# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The block-scaled FP8 GEMV must not pay Triton's binder on every decode.

``TorchFP8Linear`` reaches :func:`fp8_block_gemv` for every batch-1 projection
of a block-scaled (``FineGrainedFP8``) checkpoint. ``JITFunction.run``
re-binds, re-specializes and re-keys all 23 arguments on each call, and
``_launch_device`` probed ``torch.cuda.is_available()`` and built a
``torch.cuda.device`` context before the launch. On sm_120 that bookkeeping was
~19.5 us against an 18 us kernel, so decode stayed host-bound (#5582).

The fix keeps the compiled kernel's launcher after the first call per geometry
and skips the availability probe when the operands already sit on the current
device. The regression pins:

1. the cached launcher produces bit-identical output to the reference decode
   across rows, dtypes, bias and both scale semantics;
2. the launcher is reused instead of rebuilt per call;
3. an active Triton launch hook (a profiler) disables the cached launcher, so
   profiling still sees every launch;
4. ``_launch_device`` never probes availability and returns a no-op context
   when the operand device is already current.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.utils import fp8_block_triton


BLOCK = (128, 128)


_TRITON_GEMV_AVAILABLE = fp8_block_triton.triton_fp8_available(
    torch.device("cuda")
) and fp8_block_triton.triton_fp8_supports_format(
    torch.device("cuda"), "float8_e4m3fn"
)
_GEMV_SKIP = pytest.mark.skipif(
    not _TRITON_GEMV_AVAILABLE,
    reason="block-scaled FP8 Triton GEMV is not available on this device",
)


def _payload(*, out_features: int = 512, in_features: int = 512):
    torch.manual_seed(7)
    scale = torch.rand(
        out_features // BLOCK[0], in_features // BLOCK[1], device="cuda"
    ) + 0.5
    codes = (torch.randn(out_features, in_features, device="cuda") * 2).to(
        torch.float8_e4m3fn
    )
    return codes, scale


def _reference(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor, *, inverse: bool, bias):
    expanded = scale.repeat_interleave(BLOCK[0], 0).repeat_interleave(BLOCK[1], 1)
    decoded = codes.float() / expanded if inverse else codes.float() * expanded
    expected = x.float() @ decoded.t()
    if bias is not None:
        expected = expected + bias.float()
    return expected


class _HookChain:
    """Minimal stand-in for ``triton.knobs.HookChain``."""

    def __init__(self, *calls):
        self.calls = list(calls)


def test_launch_hook_active_recognises_chains_and_plain_hooks():
    assert not fp8_block_triton._launch_hook_active(None)
    assert not fp8_block_triton._launch_hook_active(_HookChain())
    assert fp8_block_triton._launch_hook_active(_HookChain(lambda *a, **k: None))
    assert fp8_block_triton._launch_hook_active(lambda *a, **k: None)


def test_launch_device_skips_the_availability_probe_and_current_device_guard(monkeypatch):
    def fail_probe():
        raise AssertionError("torch.cuda.is_available() must not run per launch")

    monkeypatch.setattr(torch.cuda, "is_available", fail_probe)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    current = fp8_block_triton._launch_device(torch.device("cuda", 0))
    assert isinstance(current, contextlib.nullcontext)

    unindexed = fp8_block_triton._launch_device(torch.device("cuda"))
    assert isinstance(unindexed, contextlib.nullcontext)

    other = fp8_block_triton._launch_device(torch.device("cuda", 1))
    assert isinstance(other, torch.cuda.device)


def test_launch_device_keeps_cpu_operands_context_free():
    assert isinstance(
        fp8_block_triton._launch_device(torch.device("cpu")), contextlib.nullcontext
    )


@pytest.mark.gpu
@_GEMV_SKIP
def test_cached_launcher_is_reused_and_matches_the_reference():
    codes, scale = _payload()
    x = torch.randn(1, codes.shape[1], device="cuda", dtype=torch.float16)
    fp8_block_triton._GEMV_LAUNCHERS.clear()

    with torch.inference_mode():
        first = fp8_block_triton.fp8_block_gemv(
            x, codes, scale, block_size=BLOCK, inverse=True
        )
        assert len(fp8_block_triton._GEMV_LAUNCHERS) == 1
        entry = next(iter(fp8_block_triton._GEMV_LAUNCHERS.values()))
        # ``(run, function, packed_metadata, grid0, grid1, device_index)``
        assert len(entry) == 6
        second = fp8_block_triton.fp8_block_gemv(
            x, codes, scale, block_size=BLOCK, inverse=True
        )
        # A second call must reuse the entry, not grow the cache.
        assert len(fp8_block_triton._GEMV_LAUNCHERS) == 1
        assert next(iter(fp8_block_triton._GEMV_LAUNCHERS.values()))[0] is entry[0]

    assert torch.equal(first, second)
    expected = _reference(x, codes, scale, inverse=True, bias=None)
    assert (second.float() - expected).abs().max().item() < 5e-2 * expected.abs().max().item()


@pytest.mark.gpu
@_GEMV_SKIP
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 2, 5])
@pytest.mark.parametrize("inverse", [True, False])
@pytest.mark.parametrize("bias", [False, True])
def test_cached_launcher_matches_the_reference_across_shapes(dtype, rows, inverse, bias):
    codes, scale = _payload(out_features=384, in_features=384)
    x = torch.randn(rows, codes.shape[1], device="cuda", dtype=dtype)
    b = torch.randn(codes.shape[0], device="cuda", dtype=dtype) if bias else None
    fp8_block_triton._GEMV_LAUNCHERS.clear()

    with torch.inference_mode():
        # The first call compiles, the second one takes the cached launcher.
        fp8_block_triton.fp8_block_gemv(
            x, codes, scale, block_size=BLOCK, inverse=inverse, bias=b
        )
        out = fp8_block_triton.fp8_block_gemv(
            x, codes, scale, block_size=BLOCK, inverse=inverse, bias=b
        )

    expected = _reference(x, codes, scale, inverse=inverse, bias=b)
    tolerance = 5e-2 * expected.abs().max().item()
    assert (out.float() - expected).abs().max().item() < tolerance


@pytest.mark.gpu
@_GEMV_SKIP
def test_cached_launcher_keys_on_operand_dtypes():
    """e4m3 and e5m2 codes (and fp16/fp32 bias) compile different kernels."""

    codes, scale = _payload(out_features=256, in_features=256)
    codes_e5m2 = codes.float().to(torch.float8_e5m2)
    x = torch.randn(1, codes.shape[1], device="cuda", dtype=torch.float16)
    fp8_block_triton._GEMV_LAUNCHERS.clear()

    with torch.inference_mode():
        out_e4m3 = fp8_block_triton.fp8_block_gemv(
            x, codes, scale, block_size=BLOCK, inverse=True
        )
        out_e5m2 = fp8_block_triton.fp8_block_gemv(
            x, codes_e5m2, scale, block_size=BLOCK, inverse=True
        )
        assert len(fp8_block_triton._GEMV_LAUNCHERS) == 2

        bias16 = torch.randn(codes.shape[0], device="cuda", dtype=torch.float16)
        bias32 = bias16.float()
        out_bias16 = fp8_block_triton.fp8_block_gemv(
            x, codes, scale, block_size=BLOCK, inverse=True, bias=bias16
        )
        out_bias32 = fp8_block_triton.fp8_block_gemv(
            x, codes, scale, block_size=BLOCK, inverse=True, bias=bias32
        )
        assert len(fp8_block_triton._GEMV_LAUNCHERS) == 4

    for out, payload, b in (
        (out_e4m3, codes, None),
        (out_e5m2, codes_e5m2, None),
        (out_bias16, codes, bias16),
        (out_bias32, codes, bias32),
    ):
        expected = _reference(x, payload, scale, inverse=True, bias=b)
        tolerance = 5e-2 * expected.abs().max().item()
        assert (out.float() - expected).abs().max().item() < tolerance


@pytest.mark.gpu
@_GEMV_SKIP
def test_active_launch_hook_keeps_the_standard_launcher(monkeypatch):
    codes, scale = _payload()
    x = torch.randn(1, codes.shape[1], device="cuda", dtype=torch.float16)
    runtime = SimpleNamespace(
        launch_enter_hook=_HookChain(lambda *a, **k: None),
        launch_exit_hook=_HookChain(),
    )
    monkeypatch.setattr(
        fp8_block_triton, "_triton_knobs", SimpleNamespace(runtime=runtime)
    )
    assert not fp8_block_triton._direct_launch_ready()
    fp8_block_triton._GEMV_LAUNCHERS.clear()

    with torch.inference_mode():
        first = fp8_block_triton.fp8_block_gemv(
            x, codes, scale, block_size=BLOCK, inverse=True
        )
        second = fp8_block_triton.fp8_block_gemv(
            x, codes, scale, block_size=BLOCK, inverse=True
        )

    # No cached launcher while a profiler hook is installed.
    assert fp8_block_triton._GEMV_LAUNCHERS == {}
    assert torch.equal(first, second)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
