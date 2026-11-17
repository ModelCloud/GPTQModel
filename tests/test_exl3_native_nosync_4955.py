# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for #4955: the BF16 pre-scale stays on the device.

``ExllamaV3Linear._native_bf16_input`` opened with
``float(torch.linalg.vector_norm(x.detach(), ord=inf))``, a device -> host
synchronization on *every* BF16 forward of *every* native EXL3 linear: 112
syncs per decoded token on a 1B model and 224 on 8B. The per-row decision it
guarded (#4870/#4966) never needed the host: the scale is a power of two
computed from ``row_amax``, and an unscaled row simply gets 1.0.

The decisive test runs a warm module's BF16 forward under
``torch.cuda.set_sync_debug_mode("error")``, which turns any ``.item()``,
``float()``, ``bool()`` or ``nonzero`` on a CUDA tensor into a ``RuntimeError``.
On the pre-fix code the first forward raises it; here only the one-time
``isfinite`` warmup probe may read the host, so the test warms up first and then
requires eight consecutive sync-free forwards.

The numeric contract is unchanged: rows inside ``_FP16_SAFE_AMAX`` convert to
FP16 bit-exactly, a wide row is divided by the smallest power of two that brings
it into ``(bound / 2, bound]`` and multiplied back exactly, the ordinary rows of
a mixed batch keep the plain FP16 path's accuracy, and a non-finite activation
keeps the FP32 Torch reconstruction. ``test_scale_decision_matches_the_pre_fix_code``
pins the whole decision against a verbatim copy of the pre-fix implementation.
"""

import contextlib
import math

import pytest
import torch

from gptqmodel.nn_modules import exllamav3 as exl3_bridge
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the native EXL3 dispatch needs a CUDA device",
)


def _bridge(out_features: int = 64) -> ExllamaV3Linear:
    """An ``ExllamaV3Linear`` without ``nn.Module.__init__`` for dispatch tests."""
    module = object.__new__(ExllamaV3Linear)
    module.out_features = out_features
    module.bias = None
    return module


def _unprobed(module: ExllamaV3Linear) -> ExllamaV3Linear:
    """Forget the module's warmup certificate so the next call probes again."""
    module.__dict__.pop("_native_bf16_inputs_finite", None)
    return module


def _prepared(module: ExllamaV3Linear, x: torch.Tensor):
    prepared = module._native_bf16_input(x)
    assert prepared is not None
    return prepared


def _scale_rows(scale, rows: int) -> torch.Tensor:
    """The per-row scale, with ``None`` (nothing scaled) read as all ones."""
    if scale is None:
        return torch.ones(rows, 1)
    return scale.reshape(rows, -1)


def _pre_fix_native_bf16_input(x: torch.Tensor):
    """Verbatim copy of ``_native_bf16_input`` as of main @ c52ae996d (#4966)."""
    if x.numel() == 0:
        return x.to(torch.float16), None
    amax = float(torch.linalg.vector_norm(x.detach(), ord=float("inf")))
    if not math.isfinite(amax):
        return None
    if amax <= exl3_bridge._FP16_SAFE_AMAX:
        return x.to(torch.float16), None
    row_amax = x.detach().float().abs().amax(dim=-1, keepdim=True)
    _, exponent = torch.frexp(row_amax / exl3_bridge._FP16_SAFE_AMAX)
    output_scale = torch.where(
        row_amax > exl3_bridge._FP16_SAFE_AMAX,
        torch.ldexp(torch.ones_like(row_amax), exponent),
        torch.ones_like(row_amax),
    )
    return (x / output_scale).to(torch.float16), output_scale


@contextlib.contextmanager
def _forbid_host_reads():
    """Turn every ``Tensor.item/float/bool`` into a test failure."""
    originals = (torch.Tensor.item, torch.Tensor.__float__, torch.Tensor.__bool__)

    def fail(*_args, **_kwargs):
        raise AssertionError("host read of a CUDA tensor")

    torch.Tensor.item = fail
    torch.Tensor.__float__ = fail
    torch.Tensor.__bool__ = fail
    try:
        yield
    finally:
        torch.Tensor.item, torch.Tensor.__float__, torch.Tensor.__bool__ = originals


class _FakeNative:
    """Stand-in for ``LinearEXL3``: FP16 in, FP16 out, no host reads."""

    def __init__(self, in_features: int, out_features: int):
        torch.manual_seed(4955)
        self.out_features = out_features
        self.weight = (
            torch.randn(in_features, out_features, device="cuda") * 0.02
        ).to(torch.float16)
        self.calls = 0

    def forward(self, x, params, out_dtype):
        assert x.dtype == torch.float16
        self.calls += 1
        return (x @ self.weight).to(out_dtype)


# --------------------------------------------------------------------------
# The decisive assertion: no device -> host sync on the BF16 forward path.
# --------------------------------------------------------------------------


@requires_cuda
def test_bf16_forward_is_sync_free_after_warmup():
    """Eight consecutive BF16 forwards must not synchronize the stream (#4955).

    The pre-fix code raises ``RuntimeError: called a synchronizing CUDA
    operation`` on the first forward here; the fix only reads the host once, in
    the module's ``isfinite`` warmup probe.
    """
    in_features, out_features = 256, 32
    module = _bridge(out_features)
    native = _FakeNative(in_features, out_features)
    module._ensure_inner = lambda: native
    module._ensure_torch_inner = lambda: pytest.fail(
        "a finite BF16 activation must use the native FP16 kernel"
    )
    x = (torch.randn(4, in_features, device="cuda") * 0.5).to(torch.bfloat16)

    # Warmup: the one-time isfinite probe (one host read for the module).
    module.forward(x)
    torch.cuda.synchronize()

    with pytest.warns(UserWarning, match="prototype feature"):
        torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(8):
            output = module.forward(x)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert output.dtype == torch.bfloat16
    assert native.calls == 9


@requires_cuda
def test_bf16_prescale_returns_a_device_scale_without_host_reads():
    """The scale is a CUDA tensor and the call touches no Python scalars."""
    in_features = 512
    module = _unprobed(_bridge(16))
    x = (torch.randn(4, in_features, device="cuda") * 0.5).to(torch.bfloat16)
    module._native_bf16_input(x)  # warmup probe

    with _forbid_host_reads():
        native_input, output_scale = module._native_bf16_input(x)

    assert isinstance(native_input, torch.Tensor) and native_input.is_cuda
    assert native_input.dtype == torch.float16
    assert isinstance(output_scale, torch.Tensor) and output_scale.is_cuda
    assert output_scale.dtype == torch.float32
    assert tuple(output_scale.shape) == (4, 1)


# --------------------------------------------------------------------------
# Numeric contract, CPU-testable.
# --------------------------------------------------------------------------


def test_rows_inside_the_bound_convert_bit_exactly():
    """A row FP16 can hold keeps scale 1 and converts with no rounding."""
    torch.manual_seed(4955)
    x = (torch.randn(8, 512) * 32.0).to(torch.bfloat16)
    assert 0.0 < float(x.abs().max()) <= exl3_bridge._FP16_SAFE_AMAX

    native_input, output_scale = _prepared(_unprobed(_bridge()), x)

    assert torch.equal(native_input, x.to(torch.float16))
    # ``None`` and an all-ones tensor are the same decision: scale 1.
    assert torch.equal(_scale_rows(output_scale, x.shape[0]), torch.ones(8, 1))


def test_wide_row_is_scaled_just_inside_the_bound_and_rescales_exactly():
    """A wide row lands in ``(bound / 2, bound]`` and divides back exactly."""
    x = torch.full((4, 128), 1e5, dtype=torch.bfloat16)

    native_input, output_scale = _prepared(_unprobed(_bridge()), x)
    scale = float(_scale_rows(output_scale, x.shape[0])[0, 0])

    assert scale > 1.0
    assert math.log2(scale).is_integer()
    assert bool(torch.isfinite(native_input).all())
    assert float(native_input.abs().max()) <= exl3_bridge._FP16_SAFE_AMAX
    assert float(native_input.abs().max()) > exl3_bridge._FP16_SAFE_AMAX / 2.0
    # Power-of-two scaling is exact in BF16, so the round trip is bit-exact.
    assert torch.equal(native_input.float() * scale, x.float())


def test_ordinary_rows_of_a_mixed_batch_keep_fp16_accuracy():
    """One 1e4 outlier must not touch the other 63 rows (#4870's fixture)."""
    torch.manual_seed(4955)
    rows, in_features, out_features = 64, 8192, 64
    x = (torch.randn(rows, in_features) * 0.3).to(torch.bfloat16)
    x[0, 5] = 1e4
    weight = (torch.randn(in_features, out_features) * 0.0002).to(torch.float16)

    native_input, output_scale = _prepared(_unprobed(_bridge(out_features)), x)
    kernel_output = (native_input @ weight).to(torch.float16)
    kernel_output = (
        kernel_output.float() * _scale_rows(output_scale, rows)
    ).to(torch.float16)
    reference = x.double() @ weight.double()

    actual = kernel_output[1:].double()
    relative = (actual - reference[1:]).norm(dim=1) / reference[1:].norm(dim=1)
    assert float(relative.max()) <= 5e-3
    # Ordinary rows are the plain FP16 path, bit for bit.
    assert torch.equal(native_input[1:], x[1:].to(torch.float16))


@pytest.mark.parametrize("invalid", [float("inf"), -float("inf"), float("nan")])
def test_nonfinite_input_keeps_the_fp32_fallback(invalid):
    """A non-finite activation still asks for the FP32 Torch reconstruction."""
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    x[0, 0] = invalid

    assert _unprobed(_bridge())._native_bf16_input(x) is None


def test_scale_decision_matches_the_pre_fix_code():
    """The device decision is identical to the pre-fix host decision."""
    bound = exl3_bridge._FP16_SAFE_AMAX
    cases = [
        torch.zeros(3, 128, dtype=torch.bfloat16),
        torch.full((3, 128), 1e-30, dtype=torch.bfloat16),
        torch.full((3, 128), 1.0, dtype=torch.bfloat16),
        torch.full((3, 128), bound, dtype=torch.bfloat16),
        torch.full((3, 128), bound * 2.0, dtype=torch.bfloat16),
        torch.full((3, 128), 1e5, dtype=torch.bfloat16),
        torch.tensor([[float("inf")] * 128, [1.0] * 128, [2.0] * 128], dtype=torch.bfloat16),
        torch.tensor([[float("nan")] * 128, [1.0] * 128, [2.0] * 128], dtype=torch.bfloat16),
        torch.tensor(
            [[1e5] * 128, [1.0] * 128, [3.0] * 128], dtype=torch.bfloat16
        ),
    ]
    for x in cases:
        old = _pre_fix_native_bf16_input(x)
        new = _unprobed(_bridge())._native_bf16_input(x)
        if old is None:
            assert new is None, "non-finite input must keep the FP32 fallback"
            continue
        old_input, old_scale = old
        assert new is not None
        new_input, new_scale = new
        assert torch.equal(new_input, old_input)
        assert torch.equal(
            _scale_rows(new_scale, x.shape[0]),
            _scale_rows(old_scale, x.shape[0]),
        )
