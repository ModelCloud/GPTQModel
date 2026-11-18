# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The FP8 W8A8 ``_scaled_mm`` capability probe must never run in ``forward``.

#5178 measured the rowwise ``torch._scaled_mm`` contract from
``TorchFP8Linear.forward``.  On a cold process cache that probe created a
``torch.Generator`` (opaque to dynamo) and read its verdict on the host, so the
first per-token forward failed under ``torch.compile(fullgraph=True)`` and,
inside a CUDA-graph capture, the failed measurement was cached as
"unsupported" for the rest of the process (#5195).

The fix resolves the contract eagerly (load / ``set_activation_quant_method`` /
device move) and, when a call still finds it unmeasured inside a capture or a
trace, falls back for that call **without caching**.  These tests pin both
halves: the deferral semantics on CPU and the cold-capture/cold-compile
contract on a GPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import gptqmodel.nn_modules.qlinear.fp8 as fp8_module
from tests.fp8_test_utils import pack_fp8_module


CUDA = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _fresh_scaled_mm_cache():
    """Keep the process-wide contract cache from leaking between tests."""

    saved = dict(fp8_module._SCALED_MM_ROW_SCALE_SUPPORT)
    fp8_module._SCALED_MM_ROW_SCALE_SUPPORT.clear()
    try:
        yield
    finally:
        fp8_module._SCALED_MM_ROW_SCALE_SUPPORT.clear()
        fp8_module._SCALED_MM_ROW_SCALE_SUPPORT.update(saved)


def _kernel(*, per_token: bool = True, bias: bool = False) -> fp8_module.TorchFP8Linear:
    torch.manual_seed(3)
    linear = nn.Linear(64, 64, bias=bias).eval()
    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=bias,
        pack_dtype=torch.int32,
        format="float8_e4m3fn",
        weight_scale_method="row",
        weight_block_size=None,
        register_buffers=False,
        activation_quant_method="per_token" if per_token else "none",
    )
    pack_fp8_module(kernel, linear)
    return kernel.eval()


def _clear_capability_caches(kernel: fp8_module.TorchFP8Linear) -> None:
    """Reproduce a cold process: no contract answer, no range bound."""

    fp8_module._SCALED_MM_ROW_SCALE_SUPPORT.clear()
    kernel._invalidate_compiled()


def _reference(kernel: fp8_module.TorchFP8Linear, x: torch.Tensor) -> torch.Tensor:
    """Decode the payload and matmul in FP32; independent of the path under test."""

    with torch.inference_mode():
        weight = kernel.dequantize_weight(device=x.device, dtype=torch.float32)
        return (x.float() @ weight).to(x.dtype)


def _capture(kernel: fp8_module.TorchFP8Linear, x: torch.Tensor) -> torch.Tensor:
    """Capture the *first* ``forward`` call, warming only the allocator.

    The warmup deliberately avoids ``forward`` so the capability caches stay
    cold: the probe is then reached from inside the capture, which is the
    regression from #5195.
    """

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        weight = kernel.weight.to(torch.float16)
        for _ in range(3):
            torch.mm(x, weight.t(), out_dtype=torch.float32)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.inference_mode():
        with torch.cuda.graph(graph):
            out = kernel(x)
    graph.replay()
    torch.cuda.synchronize()
    return out


# ---------------------------------------------------------------------------
# Deferral semantics (CPU-testable: no device work is performed)
# ---------------------------------------------------------------------------


def test_scaled_mm_env_override_parsing():
    parse = fp8_module._parse_scaled_mm_override
    assert parse(None) is None
    assert parse("auto") is None
    assert parse(" ON ") is True
    assert parse("1") is True
    assert parse("off") is False
    assert parse("0") is False
    with pytest.raises(ValueError, match="GPTQMODEL_FP8_SCALED_MM"):
        parse("sometimes")


def test_forced_override_skips_the_measurement(monkeypatch):
    calls = []

    def probe(*args):
        calls.append(args)
        return torch.tensor(True)

    monkeypatch.setattr(fp8_module, "_SCALED_MM_OVERRIDE", False)
    monkeypatch.setattr(fp8_module, "_probe_scaled_mm_row_scale", probe)
    fp8_module._SCALED_MM_ROW_SCALE_SUPPORT.clear()

    device = torch.device("cuda", 0)
    assert fp8_module._scaled_mm_row_scale_supported(device, torch.float16) is False
    assert calls == []
    assert not fp8_module._SCALED_MM_ROW_SCALE_SUPPORT


def test_unmeasurable_context_is_unknown_and_not_cached(monkeypatch):
    monkeypatch.setattr(fp8_module, "_SCALED_MM_OVERRIDE", None)
    monkeypatch.setattr(fp8_module, "_probe_must_wait", lambda: True)
    fp8_module._SCALED_MM_ROW_SCALE_SUPPORT.clear()

    device = torch.device("cuda", 0)
    assert fp8_module._scaled_mm_row_scale_verdict(device, torch.float16) is None
    # A capture/trace gets the correct fallback, not a process-wide "no".
    assert fp8_module._scaled_mm_row_scale_supported(device, torch.float16) is False
    assert fp8_module._SCALED_MM_ROW_SCALE_SUPPORT == {}


def test_oom_probe_is_not_cached(monkeypatch):
    def oom(*_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError("simulated")

    monkeypatch.setattr(fp8_module, "_SCALED_MM_OVERRIDE", None)
    monkeypatch.setattr(fp8_module, "_probe_must_wait", lambda: False)
    monkeypatch.setattr(fp8_module, "_probe_scaled_mm_row_scale", oom)
    fp8_module._SCALED_MM_ROW_SCALE_SUPPORT.clear()

    assert (
        fp8_module._scaled_mm_row_scale_verdict(torch.device("cuda", 0), torch.float16)
        is None
    )
    assert not fp8_module._SCALED_MM_ROW_SCALE_SUPPORT


def test_kernel_rejection_is_cached_and_not_reprobed(monkeypatch):
    calls = []

    def reject(*args):
        calls.append(args)
        raise RuntimeError("HIPBLAS_STATUS_NOT_SUPPORTED (simulated)")

    monkeypatch.setattr(fp8_module, "_SCALED_MM_OVERRIDE", None)
    monkeypatch.setattr(fp8_module, "_probe_must_wait", lambda: False)
    monkeypatch.setattr(fp8_module, "_probe_scaled_mm_row_scale", reject)
    fp8_module._SCALED_MM_ROW_SCALE_SUPPORT.clear()

    device = torch.device("cuda", 0)
    assert fp8_module._scaled_mm_row_scale_supported(device, torch.float16) is False
    assert fp8_module._SCALED_MM_ROW_SCALE_SUPPORT == {
        (fp8_module.IS_ROCM, 0, torch.float16): False
    }
    assert fp8_module._scaled_mm_row_scale_supported(device, torch.float16) is False
    assert len(calls) == 1


def test_numeric_mismatch_is_cached(monkeypatch):
    monkeypatch.setattr(fp8_module, "_SCALED_MM_OVERRIDE", None)
    monkeypatch.setattr(fp8_module, "_probe_must_wait", lambda: False)
    monkeypatch.setattr(
        fp8_module, "_probe_scaled_mm_row_scale", lambda *_a, **_k: torch.tensor(False)
    )
    fp8_module._SCALED_MM_ROW_SCALE_SUPPORT.clear()

    device = torch.device("cuda", 0)
    assert fp8_module._scaled_mm_row_scale_verdict(device, torch.float16) is False
    assert fp8_module._SCALED_MM_ROW_SCALE_SUPPORT == {
        (fp8_module.IS_ROCM, 0, torch.float16): False
    }


def test_cpu_forward_never_measures():
    kernel = _kernel(per_token=True)
    fp8_module._SCALED_MM_ROW_SCALE_SUPPORT.clear()
    x = torch.randn(4, 64)

    assert kernel._can_use_scaled_mm(x) is False
    assert fp8_module._SCALED_MM_ROW_SCALE_SUPPORT == {}


def test_post_init_warms_the_range_bound_without_a_forward():
    kernel = _kernel(per_token=True)
    assert not kernel._decoded_weight_bound_ready

    kernel.post_init()

    assert kernel._decoded_weight_bound_ready


# ---------------------------------------------------------------------------
# Cold-capture / cold-compile contract (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="GPU required")
def test_cold_first_forward_in_capture_keeps_the_fast_path():
    kernel = _kernel().to("cuda")
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    _clear_capability_caches(kernel)

    out = _capture(kernel, x)

    torch.testing.assert_close(out, _reference(kernel, x), rtol=5e-2, atol=5e-2)
    # The capture could not measure the contract; it must not have cached a
    # "no", and the module must not have disabled the native path.
    assert not kernel._scaled_mm_hard_disabled
    with torch.inference_mode():
        eager = kernel(x)
    torch.testing.assert_close(eager, _reference(kernel, x), rtol=5e-2, atol=5e-2)
    supported = fp8_module._scaled_mm_row_scale_supported(
        x.device, kernel._scaled_mm_out_dtype(x)
    )
    if supported:
        assert kernel._can_use_scaled_mm(x), "capture poisoned the fast path"


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="GPU required")
def test_cold_first_forward_under_fullgraph_compile():
    kernel = _kernel().to("cuda")
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    _clear_capability_caches(kernel)

    compiled = torch.compile(kernel, fullgraph=True, backend="eager")
    with torch.inference_mode():
        out = compiled(x)

    torch.testing.assert_close(out, _reference(kernel, x), rtol=5e-2, atol=5e-2)
    # Tracing may not measure (or cache) a device capability.
    assert fp8_module._SCALED_MM_ROW_SCALE_SUPPORT == {}
    with torch.inference_mode():
        eager = kernel(x)
    torch.testing.assert_close(eager, _reference(kernel, x), rtol=5e-2, atol=5e-2)
    assert kernel._can_use_scaled_mm(x) is fp8_module._scaled_mm_row_scale_supported(
        x.device, kernel._scaled_mm_out_dtype(x)
    )


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="GPU required")
def test_warm_fullgraph_compile_traces_the_native_path():
    kernel = _kernel().to("cuda")
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    if not kernel._can_use_scaled_mm(x):
        pytest.skip("rowwise _scaled_mm contract is not supported on this device")

    compiled = torch.compile(kernel, fullgraph=True, backend="eager")
    with torch.inference_mode():
        out = compiled(x)

    torch.testing.assert_close(out, _reference(kernel, x), rtol=5e-2, atol=5e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="GPU required")
def test_opt_in_measures_the_contract_eagerly():
    kernel = _kernel(per_token=False).to("cuda")
    _clear_capability_caches(kernel)
    assert fp8_module._SCALED_MM_ROW_SCALE_SUPPORT == {}

    kernel.set_activation_quant_method("per_token")

    # The documented W8A8 opt-in is the last safe eager point to measure.
    assert fp8_module._SCALED_MM_ROW_SCALE_SUPPORT, (
        "set_activation_quant_method('per_token') must resolve the contract"
    )
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    assert kernel._can_use_scaled_mm(x) is fp8_module._scaled_mm_row_scale_supported(
        x.device, kernel._scaled_mm_out_dtype(x)
    )


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="GPU required")
def test_unsupported_contract_falls_back_and_caches(monkeypatch):
    kernel = _kernel().to("cuda")
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    _clear_capability_caches(kernel)
    real = torch._scaled_mm

    def dropping_row_scale(x_q, w_q, scale_a=None, scale_b=None, out_dtype=None):
        # The #5174 defect: the layout is accepted, scale_a is ignored.
        return real(
            x_q,
            w_q,
            scale_a=torch.ones_like(scale_a),
            scale_b=scale_b,
            out_dtype=out_dtype,
        )

    monkeypatch.setattr(torch, "_scaled_mm", dropping_row_scale)
    with torch.inference_mode():
        out = kernel(x)

    torch.testing.assert_close(out, _reference(kernel, x), rtol=5e-2, atol=5e-2)
    assert not kernel._can_use_scaled_mm(x)
    assert fp8_module._SCALED_MM_ROW_SCALE_SUPPORT
    assert not any(fp8_module._SCALED_MM_ROW_SCALE_SUPPORT.values())


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="GPU required")
def test_warm_forward_still_takes_the_native_path(monkeypatch):
    kernel = _kernel().to("cuda")
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    if not kernel._can_use_scaled_mm(x):
        pytest.skip("rowwise _scaled_mm contract is not supported on this device")

    calls = []
    native = kernel._forward_scaled_mm

    def counted(tensor):
        calls.append(1)
        return native(tensor)

    monkeypatch.setattr(kernel, "_forward_scaled_mm", counted)
    with torch.inference_mode():
        first = kernel(x)
        second = kernel(x)

    assert len(calls) == 2, "the warm per-token path must stay on _scaled_mm"
    torch.testing.assert_close(first, second)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="GPU required")
def test_device_move_warms_the_range_bound():
    kernel = _kernel(per_token=False)
    assert not kernel._decoded_weight_bound_ready

    kernel = kernel.to("cuda")

    assert kernel._decoded_weight_bound_ready
