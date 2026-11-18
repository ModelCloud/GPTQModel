# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the lean FP8 W8A16 decode path (#5146).

The pre-fix forward materialised a full decoded weight on every call: FP8 ->
FP32, an FP32 divide, a narrow to the activation dtype, and a contiguous
transpose. At M=1 the weight traffic dominated the GEMM. The lean path casts
the exact FP8 codes straight to the activation dtype, runs the GEMM on the
``[out, in]`` weight's transposed view with an FP32 output, and applies a
row/tensor scale to that output. Block scales divide the exact codes in FP32
opmath and round once, which is the same arithmetic as before without the FP32
weight temporary or the transpose copy.

These tests pin three contracts:

* numerical equivalence with the pre-fix path and an independent FP64 oracle;
* no decoded weight is materialised or retained across a forward;
* the layer does not regress to the FP32-round-trip cost.
"""

from __future__ import annotations

import statistics

import pytest
import torch
import torch.nn as nn

import gptqmodel.nn_modules.qlinear.fp8 as fp8_module
from gptqmodel.nn_modules.qlinear._numerics import matmul_fp32_reduce


_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA FP8 decode path required",
)
_BLOCK = (32, 64)
_IN = 256
_OUT = 512


def _inverse_kernel(
    method: str,
    dtype: torch.dtype,
    *,
    fmt: str = "float8_e4m3fn",
    in_features: int = _IN,
    out_features: int = _OUT,
    seed: int = 0,
) -> fp8_module.TorchFP8Linear:
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=True, dtype=torch.float32).eval()
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.02)
        linear.bias.normal_(0.0, 0.1)
    block = _BLOCK if method == "block" else None
    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        format=fmt,
        weight_scale_method=method,
        weight_block_size=block,
        register_buffers=False,
    )
    kernel.pack_original(linear, None, None)
    return kernel.to("cuda").eval()


def _multiplier_kernel(
    method: str,
    dtype: torch.dtype,
    *,
    fmt: str = "float8_e4m3fn",
    seed: int = 0,
) -> fp8_module.TorchFP8Linear:
    """A DeepSeek/HF-style payload whose ``weight_scale_inv`` multiplies."""
    torch.manual_seed(seed)
    block = _BLOCK if method == "block" else None
    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=_IN,
        out_features=_OUT,
        bias=True,
        format=fmt,
        weight_scale_method=method,
        weight_block_size=block,
        weight_scale_semantics="multiplier",
        register_buffers=False,
    )
    codes = (torch.randn(_OUT, _IN) * 0.02).to(getattr(torch, fmt))
    if method == "tensor":
        shape = ()
    elif method == "row":
        shape = (_OUT,)
    else:
        shape = (_OUT // block[0], _IN // block[1])
    kernel.register_buffer("weight", codes)
    kernel.register_buffer("weight_scale_inv", torch.full(shape, 0.01, dtype=torch.float32))
    kernel.register_buffer("bias", torch.randn(_OUT, dtype=torch.float32))
    kernel._invalidate_compiled()
    return kernel.to("cuda").eval()


def _decoded_fp64(kernel: fp8_module.TorchFP8Linear) -> torch.Tensor:
    codes = kernel.weight.to(torch.float64)
    scale = kernel.weight_scale_inv.to(torch.float64)
    if kernel.weight_scale_method == "tensor":
        decoded = codes * scale if kernel.weight_scale_semantics == "multiplier" else codes / scale
    elif kernel.weight_scale_method == "row":
        scale = scale.view(-1, 1)
        decoded = codes * scale if kernel.weight_scale_semantics == "multiplier" else codes / scale
    else:
        block_rows, block_cols = kernel.weight_block_size
        blocks = codes.reshape(
            kernel.out_features // block_rows,
            block_rows,
            kernel.in_features // block_cols,
            block_cols,
        )
        scale = scale[:, None, :, None]
        if kernel.weight_scale_semantics == "multiplier":
            decoded = (blocks * scale).reshape(kernel.out_features, kernel.in_features)
        else:
            decoded = (blocks / scale).reshape(kernel.out_features, kernel.in_features)
        # The block path still divides the exact codes into the activation
        # dtype (FP32 opmath, one rounding), so the oracle narrows its weight.
        return decoded
    return decoded


def _fp64_oracle(kernel: fp8_module.TorchFP8Linear, x: torch.Tensor) -> torch.Tensor:
    decoded = _decoded_fp64(kernel)
    if kernel.weight_scale_method == "block":
        decoded = decoded.to(x.dtype).to(torch.float64)
    output = x.to(torch.float64) @ decoded.t()
    if kernel.bias is not None:
        output = output + kernel.bias.to(torch.float64)
    return output.to(x.dtype)


def _pre_fix_output(kernel: fp8_module.TorchFP8Linear, x: torch.Tensor) -> torch.Tensor:
    """The 902f1a42e forward: full dequantize, then the FP32-reduced GEMM."""
    weight = kernel.dequantize_weight(device=x.device, dtype=x.dtype)
    return matmul_fp32_reduce(x, weight, bias=kernel.bias)


def _normalized_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.to(torch.float64)
    expected = expected.to(torch.float64)
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()


@_CUDA
@pytest.mark.parametrize("semantics", ["inverse", "multiplier"])
@pytest.mark.parametrize("method", ["tensor", "row", "block"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 257])
def test_fp8_lean_decode_matches_fp64_oracle(method, dtype, rows, semantics):
    """Row/tensor epilogue and block in-dtype divide must stay equivalent."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")
    kernel = (
        _inverse_kernel(method, dtype, fmt="float8_e4m3fn")
        if semantics == "inverse"
        else _multiplier_kernel(method, dtype, fmt="float8_e4m3fn")
    )
    x = torch.randn(rows, _IN, device="cuda", dtype=dtype)

    with torch.inference_mode():
        oracle = _fp64_oracle(kernel, x)
        pre_fix = _pre_fix_output(kernel, x)

    decoded_calls = []
    dequantize = kernel.dequantize_weight

    def counted_dequantize(*args, **kwargs):
        decoded_calls.append(1)
        return dequantize(*args, **kwargs)

    kernel.dequantize_weight = counted_dequantize
    with torch.inference_mode():
        actual = kernel(x)

    assert actual.dtype == dtype
    assert actual.shape == (rows, _OUT)
    # The lean path must not materialise the decoded weight at all.
    assert decoded_calls == []

    new_rms = _normalized_rms(actual, oracle)
    old_rms = _normalized_rms(pre_fix, oracle)
    # No accuracy regression against the independent FP64 oracle. For
    # row/tensor scales the lean path is far more accurate because it no longer
    # rounds the whole decoded weight to the activation dtype.
    assert new_rms <= old_rms + 1e-6
    if method == "block":
        # Same arithmetic as the pre-fix path: the codes are exact in the
        # activation dtype and the FP32 divide rounds once.
        assert torch.equal(actual, pre_fix)
        assert new_rms <= 5e-3
    else:
        # Measured ~1e-5 for FP16/BF16; the pre-fix path is ~3e-4 (FP16) and
        # ~2e-3 (BF16) because it rounds the decoded weight first.
        assert new_rms <= 2e-3
        old_vs_new = _normalized_rms(actual, pre_fix)
        limit = 1e-3 if dtype == torch.float16 else 4e-3
        assert old_vs_new <= limit, (
            f"lean path drifted {old_vs_new:.2e} from the pre-fix path "
            f"(limit {limit:.0e})"
        )


@_CUDA
@pytest.mark.parametrize("fmt", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("method", ["tensor", "row", "block"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_lean_decode_retains_only_packed_weights(method, dtype, fmt):
    """Repeated forwards must leave only the packed payload on the module."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")
    kernel = _inverse_kernel(method, dtype, fmt=fmt)
    x = torch.randn(8, _IN, device="cuda", dtype=dtype)

    def packed_bytes() -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in kernel.buffers()
        )

    buffers_before = {
        name: tensor.detach().clone() for name, tensor in kernel.named_buffers()
    }
    bytes_before = packed_bytes()
    with torch.inference_mode():
        first = kernel(x)
        second = kernel(x)
    assert torch.equal(first, second)
    assert packed_bytes() == bytes_before

    buffers = dict(kernel.named_buffers())
    assert set(buffers) == {"weight", "weight_scale_inv", "bias"}
    assert buffers["weight"].dtype == getattr(torch, fmt)
    assert buffers["weight_scale_inv"].dtype == torch.float32
    assert buffers["bias"].dtype == torch.float32
    assert torch.equal(buffers["weight"], buffers_before["weight"])
    assert torch.equal(buffers["weight_scale_inv"], buffers_before["weight_scale_inv"])
    # No decoded weight may live in a plain module attribute either.
    assert not any(isinstance(value, torch.Tensor) for value in vars(kernel).values())

    def forbidden(*args, **kwargs):
        raise AssertionError("a full decoded weight was materialised for this forward")

    kernel.dequantize_weight = forbidden
    with torch.inference_mode():
        kernel(x)


def _timed(fn, *, warmup: int = 10, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@_CUDA
def test_fp8_lean_decode_does_not_regress_to_fp32_roundtrip():
    """A 4096x14336 M=1 layer must not pay the FP32 dequant + transpose copy.

    The pre-fix path is ~5x the lean cast+GEMM+epilogue cost and ~12x a dense
    FP16 ``nn.Linear`` on sm_89. The bounds are ratios measured in the same
    process, so they do not depend on absolute clocks.
    """
    in_features, out_features = 4096, 14336
    kernel = _inverse_kernel(
        "row",
        torch.float16,
        in_features=in_features,
        out_features=out_features,
    )
    dense = nn.Linear(in_features, out_features, bias=False).to(
        device="cuda", dtype=torch.float16
    ).eval()
    x = torch.randn(1, in_features, device="cuda", dtype=torch.float16)

    scale = kernel.weight_scale_inv.view(-1).to(torch.float32)

    def lean_reference():
        # The issue's emulation: cast, transposed-view GEMM with FP32 output,
        # then the row scale in the epilogue. The cast is per call, exactly as
        # the shipped lean path does it.
        codes = kernel.weight.to(torch.float16)
        return torch.mm(x, codes.t(), out_dtype=torch.float32) / scale

    with torch.inference_mode():
        lean_ms = statistics.median(_timed(lean_reference) for _ in range(3))
        fp8_ms = statistics.median(_timed(lambda: kernel(x)) for _ in range(3))
        dense_ms = statistics.median(_timed(lambda: dense(x)) for _ in range(3))

    assert fp8_ms <= 1.8 * lean_ms, (
        f"FP8 forward {fp8_ms * 1000:.1f}us is {fp8_ms / lean_ms:.1f}x the lean "
        f"reference {lean_ms * 1000:.1f}us; the FP32 round trip returned"
    )
    assert fp8_ms <= 6.0 * dense_ms, (
        f"FP8 forward {fp8_ms * 1000:.1f}us is {fp8_ms / dense_ms:.1f}x dense "
        f"FP16 {dense_ms * 1000:.1f}us"
    )
