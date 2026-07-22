# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.nn_modules.triton_utils.three_bit import (
    LAYOUT_AWQ,
    LAYOUT_GPTQ,
    dequantize_3bit,
    expand_gptq_3bit_to_uint4b8,
    matmul_3bit,
    pack_3bit,
    prepare_trilin_3bit,
    unpack_3bit,
)
from gptqmodel.utils.trilin import select_trilin_split_k, trilin_matmul


BITS = 3
GROUP_SIZE = 128
ZERO = 1 << (BITS - 1)


def _codes(k: int, n: int) -> torch.Tensor:
    values = torch.arange(k * n, dtype=torch.int64).reshape(k, n)
    # This deterministic pattern includes every 3-bit code at both word-straddling positions.
    return ((values * 5 + values // 7 + 3) & 0x7).to(torch.int32)


def _scales(
    k: int,
    n: int,
    *,
    dtype: torch.dtype = torch.float32,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    effective_group_size = k if group_size == -1 else group_size
    groups = k // effective_group_size
    values = torch.arange(groups * n, dtype=torch.float32).reshape(groups, n)
    return (0.03125 + (values.remainder(17) + 1) / 32).to(dtype=dtype)


def _dense_weight(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    effective_group_size = codes.shape[0] if group_size == -1 else group_size
    expanded_scales = scales.repeat_interleave(effective_group_size, dim=0)
    return (codes.to(torch.float32) - ZERO) * expanded_scales.to(torch.float32)


def _quality_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    difference = actual_fp32 - reference_fp32
    absolute = difference.abs()
    rmse = difference.square().mean().sqrt()
    reference_rms = reference_fp32.square().mean().sqrt().clamp_min(1e-12)
    actual_flat = actual_fp32.flatten()
    reference_flat = reference_fp32.flatten()
    cosine = torch.dot(actual_flat, reference_flat) / (
        actual_flat.norm() * reference_flat.norm()
    ).clamp_min(1e-12)
    return {
        "max_abs": absolute.max().item(),
        "mean_abs": absolute.mean().item(),
        "relative_rmse": (rmse / reference_rms).item(),
        "cosine": cosine.item(),
    }


def _packed_buffers(
    *,
    layout: str,
    k: int,
    n: int,
    device: torch.device | str = "cpu",
    scale_dtype: torch.dtype = torch.float16,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codes = _codes(k, n)
    scales = _scales(k, n, dtype=scale_dtype, group_size=group_size)
    axis = 0 if layout == LAYOUT_GPTQ else 1
    packed = pack_3bit(codes, axis=axis)
    return codes.to(device), packed.to(device), scales.to(device)


@pytest.mark.parametrize("axis,shape", [(0, (64, 96)), (1, (64, 96))])
def test_contiguous_3bit_pack_round_trip(axis: int, shape: tuple[int, int]):
    values = _codes(*shape)
    packed = pack_3bit(values, axis=axis)
    restored = unpack_3bit(packed, axis=axis, count=shape[axis])

    expected_shape = list(shape)
    expected_shape[axis] = expected_shape[axis] // 32 * 3
    assert tuple(packed.shape) == tuple(expected_shape)
    assert packed.dtype == torch.int32
    torch.testing.assert_close(restored, values, rtol=0, atol=0)


def test_contiguous_3bit_pack_matches_real_gptq_packer():
    k, n = 128, 64
    codes = _codes(k, n)
    scales_group_n = _scales(k, n)
    zeros_group_n = torch.full_like(scales_group_n, ZERO)
    dense_weight = _dense_weight(codes, scales_group_n).t().contiguous()
    linear = torch.nn.Linear(k, n, bias=False, dtype=torch.float32)
    linear.weight.data.copy_(dense_weight)

    module = TorchLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=False,
    )
    g_idx = torch.arange(k, dtype=torch.int32) // GROUP_SIZE
    module.pack_original(
        linear=linear,
        scales=scales_group_n.t().contiguous(),
        zeros=zeros_group_n.t().contiguous(),
        g_idx=g_idx,
    )

    torch.testing.assert_close(module.qweight, pack_3bit(codes, axis=0), rtol=0, atol=0)
    torch.testing.assert_close(module.qzeros, pack_3bit(zeros_group_n.to(torch.int32), axis=1), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for 3-bit native-cache conversion test")
def test_gptq_3bit_to_uint4b8_expansion_is_exact():
    k, n = 256, 128
    codes = _codes(k, n)
    qweight = pack_3bit(codes, axis=0).cuda()

    actual = expand_gptq_3bit_to_uint4b8(qweight).cpu()

    blocks = (codes + 4).to(torch.int64).reshape(k // 8, 8, n)
    expected = torch.zeros((k // 8, n), dtype=torch.int64)
    for index in range(8):
        expected |= blocks[:, index] << (4 * index)
    torch.testing.assert_close(actual, expected.to(torch.int32), rtol=0, atol=0)


def test_trilin_split_k_selector_uses_measured_small_m_split():
    assert select_trilin_split_k(1, 4096) == 32
    assert select_trilin_split_k(4, 4096) == 32
    assert select_trilin_split_k(5, 4096) == 32
    assert select_trilin_split_k(16, 4096) == 32
    assert select_trilin_split_k(128, 4096) == 1
    assert select_trilin_split_k(1, 128) == 1
    assert select_trilin_split_k(16, 256) == 2


@pytest.mark.parametrize("linear_cls", [TritonV2Linear, AwqGEMMTritonLinear])
def test_3bit_backend_capability_accepts_supported_group_sizes(linear_cls):
    common = {
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "desc_act": False,
        "sym": True,
        "in_features": 3072,
        "out_features": 96,
        "pack_dtype": torch.int32,
        "dtype": torch.float16,
        "device": DEVICE.CUDA,
    }
    valid, error = linear_cls.validate(**common)
    assert valid, error

    for group_size in linear_cls.SUPPORTS_GROUP_SIZE:
        valid, error = linear_cls.validate(**(common | {"group_size": group_size}))
        assert valid, (group_size, error)

    invalid_overrides = [
        {"group_size": 8},
        {"desc_act": True},
        {"sym": False},
        {"pack_dtype": torch.int16},
        {"in_features": 1056},
        {"out_features": 80},
    ]
    for override in invalid_overrides:
        valid, error = linear_cls.validate(**(common | override))
        assert not valid
        assert error is not None

    valid, error = linear_cls.validate(**(common | {"dynamic": {}}))
    assert valid, error


def test_awq_4bit_backend_does_not_gain_bf16_capability_from_3bit_path():
    valid, error = AwqGEMMTritonLinear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        in_features=256,
        out_features=96,
        pack_dtype=torch.int32,
        dtype=torch.bfloat16,
        device=DEVICE.CUDA,
    )
    assert not valid
    assert error is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton 3-bit kernel test")
@pytest.mark.parametrize("layout", [LAYOUT_GPTQ, LAYOUT_AWQ])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,k,n", [(1, 128, 96), (5, 256, 160), (33, 384, 224), (128, 256, 96)])
def test_raw_3bit_matmul_matches_dense_reference(layout: str, dtype: torch.dtype, m: int, k: int, n: int):
    pytest.importorskip("triton")
    torch.manual_seed(7)
    device = torch.device("cuda")
    codes, packed, scales = _packed_buffers(
        layout=layout,
        k=k,
        n=n,
        device=device,
        scale_dtype=dtype,
    )
    x = torch.randn((m, k), device=device, dtype=dtype)
    expected = torch.matmul(x.to(torch.float32), _dense_weight(codes, scales)).to(dtype)

    actual = matmul_3bit(x, packed, scales, layout=layout)

    assert actual.shape == (m, n)
    assert actual.dtype == dtype
    assert actual.device == x.device
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for grouped Triton 3-bit kernel test")
@pytest.mark.parametrize("layout", [LAYOUT_GPTQ, LAYOUT_AWQ])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [16, 32, 64, 96, 192, 256, 384, 512, 1024, -1])
def test_raw_grouped_3bit_matmul_matches_independent_fp32_reference(
    layout: str,
    dtype: torch.dtype,
    group_size: int,
):
    pytest.importorskip("triton")
    torch.manual_seed(13)
    device = torch.device("cuda")
    m, k, n = 3, 3072, 96
    codes, packed, scales = _packed_buffers(
        layout=layout,
        k=k,
        n=n,
        device=device,
        group_size=group_size,
    )
    x = torch.randn((m, k), device=device, dtype=dtype)
    expected = torch.matmul(
        x.float(),
        _dense_weight(codes, scales, group_size=group_size),
    ).to(dtype)

    actual = matmul_3bit(
        x,
        packed,
        scales,
        layout=layout,
        group_size=group_size,
    )

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.5)


@pytest.mark.parametrize("layout", [LAYOUT_GPTQ, LAYOUT_AWQ])
@pytest.mark.parametrize("group_size", [16, 32, 64, 96, 192, 256, 384, 512, 1024, -1])
def test_grouped_3bit_torch_dequant_matches_independent_fp32_reference(layout: str, group_size: int):
    k, n = 3072, 64
    codes, packed, scales = _packed_buffers(
        layout=layout,
        k=k,
        n=n,
        scale_dtype=torch.float32,
        group_size=group_size,
    )

    actual = dequantize_3bit(packed, scales, layout=layout, group_size=group_size)
    expected = _dense_weight(codes, scales, group_size=group_size)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton 3-bit stream test")
@pytest.mark.parametrize("group_size", [16, 64, 128, -1])
def test_raw_3bit_matmul_uses_current_stream(group_size: int):
    pytest.importorskip("triton")
    torch.manual_seed(11)
    device = torch.device("cuda")
    m, k, n = 3, 256, 96
    codes, packed, scales = _packed_buffers(
        layout=LAYOUT_GPTQ,
        k=k,
        n=n,
        device=device,
        group_size=group_size,
    )
    x = torch.randn((m, k), device=device, dtype=torch.float16)
    expected = torch.matmul(x.float(), _dense_weight(codes, scales, group_size=group_size)).half()
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        actual = matmul_3bit(x, packed, scales, layout=LAYOUT_GPTQ, group_size=group_size)
        completion = torch.cuda.Event()
        completion.record(stream)
    torch.cuda.current_stream(device=device).wait_event(completion)

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for grouped Triton 3-bit graph test")
@pytest.mark.parametrize("layout,group_size", [(LAYOUT_GPTQ, 16), (LAYOUT_AWQ, 64), (LAYOUT_GPTQ, -1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_raw_grouped_3bit_matmul_supports_cuda_graph_capture(
    layout: str,
    group_size: int,
    dtype: torch.dtype,
):
    pytest.importorskip("triton")
    torch.manual_seed(37)
    device = torch.device("cuda")
    m, k, n = 3, 512, 128
    _, packed, scales = _packed_buffers(
        layout=layout,
        k=k,
        n=n,
        device=device,
        group_size=group_size,
    )
    x = torch.randn((m, k), device=device, dtype=dtype)
    expected = matmul_3bit(x, packed, scales, layout=layout, group_size=group_size).clone()
    warmup_stream = torch.cuda.Stream(device=device)

    warmup_stream.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            matmul_3bit(x, packed, scales, layout=layout, group_size=group_size)
    torch.cuda.current_stream(device=device).wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = matmul_3bit(x, packed, scales, layout=layout, group_size=group_size)
    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton 3-bit module test")
@pytest.mark.parametrize("layout", [LAYOUT_GPTQ, LAYOUT_AWQ])
@pytest.mark.parametrize("m", [7, 65])
def test_3bit_quant_linear_wrapper_matches_dense_reference(layout: str, m: int):
    pytest.importorskip("triton")
    torch.manual_seed(19)
    device = torch.device("cuda")
    k, n = 256, 96
    codes, packed, scales = _packed_buffers(
        layout=layout,
        k=k,
        n=n,
        device=device,
    )
    zeros = pack_3bit(
        torch.full((k // GROUP_SIZE, n), ZERO, dtype=torch.int32),
        axis=1,
    ).to(device)
    bias = torch.linspace(-0.25, 0.25, n, dtype=torch.float16, device=device)

    linear_cls = TritonV2Linear if layout == LAYOUT_GPTQ else AwqGEMMTritonLinear
    module = linear_cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=True,
        pack_dtype=torch.int32,
        register_buffers=True,
    ).to(device)
    module.qweight.copy_(packed)
    module.qzeros.copy_(zeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)
    if layout == LAYOUT_GPTQ:
        module.g_idx.copy_(torch.arange(k, dtype=torch.int32, device=device) // GROUP_SIZE)
    module.post_init()
    module.eval()

    if layout == LAYOUT_AWQ:
        torch.testing.assert_close(module._triton_3bit_qweight, pack_3bit(codes, axis=0), rtol=0, atol=0)
        assert "_triton_3bit_qweight" not in module.state_dict()

    x = torch.randn((1, m, k), dtype=torch.float16, device=device)
    expected = torch.matmul(x.reshape(m, k).float(), _dense_weight(codes, scales)).half()
    expected = (expected + bias).reshape(1, m, n)
    actual = module(x)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for 3-bit backend quality test")
@pytest.mark.parametrize("layout", [LAYOUT_GPTQ, LAYOUT_AWQ])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 16, 33])
def test_3bit_backend_quality_matches_eager_torch_reference(
    layout: str,
    dtype: torch.dtype,
    m: int,
):
    """Compare production dispatch with TorchLinear's independent 3-bit unpack/dequant path."""
    pytest.importorskip("triton")
    torch.manual_seed(41)
    device = torch.device("cuda")
    k, n = 512, 256
    codes, packed, scales = _packed_buffers(
        layout=layout,
        k=k,
        n=n,
        device=device,
    )
    gptq_packed = pack_3bit(codes.cpu(), axis=0).to(device)
    zeros = pack_3bit(
        torch.full((k // GROUP_SIZE, n), ZERO, dtype=torch.int32),
        axis=1,
    ).to(device)
    g_idx = torch.arange(k, dtype=torch.int32, device=device) // GROUP_SIZE
    bias = torch.linspace(-0.25, 0.25, n, dtype=torch.float16, device=device)

    reference = TorchLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=True,
        pack_dtype=torch.int32,
        register_buffers=True,
    ).to(device)
    # Keep the baseline independent of both candidate kernels and Triton's dequantizer.
    reference.optimize = lambda *_args, **_kwargs: None
    reference._triton_dequant_enabled = False
    reference.qweight.copy_(gptq_packed)
    reference.qzeros.copy_(zeros)
    reference.scales.copy_(scales)
    reference.g_idx.copy_(g_idx)
    reference.bias.copy_(bias)
    reference.post_init()
    reference.eval()

    linear_cls = TritonV2Linear if layout == LAYOUT_GPTQ else AwqGEMMTritonLinear
    candidate = linear_cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=True,
        pack_dtype=torch.int32,
        register_buffers=True,
    ).to(device)
    candidate.qweight.copy_(packed)
    candidate.qzeros.copy_(zeros)
    candidate.scales.copy_(scales)
    candidate.bias.copy_(bias)
    if layout == LAYOUT_GPTQ:
        candidate.g_idx.copy_(g_idx)
    candidate.post_init()
    candidate.eval()

    x = torch.randn((1, m, k), dtype=dtype, device=device)
    with torch.inference_mode():
        expected = reference(x)
        actual = candidate(x)

    assert actual.shape == expected.shape == (1, m, n)
    assert actual.dtype == expected.dtype == dtype
    assert bool(torch.isfinite(actual).all())
    metrics = _quality_metrics(actual, expected)
    limits = {
        torch.float16: {
            "max_abs": 0.125,
            "mean_abs": 0.02,
            "relative_rmse": 0.0015,
            "cosine": 0.999995,
        },
        torch.bfloat16: {
            "max_abs": 0.5,
            "mean_abs": 0.025 if m <= 16 else 0.01,
            "relative_rmse": 0.003 if m <= 16 else 0.002,
            "cosine": 0.99999,
        },
    }[dtype]
    assert metrics["max_abs"] <= limits["max_abs"], metrics
    assert metrics["mean_abs"] <= limits["mean_abs"], metrics
    assert metrics["relative_rmse"] <= limits["relative_rmse"], metrics
    assert metrics["cosine"] >= limits["cosine"], metrics

    if dtype == torch.bfloat16 and m <= 16:
        oracle = torch.matmul(x.reshape(m, k).float(), _dense_weight(codes, scales)) + bias.float()
        oracle = oracle.to(dtype).reshape(1, m, n)
        reference_oracle_metrics = _quality_metrics(expected, oracle)
        actual_oracle_metrics = _quality_metrics(actual, oracle)
        assert actual_oracle_metrics["mean_abs"] <= reference_oracle_metrics["mean_abs"], {
            "actual": actual_oracle_metrics,
            "reference": reference_oracle_metrics,
        }
        assert actual_oracle_metrics["relative_rmse"] <= reference_oracle_metrics["relative_rmse"], {
            "actual": actual_oracle_metrics,
            "reference": reference_oracle_metrics,
        }


GROUPED_BACKEND_CASES = [
    *((LAYOUT_GPTQ, group_size) for group_size in (16, 32, 64, 96, 192, 256, 384, 512, 1024, -1)),
    *((LAYOUT_AWQ, group_size) for group_size in (16, 32, 64, 96, 192, 256, 384, 512, -1)),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for grouped 3-bit backend quality test")
@pytest.mark.parametrize("layout,group_size", GROUPED_BACKEND_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grouped_3bit_backend_matches_torch_and_fp32_references(
    layout: str,
    group_size: int,
    dtype: torch.dtype,
):
    """Exercise production fallback dispatch for every newly admitted GPTQ/AWQ group size."""
    pytest.importorskip("triton")
    torch.manual_seed(43)
    device = torch.device("cuda")
    m, k, n = 3, 3072, 128
    effective_group_size = k if group_size == -1 else group_size
    codes, packed, scales = _packed_buffers(
        layout=layout,
        k=k,
        n=n,
        device=device,
        group_size=group_size,
    )
    gptq_packed = pack_3bit(codes.cpu(), axis=0).to(device)
    zeros = pack_3bit(
        torch.full((k // effective_group_size, n), ZERO, dtype=torch.int32),
        axis=1,
    ).to(device)
    g_idx = torch.arange(k, dtype=torch.int32, device=device) // effective_group_size
    bias = torch.linspace(-0.25, 0.25, n, dtype=torch.float16, device=device)

    reference = TorchLinear(
        bits=BITS,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=True,
        pack_dtype=torch.int32,
        register_buffers=True,
    ).to(device)
    reference.optimize = lambda *_args, **_kwargs: None
    reference._triton_dequant_enabled = False
    reference.qweight.copy_(gptq_packed)
    reference.qzeros.copy_(zeros)
    reference.scales.copy_(scales)
    reference.g_idx.copy_(g_idx)
    reference.bias.copy_(bias)
    reference.post_init()
    reference.eval()

    linear_cls = TritonV2Linear if layout == LAYOUT_GPTQ else AwqGEMMTritonLinear
    candidate = linear_cls(
        bits=BITS,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=True,
        pack_dtype=torch.int32,
        register_buffers=True,
    ).to(device)
    candidate.qweight.copy_(packed)
    candidate.qzeros.copy_(zeros)
    candidate.scales.copy_(scales)
    candidate.bias.copy_(bias)
    if layout == LAYOUT_GPTQ:
        candidate.g_idx.copy_(g_idx)
    candidate.post_init()
    candidate.eval()

    assert candidate._trilin_native_3bit == (group_size in {16, 32, 64, 96, 192, 256, 384, 512, 1024})
    x = torch.randn((1, m, k), dtype=dtype, device=device)
    with torch.inference_mode():
        torch_reference = reference(x)
        actual = candidate(x)
    fp32_reference = (
        torch.matmul(
            x.reshape(m, k).float(),
            _dense_weight(codes, scales, group_size=group_size),
        )
        + bias.float()
    ).to(dtype).reshape(1, m, n)

    assert actual.shape == torch_reference.shape == fp32_reference.shape == (1, m, n)
    assert actual.dtype == torch_reference.dtype == fp32_reference.dtype == dtype
    limits = {
        "torch": {
            torch.float16: {
                "max_abs": 0.25,
                "mean_abs": 0.04,
                "relative_rmse": 0.0025,
                "cosine": 0.99999,
            },
            torch.bfloat16: {
                "max_abs": 1.0,
                "mean_abs": 0.1,
                "relative_rmse": 0.004,
                "cosine": 0.99998,
            },
        },
        "fp32": {
            torch.float16: {
                "max_abs": 0.125,
                "mean_abs": 0.04,
                "relative_rmse": 0.0025,
                "cosine": 0.99999,
            },
            torch.bfloat16: {
                "max_abs": 1.0,
                "mean_abs": 0.1,
                "relative_rmse": 0.004,
                "cosine": 0.99998,
            },
        },
    }
    for name, expected in (("torch", torch_reference), ("fp32", fp32_reference)):
        metrics = _quality_metrics(actual, expected)
        reference_limits = limits[name][dtype]
        assert metrics["max_abs"] <= reference_limits["max_abs"], (name, metrics)
        assert metrics["mean_abs"] <= reference_limits["mean_abs"], (name, metrics)
        assert metrics["relative_rmse"] <= reference_limits["relative_rmse"], (name, metrics)
        assert metrics["cosine"] >= reference_limits["cosine"], (name, metrics)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native 3-bit route test")
@pytest.mark.parametrize("layout", [LAYOUT_GPTQ, LAYOUT_AWQ])
@pytest.mark.parametrize(
    "m,dtype",
    [(1, torch.float16), (1, torch.bfloat16), (16, torch.bfloat16), (17, torch.float16)],
)
def test_3bit_quant_linear_uses_nonpersistent_native_cache(
    layout: str,
    m: int,
    dtype: torch.dtype,
    monkeypatch,
):
    pytest.importorskip("triton")
    torch.manual_seed(29)
    device = torch.device("cuda")
    k, n = 256, 128
    codes, packed, scales = _packed_buffers(
        layout=layout,
        k=k,
        n=n,
        device=device,
    )
    zeros = pack_3bit(
        torch.full((k // GROUP_SIZE, n), ZERO, dtype=torch.int32),
        axis=1,
    ).to(device)
    bias = torch.linspace(-0.25, 0.25, n, dtype=torch.float16, device=device)

    linear_cls = TritonV2Linear if layout == LAYOUT_GPTQ else AwqGEMMTritonLinear
    module = linear_cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=True,
        pack_dtype=torch.int32,
        register_buffers=True,
    ).to(device)
    module.qweight.copy_(packed)
    module.qzeros.copy_(zeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)
    if layout == LAYOUT_GPTQ:
        module.g_idx.copy_(torch.arange(k, dtype=torch.int32, device=device) // GROUP_SIZE)
    module.post_init()
    module.eval()

    if m > 16 and not hasattr(module, "_trilin_marlin_qweight"):
        pytest.skip("native Marlin JIT extension is unavailable")
    if not module._trilin_native_3bit:
        pytest.skip("native Trilin JIT extension is unavailable")

    runtime_names = {
        "_trilin_marlin_qweight",
        "_trilin_marlin_scales",
        "_trilin_marlin_workspace",
        "_trilin_marlin_empty",
    }
    assert runtime_names.isdisjoint(module.state_dict())

    from gptqmodel.nn_modules.triton_utils import three_bit as three_bit_module

    def reject_triton_fallback(*_args, **_kwargs):
        raise AssertionError("eligible native inference unexpectedly used the Triton fallback")

    monkeypatch.setattr(three_bit_module, "matmul_3bit", reject_triton_fallback)
    if m <= 16:
        monkeypatch.setattr(three_bit_module, "matmul_marlin_3bit", reject_triton_fallback)
    else:
        monkeypatch.setattr(three_bit_module, "matmul_trilin_3bit", reject_triton_fallback)
    x = torch.randn((1, m, k), dtype=dtype, device=device)
    expected = (torch.matmul(x.reshape(m, k).float(), _dense_weight(codes, scales)) + bias.float()).to(dtype)
    expected = expected.reshape(1, m, n)
    actual = module(x)

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Trilin group warning test")
def test_trilin_warns_once_for_non_primary_positive_group_size(monkeypatch):
    from gptqmodel.nn_modules.triton_utils import three_bit as three_bit_module

    class OnceRecorder:
        def __init__(self):
            self.messages = []

        def once(self, message):
            if message not in self.messages:
                self.messages.append(message)

    class LogRecorder:
        def __init__(self):
            self.warn = OnceRecorder()
            self.info = OnceRecorder()

    recorder = LogRecorder()
    monkeypatch.setattr(three_bit_module, "log", recorder)
    k, n, group_size = 256, 64, 64
    qweight = torch.zeros((k // 32 * 3, n), dtype=torch.int32, device="cuda")
    scales = torch.ones((k // group_size, n), dtype=torch.float16, device="cuda")

    assert prepare_trilin_3bit(qweight, scales, group_size)
    assert prepare_trilin_3bit(qweight, scales, group_size)
    assert len(recorder.warn.messages) == 1
    assert "group_size=128 remains the fully optimized contract" in recorder.warn.messages[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native grouped Trilin quality test")
@pytest.mark.parametrize(
    "group_size,k",
    [
        (16, 4096),
        (32, 4096),
        (64, 4096),
        (96, 3072),
        (192, 3072),
        (256, 4096),
        (384, 3072),
        (512, 4096),
        (1024, 4096),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 3])
def test_native_grouped_trilin_matches_fp32_dequant_reference(
    group_size: int,
    k: int,
    dtype: torch.dtype,
    m: int,
):
    torch.manual_seed(59)
    device = torch.device("cuda")
    n = 512
    codes, qweight, scales = _packed_buffers(
        layout=LAYOUT_GPTQ,
        k=k,
        n=n,
        device=device,
        group_size=group_size,
    )
    x = torch.randn((m, k), device=device, dtype=dtype)
    bias = torch.linspace(-0.25, 0.25, n, device=device, dtype=torch.float16)
    expected = (
        torch.matmul(x.float(), _dense_weight(codes, scales, group_size=group_size)) + bias.float()
    ).to(dtype)

    actual = trilin_matmul(x, qweight, scales, bias, group_size)

    metrics = _quality_metrics(actual, expected)
    limits = {
        torch.float16: {"max_abs": 0.125, "mean_abs": 0.002, "relative_rmse": 0.0002, "cosine": 0.999999},
        torch.bfloat16: {"max_abs": 0.5, "mean_abs": 0.01, "relative_rmse": 0.001, "cosine": 0.99999},
    }[dtype]
    assert metrics["max_abs"] <= limits["max_abs"], metrics
    assert metrics["mean_abs"] <= limits["mean_abs"], metrics
    assert metrics["relative_rmse"] <= limits["relative_rmse"], metrics
    assert metrics["cosine"] >= limits["cosine"], metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native Trilin validation test")
def test_native_trilin_rejects_unsupported_group_size():
    device = torch.device("cuda")
    k, n = 128, 64
    qweight = torch.zeros((k // 32 * 3, n), dtype=torch.int32, device=device)
    scales = torch.ones((k // 16, n), dtype=torch.float16, device=device)
    x = torch.zeros((1, k), dtype=torch.float16, device=device)

    with pytest.raises(
        RuntimeError,
        match="group_size must be 16, 32, 64, 96, 128, 192, 256, 384, 512, or 1024",
    ):
        trilin_matmul(x, qweight, scales, group_size=24)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native Trilin M=1 quality test")
@pytest.mark.parametrize(
    "dtype,bias_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_native_trilin_3bit_m1_matches_fp32_dequant_reference(
    dtype: torch.dtype,
    bias_dtype: torch.dtype,
):
    torch.manual_seed(53)
    device = torch.device("cuda")
    m, k, n = 1, 4096, 512
    codes, qweight, scales = _packed_buffers(
        layout=LAYOUT_GPTQ,
        k=k,
        n=n,
        device=device,
    )
    x = torch.randn((m, k), device=device, dtype=dtype)
    bias = torch.linspace(-0.25, 0.25, n, device=device, dtype=bias_dtype)
    expected = (torch.matmul(x.float(), _dense_weight(codes, scales)) + bias.float()).to(dtype)

    actual = trilin_matmul(x, qweight, scales, bias)

    metrics = _quality_metrics(actual, expected)
    limits = {
        torch.float16: {"max_abs": 0.0625, "mean_abs": 0.0005, "relative_rmse": 0.0001, "cosine": 0.999999},
        torch.bfloat16: {"max_abs": 0.25, "mean_abs": 0.005, "relative_rmse": 0.0005, "cosine": 0.999999},
    }[dtype]
    assert metrics["max_abs"] <= limits["max_abs"], metrics
    assert metrics["mean_abs"] <= limits["mean_abs"], metrics
    assert metrics["relative_rmse"] <= limits["relative_rmse"], metrics
    assert metrics["cosine"] >= limits["cosine"], metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native Trilin specialization test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", [1024, 4096, 11008, 14336])
def test_native_trilin_3bit_projection_width_specializations_match_torch_and_fp32_references(
    dtype: torch.dtype,
    n: int,
):
    torch.manual_seed(107)
    device = torch.device("cuda")
    m, k = 1, 4096
    codes = torch.randint(0, 8, (k, n), dtype=torch.int32)
    qweight = pack_3bit(codes, axis=0).to(device)
    scales = torch.rand((k // GROUP_SIZE, n), dtype=torch.float32).mul_(0.5).add_(0.03125).half().to(device)
    x = torch.randn((m, k), device=device, dtype=dtype)
    bias = torch.linspace(-0.25, 0.25, n, device=device, dtype=dtype)
    dense_weight = _dense_weight(codes.to(device), scales)
    oracle = (torch.matmul(x.float(), dense_weight) + bias.float()).to(dtype)
    torch_reference = (torch.matmul(x, dense_weight.to(dtype)) + bias).to(dtype)

    actual = trilin_matmul(x, qweight, scales, bias)

    torch_metrics = _quality_metrics(actual, torch_reference)
    # BF16 Torch matmul and the FP32-accumulating kernel can round isolated outputs on opposite dtype boundaries.
    torch_limits = {
        torch.float16: {"max_abs": 0.25, "mean_abs": 0.02, "relative_rmse": 0.001, "cosine": 0.99999},
        torch.bfloat16: {"max_abs": 1.0, "mean_abs": 0.125, "relative_rmse": 0.004, "cosine": 0.99999},
    }[dtype]
    assert torch_metrics["max_abs"] <= torch_limits["max_abs"], torch_metrics
    assert torch_metrics["mean_abs"] <= torch_limits["mean_abs"], torch_metrics
    assert torch_metrics["relative_rmse"] <= torch_limits["relative_rmse"], torch_metrics
    assert torch_metrics["cosine"] >= torch_limits["cosine"], torch_metrics

    oracle_metrics = _quality_metrics(actual, oracle)
    oracle_limits = {
        torch.float16: {"max_abs": 0.0625, "mean_abs": 0.0005, "relative_rmse": 0.0001, "cosine": 0.999999},
        torch.bfloat16: {"max_abs": 0.5, "mean_abs": 0.005, "relative_rmse": 0.0005, "cosine": 0.999999},
    }[dtype]
    assert oracle_metrics["max_abs"] <= oracle_limits["max_abs"], oracle_metrics
    assert oracle_metrics["mean_abs"] <= oracle_limits["mean_abs"], oracle_metrics
    assert oracle_metrics["relative_rmse"] <= oracle_limits["relative_rmse"], oracle_metrics
    assert oracle_metrics["cosine"] >= oracle_limits["cosine"], oracle_metrics

    torch_oracle_metrics = _quality_metrics(torch_reference, oracle)
    assert oracle_metrics["mean_abs"] <= torch_oracle_metrics["mean_abs"], {
        "actual": oracle_metrics,
        "reference": torch_oracle_metrics,
    }
    assert oracle_metrics["relative_rmse"] <= torch_oracle_metrics["relative_rmse"], {
        "actual": oracle_metrics,
        "reference": torch_oracle_metrics,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native Trilin BF16 quality test")
@pytest.mark.parametrize("m", [1, 16])
def test_native_trilin_3bit_bf16_random_scales_matches_torch_and_fp32_references(m: int):
    torch.manual_seed(101)
    device = torch.device("cuda")
    k, n = 4096, 512
    codes = torch.randint(0, 8, (k, n), dtype=torch.int32)
    qweight = pack_3bit(codes, axis=0).to(device)
    scales = torch.rand((k // GROUP_SIZE, n), dtype=torch.float32).mul_(0.5).add_(0.03125).half().to(device)
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    bias = torch.linspace(-0.25, 0.25, n, device=device, dtype=torch.float16)
    dense_weight = _dense_weight(codes.to(device), scales)
    oracle = (torch.matmul(x.float(), dense_weight) + bias.float()).bfloat16()
    torch_reference = (torch.matmul(x, dense_weight.bfloat16()) + bias).bfloat16()

    actual = trilin_matmul(x, qweight, scales, bias)

    torch_metrics = _quality_metrics(actual, torch_reference)
    assert torch_metrics["max_abs"] <= 1.0, torch_metrics
    assert torch_metrics["mean_abs"] <= 0.1, torch_metrics
    assert torch_metrics["relative_rmse"] <= 0.004, torch_metrics
    assert torch_metrics["cosine"] >= 0.99999, torch_metrics

    reference_oracle_metrics = _quality_metrics(torch_reference, oracle)
    actual_oracle_metrics = _quality_metrics(actual, oracle)
    assert actual_oracle_metrics["mean_abs"] <= reference_oracle_metrics["mean_abs"], {
        "actual": actual_oracle_metrics,
        "reference": reference_oracle_metrics,
    }
    assert actual_oracle_metrics["relative_rmse"] <= reference_oracle_metrics["relative_rmse"], {
        "actual": actual_oracle_metrics,
        "reference": reference_oracle_metrics,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native Trilin stream test")
@pytest.mark.parametrize("m", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_trilin_3bit_uses_current_stream(m: int, dtype: torch.dtype):
    torch.manual_seed(47)
    device = torch.device("cuda")
    k, n = 256, 128
    codes, qweight, scales = _packed_buffers(
        layout=LAYOUT_GPTQ,
        k=k,
        n=n,
        device=device,
    )
    x = torch.randn((m, k), device=device, dtype=dtype)
    expected = torch.matmul(x.float(), _dense_weight(codes, scales)).to(dtype)
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        actual = trilin_matmul(x, qweight, scales)
        completion = torch.cuda.Event()
        completion.record(stream)
    torch.cuda.current_stream(device=device).wait_event(completion)

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native Trilin graph test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_trilin_3bit_m1_supports_cuda_graph_capture(dtype: torch.dtype):
    torch.manual_seed(31)
    device = torch.device("cuda")
    k, n = 4096, 4096
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (k // 32 * 3, n),
        dtype=torch.int32,
        device=device,
    )
    scales = torch.rand((k // GROUP_SIZE, n), dtype=torch.float16, device=device)
    x = torch.randn((1, k), device=device, dtype=dtype)
    expected = trilin_matmul(x, qweight, scales).clone()
    warmup_stream = torch.cuda.Stream(device=device)

    warmup_stream.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            trilin_matmul(x, qweight, scales)
    torch.cuda.current_stream(device=device).wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = trilin_matmul(x, qweight, scales)
    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton 3-bit persistence test")
@pytest.mark.parametrize("layout", [LAYOUT_GPTQ, LAYOUT_AWQ])
@pytest.mark.parametrize("group_size", [64, 96, 128, 192, 384, -1])
def test_3bit_pack_save_reload_inference(layout: str, group_size: int):
    pytest.importorskip("triton")
    torch.manual_seed(23)
    m, k, n = 5, 384 if group_size in {96, 192, 384} else 256, 96
    effective_group_size = k if group_size == -1 else group_size
    codes = _codes(k, n)
    scales_group_n = _scales(k, n, group_size=group_size)
    zeros_group_n = torch.full_like(scales_group_n, ZERO)
    linear = torch.nn.Linear(k, n, bias=False, dtype=torch.float32)
    linear.weight.data.copy_(_dense_weight(codes, scales_group_n, group_size=group_size).t())

    linear_cls = TritonV2Linear if layout == LAYOUT_GPTQ else AwqGEMMTritonLinear
    source = linear_cls(
        bits=BITS,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=False,
    )
    g_idx = torch.arange(k, dtype=torch.int32) // effective_group_size
    if layout == LAYOUT_GPTQ:
        source.pack_original(
            linear=linear,
            scales=scales_group_n.t().contiguous(),
            zeros=zeros_group_n.t().contiguous(),
            g_idx=g_idx,
        )
    else:
        source.pack(
            linear=linear,
            scales=scales_group_n.t().contiguous(),
            zeros=zeros_group_n.t().contiguous(),
            g_idx=None,
        )

    state_buffer = io.BytesIO()
    torch.save(source.state_dict(), state_buffer)
    state_buffer.seek(0)
    restored_state = torch.load(state_buffer, map_location="cpu", weights_only=True)

    restored = linear_cls(
        bits=BITS,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    restored.load_state_dict(restored_state, strict=True)
    restored = restored.cuda()
    restored.post_init()
    restored.eval()

    x = torch.randn((1, m, k), device="cuda", dtype=torch.float16)
    expected_weight = _dense_weight(codes.cuda(), restored.scales, group_size=group_size)
    expected = torch.matmul(x.reshape(m, k).float(), expected_weight).half().reshape(1, m, n)
    actual = restored(x)

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


def test_awq_3bit_pack_rejects_non_symmetric_zero_points():
    k, n = 128, 64
    scales_group_n = _scales(k, n)
    zeros_group_n = torch.full_like(scales_group_n, ZERO)
    zeros_group_n[0, 17] = ZERO - 1
    linear = torch.nn.Linear(k, n, bias=False, dtype=torch.float32)
    module = AwqGEMMTritonLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=False,
    )

    with pytest.raises(ValueError, match="every zero point to equal 4"):
        module.pack(
            linear=linear,
            scales=scales_group_n.t().contiguous(),
            zeros=zeros_group_n.t().contiguous(),
        )


def test_awq_3bit_post_init_rejects_checkpoint_with_non_symmetric_zero_points():
    k, n = 128, 64
    module = AwqGEMMTritonLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    zeros = torch.full((k // GROUP_SIZE, n), ZERO, dtype=torch.int32)
    zeros[0, 17] = ZERO - 1
    module.qzeros.copy_(pack_3bit(zeros, axis=1))

    with pytest.raises(ValueError, match="every zero point to equal 4"):
        module.post_init()


def test_gptq_3bit_post_init_rejects_non_natural_group_indices():
    k, n = 256, 64
    module = TritonV2Linear(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    module.g_idx[127] = 1

    with pytest.raises(ValueError, match="natural group indices"):
        module.post_init()


def test_gptq_3bit_post_init_rejects_checkpoint_with_non_symmetric_zero_points():
    k, n = 128, 64
    module = TritonV2Linear(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    zeros = torch.full((k // GROUP_SIZE, n), ZERO, dtype=torch.int32)
    zeros[0, 17] = ZERO - 1
    module.qzeros.copy_(pack_3bit(zeros, axis=1))

    with pytest.raises(ValueError, match="every zero point to equal 4"):
        module.post_init()
