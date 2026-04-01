# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.utils.awq import awq_gemm_forward, awq_runtime_available
from gptqmodel.utils.paroquant import apply_paroquant_rotation, build_identity_rotation_buffers


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    assert unpacked.shape[1] % pack_factor == 0
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for i, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (i * bits)
    return packed


def _make_packed_buffers(bits: int, in_features: int, out_features: int, group_size: int):
    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, out_features), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 0.5) + 0.75

    return (
        _pack_awq_tensor(int_weight, bits),
        _pack_awq_tensor(zero_points, bits),
        scales,
    )


def _dense_reference(x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, bits: int, group_size: int):
    dense_weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bits=bits,
        group_size=group_size,
    ).to(device=x.device, dtype=x.dtype)
    return torch.matmul(x, dense_weight)


def _require_dtype_support(dtype: torch.dtype) -> None:
    if dtype != torch.bfloat16:
        return
    major, _minor = torch.cuda.get_device_capability()
    if major < 8:
        pytest.skip("BFloat16 AWQ CUDA kernels require compute capability >= 8.0.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ CUDA fp32-reduce test")
def test_awq_cuda_fp32_reduce_reduces_dense_error():
    if not awq_runtime_available():
        pytest.skip("AWQ CUDA fp32-reduce extension entrypoint unavailable.")

    torch.manual_seed(0)
    bits = 4
    batch = 1
    seq = 128
    in_features = 1024
    out_features = 1024
    group_size = 128
    qweight, qzeros, scales = _make_packed_buffers(bits, in_features, out_features, group_size)

    x = torch.randn(batch * seq, in_features, device="cuda", dtype=torch.float16)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda()

    reference = _dense_reference(x, qweight, qzeros, scales, bits=bits, group_size=group_size)

    with torch.inference_mode():
        legacy = awq_gemm_forward(x, qweight, scales, qzeros, 8, False)
        candidate = awq_gemm_forward(x, qweight, scales, qzeros, 8, True)

    legacy_abs = (legacy - reference).abs()
    candidate_abs = (candidate - reference).abs()

    assert candidate_abs.max().item() <= legacy_abs.max().item()
    assert candidate_abs.mean().item() < legacy_abs.mean().item() * 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant CUDA fp32-reduce test")
def test_paroquant_cuda_fp32_reduce_reduces_dense_error():
    if not awq_runtime_available():
        pytest.skip("AWQ CUDA fp32-reduce extension entrypoint unavailable.")

    torch.manual_seed(0)
    bits = 4
    batch = 1
    seq = 128
    in_features = 1024
    out_features = 1024
    group_size = 128
    krot = 8
    qweight, qzeros, scales = _make_packed_buffers(bits, in_features, out_features, group_size)

    x = torch.randn(batch * seq, in_features, device="cuda", dtype=torch.float16)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda()
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        device="cuda",
        dtype=torch.float16,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)
    rotated = apply_paroquant_rotation(x, pairs, theta, scales=channel_scales, group_size=group_size)

    reference = _dense_reference(rotated, qweight, qzeros, scales, bits=bits, group_size=group_size)

    with torch.inference_mode():
        legacy = awq_gemm_forward(rotated, qweight, scales, qzeros, 8, False)
        candidate = awq_gemm_forward(rotated, qweight, scales, qzeros, 8, True)

    legacy_abs = (legacy - reference).abs()
    candidate_abs = (candidate - reference).abs()

    assert candidate_abs.max().item() <= legacy_abs.max().item()
    assert candidate_abs.mean().item() < legacy_abs.mean().item() * 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ fused split-K reduction test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_cuda_fused_splitk_reduce_matches_default(dtype, monkeypatch):
    if not awq_runtime_available():
        pytest.skip("AWQ CUDA extension entrypoint unavailable.")

    _require_dtype_support(dtype)
    torch.manual_seed(0)
    bits = 4
    in_features = 1024
    out_features = 1024
    group_size = 128
    qweight, qzeros, scales = _make_packed_buffers(bits, in_features, out_features, group_size)

    x = torch.randn(128, in_features, device="cuda", dtype=dtype)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda().to(dtype=dtype)

    monkeypatch.setenv("GPTQMODEL_AWQ_DISABLE_FUSED_SPLITK_REDUCE", "1")
    with torch.inference_mode():
        baseline = awq_gemm_forward(x, qweight, scales, qzeros, 4, True)

    monkeypatch.delenv("GPTQMODEL_AWQ_DISABLE_FUSED_SPLITK_REDUCE", raising=False)
    with torch.inference_mode():
        candidate = awq_gemm_forward(x, qweight, scales, qzeros, 4, True)

    assert torch.equal(candidate, baseline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant fused split-K reduction test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paroquant_fused_splitk_reduce_matches_default(dtype, monkeypatch):
    if not awq_runtime_available():
        pytest.skip("AWQ CUDA extension entrypoint unavailable.")

    _require_dtype_support(dtype)
    torch.manual_seed(0)
    bits = 4
    in_features = 1024
    out_features = 1024
    group_size = 128
    krot = 8
    qweight, qzeros, scales = _make_packed_buffers(bits, in_features, out_features, group_size)

    x = torch.randn(128, in_features, device="cuda", dtype=dtype)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda().to(dtype=dtype)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        device="cuda",
        dtype=dtype,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)
    rotated = apply_paroquant_rotation(x, pairs, theta, scales=channel_scales, group_size=group_size)

    monkeypatch.setenv("GPTQMODEL_AWQ_DISABLE_FUSED_SPLITK_REDUCE", "1")
    with torch.inference_mode():
        baseline = awq_gemm_forward(rotated, qweight, scales, qzeros, 4, True)

    monkeypatch.delenv("GPTQMODEL_AWQ_DISABLE_FUSED_SPLITK_REDUCE", raising=False)
    with torch.inference_mode():
        candidate = awq_gemm_forward(rotated, qweight, scales, qzeros, 4, True)

    assert torch.equal(candidate, baseline)
