# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.awq.modules.triton.gemm import awq_gemm_triton
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm


_AWQ_OUTPUT_QUALITY_LIMITS = {"max_abs": 0.02, "mean_abs": 0.002, "relative_l2": 5e-4}


def _assert_awq_output_quality(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    backend: str,
    group_size: int,
    sequence_length: int,
) -> None:
    context = (
        f"method=AWQ format=GEMM bits=4 group_size={group_size} sym=True desc_act=False "
        f"dtype=torch.float16 backend={backend} sequence_length={sequence_length}"
    )
    assert output.shape == reference.shape, context
    assert output.dtype == torch.float16, context
    assert output.device == reference.device, context
    assert torch.isfinite(output).all(), context

    error = output.float() - reference
    max_abs = error.abs().max().item()
    mean_abs = error.abs().mean().item()
    relative_l2 = (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference)).item()

    assert max_abs <= _AWQ_OUTPUT_QUALITY_LIMITS["max_abs"], (
        f"{context} max_abs={max_abs:.8f} limit={_AWQ_OUTPUT_QUALITY_LIMITS['max_abs']:.8f}"
    )
    assert mean_abs <= _AWQ_OUTPUT_QUALITY_LIMITS["mean_abs"], (
        f"{context} mean_abs={mean_abs:.8f} limit={_AWQ_OUTPUT_QUALITY_LIMITS['mean_abs']:.8f}"
    )
    assert relative_l2 <= _AWQ_OUTPUT_QUALITY_LIMITS["relative_l2"], (
        f"{context} relative_l2={relative_l2:.8f} limit={_AWQ_OUTPUT_QUALITY_LIMITS['relative_l2']:.8f}"
    )


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
    bias = torch.randn(out_features, dtype=torch.float16)

    return (
        _pack_awq_tensor(int_weight, bits),
        _pack_awq_tensor(zero_points, bits),
        scales,
        bias,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for extended AWQ group-size parity test")
@pytest.mark.parametrize("group_size", [96, 192, 256, 384, 512])
@pytest.mark.parametrize("sequence_length", [1, 8, 129])
def test_awq_torch_triton_extended_group_size_output_quality(group_size, sequence_length):
    pytest.importorskip("triton")
    torch.manual_seed(1000 + group_size + sequence_length)

    bits = 4
    in_features = 1536
    out_features = 512
    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, out_features), dtype=torch.int32)
    qweight = _pack_awq_tensor(int_weight, bits).cuda()
    qzeros = _pack_awq_tensor(zero_points, bits).cuda()
    scales = ((torch.rand(groups, out_features, dtype=torch.float16) * 0.02) + 0.01).cuda()

    modules = []
    for module_cls in (AwqTorchLinear, AwqGEMMTritonLinear):
        module = module_cls(
            bits=bits,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            dtype=torch.float16,
            register_buffers=True,
        ).cuda().eval()
        module.qweight.copy_(qweight)
        module.qzeros.copy_(qzeros)
        module.scales.copy_(scales)
        module.post_init()
        modules.append(module)

    x = torch.randn(1, sequence_length, in_features, device="cuda", dtype=torch.float16)
    dense_weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bits=bits,
        group_size=group_size,
    ).to(device=x.device, dtype=x.dtype)
    reference = torch.matmul(x.float(), dense_weight.float())

    with torch.inference_mode():
        torch_output, triton_output = (module(x) for module in modules)

    _assert_awq_output_quality(
        torch_output,
        reference,
        backend="AwqTorchLinear",
        group_size=group_size,
        sequence_length=sequence_length,
    )
    _assert_awq_output_quality(
        triton_output,
        reference,
        backend="AwqGEMMTritonLinear",
        group_size=group_size,
        sequence_length=sequence_length,
    )
    torch.testing.assert_close(triton_output, torch_output, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ Triton kernel parity test")
def test_awq_triton_fp32_accum_matches_manual_dequant():
    pytest.importorskip("triton")
    torch.manual_seed(0)

    bits = 4
    in_features = 512
    out_features = 512
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    module = AwqGEMMTritonLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    ).cuda()
    module.qweight.copy_(qweight.cuda())
    module.qzeros.copy_(qzeros.cuda())
    module.scales.copy_(scales.cuda())
    module.bias.copy_(bias.cuda())
    module.post_init()
    module.eval()

    x = torch.randn(1, 128, in_features, device="cuda", dtype=torch.float16)
    dequant_weight = dequantize_gemm(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales,
        bits=bits,
        group_size=group_size,
    ).to(device=x.device, dtype=x.dtype)
    expected = torch.matmul(x.reshape(-1, in_features), dequant_weight).reshape(1, 128, out_features)
    expected = expected + module.bias

    with torch.inference_mode():
        actual = module(x)

    abs_diff = (actual - expected).abs()
    assert abs_diff.max().item() <= 1.0
    assert abs_diff.mean().item() <= 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ Triton accumulation test")
def test_awq_triton_fp32_accum_reduces_dense_error():
    pytest.importorskip("triton")
    torch.manual_seed(0)

    bits = 4
    batch = 1
    seq = 128
    in_features = 1024
    out_features = 1024
    group_size = 128
    qweight, qzeros, scales, _bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    x = torch.randn(batch * seq, in_features, device="cuda", dtype=torch.float16)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda()

    dense_weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bits=bits,
        group_size=group_size,
    ).to(device=x.device, dtype=x.dtype)
    reference = torch.matmul(x, dense_weight)

    with torch.inference_mode():
        legacy = awq_gemm_triton(
            x,
            qweight,
            scales,
            qzeros,
            split_k_iters=8,
            fp32_accum=False,
            output_dtype=x.dtype,
        )
        candidate = awq_gemm_triton(
            x,
            qweight,
            scales,
            qzeros,
            split_k_iters=8,
            fp32_accum=True,
            output_dtype=x.dtype,
        )

    legacy_abs = (legacy - reference).abs()
    candidate_abs = (candidate - reference).abs()

    assert candidate_abs.max().item() < legacy_abs.max().item()
    assert candidate_abs.mean().item() < legacy_abs.mean().item()
