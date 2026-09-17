# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.quantization.awq.modules.triton.gemm import awq_gemm_triton
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ Triton dispatch coverage")
@pytest.mark.parametrize("group_size", [32, 64, 128, -1])
@pytest.mark.parametrize("split_k_iters", [1, 2, 4, 8])
@pytest.mark.parametrize("seed,amplitude", [(0, 0.25), (7, 1.0)])
def test_awq_triton_supported_groups_splits_and_tails(group_size, split_k_iters, seed, amplitude):
    """Exercise the legal packing contract, tails and each atomic split count."""
    pytest.importorskip("triton")
    torch.manual_seed(seed)
    K, N, M = 256, 264, 3  # N and M intentionally miss the 32 tile.
    actual_group = K if group_size == -1 else group_size
    qweight, qzeros, scales, _ = _make_packed_buffers(4, K, N, actual_group)
    x = (torch.randn(M, K, device="cuda", dtype=torch.float16) * amplitude)
    qweight, qzeros, scales = qweight.cuda(), qzeros.cuda(), scales.cuda()
    expected_weight = dequantize_gemm(
        qweight=qweight, qzeros=qzeros, scales=scales, bits=4, group_size=actual_group,
    ).to(device=x.device, dtype=x.dtype)
    expected = torch.matmul(x, expected_weight)
    actual = awq_gemm_triton(
        x, qweight, scales, qzeros, split_k_iters=split_k_iters,
        block_size_m=32, block_size_n=32, block_size_k=32,
        fp32_accum=True, output_dtype=x.dtype,
    )
    diff = (actual - expected).abs()
    assert torch.isfinite(actual).all()
    assert diff.max().item() <= 1.0
    assert diff.mean().item() <= 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ Triton split-1 test")
def test_awq_triton_split1_repeated_calls_have_no_uninitialized_output():
    pytest.importorskip("triton")
    K, N = 320, 264
    qweight, qzeros, scales, _ = _make_packed_buffers(4, K, N, 64)
    qweight, qzeros, scales = qweight.cuda(), qzeros.cuda(), scales.cuda()
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        x = torch.randn(5, K, device="cuda", dtype=torch.float16)
        expected = torch.matmul(x, dequantize_gemm(
            qweight=qweight, qzeros=qzeros, scales=scales, bits=4, group_size=64,
        ).to(device=x.device, dtype=x.dtype))
        actual = awq_gemm_triton(
            x, qweight, scales, qzeros, split_k_iters=1,
            block_size_m=32, block_size_n=64, block_size_k=32,
            fp32_accum=True, output_dtype=x.dtype,
        )
        diff = (actual - expected).abs()
        assert diff.max().item() <= 1.0
        assert diff.mean().item() <= 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ Triton uneven split test")
def test_awq_triton_split8_handles_uneven_k_tile_work():
    pytest.importorskip("triton")
    torch.manual_seed(5)
    K, N, G = 320, 264, 64  # Ten BK=32 tiles distributed across eight splits.
    qweight, qzeros, scales, _ = _make_packed_buffers(4, K, N, G)
    qweight, qzeros, scales = qweight.cuda(), qzeros.cuda(), scales.cuda()
    x = torch.randn(3, K, device="cuda", dtype=torch.float16)
    expected = torch.matmul(x, dequantize_gemm(
        qweight=qweight, qzeros=qzeros, scales=scales, bits=4, group_size=G,
    ).to(device=x.device, dtype=x.dtype))
    actual = awq_gemm_triton(
        x, qweight, scales, qzeros, split_k_iters=8,
        block_size_m=32, block_size_n=32, block_size_k=32,
        fp32_accum=True, output_dtype=x.dtype,
    )
    diff = (actual - expected).abs()
    assert diff.max().item() <= 1.0
    assert diff.mean().item() <= 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ Triton module backward test")
def test_awq_triton_module_noncontiguous_bias_and_backward():
    pytest.importorskip("triton")
    torch.manual_seed(11)
    K, N, G = 128, 264, 128
    qweight, qzeros, scales, bias = _make_packed_buffers(4, K, N, G)
    module = AwqGEMMTritonLinear(
        bits=4, group_size=G, sym=True, desc_act=False,
        in_features=K, out_features=N, bias=True, register_buffers=True,
    ).cuda()
    module.qweight.copy_(qweight.cuda())
    module.qzeros.copy_(qzeros.cuda())
    module.scales.copy_(scales.cuda())
    module.bias.copy_(bias.cuda())
    module.post_init()
    module.train()
    # Shape is [batch, sequence, K], but storage is deliberately non-contiguous.
    x = torch.randn(2, 3, K, device="cuda", dtype=torch.float16).transpose(0, 1).requires_grad_()
    actual = module(x)
    weight = dequantize_gemm(
        qweight=module.qweight, qzeros=module.qzeros, scales=module.scales,
        bits=4, group_size=G,
    ).to(device=x.device, dtype=x.dtype)
    expected = torch.matmul(x.reshape(-1, K), weight).reshape(x.shape[:-1] + (N,)) + module.bias
    diff = (actual - expected).abs()
    assert diff.max().item() <= 1.0
    assert diff.mean().item() <= 0.02
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected_grad = torch.matmul(grad_output.reshape(-1, N), weight.t()).reshape_as(x)
    grad_diff = (x.grad - expected_grad).abs()
    assert grad_diff.max().item() <= 1.0
    assert grad_diff.mean().item() <= 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ Triton CUDA Graph test")
def test_awq_triton_cuda_graph_reuses_eagerly_warmed_plan():
    pytest.importorskip("triton")
    torch.manual_seed(17)
    K, N, G = 256, 264, 64
    qweight, qzeros, scales, bias = _make_packed_buffers(4, K, N, G)
    module = AwqGEMMTritonLinear(
        bits=4, group_size=G, sym=True, desc_act=False,
        in_features=K, out_features=N, bias=True, register_buffers=True,
    ).cuda().eval()
    module.qweight.copy_(qweight.cuda())
    module.qzeros.copy_(qzeros.cuda())
    module.scales.copy_(scales.cuda())
    module.bias.copy_(bias.cuda())
    module.post_init()
    static_x = torch.randn(3, K, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        module(static_x)  # Eagerly compiles and records the exact plan.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(static_x)

        replay_x = torch.randn_like(static_x)
        static_x.copy_(replay_x)
        graph.replay()
        actual = captured.clone()
        expected = module(replay_x)

    diff = (actual - expected).abs()
    assert diff.max().item() <= 1.0
    assert diff.mean().item() <= 0.02
