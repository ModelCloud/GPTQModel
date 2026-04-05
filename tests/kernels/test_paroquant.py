# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant kernel tests adapted from the ParoQuant paper and public project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""Kernel-focused tests for ParoQuant runtime behavior and backend parity."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import get_kernel_for_backend, select_quant_linear
from gptqmodel.utils.paroquant import apply_paroquant_rotation_reference, build_identity_rotation_buffers


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unpacked integer weights into the AWQ bit layout used by the kernels."""
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
    """Build synthetic packed AWQ tensors for ParoQuant runtime tests."""
    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, out_features), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 2.0) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)

    return (
        _pack_awq_tensor(int_weight, bits),
        _pack_awq_tensor(zero_points, bits),
        scales,
        bias,
    )


def _upstream_transformers_contract_reference(
    x: torch.Tensor,
    *,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    channel_scales: torch.Tensor,
    bits: int,
    group_size: int,
    out_features: int,
) -> torch.Tensor:
    """Reference for upstream RotateQuantizedLinear.forward().

    Upstream ParoQuant applies per-projection rotation to the input and then
    feeds the rotated activations into the AWQ GEMM path. We reproduce that
    contract with dense dequantization here to assess kernel accuracy without
    importing or copying upstream implementation code.
    """

    rotated = apply_paroquant_rotation_reference(
        x,
        pairs,
        theta,
        scales=channel_scales,
        group_size=group_size,
    )
    dense_weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bits=bits,
        group_size=group_size,
    ).to(device=x.device, dtype=x.dtype)
    out = torch.matmul(rotated.reshape(-1, x.shape[-1]), dense_weight).reshape(*x.shape[:-1], out_features)
    if bias is not None:
        out = out + bias.to(device=x.device, dtype=x.dtype)
    return out


def test_paroquant_identity_forward_matches_awq_torch():
    """Guard that identity ParoQuant is behaviorally equivalent to plain AWQ."""
    bits = 4
    in_features = 128
    out_features = 64
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    awq_module = AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    )
    awq_module.qweight.copy_(qweight)
    awq_module.qzeros.copy_(qzeros)
    awq_module.scales.copy_(scales)
    awq_module.bias.copy_(bias)
    awq_module.post_init()
    awq_module.eval()

    paro_module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    )
    paro_module.qweight.copy_(qweight)
    paro_module.qzeros.copy_(qzeros)
    paro_module.scales.copy_(scales)
    paro_module.bias.copy_(bias)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        dtype=torch.float16,
    )
    paro_module.pairs.copy_(pairs)
    paro_module.theta.copy_(theta)
    paro_module.channel_scales.copy_(channel_scales)
    paro_module.post_init()
    paro_module.eval()

    x = torch.randn(4, in_features, dtype=torch.float16)
    torch.testing.assert_close(paro_module(x), awq_module(x), atol=5e-3, rtol=5e-3)


def test_paroquant_forward_matches_explicit_rotated_reference():
    """Guard the dense reference contract for non-identity ParoQuant rotations."""
    bits = 4
    in_features = 128
    out_features = 64
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=1,
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=1,
        dtype=torch.float16,
    )
    theta.fill_(0.2)
    channel_scales.mul_(0.75)
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()

    x = torch.randn(3, in_features, dtype=torch.float16)
    rotated = apply_paroquant_rotation_reference(
        x,
        module.pairs,
        module.theta,
        scales=module.channel_scales,
        group_size=group_size,
    )
    dequant_weight = dequantize_gemm(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales,
        bits=bits,
        group_size=group_size,
    ).to(dtype=x.dtype)
    expected = torch.matmul(rotated, dequant_weight) + module.bias

    torch.testing.assert_close(module(x), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant kernel accuracy test")
def test_paroquant_cuda_matches_upstream_transformers_contract():
    """Compare the internal CUDA kernel path to an upstream-style contract.

    The official implementation rotates activations and then runs an AWQ-style
    packed matmul. This test reproduces that contract without importing upstream
    code and checks our fused CUDA path stays within a bounded numerical error.
    """
    bits = 4
    in_features = 128
    out_features = 128
    group_size = 128
    torch.manual_seed(0)

    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.full((groups, out_features), 2 ** (bits - 1), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 0.75) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)
    qweight = _pack_awq_tensor(int_weight, bits)
    qzeros = _pack_awq_tensor(zero_points, bits)

    module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    ).cuda()

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        device="cuda",
        dtype=torch.float16,
    )
    theta.uniform_(-0.25, 0.25)
    channel_scales.uniform_(0.7, 1.3)

    module.qweight.copy_(qweight.cuda())
    module.qzeros.copy_(qzeros.cuda())
    module.scales.copy_(scales.cuda())
    module.bias.copy_(bias.cuda())
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()

    x = torch.randn(3, 7, in_features, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        expected = _upstream_transformers_contract_reference(
            x,
            qweight=module.qweight,
            qzeros=module.qzeros,
            scales=module.scales,
            bias=module.bias,
            pairs=module.pairs,
            theta=module.theta,
            channel_scales=module.channel_scales,
            bits=bits,
            group_size=group_size,
            out_features=out_features,
        )

        original_forward_dense = module._forward_dense
        try:
            module._forward_dense = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Expected ParoQuant CUDA kernel path, but dense fallback was used.")
            )
            actual = module(x)
        finally:
            module._forward_dense = original_forward_dense

    diff = (actual - expected).abs().float()
    assert diff.max().item() <= 0.25
    assert diff.mean().item() <= 0.03


def test_paroquant_backend_selection():
    """Guard user-facing backend selection for the default CUDA runtime."""
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=None,
        backend=BACKEND.PARO,
        format=FORMAT.PAROQUANT,
        quant_method=METHOD.PARO,
        pack_dtype=torch.int32,
    )
    assert qlinear_cls is ParoLinear


def test_paroquant_triton_backend_mapping():
    """Guard registry lookup for the Triton ParoQuant runtime class."""
    assert (
        get_kernel_for_backend(BACKEND.PAROQUANT_TRITON, METHOD.PARO, FORMAT.PAROQUANT)
        is ParoQuantTritonLinear
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant Triton kernel parity test")
def test_paroquant_triton_matches_existing_cuda_kernel():
    """Guard Triton runtime accuracy against the established CUDA implementation."""
    pytest.importorskip("triton")

    bits = 4
    in_features = 128
    out_features = 128
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    baseline = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    ).cuda()
    candidate = ParoQuantTritonLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    ).cuda()

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        device="cuda",
        dtype=torch.float16,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)

    for module in (baseline, candidate):
        module.qweight.copy_(qweight.cuda())
        module.qzeros.copy_(qzeros.cuda())
        module.scales.copy_(scales.cuda())
        module.bias.copy_(bias.cuda())
        module.pairs.copy_(pairs)
        module.theta.copy_(theta)
        module.channel_scales.copy_(channel_scales)
        module.post_init()
        module.eval()

    x = torch.randn(2, 8, in_features, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        rotated = apply_paroquant_rotation_reference(
            x,
            baseline.pairs,
            baseline.theta,
            scales=baseline.channel_scales,
            group_size=group_size,
        )
        dense_weight = dequantize_gemm(
            qweight=baseline.qweight,
            qzeros=baseline.qzeros,
            scales=baseline.scales,
            bits=bits,
            group_size=group_size,
        ).to(dtype=x.dtype)
        dense_reference = torch.matmul(rotated.reshape(-1, in_features), dense_weight).reshape(2, 8, out_features)
        dense_reference = dense_reference + baseline.bias

        baseline_out = baseline(x)
        candidate_out = candidate(x)

    baseline_max_abs = (baseline_out - dense_reference).abs().max().item()
    baseline_mean_abs = (baseline_out - dense_reference).abs().mean().item()
    candidate_max_abs = (candidate_out - dense_reference).abs().max().item()
    candidate_mean_abs = (candidate_out - dense_reference).abs().mean().item()

    assert candidate_max_abs <= baseline_max_abs + 0.1
    assert candidate_mean_abs <= baseline_mean_abs + 0.02
