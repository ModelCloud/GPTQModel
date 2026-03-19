# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.qlinear.paroquant import ParoQuantQuantLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchQuantLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear
from gptqmodel.utils.paroquant import apply_paroquant_rotation_reference, build_identity_rotation_buffers


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
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 2.0) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)

    return (
        _pack_awq_tensor(int_weight, bits),
        _pack_awq_tensor(zero_points, bits),
        scales,
        bias,
    )


def test_paroquant_identity_forward_matches_awq_torch():
    bits = 4
    in_features = 128
    out_features = 64
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    awq_module = AwqTorchQuantLinear(
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

    paro_module = ParoQuantQuantLinear(
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
    bits = 4
    in_features = 128
    out_features = 64
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    module = ParoQuantQuantLinear(
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


def test_paroquant_backend_selection():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=None,
        backend=BACKEND.PAROQUANT,
        format=FORMAT.PAROQUANT,
        quant_method=METHOD.PAROQUANT,
        pack_dtype=torch.int32,
    )
    assert qlinear_cls is ParoQuantQuantLinear
