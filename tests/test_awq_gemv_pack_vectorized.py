# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.utils.gemv import calculate_zeros_width


def _reference_qscales(scales: torch.Tensor, in_features: int, group_size: int, pack_num: int) -> torch.Tensor:
    qscales = torch.zeros(
        (scales.shape[0], calculate_zeros_width(in_features, group_size) * pack_num),
        dtype=torch.float16,
        device=scales.device,
    )
    qscales[:, : scales.shape[1]] = scales
    return qscales


def _reference_gemv_awq_pack(module, linear, scales, zeros, qscales):
    """The hand-written per-column pack loop the vectorized packer replaces."""

    scale_zeros = zeros * scales
    pack_num = 32 // module.bits

    intweight = []
    for idx in range(module.in_features):
        intweight.append(
            torch.round(
                (linear.weight.data[:, idx] + scale_zeros[:, idx // module.group_size])
                / qscales[:, idx // module.group_size]
            ).to(torch.int)[:, None]
        )
    intweight = torch.cat(intweight, dim=1).to(dtype=torch.int32)
    qweight = torch.zeros(
        (intweight.shape[0], intweight.shape[1] // 32 * module.bits),
        dtype=torch.int32,
        device=intweight.device,
    )
    for col in range(intweight.shape[1] // pack_num):
        order_map = list(range(pack_num))
        for i in range(pack_num):
            qweight[:, col] |= intweight[:, col * pack_num + order_map[i]] << (i * module.bits)

    zeros = zeros.to(dtype=torch.int32)
    qzeros = torch.zeros(
        (zeros.shape[0], calculate_zeros_width(module.in_features, module.group_size)),
        dtype=torch.int32,
        device=zeros.device,
    )
    for col in range((zeros.shape[1] + pack_num - 1) // pack_num):
        order_map = list(range(pack_num))
        for i in range(pack_num):
            if col * pack_num + order_map[i] >= zeros.shape[1]:
                continue
            qzeros[:, col] |= zeros[:, col * pack_num + order_map[i]] << (i * module.bits)

    return qweight, qzeros


def test_gemv_awq_pack_is_bit_exact_with_column_loop():
    torch.manual_seed(0)

    # (out_features, in_features, group_size) with out_features % 64 == 0 to
    # satisfy the GEMMV2 tile contract; 1280/64 exercises padded zero words.
    for out_features, in_features, group_size in (
        (64, 256, 128),
        (64, 1280, 64),
        (128, 512, 128),
        (64, 128, 128),
    ):
        linear = nn.Linear(in_features, out_features, bias=False)
        module = AwqGEMVLinear(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        )

        groups = in_features // group_size
        weight = linear.weight.data
        weight_groups = weight.view(out_features, groups, group_size)
        weight_min = weight_groups.min(dim=2).values
        weight_max = weight_groups.max(dim=2).values
        scales = ((weight_max - weight_min) / 15.0).to(torch.float16)
        zeros = torch.round(-weight_min / scales.float()).to(torch.int32)

        qscales = _reference_qscales(scales, in_features, group_size, 32 // module.bits)
        expected_qweight, expected_qzeros = _reference_gemv_awq_pack(
            module, linear, scales.clone(), zeros.clone(), qscales
        )

        module.pack(linear, scales.clone(), zeros.clone())

        assert module.qweight.shape == expected_qweight.shape
        assert module.qzeros.shape == expected_qzeros.shape
        assert torch.equal(module.qweight, expected_qweight)
        assert torch.equal(module.qzeros, expected_qzeros)
