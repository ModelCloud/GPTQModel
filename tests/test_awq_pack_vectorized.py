# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.gemm_awq import _pack_awq_columns
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.awq.utils.packing_utils import AWQ_ORDER, pack_awq


def _reference_pack_columns(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Scalar layout oracle with each field confined to its declared width."""

    pack_num = 32 // bits
    rows = codes.shape[0]
    columns = codes.shape[1] // pack_num
    packed = torch.zeros((rows, columns), dtype=torch.int32, device=codes.device)
    for col in range(columns):
        for i in range(pack_num):
            packed[:, col] |= (codes[:, col * pack_num + AWQ_ORDER[i]].int() & ((1 << bits) - 1)) << (i * bits)
    return packed


def test_pack_awq_columns_matches_column_loop():
    torch.manual_seed(0)
    # Negative and oversized codes must not spill into adjacent fields.
    codes = torch.randint(-1, 17, (7, 64), dtype=torch.int32)

    assert torch.equal(_pack_awq_columns(codes, 4), _reference_pack_columns(codes, 4))


def _reference_torch_awq_pack(module, linear, scales, zeros):
    scales = scales.t().contiguous()
    zeros = zeros.t().contiguous()
    scale_zeros = zeros.float() * scales.float()

    scale_dtype = scales.dtype if scales.dtype in (torch.float16, torch.bfloat16) else torch.float16
    scales_buf = scales.clone().to(scale_dtype)

    intweight = []
    for idx in range(linear.in_features):
        intweight.append(
            torch.round(
                (linear.weight.data[:, idx].float() + scale_zeros[idx // module.group_size])
                / scales_buf[idx // module.group_size].float()
            ).clamp(0, (1 << module.bits) - 1).to(torch.int32)[:, None]
        )
    intweight = torch.cat(intweight, dim=1).t().contiguous()

    return pack_awq(intweight, zeros.to(torch.int32), module.bits)


def test_awq_torch_pack_matches_scalar_fp32_reference():
    torch.manual_seed(0)

    for out_features, in_features, group_size in ((64, 128, 128), (96, 96, 32), (8, 64, 64)):
        linear = nn.Linear(in_features, out_features, bias=False)
        module = AwqTorchLinear(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        )

        weight = linear.weight.data
        groups = in_features // group_size
        weight_groups = weight.view(out_features, groups, group_size)
        weight_min = weight_groups.min(dim=2).values
        weight_max = weight_groups.max(dim=2).values
        scales = ((weight_max - weight_min) / 15.0).to(torch.float16)
        zeros = torch.round(-weight_min / scales.float()).to(torch.int32)

        expected_qweight, expected_qzeros = _reference_torch_awq_pack(module, linear, scales, zeros)

        module.pack(linear, scales.clone(), zeros.clone())

        assert torch.equal(module.qweight, expected_qweight)
        assert torch.equal(module.qzeros, expected_qzeros)
