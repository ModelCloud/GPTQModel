# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import numpy as np  # noqa: F401  (used by the extracted reference below)
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, pack_intweight
from gptqmodel.utils.gemv import calculate_zeros_width


# The reference below is the exact pre-vectorization implementation, extracted
# verbatim from git history so the comparison is against the real old code.
def _reference_pack_intweight(unpacked_qweight, interleave, kstride):
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)
    Packed_Kernel = (
            Packed_Kernel[..., 0]
            | (Packed_Kernel[..., 1] << 4)
            | (Packed_Kernel[..., 2] << 8)
            | (Packed_Kernel[..., 3] << 12)
    )
    # reshape to (N // 4, K), FP16 format
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight = (
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight




def test_pack_intweight_matches_numpy_reference():
    torch.manual_seed(0)
    for num_rows, num_cols in ((64, 256), (128, 512), (256, 1024), (512, 2048)):
        codes = torch.randint(0, 16, (num_rows, num_cols), dtype=torch.int32)
        assert torch.equal(
            pack_intweight(codes, interleave=4, kstride=64),
            _reference_pack_intweight(codes, interleave=4, kstride=64),
        )


def _reference_gemv_fast_pack(module, linear, scales, zeros, qscales):
    """The pre-vectorization per-column intweight loop plus numpy pack."""

    scale_zeros = zeros * scales
    intweight = []
    for idx in range(module.in_features):
        intweight.append(
            torch.round(
                (linear.weight.data[:, idx] + scale_zeros[:, idx // module.group_size])
                / qscales[:, idx // module.group_size]
            ).to(torch.int)[:, None]
        )
    intweight = torch.cat(intweight, dim=1).to(dtype=torch.int32)
    qweight = _reference_pack_intweight(intweight.contiguous(), interleave=4, kstride=64)

    zeros = zeros.to(dtype=torch.int32)
    qzeros = torch.zeros_like(qscales)
    qzeros[:, : scales.shape[1]] = -(
        qscales[:, : scales.shape[1]] * zeros.to(torch.float32)
    ).to(torch.float16)
    return qweight, qzeros.transpose(1, 0).contiguous()


def test_gemv_fast_pack_is_bit_exact_with_column_loop():
    torch.manual_seed(0)

    for out_features, in_features, group_size in (
        (64, 256, 128),
        (128, 512, 64),
        (64, 1024, 32),
        (32, 128, 128),
    ):
        linear = nn.Linear(in_features, out_features, bias=False)
        module = AwqGEMVFastLinear(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=False,
        )

        groups = in_features // group_size
        weight = linear.weight.data
        weight_groups = weight.view(out_features, groups, group_size)
        weight_min = weight_groups.min(dim=2).values
        weight_max = weight_groups.max(dim=2).values
        scales = ((weight_max - weight_min) / 15.0).to(torch.float16)
        zeros = torch.round(-weight_min / scales.float()).to(torch.int32)

        qscales = torch.zeros(
            (out_features, calculate_zeros_width(in_features, group_size) * 8),
            dtype=torch.float16,
        )
        qscales[:, : scales.shape[1]] = scales

        expected_qweight, expected_qzeros = _reference_gemv_fast_pack(
            module, linear, scales.clone(), zeros.clone(), qscales
        )

        module.pack(linear, scales.clone(), zeros.clone())

        assert torch.equal(module.qweight, expected_qweight)
        assert torch.equal(getattr(module, module.zeros_name), expected_qzeros)
