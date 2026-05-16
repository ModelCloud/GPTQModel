# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import pytest  # noqa: E402
import torch  # noqa: E402

from gptqmodel.nn_modules.triton_utils.dequant import quant_matmul as dequant_quant_matmul  # noqa: E402
from gptqmodel.nn_modules.triton_utils.kernels import (  # noqa: E402
    quant_matmul_248,
    transpose_quant_matmul_248,
)


def _pack_int3_rows(values: torch.Tensor) -> torch.Tensor:
    in_features, out_features = values.shape
    values = values.to(torch.int64).reshape(in_features // 32, 32, out_features)
    word0 = torch.zeros((in_features // 32, out_features), dtype=torch.int64)
    word1 = torch.zeros_like(word0)
    word2 = torch.zeros_like(word0)

    for i in range(10):
        word0 |= (values[:, i] & 0x7) << (3 * i)
    word0 |= (values[:, 10] & 0x3) << 30
    word1 |= (values[:, 10] >> 2) & 0x1

    for i in range(11, 21):
        word1 |= (values[:, i] & 0x7) << (1 + 3 * (i - 11))
    word1 |= (values[:, 21] & 0x1) << 31
    word2 |= (values[:, 21] >> 1) & 0x3

    for i in range(22, 32):
        word2 |= (values[:, i] & 0x7) << (2 + 3 * (i - 22))

    return torch.stack((word0, word1, word2), dim=1).reshape((in_features // 32) * 3, out_features).to(torch.int32)


def _pack_int3_cols(values: torch.Tensor) -> torch.Tensor:
    groups, out_features = values.shape
    values = values.to(torch.int64).reshape(groups, out_features // 32, 32)
    word0 = torch.zeros((groups, out_features // 32), dtype=torch.int64)
    word1 = torch.zeros_like(word0)
    word2 = torch.zeros_like(word0)

    for i in range(10):
        word0 |= (values[:, :, i] & 0x7) << (3 * i)
    word0 |= (values[:, :, 10] & 0x3) << 30
    word1 |= (values[:, :, 10] >> 2) & 0x1

    for i in range(11, 21):
        word1 |= (values[:, :, i] & 0x7) << (1 + 3 * (i - 11))
    word1 |= (values[:, :, 21] & 0x1) << 31
    word2 |= (values[:, :, 21] >> 1) & 0x3

    for i in range(22, 32):
        word2 |= (values[:, :, i] & 0x7) << (2 + 3 * (i - 22))

    return torch.stack((word0, word1, word2), dim=2).reshape(groups, (out_features // 32) * 3).to(torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton int3 fused kernel test")
def test_triton_fused_int3_matches_dequant_reference():
    torch.manual_seed(11)
    device = torch.device("cuda:0")
    bits = 3
    maxq = 7
    pack_bits = 32
    group_size = 32
    batch = 3
    in_features = 64
    out_features = 64
    groups = in_features // group_size

    x = torch.randn(batch, in_features, device=device, dtype=torch.float16)
    qweight_values = torch.randint(0, 8, (in_features, out_features), dtype=torch.int64)
    qzero_values = torch.randint(0, 8, (groups, out_features), dtype=torch.int64)
    qweight = _pack_int3_rows(qweight_values).to(device)
    qzeros = _pack_int3_cols(qzero_values).to(device)
    scales = (torch.rand(groups, out_features, device=device, dtype=torch.float16) * 0.05 + 0.001).contiguous()
    g_idx = (torch.arange(in_features, device=device, dtype=torch.int32) // group_size).contiguous()

    expected = dequant_quant_matmul(x, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq)
    actual = quant_matmul_248(x, qweight, scales, qzeros, g_idx, bits, maxq)
    torch.testing.assert_close(actual, expected, rtol=0, atol=5e-3)

    grad = torch.randn(batch, out_features, device=device, dtype=torch.float16)
    expected_transpose = dequant_quant_matmul(grad, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=True)
    actual_transpose = transpose_quant_matmul_248(grad, qweight, scales, qzeros, g_idx, bits, maxq)
    torch.testing.assert_close(actual_transpose, expected_transpose, rtol=0, atol=5e-3)
