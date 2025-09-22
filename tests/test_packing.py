# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os

from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedQuantLinear

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

# isort: off
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
# isort: on
from gptqmodel import BACKEND  # noqa: E402
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.utils import dequantize_4bits_weight  # noqa: E402
from parameterized import parameterized  # noqa: E402


def gen_quant4(k, n, groupsize=-1):
    maxq = 2 ** 4 - 1
    w = torch.randn((k, n), dtype=torch.half, device="cpu")

    original_w = w.clone()

    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))

    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq

    # Quantize.
    w = torch.round(w / s).int()

    # Unsigned storage.
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)

    # Dequantize.
    ref = (w - (maxq + 1) // 2).half() * s

    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)

    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t()

    return original_w, linear, s


class TestRepacking(unittest.TestCase):
    QLINEAR_DICT = {
        # BACKEND.EXLLAMA_V1: ExllamaQuantLinear,
        BACKEND.TRITON: TritonV2QuantLinear,
        BACKEND.TORCH: TorchQuantLinear,
        BACKEND.TORCH_FUSED: TorchFusedQuantLinear,
        # BACKEND.BITBLAS: BitBLASQuantLinear,
    }


    k = 2048
    n = 1024 * 100
    group_size = 128
    pack_dtype = torch.int32

    zeros = torch.full((k // group_size, n), 8, dtype=torch.int32)
    print(f"k={k}, n={n}, shape={zeros.shape}, size={zeros.shape[0] * zeros.shape[1] * 4 / 1024 / 1024}M")

    _, linear, s = gen_quant4(k, n, group_size)

    def pack(self, qlinearCls, backend):
        qlinear = qlinearCls(
            bits=4,
            group_size=self.group_size,
            sym=True,
            desc_act=True,
            in_features=self.k,
            out_features=self.n,
            pack_dtype=self.pack_dtype,
            backend=backend,
            bias=False)

        qlinear.pack(self.linear, self.s.T, self.zeros.T, g_idx=None)

        return qlinear

    @parameterized.expand(
        list(QLINEAR_DICT.keys())
    )
    def test_compare_exllama_triton_torch(self, backend):
        triton_linear = self.pack(self.QLINEAR_DICT[backend], backend=backend)

        dequantized_weight, dequantized_qzeros = dequantize_4bits_weight(triton_linear)
        dequantized_weight = dequantized_weight.to(torch.float16)

        self.assertTrue(torch.equal(dequantized_weight, self.linear.weight))
        self.assertTrue(torch.all(dequantized_qzeros == 8))

        # validate torch packer
        torch_linear = self.pack(TorchQuantLinear, backend=BACKEND.TORCH)

        dequantized_weight, dequantized_qzeros = dequantize_4bits_weight(torch_linear)
        dequantized_weight = dequantized_weight.to(torch.float16)

        self.assertTrue(torch.equal(dequantized_weight, self.linear.weight))
        self.assertTrue(torch.all(dequantized_qzeros == 8))

        self.assertTrue(torch.allclose(triton_linear.qweight, torch_linear.qweight))
        self.assertTrue(torch.allclose(triton_linear.scales, torch_linear.scales))
        self.assertTrue(torch.allclose(triton_linear.qzeros, torch_linear.qzeros))
