# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from parameterized import parameterized

from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.pack_block_ext import pack_awq_cpu
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear


class TestAwqCpuPacking(unittest.TestCase):
    @parameterized.expand([
        (4, 64, 64, 32),
        (4, 128, 256, -1),
        (4, 96, 128, 32),
        (4, 64, 128, 128),
    ])
    def test_pack_awq_cpu_bitwise(self, bits, in_features, out_features, group_size):
        pack_num = 32 // bits
        order = [0, 2, 4, 6, 1, 3, 5, 7]

        intweight = torch.randint(0, 2**bits, (in_features, out_features), dtype=torch.int32)
        groups = 1 if group_size <= 0 else math.ceil(in_features / group_size)
        zeros = torch.randint(0, 2**bits, (groups, out_features), dtype=torch.int32)

        qweight, qzeros = pack_awq_cpu(intweight, zeros, bits)

        self.assertEqual(qweight.shape, (in_features, out_features // pack_num))
        self.assertEqual(qzeros.shape, (groups, out_features // pack_num))

        qweight_ref = torch.zeros_like(qweight)
        for col in range(out_features // pack_num):
            for i in range(pack_num):
                qweight_ref[:, col] |= intweight[:, col * pack_num + order[i]] << (i * bits)

        qzeros_ref = torch.zeros_like(qzeros)
        for col in range(out_features // pack_num):
            for i in range(pack_num):
                qzeros_ref[:, col] |= zeros[:, col * pack_num + order[i]] << (i * bits)

        self.assertTrue(torch.equal(qweight, qweight_ref))
        self.assertTrue(torch.equal(qzeros, qzeros_ref))

    def _build_linear(self, in_features, out_features, dtype):
        return nn.Linear(in_features, out_features, bias=False, dtype=dtype)

    def _build_scales_zeros(self, out_features, group_size, in_features, dtype, bits):
        groups = 1 if group_size <= 0 else math.ceil(in_features / group_size)
        scales = torch.rand(out_features, groups, dtype=dtype) * 0.05 + 1e-3
        zeros = torch.randint(0, 2**bits, (out_features, groups), dtype=dtype)
        return scales, zeros

    @parameterized.expand([
        (torch.float16, 64, 64, 32),
        (torch.bfloat16, 64, 64, 32),
        (torch.float16, 128, 256, -1),
        (torch.bfloat16, 96, 128, 32),
    ])
    def test_awq_torch_linear_pack(self, dtype, in_features, out_features, group_size):
        linear = self._build_linear(in_features, out_features, dtype)
        scales, zeros = self._build_scales_zeros(out_features, group_size, in_features, dtype, bits=4)

        q_cpu = AwqTorchLinear(
            bits=4,
            group_size=group_size,
            sym=False,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=False,
        )
        q_cpu.pack(linear, scales, zeros)

        with patch("gptqmodel.nn_modules.qlinear.torch_awq.pack_awq_cpu", side_effect=RuntimeError("unavailable")):
            q_fallback = AwqTorchLinear(
                bits=4,
                group_size=group_size,
                sym=False,
                desc_act=False,
                in_features=in_features,
                out_features=out_features,
                bias=False,
                register_buffers=False,
            )
            q_fallback.pack(linear, scales, zeros)

        self.assertTrue(torch.equal(q_cpu.qweight, q_fallback.qweight))
        self.assertTrue(torch.equal(q_cpu.qzeros, q_fallback.qzeros))
        self.assertTrue(torch.equal(q_cpu.scales, q_fallback.scales))

    @parameterized.expand([
        (torch.float16, 64, 64, 32),
        (torch.bfloat16, 64, 64, 32),
        (torch.float16, 128, 256, -1),
    ])
    def test_awq_gemm_linear_pack(self, dtype, in_features, out_features, group_size):
        linear = self._build_linear(in_features, out_features, dtype)
        scales, zeros = self._build_scales_zeros(out_features, group_size, in_features, dtype, bits=4)

        q_cpu = AwqGEMMLinear(
            bits=4,
            group_size=group_size,
            sym=False,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=False,
        )
        q_cpu.pack(linear, scales, zeros)

        with patch("gptqmodel.nn_modules.qlinear.gemm_awq.pack_awq_cpu", side_effect=RuntimeError("unavailable")):
            q_fallback = AwqGEMMLinear(
                bits=4,
                group_size=group_size,
                sym=False,
                desc_act=False,
                in_features=in_features,
                out_features=out_features,
                bias=False,
                register_buffers=False,
            )
            q_fallback.pack(linear, scales, zeros)

        self.assertTrue(torch.equal(q_cpu.qweight, q_fallback.qweight))
        self.assertTrue(torch.equal(q_cpu.qzeros, q_fallback.qzeros))
        self.assertTrue(torch.equal(q_cpu.scales, q_fallback.scales))


if __name__ == "__main__":
    unittest.main()
