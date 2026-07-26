# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.model import pack_module


class TestPackValidation(unittest.TestCase):
    """Verify that pack_module round-trip validates packed weights for each quant method."""

    in_features = 64
    out_features = 64

    def _assert_validation_succeeds(self, module, linear, q_scales, q_zeros, q_g_idx, qcfg):
        qModules = {"test": module}
        layers = {"test": linear}
        pack_module(
            "test",
            qModules,
            q_scales,
            q_zeros,
            q_g_idx,
            layers,
            type(module),
            None,
            None,
            qcfg,
        )
        # pack_module succeeding means the dequantization round-trip was valid.
        self.assertTrue(hasattr(module, "qweight") or hasattr(module, "weight"))

    @parameterized.expand([
        (4, 32),
        (4, -1),
    ])
    def test_awq_pack_module_validates(self, bits, group_size):
        """AWQ packed weights should round-trip through dequantize_weight without error."""
        in_f, out_f = self.in_features, self.out_features
        linear = nn.Linear(in_f, out_f, bias=False, dtype=torch.float16)
        groups = 1 if group_size <= 0 else math.ceil(in_f / group_size)
        scales = torch.rand(out_f, groups, dtype=torch.float16) * 0.05 + 1e-3
        zeros = torch.randint(0, 2**bits, (out_f, groups), dtype=torch.int32)

        qmodule = AwqTorchLinear(
            bits=bits,
            group_size=group_size,
            sym=False,
            desc_act=False,
            in_features=in_f,
            out_features=out_f,
            bias=False,
            register_buffers=False,
        )
        qcfg = QuantizeConfig(bits=bits, group_size=group_size, format="gemm", method="awq")
        self._assert_validation_succeeds(qmodule, linear, scales, zeros, None, qcfg)

    @parameterized.expand([
        (4, 32),
    ])
    def test_gptq_pack_module_validates(self, bits, group_size):
        """GPTQ packed weights should round-trip through dequantize_weight without error."""
        in_f, out_f = self.in_features, self.out_features
        linear = nn.Linear(in_f, out_f, bias=False, dtype=torch.float16)
        groups = 1 if group_size <= 0 else math.ceil(in_f / group_size)
        g_idx = torch.arange(in_f, dtype=torch.int32) // max(group_size, 1)
        max_q = 2**bits - 1
        scales = torch.rand(groups, out_f, dtype=torch.float16) * 0.05 + 1e-3
        zeros = torch.randint(0, max_q + 1, (groups, out_f), dtype=torch.int32)
        q_int = torch.randint(0, max_q + 1, (in_f, out_f), dtype=torch.int32)
        weight = scales[g_idx].to(torch.float32) * (q_int.to(torch.float32) - zeros[g_idx].to(torch.float32))
        linear.weight.data = weight.T.to(linear.weight.dtype)

        qmodule = TorchLinear(
            bits=bits,
            group_size=group_size,
            sym=True,
            desc_act=True,
            in_features=in_f,
            out_features=out_f,
            bias=False,
            pack_dtype=torch.int32,
            backend="torch",
        )
        qcfg = QuantizeConfig(bits=bits, group_size=group_size, format="gptq", method="gptq")
        self._assert_validation_succeeds(
            qmodule,
            linear,
            scales.t().contiguous(),
            zeros.t().contiguous(),
            g_idx,
            qcfg,
        )

    @parameterized.expand([
        (8, -1),
    ])
    def test_fp8_pack_module_validates(self, bits, group_size):
        """FP8 direct-pack should round-trip through dequantize_weight without error."""
        in_f, out_f = self.in_features, self.out_features
        linear = nn.Linear(in_f, out_f, bias=False, dtype=torch.float16)
        qmodule = TorchFP8Linear(
            bits=bits,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_f,
            out_features=out_f,
            bias=False,
            register_buffers=False,
        )
        qcfg = QuantizeConfig(bits=bits, group_size=group_size, format="fp8", method="fp8")
        self._assert_validation_succeeds(qmodule, linear, None, None, None, qcfg)


if __name__ == "__main__":
    unittest.main()
