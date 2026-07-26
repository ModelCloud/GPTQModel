# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from parameterized import parameterized

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


class TestPack3BitStrict(unittest.TestCase):
    """Strict, comprehensive accuracy tests for 3-bit CPU packing.

    3-bit packing is the non-power-of-two case and historically fell back to the
    Python path. These tests ensure the native CPU extension (when enabled) and the
    Python fallback produce bit-for-bit identical qweight/qzeros for a variety of
    shapes, group sizes, block-in sizes, worker counts, and negative g_idx values.
    """

    @staticmethod
    def _build_inputs(in_features: int, out_features: int, group_size: int, seed: int = 0):
        torch.manual_seed(seed)
        linear = nn.Linear(in_features, out_features, bias=False)

        if group_size == -1:
            groups = 1
            g_idx = torch.zeros(in_features, dtype=torch.int64)
        else:
            groups = math.ceil(in_features / group_size)
            g_idx = torch.arange(in_features, dtype=torch.int64) // group_size

        bits = 3
        max_q = 2 ** bits - 1
        scales = torch.rand(groups, out_features, dtype=torch.float32) * 0.05 + 1e-3
        zeros = torch.randint(0, max_q + 1, (groups, out_features), dtype=torch.int32)

        q_int = torch.randint(0, max_q + 1, (in_features, out_features), dtype=torch.int32)
        scales_expanded = scales[g_idx].to(torch.float32)
        zeros_expanded = zeros[g_idx].to(torch.float32)
        weight = scales_expanded * (q_int.to(torch.float32) - zeros_expanded)
        linear.weight.data = weight.T.to(linear.weight.dtype)

        return linear, scales, zeros, g_idx

    @staticmethod
    def _pack(
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor,
        group_size: int,
        in_features: int,
        out_features: int,
        block_in: int = 8192,
        workers: int = 1,
        disable_ext: bool = False,
    ):
        qlinear = TorchLinear(
            bits=3,
            group_size=group_size,
            sym=True,
            desc_act=True,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=torch.int32,
            backend=BACKEND.TORCH,
            bias=False,
        )

        env_value = "1" if disable_ext else "0"
        with patch.dict(os.environ, {"GPTQMODEL_DISABLE_PACK_EXT": env_value}, clear=False):
            qlinear.pack_block(
                linear,
                scales.T.contiguous(),
                zeros.T.contiguous(),
                g_idx=g_idx.to(torch.int32),
                block_in=block_in,
                workers=workers,
            )

        return {
            "qweight": qlinear.qweight.detach().cpu(),
            "qzeros": qlinear.qzeros.detach().cpu(),
            "scales": qlinear.scales.detach().cpu(),
            "g_idx": qlinear.g_idx.detach().cpu(),
        }

    @parameterized.expand(
        [
            # (in_features, out_features, group_size)
            (1024, 768, -1),
            (1024, 768, 32),
            (1024, 768, 64),
            (1024, 768, 128),
            (1024, 768, 256),
            (2048, 512, 128),
            (512, 2048, 64),
        ]
    )
    def test_3bit_extension_matches_python_fallback(self, in_features, out_features, group_size):
        linear, scales, zeros, g_idx = self._build_inputs(in_features, out_features, group_size)

        ext = self._pack(linear, scales, zeros, g_idx, group_size, in_features, out_features, disable_ext=False)
        py = self._pack(linear, scales, zeros, g_idx, group_size, in_features, out_features, disable_ext=True)

        self.assertTrue(torch.equal(ext["qweight"], py["qweight"]), "qweight mismatch")
        self.assertTrue(torch.equal(ext["qzeros"], py["qzeros"]), "qzeros mismatch")
        self.assertTrue(torch.equal(ext["scales"], py["scales"]), "scales mismatch")
        self.assertTrue(torch.equal(ext["g_idx"], py["g_idx"]), "g_idx mismatch")

    def test_3bit_negative_g_idx(self):
        in_features = 1024
        out_features = 768
        group_size = 32
        linear, scales, zeros, g_idx = self._build_inputs(in_features, out_features, group_size)

        g_idx_neg = g_idx.to(torch.int32)
        groups = int(g_idx.max().item() + 1)
        g_idx_neg[::7] -= groups

        ext = self._pack(linear, scales, zeros, g_idx_neg, group_size, in_features, out_features, disable_ext=False)
        py = self._pack(linear, scales, zeros, g_idx_neg, group_size, in_features, out_features, disable_ext=True)

        self.assertTrue(torch.equal(ext["qweight"], py["qweight"]), "qweight mismatch with negative g_idx")
        self.assertTrue(torch.equal(ext["qzeros"], py["qzeros"]), "qzeros mismatch with negative g_idx")
        self.assertTrue(torch.equal(ext["scales"], py["scales"]), "scales mismatch with negative g_idx")
        self.assertTrue(torch.equal(ext["g_idx"], py["g_idx"]), "g_idx mismatch with negative g_idx")

    @parameterized.expand(
        [
            # (block_in, workers)
            (32, 1),
            (32, 4),
            (64, 1),
            (64, 8),
            (128, 2),
            (256, 4),
            (1024, 8),
            (8192, 8),
        ]
    )
    def test_3bit_block_in_and_workers(self, block_in, workers):
        in_features = 1024
        out_features = 768
        group_size = 128
        linear, scales, zeros, g_idx = self._build_inputs(in_features, out_features, group_size)

        ext = self._pack(
            linear, scales, zeros, g_idx, group_size, in_features, out_features,
            block_in=block_in, workers=workers, disable_ext=False
        )
        py = self._pack(
            linear, scales, zeros, g_idx, group_size, in_features, out_features,
            block_in=block_in, workers=workers, disable_ext=True
        )

        self.assertTrue(torch.equal(ext["qweight"], py["qweight"]), "qweight mismatch")
        self.assertTrue(torch.equal(ext["qzeros"], py["qzeros"]), "qzeros mismatch")


if __name__ == "__main__":
    unittest.main()
