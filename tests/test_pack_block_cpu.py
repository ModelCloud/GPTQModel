# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Targeted correctness coverage for the CPU pack_block extension.

This test exercises the exact gotchas that were fixed during the pack_block_cpu
PR: bfloat16 round-to-nearest-even tie cases, non-uniform desc_act g_idx maps,
per-lane scale*zero recomputation, AVX-512/AVX2->scalar fallbacks, and the
workers/pack_threads threading path.
"""

import math
import os
import unittest
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from parameterized import parameterized

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.cpp import load_pack_block_extension


pytestmark = [pytest.mark.cpu]


class TestPackBlockCpu(unittest.TestCase):
    in_features = 128
    out_features = 128

    @classmethod
    def setUpClass(cls):
        cls._orig_force_ext = os.environ.get("GPTQMODEL_FORCE_PACK_EXT")
        cls._orig_disable_ext = os.environ.get("GPTQMODEL_DISABLE_PACK_EXT")
        os.environ["GPTQMODEL_FORCE_PACK_EXT"] = "1"
        os.environ.pop("GPTQMODEL_DISABLE_PACK_EXT", None)

        # Prebuild/load the JIT extension once. If it cannot be built, all
        # extension-specific tests are skipped.
        cls._ext_available = bool(load_pack_block_extension())

    @classmethod
    def tearDownClass(cls):
        if cls._orig_force_ext is None:
            os.environ.pop("GPTQMODEL_FORCE_PACK_EXT", None)
        else:
            os.environ["GPTQMODEL_FORCE_PACK_EXT"] = cls._orig_force_ext
        if cls._orig_disable_ext is None:
            os.environ.pop("GPTQMODEL_DISABLE_PACK_EXT", None)
        else:
            os.environ["GPTQMODEL_DISABLE_PACK_EXT"] = cls._orig_disable_ext

    def setUp(self):
        if not self._ext_available:
            self.skipTest("pack_block_cpu extension is not available")

    def _build_inputs(self, bits: int, group_size: int, desc_act: bool, dtype: torch.dtype):
        """Generate a Linear and (scales_T, zeros_T, g_idx) for pack parity."""
        torch.manual_seed(0)
        in_features = self.in_features
        out_features = self.out_features
        max_q = (1 << bits) - 1

        if group_size <= 0:
            groups = 1
        else:
            groups = math.ceil(in_features / group_size)

        if group_size <= 0:
            g_idx = torch.zeros(in_features, dtype=torch.long)
        elif desc_act:
            g_idx = torch.randperm(in_features, dtype=torch.long) // group_size
            g_idx = g_idx.clamp(0, max(groups - 1, 0))
        else:
            g_idx = (torch.arange(in_features, dtype=torch.long) // group_size).clamp(0, max(groups - 1, 0))

        scales = torch.rand(groups, out_features, dtype=torch.float32) * 0.05 + 1e-3
        zeros = torch.randint(0, max_q + 1, (groups, out_features), dtype=torch.int32)
        q_int = torch.randint(0, max_q + 1, (in_features, out_features), dtype=torch.int32)

        weight = scales[g_idx].to(torch.float32) * (
            q_int.to(torch.float32) - zeros[g_idx].to(torch.float32)
        )

        linear = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
        linear.weight.data = weight.T.to(dtype)

        scales_T = scales.t().contiguous()
        zeros_T = zeros.t().contiguous()
        return linear, scales_T, zeros_T, g_idx

    def _pack_original(self, bits, group_size, desc_act, dtype, linear, scales_T, zeros_T, g_idx):
        qlinear = TorchLinear(
            bits=bits,
            group_size=group_size,
            sym=True,
            desc_act=desc_act,
            in_features=self.in_features,
            out_features=self.out_features,
            pack_dtype=torch.int32,
            backend=BACKEND.TORCH,
            bias=False,
        )
        qlinear.pack_original(linear, scales_T, zeros_T, g_idx=g_idx)
        return qlinear

    def _pack_block(self, bits, group_size, desc_act, dtype, linear, scales_T, zeros_T, g_idx, workers=None):
        qlinear = TorchLinear(
            bits=bits,
            group_size=group_size,
            sym=True,
            desc_act=desc_act,
            in_features=self.in_features,
            out_features=self.out_features,
            pack_dtype=torch.int32,
            backend=BACKEND.TORCH,
            bias=False,
        )
        kwargs = {}
        if workers is not None:
            kwargs["workers"] = workers
        qlinear.pack_block(
            linear,
            scales_T,
            zeros_T,
            g_idx.to(dtype=torch.int32),
            **kwargs,
        )
        return qlinear

    def _assert_same_pack(self, a, b):
        self.assertTrue(torch.equal(a.qweight, b.qweight), "qweight mismatch")
        self.assertTrue(torch.equal(a.qzeros, b.qzeros), "qzeros mismatch")
        self.assertTrue(torch.equal(a.scales, b.scales), "scales mismatch")
        self.assertTrue(
            torch.equal(a.g_idx.to(dtype=b.g_idx.dtype), b.g_idx),
            "g_idx mismatch",
        )

    @parameterized.expand(
        [
            (bits, group_size, desc_act, dtype)
            for bits in (2, 3, 4, 8)
            for group_size in (-1, 32, 64, 128)
            for desc_act in (False, True)
            for dtype in (torch.float32, torch.bfloat16)
        ]
    )
    def test_pack_block_parity(self, bits, group_size, desc_act, dtype):
        """pack_block (C++) must match pack_original for every bit/group/desc_act/dtype."""
        linear, scales_T, zeros_T, g_idx = self._build_inputs(bits, group_size, desc_act, dtype)
        ref = self._pack_original(bits, group_size, desc_act, dtype, linear, scales_T, zeros_T, g_idx)
        ext = self._pack_block(bits, group_size, desc_act, dtype, linear, scales_T, zeros_T, g_idx)
        self._assert_same_pack(ext, ref)

    def test_pack_block_non_uniform_desc_act(self):
        """Explicit non-uniform g_idx within a 32-input block (the regression case)."""
        bits = 4
        group_size = 32
        dtype = torch.bfloat16
        linear, scales_T, zeros_T, g_idx = self._build_inputs(bits, group_size, True, dtype)

        # Force every consecutive 32-input block to contain all groups.
        in_features = self.in_features
        groups = in_features // group_size
        g_idx = torch.randperm(in_features, dtype=torch.long)
        g_idx = (g_idx % groups).contiguous()
        for b in range(groups):
            block = torch.randperm(group_size, dtype=torch.long) % groups
            g_idx[b * group_size : (b + 1) * group_size] = block

        ref = self._pack_original(bits, group_size, True, dtype, linear, scales_T, zeros_T, g_idx)
        ext = self._pack_block(bits, group_size, True, dtype, linear, scales_T, zeros_T, g_idx)
        self._assert_same_pack(ext, ref)

    def test_pack_block_bfloat16_tie_rounding(self):
        """Construct weights whose quantized float is exactly half-integer.

        This is the bfloat16 8-bit case that exposed the AVX-512 FMA
        round-to-nearest-even tie bug. With scale=1.0 the scale*zero product
        is exact, so the only rounding is the final cvtps_epi32; any future
        re-fusion of (w + scale*zero) into a single FMA will flip tie cases.
        """
        bits = 8
        group_size = 32
        max_q = (1 << bits) - 1
        zero = max_q // 2

        # Build scales/zeros with scale=1.0 and a fixed zero point.
        groups = self.in_features // group_size
        scales = torch.ones(groups, self.out_features, dtype=torch.float32)
        zeros = torch.full((groups, self.out_features), zero, dtype=torch.int32)

        # For each input i, set W[:, i] = i + 0.5 - zero, producing qf = i + 0.5
        g_idx = torch.arange(self.in_features, dtype=torch.long) // group_size
        weight_vals = (
            torch.arange(self.in_features, dtype=torch.float32).unsqueeze(1)
            + 0.5
            - zero
        ).expand(-1, self.out_features)

        linear = nn.Linear(self.in_features, self.out_features, bias=False, dtype=torch.bfloat16)
        linear.weight.data = weight_vals.T.to(torch.bfloat16)

        scales_T = scales.t().contiguous()
        zeros_T = zeros.t().contiguous()

        ref = self._pack_original(bits, group_size, True, torch.bfloat16, linear, scales_T, zeros_T, g_idx)
        ext = self._pack_block(bits, group_size, True, torch.bfloat16, linear, scales_T, zeros_T, g_idx)
        self._assert_same_pack(ext, ref)

    @parameterized.expand(
        [
            (1,),
            (2,),
            (4,),
            (8,),
            (None,),
        ]
    )
    def test_pack_block_workers(self, workers):
        """pack_block workers argument must not change output."""
        bits = 4
        group_size = 32
        dtype = torch.bfloat16
        linear, scales_T, zeros_T, g_idx = self._build_inputs(bits, group_size, True, dtype)
        ref = self._pack_original(bits, group_size, True, dtype, linear, scales_T, zeros_T, g_idx)
        ext = self._pack_block(bits, group_size, True, dtype, linear, scales_T, zeros_T, g_idx, workers=workers)
        self._assert_same_pack(ext, ref)

    @parameterized.expand(
        [
            ("avx512_off", {"GPTQMODEL_PACK_CPU_DISABLE_AVX512": "1"}),
            ("avx2_off", {"GPTQMODEL_PACK_CPU_DISABLE_AVX512": "1", "GPTQMODEL_PACK_CPU_DISABLE_AVX2": "1"}),
        ]
    )
    def test_pack_block_isa_fallback(self, _label, env_override):
        """Scalar/AVX2 fallbacks must match the reference."""
        bits = 4
        group_size = 32
        dtype = torch.bfloat16
        linear, scales_T, zeros_T, g_idx = self._build_inputs(bits, group_size, True, dtype)
        ref = self._pack_original(bits, group_size, True, dtype, linear, scales_T, zeros_T, g_idx)

        # Include force-ext so the test exercises the C++ kernels even when the
        # env would otherwise disable the extension.
        env = {"GPTQMODEL_FORCE_PACK_EXT": "1"}
        env.update(env_override)
        with patch.dict(os.environ, env, clear=False):
            ext = self._pack_block(bits, group_size, True, dtype, linear, scales_T, zeros_T, g_idx)
        self._assert_same_pack(ext, ref)


if __name__ == "__main__":
    unittest.main()
