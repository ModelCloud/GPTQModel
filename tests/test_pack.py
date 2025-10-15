# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math
import time
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized
from tabulate import tabulate

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear


class TestPackAccuracy(unittest.TestCase):
    in_features = 1024
    out_features = 768

    @staticmethod
    def _build_inputs(bits: int, group_size: int):
        torch.manual_seed(0)

        linear = nn.Linear(TestPackAccuracy.in_features, TestPackAccuracy.out_features, bias=False)

        if group_size == -1:
            groups = 1
            g_idx = torch.zeros(TestPackAccuracy.in_features, dtype=torch.int64)
        else:
            groups = math.ceil(TestPackAccuracy.in_features / group_size)
            g_idx = torch.arange(TestPackAccuracy.in_features, dtype=torch.int64) // group_size

        max_q = 2 ** bits - 1
        scales = torch.rand(groups, TestPackAccuracy.out_features, dtype=torch.float32) * 0.05 + 1e-3
        zeros = torch.randint(0, max_q + 1, (groups, TestPackAccuracy.out_features), dtype=torch.int32)

        q_int = torch.randint(0, max_q + 1, (TestPackAccuracy.in_features, TestPackAccuracy.out_features), dtype=torch.int32)
        scales_expanded = scales[g_idx].to(torch.float32)
        zeros_expanded = zeros[g_idx].to(torch.float32)
        weight = scales_expanded * (q_int.to(torch.float32) - zeros_expanded)
        linear.weight.data = weight.T.to(linear.weight.dtype)

        return linear, scales, zeros, g_idx

    def _quant_linear(self):
        qlinear = TorchQuantLinear(
            bits=self.current_bits,
            group_size=self.current_group_size,
            sym=True,
            desc_act=True,
            in_features=self.in_features,
            out_features=self.out_features,
            pack_dtype=torch.int32,
            backend=BACKEND.TORCH,
            bias=False,
        )
        return qlinear

    def _run_impl(self, impl: str, linear, scales, zeros, g_idx):
        qlinear = self._quant_linear()
        scales_T = scales.t().contiguous()
        zeros_T = zeros.t().contiguous()

        start = time.perf_counter()

        if impl == "original":
            qlinear.pack_original(linear, scales_T, zeros_T, g_idx=g_idx)
        elif impl == "pack_block":
            qlinear.pack_block(
                linear,
                scales_T,
                zeros_T,
                g_idx=g_idx.to(dtype=torch.int32),
            )
        elif impl == "gpu":
            if not torch.cuda.is_available():
                self.skipTest("CUDA device required for GPU pack comparison")
            qlinear.pack_gpu(
                linear,
                scales_T,
                zeros_T,
                g_idx=g_idx.to(dtype=torch.int32),
            )
            torch.cuda.synchronize()
        else:
            raise ValueError(f"Unknown impl `{impl}`")

        end = time.perf_counter()
        duration = end - start

        # Move buffers to CPU for comparisons
        result = {
            "qweight": qlinear.qweight.detach().cpu(),
            "qzeros": qlinear.qzeros.detach().cpu(),
            "scales": qlinear.scales.detach().cpu(),
            "g_idx": qlinear.g_idx.detach().cpu(),
        }
        if hasattr(qlinear, "bias") and qlinear.bias is not None:
            result["bias"] = qlinear.bias.detach().cpu()
        return result, duration

    @parameterized.expand(
        [
            (2, -1), (2, 32), (2, 64), (2, 128),
            (3, -1), (3, 32), (3, 64), (3, 128),
            (4, -1), (4, 32), (4, 64), (4, 128),
            (8, -1), (8, 32), (8, 64), (8, 128),
        ]
    )
    def test_pack_consistency(self, bits, group_size):
        self.current_bits = bits
        self.current_group_size = group_size

        linear, scales, zeros, g_idx = self._build_inputs(bits, group_size)

        baseline, baseline_time = self._run_impl("original", linear, scales, zeros, g_idx)
        pack_cpu, pack_cpu_time = self._run_impl("pack_block", linear, scales, zeros, g_idx)
        results = {"pack_block": (pack_cpu, pack_cpu_time)}

        if torch.cuda.is_available():
            pack_gpu, pack_gpu_time = self._run_impl("gpu", linear, scales, zeros, g_idx)
            results["pack_gpu"] = (pack_gpu, pack_gpu_time)

        rows = []
        rows.append([f"pack_original (bits={bits}, g={group_size})", 0.0, 0.0, 0.0, 0.0, baseline_time * 1e3])
        for name, (tensors, duration) in results.items():
            diff_qweight = (tensors["qweight"].to(dtype=baseline["qweight"].dtype) - baseline["qweight"]).abs().max().item()
            diff_qzeros = (tensors["qzeros"].to(dtype=baseline["qzeros"].dtype) - baseline["qzeros"]).abs().max().item()
            diff_scales = (tensors["scales"].to(dtype=baseline["scales"].dtype) - baseline["scales"]).abs().max().item()
            diff_gidx = (tensors["g_idx"].to(dtype=baseline["g_idx"].dtype) - baseline["g_idx"]).abs().max().item()
            rows.append([
                f"{name} (bits={bits}, g={group_size})",
                diff_qweight,
                diff_qzeros,
                diff_scales,
                diff_gidx,
                duration * 1e3,
            ])

            self.assertTrue(torch.equal(tensors["qweight"], baseline["qweight"]))
            self.assertTrue(torch.equal(tensors["qzeros"], baseline["qzeros"]))
            self.assertTrue(torch.equal(tensors["g_idx"].to(dtype=baseline["g_idx"].dtype), baseline["g_idx"]))
            self.assertTrue(torch.equal(tensors["scales"], baseline["scales"]))

        print(
            tabulate(
                rows,
                headers=["impl", "max|Δ qweight|", "max|Δ qzeros|", "max|Δ scales|", "max|Δ g_idx|", "time [ms]"],
                floatfmt=".3e",
            )
        )

    def test_pack_negative_g_idx(self):
        bits = 4
        group_size = 32
        self.current_bits = bits
        self.current_group_size = group_size
        linear, scales, zeros, g_idx = self._build_inputs(bits, group_size)

        groups = int(g_idx.max().item() + 1)
        g_idx_neg = g_idx.to(dtype=torch.int32)
        g_idx_neg[::7] -= groups

        baseline, _ = self._run_impl("original", linear, scales, zeros, g_idx_neg)
        pack_cpu, _ = self._run_impl("pack_block", linear, scales, zeros, g_idx_neg)

        self.assertTrue(torch.equal(pack_cpu["qweight"], baseline["qweight"]))
        self.assertTrue(torch.equal(pack_cpu["qzeros"], baseline["qzeros"]))
        self.assertTrue(torch.equal(pack_cpu["scales"], baseline["scales"]))
        self.assertTrue(torch.equal(pack_cpu["g_idx"].to(dtype=baseline["g_idx"].dtype), baseline["g_idx"]))
