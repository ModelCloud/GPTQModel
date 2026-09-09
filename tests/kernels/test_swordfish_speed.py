# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import statistics
import unittest

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.swordfish import SwordfishLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.swordfish import (
    prewarm_swordfish_extension,
    swordfish_runtime_available,
    swordfish_runtime_error,
)


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA not available"
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        return f"Swordfish requires Blackwell (sm100+); found sm{major}{minor}"
    return None


_SKIP_REASON = _skip_reason()


def _quantize_sym(weight: torch.Tensor, bits: int, group_size: int):
    out_features, in_features = weight.shape
    half_range = 1 << (bits - 1)
    num_groups = in_features // group_size

    q = torch.empty_like(weight)
    scales = torch.zeros((out_features, num_groups), dtype=weight.dtype, device=weight.device)
    for g in range(num_groups):
        mask = slice(g * group_size, (g + 1) * group_size)
        block = weight[:, mask]
        max_abs = block.abs().max(dim=1, keepdim=True).values
        max_abs[max_abs == 0] = 1.0
        scale = max_abs / (half_range - 1)
        q_block = torch.round(block / scale).clamp(-(half_range), half_range - 1) + half_range
        q[:, mask] = q_block
        scales[:, g : g + 1] = scale

    zeros = torch.full((out_features, num_groups), half_range, dtype=weight.dtype, device=weight.device)
    return q, scales, zeros


def _pack_torch_reference(bits, group_size, in_features, out_features, weight, scales, zeros, device):
    linear = nn.Linear(in_features, out_features, bias=False, dtype=weight.dtype, device="cpu")
    with torch.no_grad():
        linear.weight.copy_(weight)

    torch_linear = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        register_buffers=True,
    )
    g_idx = (torch.arange(in_features, dtype=torch.int32) // group_size)
    torch_linear.pack(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)
    torch_linear = torch_linear.to(device=device)
    torch_linear.post_init()
    return torch_linear


def _copy_to_swordfish(torch_linear: TorchLinear, dtype: torch.dtype) -> SwordfishLinear:
    sf = SwordfishLinear(
        bits=torch_linear.bits,
        group_size=torch_linear.requested_group_size,
        desc_act=torch_linear.desc_act,
        sym=torch_linear.sym,
        in_features=torch_linear.in_features,
        out_features=torch_linear.out_features,
        bias=False,
        dtype=dtype,
    )
    sf.qweight = nn.Parameter(torch_linear.qweight.data.detach().clone().contiguous(), requires_grad=False)
    sf.scales = nn.Parameter(torch_linear.scales.data.detach().clone().to(dtype).contiguous(), requires_grad=False)
    if torch_linear.g_idx is not None and torch_linear.g_idx.numel() > 0:
        sf.g_idx = nn.Parameter(torch_linear.g_idx.data.detach().clone().contiguous(), requires_grad=False)
    sf.post_init()
    return sf


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class TestSwordfishSpeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            prewarm_swordfish_extension()
        except Exception as exc:
            raise unittest.SkipTest(
                f"Swordfish runtime unavailable: {swordfish_runtime_error() or exc}"
            ) from exc
        if not swordfish_runtime_available():
            raise unittest.SkipTest(
                f"Swordfish runtime unavailable: {swordfish_runtime_error()}"
            )

    def _measure(self, fn, x, warmup=5, iters=20, blocks=5):
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        block_times = []
        for _ in range(blocks):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn(x)
            end.record()
            end.synchronize()
            block_times.append(start.elapsed_time(end) / iters)
        return statistics.median(block_times)

    def test_swordfish_speed_vs_machete(self):
        in_features = 4096
        out_features = 4096
        bits = 4
        group_size = 128
        dtype = torch.bfloat16
        device = torch.device("cuda:0")

        torch.manual_seed(42)
        weight = torch.randn((out_features, in_features), dtype=dtype, device="cpu") * 0.5
        q, scales, zeros = _quantize_sym(weight, bits, group_size)
        torch_linear = _pack_torch_reference(bits, group_size, in_features, out_features, q, scales, zeros, device)
        sf = _copy_to_swordfish(torch_linear, dtype)

        # Dense baseline in the same activation dtype and from the same checkpoint.
        with torch.no_grad():
            dense_weight = torch_linear.dequantize_weight().to(device=device, dtype=dtype)

        baseline = {}
        machete = {}
        swordfish = {}

        # Build the optional comparator once so allocator churn does not skew
        # the small-M timing samples.
        mach = None
        try:
            from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
            from gptqmodel.utils.machete import machete_runtime_available

            if machete_runtime_available():
                mach = MacheteLinear(
                    bits=bits,
                    group_size=group_size,
                    desc_act=False,
                    sym=True,
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                )
                mach.qweight = nn.Parameter(
                    torch_linear.qweight.data.detach().clone().contiguous(),
                    requires_grad=False,
                )
                mach.scales = nn.Parameter(
                    torch_linear.scales.data.detach().clone().to(dtype).contiguous(),
                    requires_grad=False,
                )
                mach.post_init()
                mach = mach.to(device)
        except Exception:
            mach = None

        m_values = [1, 8, 16, 32, 64, 128, 256]
        for m in m_values:
            x = torch.randn((m, in_features), dtype=dtype, device=device) * 0.5

            def dense_fn(x):
                return torch.matmul(x, dense_weight)

            def sf_fn(x):
                return sf(x)

            baseline_ms = self._measure(dense_fn, x)
            sf_ms = self._measure(sf_fn, x)

            flops = 2 * m * in_features * out_features
            baseline[m] = flops / (baseline_ms * 1e-3) / 1e12
            swordfish[m] = flops / (sf_ms * 1e-3) / 1e12

            if mach is not None:
                mach_ms = self._measure(mach, x)
                machete[m] = flops / (mach_ms * 1e-3) / 1e12

        print(f"\nSwordfish speed benchmark (in={in_features}, out={out_features}, bits={bits}, group={group_size})")
        print(f"{'M':>5} {'Swordfish TFLOPS':>18} {'Dense TFLOPS':>15} {'Machete TFLOPS':>18}")
        for m in m_values:
            print(f"{m:>5} {swordfish[m]:>18.3f} {baseline[m]:>15.3f} {machete.get(m, float('nan')):>18.3f}")

        ratios = {m: swordfish[m] / baseline[m] for m in m_values}
        print("Swordfish/dense ratios: " + ", ".join(f"M={m}: {ratio:.3f}" for m, ratio in ratios.items()))
        failures = [
            f"M={m} ratio={ratios[m]:.3f} "
            f"(Swordfish={swordfish[m]:.3f} TFLOPS, dense={baseline[m]:.3f} TFLOPS)"
            for m in (1, 8, 16)
            if ratios[m] < 0.8
        ]
        self.assertFalse(
            failures,
            "Swordfish must reach at least 0.8x dense throughput at decode sizes: "
            + "; ".join(failures),
        )
        median_ratio = statistics.median(ratios[m] for m in (1, 8, 16))
        self.assertGreaterEqual(
            median_ratio,
            1.0,
            "Swordfish median decode throughput ratio must be >= 1.0; "
            f"median={median_ratio:.3f}, ratios={ratios}",
        )

        if os.environ.get("GPTQMODEL_SWORDFISH_WRITE_SPEED_JSON"):
            import json
            out = {
                "in_features": in_features,
                "out_features": out_features,
                "bits": bits,
                "group_size": group_size,
                "swordfish": swordfish,
                "dense": baseline,
                "machete": machete,
                "swordfish_over_dense": ratios,
                "decode_median_ratio": median_ratio,
            }
            with open("/tmp/swordfish_speed.json", "w") as f:
                json.dump(out, f, indent=2)
