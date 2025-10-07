# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os

from gptqmodel import BACKEND


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import time  # noqa: E402
import unittest  # noqa: E402

import threadpoolctl  # noqa: E402
from parameterized import parameterized  # noqa: E402
from tabulate import tabulate  # noqa: E402


# isort: off
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
# isort: on

from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear  # noqa: E402


def gen_quant4(k: int, n: int, groupsize: int):
    """
    Generate simple symmetric 4-bit grouped quantization tensors and a Linear layer
    whose weight equals the dequantized reference. Returns:
      original_w: [k, n] (float16)
      linear: nn.Linear with weight [n, k] = ref^T
      s: scales of shape [k//groupsize, n] (if groupsize != -1) or [1, n]
    """
    bits = 4
    maxq = (1 << bits) - 1
    w = torch.randn((k, n), dtype=torch.float16, device="cpu")
    original_w = w.clone()

    if groupsize != -1:
        # reshape for per-group scaling across k dimension
        w_g = w.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))
    else:
        w_g = w

    # Per-column max-abs scale (over current w_g layout)
    s = torch.max(torch.abs(w_g), 0, keepdim=True)[0]  # [1, ...]
    s *= 2.0 / maxq

    # Quantize to integers (store unsigned, compute signed)
    q = torch.round(w_g / s).to(torch.int32)
    q += (maxq + 1) // 2
    q = torch.clamp(q, 0, maxq)

    # Dequant reference
    ref = (q - (maxq + 1) // 2).to(torch.float16) * s

    if groupsize != -1:
        def _reshape_back(x):
            x = x.reshape((groupsize, -1, n)).permute(1, 0, 2).reshape((k, n)).contiguous()
            return x
        ref = _reshape_back(ref)

    # Scales layout expected by pack call (pre-transpose here to [k//g, n])
    s = s.reshape((-1, n)).contiguous()

    # Build Linear with ref^T as weight: PyTorch Linear wants [out_features, in_features] = [n, k]
    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t().contiguous()

    return original_w, linear, s


class TestPackingSpeed(unittest.TestCase):
    # Problem size
    group_size = 128
    k = 7168
    n = 7168

    # Zeros midpoint for 4-bit (unsigned storage)
    zeros = torch.full((k // group_size, n), 8, dtype=torch.int32, device="cpu").contiguous()

    # Real group index mapping: for channel i, group = i // group_size
    g_idx = (torch.arange(k, dtype=torch.long, device="cpu") // group_size).contiguous()

    # Precompute reference Linear and scales
    print(f"k={k}, n={n}, zeros.shape={zeros.shape}, size={zeros.numel() * zeros.element_size() / 1024 / 1024:.2f} MiB")
    print("gen_quant: start")
    _, linear, s = gen_quant4(k, n, group_size)
    linear = linear.to("cpu")
    s = s.to("cpu").contiguous()
    print("gen_quant: start...end")

    def pack(self, qlinearCls, backend, impl: str = "cpu"):
        """
        Instantiate a QuantLinear and pack with:
          - linear: nn.Linear [n,k] weight
          - scales_T: [n, k//group_size]
          - zeros_T:  [n, k//group_size]
          - g_idx:    [k] (torch.long)
        """
        qlinear = qlinearCls(
            bits=4,
            group_size=self.group_size,
            sym=True,
            desc_act=True,
            in_features=self.k,
            out_features=self.n,
            pack_dtype=torch.int32,
            backend=backend,
            bias=False,
        )

        scales_T = self.s.t().contiguous()   # [n, k//group_size]
        zeros_T = self.zeros.t().contiguous()  # [n, k//group_size]
        g_idx_i32 = self.g_idx.to(dtype=torch.int32)

        impl_lower = impl.lower()
        if impl_lower == "gpu":
            qlinear.pack_gpu(
                self.linear,
                scales_T,
                zeros_T,
                g_idx=g_idx_i32,
            )
        elif impl_lower == "original":
            qlinear.pack_original(
                self.linear,
                scales_T,
                zeros_T,
                g_idx=self.g_idx,
            )
        else:
            qlinear.pack_block(
                self.linear,
                scales_T,
                zeros_T,
                g_idx=g_idx_i32,
            )
        # Keep pack-only timing; omit post_init() to avoid extra work in speed test
        return qlinear

    def _time_pack_impl(self, qlinearCls, backend, impl: str, repeats: int, threads: int = 1) -> float:
        start = time.time()
        with threadpoolctl.threadpool_limits(limits=threads):
            for _ in range(repeats):
                self.pack(qlinearCls, backend, impl=impl)
        if impl.lower() == "gpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time() - start

    @parameterized.expand(
        [
            # [ExllamaQuantLinear, BACKEND.EXLLAMA, 9.63],
            # [TritonV2QuantLinear, BACKEND.TRITON, 9.67],
            [TorchQuantLinear, BACKEND.TORCH, 21.05],  # A100 Z3 33.56 # 4090? 27.0297
        ]
    )
    def test_pack_speed_single_thread(self, qlinearCls, backend, expect_time):
        start = time.time()
        with threadpoolctl.threadpool_limits(limits=1):
            for _ in range(30):
                self.pack(qlinearCls, backend)
        time_usage = time.time() - start
        speed = self.k * self.k / time_usage
        print(f"{qlinearCls.__name__} [1 thread], time={time_usage:.4f}s, speed={speed:.4f}")

        # Allow generous tolerance to absorb CI/GIL variance
        self.assertLess((time_usage - expect_time) / expect_time, 0.10, msg=f"time: {time_usage:.4f}s")

    @parameterized.expand(
        [
            # [ExllamaQuantLinear, BACKEND.EXLLAMA, 9.63],
            # [TritonV2QuantLinear, BACKEND.TRITON, 9.67],
            [TorchQuantLinear, BACKEND.TORCH, 14.71],  # A100 Z3 33.56 # 4090? 27.0297
        ]
    )
    def test_pack_speed_two_threads(self, qlinearCls, backend, expect_time):
        start = time.time()
        with threadpoolctl.threadpool_limits(limits=2):
            for _ in range(30):
                self.pack(qlinearCls, backend)
        time_usage = time.time() - start
        speed = self.k * self.k / time_usage
        print(f"{qlinearCls.__name__} [2 threads], time={time_usage:.4f}s, speed={speed:.4f}")

        # within 2.5%
        self.assertLess((time_usage - expect_time) / expect_time, 0.05, msg=f"time: {time_usage:.4f}s")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for GPU packing speed test")
    def test_pack_speed_gpu_vs_cpu(self):
        qlinearCls = TorchQuantLinear
        backend = BACKEND.TORCH
        repeats = 10

        cpu_time = self._time_pack_impl(qlinearCls, backend, impl="cpu", repeats=repeats, threads=1)
        gpu_time = self._time_pack_impl(qlinearCls, backend, impl="gpu", repeats=repeats, threads=1)

        print(
            f"{qlinearCls.__name__} [cpu vs gpu], cpu_time={cpu_time:.4f}s, "
            f"gpu_time={gpu_time:.4f}s, speedup={cpu_time / gpu_time if gpu_time else float('inf'):.2f}x"
        )

        self.assertLess(
            gpu_time,
            cpu_time * 0.90,
            msg=f"GPU pack slower than expected (cpu={cpu_time:.4f}s vs gpu={gpu_time:.4f}s)",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for GPU packing summary test")
    def test_pack_speed_summary(self):
        qlinearCls = TorchQuantLinear
        backend = BACKEND.TORCH
        repeats = 5

        timings = {}
        for label, impl in [
            ("pack_block", "cpu"),
            ("pack_gpu", "gpu"),
            ("pack_original", "original"),
        ]:
            timings[label] = self._time_pack_impl(qlinearCls, backend, impl=impl, repeats=repeats, threads=1)

        base_cpu = timings["pack_block"]
        rows = []
        for label in ["pack_original", "pack_block", "pack_gpu"]:
            time_usage = timings[label]
            speedup = base_cpu / time_usage if time_usage else float("inf")
            rows.append([label, f"{time_usage:.4f}", f"{speedup:.2f}x", repeats])

        print(tabulate(rows, headers=["impl", "time (s)", "speedup vs pack_block", "repeats"], tablefmt="simple"))

        self.assertLess(
            timings["pack_block"],
            timings["pack_original"] * 1.05,
            msg=("Optimized CPU pack substantially slower than original pack"
                 f" (cpu={timings['pack_block']:.4f}s vs original={timings['pack_original']:.4f}s)"),
        )
        self.assertLess(
            timings["pack_gpu"],
            timings["pack_block"] * 0.90,
            msg=("GPU pack slower than expected"
                 f" (gpu={timings['pack_gpu']:.4f}s vs cpu={timings['pack_block']:.4f}s)"),
        )
