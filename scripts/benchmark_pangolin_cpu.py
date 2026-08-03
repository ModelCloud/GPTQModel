# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""CPU benchmark for the native Pangolin planar GEMV kernel.

Targets the real projection shapes from:
  - Laguna S 2.1
  - GLM-4.5-Air (public proxy for the gated GLM-4.5-Flash-0731 config)
  - DeepSeek-V4-Flash-0731

Reports latency for M=1,2,4,8 and bits=3,5,6,7, comparing the fused CPU kernel
against the dense dequant+matmul reference on the same CPU thread budget.
"""

import argparse
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import FORMAT
from gptqmodel.utils.logger import render_table
from gptqmodel.utils.pangolin import PANGOLIN_BITS, PANGOLIN_SUPPORTED_M, pangolin_gemv


# Real projection shapes as (K, N). Each must be divisible by 32 for planar pack.
SHAPE_SETS = {
    "laguna": [
        (2048, 6144),  # q_proj
        (2048, 1024),  # k_proj / v_proj
        (6144, 2048),  # o_proj
        (2048, 8192),  # dense gate / up
        (8192, 2048),  # dense down
        (2048, 512),  # moe/shared gate / up
        (512, 2048),  # moe/shared down
        (2048, 256),  # router gate
    ],
    "glm45": [
        (4096, 12288),  # q_proj
        (4096, 1024),  # k_proj / v_proj
        (12288, 4096),  # o_proj
        (4096, 10944),  # dense gate / up
        (10944, 4096),  # dense down
        (4096, 1408),  # moe/shared gate / up
        (1408, 4096),  # moe/shared down
        (4096, 128),  # router gate
    ],
    "deepseek_v4_flash_0731": [
        (4096, 1024),  # q_a_proj  (hidden -> q_lora_rank)
        (1024, 32768),  # q_b_proj  (q_lora_rank -> num_heads*head_dim)
        (4096, 512),  # kv_proj   (standard attention, single KV head)
        (4096, 8192),  # o_a_proj  (grouped, num_heads*head_dim -> o_groups*o_lora_rank)
        (8192, 4096),  # o_b_proj  (o_groups*o_lora_rank -> hidden)
        (4096, 2048),  # mlp/shared gate / up (moe_intermediate_size)
        (2048, 4096),  # mlp/shared down
        (4096, 256),  # router gate (n_routed_experts)
    ],
}


def _build_module(bits: int, k: int, n: int, group_size: int = 128) -> TorchLinear:
    torch.manual_seed(bits * 31 + k + n)
    linear = nn.Linear(k, n, bias=False)
    groups = k // group_size
    scales = torch.rand(n, groups) * 0.01 + 0.005
    zeros = torch.randint(0, (1 << bits), (n, groups)).float()
    g_idx = torch.tensor([min(i // group_size, groups - 1) for i in range(k)], dtype=torch.int32)

    kwargs = {
        "bits": bits,
        "group_size": group_size,
        "sym": False,
        "desc_act": False,
        "in_features": k,
        "out_features": n,
        "bias": False,
        "register_buffers": False,
    }
    if bits == 3:
        kwargs["format"] = FORMAT.GPTQ_P
    module = TorchLinear(**kwargs)
    module.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())
    return module


def _reference(module: TorchLinear, x: torch.Tensor) -> torch.Tensor:
    weight = module.dequantize_weight()[: module.in_features, : module.out_features]
    return torch.matmul(x, weight.to(x.dtype))


def _time(module: TorchLinear, x: torch.Tensor, bits: int, repeats: int) -> float:
    for _ in range(3):
        pangolin_gemv(x, module.qweight, module.scales, module.qzeros, module.g_idx, bits)
    start = time.perf_counter()
    for _ in range(repeats):
        pangolin_gemv(x, module.qweight, module.scales, module.qzeros, module.g_idx, bits)
    return time.perf_counter() - start


def _time_ref(module: TorchLinear, x: torch.Tensor, repeats: int) -> float:
    for _ in range(3):
        _reference(module, x)
    start = time.perf_counter()
    for _ in range(repeats):
        _reference(module, x)
    return time.perf_counter() - start


def _benchmark_shape(
    bits: int, k: int, n: int, m: int, threads: int, dtype: torch.dtype
) -> Tuple[float, float, float]:
    torch.set_num_threads(threads)
    module = _build_module(bits, k, n)
    module.scales = module.scales.to(dtype)

    x = torch.randn(m, k, dtype=dtype)
    # Adapt repeats so each timed region is roughly in the same time ballpark,
    # but cap at 15 so the full sweep stays reasonable.
    repeats = max(1, min(15, int(2.0 / (m * k * n * 1e-9))))

    kernel_time = _time(module, x, bits, repeats)
    ref_time = _time_ref(module, x, repeats)
    speedup = ref_time / kernel_time if kernel_time > 0 else float("inf")
    kernel_ms = kernel_time * 1000.0 / repeats
    ref_ms = ref_time * 1000.0 / repeats
    return kernel_ms, ref_ms, speedup


def _section(name: str, rows: List[Tuple], bits: int, threads: int) -> str:
    header = f"\n## {name} (bits={bits}, threads={threads})\n"
    table = render_table(
        rows,
        headers=["K x N", "M", "kernel (ms)", "ref (ms)", "speedup"],
        tablefmt="simple",
    )
    return header + "\n" + table + "\n"


def _sanity_speed_check(bits: int, threads: int, dtype: torch.dtype):
    """Quick fixed-shape sanity check: fused kernel must beat dequant+matmul."""
    in_features = 4096
    out_features = 11008
    group_size = 128
    torch.set_num_threads(threads)
    module = _build_module(bits, in_features, out_features, group_size)
    module.scales = module.scales.to(dtype)
    M = 8
    x = torch.randn(M, in_features, dtype=dtype)

    repeats = max(1, int(50 / (M * in_features * out_features * 1e-9)))
    repeats = min(repeats, 500)

    kernel_time = _time(module, x, bits, repeats)
    ref_time = _time_ref(module, x, repeats)
    speedup = ref_time / kernel_time if kernel_time > 0 else float("inf")

    rows = [
        ("reference (dequant+matmul)", f"{ref_time:.4f}", "1.00x"),
        ("pangolin_cpu kernel", f"{kernel_time:.4f}", f"{speedup:.2f}x"),
    ]
    print(
        render_table(
            rows,
            headers=["impl", f"time ({repeats} iters)", "speedup"],
            tablefmt="simple",
        )
    )

    if speedup <= 1.0:
        raise RuntimeError(f"Pangolin CPU kernel slower than reference (speedup={speedup:.2f}x)")


def _sanity_thread_scaling(bits: int, dtype: torch.dtype):
    """Verify wider thread counts do not regress latency."""
    in_features = 4096
    out_features = 4096
    group_size = 128
    module = _build_module(bits, in_features, out_features, group_size)
    module.scales = module.scales.to(dtype)
    x = torch.randn(1, in_features, dtype=dtype)
    repeats = 50

    rows = []
    baseline_time = None
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        t = _time(module, x, bits, repeats)
        if baseline_time is None:
            baseline_time = t
        speedup = baseline_time / t if t > 0 else float("inf")
        rows.append((threads, f"{t:.4f}", f"{speedup:.2f}x"))

    print(render_table(rows, headers=["threads", f"time ({repeats} iters)", "speedup"], tablefmt="simple"))

    multi_thread_times = [float(r[1]) for r in rows[1:]]
    if min(multi_thread_times) > baseline_time * 1.05:
        raise RuntimeError("Pangolin CPU kernel regressed with more threads")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="all", choices=["laguna", "glm45", "deepseek_v4_flash_0731", "all"])
    parser.add_argument("--bits", type=int, nargs="*", default=None)
    parser.add_argument("--batches", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--sanity", action="store_true", help="Run quick fixed-shape speed/thread checks")
    args = parser.parse_args()

    threads = args.threads or (os.cpu_count() or 1)
    dtype = getattr(torch, args.dtype)

    if args.sanity:
        bits_list = args.bits or list(PANGOLIN_BITS)
        for bits in bits_list:
            print(f"\n=== bits={bits} speed sanity ===")
            _sanity_speed_check(bits, threads, dtype)
            print(f"\n=== bits={bits} thread scaling sanity ===")
            _sanity_thread_scaling(bits, dtype)
        return

    bits_list = args.bits or list(PANGOLIN_BITS)
    batches = [m for m in args.batches if m in PANGOLIN_SUPPORTED_M]
    if not batches:
        batches = [1, 2, 4, 8]

    if args.shapes == "all":
        shape_sets = list(SHAPE_SETS.items())
    else:
        shape_sets = [(args.shapes, SHAPE_SETS[args.shapes])]

    md = [f"# Pangolin CPU GEMV benchmark ({args.dtype}, threads={threads})\n"]
    print(md[-1])

    for set_name, shapes in shape_sets:
        for bits in bits_list:
            rows = []
            for k, n in shapes:
                for m in batches:
                    kt_ms, rt_ms, sp = _benchmark_shape(bits, k, n, m, threads, dtype)
                    rows.append((f"{k} x {n}", str(m), f"{kt_ms:.3f}", f"{rt_ms:.3f}", f"{sp:.2f}x"))
                    print(
                        f"[{set_name}] bits={bits} {k}x{n} M={m} "
                        f"kernel={kt_ms:.3f}ms ref={rt_ms:.3f}ms speedup={sp:.2f}x"
                    )
            section = _section(set_name, rows, bits, threads)
            md.append(section)
            print(section)

    text = "".join(md)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
