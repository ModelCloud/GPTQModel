# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MXFP4 CPU kernel experiment: torch baseline vs Ninja-built C++ AVX-512 kernel.

This script benchmarks the on-the-fly MXFP4 (E2M1 + E8M0) dequant + matmul
kernel for BF16, FP16, and FP8-E4M3 activations.  The C++ extension supports
all three dtypes; the torch baseline only supports BF16/FP16 natively and
emulates FP8 by casting to FP32 for matmul.

Run with the repo venv that has PyTorch and ninja:
    /opt/.devin/venv/bin/python scripts/benchmark_mxfp4_cpu_kernel.py --dtype all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# MXFP4 E2M1 value table (index = 4-bit nibble).
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _derive_kimi_k3_attention_shapes() -> dict[str, dict[str, int]]:
    """Return the Kimi-3 attention projection shapes from config.json."""
    hidden = 7168
    heads = 96
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope = 128
    qk_rope = 64
    v_head = 128
    q_head = qk_nope + qk_rope  # 192

    return {
        "q_a_proj": {"k": hidden, "n": q_lora_rank},
        "q_b_proj": {"k": q_lora_rank, "n": heads * q_head},
        "kv_a_proj_with_mqa": {"k": hidden, "n": kv_lora_rank + qk_rope},
        "kv_b_proj": {"k": kv_lora_rank, "n": heads * (q_head - qk_rope + v_head)},
        "o_proj": {"k": heads * v_head, "n": hidden},
    }


def _quantization_thresholds() -> torch.Tensor:
    """Thresholds between FP4 positive magnitude bins."""
    return torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def quantize_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense (N, K) float32 weight to MXFP4.

    Returns:
        qweight: (N, K//2) uint8 with two E2M1 nibbles per byte.
        scales: (N, K//32) uint8 E8M0 scales.
    """
    weight = weight.to(torch.float32)
    N, K = weight.shape
    if K % 32 != 0:
        raise ValueError(f"K must be divisible by 32, got {K}")

    w_blocks = weight.view(N, K // 32, 32)
    max_abs = w_blocks.abs().amax(dim=-1, keepdim=True)

    needs = max_abs / 6.0
    needs = torch.where(needs <= 0, torch.tensor(1.0, dtype=torch.float32), needs)
    log2 = torch.log2(needs)
    exp = torch.ceil(log2).clamp(-127, 127)
    scale_f = torch.exp2(exp)
    scale_bits = (exp + 127).to(torch.int32).clamp(0, 254).to(torch.uint8)

    scaled = w_blocks / scale_f

    pos_values = FP4_TABLE[:8].to(weight.device)
    thresholds = _quantization_thresholds().to(weight.device)

    abs_scaled = scaled.abs()
    edges = torch.cat([torch.tensor([0.0], device=weight.device), thresholds])
    idx = torch.searchsorted(edges, abs_scaled, right=False)
    idx = idx.clamp(0, 7)
    quantized_abs = pos_values[idx]
    quantized = torch.where(scaled >= 0, quantized_abs, -quantized_abs)

    dist = (quantized.unsqueeze(-1) - FP4_TABLE.to(weight.device)).abs()
    nibbles = dist.argmin(dim=-1).to(torch.uint8)

    nibbles = nibbles.view(N, K // 32, 32)
    even = nibbles[..., 0::2]
    odd = nibbles[..., 1::2]
    bytes_ = (even & 0x0F) | ((odd & 0x0F) << 4)
    qweight = bytes_.view(N, K // 2)

    scales = scale_bits.squeeze(-1)
    return qweight, scales


def dequantize_mxfp4_to_fp8(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 weights to torch.float8_e4m3fn."""
    N, K_half = qweight.shape
    K = K_half * 2

    low = qweight & 0x0F
    high = (qweight >> 4) & 0x0F
    nibbles = torch.stack([low, high], dim=-1).view(N, K)

    scale_f = torch.exp2(scales.to(torch.float32) - 127.0)
    scale_f = scale_f.unsqueeze(-1).repeat(1, 1, 32).view(N, K)

    values = FP4_TABLE.to(qweight.device)[nibbles.to(torch.int64)] * scale_f
    return values.to(torch.float8_e4m3fn)


def make_activation(M: int, K: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Create a deterministic activation tensor in the requested dtype."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(M, K, generator=g, device="cpu", dtype=torch.float32)
    return x.to(dtype)


def torch_baseline(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Naive torch path: MXFP4 -> FP8 -> target dtype -> torch.matmul."""
    w_fp8 = dequantize_mxfp4_to_fp8(qweight, scales)
    dtype = x.dtype
    if dtype == torch.float8_e4m3fn:
        # PyTorch CPU has no native FP8 matmul, so cast both to FP32.
        out = torch.matmul(x.to(torch.float32), w_fp8.to(torch.float32).t())
        return out.to(torch.float8_e4m3fn)
    w = w_fp8.to(dtype)
    return torch.matmul(x, w.t())


def dense_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Dense FP32 reference (unrounded) used to evaluate quantization error."""
    return torch.matmul(x.to(torch.float32), weight.t())


def _ensure_ninja_on_path() -> None:
    """Make the ninja binary discoverable by torch.utils.cpp_extension."""
    if shutil.which("ninja"):
        return
    try:
        import ninja
        bin_dir = Path(ninja.BIN_DIR)
        ninja_bin = bin_dir / "ninja"
        if not ninja_bin.exists():
            return
    except (ImportError, AttributeError, OSError):
        return
    path = os.environ.get("PATH", "")
    bin_dir_str = str(bin_dir)
    if bin_dir_str not in path.split(os.pathsep):
        os.environ["PATH"] = f"{bin_dir_str}{os.pathsep}{path}"


_FP16_PROBE = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512fp16")))
__m512h probe(__m512h a, __m512h b, __m512h c) { return _mm512_fmadd_ph(a, b, c); }
int main() { return 0; }
"""


def _compiler_has_avx512fp16(compiler: str) -> bool:
    """Return True when `compiler` understands the avx512fp16 target attribute."""
    if shutil.which(compiler) is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cpp"
        src.write_text(_FP16_PROBE)
        proc = subprocess.run(
            [compiler, "-std=c++17", "-c", str(src), "-o", str(Path(tmp) / "probe.o")],
            capture_output=True,
            check=False,
        )
    return proc.returncode == 0


def _select_compiler() -> None:
    """Point torch's JIT at a compiler new enough for the AVX512-FP16 intrinsics."""
    current = os.environ.get("CXX", "c++")
    if _compiler_has_avx512fp16(current):
        return
    for candidate in ("g++-14", "g++-13", "g++-12", "clang++-16", "clang++-15", "clang++-14"):
        if _compiler_has_avx512fp16(candidate):
            os.environ["CXX"] = candidate
            print(f"  using {candidate}: default compiler lacks the avx512fp16 intrinsics")
            return
    print(f"  WARNING: no avx512fp16-capable compiler found; {current} builds the FP32/BF16 paths only")


def _mxfp4_extension():
    """Build/load the experimental C++ MXFP4 kernel via torch.utils.cpp_extension (Ninja)."""
    _ensure_ninja_on_path()
    _select_compiler()
    src = REPO_ROOT / "gptqmodel_ext" / "mxfp4_cpu_kernel.cpp"
    if not src.exists():
        raise FileNotFoundError(src)

    # No -mavx512* here on purpose: every AVX-512 routine carries its own
    # __attribute__((target(...))), so the baseline ISA stays generic and the
    # runtime-dispatched scalar fallbacks cannot be given instructions the host
    # may not support.
    extra_cflags = [
        "-O3",
        "-std=c++17",
        "-fopenmp",
    ]
    extra_cflags += os.environ.get("GPTQMODEL_MXFP4_EXTRA_CFLAGS", "").split()
    extra_ldflags = ["-fopenmp"]

    return load(
        name="mxfp4_cpu_kernel",
        sources=[str(src)],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        is_python_module=True,
        verbose=os.environ.get("GPTQMODEL_MXFP4_VERBOSE", "0") == "1",
    )


def _sync_cpu() -> None:
    if hasattr(torch, "cpu") and hasattr(torch.cpu, "synchronize"):
        torch.cpu.synchronize()


def _benchmark(fn, args, iters: int = 50, warmup: int = 10, name: str = "fn") -> dict:
    for _ in range(warmup):
        fn(*args)
        _sync_cpu()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    _sync_cpu()
    end = time.perf_counter()
    total = end - start
    return {"name": name, "iters": iters, "total_sec": total, "avg_ms": total / iters * 1000}


def _run_vnni(
    M: int,
    K: int,
    N: int,
    weight: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    ext,
    iters: int,
    warmup: int,
    seed: int,
    threads: int = 0,
) -> dict:
    """FP8 activations + AVX-512 VNNI int8 dot product on a pre-permuted MXFP4 layout."""
    dtype = torch.float8_e4m3fn
    x = make_activation(M, K, dtype, seed)
    ref = dense_reference(x, weight)
    ref_max = ref.abs().max().item() + 1e-8

    print("\n=== dtype=torch.float8_e4m3fn (VNNI int8) ===")
    prepack_start = time.perf_counter()
    qpack, spack = ext.mxfp4_prepack_vnni(qweight, scales)
    prepack_ms = (time.perf_counter() - prepack_start) * 1000
    packed_bytes = qpack.numel() + spack.numel()
    orig_bytes = qweight.numel() + scales.numel()
    print(f"  prepack {prepack_ms:.1f} ms, packed bytes {packed_bytes} vs mxfp4 {orig_bytes}")

    out_baseline = torch_baseline(x, qweight, scales)
    baseline_err = (out_baseline.to(torch.float32) - ref).abs().max().item()
    print(f"  torch baseline vs dense max abs err: {baseline_err:.4e} (rel {baseline_err / ref_max:.4e})")

    out_kernel = ext.mxfp4_linear_cpu_vnni(x, qpack, spack, N, threads)
    kernel_err = (out_kernel.to(torch.float32) - ref).abs().max().item()
    print(f"  VNNI kernel vs dense max abs err: {kernel_err:.4e} (rel {kernel_err / ref_max:.4e})")
    kernel_baseline_err = (out_kernel.to(torch.float32) - out_baseline.to(torch.float32)).abs().max().item()
    print(f"  VNNI kernel vs torch baseline max abs err: {kernel_baseline_err:.4e}")

    baseline_stats = _benchmark(torch_baseline, (x, qweight, scales), iters, warmup, "torch_baseline_fp8")
    kernel_stats = _benchmark(
        lambda x, qp, sp: ext.mxfp4_linear_cpu_vnni(x, qp, sp, N, threads),
        (x, qpack, spack),
        iters,
        warmup,
        "mxfp4_cpu_kernel_vnni",
    )

    return {
        "dtype": "torch.float8_e4m3fn (vnni)",
        "variant": 0,
        "threads": threads,
        "fp16_flush": 0,
        "M": M,
        "K": K,
        "N": N,
        "baseline_ms": baseline_stats["avg_ms"],
        "kernel_ms": kernel_stats["avg_ms"],
        "speedup": baseline_stats["avg_ms"] / kernel_stats["avg_ms"],
        "prepack_ms": prepack_ms,
        "baseline_err": baseline_err,
        "kernel_err": kernel_err,
        "kernel_baseline_err": kernel_baseline_err,
        "ref_max": ref_max - 1e-8,
    }


def _run_dtype(
    dtype: torch.dtype,
    M: int,
    K: int,
    N: int,
    weight: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    ext,
    iters: int,
    warmup: int,
    seed: int,
    variant: int = 0,
    threads: int = 0,
    fp16_flush: int = 0,
) -> dict:
    x = make_activation(M, K, dtype, seed)
    ref = dense_reference(x, weight)
    ref_max = ref.abs().max().item() + 1e-8

    print(
        f"\n=== dtype={dtype} variant={variant} threads={threads or 'auto'} "
        f"fp16_flush={fp16_flush or 'auto'} ==="
    )
    out_baseline = torch_baseline(x, qweight, scales)
    baseline_err = (out_baseline.to(torch.float32) - ref).abs().max().item()
    print(f"  torch baseline vs dense max abs err: {baseline_err:.4e} (rel {baseline_err / ref_max:.4e})")

    out_kernel = ext.mxfp4_linear_cpu(x, qweight, scales, threads, variant, fp16_flush)
    kernel_err = (out_kernel.to(torch.float32) - ref).abs().max().item()
    print(f"  C++ kernel vs dense max abs err: {kernel_err:.4e} (rel {kernel_err / ref_max:.4e})")
    kernel_baseline_err = (out_kernel.to(torch.float32) - out_baseline.to(torch.float32)).abs().max().item()
    print(f"  C++ kernel vs torch baseline max abs err: {kernel_baseline_err:.4e}")

    baseline_stats = _benchmark(torch_baseline, (x, qweight, scales), iters, warmup, f"torch_baseline_{dtype}")
    kernel_stats = _benchmark(
        lambda x, qw, s: ext.mxfp4_linear_cpu(x, qw, s, threads, variant, fp16_flush),
        (x, qweight, scales),
        iters,
        warmup,
        f"mxfp4_cpu_kernel_{dtype}",
    )

    speedup = baseline_stats["avg_ms"] / kernel_stats["avg_ms"]
    return {
        "dtype": str(dtype),
        "variant": variant,
        "threads": threads,
        "fp16_flush": fp16_flush,
        "M": M,
        "K": K,
        "N": N,
        "baseline_ms": baseline_stats["avg_ms"],
        "kernel_ms": kernel_stats["avg_ms"],
        "speedup": speedup,
        "baseline_err": baseline_err,
        "kernel_err": kernel_err,
        "kernel_baseline_err": kernel_baseline_err,
        "ref_max": ref_max - 1e-8,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", default="q_b_proj", help="Kimi-3 projection to simulate")
    parser.add_argument("--m", type=int, default=1, help="batch size")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp8", "vnni", "all"],
        help="Activation/output dtype for the kernel (bf16, fp16, fp8, or all)",
    )
    parser.add_argument(
        "--variants",
        default="0",
        help="Comma-separated kernel variants: 0=auto, 1=legacy FP32/VDPBF16PS, "
             "2=native FP16 FMA (FP16 accumulation), 3=native FP16 FMA (periodic FP32 fold)",
    )
    parser.add_argument("--threads", type=int, default=0, help="kernel threads (0 = kernel heuristic)")
    parser.add_argument(
        "--fp16-flush",
        type=int,
        default=0,
        help="fold FP16 accumulators into FP32 every N K-groups of 32 weights (0 = variant default)",
    )
    args = parser.parse_args()
    variants = [int(v) for v in args.variants.split(",") if v.strip()]

    shapes = _derive_kimi_k3_attention_shapes()
    if args.projection not in shapes:
        raise ValueError(f"unknown projection {args.projection}; choose from {list(shapes)}")

    K = shapes[args.projection]["k"]
    N = shapes[args.projection]["n"]
    M = args.m

    torch.manual_seed(args.seed)
    print(f"Benchmarking {args.projection}: M={M}, K={K}, N={N}")

    weight = torch.randn(N, K, dtype=torch.float32, device="cpu")
    print("Quantizing to MXFP4...")
    qweight, scales = quantize_mxfp4(weight)
    print(f"  qweight {tuple(qweight.shape)} dtype={qweight.dtype}")
    print(f"  scales  {tuple(scales.shape)} dtype={scales.dtype}")

    print("Building C++ MXFP4 kernel (Ninja)...")
    ext = _mxfp4_extension()

    want_vnni = args.dtype in ("vnni", "all")
    if args.dtype == "all":
        dtypes = [torch.bfloat16, torch.float16, torch.float8_e4m3fn]
    elif args.dtype == "vnni":
        dtypes = []
    elif args.dtype == "bf16":
        dtypes = [torch.bfloat16]
    elif args.dtype == "fp16":
        dtypes = [torch.float16]
    else:
        dtypes = [torch.float8_e4m3fn]

    results = []
    for dtype in dtypes:
        for variant in variants:
            results.append(
                _run_dtype(
                    dtype, M, K, N, weight, qweight, scales, ext, args.iters, args.warmup, args.seed,
                    variant, args.threads, args.fp16_flush,
                )
            )

    if want_vnni:
        results.append(
            _run_vnni(M, K, N, weight, qweight, scales, ext, args.iters, args.warmup, args.seed, args.threads)
        )

    print("\n" + "=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60)

    for r in results:
        status = "SUCCESS" if r["speedup"] >= 2.0 else "NEEDS WORK"
        print(
            f"{status}: dtype={r['dtype']} variant={r['variant']} flush={r['fp16_flush']} "
            f"speedup={r['speedup']:.2f}x "
            f"(kernel {r['kernel_ms']:.3f} ms, baseline {r['baseline_ms']:.3f} ms)"
        )


if __name__ == "__main__":
    main()
