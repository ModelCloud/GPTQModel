"""Benchmark native Pangolin GEMV for M=1,2,4,8,16,32 against an FP32 dense reference.

Usage:
    python scripts/benchmark_pangolin_m.py [bits] [K] [N] [dtype] [--symmetry affine|symmetric]
"""
import argparse
import gc
import os
import statistics
import subprocess
import sys


def preflight_idle_check(util_threshold: int = 5) -> None:
    """Stdlib-only idle check on the visible device before importing torch."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        print("preflight: nvidia-smi unavailable; skipping idle check")
        return
    for line in out.strip().splitlines():
        uuid, util, mem = (part.strip() for part in line.split(","))
        if visible and uuid not in visible:
            continue
        if int(util) > util_threshold:
            print(f"preflight: device {uuid} busy (util={util}%); aborting")
            sys.exit(2)
        print(f"preflight: device {uuid} idle (util={util}%, mem={mem} MiB)")


def bench(fn, warmup=10, iters=50):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    ordered = sorted(times)
    return statistics.median(ordered), ordered[len(ordered) // 10], ordered[(len(ordered) * 9) // 10]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bits", nargs="?", type=int, default=3)
    parser.add_argument("k", nargs="?", type=int, default=4096)
    parser.add_argument("n", nargs="?", type=int, default=4096)
    parser.add_argument("dtype", nargs="?", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--symmetry", choices=("affine", "symmetric"), default="affine")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches", nargs="+", type=int, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument(
        "--extension",
        help="load this Pangolin .so directly for cached-kernel A/B comparisons",
    )
    return parser.parse_args()


def make_zeros(torch, *, bits: int, groups: int, n: int, symmetry: str):
    maxq = (1 << bits) - 1
    midpoint = (maxq + 1) // 2
    if symmetry == "symmetric":
        return torch.full((groups, n), midpoint, dtype=torch.int32)

    # Exercise the entire affine contract deterministically. Every output tile
    # sees zero, midpoint, maximum, and group/column-varying zero points.
    group = torch.arange(groups, dtype=torch.int32).unsqueeze(1)
    column = torch.arange(n, dtype=torch.int32).unsqueeze(0)
    zeros = (group * 17 + column * 29 + 3) % (maxq + 1)
    zeros[:, 0::32] = 0
    zeros[:, 1::32] = maxq
    zeros[:, 2::32] = midpoint
    return zeros


def _theoretical_peak_tflops(dtype_name: str, prop) -> float:
    """Non-tensor FP16/BF16 peak TFLOPS for the current device.

    The kernel accumulates in FP32 for numerical stability, but the output
    dtype caps the meaningful peak at the FP16/BF16 CUDA-core rate.
    """
    clock_ghz = prop.clock_rate / 1_000_000.0  # clock_rate is in kHz
    sm_count = prop.multi_processor_count
    # Ampere sm_80: FP32 = 64 FMA/SM/clk, FP16 = 4x FP32, BF16 = 2x FP32.
    if prop.major >= 8:
        fma_per_clock = 256 if dtype_name == "float16" else 128
    else:
        fma_per_clock = 64
    return (sm_count * clock_ghz * fma_per_clock * 2.0) / 1000.0


def _env_banner(dtype_name: str):
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    dev_idx = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev_idx)
    name = torch.cuda.get_device_name(dev_idx)
    cc = f"{prop.major}.{prop.minor}"
    mem_gib = prop.total_memory / 2**30
    peak_tflops = _theoretical_peak_tflops(dtype_name, prop)
    print("=" * 70)
    print(f"device       : {name} (cc {cc}, {prop.multi_processor_count} SMs, {mem_gib:.1f} GiB)")
    print(f"peak (dtype) : {peak_tflops:.2f} TFLOPS ({dtype_name})")
    print(f"visible GPU  : {visible}")
    print(f"torch        : {torch.__version__}")
    print(f"torch.cuda   : {torch.version.cuda}")
    print(f"triton       : {getattr(torch.version, 'triton', 'n/a')}")
    print("=" * 70)


def main():
    import torch
    from gptqmodel.utils.pangolin import PANGOLIN_MAX_M, pangolin_gemv
    from gptqmodel.utils.planar_packing import (
        planar_pack_cols,
        planar_pack_rows,
        planar_unpack_cols,
        planar_unpack_rows,
    )

    args = parse_args()
    gemv = pangolin_gemv
    if args.extension:
        torch.ops.load_library(args.extension)
        gemv = torch.ops.gptqmodel_pangolin.gemv
    bits = args.bits
    k = args.k
    n = args.n
    dtype_name = args.dtype
    group_size = 128
    groups = k // group_size
    dtype = getattr(torch, dtype_name)

    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    codes = torch.randint(0, 2**bits, (k, n), dtype=torch.int32)
    zeros = make_zeros(torch, bits=bits, groups=groups, n=n, symmetry=args.symmetry)
    qweight = planar_pack_rows(codes, bits).to(dev)
    qzeros = planar_pack_cols(zeros, bits).to(dev)
    scales = (torch.rand(groups, n, dtype=dtype) * 0.01 + 0.005).to(dev)
    g_idx = (torch.arange(k, dtype=torch.int32) // group_size).to(dev)

    _env_banner(dtype_name)
    print(
        f"shape        : K={k} N={n} bits={bits} dtype={dtype_name} symmetry={args.symmetry} "
        f"PANGOLIN_MAX_M={PANGOLIN_MAX_M} warmup={args.warmup} iters={args.iters} seed={args.seed}"
    )
    print(f"extension    : {args.extension or 'current JIT source'}")
    header = (
        f"{'M':>4} | {'median_ms':>9} | {'p10_ms':>8} | {'p90_ms':>8} | {'gflops':>9} | "
        f"{'dense_ms':>8} | {'speedup':>7} | {'max_abs':>9} | {'mae':>9} | {'rmse':>9} | "
        f"{'rel_l2':>9} | {'fwd_kld':>9} | {'top1':>6} | {'nf':>3} | {'temp_MiB':>9}"
    )
    print(header)
    print("-" * len(header))
    for m in args.batches:
        if m not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f"unsupported M={m}; choose from 1,2,4,8,16,32")
        x = (torch.randn(m, k, dtype=dtype) * 0.5).to(dev)
        try:
            # warmup
            for _ in range(5):
                _ = gemv(x, qweight, scales, qzeros, g_idx, bits)
            torch.cuda.synchronize()
            ms, p10_ms, p90_ms = bench(
                lambda: gemv(x, qweight, scales, qzeros, g_idx, bits),
                warmup=args.warmup,
                iters=args.iters,
            )

            torch.cuda.synchronize()
            gc.collect()
            base_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = gemv(x, qweight, scales, qzeros, g_idx, bits)
            torch.cuda.synchronize()
            transient_mib = (torch.cuda.max_memory_allocated() - base_allocated) / 2**20

            # dense reference: recompute dequant + matmul each iteration to match
            # the on-the-fly dense baseline used in scripts/benchmark_planar_kernels.py.
            with torch.no_grad():
                unpacked = planar_unpack_rows(qweight, bits)
                zeros_dense = planar_unpack_cols(qzeros, bits)
                wq = (unpacked.float() - zeros_dense[g_idx].float()) * scales[g_idx].float()
                ref = torch.matmul(x.float(), wq)

                def dense_fn():
                    w = ((unpacked.float() - zeros_dense[g_idx].float()) * scales[g_idx].float()).to(dtype)
                    return torch.matmul(x, w)

                dense_ms, _, _ = bench(dense_fn, warmup=args.warmup, iters=args.iters)
            diff = (out.float() - ref).abs()
            mae = diff.mean().item()
            rmse = diff.square().mean().sqrt().item()
            rel_l2 = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref)).item()
            # FP64 avoids the small negative values that FP32 roundoff can
            # produce for a metric that is nonnegative by definition.
            ref_log_probs = torch.log_softmax(ref.double(), dim=-1)
            out_log_probs = torch.log_softmax(out.double(), dim=-1)
            forward_kld = (
                ref_log_probs.exp() * (ref_log_probs - out_log_probs)
            ).sum(dim=-1).mean().item()
            top1 = (ref.argmax(dim=-1) == out.argmax(dim=-1)).float().mean().item()
            nonfinite = (~torch.isfinite(out)).sum().item()
            gflops = (2.0 * m * k * n) / (ms * 1e6)
            speedup = dense_ms / ms
            print(
                f"{m:4d} | {ms:9.4f} | {p10_ms:8.4f} | {p90_ms:8.4f} | {gflops:9.1f} | "
                f"{dense_ms:8.4f} | {speedup:7.2f} | {diff.max().item():9.2e} | {mae:9.2e} | "
                f"{rmse:9.2e} | {rel_l2:9.2e} | {forward_kld:9.2e} | {top1:6.3f} | "
                f"{nonfinite:3d} | {transient_mib:9.2f}"
            )
        except Exception as exc:
            print(f"{m:>4} | FAILED: {exc}")


if __name__ == "__main__":
    preflight_idle_check()
    main()
