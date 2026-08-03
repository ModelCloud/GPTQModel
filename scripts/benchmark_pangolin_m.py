"""Benchmark native Pangolin GEMV for M=1,2,4,8,16,32 and compare to dense reference.

Usage:
    python scripts/benchmark_pangolin_m.py [bits] [K] [N] [dtype]
"""
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
    return statistics.median(times)


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

    bits = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    dtype_name = sys.argv[4] if len(sys.argv) > 4 else "float16"
    group_size = 128
    groups = k // group_size
    dtype = getattr(torch, dtype_name)

    torch.manual_seed(0)
    dev = torch.device("cuda")
    codes = torch.randint(0, 2**bits, (k, n), dtype=torch.int32)
    zeros = torch.randint(0, 2**bits, (groups, n), dtype=torch.int32)
    qweight = planar_pack_rows(codes, bits).to(dev)
    qzeros = planar_pack_cols(zeros, bits).to(dev)
    scales = (torch.rand(groups, n, dtype=dtype) * 0.01 + 0.005).to(dev)
    g_idx = (torch.arange(k, dtype=torch.int32) // group_size).to(dev)

    _env_banner(dtype_name)
    peak_tflops = _theoretical_peak_tflops(dtype_name, torch.cuda.get_device_properties(torch.cuda.current_device()))
    print(f"shape        : K={k} N={n} bits={bits} dtype={dtype_name} PANGOLIN_MAX_M={PANGOLIN_MAX_M} iters=50")
    header = (
        f"{'M':>4} | {'latency_ms':>12} | {'gflops':>10} | {'peak_tflops':>12} | "
        f"{'achieved_%':>11} | {'dense_ms':>10} | {'speedup':>8} | "
        f"{'max_diff':>10} | {'mean_sq':>10}"
    )
    print(header)
    print("-" * 95)
    for m in (1, 2, 4, 8, 16, 32):
        x = (torch.randn(m, k, dtype=dtype) * 0.5).to(dev)
        try:
            # warmup
            for _ in range(5):
                _ = pangolin_gemv(x, qweight, scales, qzeros, g_idx, bits)
            torch.cuda.synchronize()
            ms = bench(lambda: pangolin_gemv(x, qweight, scales, qzeros, g_idx, bits))
            out = pangolin_gemv(x, qweight, scales, qzeros, g_idx, bits)

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

                dense_ms = bench(dense_fn)
            diff = (out.float() - ref).abs()
            gflops = (2.0 * m * k * n) / (ms * 1e6)
            achieved_pct = (gflops / (peak_tflops * 1000.0)) * 100.0
            speedup = dense_ms / ms
            print(
                f"{m:>4} | {ms:12.4f} | {gflops:10.2f} | {peak_tflops:12.2f} | {achieved_pct:11.2f} | "
                f"{dense_ms:10.4f} | {speedup:8.2f} | {diff.max().item():10.4e} | {diff.pow(2).mean().item():10.4e}"
            )
        except Exception as exc:
            print(f"{m:>4} | FAILED: {exc}")


if __name__ == "__main__":
    preflight_idle_check()
    main()
