import random
import statistics
import time

import pytest
import torch
from tabulate import tabulate

from gptqmodel.quantization import gar, gar_ref


def _benchmark_fn(label, fn, device, warmup_runs=3, measured_runs=10):
    torch.cuda.synchronize(device)
    for _ in range(warmup_runs):
        fn()
    torch.cuda.synchronize(device)

    timings = []
    memories = []
    for _ in range(measured_runs):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)
        memories.append(torch.cuda.max_memory_allocated(device) / (1024**2))
    return {
        "label": label,
        "time_ms": timings,
        "memory_mb": memories,
    }


def test_benchmark_gar_cuda():
    if torch.cuda.device_count() == 0:
        pytest.skip("CUDA device not available for benchmark")

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_device(device)

    groupsize = 128
    num_groups = 2048
    diag_H = torch.randn(num_groups * groupsize, device=device, dtype=torch.float32)

    def optimized_call():
        local, values = gar.compute_local_perms(diag_H, groupsize, return_values=True)
        global_perm = gar.compute_global_perm(
            diag_H, groupsize, precomputed_values=values
        )
        result = gar.compose_final_perm(local, global_perm, groupsize)
        del values
        return result

    def original_call():
        local = gar_ref.compute_local_perms_original(diag_H, groupsize)
        global_perm = gar_ref.compute_global_perm_original(diag_H, groupsize)
        return gar_ref.compose_final_perm_original(local, global_perm, groupsize)

    # Ensure both implementations agree before timing to detect accuracy regressions.
    optimized_result = optimized_call()
    original_result = original_call()
    torch.testing.assert_close(
        optimized_result.to(device="cpu", dtype=torch.float64),
        original_result.to(device="cpu", dtype=torch.float64),
        rtol=0,
        atol=0,
    )

    optimized = _benchmark_fn("optimized", optimized_call, device)
    original = _benchmark_fn("original", original_call, device)

    def summarize(stats):
        return {
            "label": stats["label"],
            "time_mean_ms": statistics.mean(stats["time_ms"]),
            "time_std_ms": statistics.pstdev(stats["time_ms"]),
            "mem_mean_mb": statistics.mean(stats["memory_mb"]),
            "mem_max_mb": max(stats["memory_mb"]),
        }

    optimized_summary = summarize(optimized)
    original_summary = summarize(original)
    speedup = original_summary["time_mean_ms"] / optimized_summary["time_mean_ms"]
    mem_delta = optimized_summary["mem_mean_mb"] - original_summary["mem_mean_mb"]

    table = [
        [
            optimized_summary["label"],
            f"{optimized_summary['time_mean_ms']:.3f}",
            f"{optimized_summary['time_std_ms']:.3f}",
            f"{optimized_summary['mem_mean_mb']:.2f}",
            f"{optimized_summary['mem_max_mb']:.2f}",
        ],
        [
            original_summary["label"],
            f"{original_summary['time_mean_ms']:.3f}",
            f"{original_summary['time_std_ms']:.3f}",
            f"{original_summary['mem_mean_mb']:.2f}",
            f"{original_summary['mem_max_mb']:.2f}",
        ],
    ]

    headers = ["version", "mean_time_ms", "std_time_ms", "mean_mem_mb", "max_mem_mb"]
    print(tabulate(table, headers=headers, tablefmt="github"))
    print(
        f"Speedup (original/optimized): {speedup:.2f}x; "
        f"Mean memory delta (optimized - original): {mem_delta:.2f} MB"
    )

    assert optimized_summary["time_mean_ms"] < original_summary["time_mean_ms"]


@pytest.mark.parametrize("seed", range(64))
def test_gar_accuracy_randomized(seed):
    if torch.cuda.device_count() == 0:
        pytest.skip("CUDA device not available for accuracy sweep")

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_device(device)

    random.seed(seed)
    torch.manual_seed(seed)

    groupsize_candidates = [8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256]
    groupsize = random.choice(groupsize_candidates)
    num_groups = random.randint(0, 2048)
    remainder = random.randint(0, max(groupsize - 1, 0))
    length = num_groups * groupsize + remainder

    diag_H = torch.randn(length, device=device, dtype=torch.float32)

    opt_local, opt_values = gar.compute_local_perms(diag_H, groupsize, return_values=True)
    opt_global = gar.compute_global_perm(
        diag_H, groupsize, precomputed_values=opt_values
    )
    opt_final = gar.compose_final_perm(opt_local, opt_global, groupsize)

    orig_local = gar_ref.compute_local_perms_original(diag_H, groupsize)
    orig_global = gar_ref.compute_global_perm_original(diag_H, groupsize)
    orig_final = gar_ref.compose_final_perm_original(orig_local, orig_global, groupsize)

    opt_perm_values = diag_H[opt_final]
    orig_perm_values = diag_H[orig_final]
    torch.testing.assert_close(
        opt_perm_values.to(dtype=torch.float64),
        orig_perm_values.to(dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )
