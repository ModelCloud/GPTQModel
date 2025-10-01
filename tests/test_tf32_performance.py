import pytest
import torch

from gptqmodel.utils.torch import tf32_disable_guard, tf32_enable_guard


try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover
    tabulate = None


def _supports_bfloat16() -> bool:
    major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    return major >= 8  # Ampere or newer


def _measure_linear_time(batch: int, in_features: int, out_features: int, dtype: torch.dtype, *, runs: int = 10) -> float:
    linear = torch.nn.Linear(in_features, out_features, bias=False).cuda().to(dtype=dtype)
    torch.cuda.manual_seed(0)
    inp = torch.randn(batch, in_features, device="cuda", dtype=dtype)

    # Warmup
    for _ in range(5):
        linear(inp)

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(runs):
        linear(inp)
    end_event.record()

    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms / runs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tf32_toggle_has_no_large_perf_regression(dtype: torch.dtype):
    if dtype is torch.bfloat16 and not _supports_bfloat16():
        pytest.skip("Device does not support bfloat16")

    shapes = [
        # Llama 3 / Mistral style hidden dims
        (64, 4096, 4096),
        (128, 4096, 11008),
        # Qwen 2/3 7B/14B style
        (64, 3584, 15360),
        (64, 8192, 28672),
        # DeepSeek V3 large experts
        (32, 7168, 28672),
        (32, 12288, 49152),
    ]

    results = []

    for batch, in_features, out_features in shapes:
        times_tf32 = []
        times_no_tf32 = []
        max_diff = 0.0

        for _ in range(10):
            with tf32_enable_guard():
                linear = torch.nn.Linear(in_features, out_features, bias=False).cuda().to(dtype=dtype)
                inp = torch.randn(batch, in_features, device="cuda", dtype=dtype)
                out_tf32 = linear(inp)
                times_tf32.append(_measure_linear_time(batch, in_features, out_features, dtype))

            with tf32_disable_guard():
                out_no_tf32 = linear(inp)
                times_no_tf32.append(_measure_linear_time(batch, in_features, out_features, dtype))

            max_diff = max(max_diff, float(torch.max(torch.abs(out_tf32 - out_no_tf32)).item()))

        avg_tf32 = sum(times_tf32) / len(times_tf32)
        avg_no_tf32 = sum(times_no_tf32) / len(times_no_tf32)

        slower = max(avg_tf32, avg_no_tf32)
        faster = min(avg_tf32, avg_no_tf32)

        assert slower <= faster * 1.5, (
            f"TF32 toggle caused >50% slowdown for dtype={dtype}, shape={batch}x{in_features}->{out_features}: "
            f"tf32={avg_tf32:.3f}ms, no_tf32={avg_no_tf32:.3f}ms"
        )

        results.append(
            {
                "dtype": str(dtype).split(".")[-1],
                "shape": f"{batch}x{in_features}->{out_features}",
                "avg_tf32_ms": avg_tf32,
                "avg_no_tf32_ms": avg_no_tf32,
                "max_abs_diff": max_diff,
            }
        )

    if tabulate:
        table = tabulate(
            [
                [r["dtype"], r["shape"], f"{r['avg_tf32_ms']:.3f}", f"{r['avg_no_tf32_ms']:.3f}", f"{r['max_abs_diff']:.3e}"]
                for r in results
            ],
            headers=["dtype", "shape", "avg_tf32_ms", "avg_no_tf32_ms", "max_abs_diff"],
        )
        print("\nTF32 performance summary:\n" + table)
    else:
        print("\nTF32 performance summary:")
        for r in results:
            print(
                f"dtype={r['dtype']} shape={r['shape']} avg_tf32={r['avg_tf32_ms']:.3f}ms "
                f"avg_no_tf32={r['avg_no_tf32_ms']:.3f}ms max_abs_diff={r['max_abs_diff']:.3e}"
            )
