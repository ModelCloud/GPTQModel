import time

import pytest
import torch
from tabulate import tabulate

from gptqmodel.quantization.gptq import GPTQ


LLAMA_INPUT_SHAPES = [
    (1, 1, 4096),     # single token
    (1, 16, 4096),    # micro batch
    (4, 128, 4096),
    (8, 512, 4096),
    (8, 1024, 4096),
    (16, 1024, 4096),
    (8, 2048, 4096),
    (16, 2048, 4096),
]


def _benchmark(fn, x, n_iter=50, warmup=10):
    """Run warmup + timed iterations, return avg ms per call."""
    for _ in range(warmup):
        _ = fn(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iter):
        _ = fn(x)
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) * 1000.0 / n_iter  # ms


#@pytest.mark.cuda
def test_gptq_process_batch_vs_old():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for benchmark")

    device = torch.device("cuda")

    hidden_dim = 4096
    linear_module = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)

    quantizer = GPTQ(linear_module)

    results = []      # benchmark summary
    accuracy_rows = []  # accuracy diff summary

    for batch, seq, hidden in LLAMA_INPUT_SHAPES:
        assert hidden == hidden_dim
        x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)

        # --- Accuracy check ---
        quantizer.H.zero_(); quantizer.nsamples = 0
        quantizer.process_batch(x)
        H_new = quantizer.H.clone().detach()
        nsamples_new = quantizer.nsamples

        quantizer.H.zero_(); quantizer.nsamples = 0
        quantizer.process_batch_old(x)
        H_old = quantizer.H.clone().detach()
        nsamples_old = quantizer.nsamples

        assert nsamples_new == nsamples_old, f"nsamples mismatch for shape {(batch, seq, hidden)}"

        # compute diffs
        diff = (H_new - H_old).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        rel_diff = diff / (H_old.abs() + 1e-12)
        max_rel = rel_diff.max().item()
        mean_rel = rel_diff.mean().item()

        # log in table
        accuracy_rows.append([
            f"{batch}x{seq}x{hidden}",
            f"{max_abs:.3e}",
            f"{mean_abs:.3e}",
            f"{max_rel:.3e}",
            f"{mean_rel:.3e}"
        ])

        # assert with realistic tolerance
        torch.testing.assert_close(
            H_new, H_old, rtol=1e-5, atol=1e-5,
            msg=f"H mismatch for shape {(batch, seq, hidden)} | "
                f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
        )

        # --- Benchmark ---
        quantizer.H.zero_(); quantizer.nsamples = 0
        time_new = _benchmark(quantizer.process_batch, x)

        quantizer.H.zero_(); quantizer.nsamples = 0
        time_old = _benchmark(quantizer.process_batch_old, x)

        speedup = time_old / time_new if time_new > 0 else float("inf")

        results.append([
            f"{batch}x{seq}x{hidden}",
            f"{time_new:.3f}",
            f"{time_old:.3f}",
            f"{speedup:.2f}x"
        ])

        # cleanup
        del x, H_new, H_old, diff, rel_diff, max_abs, mean_abs, max_rel, mean_rel
        torch.cuda.empty_cache()

    # --- Print accuracy summary ---
    acc_headers = ["Shape (B x S x H)", "Max Abs", "Mean Abs", "Max Rel", "Mean Rel"]
    acc_table = tabulate(accuracy_rows, headers=acc_headers, tablefmt="github")
    print("\n=== GPTQ Accuracy Diff Summary ===\n" + acc_table + "\n")

    # --- Print benchmark summary ---
    bench_headers = ["Shape (B x S x H)", "New (ms)", "Old (ms)", "Speedup"]
    bench_table = tabulate(results, headers=bench_headers, tablefmt="github")
    print("\n=== GPTQ Benchmark Summary ===\n" + bench_table + "\n")

    del results, bench_table, bench_headers, accuracy_rows, acc_table, acc_headers
    torch.cuda.empty_cache()
