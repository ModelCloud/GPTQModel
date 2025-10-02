import pytest
import torch
import time
from tabulate import tabulate

from gptqmodel.quantization.gptq import GPTQ


LLAMA_INPUT_SHAPES = [
    (1, 1, 4096),     # single token
    (1, 16, 4096),    # micro batch
    (4, 128, 4096),   # medium
    (8, 512, 4096),   # large
    (16, 2048, 4096)  # very large
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

    # GPTQ with new process_batch (streaming version)
    quantizer = GPTQ(linear_module)

    results = []

    for batch, seq, hidden in LLAMA_INPUT_SHAPES:
        assert hidden == hidden_dim
        x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)

        # --- Accuracy test ---
        # Reset H/nsamples for new impl
        quantizer.H.zero_()
        quantizer.nsamples = 0
        quantizer.process_batch(x)
        H_new = quantizer.H.clone().detach()
        nsamples_new = quantizer.nsamples

        # Reset H/nsamples for old impl
        quantizer.H.zero_()
        quantizer.nsamples = 0
        quantizer.process_batch_old(x)
        H_old = quantizer.H.clone().detach()
        nsamples_old = quantizer.nsamples

        # Compare results
        assert nsamples_new == nsamples_old, "Mismatch in nsamples between old and new process_batch"
        torch.testing.assert_close(
            H_new, H_old, rtol=1e-5, atol=1e-5,
            msg=f"H mismatch for shape {(batch, seq, hidden)}"
        )

        # --- Benchmarking ---
        quantizer.H.zero_()
        quantizer.nsamples = 0
        time_new = _benchmark(quantizer.process_batch, x)

        quantizer.H.zero_()
        quantizer.nsamples = 0
        time_old = _benchmark(quantizer.process_batch_old, x)

        speedup = time_old / time_new if time_new > 0 else float("inf")

        results.append([f"{batch}x{seq}x{hidden}",
                        f"{time_new:.3f}",
                        f"{time_old:.3f}",
                        f"{speedup:.2f}x"])

        # cleanup
        del x, H_new, H_old, nsamples_new, nsamples_old, time_new, time_old, speedup
        torch.cuda.empty_cache()

    headers = ["Shape (B x S x H)", "New (ms)", "Old (ms)", "Speedup"]
    table = tabulate(results, headers=headers, tablefmt="github")
    print("\n=== GPTQ Benchmark & Accuracy Summary ===\n" + table + "\n")

    del results, table, headers
    torch.cuda.empty_cache()
