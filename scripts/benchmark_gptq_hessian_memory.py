"""Measure GPTQ Hessian inversion time and peak CUDA allocation.

The allocating reference uses the previous Hessian inversion operations. Run
after CUDA warmup so extension setup and allocator initialization are excluded.
"""

import argparse
import hashlib
import statistics
import time

import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def allocating_reference(hessian: torch.Tensor, damp_percent: float) -> torch.Tensor:
    diagonal = hessian.diagonal()
    original = diagonal.clone()
    diagonal.add_(damp_percent * torch.mean(original))
    chol = torch.linalg.cholesky(hessian)
    factor = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
    diagonal.copy_(original)
    return factor


def measure(fn, base: torch.Tensor):
    hessian = base.clone()
    torch.cuda.synchronize()
    resident_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    factor = fn(hessian)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    additional_bytes = torch.cuda.max_memory_allocated() - resident_bytes
    return elapsed, additional_bytes, factor, hessian


def measure_full_quantize(size: int):
    """Report the full quantization peak, which may occur outside inversion."""
    torch.manual_seed(431)
    layer = torch.nn.Linear(size, size, bias=False, device="cuda")
    config = QuantizeConfig(bits=4, group_size=128, act_group_aware=True)
    task = GPTQ(layer, qcfg=config)
    task.quantizer.configure(perchannel=True)
    inputs = torch.randn(1, 256, size, device="cuda")
    task.add_batch(inputs, None)
    del inputs

    torch.cuda.synchronize()
    resident_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = task.quantize(blocksize=128)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    additional_bytes = torch.cuda.max_memory_allocated() - resident_bytes
    hashes = [
        hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()
        for tensor in output[:4]
    ]
    print(
        f"full_quantize: resident_mib={resident_bytes / 2**20:.1f} "
        f"extra_peak_mib={additional_bytes / 2**20:.1f} seconds={elapsed:.6f}"
    )
    print(f"full_quantize_hashes={hashes} loss={output[5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--full-quantize", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.full_quantize:
        measure_full_quantize(args.size)

    config = QuantizeConfig(bits=4, group_size=128, damp_percent=0.05)
    layer = torch.nn.Linear(args.size, 1, bias=False, device="cuda")
    task = GPTQ(layer, qcfg=config)
    base = torch.full((args.size, args.size), 0.001, device="cuda", dtype=torch.float32)
    base.diagonal().fill_(2.0)

    def current(hessian):
        factor, _ = task.hessian_inverse(hessian)
        return factor

    methods = {
        "allocating_reference": lambda h: allocating_reference(h, config.damp_percent),
        "current": current,
    }
    results = {}
    for name, fn in methods.items():
        warmup = measure(fn, base)
        del warmup
        trials = [measure(fn, base) for _ in range(args.trials)]
        times = [trial[0] for trial in trials]
        peaks = [trial[1] for trial in trials]
        for _, _, factor, hessian in trials:
            torch.testing.assert_close(hessian, base, rtol=0, atol=0)
            assert factor.shape == base.shape
        results[name] = (statistics.median(times), max(peaks), trials[-1][2])

    torch.testing.assert_close(results["current"][2], results["allocating_reference"][2], rtol=0, atol=0)
    matrix_mib = base.numel() * base.element_size() / 2**20
    print(f"device={torch.cuda.get_device_name()} size={args.size} trials={args.trials} matrix_mib={matrix_mib:.1f}")
    for name, (seconds, peak_bytes, _) in results.items():
        print(f"{name}: median_seconds={seconds:.6f} extra_peak_mib={peak_bytes / 2**20:.1f}")
    print("factor_parity=bitwise_equal hessian_restored=bitwise_equal")


if __name__ == "__main__":
    main()
