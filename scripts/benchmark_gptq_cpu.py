# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the compiled CPU GPTQ block path against the eager serial loop.

Reports wall time and end-to-end Q tensor equality for common MLP shapes.
Can be run directly: python scripts/benchmark_gptq_cpu.py
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptqmodel.nn_modules.qlinear.pack_block_ext import gptq_block_cpu
from gptqmodel.quantization import gptq as gptq_module
from gptqmodel.quantization.config import QuantizeConfig, ScaleSearchConfig


def _serial_block(W1, Hinv1, scale, zero, maxq, group_size, groupwise):
    """Eager torch.addr reference for gptq_block_cpu."""
    local = W1.clone()
    rows, count = W1.shape
    Q = torch.empty_like(W1)
    Err = torch.empty_like(W1)
    for i in range(count):
        g = i // group_size
        sc = scale[:, g : g + 1]
        zv = zero[:, g : g + 1]
        w = local[:, i : i + 1]
        if groupwise:
            q = sc * torch.clamp(torch.round(w / sc), -maxq, maxq)
        else:
            q = sc * (torch.clamp(torch.round(w / sc) + zv, 0.0, maxq) - zv)
        Q[:, i] = q.squeeze(-1)
        err = (w - q) / Hinv1[i, i]
        Err[:, i] = err.squeeze(-1)
        local[:, i:] = torch.addr(local[:, i:], err.view(-1), Hinv1[i, i:], alpha=-1.0)
    return Q, Err


def bench_block(rows, block_size=128, group_size=128, iters=5):
    """Time gptq_block_cpu vs the eager serial loop for a single block."""
    torch.manual_seed(0)
    torch.set_num_threads(8)
    W1 = torch.randn(rows, block_size, dtype=torch.float32)
    H = torch.randn(block_size, block_size, dtype=torch.float32)
    H = H @ H.T
    H.diagonal().add_(0.1)
    Hinv1 = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)))
    scale = torch.rand(rows, block_size // group_size, dtype=torch.float32) * 0.05 + 0.01
    zero = torch.rand(rows, block_size // group_size, dtype=torch.float32) * 5

    # Warm-up + cache extension.
    for _ in range(10):
        _ = gptq_block_cpu(W1, Hinv1, scale, zero, 15, group_size, False)

    t0 = time.perf_counter()
    for _ in range(iters):
        Q_ext, Err_ext = gptq_block_cpu(W1, Hinv1, scale, zero, 15, group_size, False)
    t_ext = (time.perf_counter() - t0) / iters

    for _ in range(3):
        _ = _serial_block(W1, Hinv1, scale, zero, 15, group_size, False)

    t0 = time.perf_counter()
    for _ in range(iters):
        Q_ref, Err_ref = _serial_block(W1, Hinv1, scale, zero, 15, group_size, False)
    t_ref = (time.perf_counter() - t0) / iters

    assert torch.equal(Q_ext, Q_ref)
    assert torch.equal(Err_ext, Err_ref)
    return t_ext, t_ref


def bench_quantize(in_features, out_features, scale_search=None, iters=3):
    """Time GPTQ.quantize with and without the compiled CPU block path."""
    torch.manual_seed(0)
    torch.set_num_threads(8)
    layer = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        act_group_aware=False,
        scale_search=scale_search,
        offload_to_disk=False,
    )
    inp = torch.randn(2048, in_features, dtype=torch.float32)

    # Keep the quantization block on CPU so both runs use the same backend.
    os.environ["GPTQMODEL_CUDA_BLOCK"] = "0"
    os.environ["GPTQMODEL_SCALE_SEARCH_TRITON"] = "0"
    os.environ["GPTQMODEL_SCALE_SEARCH_CPU"] = "0"

    def run(cpu_block):
        os.environ["GPTQMODEL_BLOCK_CPU"] = "1" if cpu_block else "0"
        gptq = gptq_module.GPTQ(layer, qcfg=qcfg)
        gptq.quantizer.configure(perchannel=True, sym=True)
        gptq.add_batch(inp, None)
        t0 = time.perf_counter()
        Q, *_ = gptq.quantize(blocksize=128)
        return time.perf_counter() - t0, Q

    # Warm-up.
    _, Q_cpu = run(True)
    _, Q_ref = run(False)
    assert torch.equal(Q_cpu, Q_ref)

    t_cpu = min(run(True)[0] for _ in range(iters))
    t_ref = min(run(False)[0] for _ in range(iters))
    return t_cpu, t_ref


def main():
    print("Benchmarking compiled CPU GPTQ block (lower is better)\n")
    print(f"{'shape':>20}  {'cpu (ms)':>10}  {'eager (ms)':>12}  {'speedup':>8}  notes")
    print("-" * 72)

    for rows in [2048, 4096, 8192]:
        t_ext, t_ref = bench_block(rows, block_size=128, iters=5)
        print(
            f"block [{rows:>5}x{128:<5}]  "
            f"{t_ext * 1000:>10.2f}  {t_ref * 1000:>12.2f}  "
            f"{t_ref / t_ext:>8.2f}  bit-exact"
        )

    print()
    print("Benchmarking end-to-end GPTQ.quantize (lower is better)\n")
    print(f"{'shape':>20}  {'search':>12}  {'cpu (s)':>10}  {'eager (s)':>12}  {'speedup':>8}")
    print("-" * 82)

    shapes = [
        ("mlp.down 4k x 11k", 11008, 4096),
        ("mlp.down 2k x 8k", 8192, 2048),
    ]
    for label, in_features, out_features in shapes:
        for search in (None, ScaleSearchConfig.ACTIVATION):
            search_name = "none" if search is None else "activation"
            t_cpu, t_ref = bench_quantize(in_features, out_features, scale_search=search, iters=2)
            print(
                f"{label:>20}  {search_name:>12}  "
                f"{t_cpu:>10.3f}  {t_ref:>12.3f}  {t_ref / t_cpu:>8.2f}"
            )


if __name__ == "__main__":
    main()
