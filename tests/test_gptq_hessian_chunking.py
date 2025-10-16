# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math
import time
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available()) or torch.cuda.device_count() <= 6,
    reason="CUDA device 6 is required for this benchmark test",
)


def _make_module(hidden_dim: int, device: torch.device) -> nn.Linear:
    layer = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.float16)
    return layer.to(device).eval()


def _run_add_batch(
    hidden_dim: int,
    *,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    total_batches: int,
    warmup_batches: int,
    chunk_bytes: Optional[int],
) -> Dict[str, float]:
    qcfg = QuantizeConfig()
    qcfg.hessian_chunk_bytes = chunk_bytes

    module = _make_module(hidden_dim, device)
    gptq = GPTQ(module, qcfg=qcfg)
    dummy_outputs = torch.empty(0, device=device)

    def _one_batch(idx: int):
        activations = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16)
        gptq.add_batch(activations, dummy_outputs, batch_index=idx)

    for idx in range(warmup_batches):
        _one_batch(idx)

    torch.cuda.synchronize(device)
    baseline_alloc = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    measured = 0
    start = time.perf_counter()
    for idx in range(warmup_batches, total_batches):
        _one_batch(idx)
        measured += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_alloc = torch.cuda.max_memory_allocated(device)
    gptq_mod._WORKSPACE_CACHE.clear()

    per_batch = elapsed / measured if measured else 0.0
    activation_mb = (batch_size * seq_len * hidden_dim * 2) / (1024**2)
    peak_delta_mb = max(0.0, (peak_alloc - baseline_alloc) / (1024**2))

    chunk_rows = gptq._resolve_hessian_chunk_size(batch_size * seq_len, torch.float32)

    return {
        "chunk_bytes": chunk_bytes,
        "per_batch_sec": per_batch,
        "total_sec": elapsed,
        "peak_delta_mb": peak_delta_mb,
        "activation_mb": activation_mb,
        "chunk_rows": chunk_rows,
    }


def test_hessian_chunking_vram_vs_latency():
    device = torch.device("cuda", 6)
    torch.cuda.set_device(device)

    configs: List[Tuple[str, int]] = [
        ("llama3", 4096),
        ("qwen3", 3584),
    ]
    chunk_options = [None, 64 << 20, 32 << 20, 16 << 20, 8 << 20, 4 << 20]

    total_batches = 6
    warmup_batches = 2
    batch_size = 4
    seq_len = 512

    for name, hidden_dim in configs:
        results: List[Dict[str, float]] = []
        for chunk_bytes in chunk_options:
            stats = _run_add_batch(
                hidden_dim,
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                total_batches=total_batches,
                warmup_batches=warmup_batches,
                chunk_bytes=chunk_bytes,
            )
            results.append(stats)

        baseline = results[0]
        best = min(results, key=lambda x: x["peak_delta_mb"])

        print(f"\n[{name.upper()}] activation ~{baseline['activation_mb']:.2f} MiB")
        for stats in results:
            chunk_label = "none" if stats["chunk_bytes"] is None else f"{stats['chunk_bytes'] // (1<<20)} MiB"
            print(
                f"  chunk={chunk_label:<5} | chunk_rows={stats['chunk_rows']} | "
                f"peak ΔVRAM {stats['peak_delta_mb']:.2f} MiB | per-batch {stats['per_batch_sec'] * 1e3:.2f} ms"
            )

        assert math.isclose(baseline["activation_mb"], best["activation_mb"], rel_tol=1e-6)

        smallest_chunk = results[-1]
        assert smallest_chunk["peak_delta_mb"] >= baseline["peak_delta_mb"]
        assert smallest_chunk["per_batch_sec"] <= baseline["per_batch_sec"] * 4.0
