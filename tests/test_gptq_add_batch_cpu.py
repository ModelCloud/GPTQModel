# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass
from typing import List, Tuple

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.gptq import GPTQ


pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available()) or torch.cuda.device_count() <= 6,
    reason="CUDA device 6 is required for this benchmark test",
)


@dataclass
class PathStats:
    per_batch_seconds: float
    total_seconds: float
    peak_bytes: int
    batches_measured: int


def _make_module(hidden_dim: int, device: torch.device) -> nn.Linear:
    layer = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.float16)
    return layer.to(device).eval()


def _generate_input(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16)


def _benchmark_add_batch(
    module: nn.Module,
    device: torch.device,
    hidden_dim: int,
    *,
    total_batches: int,
    warmup_batches: int,
    batch_size: int,
    seq_len: int,
    use_cpu_queue: bool,
) -> PathStats:
    gptq = GPTQ(module)
    dummy_outputs = torch.empty(0, device=device)

    def _run_batch(idx: int) -> None:
        activations = _generate_input(batch_size, seq_len, hidden_dim, device=device)
        if use_cpu_queue:
            cpu_activations = activations.detach().to(device="cpu")
            del activations
            gptq.add_batch(cpu_activations, dummy_outputs, batch_index=idx)
        else:
            gptq.add_batch(activations, dummy_outputs, batch_index=idx)

    for idx in range(warmup_batches):
        _run_batch(idx)

    torch.cuda.synchronize(device)
    baseline_alloc = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    measured = 0
    start = time.perf_counter()

    for idx in range(warmup_batches, total_batches):
        _run_batch(idx)
        measured += 1

    torch.cuda.synchronize(device)
    total = time.perf_counter() - start
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_bytes = max(0, peak_alloc - baseline_alloc)
    per_batch = total / measured if measured else 0.0
    return PathStats(per_batch_seconds=per_batch, total_seconds=total, peak_bytes=peak_bytes, batches_measured=measured)


def test_gptq_add_batch_cpu_vs_gpu_queue():
    device = torch.device("cuda", 6)
    torch.cuda.set_device(device)

    configs: List[Tuple[str, int]] = [
        ("llama3", 4096),
        ("qwen3", 3584),
    ]

    total_batches = 8
    warmup_batches = 2
    batch_size = 4
    seq_len = 512

    for name, hidden_dim in configs:
        module_gpu = _make_module(hidden_dim, device=device)
        gpu_stats = _benchmark_add_batch(
            module_gpu,
            device,
            hidden_dim,
            total_batches=total_batches,
            warmup_batches=warmup_batches,
            batch_size=batch_size,
            seq_len=seq_len,
            use_cpu_queue=False,
        )

        module_cpu_queue = _make_module(hidden_dim, device=device)
        cpu_stats = _benchmark_add_batch(
            module_cpu_queue,
            device,
            hidden_dim,
            total_batches=total_batches,
            warmup_batches=warmup_batches,
            batch_size=batch_size,
            seq_len=seq_len,
            use_cpu_queue=True,
        )

        assert gpu_stats.batches_measured == cpu_stats.batches_measured == total_batches - warmup_batches

        print(
            f"[{name.upper()}] GPU queue: {gpu_stats.per_batch_seconds*1e3:.3f} ms/batch "
            f"(total {gpu_stats.total_seconds:.3f} s, peak GPU alloc {gpu_stats.peak_bytes/1024/1024:.2f} MiB) | "
            f"CPU queue: {cpu_stats.per_batch_seconds*1e3:.3f} ms/batch "
            f"(total {cpu_stats.total_seconds:.3f} s, peak GPU alloc {cpu_stats.peak_bytes/1024/1024:.2f} MiB)"
        )

        assert cpu_stats.per_batch_seconds >= gpu_stats.per_batch_seconds
        assert cpu_stats.peak_bytes <= gpu_stats.peak_bytes
