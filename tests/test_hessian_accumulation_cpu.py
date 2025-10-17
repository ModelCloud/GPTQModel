# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass
from typing import List, Tuple

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.safe import THREADPOOLCTL


pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available()) or torch.cuda.device_count() <= 6,
    reason="CUDA device 6 is required for this benchmark test",
)


@dataclass
class BenchmarkResult:
    per_batch_seconds: float
    total_seconds: float
    batches_measured: int


def _make_module(hidden_dim: int, device: torch.device) -> nn.Linear:
    module = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.float16).to(device)
    module.eval()
    return module


def _generate_samples(
    batches: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    device: torch.device,
) -> List[torch.Tensor]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    samples = []
    for _ in range(batches):
        sample = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16)
        samples.append(sample.contiguous())
    return samples


def _to_pinned_cpu(samples: List[torch.Tensor]) -> List[torch.Tensor]:
    pinned = []
    for tensor in samples:
        host = tensor.to(device="cpu", non_blocking=False).contiguous()
        pinned.append(host.pin_memory())
    return pinned


def _benchmark_add_batch(
    module: nn.Module,
    samples: List[torch.Tensor],
    warmup_batches: int,
    device: torch.device,
) -> BenchmarkResult:
    gptq = GPTQ(module)
    dummy_outputs = torch.empty(0, device=samples[0].device)

    # Warmup to populate internal workspaces and caches
    for idx in range(warmup_batches):
        gptq.add_batch(samples[idx], dummy_outputs, batch_index=idx)

    if samples[0].device.type == "cuda":
        torch.cuda.synchronize(device)

    measured = 0
    start = time.perf_counter()

    for idx in range(warmup_batches, len(samples)):
        gptq.add_batch(samples[idx], dummy_outputs, batch_index=idx)
        measured += 1

    if samples[0].device.type == "cuda":
        torch.cuda.synchronize(device)

    total = time.perf_counter() - start
    per_batch = total / measured if measured else 0.0
    return BenchmarkResult(per_batch_seconds=per_batch, total_seconds=total, batches_measured=measured)


def test_hessian_accumulation_cpu_vs_gpu():
    device = torch.device("cuda", 6)
    torch.cuda.set_device(device)

    configs: List[Tuple[str, int]] = [
        ("llama3", 4096),
        ("qwen3", 3584),
    ]

    total_batches = 6
    warmup_batches = 2
    batch_size = 4
    seq_len = 256

    for name, hidden_dim in configs:
        module_gpu = _make_module(hidden_dim, device=device)
        gpu_samples = _generate_samples(total_batches, batch_size, seq_len, hidden_dim, device)
        cpu_samples = _to_pinned_cpu(gpu_samples)

        gpu_result = _benchmark_add_batch(module_gpu, gpu_samples, warmup_batches, device)

        module_cpu = _make_module(hidden_dim, device=device)
        with THREADPOOLCTL.threadpool_limits(limits=16):
            cpu_result = _benchmark_add_batch(module_cpu, cpu_samples, warmup_batches, device)

        assert gpu_result.batches_measured == cpu_result.batches_measured == total_batches - warmup_batches

        print(
            f"[{name.upper()}] GPU add_batch: {gpu_result.per_batch_seconds*1e3:.3f} ms/batch "
            f"(total {gpu_result.total_seconds:.3f} s) | "
            f"CPU add_batch (threads=16): {cpu_result.per_batch_seconds*1e3:.3f} ms/batch "
            f"(total {cpu_result.total_seconds:.3f} s)"
        )

        assert cpu_result.per_batch_seconds >= gpu_result.per_batch_seconds
