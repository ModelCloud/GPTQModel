# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import statistics
import time

import pytest
import torch
import torch.nn as nn

from gptqmodel.utils.cuda_activation_buffer import CudaEventActivationBuffer


class PinnedHostPool:
    def __init__(self) -> None:
        self._store = {}
        self.hits = 0
        self.misses = 0

    def acquire(self, shape: torch.Size, dtype: torch.dtype, layout: torch.layout) -> torch.Tensor:
        key = (tuple(shape), dtype, layout)
        bucket = self._store.get(key)
        if bucket:
            self.hits += 1
            return bucket.pop()
        self.misses += 1
        return torch.empty(shape, dtype=dtype, layout=layout, device="cpu", pin_memory=True)

    def release(self, tensor: torch.Tensor) -> None:
        key = (tuple(tensor.shape), tensor.dtype, tensor.layout)
        self._store.setdefault(key, []).append(tensor)


pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available()) or torch.cuda.device_count() <= 6,
    reason="CUDA device 6 is required for this test",
)


class ActivationEmitter(nn.Module):
    """
    Minimal module that mimics a transformer block stage emitting large activations.

    We keep computation intentionally light so that host transfer dominates timing,
    highlighting the benefit of async CUDA stream copies.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class ForwardBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.emitter = ActivationEmitter(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.emitter(x))


def _run_variant(
    mode: str,
    model: nn.Module,
    batch_input: torch.Tensor,
    *,
    warmup: int = 2,
    steps: int = 6,
    buffer_kwargs: dict | None = None,
    recycle_packets: bool = False,
):
    assert mode in {"gpu", "sync", "async"}

    device = batch_input.device
    forward_latencies = []
    drain_latencies = []
    captured_outputs = []
    tmp_store = []

    buffer_kwargs = buffer_kwargs or {}
    buffer = CudaEventActivationBuffer(device=device, **buffer_kwargs) if mode == "async" else None

    def _hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        tensor = tensor.detach()
        if mode == "async":
            assert buffer is not None
            buffer.capture_async(tensor)
        elif mode == "sync":
            tmp_store.append(tensor.to("cpu"))
        else:
            tmp_store.append(tensor)

    handle = model.emitter.register_forward_hook(_hook)

    try:
        current_stream = torch.cuda.current_stream(device)
        with torch.inference_mode():
            total_steps = warmup + steps
            for idx in range(total_steps):
                tmp_store.clear()
                current_stream.synchronize()
                t0 = time.perf_counter()
                _ = model(batch_input)
                current_stream.synchronize()
                elapsed = time.perf_counter() - t0
                if idx >= warmup:
                    forward_latencies.append(elapsed)

                if mode == "async":
                    assert buffer is not None
                    t1 = time.perf_counter()
                    drained = buffer.drain(wait=True)
                    drain_elapsed = time.perf_counter() - t1
                    if idx >= warmup:
                        drain_latencies.append(drain_elapsed)
                        for pkt in drained:
                            captured_outputs.append(pkt.host_tensor.clone())
                            if recycle_packets:
                                buffer.recycle(pkt)
                else:
                    t1 = time.perf_counter()
                    drained = list(tmp_store)
                    drain_elapsed = time.perf_counter() - t1
                    if idx >= warmup:
                        drain_latencies.append(drain_elapsed)
                        captured_outputs.extend(drained)
    finally:
        handle.remove()
        if buffer is not None:
            leftover = buffer.drain(wait=True)
            for pkt in leftover:
                captured_outputs.append(pkt.host_tensor.clone())
                if recycle_packets:
                    buffer.recycle(pkt)

    return forward_latencies, drain_latencies, captured_outputs


def test_cuda_event_stream_activation_buffer_benchmarks():
    """
    Benchmarks three capture strategies that mirror GPTQ forward hooks:

    - gpu: baseline, keep activations resident on the device.
    - sync: copy to CPU immediately via blocking `.cpu()`.
    - async: enqueue D2H on a dedicated CUDA stream and wait only once drained.
    """
    device = torch.device("cuda", 6)
    torch.cuda.set_device(device)

    batch = 4
    seq = 2048
    hidden_dim = 4096
    dtype = torch.float16

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    template_model = ForwardBlock(hidden_dim).to(device=device, dtype=dtype).eval()
    state = template_model.state_dict()
    del template_model

    model_gpu = ForwardBlock(hidden_dim).to(device=device, dtype=dtype).eval()
    model_gpu.load_state_dict(state)

    model_sync = ForwardBlock(hidden_dim).to(device=device, dtype=dtype).eval()
    model_sync.load_state_dict(state)

    model_async = ForwardBlock(hidden_dim).to(device=device, dtype=dtype).eval()
    model_async.load_state_dict(state)

    batch_input = torch.randn(batch, seq, hidden_dim, device=device, dtype=dtype)

    # Warm everything so subsequent measurements reflect steady-state timings.
    for _ in range(3):
        _ = model_gpu(batch_input)
        _ = model_sync(batch_input)
        _ = model_async(batch_input)

    gpu_forward, gpu_drain, gpu_outputs = _run_variant("gpu", model_gpu, batch_input)
    sync_forward, sync_drain, sync_outputs = _run_variant("sync", model_sync, batch_input)
    async_forward, async_drain, async_outputs = _run_variant("async", model_async, batch_input)

    pool = PinnedHostPool()
    pool_kwargs = {
        "host_allocator": pool.acquire,
        "host_reclaimer": pool.release,
    }
    async_pool_forward, async_pool_drain, async_pool_outputs = _run_variant(
        "async",
        model_async,
        batch_input,
        warmup=0,
        steps=5,
        buffer_kwargs=pool_kwargs,
        recycle_packets=True,
    )

    gpu_outputs_cpu = [t.detach().cpu() for t in gpu_outputs]

    assert len(gpu_outputs) == len(sync_outputs) == len(async_outputs) > 0
    for baseline, candidate in zip(sync_outputs, async_outputs):
        assert torch.allclose(baseline, candidate, atol=0, rtol=0)
    for baseline, candidate in zip(sync_outputs, gpu_outputs_cpu):
        assert torch.allclose(baseline, candidate, atol=0, rtol=0)
    reference_cpu = sync_outputs[0]
    for candidate in async_pool_outputs:
        assert torch.allclose(reference_cpu, candidate, atol=0, rtol=0)

    mean_gpu_forward = statistics.mean(gpu_forward)
    mean_sync_forward = statistics.mean(sync_forward)
    mean_async_forward = statistics.mean(async_forward)

    mean_gpu_drain = statistics.mean(gpu_drain)
    mean_sync_drain = statistics.mean(sync_drain)
    mean_async_drain = statistics.mean(async_drain)

    async_combined = [f + d for f, d in zip(async_forward, async_drain)]
    combined_mean = statistics.mean(async_combined)

    # Async capture should avoid additional forward blocking relative to sync copies.
    assert mean_async_forward <= mean_sync_forward

    # Async totals should be bounded by the synchronous copy baseline.
    assert combined_mean <= mean_sync_forward * 1.1

    miss_forward = async_pool_forward[0]
    hit_forward_mean = statistics.mean(async_pool_forward[1:]) if len(async_pool_forward) > 1 else miss_forward
    miss_drain = async_pool_drain[0]
    hit_drain_mean = statistics.mean(async_pool_drain[1:]) if len(async_pool_drain) > 1 else miss_drain

    assert pool.misses >= 1
    assert pool.hits >= 1
    assert hit_forward_mean <= miss_forward * 0.75

    print(
        "[CUDA6 Activation Copy Benchmark]\n"
        f"  gpu forward mean:  {mean_gpu_forward * 1e3:.2f} ms\n"
        f"  gpu drain mean:    {mean_gpu_drain * 1e3:.2f} ms\n"
        f"  sync forward mean: {mean_sync_forward * 1e3:.2f} ms\n"
        f"  sync drain mean:   {mean_sync_drain * 1e3:.2f} ms\n"
        f"  async forward mean:{mean_async_forward * 1e3:.2f} ms\n"
        f"  async drain mean:  {mean_async_drain * 1e3:.2f} ms\n"
        f"  async combined:    {combined_mean * 1e3:.2f} ms\n"
        f"  pool miss forward: {miss_forward * 1e3:.2f} ms\n"
        f"  pool hit forward:  {hit_forward_mean * 1e3:.2f} ms\n"
        f"  pool miss drain:   {miss_drain * 1e3:.2f} ms\n"
        f"  pool hit drain:    {hit_drain_mean * 1e3:.2f} ms\n"
        f"  pool stats (hits/misses): {pool.hits}/{pool.misses}"
    )
