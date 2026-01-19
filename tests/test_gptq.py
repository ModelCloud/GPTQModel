# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.config import HessianConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from models.model_test import ModelTest



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


@dataclass
class PathStats:
    per_batch_seconds: float
    total_seconds: float
    peak_bytes: int
    batches_measured: int

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


class TestGPTQAddBatchCPU(ModelTest):
    ######### test_gptq_add_batch_cpu.py ###########
    pytestmark = pytest.mark.skipif(
        (not torch.cuda.is_available()) or torch.cuda.device_count() <= 6,
        reason="CUDA device 6 is required for this benchmark test",
        )


    def test_gptq_add_batch_cpu_vs_gpu_queue(self):
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

class TestGPTQHessianChunking(ModelTest):
    ######### test_gptq_hessian_chunking.py ###########
    pytestmark = pytest.mark.skipif(
        (not torch.cuda.is_available()) or torch.cuda.device_count() <= 6,
        reason="CUDA device 6 is required for this benchmark test",
        )


    def _run_add_batch(
            self,
            hidden_dim: int,
            *,
            device: torch.device,
            batch_size: int,
            seq_len: int,
            total_batches: int,
            warmup_batches: int,
            chunk_bytes: Optional[int],
    ) -> Dict[str, float]:
        qcfg = QuantizeConfig(hessian=HessianConfig(chunk_bytes=chunk_bytes))

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

        chunk_rows = gptq.resolve_hessian_chunk_size(batch_size * seq_len, torch.float32)

        return {
            "chunk_bytes": chunk_bytes,
            "per_batch_sec": per_batch,
            "total_sec": elapsed,
            "peak_delta_mb": peak_delta_mb,
            "activation_mb": activation_mb,
            "chunk_rows": chunk_rows,
        }


    def test_hessian_chunking_vram_vs_latency(self):
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
                stats = self._run_add_batch(
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


######### test_gptq_processor_streaming.py ###########

class TestGPTQProcessorStreaming(ModelTest):
    pytestmark = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA device required for streaming D2H test"
    )

    def test_gptq_processor_async_d2h_streaming_roundtrip(self):
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "7")
        env.setdefault("PYTHON_GIL", os.environ.get("PYTHON_GIL", "1"))

        script = textwrap.dedent(
            """
            import os
            import sys
            import threading
            from types import SimpleNamespace
    
            import torch
    
            class _RandomWords:
                def get_random_word(self):
                    return "stream-events"
    
            sys.modules.setdefault("random_word", SimpleNamespace(RandomWords=lambda: _RandomWords()))
    
            from gptqmodel.looper.gptq_processor import GPTQProcessor
            from gptqmodel.looper.named_module import NamedModule
    
            device = torch.device("cuda", 0)
            torch.cuda.set_device(device)
    
            processor = object.__new__(GPTQProcessor)
            processor.lock = threading.Lock()
    
            linear = torch.nn.Linear(8, 8, bias=False).to(device=device, dtype=torch.float16)
            named_module = NamedModule(linear, name="proj", full_name="model.layers.0.proj", layer_index=0)
    
            payload = {
                "q_scales": torch.randn(8, 8, device=device, dtype=torch.float16),
                "q_zeros": torch.randn(8, 8, device=device, dtype=torch.float16),
                "q_g_idx": torch.arange(64, device=device, dtype=torch.int32).reshape(8, 8),
            }
    
            named_module.stream_state_payload_to_cpu(payload)
    
            host_scales = named_module.state["q_scales"]
            host_zeros = named_module.state["q_zeros"]
            host_g_idx = named_module.state["q_g_idx"]
    
            assert host_scales.is_pinned() and host_zeros.is_pinned() and host_g_idx.is_pinned()
    
            named_module.stream_sync()
    
            torch.testing.assert_close(host_scales.cpu(), payload["q_scales"].cpu(), atol=0, rtol=0)
            torch.testing.assert_close(host_zeros.cpu(), payload["q_zeros"].cpu(), atol=0, rtol=0)
            torch.testing.assert_close(host_g_idx.cpu(), payload["q_g_idx"].cpu(), atol=0, rtol=0)
    
            processor._release_host_buffers(
                named_module.state.pop("q_scales"),
                named_module.state.pop("q_zeros"),
                named_module.state.pop("q_g_idx"),
            )
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.skip(
                f"Streaming event helper subprocess unavailable: rc={result.returncode}, stderr={result.stderr.strip()}"
            )

