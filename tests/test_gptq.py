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
from models.model_test import ModelTest

from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.config import FallbackStrategy, HessianConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CPU fallback regression coverage")
def test_gptq_cpu_hessian_fallback_returns_quantized_weights_to_original_cuda_device(monkeypatch):
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    layer = _make_module(hidden_dim=8, device=device)
    qcfg = QuantizeConfig(bits=4, group_size=2, act_group_aware=True)
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)

    inp = _generate_input(batch_size=1, seq_len=4, hidden_dim=8, device=device)
    gptq.add_batch(inp, None)

    calls = {"cuda": 0, "cpu": 0}

    def _patched_hessian_inverse(self, hessian: torch.Tensor):
        if hessian.device.type == "cuda":
            calls["cuda"] += 1
            raise RuntimeError("CUDA out of memory. simulated for regression test")

        calls["cpu"] += 1
        identity = torch.eye(hessian.shape[0], dtype=torch.float32, device=hessian.device)
        return identity, self.qcfg.damp_percent

    monkeypatch.setattr(GPTQ, "hessian_inverse", _patched_hessian_inverse)
    log_messages = []

    def _capture_warn(message, *args, **kwargs):
        log_messages.append(message % args if args else message)

    def _capture_info(message, *args, **kwargs):
        log_messages.append(message % args if args else message)

    monkeypatch.setattr(gptq_mod.log, "warn", _capture_warn)
    monkeypatch.setattr(gptq_mod.log, "info", _capture_info)

    qweight, _, _, _, *_ = gptq.quantize(blocksize=4)

    assert calls == {"cuda": 1, "cpu": 1}
    assert qweight.device == device
    joined_logs = "\n".join(log_messages)
    assert "falling back to CPU" in joined_logs
    assert "may take much longer than normal" in joined_logs
    assert "moving final quantized weights back" in joined_logs


def test_gptq_act_group_aware_accepts_effective_columns_with_tail_group():
    layer = nn.Linear(10, 6, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(bits=4, group_size=4, act_group_aware=True)

    gptq = GPTQ(layer, qcfg=qcfg)
    assert gptq.columns == 10


def test_gptq_act_group_aware_rejects_non_positive_group_size():
    layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(bits=4, group_size=-1, act_group_aware=True)

    with pytest.raises(ValueError, match="group_size > 0"):
        GPTQ(layer, qcfg=qcfg)


def test_gptq_dense_loss_uses_scalar_accumulator(monkeypatch):
    torch.manual_seed(0)

    layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(bits=4, group_size=4, desc_act=False)
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.randn(1, 4, 8), None)

    full_weight_shape = tuple(layer.weight.shape)
    full_zero_like_calls = 0
    full_empty_like_dtypes = []
    original_zeros_like = torch.zeros_like
    original_empty_like = torch.empty_like

    def _record_zeros_like(input_tensor, *args, **kwargs):
        nonlocal full_zero_like_calls
        if tuple(input_tensor.shape) == full_weight_shape:
            full_zero_like_calls += 1
        return original_zeros_like(input_tensor, *args, **kwargs)

    def _record_empty_like(input_tensor, *args, **kwargs):
        if tuple(input_tensor.shape) == full_weight_shape:
            full_empty_like_dtypes.append(kwargs.get("dtype", input_tensor.dtype))
        return original_empty_like(input_tensor, *args, **kwargs)

    # Dense GPTQ must not zero-initialize any full weight-shaped tensors. The
    # retained output buffer is allocated once with empty_like in final dtype.
    monkeypatch.setattr(torch, "zeros_like", _record_zeros_like)
    monkeypatch.setattr(torch, "empty_like", _record_empty_like)

    qweight, scales, zeros, g_idx, _, avg_loss, _, nsamples = gptq.quantize(blocksize=4)

    assert full_zero_like_calls == 0
    assert full_empty_like_dtypes == [layer.weight.dtype]
    assert qweight.shape == layer.weight.shape
    assert scales.shape == zeros.shape == (6, 2)
    assert g_idx.shape == (8,)
    assert nsamples == 4
    assert math.isfinite(avg_loss)
    assert avg_loss >= 0


def test_gptq_dense_hessian_released_before_output_allocation(monkeypatch):
    torch.manual_seed(0)

    layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(bits=4, group_size=4, desc_act=False)
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.randn(1, 4, 8), None)

    full_weight_shape = tuple(layer.weight.shape)
    hessian_states_at_full_alloc = []
    original_empty_like = torch.empty_like

    def _record_hessian_state(input_tensor, *args, **kwargs):
        if tuple(input_tensor.shape) == full_weight_shape:
            hessian_states_at_full_alloc.append(getattr(gptq, "H", "missing"))
        return original_empty_like(input_tensor, *args, **kwargs)

    # Once Hinv is materialized, the dense Hessian should be gone before GPTQ
    # allocates the full output buffer and block scratch tensors.
    monkeypatch.setattr(torch, "empty_like", _record_hessian_state)

    qweight, *_ = gptq.quantize(blocksize=4)

    assert qweight.shape == layer.weight.shape
    assert hessian_states_at_full_alloc
    assert all(state is None for state in hessian_states_at_full_alloc)


def test_gptq_dense_final_dtype_output_buffer_preserves_quantized_weight(monkeypatch):
    torch.manual_seed(0)

    base_layer = nn.Linear(8, 6, bias=False, dtype=torch.float16).eval()
    calibration = torch.randn(1, 4, 8, dtype=torch.float16)
    full_weight_shape = tuple(base_layer.weight.shape)

    def _run_quantize() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        layer = nn.Linear(8, 6, bias=False, dtype=torch.float16).eval()
        layer.weight.data.copy_(base_layer.weight.data)
        qcfg = QuantizeConfig(bits=4, group_size=4, desc_act=False)
        gptq = GPTQ(layer, qcfg=qcfg)
        gptq.quantizer.configure(perchannel=True)
        gptq.add_batch(calibration, None)
        qweight, scales, zeros, g_idx, _, avg_loss, *_ = gptq.quantize(blocksize=4)
        return qweight, scales, zeros, g_idx, avg_loss

    default_result = _run_quantize()

    original_empty_like = torch.empty_like

    def _force_full_output_buffer_fp32(input_tensor, *args, **kwargs):
        if tuple(input_tensor.shape) == full_weight_shape:
            kwargs["dtype"] = torch.float32
        return original_empty_like(input_tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", _force_full_output_buffer_fp32)
    fp32_buffer_result = _run_quantize()

    for default_tensor, fp32_buffer_tensor in zip(default_result[:4], fp32_buffer_result[:4]):
        assert torch.equal(default_tensor, fp32_buffer_tensor)
    assert default_result[4] == fp32_buffer_result[4]


def test_gptq_dense_block_scratch_buffers_are_fully_written(monkeypatch):
    torch.manual_seed(0)

    base_layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
    calibration = torch.randn(1, 4, 8, dtype=torch.float32)
    block_shape = (base_layer.out_features, 4)

    def _run_quantize() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
        layer.weight.data.copy_(base_layer.weight.data)
        qcfg = QuantizeConfig(bits=4, group_size=4, desc_act=False)
        gptq = GPTQ(layer, qcfg=qcfg)
        gptq.quantizer.configure(perchannel=True)
        gptq.add_batch(calibration, None)
        qweight, scales, zeros, g_idx, _, avg_loss, *_ = gptq.quantize(blocksize=4)
        return qweight, scales, zeros, g_idx, avg_loss

    baseline_result = _run_quantize()
    original_empty_like = torch.empty_like

    def _poison_block_scratch(input_tensor, *args, **kwargs):
        result = original_empty_like(input_tensor, *args, **kwargs)
        if tuple(input_tensor.shape) == block_shape:
            result.fill_(float("nan"))
        return result

    # Block scratch buffers now use empty_like. Poison the allocation so this
    # test fails if Q1 or Err1 ever reads an element before writing it.
    monkeypatch.setattr(torch, "empty_like", _poison_block_scratch)
    poisoned_result = _run_quantize()

    for baseline_tensor, poisoned_tensor in zip(baseline_result[:4], poisoned_result[:4]):
        assert torch.equal(baseline_tensor, poisoned_tensor)
    assert baseline_result[4] == poisoned_result[4]


def test_gptq_hessian_chunk_materialization_direct_copy_preserves_xtx():
    torch.manual_seed(0)

    layer = nn.Linear(8, 6, bias=False, dtype=torch.float16).eval()
    qcfg = QuantizeConfig(hessian=HessianConfig(chunk_size=2, staging_dtype=torch.float32))
    gptq = GPTQ(layer, qcfg=qcfg)
    matrix = torch.randn(5, 8, dtype=torch.float16)

    actual = gptq.compute_hessian_xtx(matrix)
    expected = torch.zeros(8, 8, dtype=torch.float32)
    for start in range(0, matrix.shape[0], 2):
        chunk = matrix[start:start + 2].to(torch.float32)
        expected.add_(chunk.T.matmul(chunk))

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_gptq_hessian_chunk_size_uses_dtype_itemsize_without_tensor_alloc(monkeypatch):
    layer = nn.Linear(8, 6, bias=False, dtype=torch.float16).eval()
    qcfg = QuantizeConfig(hessian=HessianConfig(chunk_bytes=64, staging_dtype=torch.float16))
    gptq = GPTQ(layer, qcfg=qcfg)

    def _reject_tensor_alloc(*args, **kwargs):
        raise AssertionError("chunk sizing should use dtype.itemsize instead of torch.tensor")

    monkeypatch.setattr(torch, "tensor", _reject_tensor_alloc)

    assert gptq.resolve_hessian_chunk_size(rows=16, stage_dtype=torch.float16) == 4


def test_gptq_dense_hessian_ordering_uses_diagonal_view(monkeypatch):
    torch.manual_seed(0)

    layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(bits=4, group_size=4, desc_act=True)
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.randn(1, 4, 8), None)

    original_diag = torch.diag

    def _reject_dense_hessian_diag(input_tensor, *args, **kwargs):
        if isinstance(input_tensor, torch.Tensor) and tuple(input_tensor.shape) == (8, 8):
            raise AssertionError("dense GPTQ should read the Hessian diagonal through Tensor.diagonal()")
        return original_diag(input_tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "diag", _reject_dense_hessian_diag)

    qweight, *_ = gptq.quantize(blocksize=4)

    assert qweight.shape == layer.weight.shape


def test_gptq_group_index_builder_matches_legacy_python_list():
    columns = 17
    group_size = 4
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    direct = GPTQ.build_group_index(columns, group_size, device)
    legacy = torch.tensor(
        [i // group_size for i in range(columns)],
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(direct, legacy, atol=0, rtol=0)
    assert direct.dtype == torch.int32
    assert direct.device == device

    perm = torch.tensor(
        [16, 0, 7, 8, 3, 4, 15, 1, 2, 5, 6, 9, 10, 11, 12, 13, 14],
        dtype=torch.long,
        device=device,
    )
    perm_before = perm.clone()

    perm_direct = GPTQ.build_group_index(
        columns,
        group_size,
        device,
        source_perm=perm,
    )
    perm_legacy = torch.tensor(
        [int(perm_before[i].item()) // group_size for i in range(columns)],
        dtype=torch.int32,
        device=device,
    )

    torch.testing.assert_close(perm_direct, perm_legacy, atol=0, rtol=0)
    torch.testing.assert_close(perm, perm_before, atol=0, rtol=0)


def test_gptq_fallback_quantize_reuses_group_index_builder(monkeypatch):
    torch.manual_seed(0)

    layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(bits=4, group_size=4, desc_act=False)
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)

    calls = []
    original_builder = GPTQ.build_group_index

    def _record_builder(columns, group_size, device, *, source_perm=None):
        calls.append((columns, group_size, torch.device(device), source_perm))
        return original_builder(columns, group_size, device, source_perm=source_perm)

    monkeypatch.setattr(GPTQ, "build_group_index", staticmethod(_record_builder))

    qweight, _scales, _zeros, g_idx, *_ = gptq._fallback_quantize(FallbackStrategy.RTN, blocksize=4)

    assert qweight.shape == layer.weight.shape
    torch.testing.assert_close(
        g_idx,
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=g_idx.device),
        atol=0,
        rtol=0,
    )
    assert calls == [(8, 4, g_idx.device, None)]


def test_gptq_embedding_act_group_aware_reorders_scale_once(monkeypatch):
    torch.manual_seed(0)

    layer = nn.Embedding(8, 6, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(bits=4, group_size=4, desc_act=False, act_group_aware=True)
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.tensor([[0, 1, 2, 3]], dtype=torch.long), None)

    original_tolist = torch.Tensor.tolist
    inverse_group_perm_calls = 0

    def _record_tolist(tensor, *args, **kwargs):
        nonlocal inverse_group_perm_calls
        if tensor.dtype == torch.long and tensor.dim() == 1 and tensor.numel() == 2:
            inverse_group_perm_calls += 1
        return original_tolist(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "tolist", _record_tolist)

    qweight, *_ = gptq.quantize(blocksize=4)

    assert qweight.shape == layer.weight.shape
    assert inverse_group_perm_calls == 1


def test_gptq_embedding_loss_uses_scalar_accumulator(monkeypatch):
    torch.manual_seed(0)

    layer = nn.Embedding(8, 6, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(bits=4, group_size=4, desc_act=False)
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.tensor([[0, 1, 2, 3]], dtype=torch.long), None)

    operating_shape = (layer.embedding_dim, layer.num_embeddings)
    full_zero_like_calls = 0
    full_empty_like_dtypes = []
    original_zeros_like = torch.zeros_like
    original_empty_like = torch.empty_like

    def _record_zeros_like(input_tensor, *args, **kwargs):
        nonlocal full_zero_like_calls
        if tuple(input_tensor.shape) == operating_shape:
            full_zero_like_calls += 1
        return original_zeros_like(input_tensor, *args, **kwargs)

    def _record_empty_like(input_tensor, *args, **kwargs):
        if tuple(input_tensor.shape) == operating_shape:
            full_empty_like_dtypes.append(kwargs.get("dtype", input_tensor.dtype))
        return original_empty_like(input_tensor, *args, **kwargs)

    # Embedding GPTQ works on transposed weights. It must not zero-initialize a
    # full [embedding_dim, vocab] tensor; the output buffer is empty final dtype.
    monkeypatch.setattr(torch, "zeros_like", _record_zeros_like)
    monkeypatch.setattr(torch, "empty_like", _record_empty_like)

    qweight, scales, zeros, g_idx, _, avg_loss, _, nsamples = gptq.quantize(blocksize=4)

    assert full_zero_like_calls == 0
    assert full_empty_like_dtypes == [layer.weight.dtype]
    assert qweight.shape == layer.weight.shape
    assert scales.shape == zeros.shape == (6, 2)
    assert g_idx.shape == (8,)
    assert nsamples == 4
    assert math.isfinite(avg_loss)
    assert avg_loss >= 0


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
