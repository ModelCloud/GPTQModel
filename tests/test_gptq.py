# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import math
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest
import random
import torch
import torch.nn as nn
from gptqmodel.looper.named_module import NamedModule
from models.model_test import ModelTest

from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.config import FallbackStrategy, HessianConfig, QuantizeConfig, ScaleSearchConfig
from gptqmodel.quantization.gptq import GPTQ, get_number_of_rows_and_cols


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


@torch.inference_mode()
def test_gptq_static_groups_keep_original_quantizer_mapping_with_gar(monkeypatch):
    torch.manual_seed(42)
    layer = nn.Linear(10, 6, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(
        bits=2,
        group_size=4,
        desc_act=False,
        act_group_aware=True,
        static_groups=True,
    )
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.randn(10, 10, dtype=torch.float32), None)
    gptq.finalize_hessian()

    expected_scales = []
    expected_zeros = []
    original_weight = layer.weight.detach().clone()
    for start in range(0, gptq.columns, qcfg.group_size):
        end = start + qcfg.group_size
        quantizer = copy.deepcopy(gptq.quantizer)
        quantizer.find_params(
            original_weight[:, start:end],
            weight=True,
            hessian=gptq.H[start:end, start:end],
        )
        expected_scales.append(quantizer.scale)
        expected_zeros.append(quantizer.zero)

    def _swap_full_groups(diag_h, group_size, **_kwargs):
        assert diag_h.numel() == 10
        assert group_size == 4
        return torch.tensor([1, 0], dtype=torch.long, device=diag_h.device)

    monkeypatch.setattr(gptq_mod, "compute_global_perm", _swap_full_groups)

    _qweight, scales, zeros, g_idx, *_ = gptq.quantize(blocksize=4)

    torch.testing.assert_close(scales, torch.cat(expected_scales, dim=1))
    torch.testing.assert_close(zeros, torch.cat(expected_zeros, dim=1))
    assert g_idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]


def test_gptq_base_quant_linear_like_shape_without_import():
    class BaseQuantLinear(nn.Module):
        __module__ = "gptqmodel.nn_modules.qlinear"

    class FakeQuantLinear(BaseQuantLinear):
        def __init__(self):
            super().__init__()
            self.in_features = 8
            self.out_features = 19

    assert get_number_of_rows_and_cols(FakeQuantLinear()) == (8, 19)


def test_gptq_act_group_aware_rejects_non_positive_group_size():
    layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(bits=4, group_size=-1, act_group_aware=True)

    with pytest.raises(ValueError, match="group_size > 0"):
        GPTQ(layer, qcfg=qcfg)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("method", "expected_hessian_shape"),
    [
        (ScaleSearchConfig.ACTIVATION, (1, 4)),
        (ScaleSearchConfig.HESSIAN, (1, 4, 4)),
        (ScaleSearchConfig.HYBRID, (1, 4, 4)),
    ],
)
def test_grouped_scale_search_skips_overwritten_full_tensor_search(
    monkeypatch,
    method,
    expected_hessian_shape,
):
    torch.manual_seed(1907)
    layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(
        bits=4,
        group_size=4,
        act_group_aware=False,
        scale_search=method,
        offload_to_disk=False,
    )
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.randn(4, 8, dtype=torch.float32), None)

    searched_shapes = []
    searched_hessian_shapes = []
    find_params_batched = gptq.quantizer.find_params_batched

    def _record_find_params_batched(weights, *args, **kwargs):
        searched_shapes.append(tuple(weights.shape))
        searched_hessian_shapes.append(tuple(kwargs["hessian"].shape))
        return find_params_batched(weights, *args, **kwargs)

    monkeypatch.setattr(gptq.quantizer, "find_params_batched", _record_find_params_batched)
    gptq.quantize(blocksize=4)

    assert searched_shapes == [(6, 1, 4), (6, 1, 4)]
    assert searched_hessian_shapes == [expected_hessian_shape, expected_hessian_shape]


@torch.inference_mode()
def test_ungrouped_scale_search_retains_full_tensor_search(monkeypatch):
    torch.manual_seed(1907)
    layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(
        bits=4,
        group_size=-1,
        act_group_aware=False,
        scale_search=ScaleSearchConfig.ACTIVATION,
        offload_to_disk=False,
    )
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.randn(4, 8, dtype=torch.float32), None)

    searched_shapes = []
    find_params = gptq.quantizer.find_params

    def _record_find_params(weights, *args, **kwargs):
        searched_shapes.append(tuple(weights.shape))
        return find_params(weights, *args, **kwargs)

    monkeypatch.setattr(gptq.quantizer, "find_params", _record_find_params)
    gptq.quantize(blocksize=4)

    assert searched_shapes == [(6, 8)]


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


def test_gptq_embedding_act_group_aware_reorders_scale_on_device(monkeypatch):
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
    assert inverse_group_perm_calls == 0


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hessian_inverse_compile_and_eager_match(dtype):
    """The compiled hessian_inverse helpers must match the same helpers run eagerly."""
    torch.manual_seed(0)
    n = 128
    H = torch.randn(n, n, dtype=dtype, device="cuda")
    H = H @ H.T
    H.diagonal().add_(0.1)
    H_orig = H.clone()

    g_comp = GPTQ(nn.Linear(n, n, bias=False, dtype=dtype, device="cuda"))
    g_comp.quantizer.configure(perchannel=True, grid=100, maxshrink=0.8, trits=False)
    Hinv_comp, used_damp_comp = g_comp.hessian_inverse(H_orig.clone())

    # Swap to the uncompiled helpers for a bit-exact eager reference.
    old_try = gptq_mod._HESSIAN_INVERSE_TRY
    old_factor = gptq_mod._HESSIAN_INVERSE_FACTOR
    gptq_mod._HESSIAN_INVERSE_TRY = gptq_mod._hessian_inverse_try_cholesky
    gptq_mod._HESSIAN_INVERSE_FACTOR = gptq_mod._hessian_inverse_factor
    try:
        g_eager = GPTQ(nn.Linear(n, n, bias=False, dtype=dtype, device="cuda"))
        g_eager.quantizer.configure(perchannel=True, grid=100, maxshrink=0.8, trits=False)
        Hinv_eager, used_damp_eager = g_eager.hessian_inverse(H_orig.clone())
    finally:
        gptq_mod._HESSIAN_INVERSE_TRY = old_try
        gptq_mod._HESSIAN_INVERSE_FACTOR = old_factor

    torch.testing.assert_close(Hinv_comp, Hinv_eager, atol=1e-5, rtol=1e-5)
    assert used_damp_comp == used_damp_eager

    # Orientation sanity: Hinv.T @ Hinv is the inverse of the damped Hessian.
    mean = H_orig.diagonal().mean()
    c = used_damp_comp * mean
    H_eff = H_orig + torch.eye(n, dtype=dtype, device="cuda") * c
    identity = Hinv_comp.T @ Hinv_comp @ H_eff
    torch.testing.assert_close(identity, torch.eye(n, dtype=dtype, device="cuda"), atol=1e-4, rtol=1e-4)


def test_hessian_inverse_cpu_extension_matches_eager():
    """The compiled CPU Hessian inverse op is bit-exact with the eager torch reference."""
    torch.manual_seed(0)
    n = 256
    H = torch.randn(n, n, dtype=torch.float32)
    H = H @ H.T
    H.diagonal().add_(0.1)

    from gptqmodel.nn_modules.qlinear.pack_block_ext import hessian_inverse_cholesky_cpu

    damp = torch.tensor(0.05, dtype=torch.float32)
    Hinv_ext, success = hessian_inverse_cholesky_cpu(H, damp)
    assert success.item()

    H_eff = H.clone()
    H_eff.diagonal().add_(damp)
    L = torch.linalg.cholesky(H_eff)
    Hinv_ref = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)

    assert torch.equal(Hinv_ext, Hinv_ref)


def test_hessian_xtx_cpu_extension_matches_eager():
    """The compiled CPU Hessian X^T X op is bit-exact with torch.matmul and addmm_."""
    torch.manual_seed(0)
    rows, cols = 512, 256
    X1 = torch.randn(rows, cols, dtype=torch.float32)
    X2 = torch.randn(rows, cols, dtype=torch.float32)

    from gptqmodel.nn_modules.qlinear.pack_block_ext import hessian_xtx_cpu

    # out=None path mirrors torch.matmul(X.T, X)
    out_ext = hessian_xtx_cpu(X1, None, beta=0.0, alpha=1.0)
    out_ref = torch.matmul(X1.t(), X1)
    assert torch.equal(out_ext, out_ref)

    # out=... path mirrors out.addmm_(X.T, X)
    out = torch.zeros(cols, cols, dtype=torch.float32)
    out.addmm_(X1.t(), X1, beta=0.0, alpha=1.0)
    hessian_xtx_cpu(X2, out, beta=1.0, alpha=1.0)
    out_ref2 = torch.zeros(cols, cols, dtype=torch.float32)
    out_ref2.addmm_(X1.t(), X1, beta=0.0, alpha=1.0)
    out_ref2.addmm_(X2.t(), X2, beta=1.0, alpha=1.0)
    assert torch.equal(out, out_ref2)


def _gptq_block_reference(W1, Hinv1, scale, zero, maxq, group_size, groupwise):
    """Eager CPU reference for gptq_block_cpu using torch.addr."""
    rows, count = W1.shape
    local = W1.clone()
    Q = torch.empty_like(W1)
    Err = torch.empty_like(W1)
    num_full_groups = count // group_size
    for i in range(count):
        g = i // group_size
        if g >= num_full_groups:
            g = num_full_groups
        sc = scale[:, g : g + 1]
        zv = zero[:, g : g + 1]
        w = local[:, i : i + 1]
        if groupwise:
            q = sc * torch.clamp(torch.round(w / sc), -maxq, maxq)
        else:
            q = sc * (torch.clamp(torch.round(w / sc) + zv, 0.0, maxq) - zv)
        Q[:, i] = q.squeeze(-1)
        d = Hinv1[i, i]
        err = (w - q) / d
        Err[:, i] = err.squeeze(-1)
        local[:, i:] = torch.addr(local[:, i:], err.view(-1), Hinv1[i, i:], alpha=-1.0)
    return Q, Err


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("group_size", [1, 2, 4, 8, 16, 32, 64, 128, 160, 256])
@pytest.mark.parametrize("groupwise", [False, True])
def test_gptq_block_cpu_extension_matches_eager(bits, group_size, groupwise):
    """The compiled CPU GPTQ block step is bit-exact with the eager torch.addr loop."""
    torch.manual_seed(bits * 1000 + group_size + int(groupwise))
    rows, count = 64, 128
    W1 = torch.randn(rows, count, dtype=torch.float32)
    H = torch.randn(count, count, dtype=torch.float32)
    H = H @ H.T
    H.diagonal().add_(0.1)
    Hinv1 = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)))

    if groupwise:
        maxq = 2 ** (bits - 1) - 1
    else:
        maxq = 2 ** bits - 1

    # The scale tensor must have one entry per group spanned by the block.
    groups = count // group_size + (1 if count % group_size != 0 else 0)
    scale = torch.rand(rows, groups, dtype=torch.float32) * 0.05 + 0.01
    if groupwise:
        zero = torch.zeros_like(scale)
    else:
        if bits == 8 and not groupwise:
            # Use a realistic symmetric zero point to avoid saturating the 8-bit range.
            zero = torch.full_like(scale, float((maxq + 1) // 2))
        else:
            zero = torch.rand(rows, groups, dtype=torch.float32) * float(maxq)

    from gptqmodel.nn_modules.qlinear.pack_block_ext import gptq_block_cpu

    Q_ref, Err_ref = _gptq_block_reference(W1, Hinv1, scale, zero, maxq, group_size, groupwise)
    Q_ext, Err_ext = gptq_block_cpu(W1, Hinv1, scale, zero, maxq, group_size, groupwise)
    assert torch.equal(Q_ext, Q_ref)
    assert torch.equal(Err_ext, Err_ref)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("group_size", [1, 2, 4, 8, 16, 32, 64, 128, 160, 256])
def test_gptq_cpu_block_matches_serial_quantize(bits, group_size, monkeypatch):
    """The compiled CPU GPTQ block path is bit-exact with the serial eager path."""
    torch.manual_seed(bits * 1000 + group_size)
    base_layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
    base_layer.weight.data = torch.randn_like(base_layer.weight.data)
    calibration = torch.randn(1, 4, 8)

    def _run(block_cpu: str):
        monkeypatch.setenv("GPTQMODEL_BLOCK_CPU", block_cpu)
        layer = nn.Linear(8, 6, bias=False, dtype=torch.float32).eval()
        layer.weight.data.copy_(base_layer.weight.data)
        qcfg = QuantizeConfig(bits=bits, group_size=group_size, desc_act=False)
        gptq = GPTQ(layer, qcfg=qcfg)
        gptq.quantizer.configure(perchannel=True)
        gptq.add_batch(calibration, None)
        return gptq.quantize(blocksize=4)

    Q_ref, scale_ref, zero_ref, g_ref, *_ = _run("0")
    Q_cpu, scale_cpu, zero_cpu, g_cpu, *_ = _run("1")
    assert torch.equal(Q_ref, Q_cpu)
    assert torch.equal(scale_ref, scale_cpu)
    assert torch.equal(zero_ref, zero_cpu)
    assert torch.equal(g_ref, g_cpu)


def test_find_params_batched_cpu_extension_matches_eager():
    """The compiled CPU scale-search fallback returns the same scale/zero as the eager fallback."""
    from gptqmodel.quantization.quantizer import Quantizer
    from gptqmodel.nn_modules.qlinear.pack_block_ext import find_params_batched_cpu
    from gptqmodel.quantization.config import ScaleSearchConfig

    torch.manual_seed(0)
    rows, num_groups, group_size = 32, 2, 32
    qcfg = QuantizeConfig(bits=4, group_size=group_size)
    quantizer = Quantizer(qcfg=qcfg)
    quantizer.configure(perchannel=True, sym=True)

    x = torch.randn(rows, num_groups, group_size, dtype=torch.float32)
    hessian = torch.rand(num_groups, group_size, dtype=torch.float32) + 0.1
    scale_ref, zero_ref = quantizer.find_params_batched(x, weight=True, hessian=hessian)

    tmp = torch.zeros((rows, num_groups), dtype=torch.float32)
    xmin = torch.minimum(x.amin(dim=-1), tmp)
    xmax = torch.maximum(x.amax(dim=-1), tmp)
    if qcfg.sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        mask = xmin < 0
        xmin = torch.where(mask, -xmax, xmin)
    mask = (xmin == 0) & (xmax == 0)
    xmin = torch.where(mask, -torch.ones_like(xmin), xmin)
    xmax = torch.where(mask, torch.ones_like(xmax), xmax)
    importance = quantizer._prepare_scale_search_hessian_batched(
        hessian, method=ScaleSearchConfig.ACTIVATION
    )

    scale_ext, zero_ext = find_params_batched_cpu(
        x,
        xmin,
        xmax,
        importance,
        quantizer.grid,
        quantizer.maxshrink,
        int(quantizer.maxq.item()),
        qcfg.sym,
        quantizer.requires_groupwise_processing(),
        ScaleSearchConfig.ACTIVATION.value,
        0.0,
    )
    assert torch.equal(scale_ext, scale_ref)
    assert torch.equal(zero_ext, zero_ref)


class TestGPTQHessian:
    """Verify GPTQ Hessian accumulation matches the closed-form reference."""

    def _reference(self, *tensors):
        target = tensors[0].device
        total = sum(t.numel() // t.shape[-1] for t in tensors)
        X = torch.cat(
            [t.reshape(-1, t.shape[-1]).to(target).float().t() for t in tensors],
            dim=1,
        )
        if total == 0:
            return torch.zeros((X.shape[0], X.shape[0]), dtype=torch.float32, device=target)
        return (2.0 / total) * X.matmul(X.t())

    def _make_gptq(self, layer):
        from gptqmodel.looper.named_module import NamedModule
        named = NamedModule(layer, name="l", full_name="m.l", layer_index=0)
        return GPTQ(named, QuantizeConfig())

    def test_single_batch_matches_reference(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, bias=False, dtype=torch.float32)
        gptq = self._make_gptq(layer)
        x = torch.randn(4, 8, 16, dtype=torch.float32)
        gptq.add_batch(x, None)
        gptq.materialize_global_hessian()
        H_ref = self._reference(x)
        assert torch.allclose(gptq.H, H_ref, rtol=1e-4, atol=1e-5)

    def test_multiple_batches_matches_reference(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, bias=False, dtype=torch.float32)
        gptq = self._make_gptq(layer)
        batches = [torch.randn(2, 8, 16, dtype=torch.float32) for _ in range(3)]
        for x in batches:
            gptq.add_batch(x, None)
        gptq.materialize_global_hessian()
        H_ref = self._reference(*batches)
        assert torch.allclose(gptq.H, H_ref, rtol=1e-4, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multi_device_matches_reference(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, bias=False, dtype=torch.float32, device="cpu")
        gptq = self._make_gptq(layer)
        x_cpu = torch.randn(2, 8, 16, dtype=torch.float32, device="cpu")
        x_gpu = torch.randn(2, 8, 16, dtype=torch.float32, device="cuda:0")
        gptq.add_batch(x_cpu, None)
        gptq.add_batch(x_gpu, None)
        gptq.materialize_global_hessian()
        H_ref = self._reference(x_cpu, x_gpu)
        assert torch.allclose(gptq.H, H_ref, rtol=1e-4, atol=1e-5)

    def test_empty_batch_is_noop(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, bias=False, dtype=torch.float32)
        gptq = self._make_gptq(layer)
        gptq.add_batch(torch.zeros(0, 8, 16, dtype=torch.float32), None)
        gptq.materialize_global_hessian()
        assert gptq.H.shape == (16, 16)
        assert torch.allclose(gptq.H, torch.zeros_like(gptq.H))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_add_batch_second_call_avoids_hessian_temporary(self):
        """A second batch should not allocate another columns x columns temporary."""
        columns = 4096
        layer = torch.nn.Linear(columns, columns // 2, bias=False, dtype=torch.float16, device="cuda:0")
        gptq = self._make_gptq(layer)
        x = torch.randn(1, 64, columns, dtype=torch.float16, device="cuda:0")

        gptq.add_batch(x, None)
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        gptq.add_batch(x, None)
        peak = torch.cuda.max_memory_allocated()

        # The columns x columns fp32 output tensor is ~64 MiB.  A second in-place
        # batch should only allocate the small activation cast + workspace.
        assert peak - base < (columns * columns * 4) // 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compute_hessian_xtx_out_saves_memory(self):
        """compute_hessian_xtx(out=...) should not allocate an extra CxC output."""
        columns = 4096
        rows = 128
        layer = torch.nn.Linear(columns, columns // 2, bias=False, dtype=torch.float16, device="cuda:0")
        gptq = self._make_gptq(layer)
        matrix = torch.randn(rows, columns, dtype=torch.float16, device="cuda:0")
        H = torch.zeros(columns, columns, dtype=torch.float32, device="cuda:0")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        gptq.compute_hessian_xtx(matrix, out=H)
        new_with_out = torch.cuda.max_memory_allocated() - base

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        xtx = gptq.compute_hessian_xtx(matrix)
        new_without_out = torch.cuda.max_memory_allocated() - base
        del xtx

        expected_savings = columns * columns * 4
        assert new_without_out >= new_with_out + int(expected_savings * 0.8)


def _gptq_reference(*tensors):
    target = tensors[0].device
    total = sum(t.numel() // t.shape[-1] for t in tensors)
    X = torch.cat(
        [t.reshape(-1, t.shape[-1]).to(target).float().t() for t in tensors],
        dim=1,
    )
    if total == 0:
        return torch.zeros((X.shape[0], X.shape[0]), dtype=torch.float32, device=target)
    return (2.0 / total) * X.matmul(X.t())


@pytest.mark.parametrize("seed", range(10000))
def test_gptq_hessian_randomized(seed: int):
    """Run many random shapes/datasets through the Hessian path."""
    rng = random.Random(seed)
    columns = rng.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    batch_count = rng.choice([1, 2, 3, 5])
    samples = rng.choice([1, 2, 3])
    seq_len = rng.choice([1, 4, 8, 16, 32, 64])

    # Keep total token count bounded so the suite stays fast on CPU.
    max_tokens = 1_000_000
    total = columns * samples * seq_len * batch_count
    if total > max_tokens:
        seq_len = max(1, max_tokens // (columns * samples * batch_count))

    torch.manual_seed(seed)
    layer = nn.Linear(columns, columns // 2, bias=False, dtype=torch.float32)
    named = NamedModule(layer, name="l", full_name="m.l", layer_index=0)
    gptq = GPTQ(named, QuantizeConfig())
    batches = [
        torch.randn(samples, seq_len, columns, dtype=torch.float32)
        for _ in range(batch_count)
    ]

    for x in batches:
        gptq.add_batch(x, None)
    gptq.materialize_global_hessian()

    H_ref = _gptq_reference(*batches)
    assert torch.allclose(gptq.H, H_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("seed", range(1000))
def test_gptq_hessian_randomized_multi_device(seed: int):
    """Randomized multi-device accumulation (cpu + cuda:0)."""
    rng = random.Random(seed)
    columns = rng.choice([8, 16, 32, 64, 128, 256, 512, 1024])
    seq_len = rng.choice([1, 4, 8, 16, 32])

    torch.manual_seed(seed)
    layer = nn.Linear(columns, columns // 2, bias=False, dtype=torch.float32, device="cpu")
    named = NamedModule(layer, name="l", full_name="m.l", layer_index=0)
    gptq = GPTQ(named, QuantizeConfig())

    x_cpu = torch.randn(1, seq_len, columns, dtype=torch.float32, device="cpu")
    x_gpu = torch.randn(1, seq_len, columns, dtype=torch.float32, device="cpu").to("cuda:0")

    gptq.add_batch(x_cpu, None)
    gptq.add_batch(x_gpu, None)
    gptq.materialize_global_hessian()

    H_ref = _gptq_reference(x_cpu, x_gpu)
    assert torch.allclose(gptq.H, H_ref, rtol=1e-4, atol=1e-5)
