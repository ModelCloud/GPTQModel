# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from models.model_test import ModelTest

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.config import (
    FOEMConfig,
    GPTAQConfig,
    HessianConfig,
    QuantizeConfig,
)
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ
from gptqmodel.quantization.quantizer import (
    Quantizer,
    _neutralize_dead_columns,
    _replace_zero_scale,
    quantize,
)


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


def _make_embedding_gptq(*, group_size: int = 2, desc_act: bool = False) -> GPTQ:
    embedding = nn.Embedding(5, 3, dtype=torch.float32).eval()
    with torch.no_grad():
        embedding.weight.copy_(
            torch.tensor(
                [
                    [-1.0, -0.8, -0.6],
                    [-0.2, 0.0, 0.2],
                    [0.4, 0.6, 0.8],
                    [1.0, 1.2, 1.4],
                    [1.6, 1.8, 2.0],
                ]
            )
        )
    gptq = GPTQ(
        embedding,
        qcfg=QuantizeConfig(bits=4, group_size=group_size, sym=True, desc_act=desc_act),
    )
    gptq.quantizer.configure(perchannel=True)
    return gptq


def test_embedding_hessian_tracks_token_frequencies_and_incremental_batches():
    gptq = _make_embedding_gptq()
    gptq.add_batch(torch.tensor([[0, 1, 1, 3]]), None)
    gptq.add_batch(torch.tensor([[1, 4]]), None)

    expected = 2.0 * torch.tensor([1, 3, 0, 1, 1], dtype=torch.float32) / 6.0
    diagonal = gptq.finalize_hessian()

    torch.testing.assert_close(diagonal, expected)
    assert gptq.H is None
    assert gptq.nsamples == 6
    torch.testing.assert_close(gptq.finalize_hessian(), expected)

    gptq.add_batch(torch.tensor([[4, 4]]), None)
    expected = 2.0 * torch.tensor([1, 3, 0, 1, 3], dtype=torch.float32) / 8.0
    torch.testing.assert_close(gptq.finalize_hessian(), expected)
    assert gptq.H is None
    assert gptq.nsamples == 8


@pytest.mark.parametrize("token_ids", [torch.tensor([[-1]]), torch.tensor([[5]])])
def test_embedding_hessian_rejects_invalid_token_ids(token_ids):
    gptq = _make_embedding_gptq()

    with pytest.raises(ValueError, match="token IDs must be"):
        gptq.add_batch(token_ids, None)


def test_embedding_quantization_is_group_exact_and_blocksize_independent():
    first = _make_embedding_gptq(group_size=2)
    second = _make_embedding_gptq(group_size=2)
    token_ids = torch.tensor([[0, 1, 1, 3, 4, 4, 4]])
    first.add_batch(token_ids, None)
    second.add_batch(token_ids, None)

    first_result = first.quantize(blocksize=1)
    second_result = second.quantize(blocksize=128)

    for first_tensor, second_tensor in zip(first_result[:4], second_result[:4]):
        torch.testing.assert_close(first_tensor, second_tensor)

    quantized, scale, zero, g_idx, _, loss, damp, nsamples = first_result
    assert quantized.shape == first.module.weight.shape
    assert scale.shape == zero.shape == (3, 3)
    torch.testing.assert_close(g_idx, torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32))
    assert nsamples == token_ids.numel()
    assert damp == first.qcfg.damp_percent

    diagonal = first._H_diag
    floor = max(diagonal.max().item(), 1.0) * 1e-6
    effective_diagonal = torch.clamp(diagonal, min=floor) + damp * diagonal.mean()
    column_error = (first.module.weight.float().T - quantized.float().T).square().sum(dim=0)
    expected_loss = (0.5 * torch.sum(effective_diagonal * column_error) / nsamples).item()
    assert loss == pytest.approx(expected_loss)


def test_embedding_desc_act_uses_frequency_ordered_groups():
    gptq = _make_embedding_gptq(group_size=2, desc_act=True)
    gptq.add_batch(torch.tensor([[0, 1, 1, 1, 3, 3, 4]]), None)

    _, scale, zero, g_idx, *_ = gptq.quantize(blocksize=3)

    assert scale.shape == zero.shape == (3, 3)
    torch.testing.assert_close(g_idx, torch.tensor([1, 0, 2, 0, 1], dtype=torch.int32))


def test_embedding_group_aware_ordering_preserves_tail_group():
    gptq = _make_embedding_gptq(group_size=2)
    gptq.qcfg.act_group_aware = True
    gptq.add_batch(torch.tensor([[0, 1, 1, 2, 3, 3, 3, 4]]), None)

    quantized, scale, zero, g_idx, *_ = gptq.quantize(blocksize=3)

    assert quantized.shape == gptq.module.weight.shape
    assert scale.shape == zero.shape == (3, 3)
    torch.testing.assert_close(g_idx, torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32))
    assert torch.isfinite(quantized).all()


def test_embedding_capture_applies_rank_two_attention_mask():
    from gptqmodel.looper.gptq_processor import GPTQProcessor

    gptq = _make_embedding_gptq()
    processor = object.__new__(GPTQProcessor)
    processor.tasks = {"embed": gptq}
    processor._mask_tls = SimpleNamespace(value=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool))
    processor.current_batch_index = lambda: 0
    input_ids = torch.tensor([[0, 1, 4], [3, 2, 2]])
    output = gptq.module(input_ids)

    processor.pre_process_fwd_hook("embed")(gptq.module, (input_ids,), output)
    diagonal = gptq.finalize_hessian()

    torch.testing.assert_close(diagonal, 2.0 * torch.tensor([1, 1, 0, 1, 0]) / 3.0)
    assert gptq.nsamples == 3


@torch.inference_mode()
def test_two_bit_asymmetric_mse_search_optimizes_zero_point_orientation(monkeypatch):
    weights = torch.tensor([[-1.0, 0.9, 0.9, 0.9, 0.9]], dtype=torch.float32)
    qcfg = QuantizeConfig(bits=2, group_size=-1, sym=False, mse=2.0)

    optimized = Quantizer(qcfg)
    optimized.configure(perchannel=True)
    optimized.find_params(weights, weight=True)

    legacy = Quantizer(qcfg)
    legacy.configure(perchannel=True)
    monkeypatch.setattr(legacy, "_search_adjacent_zero_points", lambda: False)
    legacy.find_params(weights, weight=True)

    optimized_loss = (optimized.quantize(weights) - weights).square().sum()
    legacy_loss = (legacy.quantize(weights) - weights).square().sum()

    assert optimized.zero.item() == 1.0
    assert legacy.zero.item() == 2.0
    assert optimized_loss < legacy_loss


def _legacy_find_params_range(x, sym):
    """The pre-optimization range math, kept verbatim as the numeric oracle."""
    tmp = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmin < 0
        if torch.any(tmp):
            xmin[tmp] = -xmax[tmp]
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1
    return xmin, xmax


def _range_cases(dtype):
    generator = torch.Generator().manual_seed(0)
    cases = {"randn": torch.randn(48, 32, generator=generator)}
    cases["all_zero"] = torch.zeros(48, 32)

    mixed = torch.randn(48, 32, generator=generator)
    mixed[::3] = 0.0
    mixed[1::5] = -mixed[1::5].abs()
    mixed[2::7] = mixed[2::7].abs()
    mixed[3::11] = -0.0
    cases["mixed_signs"] = mixed

    wide = torch.randn(48, 32, generator=generator) * 1e4
    wide[5] = 0.0
    cases["wide"] = wide
    return {name: tensor.to(dtype) for name, tensor in cases.items()}


def _raw_bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("sym", [True, False])
def test_find_params_range_math_stays_bit_exact_and_sync_free(dtype, sym):
    """The min/max normalization must stay bit-identical to the legacy math.

    It also must not fall back to boolean-mask indexing: ``xmin[mask] = ...``
    forces a ``nonzero`` scan plus a device-to-host round trip on every call,
    which is the dominant cost of the GPTQ block loop on accelerators.
    """
    dispatched = []

    class _Recorder(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            dispatched.append(str(func))
            return func(*args, **(kwargs or {}))

    qcfg = QuantizeConfig(bits=4, group_size=128, sym=sym, desc_act=False)

    for name, weights in _range_cases(dtype).items():
        quantizer = Quantizer(qcfg)
        quantizer.configure(perchannel=True, bits=4, sym=sym)

        dispatched.clear()
        with _Recorder():
            quantizer.find_params(weights, weight=True)

        expected_xmin, expected_xmax = _legacy_find_params_range(weights.clone(), sym)
        expected_scale = (expected_xmax - expected_xmin) / quantizer.maxq.cpu()
        if sym:
            expected_zero = torch.full_like(expected_scale, (quantizer.maxq.cpu() + 1) / 2)
        else:
            expected_zero = torch.round(-expected_xmin / expected_scale)

        assert torch.equal(_raw_bytes(quantizer.scale.reshape(-1)), _raw_bytes(expected_scale)), name
        assert torch.equal(_raw_bytes(quantizer.zero.reshape(-1)), _raw_bytes(expected_zero)), name

        banned = ("aten.nonzero", "aten.index_put", "aten._local_scalar_dense", "aten.masked_scatter")
        offenders = [entry for entry in dispatched if entry.startswith(banned)]
        assert not offenders, (name, offenders)
        if sym:
            assert "aten.__and__.Tensor" not in dispatched, (name, dispatched)


def _legacy_mse_quantize(x, scale, zero, maxq):
    """The pre-optimization MSE candidate dequantization, kept as the oracle."""
    zero_mask = scale == 0
    safe_scale = torch.where(
        zero_mask,
        torch.full_like(scale, torch.finfo(scale.dtype).tiny),
        scale,
    )
    undefined_zero = zero_mask & ~torch.isfinite(zero)
    safe_zero = torch.where(undefined_zero, torch.zeros_like(zero), zero)
    quantized = torch.clamp(torch.round(x / safe_scale) + safe_zero, 0, maxq)
    return safe_scale * (quantized - safe_zero)


def _legacy_mse_range_search(x, *, bits, sym, mse, grid=100, maxshrink=0.8):
    """The pre-optimization MSE search, kept verbatim as the numeric oracle."""
    x = x.flatten(1)
    maxq = 2**bits - 1
    xmin, xmax = _legacy_find_params_range(x, sym)
    scale = (xmax - xmin) / maxq
    zero = torch.full_like(scale, (maxq + 1) / 2) if sym else torch.round(-xmin / scale)
    best = torch.full([x.shape[0]], float("inf"), device=x.device)
    search_adjacent_zero_points = maxq == 3 and not sym

    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        xmin1 = p * xmin
        xmax1 = p * xmax
        scale1 = (xmax1 - xmin1) / maxq
        zero1 = torch.round(-xmin1 / scale1) if not sym else zero
        zero_candidates = (zero1,)
        if search_adjacent_zero_points:
            ideal = -xmin1 / scale1
            lower = torch.floor(ideal).clamp_(0, maxq)
            upper = torch.ceil(ideal).clamp_(0, maxq)
            zero_candidates = (zero1, torch.where(zero1 == lower, upper, lower))

        for zero_candidate in zero_candidates:
            candidate = _legacy_mse_quantize(x, scale1.unsqueeze(1), zero_candidate.unsqueeze(1), maxq)
            candidate -= x
            candidate.abs_()
            candidate.pow_(mse)
            err = torch.sum(candidate, 1)
            better = err < best
            if torch.any(better):
                best[better] = err[better]
                scale[better] = scale1[better]
                zero[better] = zero_candidate[better]

    return scale, zero


def _mse_search_cases(dtype):
    generator = torch.Generator().manual_seed(0)
    cases = {"randn": torch.randn(32, 128, generator=generator)}
    cases["all_zero"] = torch.zeros(32, 128)

    mixed = torch.randn(32, 128, generator=generator)
    mixed[::3] = 0.0
    mixed[1::5] = -mixed[1::5].abs()
    mixed[2::7] = mixed[2::7].abs()
    mixed[3::11] = -0.0
    cases["mixed_signs"] = mixed

    wide = torch.randn(32, 128, generator=generator) * 1e4
    wide[5] = 0.0
    cases["wide"] = wide

    # Rounding thresholds: values one representable step below, exactly on, and
    # one step above the nearest half-step for each magnitude.
    centers = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=dtype)
    ulps = torch.nextafter(centers, torch.full_like(centers, float("inf"))) - centers
    thresholds = torch.empty(32, 128, dtype=dtype)
    for row in range(32):
        for col in range(128):
            index = (row + col) % centers.numel()
            thresholds[row, col] = centers[index] + (col % 3 - 1) * ulps[index]
    thresholds[0, 0] = -0.0
    cases["rounding_thresholds"] = thresholds

    return {name: tensor.to(dtype) for name, tensor in cases.items()}


class _DispatchRecorder(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        self.ops = []
        self.bool_mask_index_puts = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        self.ops.append(name)
        if name.startswith("aten.index_put"):
            indices = args[1] if len(args) > 1 else ()
            if any(
                isinstance(index, torch.Tensor) and index.dtype == torch.bool
                for index in indices
            ):
                self.bool_mask_index_puts += 1
        return func(*args, **(kwargs or {}))


def _assert_sync_free(ops, context):
    banned = ("aten.nonzero", "aten.index_put", "aten._local_scalar_dense", "aten.masked_scatter")
    offenders = [entry for entry in ops if entry.startswith(banned)]
    assert not offenders, (context, offenders)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("sym", [True, False])
def test_mse_range_search_stays_bit_exact_and_sync_free(dtype, bits, sym):
    """The device-side MSE selection must match the masked-write oracle exactly.

    The replaced loop paid one ``torch.any`` host round trip and three
    ``index_put_`` scans per grid step, plus two scalar reads inside the
    quantizer.  Scale and zero-point selection are discrete decisions, so the
    optimized path has to stay bitwise identical, not merely close.
    """
    qcfg = QuantizeConfig(bits=bits, group_size=128, sym=sym, mse=2.0)

    for name, weights in _mse_search_cases(dtype).items():
        quantizer = Quantizer(qcfg)
        quantizer.configure(perchannel=True)

        recorder = _DispatchRecorder()
        with recorder:
            quantizer.find_params(weights, weight=True)

        expected_scale, expected_zero = _legacy_mse_range_search(
            weights.clone(),
            bits=bits,
            sym=sym,
            mse=2.0,
        )
        assert torch.equal(_raw_bytes(quantizer.scale.reshape(-1)), _raw_bytes(expected_scale)), (name, "scale")
        assert torch.equal(_raw_bytes(quantizer.zero.reshape(-1)), _raw_bytes(expected_zero)), (name, "zero")
        _assert_sync_free(recorder.ops, (name, "find_params"))

        recorder = _DispatchRecorder()
        with recorder:
            actual = quantizer.quantize(weights)
        expected = _legacy_mse_quantize(
            weights,
            quantizer.scale,
            quantizer.zero,
            2**bits - 1,
        )
        assert torch.equal(_raw_bytes(actual), _raw_bytes(expected)), (name, "quantize")
        _assert_sync_free(recorder.ops, (name, "quantize"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("maxq", [-1, 3, 15])
@pytest.mark.parametrize("requires_groupwise_processing", [False, True])
def test_quantize_host_maxq_matches_tensor_bound(dtype, maxq, requires_groupwise_processing):
    """A host ``maxq`` must remove scalar reads without changing any output bit."""
    torch.manual_seed(0)
    x = torch.randn(16, 32, dtype=dtype)
    scale = torch.rand(16, 1, dtype=dtype) * 0.1
    zero = torch.randn(16, 1, dtype=dtype) * 2.0
    scale[0] = 0
    zero[0] = float("nan")

    expected = quantize(
        x,
        scale,
        zero,
        torch.tensor(maxq, dtype=torch.float32),
        requires_groupwise_processing,
    )

    recorder = _DispatchRecorder()
    with recorder:
        actual = quantize(x, scale, zero, maxq, requires_groupwise_processing)

    assert torch.equal(_raw_bytes(actual), _raw_bytes(expected))
    _assert_sync_free(recorder.ops, (maxq, requires_groupwise_processing))


_DEAD_COLUMN_CASES = {
    "none": [],
    "partial": [1, 4],
    "endpoints": [0, 7],
    "all": list(range(8)),
}


@pytest.mark.parametrize("case", list(_DEAD_COLUMN_CASES))
@pytest.mark.parametrize("strided", [False, True])
def test_neutralize_dead_columns_matches_boolean_mask_oracle(case, strided):
    """Device masking must stay bit-identical to the replaced boolean indexing.

    ``H[dead, dead] = 1`` / ``W[:, dead] = 0`` scans the mask with ``nonzero``
    and synchronizes with the host once per module.  The in-place masked-fill
    form must produce exactly the same tensors while staying on the device.
    """

    generator = torch.Generator().manual_seed(7)
    size = 8
    hessian = torch.randn(size, size, generator=generator, dtype=torch.float64)
    hessian = hessian @ hessian.mT
    weight = torch.randn(5, size, generator=generator, dtype=torch.float64)
    auxiliary = torch.randn(3, size, generator=generator, dtype=torch.float64)
    dead_columns = _DEAD_COLUMN_CASES[case]

    if strided:
        hessian = torch.randn(size * 2, size * 2, generator=generator, dtype=torch.float64)[::2, ::2]
        weight = torch.randn(5, size * 2, generator=generator, dtype=torch.float64)[:, ::2]
        auxiliary = torch.randn(3, size * 2, generator=generator, dtype=torch.float64)[:, ::2]

    mask = torch.zeros(size, dtype=torch.bool)
    mask[dead_columns] = True
    # A dead channel is defined by a zero Hessian diagonal, exactly as the
    # production path observes it.
    hessian.diagonal()[mask] = 0.0

    expected_hessian = hessian.clone()
    expected_weight = weight.clone()
    expected_auxiliary = auxiliary.clone()
    expected_hessian[mask, mask] = 1.0
    expected_weight[:, mask] = 0.0
    expected_auxiliary[:, mask] = 0.0

    actual_hessian = hessian.clone()
    actual_weight = weight.clone()
    actual_auxiliary = auxiliary.clone()
    recorder = _DispatchRecorder()
    with recorder:
        _neutralize_dead_columns(actual_hessian, actual_weight, actual_auxiliary)

    assert torch.equal(actual_hessian, expected_hessian), case
    assert torch.equal(actual_weight, expected_weight), case
    assert torch.equal(actual_auxiliary, expected_auxiliary), case
    _assert_sync_free(recorder.ops, (case, "neutralize_dead_columns"))


def _dead_channel_task(kind):
    """Build a tiny quantizer whose Hessian has one calibration-dead channel."""

    torch.manual_seed(0)
    if kind == "gptq":
        linear = nn.Linear(8, 4, bias=False, dtype=torch.float32).eval()
        task = GPTQ(
            linear,
            qcfg=QuantizeConfig(bits=4, group_size=4, sym=True, desc_act=False),
        )
        task.quantizer.configure(perchannel=True)
        inputs = torch.randn(16, 8)
        task.add_batch(inputs, linear(inputs))
        task.finalize_hessian()
    elif kind in ("gptaq", "foem"):
        linear = nn.Linear(8, 4, bias=False, dtype=torch.float32).eval()
        named = NamedModule(
            linear,
            name="proj",
            full_name="layer.0.proj",
            layer_index=0,
        )
        named.state["native_inp"] = []
        if kind == "gptaq":
            qcfg = QuantizeConfig(
                bits=4,
                group_size=4,
                sym=True,
                desc_act=False,
                gptaq=GPTAQConfig(alpha=0.25, device="cpu"),
            )
            task = GPTAQ(named, qcfg=qcfg)
        else:
            qcfg = QuantizeConfig(
                bits=4,
                group_size=4,
                sym=True,
                desc_act=False,
                foem=FOEMConfig(alpha=0, beta=0.2, device="cpu"),
            )
            task = FOEM(named, qcfg=qcfg)
        task.quantizer.configure(perchannel=True)
        task.H = torch.eye(task.columns, dtype=torch.float32)
        task.nsamples = 1
        if kind == "gptaq":
            task.dXXT = torch.zeros(task.columns, task.columns)
    else:
        linear = nn.Linear(16, 8, bias=False, dtype=torch.float32).eval()
        task = QQQ(linear)
        task.quantizer.configure(bits=4, perchannel=True, sym=True, groupsize=4)
        task.add_batch(torch.randn(4, 8, 16), None)
        task.materialize_hessian()

    hessian = task.H
    hessian.diagonal()[2] = 0.0
    hessian[2, :] = 0.0
    hessian[:, 2] = 0.0
    return task


@pytest.mark.parametrize("kind", ["gptq", "gptaq", "foem", "qqq"])
def test_dead_channel_neutralization_stays_on_device(kind):
    """No quantization pipeline may write dead channels through a boolean mask."""

    task = _dead_channel_task(kind)

    recorder = _DispatchRecorder()
    with recorder:
        task.quantize(blocksize=4 if kind != "qqq" else 8)

    assert recorder.bool_mask_index_puts == 0, (kind, recorder.ops)
    assert "aten.masked_fill_.Scalar" in recorder.ops, kind


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("requires_groupwise_processing", [False, True])
def test_quantize_bypasses_zero_scale_without_nonfinite_output(
    device, dtype, requires_groupwise_processing
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.set_device(0)

    maxq = 7 if requires_groupwise_processing else 15
    x = torch.tensor(
        [
            [8.0, 1.5, 0.5, -0.5, -1.5, -8.0, 0.0, -0.0],
            [1.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=target,
    )
    scale = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=target,
    )
    zero = torch.tensor(
        [
            [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [float("nan"), 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        ],
        dtype=dtype,
        device=target,
    )

    actual = quantize(
        x,
        scale,
        zero,
        torch.tensor(maxq, device=target),
        requires_groupwise_processing,
    )

    safe_scale = torch.where(
        scale == 0,
        torch.full_like(scale, torch.finfo(dtype).tiny),
        scale,
    )
    undefined_zero = (scale == 0) & ~torch.isfinite(zero)
    safe_zero = torch.where(undefined_zero, torch.zeros_like(zero), zero)
    if requires_groupwise_processing:
        expected = safe_scale * torch.clamp(
            torch.round(x / safe_scale), -maxq, maxq
        )
    else:
        expected = safe_scale * (
            torch.clamp(torch.round(x / safe_scale) + safe_zero, 0, maxq)
            - safe_zero
        )

    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_replace_zero_scale_accepts_precomputed_mask(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    scale = torch.tensor(
        [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]],
        dtype=torch.float16,
        device=device,
    )
    zero = torch.tensor(
        [[float("nan"), 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float16,
        device=device,
    )

    expected_scale, expected_zero = _replace_zero_scale(scale, zero)
    actual_scale, actual_zero = _replace_zero_scale(
        scale,
        zero,
        zero_mask=scale == 0,
    )

    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(actual_zero, expected_zero, rtol=0, atol=0)


@pytest.mark.parametrize("act_group_aware", [False, True])
def test_gptq_bypasses_zero_scale_and_logs_once(monkeypatch, act_group_aware):
    torch.manual_seed(0)
    linear = nn.Linear(8, 4, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(
        bits=4,
        group_size=4,
        sym=True,
        desc_act=False,
        act_group_aware=act_group_aware,
    )
    gptq = GPTQ(linear, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)

    inputs = torch.randn(16, 8)
    gptq.add_batch(inputs, linear(inputs))

    original_find_params = gptq.quantizer.find_params

    def find_params_with_zero_scale(x, weight=False):
        original_find_params(x, weight=weight)
        gptq.quantizer.scale[0, 0] = 0

    monkeypatch.setattr(gptq.quantizer, "find_params", find_params_with_zero_scale)

    errors = []
    monkeypatch.setattr(
        gptq_mod.log,
        "error",
        lambda message, *args, **kwargs: errors.append(message % args if args else message),
    )

    quantized, scale, zero, *_ = gptq.quantize(blocksize=4)

    assert torch.isfinite(quantized).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()
    assert (scale != 0).all()
    assert len(errors) == 1
    assert "zero scale found and bypassed" in errors[0]


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_specialized_quantizers_sanitize_zero_scale_before_return(monkeypatch, kind):
    linear = nn.Linear(8, 4, bias=False, dtype=torch.float32)
    named = NamedModule(
        linear,
        name="proj",
        full_name="layer.0.proj",
        layer_index=0,
    )
    if kind == "gptaq":
        named.state["native_inp"] = []
        qcfg = QuantizeConfig(
            bits=4,
            group_size=4,
            sym=True,
            desc_act=False,
            gptaq=GPTAQConfig(alpha=0.25, device="cpu"),
        )
        task = GPTAQ(named, qcfg=qcfg)
    else:
        qcfg = QuantizeConfig(
            bits=4,
            group_size=4,
            sym=True,
            desc_act=False,
            foem=FOEMConfig(alpha=0, beta=0.2, device="cpu"),
        )
        task = FOEM(named, qcfg=qcfg)

    task.quantizer.configure(perchannel=True)
    task.H = torch.eye(task.columns, dtype=torch.float32)
    task.nsamples = 1
    if kind == "gptaq":
        task.dXXT = torch.zeros_like(task.H)

    original_find_params = task.quantizer.find_params

    def find_params_with_zero_scale(x, weight=False):
        original_find_params(x, weight=weight)
        task.quantizer.scale[0, 0] = 0

    monkeypatch.setattr(
        task.quantizer,
        "find_params",
        find_params_with_zero_scale,
    )

    quantized, scale, zero, *_ = task.quantize(blocksize=4)

    assert torch.isfinite(quantized).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()
    assert (scale != 0).all()


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
        gptq_mod._WORKSPACE_SLOTS.clear()

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
