# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4313: AWQ releases its allocator cache per layer on ROCm.

AWQ's clip search allocates a ~256 MiB scratch buffer per clip-eligible module.
On ROCm the default ``max_split_size_mb:256`` prevents the caching allocator
from reusing a freed 256 MiB block for the next module's slightly smaller
request, so reserved VRAM grew by roughly 1 GiB per decoder layer. ``apply_clip``
is the once-per-layer boundary where those scratch tensors have already been
released, so it must hand the unused cache back to the driver exactly once.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
from torch import nn

from gptqmodel import _alloc_conf
from gptqmodel.looper import awq_processor as awq_processor_module
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.quantization.awq.quantize import scale
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig


pytestmark = [pytest.mark.cpu]

_MIB = 1 << 20


def _module_with_two_linears() -> nn.Sequential:
    module = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    with torch.no_grad():
        module[0].weight.fill_(2.0)
        module[1].weight.fill_(-2.0)
    return module


def _clip_list() -> list[tuple[str, torch.Tensor]]:
    return [("0", torch.ones(8, 1, 1)), ("1", torch.ones(8, 1, 1))]


def test_apply_clip_releases_rocm_cache_once_per_layer(monkeypatch):
    module = _module_with_two_linears()
    calls: list[str] = []

    monkeypatch.setattr(scale, "running_on_rocm", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    # Two clipped modules still belong to one layer, so the release runs once.
    scale.apply_clip(module, _clip_list())

    assert calls == ["empty"]
    # The clamp still ran (the objective must not change).
    assert torch.all(module[0].weight <= 1.0)
    assert torch.all(module[1].weight >= -1.0)


def test_apply_clip_releases_cache_on_resolved_quant_device(monkeypatch):
    module = _module_with_two_linears()
    calls: list[torch.device] = []

    monkeypatch.setattr(scale, "running_on_rocm", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        scale,
        "_resolve_module_exec_device",
        lambda _module: torch.device("cuda", 1),
    )
    # The device is only recorded here; do not require a real GPU to move to it.
    monkeypatch.setattr(scale, "move_to", lambda _module, device=None: None)

    def _record_device(device):
        calls.append(torch.device(device))
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.cuda, "device", _record_device)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    scale.apply_clip(module, _clip_list())

    assert calls == [torch.device("cuda", 1)]


def test_apply_clip_does_not_release_cache_on_cuda(monkeypatch):
    module = _module_with_two_linears()
    calls: list[str] = []

    monkeypatch.setattr(scale, "running_on_rocm", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    scale.apply_clip(module, _clip_list())

    assert calls == []


def test_apply_clip_skips_release_without_cuda(monkeypatch):
    module = _module_with_two_linears()
    calls: list[str] = []

    monkeypatch.setattr(scale, "running_on_rocm", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    scale.apply_clip(module, _clip_list())

    assert calls == []


def test_apply_clip_never_fails_on_allocator_error(monkeypatch):
    module = _module_with_two_linears()

    monkeypatch.setattr(scale, "running_on_rocm", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())

    def _boom():
        raise RuntimeError("allocator is unavailable")

    monkeypatch.setattr(torch.cuda, "empty_cache", _boom)

    # Cleanup is best-effort: a failure must not abort quantization.
    scale.apply_clip(module, _clip_list())
    assert torch.all(module[0].weight <= 1.0)


# ---------------------------------------------------------------------------
# Within-layer MoE growth (#4401 item 1)
#
# ``apply_clip`` releases the allocator cache once per layer (#4313), but a MoE
# layer runs the clip search for every expert before that boundary. When the
# scratch request lands in the last 2 MiB below ``max_split_size_mb`` the
# allocator rounds the segment up to the cap and then refuses to serve the next
# (slightly smaller) request from it, so reserved VRAM grows by one segment per
# expert projection while allocated memory stays flat.
#
# The allocator below is a CPU model of the relevant Torch paths
# (``round_size``, ``get_allocation_size``, ``get_free_block``'s split-cap
# rejection, and ``should_split``). Only the actual ``torch.empty`` sizes come
# from the real clip search; no GPU is required.
# ---------------------------------------------------------------------------

_SMALL_SIZE = 1 * _MIB  # kSmallSize
_SMALL_BUFFER = 2 * _MIB  # kSmallBuffer
_MIN_LARGE_ALLOC = 10 * _MIB  # kMinLargeAlloc
_ROUND_LARGE = 2 * _MIB  # kRoundLarge
_MIN_BLOCK = 512  # kMinBlockSize
_MAX_NON_SPLIT_ROUNDING = 20 * _MIB  # max_non_split_rounding_size default


class _SplitCapAllocator:
    """Caching-allocator reuse model for a configured ``max_split_size``."""

    def __init__(self, max_split: int) -> None:
        self.max_split = max_split
        # Blocks live in separate small/large pools, as in Torch.
        self.free: dict[bool, list[int]] = {True: [], False: []}
        self.live: list[tuple[bool, int]] = []
        self.reserved = 0
        self.requests: list[int] = []
        self.reserved_after_module: list[int] = []

    @staticmethod
    def _round_size(size: int) -> int:
        return max(_MIN_BLOCK, -(-size // _MIN_BLOCK) * _MIN_BLOCK)

    @staticmethod
    def _allocation_size(size: int) -> int:
        if size <= _SMALL_SIZE:
            return _SMALL_BUFFER
        if size < _MIN_LARGE_ALLOC:
            return 20 * _MIB  # large_segment_size default
        return -(-size // _ROUND_LARGE) * _ROUND_LARGE

    def _should_split(self, is_small: bool, block: int, size: int) -> bool:
        remaining = block - size
        if is_small:
            return remaining >= _MIN_BLOCK
        return size < self.max_split and remaining > _SMALL_SIZE

    def _take_free(self, is_small: bool, size: int) -> int | None:
        for block in sorted(self.free[is_small]):
            if block < size:
                continue
            # get_free_block: a sub-cap request may not use an at-or-above-cap
            # block, and a cap-sized request may not round up by >= 20 MiB.
            if size < self.max_split and block >= self.max_split:
                continue
            if (
                size >= self.max_split
                and block >= size + _MAX_NON_SPLIT_ROUNDING
            ):
                continue
            self.free[is_small].remove(block)
            if self._should_split(is_small, block, size):
                self.free[is_small].append(block - size)
                return size
            return block
        return None

    def allocate(self, nbytes: int) -> None:
        size = self._round_size(nbytes)
        is_small = size <= _SMALL_SIZE
        self.requests.append(size)
        block = self._take_free(is_small, size)
        if block is None:
            allocation = self._allocation_size(size)
            self.reserved += allocation
            if self._should_split(is_small, allocation, size):
                self.free[is_small].append(allocation - size)
                block = size
            else:
                block = allocation
        self.live.append((is_small, block))

    def end_module(self) -> None:
        for is_small, block in self.live:
            self.free[is_small].append(block)
        self.live.clear()
        self.reserved_after_module.append(self.reserved)


class _ClipTestAWQProcessor(AWQProcessor):
    def __init__(self, qcfg: QuantizeConfig) -> None:
        super().__init__(
            tokenizer=None,
            qcfg=qcfg,
            calibration=CalibrationConfig(),
            gptq_model=None,
            model=None,
            calculate_w_wq_diff=False,
        )

    def _module_forward(self, x, module, module_kwargs):
        return module(x)


def _run_expert_layer(monkeypatch, *, clamp_enabled: bool, modules: int = 4):
    """Run one MoE layer's clip search, modelling allocator reservations."""

    max_split = 16 * _MIB
    monkeypatch.setattr(
        _alloc_conf, "allocator_max_split_bytes", lambda: max_split
    )
    if not clamp_enabled:
        # Reproduce the pre-fix behaviour: the resolved 256 MiB-class budget is
        # used as-is.
        monkeypatch.setattr(
            awq_processor_module,
            "clamp_budget_below_allocator_split",
            lambda budget: budget,
            raising=False,
        )

    allocator = _SplitCapAllocator(max_split)
    real_empty = torch.empty
    real_empty_like = torch.empty_like

    def fake_empty(*args, **kwargs):
        tensor = real_empty(*args, **kwargs)
        allocator.allocate(tensor.numel() * tensor.element_size())
        return tensor

    def fake_empty_like(source, *args, **kwargs):
        tensor = real_empty_like(source, *args, **kwargs)
        allocator.allocate(tensor.numel() * tensor.element_size())
        return tensor

    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch, "empty_like", fake_empty_like)

    qcfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        bits=4,
        group_size=128,
        sym=True,
        # 16 MiB - 1 so the unclamped product chunk rounds up to the 16 MiB cap.
        mse={"clip_grid_budget": (16 * _MIB) - 1},
    )
    processor = _ClipTestAWQProcessor(qcfg)
    torch.manual_seed(0)
    weight = torch.randn(64, 256)
    input_feat = torch.randn(255, 256)

    results = []
    for _ in range(modules):
        results.append(processor._compute_best_clip(weight, input_feat, max_shrink=0.1))
        allocator.end_module()
    return allocator, results


def test_moe_clip_scratch_grows_within_layer_without_budget_clamp():
    """The unclamped workspace reproduces the reported within-layer growth."""

    with pytest.MonkeyPatch.context() as monkeypatch:
        allocator, _ = _run_expert_layer(monkeypatch, clamp_enabled=False)

    # The large product chunk lands in the last 2 MiB below the cap, so each
    # expert maps a fresh 16 MiB segment.
    assert max(allocator.requests) < 16 * _MIB
    growth = allocator.reserved_after_module[-1] - allocator.reserved_after_module[0]
    assert growth >= 2 * 16 * _MIB


def test_moe_clip_scratch_reserved_flat_within_layer():
    """Clamping below the split cap keeps a MoE layer's reservation flat."""

    with pytest.MonkeyPatch.context() as monkeypatch:
        old_allocator, old_results = _run_expert_layer(
            monkeypatch, clamp_enabled=False
        )
    with pytest.MonkeyPatch.context() as monkeypatch:
        new_allocator, new_results = _run_expert_layer(
            monkeypatch, clamp_enabled=True
        )

    # Every scratch request now fits in a splittable segment.
    assert max(new_allocator.requests) < 16 * _MIB
    assert new_allocator.reserved_after_module[-1] == new_allocator.reserved_after_module[0]
    assert new_allocator.reserved_after_module[-1] < old_allocator.reserved_after_module[-1]

    # Chunking is a scheduling detail: the clip bound must be bitwise identical.
    for old, new in zip(old_results, new_results):
        assert torch.equal(old, new)
