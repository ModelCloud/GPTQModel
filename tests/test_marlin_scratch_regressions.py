# SPDX-License-Identifier: Apache-2.0
"""Ownership and accounting regressions discovered during scratch review."""
import threading

import pytest
import torch

from gptqmodel.utils import marlin
from gptqmodel.utils.marlin_scratch import (
    MarlinScratchContext,
    active_marlin_scratch_context,
)
from test_marlin_scratch_context import _gemm_args, _patch_ops


def ordered_args(m, dtype=torch.float16, fp32=True):
    return _gemm_args(torch.ones(m, 8, dtype=dtype), fp32=fp32,
                      g_idx=torch.zeros(8, dtype=torch.int32),
                      perm=torch.arange(8, dtype=torch.int32))


def cached_bytes(context):
    return sum(t.numel() * t.element_size() for t in
               (context.c_tmp, context.a_tmp, context.workspace) if t is not None)


def test_allocator_hits_growth_and_smaller_rows(monkeypatch):
    queries, calls = _patch_ops(monkeypatch, lambda a, m, k, *_: (m * 2, m * k))
    inputs = {m: ordered_args(m) for m in (1, 4, 2)}
    original = torch.empty
    allocations = []
    def empty(*args, **kwargs):
        result = original(*args, **kwargs)
        allocations.append(result)
        return result
    monkeypatch.setattr(torch, "empty", empty)
    with MarlinScratchContext("cpu", max_cached_bytes=256) as context:
        marlin.gptq_marlin_gemm(**inputs[1])
        first = (context.c_tmp, context.a_tmp, context.workspace)
        count = len(allocations)
        marlin.gptq_marlin_gemm(**inputs[1])
        assert len(allocations) == count
        assert all(a is b for a, b in zip(first, (context.c_tmp, context.a_tmp, context.workspace)))
        marlin.gptq_marlin_gemm(**inputs[4])
        assert context.c_tmp is not first[0] and context.a_tmp is not first[1]
        assert context.workspace is first[2]
        assert len(allocations) == count + 2
        marlin.gptq_marlin_gemm(**inputs[2])
        assert len(allocations) == count + 2
        assert len(queries) == 3
        assert calls[-1][-2] is context.c_tmp and calls[-1][-1] is context.a_tmp


def test_budget_counts_inactive_retained_capacity(monkeypatch):
    _patch_ops(monkeypatch, lambda a, m, *_: (10, 0) if m == 1 else (0, 40))
    first = _gemm_args(torch.ones(1, 8, dtype=torch.float16))
    second = ordered_args(2, fp32=False)
    with MarlinScratchContext("cpu", max_cached_bytes=100) as context:
        marlin.gptq_marlin_gemm(**first)
        old = context.c_tmp
        marlin.gptq_marlin_gemm(**second)
        assert context.c_tmp is old
        assert context.a_tmp is None
        assert cached_bytes(context) <= 100


def test_oversized_permutation_preserves_reduction_hit(monkeypatch):
    _, calls = _patch_ops(monkeypatch, lambda a, m, *_: (5, 4) if m == 1 else (5, 1000))
    with MarlinScratchContext("cpu", max_cached_bytes=64) as context:
        marlin.gptq_marlin_gemm(**ordered_args(1))
        old = context.c_tmp
        marlin.gptq_marlin_gemm(**ordered_args(2))
        assert calls[-1][-2] is old and calls[-1][-1] is None
        assert cached_bytes(context) <= 64


@pytest.mark.parametrize("budget", [0, 1, 3])
def test_tiny_budget_uses_private_transient_locks(monkeypatch, budget):
    _, calls = _patch_ops(monkeypatch, (5, 10))
    args = ordered_args(1)
    with MarlinScratchContext("cpu", max_cached_bytes=budget) as context:
        marlin.gptq_marlin_gemm(**args)
        assert calls[-1][9] is not args["workspace"]
        assert calls[-1][9].dtype == torch.int32
        assert context.workspace is None
        assert cached_bytes(context) == 0


def test_loader_failure_returns_lease(monkeypatch):
    _patch_ops(monkeypatch, (5, 10))
    good = marlin._marlin_resolve_op
    def fail(*, dtype, op_name):
        if op_name != "marlin_scratch_sizes":
            raise RuntimeError("loader failed")
        return good(dtype=dtype, op_name=op_name)
    with MarlinScratchContext("cpu") as context:
        monkeypatch.setattr(marlin, "_marlin_resolve_op", fail)
        with pytest.raises(RuntimeError, match="loader failed"):
            marlin.gptq_marlin_gemm(**ordered_args(1))
        assert not context._borrowed
        monkeypatch.setattr(marlin, "_marlin_resolve_op", good)
        marlin.gptq_marlin_gemm(**ordered_args(1))


def cuda_controls(monkeypatch):
    state = dict(stream=object(), device=0, capture=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["device"])
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: state["stream"])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state["capture"])
    return state


def test_context_reentry_keeps_stream_affinity(monkeypatch):
    state = cuda_controls(monkeypatch)
    context = MarlinScratchContext("cuda:0")
    with context:
        pass
    state["stream"] = object()
    with pytest.raises(RuntimeError, match="stream"):
        context.__enter__()
    assert active_marlin_scratch_context() is None


def test_failed_exit_resets_active_context(monkeypatch):
    state = cuda_controls(monkeypatch)
    context = MarlinScratchContext("cuda:0")
    context.__enter__()
    state["stream"] = object()
    with pytest.raises(RuntimeError, match="stream"):
        context.__exit__(None, None, None)
    assert active_marlin_scratch_context() is None


def test_capture_started_after_context_entry_is_rejected(monkeypatch):
    state = cuda_controls(monkeypatch)
    context = MarlinScratchContext("cuda:0")
    with context:
        state["capture"] = True
        try:
            with pytest.raises(RuntimeError, match="capture"):
                context.acquire(torch.empty(1, 8), size_m=1, size_k=8,
                                use_fp32_reduce=True, has_act_order=True)
        finally:
            state["capture"] = False
        assert not context._borrowed


def test_other_thread_cannot_clear_inactive_cache(monkeypatch):
    _patch_ops(monkeypatch, (5, 10))
    context = MarlinScratchContext("cpu")
    with context:
        marlin.gptq_marlin_gemm(**ordered_args(1))
    old = context.c_tmp
    errors = []
    def clear():
        try:
            context.clear()
        except RuntimeError as exc:
            errors.append(str(exc))
    worker = threading.Thread(target=clear)
    worker.start()
    worker.join()
    assert errors and "thread" in errors[0]
    assert context.c_tmp is old
    context.clear()


def test_metadata_capacity_is_bounded(monkeypatch):
    _patch_ops(monkeypatch, (5, 10))
    with MarlinScratchContext("cpu") as context:
        for m in range(1, 80):
            marlin.gptq_marlin_gemm(**ordered_args(m))
        assert context.metadata_cache_size <= 64


def test_direct_empty_lease_can_be_released(monkeypatch):
    queries, _ = _patch_ops(monkeypatch, (1000, 1000))
    with MarlinScratchContext("cpu") as context:
        c_tmp, a_tmp, _ = context.acquire(torch.empty(0, 8, dtype=torch.float16),
                                         size_m=0, size_k=8,
                                         use_fp32_reduce=True, has_act_order=True)
        assert c_tmp is None and a_tmp is None and not queries
        context.release()
        context.clear()
