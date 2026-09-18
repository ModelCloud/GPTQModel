# SPDX-License-Identifier: Apache-2.0
"""CPU-side lifecycle and dispatch tests for Marlin caller-owned scratch."""

from __future__ import annotations

import threading

import pytest
import torch

from gptqmodel.utils import marlin
from gptqmodel.utils.marlin_scalar_type import scalar_types


def _gemm_args(a: torch.Tensor, *, g_idx=None, perm=None, fp32=True):
    return dict(
        a=a,
        c=None,
        b_q_weight=torch.zeros((1, 1), dtype=torch.int32),
        b_bias=None,
        b_scales=torch.ones((1, 1), dtype=a.dtype),
        global_scale=None,
        b_zeros=None,
        g_idx=g_idx,
        perm=perm,
        workspace=torch.zeros(1, dtype=torch.int32),
        b_q_type=scalar_types.uint4b8,
        size_m=a.shape[0],
        size_n=1,
        size_k=a.shape[1],
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=fp32,
        is_zp_float=False,
    )


def _patch_ops(monkeypatch, sizes):
    queries = []
    calls = []

    def query(a, size_m, size_k, use_fp32_reduce, has_act_order):
        queries.append((tuple(a.shape), size_m, size_k, use_fp32_reduce, has_act_order))
        if callable(sizes):
            return sizes(a, size_m, size_k, use_fp32_reduce, has_act_order)
        return sizes

    def gemm(*args):
        calls.append(args)
        return torch.zeros((args[11], args[12]), dtype=args[0].dtype)

    def resolve(*, dtype, op_name):
        del dtype
        return query if op_name == "marlin_scratch_sizes" else gemm

    monkeypatch.setattr(marlin, "_marlin_resolve_op", resolve)
    return queries, calls


def test_default_dispatch_keeps_legacy_arity_and_explicit_scratch_appends(monkeypatch):
    _, calls = _patch_ops(monkeypatch, (3, 4))
    args = _gemm_args(torch.ones((2, 8), dtype=torch.float16), fp32=False)

    marlin.gptq_marlin_gemm(**args)
    assert len(calls[-1]) == 18

    c_tmp = torch.empty(3, dtype=torch.float32)
    marlin.gptq_marlin_gemm(**args, c_tmp=c_tmp)
    assert len(calls[-1]) == 20
    assert calls[-1][-2] is c_tmp
    assert calls[-1][-1] is None


def test_context_cache_hit_uses_padded_shape_and_owns_workspace(monkeypatch):
    queries, calls = _patch_ops(monkeypatch, (5, 7))
    args = _gemm_args(torch.ones((2, 16), dtype=torch.float16), fp32=True)
    module_workspace = args["workspace"]

    with marlin.MarlinScratchContext("cpu", max_cached_bytes=64) as context:
        # size_k deliberately differs from the actual post-padding A shape.
        args["size_k"] = 8
        marlin.gptq_marlin_gemm(**args)
        marlin.gptq_marlin_gemm(**args)
        assert len(queries) == 1
        assert queries[0] == ((2, 16), 2, 16, True, False)
        assert len(calls[-1]) == 20
        assert calls[-1][9] is context.workspace
        assert calls[-1][9] is not module_workspace
        assert calls[-1][-2] is context.c_tmp
        assert calls[-1][-1] is None


def test_context_disabled_flags_skips_temp_buffers_but_keeps_owned_locks(monkeypatch):
    queries, calls = _patch_ops(monkeypatch, (500, 700))
    args = _gemm_args(torch.ones((2, 8), dtype=torch.float16), fp32=False)
    args["g_idx"] = torch.empty(0, dtype=torch.int32)
    args["perm"] = torch.empty(0, dtype=torch.int32)

    with marlin.MarlinScratchContext("cpu", max_cached_bytes=64) as context:
        marlin.gptq_marlin_gemm(**args)
        assert len(queries) == 1
        assert len(calls[-1]) == 18
        assert calls[-1][9] is context.workspace
        assert context.c_tmp is None
        assert context.a_tmp is None


def test_context_budget_falls_back_and_retains_bounded_cache(monkeypatch):
    def sizes(a, *_):
        return (2, 2) if a.shape[0] == 1 else (100, 100)

    queries, calls = _patch_ops(monkeypatch, sizes)
    first = _gemm_args(torch.ones((1, 8), dtype=torch.float16))
    second = _gemm_args(torch.ones((2, 8), dtype=torch.float16))

    # Lock workspace is four bytes in CPU test mode; the first request costs
    # 12 bytes, while the larger request cannot fit in the retained budget.
    with marlin.MarlinScratchContext("cpu", max_cached_bytes=16) as context:
        marlin.gptq_marlin_gemm(**first)
        old_c, old_a = context.c_tmp, context.a_tmp
        marlin.gptq_marlin_gemm(**second)
        assert len(queries) == 2
        assert context.c_tmp is old_c
        assert context.a_tmp is old_a
        assert len(calls[-1]) == 18


def test_context_clear_releases_cache_and_requeries(monkeypatch):
    queries, _ = _patch_ops(monkeypatch, (2, 2))
    args = _gemm_args(torch.ones((1, 8), dtype=torch.float16))
    context = marlin.MarlinScratchContext("cpu", max_cached_bytes=32)
    with context:
        marlin.gptq_marlin_gemm(**args)
        assert context.c_tmp is not None
        context.clear()
        assert context.c_tmp is None
        assert context.a_tmp is None
        assert context.workspace is None
        marlin.gptq_marlin_gemm(**args)
    context.clear()
    assert len(queries) == 2


def test_context_dtype_change_replaces_both_temporaries(monkeypatch):
    _patch_ops(monkeypatch, (2, 2))
    context = marlin.MarlinScratchContext("cpu", max_cached_bytes=32)
    with context:
        first = _gemm_args(
            torch.ones((1, 8), dtype=torch.float16),
            g_idx=torch.ones(8, dtype=torch.int32),
            perm=torch.ones(8, dtype=torch.int32),
        )
        marlin.gptq_marlin_gemm(**first)
        old_c, old_a = context.c_tmp, context.a_tmp
        second = _gemm_args(
            torch.ones((1, 8), dtype=torch.bfloat16),
            g_idx=torch.ones(8, dtype=torch.int32),
            perm=torch.ones(8, dtype=torch.int32),
        )
        marlin.gptq_marlin_gemm(**second)
        assert context.c_tmp is not old_c
        assert context.a_tmp is not old_a
        assert context.a_tmp.dtype == torch.bfloat16


def test_context_releases_borrow_on_backend_exception(monkeypatch):
    queries, calls = _patch_ops(monkeypatch, (2, 2))
    args = _gemm_args(torch.ones((1, 8), dtype=torch.float16))
    original = marlin._marlin_resolve_op
    failed = True

    def resolver(*, dtype, op_name):
        op = original(dtype=dtype, op_name=op_name)
        if op_name == "gptq_marlin_gemm_fp16" and failed:
            def raising(*_args):
                raise RuntimeError("backend failure")
            return raising
        return op

    # Keep the fake query and swap only the GEMM operation.
    def fake_query(*query_args):
        queries.append(query_args)
        return 2, 2
    monkeypatch.setattr(
        marlin,
        "_marlin_resolve_op",
        lambda *, dtype, op_name: fake_query if op_name == "marlin_scratch_sizes" else resolver(dtype=dtype, op_name=op_name),
    )
    context = marlin.MarlinScratchContext("cpu", max_cached_bytes=32)
    with context:
        with pytest.raises(RuntimeError, match="backend failure"):
            marlin.gptq_marlin_gemm(**args)
        assert not context._borrowed
        failed = False
        marlin.gptq_marlin_gemm(**args)
    assert len(calls) == 1


def test_context_rejects_mixing_explicit_scratch(monkeypatch):
    _patch_ops(monkeypatch, (2, 2))
    args = _gemm_args(torch.ones((1, 8), dtype=torch.float16))
    with marlin.MarlinScratchContext("cpu"):
        with pytest.raises(ValueError, match="mixed"):
            marlin.gptq_marlin_gemm(**args, c_tmp=torch.empty(2, dtype=torch.float32))


def test_m0_does_not_query_or_allocate_manager_scratch(monkeypatch):
    queries, calls = _patch_ops(monkeypatch, (500, 700))
    args = _gemm_args(torch.empty((0, 8), dtype=torch.float16))
    module_workspace = args["workspace"]
    with marlin.MarlinScratchContext("cpu", max_cached_bytes=0) as context:
        marlin.gptq_marlin_gemm(**args)
        assert queries == []
        assert len(calls[-1]) == 18
        assert calls[-1][9] is module_workspace
        assert context.workspace is None


def test_context_nested_and_reused_entries_are_controlled(monkeypatch):
    _patch_ops(monkeypatch, (1, 1))
    context = marlin.MarlinScratchContext("cpu")
    with context:
        with pytest.raises(RuntimeError, match="nested|re-entered"):
            with context:
                pass
    with context:
        pass


def test_context_rejects_other_thread_and_concurrent_borrow(monkeypatch):
    _patch_ops(monkeypatch, (1, 1))
    context = marlin.MarlinScratchContext("cpu")
    errors = []
    with context:
        worker = threading.Thread(
            target=lambda: errors.append(_capture_error(context._check_owner))
        )
        worker.start()
        worker.join()
        assert isinstance(errors[0], RuntimeError)
        args = _gemm_args(torch.ones((1, 8), dtype=torch.float16))
        context.acquire(args["a"], size_m=1, size_k=8, use_fp32_reduce=True, has_act_order=False)
        with pytest.raises(RuntimeError, match="borrowed"):
            context.acquire(args["a"], size_m=1, size_k=8, use_fp32_reduce=True, has_act_order=False)
        context.release()


def _capture_error(fn):
    try:
        fn()
    except BaseException as exc:  # test helper
        return exc
    return None


def test_cuda_affinity_and_capture_checks_are_eager(monkeypatch):
    stream = object()
    current = {"device": 0, "stream": stream, "capture": False}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["device"])
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: current["stream"])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: current["capture"])
    context = marlin.MarlinScratchContext("cuda:0")
    context.__enter__()
    try:
        current["capture"] = True
        with pytest.raises(RuntimeError, match="capture"):
            context._check_owner()
        current["capture"] = False
        current["stream"] = object()
        with pytest.raises(RuntimeError, match="stream"):
            context._check_owner()
    finally:
        # Exit itself validates ownership, so restore the original stream.
        current["stream"] = stream
        context.__exit__(None, None, None)
