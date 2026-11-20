# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Chunked Hessian staging must cast directly into its leased workspaces (#5852).

``borrow_materialized_chunk_fp32`` used to allocate a standalone cast tensor for
every chunk (``chunk.to(stage_dtype)``), plus a second one when promoting a
lower-precision staging buffer (``staging_view.to(torch.float32)``).  A warmed
capture therefore allocated up to two full-chunk temporaries per chunk even
though both destination workspaces already exist.  ``copy_`` performs the cast
into the destination, so the same two-step rounding is preserved without the
temporaries.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.config import GPTQConfig, HessianConfig
from gptqmodel.quantization.gptq import GPTQ, release_workspace_cache


ROWS = 1024
COLUMNS = 128
CHUNK = 256


def _task(*, staging_dtype, chunk_size=CHUNK, desc_act=False) -> GPTQ:
    layer = torch.nn.Linear(COLUMNS, 64, bias=False)
    cfg = GPTQConfig(
        bits=4,
        group_size=32,
        device="cpu",
        desc_act=desc_act,
        hessian=HessianConfig(chunk_size=chunk_size, staging_dtype=staging_dtype),
    )
    return GPTQ(layer, qcfg=cfg)


def _activations(dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(9881)
    return torch.randn(ROWS, COLUMNS).to(dtype)


def _two_step_xtx(matrix, *, staging_dtype, chunk_size, cap_scale=None):
    """The pre-fix two-step staging casts, accumulated in fp32 chunk order.

    The GEMM uses the same ``addmm_`` formulation the capture now uses (#5731):
    accumulating in the BLAS epilogue rounds differently from a separate
    ``add_``, so the oracle has to match it for an exact comparison. The
    staging casts above -- the contract this file guards -- are unchanged.
    """

    xtx = torch.zeros((COLUMNS, COLUMNS), dtype=torch.float32)
    for start in range(0, matrix.shape[0], chunk_size):
        chunk = matrix[start : start + chunk_size]
        staged = chunk.to(dtype=staging_dtype).to(torch.float32)
        if cap_scale is not None:
            staged.mul_(cap_scale[start : start + chunk.size(0)].unsqueeze(1))
        xtx.addmm_(staged.T, staged)
    return xtx


@pytest.mark.parametrize(
    "staging_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
def test_warmed_chunk_capture_allocates_no_cast_temporaries(staging_dtype):
    matrix = _activations(torch.float16)
    release_workspace_cache()
    task = _task(staging_dtype=staging_dtype)
    task.compute_hessian_xtx(matrix)  # warm the workspace slots

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as profile:
        task.compute_hessian_xtx(matrix)

    allocations = [
        event.cpu_memory_usage
        for event in profile.events()
        if event.name == "aten::_to_copy" and event.cpu_memory_usage > 0
    ]
    assert allocations == []


@pytest.mark.parametrize(
    "staging_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
def test_direct_copy_keeps_the_staging_rounding(staging_dtype):
    matrix = _activations(torch.float16)
    release_workspace_cache()
    task = _task(staging_dtype=staging_dtype)

    actual = task.compute_hessian_xtx(matrix)
    expected = _two_step_xtx(matrix, staging_dtype=staging_dtype, chunk_size=CHUNK)

    assert torch.equal(actual, expected)


def test_desc_act_row_cap_keeps_the_staging_rounding():
    matrix = _activations(torch.bfloat16)
    release_workspace_cache()
    task = _task(staging_dtype=torch.float32, desc_act=True)
    cap_scale = task.resolve_sink_row_cap_scale(matrix, CHUNK)
    assert cap_scale is not None

    actual = task.compute_hessian_xtx(matrix)
    expected = _two_step_xtx(
        matrix, staging_dtype=torch.float32, chunk_size=CHUNK, cap_scale=cap_scale
    )

    assert torch.equal(actual, expected)


def test_unchunked_capture_still_matches_the_staged_reference():
    matrix = _activations(torch.float16)
    release_workspace_cache()
    task = _task(staging_dtype=torch.float32, chunk_size=None)

    actual = task.compute_hessian_xtx(matrix)
    fp32 = matrix.to(torch.float32)
    expected = torch.matmul(fp32.T, fp32)

    assert torch.equal(actual, expected)
