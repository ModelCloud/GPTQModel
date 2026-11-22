# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""NVFP4 accumulator bounds must be computed in 64 bits (#4551).

``_accumulate_fp4_planes`` masks with ``index < M * N``. ``M`` is a runtime i32
and ``N`` a constexpr, so the product wrapped negative once ``rows * N >= 2^31``
and every lane was masked off, leaving ``torch.empty`` output uninitialized.
The full output cannot be allocated on the available GPUs, so the regression
runs one 256-element block with an ``M`` that crosses the int32 product
boundary, and separately inspects the emitted Triton IR for the widened mask.
"""

import re

import pytest
import torch


# The accumulate kernel and its IR come from Triton.
pytest.importorskip("triton")

from gptqmodel.nn_modules.qlinear import w4a_nvfp4_triton
from gptqmodel.utils.rocm import IS_ROCM


pytestmark = pytest.mark.skipif(IS_ROCM, reason="NVFP4 accumulate kernel is CUDA-only")


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device to compile and run the Triton kernel")


def _kernel_args(rows: int, columns: int):
    both = torch.zeros((2 * columns,), device="cuda", dtype=torch.uint8)
    group_scales = torch.ones((columns,), device="cuda", dtype=torch.float32)
    global_scale = torch.ones((), device="cuda", dtype=torch.float32)
    accumulator = torch.zeros((256,), device="cuda", dtype=torch.float32)
    output = torch.full((256,), float("nan"), device="cuda", dtype=torch.float32)
    return both, group_scales, global_scale, accumulator, output


def _launch(rows: int, columns: int):
    both, group_scales, global_scale, accumulator, output = _kernel_args(rows, columns)
    w4a_nvfp4_triton._accumulate_fp4_planes[(1,)](
        both, group_scales, global_scale, None, None, accumulator, output,
        rows, columns, True, True, False, False, 256,
        num_warps=4, enable_fp_fusion=False,
    )
    torch.cuda.synchronize()
    return output


def test_accumulate_mask_stops_at_the_row_limit():
    """The mask still bounds the last partial block by rows * columns."""

    _require_cuda()
    output = _launch(rows=1, columns=4)

    assert not torch.isnan(output[:4]).any()
    assert torch.isnan(output[4:]).all()
    assert torch.equal(output[:4], torch.zeros(4, device="cuda"))


def test_accumulate_mask_covers_indices_past_the_int32_product():
    """rows * N >= 2^31 must still store every index of the first block."""

    _require_cuda()
    columns = 14336
    rows = (2**31) // columns + 1
    assert rows * columns >= 2**31
    assert rows <= 2**31 - 1  # still a valid runtime i32 row count

    output = _launch(rows=rows, columns=columns)

    assert not torch.isnan(output).any(), "bounds mask wrapped negative and stored nothing"
    assert torch.equal(output, torch.zeros(256, device="cuda"))


def test_accumulate_output_matches_torch_oracle():
    """The widened mask must not change the accumulated values it stores."""

    _require_cuda()
    torch.manual_seed(0)
    rows, columns = 3, 4
    total = rows * columns
    both = torch.randn((rows, 2 * columns), device="cuda", dtype=torch.float32)
    group_scales = torch.rand((columns,), device="cuda", dtype=torch.float32) + 0.5
    global_scale = torch.tensor(0.75, device="cuda", dtype=torch.float32)
    bias = torch.randn((columns,), device="cuda", dtype=torch.float32)
    token_scale = torch.rand((rows,), device="cuda", dtype=torch.float32) + 0.5
    accumulator = torch.zeros((256,), device="cuda", dtype=torch.float32)
    output = torch.full((256,), float("nan"), device="cuda", dtype=torch.float32)

    w4a_nvfp4_triton._accumulate_fp4_planes[(1,)](
        both.reshape(-1), group_scales, global_scale, token_scale, bias,
        accumulator, output, rows, columns, True, True, True, True, 256,
        num_warps=4, enable_fp_fusion=False,
    )
    torch.cuda.synchronize()

    index = torch.arange(total, device="cuda")
    plane_offset = (index // columns) * (2 * columns) + index % columns
    low = both.reshape(-1)[plane_offset]
    high = both.reshape(-1)[plane_offset + columns]
    expected = (low + 4.0 * high) * group_scales[index % columns]
    expected = expected * global_scale * token_scale[index // columns] + bias[index % columns]

    assert torch.allclose(output[:total], expected, rtol=1e-6, atol=1e-6)
    assert torch.isnan(output[total:]).all()


def test_accumulate_mask_product_is_widened_in_the_emitted_ir():
    """The Triton IR must multiply the widened row count, not the i32 M."""

    _require_cuda()
    both, group_scales, global_scale, accumulator, output = _kernel_args(rows=4, columns=4)
    compiled = w4a_nvfp4_triton._accumulate_fp4_planes.warmup(
        both, group_scales, global_scale, None, None, accumulator, output,
        4, 4, True, True, False, False, 256,
        grid=(1,), num_warps=4, enable_fp_fusion=False,
    )
    ttir = compiled.asm["ttir"]

    assert "arith.extsi %M : i32 to i64" in ttir
    assert not re.search(r"arith\.muli %M, %c\d+_i32 : i32", ttir), "M * N is still computed in int32"
    assert re.search(r"arith\.muli %\w+, %c\d+_i64 : i64", ttir)
