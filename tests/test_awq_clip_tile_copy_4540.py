# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ clip search must reduce real tile rows, not a reshape copy (issue #4540).

#4474 let clip search process a partial final output tile by cropping the
pre-allocated scratch to that tile's row count.  When the candidate grid is
chunked (``cand_chunk > 1``, the default workspace budget) and
``out_features % 64 != 0``, that crop is non-contiguous, so ``reshape`` returns
a copy.  ``_pseudo_quantize_tensor_into`` wrote the quantized candidates into
the copy, the reduction read stale rows, and every tail row got an arbitrary
clip bound.

These tests pin the tail bounds against the independent per-row oracle with the
default budget, and check that the bounds do not depend on how the candidate
grid is chunked.  A width divisible by the tile size is the control.
"""

import os


# The repro is CPU-only; keep model construction off any visible GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig
from tests.test_awq_clip_consistency import _ClipTestAWQProcessor, _sequential_clip_grid


pytestmark = [pytest.mark.cpu]


@pytest.mark.parametrize(
    "out_features",
    [96, 72, 104, 128],
    ids=["partial-96", "partial-72", "partial-104", "control-128"],
)
def test_partial_tile_matches_per_row_oracle(out_features):
    """Every output row must match the row-by-row oracle under the default budget."""

    generator = torch.Generator().manual_seed(4540)
    weight = torch.randn(out_features, 96, generator=generator)
    input_feat = torch.randn(13, 96, generator=generator)
    # No `mse` override: the default CPU budget (16 MiB) yields cand_chunk ==
    # steps, the chunking that corrupted the tail tile before the fix.
    qcfg = QuantizeConfig(
        quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=32,
    )
    processor = _ClipTestAWQProcessor(qcfg)

    actual = processor._compute_best_clip(weight, input_feat)
    expected = _sequential_clip_grid(qcfg, weight, input_feat)

    # One bound per (output row, group); the trailing singleton is the group axis
    # `apply_clip` reshapes against.
    assert actual.shape == (out_features, 3, 1)
    assert torch.isfinite(actual).all()
    assert (actual >= 0).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "out_features",
    [96, 72, 104],
    ids=["partial-96", "partial-72", "partial-104"],
)
def test_partial_tile_bounds_do_not_depend_on_candidate_chunk(out_features):
    """Chunking the candidate grid must not change a single clip bound."""

    generator = torch.Generator().manual_seed(4540)
    weight = torch.randn(out_features, 96, generator=generator)
    input_feat = torch.randn(13, 96, generator=generator)
    chunked_cfg = QuantizeConfig(
        quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=32,
    )
    single_cfg = QuantizeConfig(
        quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=32,
        mse={"clip_grid_budget": 0},
    )

    chunked = _ClipTestAWQProcessor(chunked_cfg)._compute_best_clip(weight, input_feat)
    single = _ClipTestAWQProcessor(single_cfg)._compute_best_clip(weight, input_feat)

    torch.testing.assert_close(chunked, single, rtol=0, atol=0)
    torch.testing.assert_close(
        chunked, _sequential_clip_grid(chunked_cfg, weight, input_feat), rtol=0, atol=0
    )
