# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ clip search must reuse one scratch set across full tiles (issue #4682).

#4558 sized the clip-search scratch per tile so the partial final tile kept
contiguous views (#4540).  Each name was rebound only after the new tensor
existed and the previous tile's buffers were still reachable through the
``prod``/``clamp``/``quant``/``cur_out`` views, so every tile after the first
ran with two live workspace sets and every AWQ run paid +52-62% peak memory.

The scratch is now allocated once for a full tile and only the partial final
tile gets its own exact-size set.  These tests pin that allocation shape and
check that sharing the buffers does not leak state between tiles: the bounds
must still equal the independent sequential oracle.
"""

import os


# The repro is CPU-only; keep model construction off any visible GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig
from tests.test_awq_clip_consistency import _ClipTestAWQProcessor, _sequential_clip_grid


pytestmark = [pytest.mark.cpu]


def _clip_shapes(monkeypatch, processor, weight, input_feat):
    """Row counts of every clamp-shaped scratch tensor clip search allocates."""

    rows = []
    real_empty = torch.empty

    def recording_empty(*args, **kwargs):
        tensor = real_empty(*args, **kwargs)
        shape = tuple(tensor.shape)
        # The clamp/quant pair is the only rank-5 workspace with a singleton
        # candidate axis; the product slice uses `token_chunk` there.
        if len(shape) == 5 and shape[2] == 1:
            rows.append(shape[1])
        return tensor

    monkeypatch.setattr(torch, "empty", recording_empty)
    try:
        processor._compute_best_clip(weight, input_feat)
    finally:
        monkeypatch.setattr(torch, "empty", real_empty)
    return rows


def test_full_tiles_share_one_scratch_set(monkeypatch):
    """128 rows at the 64-row tile must allocate one set, not one per tile."""

    generator = torch.Generator().manual_seed(4682)
    weight = torch.randn(128, 96, generator=generator)
    input_feat = torch.randn(13, 96, generator=generator)
    processor = _ClipTestAWQProcessor(
        QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=32)
    )

    assert _clip_shapes(monkeypatch, processor, weight, input_feat) == [64]


def test_partial_tile_adds_exactly_one_exact_size_set(monkeypatch):
    """96 rows at the 64-row tile is one full tile plus one 32-row tile."""

    generator = torch.Generator().manual_seed(4682)
    weight = torch.randn(96, 96, generator=generator)
    input_feat = torch.randn(13, 96, generator=generator)
    processor = _ClipTestAWQProcessor(
        QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=32)
    )

    assert _clip_shapes(monkeypatch, processor, weight, input_feat) == [64, 32]


@pytest.mark.parametrize("out_features", [128, 192], ids=["two-tiles", "three-tiles"])
def test_shared_scratch_matches_sequential_oracle(out_features):
    """Reusing the buffers between tiles must not change a single bound."""

    generator = torch.Generator().manual_seed(4682)
    weight = torch.randn(out_features, 96, generator=generator)
    input_feat = torch.randn(13, 96, generator=generator)
    qcfg = QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=32)
    processor = _ClipTestAWQProcessor(qcfg)

    actual = processor._compute_best_clip(weight, input_feat)
    expected = _sequential_clip_grid(qcfg, weight, input_feat)

    assert actual.shape == (out_features, 3, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
