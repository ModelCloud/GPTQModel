# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""NF12 escape-index validation (issue #3971).

The packed stream stores an escape-table index per escaped group; it is not
bounded by the checkpoint loader.
"""

from pathlib import Path

import pytest
import torch


def _nf12_packed(escape_index: int, *, escaped: bool = True) -> torch.Tensor:
    """One 16x64 NF12 super-atom with an optional escaped first group."""

    meta = torch.ones((128, 4), dtype=torch.uint8)
    lo = torch.zeros((128, 8), dtype=torch.uint8)
    if escaped:
        meta[0] = 0
        lo[0, :4] = torch.tensor(
            [escape_index & 0xFF, (escape_index >> 8) & 0xFF,
             (escape_index >> 16) & 0xFF, (escape_index >> 24) & 0xFF],
            dtype=torch.uint8,
        )
    plane0 = meta.reshape(1, 4, 32, 4).permute(0, 2, 1, 3).reshape(512)
    planes12 = lo.reshape(1, 2, 2, 32, 8).permute(0, 1, 3, 2, 4).reshape(1024)
    return torch.cat([plane0, planes12]).reshape(16, 96)


def test_nf12_escape_index_accepts_in_range():
    from gptqmodel_nf12.api import validate_escape_indices

    validate_escape_indices(_nf12_packed(3), torch.zeros((4, 8), dtype=torch.int16))


def test_nf12_escape_index_rejects_out_of_table():
    from gptqmodel_nf12.api import validate_escape_indices

    with pytest.raises(ValueError, match="out of range"):
        validate_escape_indices(_nf12_packed(3), torch.zeros((3, 8), dtype=torch.int16))


def test_nf12_escape_index_rejects_escape_with_empty_table():
    from gptqmodel_nf12.api import validate_escape_indices

    with pytest.raises(ValueError, match="out of range"):
        validate_escape_indices(_nf12_packed(0), torch.zeros((0, 8), dtype=torch.int16))


def test_nf12_escape_index_noop_without_escapes():
    from gptqmodel_nf12.api import validate_escape_indices

    validate_escape_indices(_nf12_packed(0, escaped=False), torch.zeros((0, 8), dtype=torch.int16))


def test_nf12_weight_from_state_dict_validates_escape_indices():
    from gptqmodel_nf12.api import NF12Weight

    state = {
        "nf12_packed": _nf12_packed(5),
        "nf12_escapes": torch.zeros((2, 8), dtype=torch.int16),
        "nf12_exp_base": torch.tensor([0], dtype=torch.uint8),
    }
    with pytest.raises(ValueError, match="out of range"):
        NF12Weight.from_state_dict(state)


SMALL_M_ASYNC = (
    Path(__file__).resolve().parents[1]
    / "gptqmodel_ext/nf12/csrc/linear/small_m_async.cu"
)


def _squashed(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text)


def test_sm103_staged_escape_producer_and_consumer_bound_the_index():
    """The staged path must honour ``num_escapes`` like every other path.

    ``small_m_async_kernel`` copies the escaped rows into shared memory and reads
    them back. Both halves must reject an index at or past the expert's slice
    count; otherwise SM103 alone reads outside the table (issue #4552).
    """

    source = SMALL_M_ASYNC.read_text(encoding="utf-8")
    squashed = _squashed(source)

    producer = _squashed(
        source[
            source.index("} else if (StageEscapes && warp < 4) {") :
            source.index("arrive_async(smem_addr(barriers + stage * Bars + Full));", source.index("StageEscapes && warp < 4"))
        ]
    )
    assert "if (metadata[group] == 0u && indices[group] < p.num_escapes)" in producer
    assert "cp_async_16(target, p.escapes + indices[group]);" in producer

    consumer = _squashed(
        source[
            source.index("if constexpr (StageEscapes && S == WeightSource::kNF12) {") :
            source.index("} else {", source.index("if constexpr (StageEscapes && S == WeightSource::kNF12) {"))
        ]
    )
    assert "if (groups[group].meta == 0u && groups[group].lo0 < p.num_escapes)" in consumer

    # Reference: the unstaged path bounds the same value through the codec.
    assert "T::fragments(planes, p.base, p.escapes, p.num_escapes, fragments);" in squashed
