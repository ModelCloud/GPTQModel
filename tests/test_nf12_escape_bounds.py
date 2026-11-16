# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""NF12 escape-index validation (issue #3971).

The packed stream stores an escape-table index per escaped group; it is not
bounded by the checkpoint loader.
"""

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
