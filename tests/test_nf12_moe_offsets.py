# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""NF12 MoE escape-offset validation (issue #3971).

Each expert selects its local escape slice with a checkpoint offset pair. The
loader must reject negative, decreasing, or out-of-table offsets.
"""

import pytest
import torch


def _moe_weight(offsets):
    from gptqmodel_nf12.moe import NF12MoEWeight

    return NF12MoEWeight(
        packed=torch.zeros((1, 16, 96), dtype=torch.uint8),
        escapes=torch.zeros((4, 8), dtype=torch.int16),
        escape_offsets=torch.tensor(offsets, dtype=torch.int64),
        exp_base=torch.zeros((1,), dtype=torch.uint8),
    )


def test_nf12_moe_offsets_accept_valid():
    _moe_weight([0, 4])


@pytest.mark.parametrize(
    "offsets",
    [
        [-1, 4],   # negative start
        [2, 1],    # decreasing
        [0, 5],    # past the escape table
    ],
)
def test_nf12_moe_offsets_reject_invalid(offsets):
    with pytest.raises(ValueError, match="escape_offsets"):
        _moe_weight(offsets)
