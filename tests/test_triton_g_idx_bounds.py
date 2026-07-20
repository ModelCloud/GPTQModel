# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the Triton g_idx bounds validation (see issue #2949).

The Triton dequant kernel indexes scales/qzeros with the checkpoint's g_idx and
performs no upper-bound check, so a crafted g_idx >= num_groups reads out of
bounds on the device. ``_validate_g_idx_bounds`` rejects such checkpoints at
load. These tests exercise the pure validator with small CPU tensors, so they
need neither a GPU nor Triton.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.tritonv2 import _validate_g_idx_bounds


def _scales(num_groups: int, out_features: int = 8) -> torch.Tensor:
    return torch.zeros((num_groups, out_features), dtype=torch.float16)


def test_valid_g_idx_passes():
    num_groups = 2
    g_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    # in-range values must not raise
    _validate_g_idx_bounds(g_idx, _scales(num_groups))


def test_negative_wrap_within_range_passes():
    # The kernel wraps negative entries by + num_groups, so [-num_groups, -1]
    # is still in range after the wrap.
    num_groups = 2
    g_idx = torch.tensor([-1, -2, 0, 1], dtype=torch.int32)
    _validate_g_idx_bounds(g_idx, _scales(num_groups))


def test_g_idx_above_num_groups_rejected():
    num_groups = 2
    g_idx = torch.tensor([0, 0, 1, 6], dtype=torch.int32)  # 6 >= num_groups
    with pytest.raises(ValueError, match="out of range"):
        _validate_g_idx_bounds(g_idx, _scales(num_groups))


def test_g_idx_below_negative_num_groups_rejected():
    num_groups = 2
    g_idx = torch.tensor([0, 0, 1, -3], dtype=torch.int32)  # wraps to -1 -> OOB
    with pytest.raises(ValueError, match="out of range"):
        _validate_g_idx_bounds(g_idx, _scales(num_groups))


def test_none_and_empty_are_noops():
    # Missing buffers or an empty g_idx must not raise.
    _validate_g_idx_bounds(None, _scales(2))
    _validate_g_idx_bounds(torch.tensor([0], dtype=torch.int32), None)
    _validate_g_idx_bounds(torch.empty(0, dtype=torch.int32), _scales(2))
