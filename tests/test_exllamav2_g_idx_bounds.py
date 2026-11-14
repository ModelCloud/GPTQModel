# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""ExLlamaV2 act-order g_idx bounds (issue #3971).

``make_sequential`` indexes a host histogram with the checkpoint's ``g_idx``
and never bounds it, so a crafted value writes host heap memory at a chosen
offset. The Triton kernel wraps negative indices by ``num_groups``; ExLlamaV2
indexes directly and must reject them.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear._g_idx import validate_g_idx_bounds


def _scales(num_groups: int, out_features: int = 8) -> torch.Tensor:
    return torch.zeros((num_groups, out_features), dtype=torch.float16)


def test_exllamav2_g_idx_accepts_in_range_values():
    validate_g_idx_bounds(
        torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        _scales(2),
        allow_negative_wrap=False,
    )


@pytest.mark.parametrize("bad", [-1, -2, 2, 99])
def test_exllamav2_g_idx_rejects_out_of_range(bad):
    g_idx = torch.tensor([0, 0, 1, bad], dtype=torch.int32)
    with pytest.raises(ValueError, match="out of range"):
        validate_g_idx_bounds(g_idx, _scales(2), allow_negative_wrap=False)


def test_triton_g_idx_keeps_negative_wrap_contract():
    # The Triton kernel adds num_groups to negative indices, so [-2, -1] is in
    # range there while ExLlamaV2's direct histogram index is not.
    g_idx = torch.tensor([-1, -2, 0, 1], dtype=torch.int32)
    validate_g_idx_bounds(g_idx, _scales(2))
    with pytest.raises(ValueError, match="out of range"):
        validate_g_idx_bounds(g_idx, _scales(2), allow_negative_wrap=False)
