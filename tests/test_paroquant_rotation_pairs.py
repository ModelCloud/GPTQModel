# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""ParoQuant rotation pair bounds (issue #3971).

The fused CUDA rotation uses the checkpoint's int16 pair values as
shared-memory row indices; valid pairs are ``[0, group_size)``.
"""

import pytest
import torch

from gptqmodel.utils.paroquant import validate_rotation_buffers


def _pairs(values):
    return torch.tensor(values, dtype=torch.int16).reshape(1, -1)


def test_paroquant_pairs_accept_in_range():
    validate_rotation_buffers(
        _pairs([0, 1, 2, 3]),
        torch.zeros((1, 2), dtype=torch.float16),
        torch.ones((1, 4), dtype=torch.float16),
        group_size=4,
        in_features=4,
        krot=1,
    )


@pytest.mark.parametrize("bad", [-1, 4, 300])
def test_paroquant_pairs_reject_out_of_group(bad):
    with pytest.raises(ValueError, match="outside"):
        validate_rotation_buffers(
            _pairs([0, 1, 2, bad]),
            torch.zeros((1, 2), dtype=torch.float16),
            torch.ones((1, 4), dtype=torch.float16),
            group_size=4,
            in_features=4,
            krot=1,
        )


def test_paroquant_pairs_reject_wrong_theta_shape():
    with pytest.raises(ValueError, match="theta"):
        validate_rotation_buffers(
            _pairs([0, 1, 2, 3]),
            torch.zeros((1, 3), dtype=torch.float16),
            torch.ones((1, 4), dtype=torch.float16),
            group_size=4,
            in_features=4,
            krot=1,
        )
