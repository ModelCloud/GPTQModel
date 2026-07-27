# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.utils.marlin import (
    get_scale_perms,
    marlin_permute_bias,
    marlin_permute_scales,
)


@pytest.mark.parametrize("size_k,size_n,group_size", [
    (3072, 3072, 128),
    (3072, 12288, 128),
    (12288, 3072, 128),
    (3072, 1024, 128),
    (1024, 3072, 128),
])
def test_marlin_permute_scales_matches_reference(size_k, size_n, group_size):
    """marlin_permute_scales output must match a simple Python-list reference."""
    scale_perm, scale_perm_single = get_scale_perms()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scales = torch.randn(size_k // group_size, size_n, dtype=torch.float32, device=device)

    out = marlin_permute_scales(scales, size_k, size_n, group_size)

    if group_size < size_k and group_size != -1:
        expected = scales.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        expected = scales.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    expected = expected.reshape((-1, size_n)).contiguous()

    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected)


def test_marlin_permute_bias_matches_reference():
    """marlin_permute_bias output must match a simple Python-list reference."""
    _, scale_perm_single = get_scale_perms()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bias = torch.randn(3072, dtype=torch.float32, device=device)

    out = marlin_permute_bias(bias)
    expected = bias.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    expected = expected.reshape(bias.shape).contiguous()

    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected)
