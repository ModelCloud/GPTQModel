# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Group-order ``ret_idx`` must stay bit-identical to the per-group loop."""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear._runtime_cache import build_group_ret_idx
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear


def _reference_ret_idx(g_idx: torch.Tensor, group_size: int) -> torch.Tensor:
    """Independent oracle: the pre-argsort per-group offset accumulation."""

    total = g_idx.shape[0]
    device = g_idx.device
    ret_idx = torch.zeros(total, dtype=torch.int32, device=device)
    group_size = max(int(group_size), 1)
    groups = total // group_size
    remainder = total % group_size
    g_idx = g_idx.to(torch.int32)
    g_idx_2 = g_idx * group_size

    if remainder > 0:
        mask = g_idx == groups
        if mask.any():
            g_idx_2[mask] += torch.arange(remainder, device=device, dtype=torch.int32)

    if groups > 0:
        base = torch.arange(group_size, device=device, dtype=torch.int32)
        for i in range(groups):
            mask = g_idx == i
            if not mask.any():
                continue
            count = int(mask.sum().item())
            g_idx_2[mask] += base[:count]

    ret_idx[g_idx_2] = torch.arange(total, device=device, dtype=torch.int32)
    return ret_idx


def _valid_g_idx(total: int, group_size: int, *, permute: bool, seed: int) -> torch.Tensor:
    """Checkpoint-valid layout: full groups repeated, trailing partial group."""

    g_idx = torch.arange(total, dtype=torch.int32) // group_size
    if permute:
        generator = torch.Generator().manual_seed(seed)
        g_idx = g_idx[torch.randperm(total, generator=generator)]
    return g_idx


SHAPES = [
    (4096, 32),
    (14336, 112),
    (8192, 64),
    (4100, 32),
    (129, 32),
    (33, 32),
    (7, 3),
    (32, 32),
    (1, 1),
]


@pytest.mark.parametrize("total,group_size", SHAPES)
@pytest.mark.parametrize("permute", (False, True))
def test_build_group_ret_idx_matches_reference(total, group_size, permute):
    g_idx = _valid_g_idx(total, group_size, permute=permute, seed=total + group_size)

    expected = _reference_ret_idx(g_idx, group_size)
    actual = build_group_ret_idx(g_idx)

    assert actual.dtype == torch.int32
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    assert torch.equal(
        actual.sort().values,
        torch.arange(total, dtype=torch.int32),
    )


@pytest.mark.parametrize("layer_cls", (TorchFusedLinear, TorchAtenLinear))
def test_layer_ret_idx_matches_reference_and_invalidates(layer_cls):
    total, group_size = 4100, 32
    layer = layer_cls(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=True,
        in_features=total,
        out_features=32,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )

    first = _valid_g_idx(total, group_size, permute=True, seed=1)
    layer.g_idx = first
    actual = layer._build_ret_idx()
    assert torch.equal(actual, _reference_ret_idx(first, group_size))

    # Same tensor: the cached permutation is reused.
    assert layer._build_ret_idx() is actual

    second = _valid_g_idx(total, group_size, permute=True, seed=2)
    layer.g_idx = second
    refreshed = layer._build_ret_idx()
    assert refreshed is not actual
    assert torch.equal(refreshed, _reference_ret_idx(second, group_size))


def test_aten_identity_shortcut_still_matches_reference():
    total, group_size = 4096, 32
    layer = TorchAtenLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=total,
        out_features=32,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    layer.g_idx = _valid_g_idx(total, group_size, permute=False, seed=0)

    assert layer._ret_idx_is_identity()
    assert torch.equal(
        layer._build_ret_idx(),
        torch.arange(total, dtype=torch.int32),
    )
