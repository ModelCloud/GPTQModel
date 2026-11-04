# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``ret_idx`` must be rebuilt when ``g_idx`` is replaced by a checkpoint load.

``TorchFusedLinear`` and ``TorchAtenLinear`` both derive an activation-reorder
index from ``g_idx``. The cache used to be keyed on ``numel()`` alone, so
``load_state_dict`` copying a different ``g_idx`` (a desc_act checkpoint) into
the same buffer silently reused the previous permutation. These tests drive the
real load path and compare against an independent re-derivation of the index.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear


IN_FEATURES = 128
OUT_FEATURES = 64
GROUP_SIZE = 32


def _module(cls):
    module = cls(
        bits=4, group_size=GROUP_SIZE, sym=True, desc_act=True,
        in_features=IN_FEATURES, out_features=OUT_FEATURES, bias=False,
        pack_dtype=torch.int32,
    )
    module.qweight = torch.zeros((IN_FEATURES // 8, OUT_FEATURES), dtype=torch.int32)
    module.qzeros = torch.zeros((IN_FEATURES // GROUP_SIZE, OUT_FEATURES // 8), dtype=torch.int32)
    module.scales = torch.ones((IN_FEATURES // GROUP_SIZE, OUT_FEATURES), dtype=torch.float16)
    return module


def _group_labels(order) -> torch.Tensor:
    return torch.cat([torch.full((GROUP_SIZE,), i, dtype=torch.int32) for i in order])


def _oracle_ret_idx(g_idx: torch.Tensor, group_size: int) -> torch.Tensor:
    """Independent re-derivation of the permutation (flattened group offsets)."""
    total = g_idx.shape[0]
    labels = g_idx.to(torch.int32)
    flat = labels * group_size
    for label in range(int(labels.max()) + 1):
        mask = labels == label
        # Each group label restarts its within-group offset at zero.
        flat[mask] += torch.arange(int(mask.sum()), dtype=torch.int32)
    out = torch.zeros(total, dtype=torch.int32)
    out[flat] = torch.arange(total, dtype=torch.int32)
    return out


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
def test_load_state_dict_with_new_g_idx_rebuilds_ret_idx(cls):
    module = _module(cls)
    module.g_idx = _group_labels((0, 1, 2, 3))
    first = module._build_ret_idx().clone()

    state = {name: value.clone() for name, value in module.state_dict().items()}
    new_g_idx = _group_labels((3, 0, 2, 1))
    state["g_idx"] = new_g_idx
    module.load_state_dict(state)

    assert torch.equal(module.g_idx, new_g_idx)
    rebuilt = module._build_ret_idx()
    expected = _oracle_ret_idx(new_g_idx, GROUP_SIZE)
    assert not torch.equal(rebuilt, first), "the stale permutation was reused"
    assert torch.equal(rebuilt, expected)


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
def test_explicit_clear_weight_cache_drops_ret_idx(cls):
    module = _module(cls)
    module.g_idx = _group_labels((0, 1, 2, 3))
    module._build_ret_idx()

    assert callable(getattr(module, "clear_weight_cache", None))
    module.clear_weight_cache()

    assert module.ret_idx is None
    assert getattr(module, "_ret_idx_identity", None) is None


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
def test_repeated_build_reuses_the_cache(cls):
    module = _module(cls)
    module.g_idx = _group_labels((0, 1, 2, 3))

    first = module._build_ret_idx()
    second = module._build_ret_idx()

    assert first is second


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
def test_in_place_g_idx_mutation_is_observed(cls):
    module = _module(cls)
    module.g_idx = _group_labels((0, 1, 2, 3))
    module._build_ret_idx()

    with torch.no_grad():
        module.g_idx.copy_(_group_labels((3, 0, 2, 1)))

    rebuilt = module._build_ret_idx()
    assert torch.equal(rebuilt, _oracle_ret_idx(module.g_idx, GROUP_SIZE))
