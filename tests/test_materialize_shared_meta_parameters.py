# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Materializing a lazy meta shell keeps parameters shared by identity shared.

A caller can tie ``lm_head.weight`` to the input embedding in session while the
model is still a checkpoint-backed meta shell. When a run exits (for example
because tied-head quantization is rejected), recovery fills every remaining meta
module from the checkpoint one module at a time. Filling each owner from its own
checkpoint key silently split the shared tensor into two independent ones.
"""

from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.utils.offload import materialize_remaining_meta_modules


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4, device="meta")
        self.lm_head = nn.Linear(4, 8, bias=False, device="meta")
        self.norm = nn.LayerNorm(4, device="meta")


def _checkpoint_filler(model):
    """Fill each module's meta parameters with a distinct per-key checkpoint value."""

    values = {}

    def materialize(*, target_submodule, device, module_path):
        assert model.get_submodule(module_path) is target_submodule
        for submodule in target_submodule.modules():
            for attr, param in list(submodule.named_parameters(recurse=False)):
                if not param.is_meta:
                    continue
                fill = float(len(values) + 1)
                fresh = nn.Parameter(torch.full(param.shape, fill, device=device))
                values[(id(submodule), attr)] = fill
                setattr(submodule, attr, fresh)

    return SimpleNamespace(model=model, shell_direct_meta_materialize=materialize)


def test_shared_meta_parameter_stays_shared_after_materialization():
    model = _Tiny()
    model.lm_head.weight = model.embed_tokens.weight

    assert materialize_remaining_meta_modules(_checkpoint_filler(model)) == []

    assert model.lm_head.weight is model.embed_tokens.weight
    assert not model.embed_tokens.weight.is_meta
    assert not model.norm.weight.is_meta


def test_unshared_meta_parameters_materialize_independently():
    model = _Tiny()

    materialize_remaining_meta_modules(_checkpoint_filler(model))

    assert model.lm_head.weight is not model.embed_tokens.weight
    assert not torch.equal(model.lm_head.weight, model.embed_tokens.weight)
