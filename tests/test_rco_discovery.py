# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""RCO must discover the projections a model actually has.

The GSQ paper's Llama configuration stages seven names. A model that adds a
projection (Ripple-65B has a gated attention `self_attn.g_proj`) would otherwise
have those tensors dropped from the allocation, the budget, and the merge -- the
assignment then covers fewer parameters than the checkpoint contains, and the
materializer rejects the plan as incomplete.
"""

import pytest
import torch

from gptqmodel.quantization.rco import (
    DEFAULT_PROJECTIONS,
    block_of,
    discover_projections,
)


def _llama_like(layers=2, extra=()):
    """A decoder whose block is an nn.Module with the paper's 7 projections."""
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(16, 8)
    model.lm_head = torch.nn.Linear(8, 16, bias=False)
    blocks = torch.nn.ModuleList()
    for _ in range(layers):
        block = torch.nn.Module()
        block.self_attn = torch.nn.Module()
        block.mlp = torch.nn.Module()
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(block.self_attn, proj, torch.nn.Linear(8, 8, bias=False))
        for proj in ("gate_proj", "up_proj", "down_proj"):
            setattr(block.mlp, proj, torch.nn.Linear(8, 8, bias=False))
        for name in extra:
            parent, _, leaf = name.rpartition(".")
            holder = getattr(block, parent)
            setattr(holder, leaf, torch.nn.Linear(8, 8, bias=False))
        blocks.append(block)
    model.model.layers = blocks
    return model


def test_discovers_the_paper_configuration():
    model = _llama_like(layers=2)
    found = discover_projections(model)
    assert len(found) == 14
    assert {name.rsplit(".", 1)[-1] for name in found} == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }


def test_discovers_an_extra_projection_the_hardcoded_tuple_misses():
    """The regression: Ripple's gated attention adds `self_attn.g_proj`."""
    model = _llama_like(layers=3, extra=("self_attn.g_proj",))
    found = discover_projections(model)

    assert len(found) == 24  # 3 blocks x 8 projections
    gated = [name for name in found if name.endswith("self_attn.g_proj")]
    assert len(gated) == 3
    # The hardcoded tuple cannot express these, which is why they used to vanish.
    assert not any(name.endswith("g_proj") for name in DEFAULT_PROJECTIONS)


def test_never_returns_the_vocabulary_projections():
    model = _llama_like(layers=1)
    found = discover_projections(model)
    assert not any(name.endswith(("lm_head", "embed_tokens")) for name in found)


def test_only_linear_modules_under_an_indexed_block():
    model = _llama_like(layers=2)
    # A norm is not a projection, and a top-level linear is not block-indexed.
    model.model.layers[0].input_layernorm = torch.nn.LayerNorm(8)
    model.proj = torch.nn.Linear(8, 8, bias=False)
    found = discover_projections(model)
    assert not any("input_layernorm" in name for name in found)
    assert "proj" not in found
    assert len(found) == 14


def test_suffix_filter_and_exclusion():
    model = _llama_like(layers=2, extra=("self_attn.g_proj",))

    attention_only = discover_projections(model, suffixes=["q_proj", "k_proj", "v_proj", "o_proj", "g_proj"])
    assert {name.rsplit(".", 1)[-1] for name in attention_only} == {
        "q_proj", "k_proj", "v_proj", "o_proj", "g_proj"
    }

    excluded = {name for name in discover_projections(model) if name.endswith("g_proj")}
    without_gated = discover_projections(model, exclude=excluded)
    assert len(without_gated) == len(discover_projections(model)) - len(excluded)


def test_result_is_ordered_by_block_then_name():
    model = _llama_like(layers=3)
    found = discover_projections(model)
    assert [block_of(name) for name in found] == sorted(block_of(name) for name in found)
    # Blocks must not interleave.
    seen = []
    for name in found:
        if not seen or seen[-1] != block_of(name):
            seen.append(block_of(name))
    assert seen == [0, 1, 2]


@pytest.mark.parametrize("layers", [1, 4])
def test_block_of_extracts_the_index(layers):
    model = _llama_like(layers=layers)
    found = discover_projections(model)
    assert {block_of(name) for name in found} == set(range(layers))
