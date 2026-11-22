# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Real-model module-tree coverage for Decision Transformer (#4755).

The definition wraps a GPT-2 decoder under ``encoder`` and keeps the tiny
``embed_*``/``predict_*`` heads native. This test builds the installed
``DecisionTransformerModel`` on ``meta`` and checks the same plan consumers that
quantization uses: the decoder prefix, ``simple_layer_modules``, every
checkpoint-owned block weight, and the explicitly non-quantized norms/heads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.ports.native.decision_transformer.configuration_decision_transformer import (
    DecisionTransformerConfig,
)
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)
MODEL_TYPE = "decision_transformer"
HEAD_MODULES = (
    "embed_timestep",
    "embed_return",
    "embed_state",
    "embed_action",
    "embed_ln",
    "predict_state",
    "predict_action",
    "predict_return",
)
BLOCK_QUANTIZABLE = {"attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"}
BLOCK_NORMS = {"ln_1", "ln_2"}


@pytest.fixture(scope="module")
def case():
    definition = MODEL_MAP[MODEL_TYPE]
    config = DecisionTransformerConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        hidden_size=32,
        n_inner=64,
        state_dim=8,
        act_dim=3,
        max_length=16,
        max_ep_len=16,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )

    with torch.device("meta"):
        model = definition.loader.from_config(config)

    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())
    specs = tuple(
        name.split(":", 1)[0]
        for block in definition.simple_layer_modules(config, QC)
        for name in block
    )
    wrapper = definition(model=model, quantized=False, quantize_config=None)
    return SimpleNamespace(
        config=config,
        definition=definition,
        model=model,
        wrapper=wrapper,
        layers=layers,
        layer_names=layer_names,
        specs=specs,
        plan=wrapper.module_plan(),
    )


def test_decoder_prefix_and_layer_count(case):
    assert case.definition.extract_layers_node() == ["encoder.h"]
    assert len(case.layers) == case.config.n_layer
    assert case.layer_names == ["encoder.h.0", "encoder.h.1"]
    assert "encoder.wte" in case.definition.get_base_modules(case.model)


def test_quantizable_specs_resolve_on_the_real_model(case):
    assert set(case.specs) == BLOCK_QUANTIZABLE
    for layer in case.layers:
        assert set(find_modules(layer)) == BLOCK_QUANTIZABLE
        for name in BLOCK_QUANTIZABLE:
            spec = case.plan.find(name)
            assert spec is not None and spec.quantize, name


def test_every_checkpoint_weight_owner_is_declared(case):
    for layer in case.layers:
        modules = {name for name, _ in layer.named_modules() if name}
        quantizable = find_modules(layer)
        for weight_path in layer.state_dict():
            owner = max(
                (
                    name
                    for name in modules
                    if weight_path.startswith(f"{name}.")
                    and "." not in weight_path[len(name) + 1 :]
                ),
                key=len,
                default="",
            )
            assert owner, weight_path
            spec = case.plan.find(owner)
            assert spec is not None, (weight_path, owner)
            if spec.quantize:
                assert owner in quantizable, (weight_path, owner)
                assert owner in case.specs, (weight_path, owner)
            else:
                assert {"!", "?"}.intersection(spec.flags), (weight_path, owner, spec)


def test_norms_are_explicitly_not_quantized(case):
    for name in BLOCK_NORMS:
        spec = case.plan.find(name)
        assert spec is not None and not spec.quantize, name
        assert "!" in spec.flags and "norm" in spec.flags, name


def test_heads_stay_native_and_out_of_the_block_plan(case):
    base_modules = set(case.definition.get_base_modules(case.model))
    for name in HEAD_MODULES:
        # The head exists on the checkpoint and is excluded from quantization.
        assert getattr(case.model, name, None) is not None, name
        assert name in base_modules, name
        assert case.plan.find(name) is None, name
        for layer in case.layers:
            assert name not in find_modules(layer), name

    # The heads keep their checkpoint shapes and are not 4-bit groups.
    assert tuple(case.model.embed_state.weight.shape) == (case.config.hidden_size, case.config.state_dim)
    assert tuple(case.model.embed_action.weight.shape) == (case.config.hidden_size, case.config.act_dim)
    assert tuple(case.model.predict_return.weight.shape) == (1, case.config.hidden_size)


def test_undeclared_descendants_are_rejected(case):
    assert case.plan.find("attn.unknown_proj") is None
    assert case.plan.find("mlp.unknown_proj") is None
    assert case.plan.find("ln_1.extra") is None
    assert case.plan.find("embed_state") is None
