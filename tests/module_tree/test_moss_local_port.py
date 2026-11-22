# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real MOSS shells: every persistent checkpoint owner and neutral AWQ folds."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.moss import MossQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.moss.configuration_moss import MossConfig
from gptqmodel.models.ports.moss.modeling_moss import MossForCausalLM
from tests.test_moss_local_port import _config


RELEASED_KEYS_SHA256 = "97ce41222e2a95a3a038d33506d72b5adad28ee6fadef83e22333ccfd3e000a2"
EXPECTED_PLAN = {"layers": ["transformer.h"], "blocks": [
    ["ln_1:!"], ["attn:!"], ["attn.qkv_proj"], ["attn.out_proj"], ["mlp.fc_in"], ["mlp.fc_out"],
]}


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("heads,width,rotary,inner", [(4, 32, 4, 64), (8, 64, 8, None), (12, 48, 2, 80)])
def test_moss_actual_owners_and_plan(device, heads, width, rotary, inner):
    config = _config(n_head=heads, n_embd=width, rotary_dim=rotary, n_inner=inner)
    with torch.device(device):
        model = MossForCausalLM(config)
    _convert_model_with_defuser(MossQModel, model, cleanup_original=False)
    wrapper = MossQModel.__new__(MossQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    plan = ModulePlan.from_tree(wrapper.module_tree)
    assert MODEL_MAP["moss"] is MossQModel
    linears = {"attn.qkv_proj", "attn.out_proj", "mlp.fc_in", "mlp.fc_out"}
    declared = {p for group in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for p in group}
    assert declared == linears
    for layer in model.transformer.h:
        expected = {"ln_1.weight", "ln_1.bias", "attn.qkv_proj.weight", "attn.out_proj.weight",
                    "attn.causal_mask", "mlp.fc_in.weight", "mlp.fc_in.bias", "mlp.fc_out.weight", "mlp.fc_out.bias"}
        assert set(layer.state_dict()) == expected
        for key in expected:
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert owner in declared and isinstance(layer.get_submodule(owner), nn.Linear)
            else:
                assert set(spec.flags) & {"!", "?"}
        assert layer.attn.qkv_proj.bias is None and layer.attn.out_proj.bias is None
        assert layer.mlp.fc_in.bias is not None and layer.mlp.fc_out.bias is not None
    for key in ["attn_extra", "attn.qkv_proj_extra", "attn.qkv_proj.undeclared", "mlp.fc_input"]:
        assert plan.find(key) is None
    assert wrapper.fusable is None


@pytest.mark.parametrize("rotary,inner", [(4, 64), (8, None)])
def test_moss_shared_norm_folds_once_into_attention_and_mlp(rotary, inner):
    from awq_neutrality import assert_awq_folds_neutral

    from tests.test_moss_local_port import _model

    model = _model(rotary_dim=rotary, n_inner=inner)
    ids = torch.randint(2, 250, (1, 8))

    def forward():
        return model(input_ids=ids).logits

    for layer in model.transformer.h:
        checked = assert_awq_folds_neutral(MossQModel, model, layer, forward)
        # `ln_1` feeds the packed Q/V/K projection and the GELU MLP in parallel:
        # one group, folded once. The packed Q/V/K rows are not `out_proj`'s
        # columns and GELU blocks `fc_in -> fc_out`.
        assert [(source, targets) for source, targets, _consumers in checked] == [
            ("ln_1", ["attn.qkv_proj", "mlp.fc_in"]),
        ]


def test_moss_complete_released_index_ownership():
    # fnlp/moss-moon-003-base@ea9e1340 config; sft/sft-plugin differ only in eos_token_id.
    config = MossConfig(**json.loads((Path(__file__).parents[1] / "fixtures/ports/moss/released_config.json").read_text()))
    with torch.device("meta"):
        model = MossForCausalLM(config)
    # Includes all 34 persistent causal_mask buffers, not just Parameters.
    keys = sorted(model.state_dict())
    assert len(keys) == 311
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == RELEASED_KEYS_SHA256
    assert model.transformer.h[0].attn.qkv_proj.weight.shape == (18432, 6144)
    assert model.transformer.h[0].mlp.fc_in.weight.shape == (24576, 6144)
    assert model.lm_head.weight.shape == (107008, 6144)
    plan = ModulePlan.from_tree(MossQModel.module_tree)
    for key in keys:
        if key.startswith("transformer.h."):
            owner = key.split(".", 3)[-1].rsplit(".", 1)[0]
            assert plan.find(owner) is not None, key
