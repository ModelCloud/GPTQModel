# SPDX-License-Identifier: Apache-2.0
"""Real LLaDA2 packed-QKV/dense/shared/routed checkpoint weight ownership."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.llada2 import LLaDA2MoeQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.llada2.modeling_llada2_moe import LLaDA2MoeModelLM
from tests.test_llada2_local_port import VERSIONS, tiny_config


def test_exact_central_plan_proposal_and_no_fusion_admission():
    from gptqmodel.models.fusable import require_fusable_config

    proposal = json.loads(
        (Path(__file__).parents[2] / "gptqmodel/models/ports/llada2/module_plan_cases.json").read_text()
    )
    expected = proposal["registered_module_plans"]["cases"]["LLaDA2MoeQModel/gptq"]
    assert LLaDA2MoeQModel.extract_layers_node() == expected["layers"]
    assert (
        LLaDA2MoeQModel.build_layer_modules(LLaDA2MoeQModel.module_tree, include_capture_only=True)
        == expected["blocks"]
    )
    with pytest.raises(ValueError):
        require_fusable_config(LLaDA2MoeQModel, "llada2_moe")


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("kv,bias,shared,dense", [(4, False, None, 0), (2, True, 1, 1), (1, False, 2, 2)])
def test_real_checkpoint_owned_modules_and_aliases(version, kv, bias, shared, dense):
    config = tiny_config(
        version,
        num_key_value_heads=kv,
        use_qkv_bias=bias,
        use_bias=bias,
        num_shared_experts=shared,
        first_k_dense_replace=dense,
    )
    with torch.device("meta"):
        model = LLaDA2MoeModelLM(config)
    _convert_model_with_defuser(LLaDA2MoeQModel, model, cleanup_original=False)
    wrapper = LLaDA2MoeQModel.__new__(LLaDA2MoeQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    plan = ModulePlan.from_tree(wrapper.module_tree)
    declared = {
        name for group in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in group
    }
    for layer in model.model.layers:
        for name, tensor in layer.state_dict().items():
            owner = name.rsplit(".", 1)[0] if name.rsplit(".", 1)[-1] in ("weight", "bias") else name
            spec = plan.find(owner)
            assert spec is not None, name
            if spec.quantize:
                assert owner in declared, name
                assert isinstance(layer.get_submodule(owner), nn.Linear)
            else:
                assert "!" in spec.flags
            assert tensor.is_meta
        if hasattr(layer.mlp, "gate"):
            assert not plan.find("mlp.gate").quantize
            assert not plan.find("mlp.gate.expert_bias").quantize
        assert "0" in plan.find("attention.query_key_value").flags
        assert "1" in plan.find("attention.dense").flags
        assert not plan.find("attention.query_layernorm").quantize
        assert not plan.find("attention.key_layernorm").quantize
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    assert plan.find("attention.q_proj") is None
    assert plan.find("attention.query_key_value.unknown") is None
