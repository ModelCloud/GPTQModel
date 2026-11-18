# SPDX-License-Identifier: Apache-2.0
"""Real LLaDA2 packed-QKV/dense/shared/routed checkpoint weight ownership."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.llada2 import LLaDA2MoeQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.llada2.modeling_llada2_moe import LLaDA2MoeModelLM
from awq_neutrality import assert_awq_folds_neutral

from tests.test_llada2_local_port import VERSIONS, tiny_config, tiny_model


def test_exact_central_plan_proposal_and_no_fusion_admission():
    from gptqmodel.models.fusable import require_fusable_config

    assert LLaDA2MoeQModel.extract_layers_node() == ["model.layers"]
    assert LLaDA2MoeQModel.build_layer_modules(LLaDA2MoeQModel.module_tree, include_capture_only=True) == [
        ["input_layernorm:!"],
        ["attention.query_key_value", "attention.query_layernorm:!", "attention.key_layernorm:!"],
        ["attention.dense"],
        ["post_attention_layernorm:!"],
        ["mlp:?"],
        ["mlp.gate:!", "mlp.gate.expert_bias:!"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
        ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"],
        ["mlp.shared_experts.down_proj"],
        ["mlp.experts.{expert_index}.gate_proj", "mlp.experts.{expert_index}.up_proj"],
        ["mlp.experts.{expert_index}.down_proj"],
    ]
    with pytest.raises(ValueError):
        require_fusable_config(LLaDA2MoeQModel, "llada2_moe")


@pytest.mark.parametrize(
    "version,kv,bias,shared,dense",
    [(version, *layout) for version, layout in zip(VERSIONS, [(4, False, None, 0), (2, True, 1, 1), (1, False, 2, 2)] * 2)],
)
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
    assert plan.find("attention.q_proj") is None
    assert plan.find("attention.query_key_value.unknown") is None


@pytest.mark.parametrize(
    "version,overrides",
    [
        ("2.1", {}),
        ("2.0", {"num_key_value_heads": 4, "use_bias": True, "use_qkv_bias": True}),
        ("2.2", {"use_qk_norm": False}),
    ],
)
def test_awq_scale_folds_preserve_dense_and_moe_layer_outputs(version, overrides):
    model = tiny_model(version, **overrides)
    ids = (torch.arange(12).view(1, 12) % 30) + 2

    def forward():
        return model(input_ids=ids).logits

    dense_layer, moe_layer = model.model.layers
    dense = {source: (targets, consumers)
             for source, targets, consumers in assert_awq_folds_neutral(LLaDA2MoeQModel, model, dense_layer, forward)}
    assert dense == {
        "input_layernorm": (["attention.query_key_value"], []),
        "post_attention_layernorm": (["mlp.gate_proj", "mlp.up_proj"], []),
        "mlp.up_proj": (["mlp.down_proj"], []),
    }

    checked = assert_awq_folds_neutral(LLaDA2MoeQModel, model, moe_layer, forward)
    moe = {source: (targets, consumers) for source, targets, consumers in checked}
    assert moe["input_layernorm"] == (["attention.query_key_value"], [])
    targets, consumers = moe["post_attention_layernorm"]
    # Shared and every routed gate/up read one norm; the biased sigmoid router
    # reads it too and is compensated, so expert selection is unchanged.
    assert targets == ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"] + [
        f"mlp.experts.{i}.{name}" for i in range(4) for name in ("gate_proj", "up_proj")
    ]
    assert consumers == ["mlp.gate"]
    assert moe["mlp.shared_experts.up_proj"] == (["mlp.shared_experts.down_proj"], [])
    routed = [source for source in moe if source.startswith("mlp.experts.")]
    assert routed and all(moe[source][0] == [source.replace("up_proj", "down_proj")] for source in routed)
    # attention.dense reads the attention output and never receives a fold.
    for layer_groups in (dense, moe):
        assert all("attention.dense" not in targets for targets, _ in layer_groups.values())
