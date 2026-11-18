# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-owned original Qwen shapes and actual module-plan consumers."""

import json
from types import SimpleNamespace

import pytest
import torch
from test_qwen_local_port import ASSETS, tiny_model

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.qwen import QwenQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.qwen.configuration_qwen import QWenConfig
from gptqmodel.models.ports.qwen.modeling_qwen import QWenLMHeadModel
from tests.port_assets import read_port_json


EXPECTED_PLAN = {
    "layers": ["transformer.h"],
    "blocks": [["ln_1:!"], ["attn.c_attn"], ["attn.c_proj"], ["ln_2:!"], ["mlp.w1", "mlp.w2"], ["mlp.c_proj"]],
}


def expected_shapes(c):
    h, mid = c.hidden_size, c.intermediate_size // 2
    shapes = {
        "transformer.wte.weight": (c.vocab_size, h),
        "transformer.ln_f.weight": (h,),
        "lm_head.weight": (c.vocab_size, h),
    }
    block = {
        "ln_1.weight": (h,),
        "attn.c_attn.weight": (3 * h, h),
        "attn.c_attn.bias": (3 * h,),
        "attn.c_proj.weight": (h, h),
        "ln_2.weight": (h,),
        "mlp.w1.weight": (mid, h),
        "mlp.w2.weight": (mid, h),
        "mlp.c_proj.weight": (h, mid),
    }
    if not c.no_bias:
        block.update({"attn.c_proj.bias": (h,), "mlp.w1.bias": (mid,), "mlp.w2.bias": (mid,), "mlp.c_proj.bias": (h,)})
    for idx in range(c.num_hidden_layers):
        shapes.update({f"transformer.h.{idx}.{key}": value for key, value in block.items()})
    return shapes


@pytest.mark.parametrize("release", ["Qwen-1_8B", "Qwen-7B", "Qwen-14B", "Qwen-72B", "Qwen-7B-Chat"])
def test_released_indices_headers_exact_live_meta_owners(release):
    manifest = json.loads((ASSETS / "provenance.json").read_text())["releases"][release]
    headers = read_port_json(ASSETS / "fixtures/checkpoint_headers.json")[release]
    c = QWenConfig.from_dict(manifest["config"])
    with torch.device("meta"):
        model = QWenLMHeadModel(c)
    state = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert set(manifest["keys"]) == set(state) == set(headers)
    assert state == expected_shapes(c) == {key: tuple(value["shape"]) for key, value in headers.items()}
    assert all(p.device.type == "meta" for p in model.parameters())
    del model


@pytest.mark.parametrize("no_bias", [False, True])
@pytest.mark.parametrize("rotary_pct", [0.5, 1.0])
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_every_cpu_checkpoint_weight_resolves_through_live_plan(no_bias, rotary_pct, heads):
    model = tiny_model(no_bias=no_bias, rotary_pct=rotary_pct, num_attention_heads=heads, kv_channels=32 // heads)
    _convert_model_with_defuser(QwenQModel, model, cleanup_original=False)
    assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == expected_shapes(model.config)
    for method in ["gptq", "awq"]:
        tree = QwenQModel._resolve_effective_module_tree(model, SimpleNamespace(method=method))
        assert QwenQModel.build_layer_modules(tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]
        plan = ModulePlan.from_tree(tree)
        for block in model.transformer.h:
            targets = set()
            for key in block.state_dict():
                owner = key.rsplit(".", 1)[0]
                spec = plan.find(owner)
                assert spec is not None, key
                assert spec.quantize == (owner not in {"ln_1", "ln_2"})
                if spec.quantize:
                    targets.add(owner)
                else:
                    assert "!" in spec.flags
            assert targets == {"attn.c_attn", "attn.c_proj", "mlp.w1", "mlp.w2", "mlp.c_proj"}
        assert "nosrc" in plan.find("mlp.w2").flags
        for name in ["attn.c_attn.extra", "other.attn.c_attn", "attn.c_attn_suffix"]:
            assert plan.find(name) is None
    assert MODEL_MAP["qwen"] is QwenQModel and QwenQModel.fusable is None
    assert QwenQModel.extract_layers_node() == EXPECTED_PLAN["layers"]


def test_plain_awq_definition_policy_preserves_all_five_targets():
    model = tiny_model()
    wrapper = QwenQModel.__new__(QwenQModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    assert wrapper.awq_get_modules_for_scaling(model.transformer.h[0], {}, {}) == []
    assert [
        name for group in wrapper.simple_layer_modules(model.config, SimpleNamespace(method="awq")) for name in group
    ] == ["attn.c_attn", "attn.c_proj", "mlp.w1", "mlp.w2", "mlp.c_proj"]
