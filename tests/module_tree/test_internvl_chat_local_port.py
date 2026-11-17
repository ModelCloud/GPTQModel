# SPDX-License-Identifier: Apache-2.0
"""Inventory real decoder checkpoint owners after normal Defuser conversion."""

import re

import pytest
import torch
from test_internvl_chat_local_port import ARCHITECTURES, tiny_config

from gptqmodel.models.definitions.internvl_chat import InternVLChatQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.internvl_chat.modeling_internvl_chat import InternVLChatModel
from gptqmodel.quantization import QuantizeConfig


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_real_checkpoint_owners_quantization_consumers_and_plain_awq(architecture):
    model = InternVLChatModel(tiny_config(architecture)).eval()
    ids = torch.tensor([[7, 8, 9]])
    with torch.no_grad():
        before = model(input_ids=ids).logits
    _convert_model_with_defuser(InternVLChatQModel, model, cleanup_original=False)
    qmodel = InternVLChatQModel(model, quantized=False, quantize_config=QuantizeConfig(bits=4, group_size=8))
    plan = qmodel.module_plan()
    quantizable = {name for block in qmodel.simple_layer_modules(model.config, qmodel.quantize_config) for name in block}
    assert quantizable
    resolved = set()
    for layer in model.language_model.model.layers:
        # state_dict includes persistent buffers as well as parameters.
        for key in layer.state_dict():
            owner = key.rpartition(".")[0]
            spec = plan.find(owner)
            assert spec is not None, (architecture, owner)
            if spec.quantize:
                assert owner in quantizable, (architecture, owner)
                assert isinstance(layer.get_submodule(owner), torch.nn.Linear)
                resolved.add(owner)
            else:
                assert "!" in spec.flags or spec.capture_only
            assert plan.find(owner + ".undeclared") is None
        assert qmodel.awq_get_modules_for_scaling(layer, {}, {}) == []
    assert resolved == quantizable
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids).logits, before, atol=1e-6, rtol=1e-5)
    if architecture == "Qwen3MoeForCausalLM":
        assert qmodel.get_num_experts(model.config) == 2
        assert plan.moe_roots()
        assert plan.is_expert("mlp.experts.1.gate_proj")
        assert not plan.find("mlp.gate").quantize
    else:
        assert not plan.moe_roots()
    assert re.fullmatch(r"language_model.model.layers", qmodel.extract_layers_node()[0])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_prequant_hooks_preserve_embedding_and_rotary_owners(architecture, monkeypatch):
    model = InternVLChatModel(tiny_config(architecture)).eval()
    qmodel = InternVLChatQModel(model, quantized=False, quantize_config=QuantizeConfig(bits=4, group_size=16, offload_to_disk=False))
    visited = []
    def prequant(module):
        visited.append(module)
        return module
    monkeypatch.setattr(qmodel, "pre_quantize", prequant)
    qmodel.pre_quantize_generate_hook_start()
    qmodel.pre_quantize_generate_hook_end()
    assert model.get_input_embeddings() is visited[0]
    rotary_paths = qmodel._decoder_rotary_paths()
    assert len(rotary_paths) == (2 if architecture == "InternLM2ForCausalLM" else 1)
    assert all(model.get_submodule(path) in visited for path in rotary_paths)
