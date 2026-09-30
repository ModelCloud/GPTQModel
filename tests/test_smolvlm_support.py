# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoConfig, SmolVLMConfig, SmolVLMForConditionalGeneration

from gptqmodel.models import auto
from gptqmodel.models.definitions import smolvlm as smolvlm_definition
from gptqmodel.models.definitions.smolvlm import SmolVLMQModel


def _tiny_smolvlm():
    config = SmolVLMConfig(
        text_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
        ),
        vision_config=dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=32,
            patch_size=8,
        ),
    )
    return config, SmolVLMForConditionalGeneration(config)


def test_smolvlm_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="smolvlm")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/smolvlm") is SmolVLMQModel
    assert auto.MODEL_MAP["smolvlm"] is SmolVLMQModel


def test_smolvlm_local_config_selects_definition():
    config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", trust_remote_code=False)

    assert config.model_type == "smolvlm"
    assert auto.check_and_get_model_definition("HuggingFaceTB/SmolVLM2-2.2B-Instruct") is SmolVLMQModel


def test_smolvlm_definition_metadata():
    assert SmolVLMQModel.require_load_processor is True
    assert SmolVLMQModel.require_trust_remote_code is False
    assert SmolVLMQModel.require_pkgs == ["num2words"]
    assert SmolVLMQModel.pre_lm_head_norm_module == "model.text_model.norm"
    assert SmolVLMQModel.rotary_embedding == "model.text_model.rotary_emb"
    assert SmolVLMQModel.extract_layers_node() == ["model.text_model.layers"]


def test_smolvlm_definition_is_exported_from_definitions_package():
    from gptqmodel.models import definitions

    assert definitions.SmolVLMQModel is SmolVLMQModel


def test_smolvlm_module_tree_covers_decoder_paths():
    config, model = _tiny_smolvlm()
    layer_modules = SmolVLMQModel.simple_layer_modules(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert layer_modules == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]

    full_modules = {name for block in SmolVLMQModel.full_layer_modules(config) for name in block}
    assert "input_layernorm:!" in full_modules
    assert "post_attention_layernorm:!" in full_modules

    decoder_layer = model.model.text_model.layers[0]
    for module_name in {name.split(":")[0] for name in full_modules}:
        module = decoder_layer
        for attribute in module_name.split("."):
            module = getattr(module, attribute)
        assert module is not None


def test_smolvlm_base_modules_include_vision_stack_and_text_endpoints():
    _, model = _tiny_smolvlm()
    base_modules = set(SmolVLMQModel.get_base_modules(model))

    assert "model.vision_model" in base_modules
    assert "model.connector" in base_modules
    assert "model.text_model.embed_tokens" in base_modules
    assert "model.text_model.norm" in base_modules
    assert "model.text_model.rotary_emb" in base_modules
    assert "model.text_model.layers" not in base_modules


def test_smolvlm_prepare_dataset_uses_processor_chat_template():
    calls = {}

    class _FakeProcessor:
        def apply_chat_template(self, conversations, **kwargs):
            calls["conversations"] = conversations
            calls["kwargs"] = kwargs
            return {"input_ids": [[1, 2, 3]]}

    conversations = [[{"role": "user", "content": [{"type": "text", "text": "hi"}]}]]

    result = SmolVLMQModel.prepare_inputs_for_conversations(_FakeProcessor(), conversations)

    assert result == {"input_ids": [[1, 2, 3]]}
    assert calls["conversations"] is conversations
    assert calls["kwargs"]["add_generation_prompt"] is True
    assert calls["kwargs"]["tokenize"] is True
    assert calls["kwargs"]["return_dict"] is True
    assert calls["kwargs"]["return_tensors"] == "pt"


def _fake_smolvlm_qmodel():
    model = nn.Module()
    model.model = nn.Module()
    text_model = nn.Module()
    text_model.embed_tokens = nn.Embedding(4, 4)
    text_model.norm = nn.LayerNorm(4)
    text_model.rotary_emb = nn.Linear(4, 4)
    model.model.text_model = text_model
    model.model.vision_model = nn.Linear(4, 4)
    model.model.connector = nn.Linear(4, 4)

    qmodel = object.__new__(SmolVLMQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    qmodel.turtle_model = None
    qmodel._direct_parameter_names = ()
    return qmodel, model


def test_smolvlm_pre_quantize_hook_materializes_base_modules():
    qmodel, model = _fake_smolvlm_qmodel()
    qmodel.quantize_config = SimpleNamespace(device=torch.device("cpu"))
    materialized = []

    def shell_module_materialize(module, device):
        materialized.append((module, device))
        return module

    qmodel.shell_module_materialize = shell_module_materialize

    qmodel.pre_quantize_generate_hook_start()

    assert materialized == [
        (model.model.text_model.embed_tokens, torch.device("cpu")),
        (model.model.text_model.norm, torch.device("cpu")),
        (model.model.text_model.rotary_emb, torch.device("cpu")),
        (model.model.vision_model, torch.device("cpu")),
        (model.model.connector, torch.device("cpu")),
    ]


def test_smolvlm_pre_quantize_hook_end_offloads_base_modules(monkeypatch):
    qmodel, model = _fake_smolvlm_qmodel()
    qmodel.quantize_config = SimpleNamespace(
        device=torch.device("cpu"),
        offload_to_disk=True,
        offload_to_disk_path="/tmp/smolvlm-offload",
    )
    offloaded = []

    def offload_to_disk(model, module, disk_path):
        offloaded.append((model, module, disk_path))

    monkeypatch.setattr(smolvlm_definition, "offload_to_disk", offload_to_disk)

    qmodel.pre_quantize_generate_hook_end()

    text_model = model.model.text_model
    assert offloaded == [
        (text_model, text_model.embed_tokens, "/tmp/smolvlm-offload"),
        (text_model, text_model.norm, "/tmp/smolvlm-offload"),
        (text_model, text_model.rotary_emb, "/tmp/smolvlm-offload"),
        (model.model, model.model.vision_model, "/tmp/smolvlm-offload"),
        (model.model, model.model.connector, "/tmp/smolvlm-offload"),
    ]
