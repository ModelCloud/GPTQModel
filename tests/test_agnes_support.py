# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from gptqmodel.models import auto
from gptqmodel.models.definitions import agnes
from gptqmodel.models.definitions.agnes import AgnesQModel
from gptqmodel.utils.model import MODALITY


def test_agnes_registry_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="agnes")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda path: None)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.MODEL_MAP["agnes"] is AgnesQModel
    assert auto.check_and_get_model_definition("/tmp/agnes", trust_remote_code=True) is AgnesQModel


def test_agnes_definition_contract_and_module_tree():
    assert AgnesQModel.loader.__name__ == "AutoModelForImageTextToText"
    assert AgnesQModel.require_trust_remote_code is True
    assert AgnesQModel.require_load_processor is True
    assert AgnesQModel.layer_modules_strict is False
    assert AgnesQModel.modality == [MODALITY.TEXT]
    assert AgnesQModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert AgnesQModel.rotary_embedding == "model.language_model.rotary_emb"
    assert AgnesQModel.out_of_model_tensors == {"prefixes": ["mtp"]}
    assert AgnesQModel.awq_scale_optimize_shape_dependent_modules == ["global_attn.o_proj"]
    assert AgnesQModel.extract_layers_node() == ["model.language_model.layers"]
    assert "shared_input_verified_model_types" not in AgnesQModel.__dict__


def test_agnes_simple_and_full_layer_modules_cover_delta_global_and_parallel_ffn():
    quantize_config = SimpleNamespace(dynamic=None)
    simple = AgnesQModel.simple_layer_modules(SimpleNamespace(), quantize_config)
    full = AgnesQModel.full_layer_modules(SimpleNamespace())

    assert simple == [
        ["delta_attn.in_proj_qkv"],
        ["delta_attn.in_proj_z"],
        ["delta_attn.out_proj"],
        ["global_attn.q_proj", "global_attn.k_proj", "global_attn.v_proj"],
        ["global_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
        ["mlp.parallel_ffn.gate_proj", "mlp.parallel_ffn.up_proj"],
        ["mlp.parallel_ffn.down_proj"],
    ]

    full_names = {name for block in full for name in block}
    for excluded in (
        "input_layernorm:!",
        "post_attention_layernorm:!",
        "delta_attn.norm:!",
        "delta_attn.conv1d:!",
        "delta_attn.in_proj_b:!",
        "delta_attn.in_proj_a:!",
        "global_attn.q_norm:!",
        "global_attn.k_norm:!",
    ):
        assert excluded in full_names

    simple_names = {name for block in simple for name in block}
    assert not any(name.endswith(":!") for name in simple_names)


def test_agnes_replay_keeps_original_2d_mask_when_base_hook_changes_it(monkeypatch):
    original_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    hidden_states = torch.zeros(1, 4, 8)
    calls = []

    def base_prepare(_self, _layer, _layer_input, additional_inputs, _target_device):
        additional_inputs["attention_mask"] = "base-transformed-mask"
        return additional_inputs

    def fake_create_causal_mask(**kwargs):
        calls.append(kwargs)
        return "causal-mask"

    monkeypatch.setattr(AgnesQModel.__mro__[1], "prepare_layer_replay_kwargs", base_prepare)
    monkeypatch.setattr(agnes, "create_causal_mask", fake_create_causal_mask)

    qmodel = object.__new__(AgnesQModel)
    global_layer = SimpleNamespace(
        layer_type="agnes_global_attention",
        global_attn=SimpleNamespace(config=SimpleNamespace(), layer_idx=3),
    )
    replay_kwargs = qmodel.prepare_layer_replay_kwargs(
        global_layer,
        [hidden_states],
        {"attention_mask": original_mask, "position_ids": torch.arange(4).unsqueeze(0)},
        torch.device("cpu"),
    )

    assert replay_kwargs["attention_mask"] == "causal-mask"
    assert calls[0]["attention_mask"] is original_mask
    assert calls[0]["inputs_embeds"] is hidden_states
    assert calls[0]["layer_idx"] == 3

    delta_layer = SimpleNamespace(layer_type="agnes_delta_attention", delta_attn=SimpleNamespace())
    replay_kwargs = qmodel.prepare_layer_replay_kwargs(
        delta_layer,
        [hidden_states],
        {"attention_mask": original_mask},
        torch.device("cpu"),
    )
    assert replay_kwargs["attention_mask"] is original_mask


def test_agnes_replay_reduces_legacy_extended_mask_for_delta_layer():
    hidden_states = torch.zeros(1, 4, 8)
    extended_mask = torch.tensor([[[[0, 0, -10000, -10000]]]], dtype=torch.float32)
    qmodel = object.__new__(AgnesQModel)
    replay_kwargs = qmodel.prepare_layer_replay_kwargs(
        SimpleNamespace(layer_type="agnes_delta_attention"),
        [hidden_states],
        {"attention_mask": extended_mask},
        torch.device("cpu"),
    )

    assert replay_kwargs["attention_mask"].shape == (1, 4)
    assert replay_kwargs["attention_mask"].dtype == torch.bool
    assert replay_kwargs["attention_mask"].tolist() == [[True, True, False, False]]


def test_agnes_remote_code_compat_is_scoped_to_agnes_loader(monkeypatch):
    from transformers import cache_utils, video_processing_utils

    monkeypatch.delattr(cache_utils, "LAYER_TYPE_CACHE_MAPPING", raising=False)
    monkeypatch.delattr(video_processing_utils, "BASE_VIDEO_PROCESSOR_DOCSTRING", raising=False)

    # The generic remote-code compatibility path must not restore Agnes-only
    # symbols for unrelated models.
    from gptqmodel.utils import hf as hf_utils

    hf_utils._patch_transformers_remote_code_compat()
    assert not hasattr(cache_utils, "LAYER_TYPE_CACHE_MAPPING")
    assert not hasattr(video_processing_utils, "BASE_VIDEO_PROCESSOR_DOCSTRING")

    AgnesQModel.before_model_load(AgnesQModel, "/tmp/agnes", False)

    assert cache_utils.LAYER_TYPE_CACHE_MAPPING is cache_utils.DYNAMIC_LAYER_TYPE_MAPPING
    assert video_processing_utils.BASE_VIDEO_PROCESSOR_DOCSTRING == ""
