# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

from gptqmodel.models import auto
from gptqmodel.models.definitions.lfm2_vl import LFM2VLQModel
from gptqmodel.utils.structure import LazyTurtle


MODEL_PATH = Path("/monster/data/model/LFM2.5-VL-1.6B")


def test_lfm2_vl_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="lfm2_vl")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/lfm2-vl") is LFM2VLQModel


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="LFM2.5-VL model not found")
def test_lfm2_vl_local_config_selects_definition():
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=False)

    assert config.model_type == "lfm2_vl"
    assert auto.check_and_get_model_definition(MODEL_PATH) is LFM2VLQModel


def test_lfm2_vl_module_tree_covers_conv_attention_and_mlp_paths():
    layer_modules = LFM2VLQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert LFM2VLQModel.layer_modules_strict is False
    assert LFM2VLQModel.require_load_processor is True
    assert LFM2VLQModel.require_trust_remote_code is False
    assert LFM2VLQModel.pre_lm_head_norm_module == "model.language_model.embedding_norm"
    assert LFM2VLQModel.rotary_embedding == "model.language_model.rotary_emb"
    assert LFM2VLQModel.extract_layers_node() == ["model.language_model.layers"]

    assert "conv.in_proj" in flat_modules
    assert "conv.out_proj" in flat_modules
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.out_proj" in flat_modules
    assert "feed_forward.w1" in flat_modules
    assert "feed_forward.w3" in flat_modules
    assert "feed_forward.w2" in flat_modules


def test_lfm2_vl_base_modules_include_non_language_children():
    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity()])
            self.embed_tokens = nn.Embedding(4, 4)
            self.rotary_emb = nn.Identity()
            self.embedding_norm = nn.LayerNorm(4)

    class _LFM2VLCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _LanguageModel()
            self.vision_tower = nn.Identity()
            self.multi_modal_projector = nn.Identity()

    class _LFM2VLWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _LFM2VLCore()
            self.lm_head = nn.Linear(4, 4, bias=False)

    base_modules = set(LFM2VLQModel.get_base_modules(_LFM2VLWrapper()))

    assert "model.vision_tower" in base_modules
    assert "model.multi_modal_projector" in base_modules
    assert "model.language_model.embed_tokens" in base_modules
    assert "model.language_model.rotary_emb" in base_modules
    assert "model.language_model.embedding_norm" in base_modules


def test_lfm2_vl_replay_kwargs_convert_long_mask_for_attention_layers():
    config = Lfm2Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        layer_types=["conv", "conv", "full_attention"],
    )
    config._attn_implementation = "eager"
    hidden_states = torch.randn(1, 4, config.hidden_size, dtype=torch.bfloat16)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    position_ids = torch.arange(4).unsqueeze(0)
    model_def = object.__new__(LFM2VLQModel)

    attention_layer = SimpleNamespace(
        is_attention_layer=True,
        self_attn=SimpleNamespace(config=config),
    )
    replay_kwargs = model_def.prepare_layer_replay_kwargs(
        attention_layer,
        [hidden_states],
        {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        torch.device("cpu"),
    )

    replay_mask = replay_kwargs["attention_mask"]
    assert replay_mask is None or replay_mask.dtype in (torch.bool, hidden_states.dtype, torch.float32)

    conv_layer = SimpleNamespace(is_attention_layer=False)
    replay_kwargs = model_def.prepare_layer_replay_kwargs(
        conv_layer,
        [hidden_states],
        {"attention_mask": attention_mask},
        torch.device("cpu"),
    )
    assert replay_kwargs["attention_mask"].dtype == torch.bool


def test_lfm2_vl_lazy_turtle_resolves_siglip2_vision_model_prefix(tmp_path):
    checkpoint_tensors = {
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight": torch.zeros(2, 2),
    }
    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    save_file(checkpoint_tensors, str(model_dir / shard_name))
    (model_dir / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.vision_tower.vision_model.embeddings.patch_embedding.weight":"model.safetensors"}}',
        encoding="utf-8",
    )

    class _PatchEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, 2, device="meta"))

    class _Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embedding = _PatchEmbedding()

    class _VisionTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = _Embeddings()

    class _Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = _VisionTower()

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Core()

    shell = _Wrapper()
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
        module_tree=LFM2VLQModel.module_tree,
        hf_conversion_map_reversed=LFM2VLQModel.resolve_hf_conversion_map_reversed(),
        target_model=shell,
    )

    assert turtle is not None
    assert (
        turtle._resolve_checkpoint_tensor_name(
            "model.vision_tower.embeddings.patch_embedding",
            "weight",
        )
        == "model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    )

    tensors = turtle.checkpoint_tensors_for_submodule(
        target_model=shell,
        target_submodule=shell.model.vision_tower.embeddings.patch_embedding,
        recurse=False,
    )
    assert set(tensors) == {"weight"}


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="LFM2.5-VL model not found")
def test_lfm2_vl_native_shell_matches_definition_tree():
    from accelerate import init_empty_weights

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=False)
    with init_empty_weights(include_buffers=True):
        shell = AutoModelForImageTextToText.from_config(config, trust_remote_code=False)

    conv_layer = shell.model.language_model.layers[0]
    attention_layer = shell.model.language_model.layers[2]

    assert hasattr(shell.model, "language_model")
    assert hasattr(shell.model, "vision_tower")
    assert hasattr(shell.model, "multi_modal_projector")
    assert hasattr(conv_layer, "conv")
    assert hasattr(attention_layer, "self_attn")
    assert hasattr(attention_layer.self_attn, "out_proj")
    assert hasattr(attention_layer.self_attn, "q_layernorm")
