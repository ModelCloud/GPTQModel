# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.models.definitions.mllama import MLlamaQModel
from gptqmodel.utils.structure import LazyTurtle
from gptqmodel.utils.model import get_layers_with_prefixes


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = _Attention()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = _MLP()
        self.last_kwargs = None

    def forward(self, hidden_states, **kwargs):
        self.last_kwargs = kwargs
        return hidden_states


class _Rotary(nn.Module):
    def forward(self, x, position_ids=None):
        shape = (*x.shape[:2], x.shape[-1] // 2)
        return x.new_zeros(shape), x.new_zeros(shape)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([_DecoderLayer(), _DecoderLayer()])
        self.norm = nn.LayerNorm(4)
        self.rotary_emb = _Rotary()


class _CurrentMllamaWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _LanguageModel()
        self.model.vision_model = nn.Identity()
        self.model.multi_modal_projector = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 4, bias=False)


class _LegacyMllamaWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = _LanguageModel()
        self.lm_head = nn.Linear(4, 4, bias=False)


def test_mllama_module_tree_supports_current_transformers_layout():
    model = _CurrentMllamaWrapper()

    layers, layer_names = get_layers_with_prefixes(
        model,
        MLlamaQModel.extract_layers_node(),
    )

    assert MLlamaQModel.extract_layers_node()[0] == "model.language_model.layers"
    assert len(layers) == 2
    assert layer_names == [
        "model.language_model.layers.0",
        "model.language_model.layers.1",
    ]
    assert MLlamaQModel.pre_lm_head_norm_module == "model.language_model.norm"
    base_modules = set(MLlamaQModel.get_base_modules(model))
    assert "model.vision_model" in base_modules
    assert "model.multi_modal_projector" in base_modules


def test_mllama_module_tree_keeps_legacy_language_model_layout():
    model = _LegacyMllamaWrapper()

    layers, layer_names = get_layers_with_prefixes(
        model,
        MLlamaQModel.extract_layers_node(),
    )

    assert len(layers) == 2
    assert layer_names == [
        "language_model.model.layers.0",
        "language_model.model.layers.1",
    ]


def test_mllama_pre_quantize_hooks_materialize_text_input_modules():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.model = _CurrentMllamaWrapper()
    instance.quantize_config = SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )

    instance.pre_quantize_generate_hook_start()
    instance.pre_quantize_generate_hook_end()

    language_model = instance.model.model.language_model
    assert language_model.embed_tokens.weight.device.type == "cpu"
    assert next(language_model.rotary_emb.parameters(), None) is None


def test_mllama_run_input_capture_calls_first_decoder_layer_directly():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.model = _CurrentMllamaWrapper()
    instance.quantize_config = SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )

    example = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.bool),
    }

    instance.pre_quantize_generate_hook_start()
    instance.run_input_capture(example, use_cache=False, data_device=torch.device("cpu"))

    first_layer = instance.model.model.language_model.layers[0]
    assert first_layer.last_kwargs["attention_mask"] is None
    assert first_layer.last_kwargs["position_ids"].shape == (1, 3)
    assert first_layer.last_kwargs["position_embeddings"][0].shape == (1, 3, 2)


def test_mllama_lazy_turtle_conversion_map_matches_checkpoint_prefixes():
    renamings = LazyTurtle._normalize_runtime_to_checkpoint_renamings(
        MLlamaQModel.resolve_hf_conversion_map_reversed(target_model=_CurrentMllamaWrapper())
    )

    def convert(path):
        out = [path]
        for renaming in renamings:
            renamed, _ = renaming.rename_source_key(path)
            out.append(renamed)
        return out

    assert "language_model.model.embed_tokens" in convert("model.language_model.embed_tokens")
    assert "language_model.model.layers.0" in convert("model.language_model.layers.0")
    assert "language_model.lm_head" in convert("lm_head")
    assert "vision_model" in convert("model.vision_model")
    assert "multi_modal_projector" in convert("model.multi_modal_projector")


def test_mllama_run_input_capture_moves_input_ids_to_embedding_device():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.model = _CurrentMllamaWrapper()
    instance.quantize_config = SimpleNamespace(
        device="meta",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )

    language_model = instance.model.model.language_model
    language_model.embed_tokens = language_model.embed_tokens.to("meta")

    example = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.bool),
    }

    try:
        instance.run_input_capture(example, use_cache=False, data_device=torch.device("cpu"))
    except RuntimeError as exc:
        assert "produced meta inputs_embeds" in str(exc)
