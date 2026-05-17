import json
from types import SimpleNamespace

from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.mimo_v2 import MimoV2QModel


_LOCAL_MIMO_V2_5_BASE_MODELING_SIGNATURE = {
    "architectures": ["MiMoV2ForCausalLM"],
    "attention_projection_layout": "fused_qkv",
    "hidden_size": 4096,
    "intermediate_size": 16384,
    "model_type": "mimo_v2",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "num_key_value_heads": 4,
}


class _FakeVisualMerger(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_q = nn.LayerNorm(8)
        self.mlp = nn.Sequential(
            nn.Linear(8, 8),
            nn.GELU(),
            nn.Linear(8, 4),
        )


class _FakeAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_local_transformer = nn.Module()
        self.input_local_transformer.embed_tokens = nn.Embedding(16, 8)


def test_mimo_v2_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="mimo_v2")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/monster/data/model/MiMo-V2.5-Base") is MimoV2QModel


def test_mimo_v2_5_base_local_modeling_signature_snapshot():
    assert _LOCAL_MIMO_V2_5_BASE_MODELING_SIGNATURE == {
        "architectures": ["MiMoV2ForCausalLM"],
        "attention_projection_layout": "fused_qkv",
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "model_type": "mimo_v2",
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_attention_heads": 64,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
    }


def test_mimo_v2_module_tree_expands_fused_attention_dense_mlp_and_moe_paths():
    layer_modules = MimoV2QModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=4),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert MimoV2QModel.require_trust_remote_code is True
    assert MimoV2QModel.layer_modules_strict is False
    assert MimoV2QModel.pre_lm_head_norm_module == "model.norm"
    assert MimoV2QModel.rotary_embedding == "model.rotary_emb"
    assert "self_attn.qkv_proj" in flat_modules
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.experts.0.up_proj" in flat_modules
    assert "mlp.experts.0.down_proj" in flat_modules
    assert "mlp.gate" not in flat_modules


def test_mimo_v2_drops_visual_merger_biases_when_checkpoint_omits_them(tmp_path):
    model = SimpleNamespace(
        visual=SimpleNamespace(
            merger=_FakeVisualMerger()
        )
    )
    index = {
        "metadata": {},
        "weight_map": {
            "visual.merger.ln_q.weight": "model.safetensors",
            "visual.merger.mlp.0.weight": "model.safetensors",
            "visual.merger.mlp.2.weight": "model.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    MimoV2QModel._drop_visual_merger_biases_if_checkpoint_omits_them(model, str(tmp_path))

    assert model.visual.merger.ln_q.bias is None
    assert model.visual.merger.mlp[0].bias is None
    assert model.visual.merger.mlp[2].bias is None


def test_mimo_v2_keeps_visual_merger_biases_when_checkpoint_has_them(tmp_path):
    model = SimpleNamespace(
        visual=SimpleNamespace(
            merger=_FakeVisualMerger()
        )
    )
    index = {
        "metadata": {},
        "weight_map": {
            "visual.merger.ln_q.weight": "model.safetensors",
            "visual.merger.ln_q.bias": "model.safetensors",
            "visual.merger.mlp.0.weight": "model.safetensors",
            "visual.merger.mlp.0.bias": "model.safetensors",
            "visual.merger.mlp.2.weight": "model.safetensors",
            "visual.merger.mlp.2.bias": "model.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    MimoV2QModel._drop_visual_merger_biases_if_checkpoint_omits_them(model, str(tmp_path))

    assert model.visual.merger.ln_q.bias is not None
    assert model.visual.merger.mlp[0].bias is not None
    assert model.visual.merger.mlp[2].bias is not None


def test_mimo_v2_drops_audio_input_embedding_when_checkpoint_omits_it(tmp_path):
    model = SimpleNamespace(audio_encoder=_FakeAudioEncoder())
    index = {
        "metadata": {},
        "weight_map": {
            "audio_encoder.input_local_transformer.layers.0.input_layernorm.weight": "model.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    MimoV2QModel._drop_checkpoint_omitted_audio_tensors(model, str(tmp_path))

    assert model.audio_encoder.input_local_transformer.embed_tokens.weight is None


def test_mimo_v2_keeps_audio_input_embedding_when_checkpoint_has_it(tmp_path):
    model = SimpleNamespace(audio_encoder=_FakeAudioEncoder())
    index = {
        "metadata": {},
        "weight_map": {
            "audio_encoder.input_local_transformer.embed_tokens.weight": "model.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    MimoV2QModel._drop_checkpoint_omitted_audio_tensors(model, str(tmp_path))

    assert model.audio_encoder.input_local_transformer.embed_tokens.weight is not None
