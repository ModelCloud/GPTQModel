import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from defuser import convert_model
from safetensors.torch import save_file
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

from gptqmodel.models import auto
from gptqmodel.models.definitions.glm_moe_dsa import GlmMoeDsaQModel
from gptqmodel.utils.structure import LazyTurtle, alias_from_turtle_for_submodule


_UPSTREAM_GLM5_MODELING_SIGNATURE = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "first_k_dense_replace": 3,
    "hidden_act": "silu",
    "hidden_size": 6144,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 2048,
    "intermediate_size": 12288,
    "kv_lora_rank": 512,
    "max_position_embeddings": 202752,
    "model_type": "glm_moe_dsa",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 78,
    "num_key_value_heads": 64,
    "q_lora_rank": 2048,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "v_head_dim": 256,
}

_UPSTREAM_GLM5_1_MODELING_SIGNATURE = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "first_k_dense_replace": 3,
    "hidden_act": "silu",
    "hidden_size": 6144,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 2048,
    "intermediate_size": 12288,
    "kv_lora_rank": 512,
    "max_position_embeddings": 202752,
    "model_type": "glm_moe_dsa",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 78,
    "num_key_value_heads": 64,
    "q_lora_rank": 2048,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "v_head_dim": 256,
}


def _tiny_glm_moe_dsa_config(num_hidden_layers: int = 4) -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _build_lazy_turtle(tmp_path: Path, model: GlmMoeDsaForCausalLM) -> LazyTurtle:
    # Persist a tiny real GLM checkpoint so the test exercises the checkpoint-backed lazy path.
    model_dir = tmp_path / "glm_source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    save_file(state_dict, str(model_dir / shard_name))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(state_dict, shard_name)}),
        encoding="utf-8",
    )
    source = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert source is not None
    return source


@pytest.mark.parametrize("model_path", ["/tmp/glm-5", "/tmp/glm-5.1"])
def test_glm_moe_dsa_model_type_selects_definition_for_glm5_variants(monkeypatch, model_path):
    fake_config = SimpleNamespace(model_type="glm_moe_dsa")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition(model_path) is GlmMoeDsaQModel


def test_glm5_and_glm5_1_share_same_upstream_modeling_signature():
    # Snapshot from the current upstream config.json files fetched on 2026-04-07.
    assert _UPSTREAM_GLM5_MODELING_SIGNATURE == _UPSTREAM_GLM5_1_MODELING_SIGNATURE


def test_glm_moe_dsa_module_tree_expands_dense_and_sparse_paths():
    layer_modules = GlmMoeDsaQModel.simple_layer_modules(
        model_config=_tiny_glm_moe_dsa_config(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert GlmMoeDsaQModel.layer_modules_strict is False
    assert "self_attn.q_a_proj" in flat_modules
    assert "self_attn.kv_a_proj_with_mqa" in flat_modules
    assert "self_attn.indexer.wk" in flat_modules
    assert "self_attn.indexer.wq_b" in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.experts.0.up_proj" in flat_modules
    assert "mlp.experts.0.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_proj" in flat_modules
    assert "mlp.shared_experts.up_proj" in flat_modules
    assert "mlp.shared_experts.down_proj" in flat_modules


def test_glm_moe_dsa_tiny_model_matches_definition():
    model = GlmMoeDsaForCausalLM(_tiny_glm_moe_dsa_config())
    convert_model(model, cleanup_original=False)

    dense_layer = model.model.layers[0]
    moe_layer = model.model.layers[3]

    assert hasattr(dense_layer.self_attn, "q_a_proj")
    assert hasattr(dense_layer.self_attn, "kv_a_proj_with_mqa")
    assert hasattr(dense_layer.self_attn.indexer, "wk")
    assert hasattr(dense_layer.self_attn.indexer, "wq_b")
    assert hasattr(dense_layer.mlp, "gate_proj")
    assert hasattr(dense_layer.mlp, "up_proj")
    assert hasattr(dense_layer.mlp, "down_proj")
    assert not hasattr(dense_layer.mlp, "experts")

    assert hasattr(moe_layer.mlp, "gate")
    assert hasattr(moe_layer.mlp, "experts")
    assert hasattr(moe_layer.mlp, "shared_experts")
    assert len([name for name, _ in moe_layer.mlp.experts.named_children() if name.isdigit()]) == model.config.n_routed_experts

    expert0 = getattr(moe_layer.mlp.experts, "0")
    assert hasattr(expert0, "gate_proj")
    assert hasattr(expert0, "up_proj")
    assert hasattr(expert0, "down_proj")


def test_glm_moe_dsa_lazy_turtle_restores_rotary_buffers_from_module_init(tmp_path):
    source_model = GlmMoeDsaForCausalLM(_tiny_glm_moe_dsa_config())
    convert_model(source_model, cleanup_original=False)
    shell_model = GlmMoeDsaForCausalLM(_tiny_glm_moe_dsa_config())
    convert_model(shell_model, cleanup_original=False)
    shell_model.load_state_dict(source_model.state_dict())

    rotary = shell_model.model.rotary_emb
    rotary.register_buffer("inv_freq", torch.empty_like(rotary.inv_freq, device="meta"), persistent=False)
    rotary.register_buffer(
        "original_inv_freq",
        torch.empty_like(rotary.original_inv_freq, device="meta"),
        persistent=False,
    )

    turtle = _build_lazy_turtle(tmp_path, source_model)
    alias_from_turtle_for_submodule(
        target_model=shell_model,
        turtle_model=turtle,
        target_submodule=shell_model.model.rotary_emb,
        device=torch.device("cpu"),
    )

    rebuilt_rotary = shell_model.model.rotary_emb
    assert hasattr(rebuilt_rotary, "inv_freq")
    assert hasattr(rebuilt_rotary, "original_inv_freq")
    torch.testing.assert_close(rebuilt_rotary.inv_freq, source_model.model.rotary_emb.inv_freq)
    torch.testing.assert_close(rebuilt_rotary.original_inv_freq, source_model.model.rotary_emb.original_inv_freq)
