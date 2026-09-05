from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.glm5_next import Glm5NextQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks
from gptqmodel.utils.model import find_modules
from gptqmodel.utils.structure import LazyTurtle
from defuser.model_registry import MODEL_CONFIG


def _tiny_text_config():
    glm5_next = pytest.importorskip("transformers.models.glm5_next")
    return glm5_next.Glm5NextTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        index_topk=4,
        index_kpool=2,
        index_head_dim=8,
        index_n_heads=2,
        linear_head_dim=8,
        linear_num_heads=4,
        hc_mult=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_glm5_next_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="glm5_next")
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("glm-5.3-flash-fixture") is Glm5NextQModel


def test_glm5_next_module_tree_matches_reference_quantization_boundary():
    config = SimpleNamespace(text_config=SimpleNamespace(n_routed_experts=2))
    layer_modules = Glm5NextQModel.simple_layer_modules(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat = {name for block in layer_modules for name in block}

    assert {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_b_proj",
        "self_attn.o_proj",
        "mlp.experts.0.gate_proj",
        "mlp.experts.1.up_proj",
        "mlp.experts.1.down_proj",
    } <= flat
    assert {
        "self_attn.q_a_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.indexer.wq_b",
        "self_attn.indexer.wk",
        "self_attn.indexer.weights_proj",
        "mlp.shared_experts.gate_proj",
    }.isdisjoint(flat)

    full = {
        name
        for block in Glm5NextQModel.full_layer_modules(model_config=config)
        for name in block
    }
    assert "self_attn.q_a_proj:!" in full
    assert "mlp.shared_experts.gate_proj:!" in full
    assert Glm5NextQModel.out_of_model_tensors == {
        "prefixes": ["model.language_model.layers.45"]
    }


def test_glm5_next_quantizes_every_loaded_decoder_layer():
    layer = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=45))
    assert "should_quantize_layer" not in Glm5NextQModel.__dict__
    assert all(
        Glm5NextQModel.should_quantize_layer(
            layer=layer,
            layer_name=f"model.language_model.layers.{index}",
            layer_index=index,
            quantize_config=SimpleNamespace(),
        )
        for index in range(45)
    )


def test_glm5_next_defuser_registry_preserves_expert_forward():
    glm5_next = pytest.importorskip("transformers.models.glm5_next")
    assert "glm5_next" in MODEL_CONFIG
    text_model = glm5_next.Glm5NextTextModel(_tiny_text_config()).eval()

    class TinyOuter(nn.Module):
        def __init__(self, language_model):
            super().__init__()
            self.config = SimpleNamespace(model_type="glm5_next")
            self.model = nn.Module()
            self.model.language_model = language_model

    model = TinyOuter(text_model)
    experts = model.model.language_model.layers[3].mlp.experts
    hidden_states = torch.randn(4, text_model.config.hidden_size)
    topk_indices = torch.tensor([[0], [1], [0], [1]])
    topk_weights = torch.ones(4, 1)

    with torch.no_grad():
        expected = experts(hidden_states, topk_indices, topk_weights)

    assert _convert_model_with_defuser(Glm5NextQModel, model, cleanup_original=False) is True

    with torch.no_grad():
        actual = experts(hidden_states, topk_indices, topk_weights)
    torch.testing.assert_close(actual, expected)

    modules = find_modules(model)
    assert "model.language_model.layers.0.self_attn.q_proj" in modules
    assert "model.language_model.layers.3.self_attn.q_b_proj" in modules
    assert "model.language_model.layers.3.self_attn.kv_b_proj" in modules
    assert "model.language_model.layers.3.mlp.experts.0.gate_proj" in modules
    assert "model.language_model.layers.3.mlp.experts.0.up_proj" in modules
    assert "model.language_model.layers.3.mlp.experts.0.down_proj" in modules

    layer_modules = Glm5NextQModel.simple_layer_modules(
        model_config=SimpleNamespace(text_config=text_model.config),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    suffixes = {name for block in layer_modules for name in block}
    matched = {name for name in modules if any(name.endswith(suffix) for suffix in suffixes)}
    assert "model.language_model.layers.0.self_attn.q_proj" in matched
    assert "model.language_model.layers.3.self_attn.q_b_proj" in matched
    assert "model.language_model.layers.3.mlp.experts.0.gate_proj" in matched
    assert not any(name.endswith("self_attn.indexer.weights_proj") for name in matched)


def test_glm5_next_lazy_turtle_materializes_defused_experts(tmp_path):
    glm5_next = pytest.importorskip("transformers.models.glm5_next")
    config = glm5_next.Glm5NextConfig(
        text_config=_tiny_text_config().to_dict(),
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "projection_intermediate_size": 32,
            "out_hidden_size": 32,
            "num_heads": 4,
            "in_channels": 3,
            "image_size": 4,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
        },
        image_token_id=60,
        video_token_id=61,
        image_start_token_id=62,
        image_end_token_id=63,
        video_start_token_id=58,
        video_end_token_id=59,
    )
    source = glm5_next.Glm5NextForConditionalGeneration(config).eval()
    packed_experts = source.model.language_model.layers[3].mlp.experts
    expected_gate, expected_up = packed_experts.gate_up_proj[0].detach().chunk(2, dim=0)
    expected_down = packed_experts.down_proj[0].detach().clone()
    source.save_pretrained(tmp_path)

    with torch.device("meta"):
        shell = glm5_next.Glm5NextForConditionalGeneration(config).eval()
    assert _convert_model_with_defuser(Glm5NextQModel, shell, cleanup_original=False) is True

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(tmp_path),
        config=shell.config,
        model_init_kwargs={"device_map": {"": "cpu"}},
        module_tree=Glm5NextQModel.module_tree,
        hf_conversion_map_reversed=Glm5NextQModel.resolve_hf_conversion_map_reversed(
            target_model=shell
        ),
        target_model=shell,
    )
    assert turtle is not None

    layer = shell.model.language_model.layers[3]
    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=layer,
        device=torch.device("cpu"),
        module_path="model.language_model.layers.3",
        show_progress=False,
    )

    expert0 = layer.mlp.experts[0]
    torch.testing.assert_close(expert0.gate_proj.weight, expected_gate)
    torch.testing.assert_close(expert0.up_proj.weight, expected_up)
    torch.testing.assert_close(expert0.down_proj.weight, expected_down)


def test_glm5_next_replay_propagates_dsa_topk_indices():
    model_def = Glm5NextQModel.__new__(Glm5NextQModel)
    topk_indices = torch.tensor([[[1, 2], [0, 3]]])
    kwargs = {}

    returned = model_def.update_layer_replay_kwargs_from_output(
        layer=SimpleNamespace(),
        layer_output=(torch.zeros(1, 2, 4), topk_indices),
        layer_input_kwargs=kwargs,
        target_device=torch.device("cpu"),
    )
    assert returned is kwargs
    torch.testing.assert_close(kwargs["prev_topk_indices"], topk_indices)


def test_glm5_next_moe_calibration_uses_clamped_fused_gate():
    class Experts:
        @staticmethod
        def _apply_gate(gate_up):
            gate, up = gate_up.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate.clamp(max=1.0)) * up.clamp(-1.0, 1.0)

    gate = torch.tensor([[3.0, -2.0]])
    up = torch.tensor([[4.0, -4.0]])
    actual = GateUpDownMoELifecycleHooks().apply_expert_activation(
        experts_module=Experts(),
        expert=SimpleNamespace(),
        gate_out=gate,
        up_out=up,
    )
    expected = Experts._apply_gate(torch.cat([gate, up], dim=-1))
    torch.testing.assert_close(actual, expected)
