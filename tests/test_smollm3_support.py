from types import SimpleNamespace

from transformers import SmolLM3Config, SmolLM3ForCausalLM

from gptqmodel.models import auto
from gptqmodel.models.definitions.llama import LlamaQModel


def test_smollm3_model_type_selects_llama_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="smollm3")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/smollm3") is LlamaQModel


def test_smollm3_quantization_groups_match_runtime_projection_order():
    layer_modules = LlamaQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert layer_modules == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]


def test_smollm3_transformers_runtime_uses_llama_projection_paths():
    config = SmolLM3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SmolLM3ForCausalLM(config)
    layer_modules = set(dict(model.model.layers[0].named_modules()))

    assert {
        "input_layernorm",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "post_attention_layernorm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    } <= layer_modules
    assert LlamaQModel.pre_lm_head_norm_module == "model.norm"
