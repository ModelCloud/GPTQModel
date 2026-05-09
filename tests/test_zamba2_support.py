# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from transformers.models.zamba2.configuration_zamba2 import Zamba2Config
from transformers.models.zamba2.modeling_zamba2 import Zamba2ForCausalLM

from gptqmodel.models import auto
from gptqmodel.models.definitions.zamba2 import Zamba2QModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear


def _tiny_zamba2_config() -> Zamba2Config:
    return Zamba2Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        n_mamba_heads=4,
        chunk_size=16,
        max_position_embeddings=128,
        layers_block_type=["mamba", "hybrid"],
        num_mem_blocks=1,
        use_shared_attention_adapter=False,
        use_mem_rope=False,
        use_mamba_kernels=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )


def test_zamba2_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="zamba2")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/zamba2") is Zamba2QModel


def test_zamba2_module_tree_expands_mamba_and_hybrid_paths():
    layer_modules = Zamba2QModel.simple_layer_modules(
        model_config=_tiny_zamba2_config(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}
    full_layer_modules = Zamba2QModel.full_layer_modules(model_config=_tiny_zamba2_config())
    full_flat_modules = {name for block in full_layer_modules for name in block}

    assert Zamba2QModel.layer_modules_strict is False
    assert Zamba2QModel.pre_lm_head_norm_module == "lm_head"
    assert Zamba2QModel.extract_layers_node() == ["model.layers"]
    assert "input_layernorm:!" in full_flat_modules
    assert "mamba.in_proj" in flat_modules
    assert "mamba.out_proj" in flat_modules
    assert "linear" in flat_modules
    assert "mamba_decoder.mamba.in_proj" in flat_modules
    assert "mamba_decoder.mamba.out_proj" in flat_modules


def test_zamba2_base_modules_include_embeddings_and_final_norm():
    model = Zamba2ForCausalLM(_tiny_zamba2_config())
    base_modules = set(Zamba2QModel.get_base_modules(model))

    assert "model.embed_tokens" in base_modules
    assert "model.final_layernorm" in base_modules
    assert "model.layers" not in base_modules


class _DenseProjection:
    def __init__(self, device_type: str = "cuda"):
        self.weight = SimpleNamespace(device=SimpleNamespace(type=device_type))


class _QuantizedProjection(BaseQuantLinear):
    def __init__(self, device_type: str = "cuda"):
        torch.nn.Module.__init__(self)
        self.qweight = SimpleNamespace(device=SimpleNamespace(type=device_type))


def _build_dummy_zamba2_mixer(in_proj, out_proj=None):
    out_proj = _DenseProjection() if out_proj is None else out_proj

    def __init__(self):
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.path = None
        self.call = None

    def cuda_kernels_forward(self, hidden_states, cache_params=None, attention_mask=None):
        self.path = "fast"
        self.call = {
            "hidden_states": hidden_states,
            "cache_params": cache_params,
            "attention_mask": attention_mask,
        }
        return "fast"

    def torch_forward(self, hidden_states, cache_params=None, attention_mask=None):
        self.path = "slow"
        self.call = {
            "hidden_states": hidden_states,
            "cache_params": cache_params,
            "attention_mask": attention_mask,
        }
        return "slow"

    mixer_class = type(
        "Zamba2MambaMixer",
        (),
        {
            "__init__": __init__,
            "cuda_kernels_forward": cuda_kernels_forward,
            "torch_forward": torch_forward,
        },
    )
    return mixer_class()


def test_zamba2_quantized_mixer_uses_torch_path(monkeypatch):
    from transformers.models.zamba2 import modeling_zamba2

    monkeypatch.setattr(modeling_zamba2, "selective_state_update", object())
    monkeypatch.setattr(modeling_zamba2, "mamba_chunk_scan_combined", object())
    monkeypatch.setattr(modeling_zamba2, "mamba_split_conv1d_scan_combined", object())
    monkeypatch.setattr(modeling_zamba2, "causal_conv1d_fn", object())
    monkeypatch.setattr(modeling_zamba2, "causal_conv1d_update", object())
    monkeypatch.setattr(modeling_zamba2, "is_torchdynamo_compiling", lambda: False)

    mixer = _build_dummy_zamba2_mixer(in_proj=_QuantizedProjection())
    layer = SimpleNamespace(mamba=mixer)
    qmodel = Zamba2QModel.__new__(Zamba2QModel)
    qmodel.model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    qmodel.monkey_patch()

    hidden_states = object()
    cache_params = object()
    attention_mask = object()

    result = mixer.forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask)

    assert result == "slow"
    assert mixer.path == "slow"
    assert mixer.call == {
        "hidden_states": hidden_states,
        "cache_params": cache_params,
        "attention_mask": attention_mask,
    }


def test_zamba2_dense_cuda_mixer_keeps_fast_path(monkeypatch):
    from transformers.models.zamba2 import modeling_zamba2

    monkeypatch.setattr(modeling_zamba2, "selective_state_update", object())
    monkeypatch.setattr(modeling_zamba2, "mamba_chunk_scan_combined", object())
    monkeypatch.setattr(modeling_zamba2, "mamba_split_conv1d_scan_combined", object())
    monkeypatch.setattr(modeling_zamba2, "causal_conv1d_fn", object())
    monkeypatch.setattr(modeling_zamba2, "causal_conv1d_update", object())
    monkeypatch.setattr(modeling_zamba2, "is_torchdynamo_compiling", lambda: False)

    mixer = _build_dummy_zamba2_mixer(in_proj=_DenseProjection())
    layer = SimpleNamespace(mamba_decoder=SimpleNamespace(mamba=mixer))
    qmodel = Zamba2QModel.__new__(Zamba2QModel)
    qmodel.model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    qmodel.monkey_patch()

    hidden_states = object()
    cache_params = object()
    attention_mask = object()

    result = mixer.forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask)

    assert result == "fast"
    assert mixer.path == "fast"
    assert mixer.call == {
        "hidden_states": hidden_states,
        "cache_params": cache_params,
        "attention_mask": attention_mask,
    }


def test_zamba2_capture_and_replay_kwargs_preserve_original_hidden_states_and_layer_index():
    model_def = Zamba2QModel.__new__(Zamba2QModel)
    hidden_states = torch.randn(1, 3, 8)
    original_hidden_states = torch.randn(1, 3, 8)
    attention_mask = torch.ones(1, 3)
    causal_mask = torch.zeros(1, 1, 3, 3)

    captured_kwargs = model_def.capture_first_layer_input_kwargs(
        args=(hidden_states, original_hidden_states, 0, attention_mask, causal_mask),
        kwargs={},
        batch_device=torch.device("cpu"),
        layer_input_kwargs={},
    )

    layer = SimpleNamespace(mamba_decoder=SimpleNamespace(layer_idx=7))
    replay_kwargs = model_def.prepare_layer_replay_kwargs(
        layer=layer,
        layer_input=[hidden_states],
        additional_inputs=dict(captured_kwargs),
        target_device=torch.device("cpu"),
    )

    assert torch.equal(replay_kwargs["original_hidden_states"], original_hidden_states)
    assert torch.equal(replay_kwargs["attention_mask"], attention_mask)
    assert torch.equal(replay_kwargs["causal_mask"], causal_mask)
    assert replay_kwargs["layer_idx"] == 7
