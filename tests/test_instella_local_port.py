# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/instella/SPEC.md; see NOTICE.
"""Black-box behavior tests for the package-owned Instella port.

Numerical golden values live in ``tests/test_instella_cleanroom_fixtures.py``;
this module checks the runtime contracts (cache, reload, dispatch, RoPE state).
"""

import json

import pytest
import torch
from transformers.cache_utils import DynamicCache, StaticCache

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.instella import InstellaQModel
from gptqmodel.models.ports.instella import InstellaConfig, InstellaForCausalLM, InstellaModel


def tiny_config(**overrides):
    values = {
        "vocab_size": 47, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 32,
        "pad_token_id": 1, "bos_token_id": 2, "eos_token_id": 3,
    }
    values.update(overrides)
    return InstellaConfig(**values)


def tiny_model(seed=0, attn_implementation="eager", **overrides):
    torch.manual_seed(seed)
    model = InstellaForCausalLM._from_config(tiny_config(**overrides), attn_implementation=attn_implementation)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:
                param.copy_(torch.linspace(0.6, 1.4, param.numel()).reshape(param.shape))
            elif name.endswith(".bias"):
                param.normal_(0, 0.1)
            else:
                param.normal_(0, 0.2)
    return model.eval()


def _clone(model, **kwargs):
    other = InstellaForCausalLM._from_config(model.config, **kwargs).eval()
    other.load_state_dict(model.state_dict())
    return other


IDS = torch.tensor([[2, 5, 9, 11, 4, 7, 13, 21, 6, 8], [1, 1, 2, 30, 17, 9, 40, 12, 5, 33]])
MASK = (IDS != 1).long()


def test_public_surface_and_layer_layout():
    import gptqmodel.models.ports.instella as package

    assert sorted(package.__all__) == ["InstellaConfig", "InstellaForCausalLM", "InstellaModel"]
    assert [key for key, value in MODEL_MAP.items() if value is InstellaQModel] == ["instella"]
    model = tiny_model()
    assert isinstance(model.model, InstellaModel)
    assert InstellaForCausalLM._no_split_modules == ["InstellaDecoderLayer"]
    assert model.config._attn_implementation == "eager"
    assert InstellaForCausalLM(tiny_config()).config._attn_implementation == "sdpa"
    layer = model.model.layers[0]
    assert type(layer).__name__ == "InstellaDecoderLayer"
    assert not hasattr(layer, "input_layernorm") and not hasattr(layer, "post_attention_layernorm")
    attn = layer.self_attn
    assert attn.q_norm.weight.shape == (32,) and attn.k_norm.weight.shape == (16,)
    assert attn.is_causal and attn.head_dim == 8 and attn.num_key_value_groups == 2
    assert attn.rotary_emb is not model.model.layers[1].self_attn.rotary_emb
    assert attn.rotary_emb.inv_freq.dtype == torch.float32 and attn.rotary_emb.scaling_factor == 1.0
    assert not any("rotary" in key or "inv_freq" in key for key in model.state_dict())


@pytest.mark.parametrize("implementation", ["flash_attention_2", "flex_attention"])
def test_rejects_unsupported_attention_backends(implementation):
    with pytest.raises(ValueError, match="Flash Attention 2" if implementation == "flash_attention_2" else None):
        InstellaForCausalLM._from_config(tiny_config(), attn_implementation=implementation)


@torch.no_grad()
def test_input_validation_and_return_forms():
    model = tiny_model()
    with pytest.raises(ValueError):
        model(input_ids=IDS, inputs_embeds=model.model.embed_tokens(IDS))
    with pytest.raises(ValueError):
        model()
    legacy = tuple((torch.zeros(2, 2, 1, 8), torch.zeros(2, 2, 1, 8)) for _ in range(2))
    with pytest.raises(TypeError, match="Transformers Cache"):
        model(IDS, past_key_values=legacy)
    out = model(IDS, labels=IDS, return_dict=False, use_cache=False)
    assert isinstance(out, tuple) and torch.isfinite(out[0])
    full = model(IDS, use_cache=False, output_hidden_states=True, output_attentions=True)
    assert len(full.hidden_states) == 3 and len(full.attentions) == 2
    torch.testing.assert_close(full.hidden_states[-1], model.model(IDS, use_cache=False).last_hidden_state,
                               atol=0, rtol=0)
    assert full.past_key_values is None
    torch.testing.assert_close(model(IDS, logits_to_keep=1).logits, full.logits[:, -1:], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("rope", [None, {"type": "linear", "factor": 2.0}])
@torch.no_grad()
def test_cache_continuation_causality_and_generation(dtype, implementation, rope):
    model = tiny_model(attn_implementation=implementation, rope_scaling=rope).to(dtype)
    atol = 2e-6 if dtype == torch.float32 else 5e-2
    full = model(IDS, attention_mask=MASK, use_cache=False).logits
    for cache in (DynamicCache(config=model.config), StaticCache(config=model.config, max_cache_len=16)):
        first = model(IDS[:, :6], attention_mask=MASK[:, :6], past_key_values=cache, use_cache=True).logits
        second = model(IDS[:, 6:], attention_mask=MASK, past_key_values=cache, use_cache=True).logits
        stitched = torch.cat([first, second], dim=1)
        valid = MASK.bool()
        torch.testing.assert_close(stitched[valid].float(), full[valid].float(), atol=atol, rtol=1e-2)
    changed = IDS.clone()
    changed[:, -1] = 44
    # Padded query rows are undefined (SPEC 3.5), so causality is checked without padding.
    torch.testing.assert_close(model(changed, use_cache=False).logits[:, :-1],
                               model(IDS, use_cache=False).logits[:, :-1], atol=0, rtol=0)
    kwargs = {"attention_mask": MASK[:, :6], "max_new_tokens": 6, "do_sample": False, "eos_token_id": None, "pad_token_id": 1}
    cached = model.generate(IDS[:, :6], use_cache=True, **kwargs)
    uncached = model.generate(IDS[:, :6], use_cache=False, **kwargs)
    assert torch.equal(cached, uncached)
    beam = {"num_beams": 3, "max_new_tokens": 4, "do_sample": False, "eos_token_id": None, "pad_token_id": 1}
    assert torch.equal(model.generate(IDS[:1, :4], use_cache=True, **beam),
                       model.generate(IDS[:1, :4], use_cache=False, **beam))


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@torch.no_grad()
def test_safetensors_round_trip_restores_cold_rope(tmp_path, tied, implementation):
    model = tiny_model(attn_implementation=implementation, tie_word_embeddings=tied, max_position_embeddings=8,
                       rope_scaling={"type": "dynamic", "factor": 2.0}, rope_theta=5000.0)
    assert (model.lm_head.weight is model.model.embed_tokens.weight) is tied
    cold = model.model.layers[0].self_attn.rotary_emb.inv_freq.clone()
    baseline = model(IDS[:, :6], use_cache=False).logits
    model(IDS, use_cache=False)  # 10 > 8 positions: warms every layer's buffer
    assert not torch.equal(model.model.layers[0].self_attn.rotary_emb.inv_freq, cold)
    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists() and not list(tmp_path.glob("*.bin"))
    saved = json.loads((tmp_path / "config.json").read_text())
    assert "rope_scaling" not in saved and saved["rope_theta"] == 5000.0
    restored = InstellaQModel.loader.from_pretrained(tmp_path, attn_implementation=implementation).eval()
    assert (restored.lm_head.weight is restored.model.embed_tokens.weight) is tied
    assert set(restored.state_dict()) == set(model.state_dict())
    for layer in restored.model.layers:
        torch.testing.assert_close(layer.self_attn.rotary_emb.inv_freq, cold, atol=0, rtol=0)
        assert layer.self_attn.rotary_emb.scaling_factor == 2.0
    assert restored.config.rope_theta == 5000.0
    torch.testing.assert_close(restored(IDS[:, :6], use_cache=False).logits, baseline, atol=0, rtol=0)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("assign", [False, True])
@torch.no_grad()
def test_load_state_dict_strict_bit_exact(tied, assign):
    model = tiny_model(tie_word_embeddings=tied, attention_bias=True)
    other = InstellaForCausalLM._from_config(model.config, attn_implementation="eager").eval()
    other.load_state_dict({k: v.clone() for k, v in model.state_dict().items()}, strict=True, assign=assign)
    other.tie_weights()
    assert (other.lm_head.weight is other.model.embed_tokens.weight) is tied
    torch.testing.assert_close(other(IDS).logits, model(IDS).logits, atol=0, rtol=0)


def test_missing_weights_raise_and_retry_succeeds(tmp_path):
    model = tiny_model()
    model.config.save_pretrained(tmp_path)
    with pytest.raises(OSError):
        InstellaQModel.loader.from_pretrained(tmp_path)
    model.save_pretrained(tmp_path)
    restored = InstellaQModel.loader.from_pretrained(tmp_path, attn_implementation="eager").eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(IDS).logits, model(IDS).logits, atol=0, rtol=0)


@torch.no_grad()
def test_partial_cache_failure_reset_and_retry(monkeypatch):
    model = tiny_model()
    expected = model(IDS, past_key_values=DynamicCache(config=model.config), use_cache=True).logits
    cache = DynamicCache(config=model.config)
    broken = model.model.layers[1]

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(broken, "forward", fail)
    with pytest.raises(RuntimeError, match="boom"):
        model(IDS, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length(0) == IDS.shape[1]  # layer 0 wrote before the failure; no rollback
    monkeypatch.undo()
    cache.reset()
    retry = model(IDS, past_key_values=cache, use_cache=True).logits
    torch.testing.assert_close(retry, expected, atol=0, rtol=0)


@torch.no_grad()
def test_dynamic_rope_attention_keeps_written_keys_and_publishes_warm_buffer():
    model = tiny_model(max_position_embeddings=8, rope_scaling={"type": "dynamic", "factor": 2.0})
    attn = model.model.layers[0].self_attn
    hidden = torch.randn(1, 10, 32)
    cache = DynamicCache(config=model.config)
    attn(hidden[:, :6], attention_mask=None, position_ids=torch.arange(6)[None], past_key_values=cache)
    first_keys = cache.layers[0].keys.clone()
    cold = attn.rotary_emb.inv_freq.clone()
    attn(hidden[:, 6:], attention_mask=None, position_ids=torch.arange(6, 10)[None], past_key_values=cache)
    torch.testing.assert_close(cache.layers[0].keys[:, :, :6], first_keys, atol=0, rtol=0)
    base = 10000.0 * (2.0 * 10 / 8 - 1.0) ** (8 / 6)
    warm = 1.0 / (base ** (torch.arange(0, 8, 2).float() / 8))
    torch.testing.assert_close(attn.rotary_emb.inv_freq, warm)
    assert not torch.equal(warm, cold)
    # A later short call keeps using the warm buffer.
    attn.rotary_emb(hidden, torch.arange(4)[None])
    torch.testing.assert_close(attn.rotary_emb.inv_freq, warm)


@torch.no_grad()
def test_dynamic_rope_uses_local_snapshot_after_buffer_replacement():
    model = tiny_model(max_position_embeddings=8, rope_scaling={"type": "dynamic", "factor": 2.0})
    rotary = model.model.layers[0].self_attn.rotary_emb
    reference = InstellaForCausalLM._from_config(model.config).model.layers[0].self_attn.rotary_emb
    expected_cos, expected_sin = reference(torch.zeros(1, 2, 8), torch.arange(12)[None])
    assignments = []

    class Swapping(type(rotary)):
        def __setattr__(self, name, value):
            if name == "inv_freq":
                assignments.append(value.clone())
                value = torch.full_like(value, 123.0)
            super().__setattr__(name, value)

    rotary.__class__ = Swapping
    cos, sin = rotary(torch.zeros(1, 2, 8), torch.arange(12)[None])
    assert len(assignments) == 1
    torch.testing.assert_close(cos, expected_cos, atol=0, rtol=0)
    torch.testing.assert_close(sin, expected_sin, atol=0, rtol=0)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_gradient_checkpointing_gradients_bit_identical(implementation):
    torch.manual_seed(0)
    model = tiny_model(attn_implementation=implementation).train()
    other = _clone(model, attn_implementation=implementation).train()
    other.activation_checkpointing_enable()
    for candidate in (model, other):
        candidate(IDS, attention_mask=MASK, labels=IDS, use_cache=False).loss.backward()
    for (name, left), (_, right) in zip(model.named_parameters(), other.named_parameters()):
        torch.testing.assert_close(right.grad, left.grad, atol=0, rtol=0, msg=name)


def test_public_dispatch_ignores_malicious_checkpoint_code(tmp_path):
    from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code

    model = tiny_model()
    model.save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "poison.EvilConfig", "AutoModelForCausalLM": "poison.EvilModel"}
    (tmp_path / "config.json").write_text(json.dumps(payload))
    marker = tmp_path / "EXECUTED"
    (tmp_path / "poison.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n"
    )
    assert resolve_trust_remote_code(str(tmp_path), trust_remote_code=True) is False
    assert type(load_model_config(tmp_path, trust_remote_code=True)) is InstellaConfig
    restored = InstellaQModel.loader.from_pretrained(tmp_path, trust_remote_code=True, attn_implementation="eager")
    assert type(restored) is InstellaForCausalLM and not marker.exists()
    with torch.no_grad():
        torch.testing.assert_close(restored.eval()(IDS).logits, model(IDS).logits, atol=0, rtol=0)
