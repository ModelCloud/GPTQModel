# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""EXAONE's real decoder, public loader, numerical and cache transitions."""

import copy
import json
import math

import pytest
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache, StaticCache

from gptqmodel.models.definitions.exaone import ExaOneQModel
from gptqmodel.models.ports.exaone.configuration_exaone import ExaoneConfig
from gptqmodel.models.ports.exaone.modeling_exaone import ExaoneForCausalLM


ROPE = {
    "default": None,
    "linear": {"rope_type": "linear", "factor": 2.0},
    "dynamic": {"rope_type": "dynamic", "factor": 2.0},
    "yarn": {"rope_type": "yarn", "factor": 2.0},
    "llama3": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
               "high_freq_factor": 4.0, "original_max_position_embeddings": 4},
    "longrope": {"rope_type": "longrope", "factor": 2.0, "short_factor": [1., 1., 1., 1.],
                 "long_factor": [2., 3., 4., 5.]},
}


def tiny_config(**kwargs):
    values = {"vocab_size": 37, "hidden_size": 32, "intermediate_size": 48, "num_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 8,
              "attn_implementation": "eager", "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
              "tie_word_embeddings": False}
    values.update(kwargs)
    return ExaoneConfig(**values)


def tiny_model(**kwargs):
    torch.manual_seed(112)
    return ExaoneForCausalLM(tiny_config(**kwargs)).eval()


def upstream_computation(weights, config, tokens, positions, padding=None, past=None):
    """Independent functional equations from pinned upstream EXAONE eager code.

    Uses no port modules, mask, cache or RoPE helper. Source revisions and
    Apache-2.0 attribution are recorded in the port's PROVENANCE.md.
    """
    batch, length = tokens.shape
    hidden = F.embedding(tokens, weights["transformer.wte.weight"])
    dim, heads, kv_heads = config.head_dim, config.num_attention_heads, config.num_key_value_heads
    params = config.rope_parameters
    base, kind = params["rope_theta"], params["rope_type"]
    if kind == "dynamic":
        seq = max(positions.max().item() + 1, config.max_position_embeddings)
        base *= (params["factor"] * seq / config.max_position_embeddings - (params["factor"] - 1)) ** (dim / (dim - 2))
    inverse = base ** (-torch.arange(dim // 2).float() * (2 / dim))
    scaling = 1.0
    if kind == "linear":
        inverse /= params["factor"]
    elif kind == "yarn":
        def correction(rotations):
            return dim * math.log(config.max_position_embeddings / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(correction(params.get("beta_fast") or 32)), 0)
        high = min(math.ceil(correction(params.get("beta_slow") or 1)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2).float() - low) / (high - low)).clamp(0, 1)
        inverse = inverse / params["factor"] * ramp + inverse * (1 - ramp)
        scaling = params.get("attention_factor", 0.1 * math.log(params["factor"]) + 1.0)
    elif kind == "llama3":
        wavelength = 2 * math.pi / inverse
        low = params["original_max_position_embeddings"] / params["low_freq_factor"]
        high = params["original_max_position_embeddings"] / params["high_freq_factor"]
        smooth = (params["original_max_position_embeddings"] / wavelength - params["low_freq_factor"]) / (
            params["high_freq_factor"] - params["low_freq_factor"])
        middle = (1 - smooth) * inverse / params["factor"] + smooth * inverse
        inverse = torch.where(wavelength > low, inverse / params["factor"], inverse)
        inverse = torch.where((wavelength >= high) & (wavelength <= low), middle, inverse)
    elif kind == "longrope":
        factors = params["long_factor"] if positions.max() + 1 > config.max_position_embeddings else params["short_factor"]
        inverse /= torch.tensor(factors)
        scaling = params.get("attention_factor", math.sqrt(1 + math.log(params["factor"]) / math.log(config.max_position_embeddings)))

    def linear(x, key):
        return F.linear(x, weights[key + ".weight"])

    def norm(x, key):
        normalized = x.float() / (x.float().square().mean(-1, keepdim=True) + config.layer_norm_epsilon).sqrt()
        return weights[key + ".weight"] * normalized.to(x.dtype)

    theta = positions.float()[..., None] * inverse
    cosine, sine = (theta.cos() * scaling).to(hidden.dtype)[:, None], (theta.sin() * scaling).to(hidden.dtype)[:, None]
    def rotate(x):
        left, right = x[..., :dim // 2], x[..., dim // 2:]
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1)

    new_cache = []
    for index in range(config.num_layers):
        prefix = f"transformer.h.{index}."
        x = norm(hidden, prefix + "ln_1")
        q, k, v = [linear(x, prefix + "attn.attention." + name).view(batch, length, count, dim).transpose(1, 2)
                   for name, count in [("q_proj", heads), ("k_proj", kv_heads), ("v_proj", kv_heads)]]
        q, k = rotate(q), rotate(k)
        if past is not None:
            k, v = torch.cat((past[index][0], k), -2), torch.cat((past[index][1], v), -2)
        new_cache.append((k, v))
        key = k.repeat_interleave(heads // kv_heads, dim=1)
        value = v.repeat_interleave(heads // kv_heads, dim=1)
        key_length = key.shape[-2]
        allowed = torch.arange(key_length)[None, :] <= torch.arange(key_length - length, key_length)[:, None]
        allowed = allowed[None, None]
        if padding is not None:
            allowed = allowed & padding[:, None, None, :].bool()
        scores = (q @ key.transpose(-1, -2) / dim ** .5).masked_fill(~allowed, torch.finfo(q.dtype).min)
        attended = scores.float().softmax(-1).to(q.dtype) @ value
        hidden = hidden + linear(attended.transpose(1, 2).reshape(batch, length, -1), prefix + "attn.attention.out_proj")
        x = norm(hidden, prefix + "ln_2")
        hidden = hidden + linear(F.silu(linear(x, prefix + "mlp.c_fc_0")) * linear(x, prefix + "mlp.c_fc_1"), prefix + "mlp.c_proj")
    return linear(norm(hidden, "transformer.ln_f"), "lm_head").float(), new_cache


@pytest.mark.parametrize("rope", ROPE)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.no_grad()
def test_matches_upstream_forward_and_rotated_key_cache(rope, dtype):
    model = tiny_model(rope_scaling=ROPE[rope]).to(dtype)
    tokens = torch.tensor([[0, 1, 9, 7, 3, 6], [1, 4, 8, 5, 2, 9]])
    padding = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    positions = (padding.cumsum(-1) - 1).clamp(min=0)
    tolerance = 3e-6 if dtype == torch.float32 else 0.006
    expected, _ = upstream_computation(model.state_dict(), model.config, tokens, positions, padding)
    actual = model(tokens, attention_mask=padding, position_ids=positions)
    # Fully masked pad queries differ between eager/SDPA; real tokens must agree.
    torch.testing.assert_close(actual.logits[padding.bool()], expected[padding.bool()], atol=tolerance, rtol=tolerance)
    cache, reference = None, None
    for start, end in [(0, 3), (3, 4), (4, 6)]:
        expected, reference = upstream_computation(model.state_dict(), model.config, tokens[:, start:end],
                                                    positions[:, start:end], padding[:, :end], reference)
        actual = model(tokens[:, start:end], attention_mask=padding[:, :end], position_ids=positions[:, start:end],
                       past_key_values=cache, use_cache=True)
        cache = actual.past_key_values
        assert cache.get_seq_length() == end
        valid = padding[:, start:end].bool()
        torch.testing.assert_close(actual.logits[valid], expected[valid], atol=tolerance, rtol=tolerance)
        for index, (keys, values) in enumerate(reference):
            torch.testing.assert_close(cache.layers[index].keys, keys, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(cache.layers[index].values, values, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("tied", [False, True])
@torch.no_grad()
def test_generation_cache_causality_safetensors_reload(tmp_path, implementation, tied):
    model = tiny_model(attn_implementation=implementation, tie_word_embeddings=tied)
    tokens = torch.tensor([[1, 3, 7, 8, 9], [1, 5, 6, 4, 3]])
    baseline = model(tokens, use_cache=False).logits
    for cache in (DynamicCache(config=model.config), StaticCache(config=model.config, max_cache_len=10)):
        model(tokens[:, :3], past_key_values=cache)
        output = model(tokens[:, 3:], past_key_values=cache)
        torch.testing.assert_close(output.logits, baseline[:, 3:], atol=2e-7, rtol=1e-5)
        assert cache.layers[0].keys.shape[1] == 2
    changed = tokens.clone()
    changed[:, -1] = 12
    torch.testing.assert_close(model(changed).logits[:, :-1], baseline[:, :-1], atol=2e-7, rtol=1e-5)
    attention = model(tokens, output_attentions=True, output_hidden_states=True)
    assert len(attention.attentions) == 2 and len(attention.hidden_states) == 3
    for beams in (1, 3):
        args = {"max_new_tokens": 3, "do_sample": False, "num_beams": beams, "eos_token_id": None}
        assert torch.equal(model.generate(tokens, **args), model.generate(tokens, use_cache=False, **args))
    model.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
    assert not list(tmp_path.glob("*.py"))
    restored = ExaOneQModel.loader.from_pretrained(tmp_path, attn_implementation=implementation).eval()
    torch.testing.assert_close(restored(tokens).logits, baseline, atol=2e-7, rtol=1e-5)
    assert set(restored.state_dict()) == set(model.state_dict())
    assert (restored.lm_head.weight is restored.transformer.wte.weight) is tied


def write_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
                                  **{str(i): i for i in range(3, 37)}}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]").save_pretrained(path)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@torch.no_grad()
def test_public_dispatch_ignores_malicious_checkpoint_code(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel
    from gptqmodel.utils.hf import load_model_config
    model = tiny_model()
    model.save_pretrained(tmp_path)
    write_tokenizer(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "poison.EvilConfig", "AutoModelForCausalLM": "poison.EvilModel"}
    (tmp_path / "config.json").write_text(json.dumps(payload))
    marker = tmp_path / "EXECUTED"
    (tmp_path / "poison.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n")
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")
    (tmp_path / "config.json").write_text(json.dumps({**payload, "head_dim": 7}))
    with pytest.raises(ValueError, match="head_dim"):
        GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert not marker.exists()
    (tmp_path / "config.json").write_text(json.dumps(payload))
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is ExaoneConfig
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert isinstance(loaded, ExaOneQModel)
    assert type(loaded.model) is ExaoneForCausalLM
    assert loaded.trust_remote_code is False and not marker.exists()
    ids = torch.tensor([[1, 7, 9]])
    torch.testing.assert_close(loaded.model(ids).logits, model(ids).logits, atol=2e-7, rtol=1e-5)


@torch.no_grad()
def test_cache_reorder_crop_reset_and_failed_mask_retry():
    model = tiny_model()
    ids = torch.tensor([[1, 7, 9], [1, 6, 8]])
    cache = model(ids[:, :2]).past_key_values
    before = [layer.keys.clone() for layer in cache.layers]
    with pytest.raises(ValueError, match="attention_mask"):
        model(ids[:, 2:], past_key_values=cache, attention_mask=torch.ones(2, 2))
    assert cache.get_seq_length() == 2
    for layer, keys in zip(cache.layers, before):
        assert torch.equal(layer.keys, keys)
    output = model(ids[:, 2:], past_key_values=cache)
    torch.testing.assert_close(output.logits, model(ids).logits[:, -1:], atol=2e-7, rtol=1e-5)
    cache.reorder_cache(torch.tensor([1, 0, 1]))
    cache.crop(-1)
    reordered = model(ids[[1, 0, 1], 2:], past_key_values=cache)
    torch.testing.assert_close(reordered.logits, output.logits[[1, 0, 1]], atol=2e-7, rtol=1e-5)
    cache.reset()
    assert cache.get_seq_length() == 0
    torch.testing.assert_close(model(ids, past_key_values=cache).logits, model(ids).logits)
    with pytest.raises(TypeError, match="legacy tuple"):
        model(ids, past_key_values=())


@pytest.mark.parametrize("rope", ["dynamic", "longrope"])
@torch.no_grad()
def test_long_request_does_not_change_subsequent_rope_request(rope):
    model = tiny_model(rope_scaling=ROPE[rope], max_position_embeddings=4)
    ids = torch.tensor([[1, 7, 9, 5, 6, 8, 3, 4, 9, 5, 8, 6]])
    reference = copy.deepcopy(model)
    model(ids)
    for length in (6, 2):
        torch.testing.assert_close(model(ids[:, :length]).logits, reference(ids[:, :length]).logits)


def test_gradient_checkpointing_backward_disables_cache():
    model = tiny_model().train()
    model.gradient_checkpointing_enable()
    ids = torch.tensor([[1, 3, 7, 5]])
    output = model(ids, labels=ids, use_cache=True)
    assert output.past_key_values is None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.transformer.h[0].attn.attention.q_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("cache_implementation", ["dynamic", "static"])
@torch.no_grad()
def test_padded_generation_uses_current_transformers_cache_api(implementation, cache_implementation):
    model = tiny_model(attn_implementation=implementation)
    ids = torch.tensor([[0, 0, 1, 7, 9], [1, 6, 8, 4, 5]])
    mask = ids.ne(0).long()
    args = {"attention_mask": mask, "max_new_tokens": 3, "eos_token_id": None, "do_sample": False}
    expected = model.generate(ids, use_cache=False, **args)
    actual = model.generate(ids, cache_implementation=cache_implementation, **args)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@torch.no_grad()
def test_config_aliases_input_embeddings_labels_and_tuple_outputs(kv_heads):
    config = tiny_config(num_hidden_layers=2, num_key_value_heads=kv_heads,
                         hidden_act="silu", rms_norm_eps=1e-5,
                         rope_scaling={"type": "linear", "factor": 2.0})
    assert config.num_layers == config.num_hidden_layers == 2
    assert config.activation_function == config.hidden_act == "silu"
    assert config.layer_norm_epsilon == config.rms_norm_eps == 1e-5
    assert config.rope_parameters["rope_type"] == "linear"
    model = ExaoneForCausalLM(config).eval()
    ids = torch.tensor([[1, 7, 9, 6]])
    logits, _ = upstream_computation(model.state_dict(), config, ids, torch.arange(4)[None])
    actual = model(inputs_embeds=model.get_input_embeddings()(ids), labels=ids, use_cache=False)
    torch.testing.assert_close(actual.logits, logits, atol=2e-7, rtol=1e-5)
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten())
    torch.testing.assert_close(actual.loss, loss)
    sequence = model(ids, labels=ids, use_cache=False, return_dict=False)
    assert len(sequence) == 2
    torch.testing.assert_close(sequence[0], actual.loss)
    torch.testing.assert_close(sequence[1], actual.logits)
    with pytest.raises(ValueError, match="exactly one"):
        model(ids, inputs_embeds=model.get_input_embeddings()(ids))
    with pytest.raises(ValueError, match="exactly one"):
        model()


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@torch.no_grad()
def test_invalid_4d_mask_retry_does_not_mutate_existing_cache(static, implementation):
    model = tiny_model(attn_implementation=implementation)
    ids = torch.tensor([[1, 7, 9, 3], [1, 6, 8, 5]])
    cache = StaticCache(config=model.config, max_cache_len=8) if static else DynamicCache(config=model.config)
    model(ids[:, :2], past_key_values=cache)
    before = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    with pytest.raises(ValueError, match="attention_mask"):
        model(ids[:, 2:], past_key_values=cache, attention_mask=torch.zeros(2, 1, 2, 3))
    assert cache.get_seq_length() == 2
    for layer, (keys, values) in zip(cache.layers, before):
        assert torch.equal(layer.keys, keys) and torch.equal(layer.values, values)
    actual = model(ids[:, 2:], past_key_values=cache)
    torch.testing.assert_close(actual.logits, model(ids, use_cache=False).logits[:, 2:], atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure_site", ["decoder", "norm", "head", "loss", "encoding"])
@pytest.mark.parametrize("prefix_length", [0, 2])
@torch.no_grad()
def test_failure_after_cache_writes_rolls_back_and_retries(monkeypatch, static, failure_site, prefix_length):
    model = tiny_model()
    ids = torch.tensor([[1, 7, 9, 3], [1, 6, 8, 5]])
    cache = StaticCache(config=model.config, max_cache_len=8) if static else DynamicCache(config=model.config)
    if prefix_length:
        model(ids[:, :prefix_length], past_key_values=cache)
    before = [(layer.keys.clone(), layer.values.clone()) if layer.is_initialized else None for layer in cache.layers]
    pointers = [(layer.keys.data_ptr(), layer.values.data_ptr()) if layer.is_initialized else None for layer in cache.layers]
    fired = False

    def fail_after_write(*args, **kwargs):
        nonlocal fired
        assert cache.layers[0].get_seq_length() == 4, "failure must occur after a completed mutating block"
        fired = True
        raise RuntimeError("injected after cache write")

    with monkeypatch.context() as patch:
        handles = []
        if failure_site == "loss":
            patch.setattr(model, "_loss_function", fail_after_write, raising=False)
        elif failure_site == "encoding":
            from transformers.modeling_outputs import CausalLMOutputWithPast
            patch.setattr(CausalLMOutputWithPast, "to_tuple", fail_after_write)
        else:
            target = {"decoder": model.transformer.h[1].attn.attention.out_proj,
                      "norm": model.transformer.ln_f, "head": model.lm_head}[failure_site]
            handles.append(target.register_forward_hook(fail_after_write))
        with pytest.raises(RuntimeError, match="injected after cache write"):
            try:
                model(ids[:, prefix_length:], past_key_values=cache, labels=ids[:, prefix_length:], return_dict=False)
            finally:
                for handle in handles:
                    handle.remove()
    assert fired
    for layer, previous, addresses in zip(cache.layers, before, pointers):
        assert layer.get_seq_length() == prefix_length
        if previous is None:
            assert not layer.is_initialized and layer.keys is None and layer.values is None
        else:
            keys, values = previous
            assert torch.equal(layer.keys, keys) and torch.equal(layer.values, values)
            if static:
                assert (layer.keys.data_ptr(), layer.values.data_ptr()) == addresses
    actual = model(ids[:, prefix_length:], past_key_values=cache)
    assert actual.past_key_values is cache
    torch.testing.assert_close(actual.logits, model(ids, use_cache=False).logits[:, prefix_length:], atol=2e-7, rtol=1e-5)
