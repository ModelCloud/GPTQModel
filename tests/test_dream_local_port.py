# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Dream must select packaged diffusion code without executing checkpoint Python."""

import json

import pytest
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from gptqmodel.models.definitions.dream import DreamQModel
from gptqmodel.models.ports.dream.configuration_dream import DreamConfig
from gptqmodel.models.ports.dream.generation_utils import DreamGenerationConfig, sample_tokens
from gptqmodel.models.ports.dream.modeling_dream import DreamModel
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


@pytest.mark.parametrize("model_type", ["dream", "Dream"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_packaged_selection_over_malicious_auto_map(tmp_path, monkeypatch, model_type, trust, bypass):
    marker = tmp_path / "EXECUTED"
    (tmp_path / "evil.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote')\n")
    payload = {"model_type": model_type, "vocab_size": 31, "hidden_size": 16, "intermediate_size": 24,
                   "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                   "pad_token_id": 0, "mask_token_id": 30, "max_position_embeddings": 32,
                   "auto_map": {"AutoConfig": "evil.Config", "AutoModel": "evil.Model"}}
    (tmp_path / "config.json").write_text(json.dumps(payload))
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security bypass is the contract under test.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config).__module__ == "gptqmodel.models.ports.dream.configuration_dream"
    with torch.device("meta"):
        model = DreamQModel.loader.from_config(config, trust_remote_code=trust)
    assert type(model).__module__ == "gptqmodel.models.ports.dream.modeling_dream"
    assert next(model.parameters()).device.type == "meta"
    assert not marker.exists()


def tiny_config(**kwargs):
    values = {"vocab_size": 31, "hidden_size": 32, "intermediate_size": 64,
              "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
              "pad_token_id": 0, "mask_token_id": 30, "max_position_embeddings": 32}
    return DreamConfig(**(values | kwargs))


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    torch.set_num_threads(old)


def oracle(model, ids, mask=None, positions=None, past=None):
    """Functional transcription of pinned Dream's equations; no modeling helpers/SDPA."""
    config, state = model.config, model.state_dict()
    x = F.embedding(ids, state["model.embed_tokens.weight"])
    dim = config.hidden_size // config.num_attention_heads
    offset = 0 if past is None else past[0][0].shape[-2]
    if positions is None:
        positions = torch.arange(offset, offset + ids.shape[1])[None]
    frequencies = positions[..., None].float() / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
    embedding = torch.cat([frequencies, frequencies], dim=-1)
    cosine, sine = embedding.cos()[:, None].to(x.dtype), embedding.sin()[:, None].to(x.dtype)

    def norm(value, weight):
        normalized = value.float() * (value.float().square().mean(-1, keepdim=True) + config.rms_norm_eps).rsqrt()
        return normalized.to(value.dtype) * weight

    def rotate(value):
        first, second = value.chunk(2, dim=-1)
        return value * cosine + torch.cat([-second, first], dim=-1) * sine

    present = []
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        hidden = norm(x, state[prefix + "input_layernorm.weight"])
        qkv = []
        for name, heads in [("q", config.num_attention_heads), ("k", config.num_key_value_heads), ("v", config.num_key_value_heads)]:
            value = F.linear(hidden, state[prefix + f"self_attn.{name}_proj.weight"], state[prefix + f"self_attn.{name}_proj.bias"])
            qkv.append(value.reshape(ids.shape[0], ids.shape[1], heads, dim).transpose(1, 2))
        q, k, v = rotate(qkv[0]), rotate(qkv[1]), qkv[2]
        if past is not None:
            k = torch.cat([past[index][0], k], dim=-2)
            v = torch.cat([past[index][1], v], dim=-2)
        present.append((k, v))
        k = k.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
        v = v.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
        scores = q @ k.transpose(-1, -2) / dim ** 0.5
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
        probabilities = torch.nan_to_num(scores.float().softmax(-1).to(q.dtype), nan=0.0)
        attention = (probabilities @ v).transpose(1, 2).reshape_as(hidden)
        x = x + F.linear(attention, state[prefix + "self_attn.o_proj.weight"])
        hidden = norm(x, state[prefix + "post_attention_layernorm.weight"])
        gate = F.silu(F.linear(hidden, state[prefix + "mlp.gate_proj.weight"]))
        up = F.linear(hidden, state[prefix + "mlp.up_proj.weight"])
        x = x + F.linear(gate * up, state[prefix + "mlp.down_proj.weight"])
    return F.linear(norm(x, state["model.norm.weight"]), state["lm_head.weight"]), present


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bidirectional_forward_matches_independent_oracle(implementation, kv_heads, dtype):
    config = tiny_config(num_key_value_heads=kv_heads)
    config._attn_implementation = implementation
    model = DreamModel(config).to(dtype).eval()
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]])
    keep = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.bool)
    mask = keep[:, None, :, None] & keep[:, None, None, :]
    positions = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2]])
    with torch.no_grad():
        output = model(ids, mask, positions, output_attentions=implementation == "eager")
        expected, _ = oracle(model, ids, mask, positions)
        tolerance = {torch.float32: 2e-6, torch.float16: 7e-4, torch.bfloat16: 5e-3}[dtype]
        torch.testing.assert_close(output.logits, expected, atol=tolerance, rtol=tolerance)
        assert output.logits.dtype == dtype
        changed = ids.clone()
        changed[:, -1] = 9
        assert not torch.allclose(model(changed, mask, positions).logits[:, 0], output.logits[:, 0], atol=1e-5)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_padding_mask_forms_and_failed_call_retry(implementation):
    config = tiny_config()
    config._attn_implementation = implementation
    model = DreamModel(config).eval()
    ids = torch.tensor([[1, 2, 3], [4, 0, 5]])
    keep = ids.ne(0)
    bool_mask = keep[:, None, None, :]
    additive = torch.zeros(2, 1, 1, 3).masked_fill(~bool_mask, float("-inf"))
    baseline = model(ids, keep).logits
    torch.testing.assert_close(baseline, model(ids, bool_mask).logits)
    torch.testing.assert_close(baseline, model(ids, additive).logits)
    with pytest.raises(ValueError, match="rank 2 or 4"):
        model(ids, torch.ones(2, 1, 3))
    torch.testing.assert_close(model(ids, keep).logits, baseline)
    torch.testing.assert_close(model(ids, "full").logits, model(ids).logits)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_explicit_cache_continuation_matches_pinned_incremental_semantics(implementation):
    config = tiny_config()
    config._attn_implementation = implementation
    model = DreamModel(config).eval()
    cache = DynamicCache()
    prefix, continuation = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])
    expected, old = oracle(model, prefix)
    torch.testing.assert_close(model(prefix, past_key_values=cache, use_cache=True).logits, expected, atol=2e-6, rtol=2e-6)
    assert cache.get_seq_length() == 3
    expected, new = oracle(model, continuation, past=old)
    output = model(continuation, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(output.logits, expected, atol=2e-6, rtol=2e-6)
    assert cache.get_seq_length() == 5
    for index, (keys, values) in enumerate(new):
        torch.testing.assert_close(cache.layers[index].keys, keys, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(cache.layers[index].values, values, atol=2e-6, rtol=2e-6)
    # Bidirectional old states cannot equal a recomputation with future context.
    assert not torch.allclose(output.logits, model(torch.cat([prefix, continuation], dim=1)).logits[:, -2:], atol=1e-5)
    cache.reset()
    torch.testing.assert_close(model(prefix, past_key_values=cache).logits, oracle(model, prefix)[0], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("algorithm", ["origin", "maskgit_plus", "topk_margin", "entropy"])
def test_diffusion_generation_shift_schedule_hooks_and_retry(algorithm):
    model = DreamModel(tiny_config(use_cache=True)).eval()
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    inputs = []

    def capture(module, args, kwargs):
        assert kwargs["use_cache"] is False
        inputs.append((args[0].clone(), args[1], args[2]))

    def logits_hook(step, x, logits):
        expected = oracle(model, x, inputs[-1][1], inputs[-1][2])[0]
        expected = torch.cat([expected[:, :1], expected[:, :-1]], dim=1)
        torch.testing.assert_close(logits, expected, atol=2e-6, rtol=2e-6)
        # Distinct deterministic choices by denoising step and sequence position.
        result = torch.full_like(logits, -1000.0)
        tokens = (torch.arange(x.shape[1]) + step + 6).expand(x.shape[0], -1)
        return result.scatter(-1, tokens[..., None], 1000.0)

    handle = model.register_forward_pre_hook(capture, with_kwargs=True)
    config = DreamGenerationConfig(max_new_tokens=4, steps=3, alg=algorithm,
                                   output_history=True, return_dict_in_generate=True)
    try:
        result = model.generate(input_ids=ids, attention_mask=ids.ne(0), generation_config=config,
                                generation_logits_hook_func=logits_hook)
    finally:
        handle.remove()
    assert len(inputs) == len(result.history) == 3
    assert result.sequences.shape == (2, 7)
    torch.testing.assert_close(result.sequences[:, :3], ids)
    assert not result.sequences[:, 3:].eq(30).any()
    # Forward hook failures must not contaminate the caller's reusable config/model.
    def fail(step, x, logits):
        if step == 1:
            raise RuntimeError("injected after a completed denoising step")
        return x
    torch.manual_seed(21)
    with pytest.raises(RuntimeError, match="injected"):
        model.generate(ids, generation_config=config, generation_tokens_hook_func=fail)
    torch.manual_seed(22)
    retry = model.generate(ids, generation_config=config)
    torch.manual_seed(22)
    fresh = model.diffusion_generate(ids, generation_config=config)
    torch.testing.assert_close(retry.sequences, fresh.sequences)
    assert config.max_length == 20 and config.mask_token_id is None


@pytest.mark.parametrize("tied", [False, True])
def test_safetensors_round_trip_and_direct_reload_equivalence(tmp_path, tied):
    model = DreamModel(tiny_config(tie_word_embeddings=tied)).eval()
    ids = torch.tensor([[1, 2, 3]])
    expected = model(ids).logits
    for _ in range(2):
        model.save_pretrained(tmp_path)
        assert list(tmp_path.glob("*.safetensors"))
        assert not list(tmp_path.glob("*.bin"))
        config_payload = json.loads((tmp_path / "config.json").read_text())
        config_payload["auto_map"] = {"AutoConfig": "evil.Config", "AutoModel": "evil.Model"}
        (tmp_path / "config.json").write_text(json.dumps(config_payload))
        model = DreamQModel.loader.from_pretrained(tmp_path, trust_remote_code=True).eval()
        torch.testing.assert_close(model(ids).logits, expected, atol=0, rtol=0)
        assert (model.lm_head.weight is model.model.embed_tokens.weight) is tied
        assert isinstance(model.generation_config, DreamGenerationConfig)
    assert model.generate(ids, max_new_tokens=2, steps=1).shape == (1, 5)


def test_rope_legacy_metadata_does_not_mutate_caller_and_linear_scaling_is_real():
    scaling = {"type": "linear", "factor": 2.0}
    model = DreamModel(tiny_config(rope_scaling=scaling)).eval()
    assert scaling == {"type": "linear", "factor": 2.0}
    regular = DreamModel(tiny_config()).eval()
    regular.load_state_dict(model.state_dict())
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = regular(ids, position_ids=torch.arange(4).float()[None] / 2).logits
    torch.testing.assert_close(model(ids).logits, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("mode", ["sample_probability", "margin", "negative_entropy"])
def test_positive_temperature_truncation_and_confidence_match_independent_probabilities(mode):
    logits = torch.tensor([[3., 2., 0., -1.], [-1., 0., 2., 3.]]).repeat(32, 1)
    # At temperature .6 / nucleus .85, only the two largest entries survive.
    # This explicitly assembled distribution is independent of the filtering helpers.
    pair = torch.softmax(torch.tensor([3., 2.]) / .6, dim=0)
    probabilities = torch.tensor([[pair[0], pair[1], 0., 0.], [0., 0., pair[1], pair[0]]]).repeat(32, 1)
    torch.manual_seed(37)
    expected_tokens = torch.distributions.Categorical(probs=probabilities).sample()
    torch.manual_seed(37)
    confidence, tokens = sample_tokens(logits, temperature=.6, top_p=.85, top_k=3,
                                       margin_confidence=mode == "margin", neg_entropy=mode == "negative_entropy")
    torch.testing.assert_close(tokens, expected_tokens)
    if mode == "margin":
        expected_confidence = (pair[0] - pair[1]).expand_as(confidence)
    elif mode == "negative_entropy":
        expected_confidence = (probabilities * (probabilities + 1e-10).log()).sum(-1)
    else:
        expected_confidence = probabilities.gather(-1, tokens[:, None]).squeeze(-1)
    torch.testing.assert_close(confidence, expected_confidence)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_checkpoint_load_uses_packaged_model_and_diffusion_generate(tmp_path, monkeypatch, trust, bypass):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import GPTQModel

    original = DreamModel(tiny_config()).eval()
    original.save_pretrained(tmp_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "hello": 1, "<unk>": 2})),
                                        pad_token="<pad>", unk_token="<unk>")
    tokenizer.save_pretrained(tmp_path)
    marker = tmp_path / "EXECUTED"
    (tmp_path / "evil.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote')\n")
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "evil.Config", "AutoModel": "evil.Model"}
    (tmp_path / "config.json").write_text(json.dumps(payload))
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert isinstance(loaded, DreamQModel)
    assert isinstance(loaded.model, DreamModel)
    ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(loaded.model(ids).logits, original(ids).logits, atol=0, rtol=0)
    torch.manual_seed(31)
    actual = loaded.generate(ids, max_new_tokens=3, steps=2)
    torch.manual_seed(31)
    expected = original.diffusion_generate(ids, max_new_tokens=3, steps=2)
    torch.testing.assert_close(actual, expected)
    assert loaded.module_plan().find("self_attn.v_proj").quantize
    assert not marker.exists()


def test_native_init_rebuilds_rope_buffers_after_load_and_checkpointing_matches(tmp_path):
    model = DreamModel(tiny_config(rope_scaling={"type": "linear", "factor": 2.0})).eval()
    dim = model.config.hidden_size // model.config.num_attention_heads
    expected = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim)) / 2.0
    model.save_pretrained(tmp_path)
    loaded = DreamQModel.loader.from_pretrained(tmp_path)
    rotaries = [loaded.model.rotary_emb, *(layer.self_attn.rotary_emb for layer in loaded.model.layers)]
    for rotary in rotaries:
        torch.testing.assert_close(rotary.inv_freq, expected, rtol=0, atol=0)
        torch.testing.assert_close(rotary.original_inv_freq, expected, rtol=0, atol=0)
    ids = torch.tensor([[1, 2, 3, 4]])
    loaded.train()
    loaded.model(ids).last_hidden_state.square().sum().backward()
    reference = {name: param.grad.clone() for name, param in loaded.named_parameters() if param.grad is not None}
    loaded.zero_grad(set_to_none=True)
    loaded.activation_checkpointing_enable()
    loaded.model(ids).last_hidden_state.square().sum().backward()
    for name, grad in reference.items():
        torch.testing.assert_close(dict(loaded.named_parameters())[name].grad, grad, rtol=0, atol=0)
