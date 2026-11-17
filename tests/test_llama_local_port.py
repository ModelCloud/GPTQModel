# SPDX-License-Identifier: Apache-2.0
"""Definition-owned Llama aliases must not consult checkpoint Python."""

import json
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code


NATIVE_TYPES = [
    "cohere",
    "cohere2",
    "gemma",
    "granite",
    "helium",
    "llama",
    "mistral",
    "olmo",
    "seed_oss",
    "smollm3",
    "stablelm",
]


def local_model(model_type="longllama", *, dtype=None, **overrides):
    from gptqmodel.models.ports.llama.dispatch import _config_class

    return LlamaQModel.loader.from_config(
        _config_class(model_type).from_dict(tiny_payload(model_type, **overrides)),
        **({"dtype": dtype} if dtype is not None else {}),
    ).eval()


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_payload(model_type, **overrides):
    payload = {
        "model_type": model_type,
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "max_position_embeddings": 8,
        "last_context_length": 4,
        "mem_layers": [0],
        "mem_dtype": "float32",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    payload.update(overrides)
    return payload


def hostile_config(path, model_type, **overrides):
    path.mkdir(exist_ok=True)
    payload = tiny_payload(model_type, **overrides)
    payload["auto_map"] = {
        "AutoConfig": "hostile.Config",
        "AutoModelForCausalLM": "hostile.Model",
    }
    (path / "config.json").write_text(json.dumps(payload))
    (path / "hostile.py").write_text("raise AssertionError('checkpoint Python executed')\n")


@pytest.mark.parametrize("model_type", NATIVE_TYPES + ["longllama", "stablelm_epoch", "yi"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_alias_ignores_hostile_code(tmp_path, monkeypatch, model_type, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    hostile_config(tmp_path, model_type)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert config.model_type == model_type
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    model = LlamaQModel.loader.from_config(config, trust_remote_code=trust)
    assert model.config.model_type == model_type
    with torch.inference_mode():
        logits = model(torch.tensor([[3, 7, 5]]), use_cache=False).logits
    assert logits.shape == (1, 3, 64)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_type", NATIVE_TYPES)
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_exact_native_alias_forward_cache_generation_reload(tmp_path, model_type, implementation):
    from transformers import AutoModelForCausalLM

    hostile_config(tmp_path, model_type, head_dim=8, attn_implementation=implementation)
    config = load_model_config(tmp_path, trust_remote_code=True)
    config._attn_implementation = implementation
    model = LlamaQModel.loader.from_config(config).eval()
    assert type(model) is AutoModelForCausalLM._model_mapping[type(config)]
    ids = torch.tensor([[3, 7, 5, 9], [0, 4, 8, 6]])
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    with torch.inference_mode():
        expected = model(ids, attention_mask=mask, output_hidden_states=True).logits
        first = model(ids[:, :2], attention_mask=mask[:, :2])
        continued = model(ids[:, 2:], attention_mask=mask, past_key_values=first.past_key_values).logits
        torch.testing.assert_close(continued, expected[:, 2:], atol=3e-6, rtol=3e-5)
        generation = model.generate(ids[:1], max_new_tokens=2, do_sample=False, eos_token_id=None)
    model.save_pretrained(tmp_path)
    loaded = LlamaQModel.loader.from_pretrained(
        tmp_path, config=load_model_config(tmp_path), attn_implementation=implementation
    ).eval()
    assert type(loaded) is type(model)
    assert set(loaded.state_dict()) == set(model.state_dict())
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids, attention_mask=mask).logits, expected, atol=3e-6, rtol=3e-5)
        assert torch.equal(loaded.generate(ids[:1], max_new_tokens=2, do_sample=False, eos_token_id=None), generation)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("model_type", ["longllama", "stablelm_epoch", "yi"])
def test_legacy_safetensors_replacement_and_retry(tmp_path, dtype, model_type):
    from safetensors.torch import load_file, save_file

    model = local_model(model_type, dtype=dtype)
    ids = torch.tensor([[3, 7, 5, 9, 4, 8, 6, 12, 3, 6, 5, 9]])
    with torch.inference_mode():
        expected = model(ids, use_cache=True).logits
    model.save_pretrained(tmp_path)
    payload = load_file(tmp_path / "model.safetensors")
    assert set(payload) == set(model.state_dict())
    loaded = LlamaQModel.loader.from_pretrained(tmp_path, config=load_model_config(tmp_path), dtype=dtype).eval()
    assert {p.dtype for p in loaded.parameters()} == {dtype}
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, expected, atol=0, rtol=0)
    # Malformed tensor payload fails before publishing a usable model; repairing
    # the same path and retrying must neither retain state nor alter default dtype.
    wrong = dict(payload)
    wrong["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(7, 7, dtype=dtype)
    save_file(wrong, tmp_path / "model.safetensors", metadata={"format": "pt"})
    default_dtype = torch.get_default_dtype()
    with pytest.raises((RuntimeError, ValueError), match="[Ss]ize mismatch|[Ss]hape mismatch|ignore_mismatched_sizes"):
        LlamaQModel.loader.from_pretrained(tmp_path, config=load_model_config(tmp_path), dtype=dtype)
    assert torch.get_default_dtype() == default_dtype
    replacement = dict(payload)
    replacement["lm_head.weight"] = payload["lm_head.weight"] + 0.2 * torch.randn_like(payload["lm_head.weight"])
    save_file(replacement, tmp_path / "model.safetensors", metadata={"format": "pt"})
    repaired = LlamaQModel.loader.from_pretrained(tmp_path, config=load_model_config(tmp_path), dtype=dtype).eval()
    torch.testing.assert_close(repaired.lm_head.weight, replacement["lm_head.weight"], atol=0, rtol=0)
    with torch.inference_mode():
        assert not torch.allclose(repaired(ids).logits, expected)
    assert not list(tmp_path.glob("*.bin"))


def _rotate_oracle(states, positions, base=10000):
    dim = states.shape[-1]
    angles = positions.to(states.dtype)[..., None] / base ** (torch.arange(0, dim, 2, dtype=states.dtype) / dim)
    angles = torch.cat((angles, angles), dim=-1)[:, None]
    first, second = states.chunk(2, dim=-1)
    return states * angles.cos() + torch.cat((-second, first), dim=-1) * angles.sin()


@pytest.mark.parametrize("positionals", [False, True])
@pytest.mark.parametrize("sdpa", [False, True])
@pytest.mark.parametrize("grouping", [None, (1, 2)])
def test_longllama_attention_memory_eviction_independent_oracle(positionals, sdpa, grouping):
    from gptqmodel.models.ports.llama.longllama_utils import LongLlamaMemCache

    torch.manual_seed(247)
    model = local_model(mem_positionals=positionals, torch_attention=sdpa, mem_attention_grouping=grouping).double()
    attention = model.model.layers[0].self_attn
    x = torch.randn(2, 3, 32, dtype=torch.float64)
    old_k, old_v = (torch.randn(2, 4, 6, 8, dtype=x.dtype) for _ in range(2))
    old_pos = torch.arange(6).view(1, 1, 6, 1).expand(2, 1, -1, -1)
    memory = LongLlamaMemCache(
        torch.randn(2, 4, 2, 8),
        torch.randn(2, 4, 2, 8),
        torch.zeros(2, 1, 2, 1, dtype=x.dtype),
    )
    memory.masks[0, :, 0] = torch.finfo(x.dtype).min
    positions = torch.arange(6, 9).expand(2, -1)
    mask = torch.zeros(2, 1, 3, 9, dtype=x.dtype)
    mask[..., 6:] = torch.full((3, 3), torch.finfo(x.dtype).min, dtype=x.dtype).triu(1)
    mask[1, :, :, 0] = torch.finfo(x.dtype).min
    q, k, v = [
        F.linear(x, projection.weight).view(2, 3, 4, 8).transpose(1, 2)
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    ]
    if positionals:
        q = _rotate_oracle(q, torch.arange(3).expand(2, -1))
        rotated_k = _rotate_oracle(k, torch.arange(3).expand(2, -1))
    else:
        rotated_k = k
    # Overflow moves the complete preceding local window into FoT memory;
    # neither ordinary Llama cache concatenation nor a sliding cache is valid.
    mem_k = torch.cat((memory.keys, old_k.float()), dim=-2).to(x.dtype)
    mem_v = torch.cat((memory.values, old_v.float()), dim=-2).to(x.dtype)
    mem_mask = torch.cat((memory.masks, mask[..., -1, :6, None]), dim=-2).transpose(-1, -2)
    keys = torch.cat((rotated_k, mem_k), dim=-2)
    values = torch.cat((v, mem_v), dim=-2)
    total_mask = torch.cat((mask[..., 6:], mem_mask.expand(-1, -1, 3, -1)), dim=-1)
    probs = torch.softmax(((q @ keys.transpose(-1, -2)) / 8**0.5 + total_mask).float(), dim=-1).to(x.dtype)
    expected = F.linear((probs @ values).transpose(1, 2).reshape(2, 3, 32), attention.o_proj.weight)
    with torch.inference_mode():
        actual, _, local, new_memory = attention(
            x,
            attention_mask=mask,
            position_ids=positions,
            past_key_value=(old_k, old_v, old_pos),
            use_cache=True,
            mem_cache=memory,
        )
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(local[0], k, atol=0, rtol=0)
    torch.testing.assert_close(new_memory.keys, mem_k.float(), atol=0, rtol=0)
    assert local[0].shape[-2] == 3 and new_memory.keys.shape[-2] == 8
    assert memory.keys.shape[-2] == 2, "caller-owned memory must not be changed"


@pytest.mark.parametrize("positionals", [False, True])
@pytest.mark.parametrize("continuation", [1, 4])
def test_long_memory_continuation_generation_and_failure_retry(positionals, continuation):
    from gptqmodel.models.ports.llama.modeling_longllama import LongLlamaCache

    torch.manual_seed(62)
    eager = local_model(mem_positionals=positionals, mem_attention_grouping=(1, 2))
    sdpa = deepcopy(eager)
    sdpa.config.torch_attention = True
    for layer in sdpa.model.layers:
        layer.self_attn.config = sdpa.config
    ids = torch.tensor([[3, 7, 5, 9, 4, 8, 6, 12, 3, 6, 5, 9], [0, 0, 3, 8, 6, 4, 5, 9, 6, 3, 5, 7]])
    mask = (ids != 0).long()
    with torch.inference_mode():
        first = eager(ids, attention_mask=mask, output_hidden_states=True)
        assert isinstance(first.past_key_values, LongLlamaCache)
        assert len(first.hidden_states) == 3 and first.hidden_states[0].shape == (2, 12, 32)
        assert first.past_key_values.get_seq_length() == 12
        snapshots = [[tensor.clone() for tensor in layer] for layer in first.past_key_values]
        suffix = torch.tensor([[5, 4, 8, 6], [9, 3, 4, 7]])[:, :continuation]
        full_mask = torch.cat((mask, torch.ones_like(suffix)), dim=-1)
        with pytest.raises(ValueError, match="Attention mask"):
            eager(suffix, past_key_values=first.past_key_values, attention_mask=torch.ones(2, 2))
        reference = eager(suffix, attention_mask=full_mask, past_key_values=first.past_key_values).logits
        observed = sdpa(suffix, attention_mask=full_mask, past_key_values=first.past_key_values).logits
        torch.testing.assert_close(observed, reference, atol=2e-6, rtol=3e-5)
        for actual_layer, expected_layer in zip(first.past_key_values, snapshots):
            for actual, expected in zip(actual_layer, expected_layer):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert eager.generate(ids[:1], max_new_tokens=3, num_beams=2, eos_token_id=None).shape == (1, 15)
        manual = ids[:1].clone()
        cache = None
        for _ in range(3):
            output = eager(manual if cache is None else manual[:, -1:], past_key_values=cache)
            manual = torch.cat((manual, output.logits[:, -1:].argmax(-1)), dim=-1)
            cache = output.past_key_values
        assert torch.equal(eager.generate(ids[:1], max_new_tokens=3, eos_token_id=None), manual)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("rope_pct", [0.25, 1.0])
def test_epoch_native_mapping_matches_independent_sequential_equations(kv_heads, rope_pct):
    torch.manual_seed(83)
    model = local_model(
        "stablelm_epoch",
        num_hidden_layers=1,
        num_key_value_heads=kv_heads,
        rope_pct=rope_pct,
        norm_eps=0.031,
        partial_rotary_factor=1.0,
        layer_norm_eps=0.2,
        use_parallel_residual=True,
        qk_layernorm=True,
        use_qkv_bias=True,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        hidden_act="gelu",
        rotary_scaling_factor=7.0,
    )
    assert model.config.rope_parameters["partial_rotary_factor"] == rope_pct
    layer = model.model.layers[0]
    with torch.no_grad():
        layer.input_layernorm.weight.copy_(torch.linspace(0.6, 1.4, 32))
        layer.input_layernorm.bias.copy_(torch.linspace(-0.3, 0.4, 32))
    ids = torch.tensor([[3, 7, 5, 9]])
    x = model.model.embed_tokens(ids)
    normed = F.layer_norm(x, (32,), layer.input_layernorm.weight, layer.input_layernorm.bias, 0.031)
    q = F.linear(normed, layer.self_attn.q_proj.weight).view(1, 4, 4, 8).transpose(1, 2)
    k = F.linear(normed, layer.self_attn.k_proj.weight).view(1, 4, kv_heads, 8).transpose(1, 2)
    v = F.linear(normed, layer.self_attn.v_proj.weight).view(1, 4, kv_heads, 8).transpose(1, 2)
    rotary = int(8 * rope_pct)
    positions = torch.arange(4).view(1, -1)
    q = torch.cat((_rotate_oracle(q[..., :rotary], positions), q[..., rotary:]), dim=-1)
    k = torch.cat((_rotate_oracle(k[..., :rotary], positions), k[..., rotary:]), dim=-1)
    k, v = (t.repeat_interleave(4 // kv_heads, dim=1) for t in (k, v))
    causal = torch.full((4, 4), -65504.0).triu(1)
    probs = ((q @ k.transpose(-1, -2)) / 8**0.5 + causal).float().softmax(-1)
    x = x + F.linear((probs @ v).transpose(1, 2).reshape(1, 4, 32), layer.self_attn.o_proj.weight)
    normed = F.layer_norm(x, (32,), layer.post_attention_layernorm.weight, layer.post_attention_layernorm.bias, 0.031)
    x = x + F.linear(
        F.silu(F.linear(normed, layer.mlp.gate_proj.weight)) * F.linear(normed, layer.mlp.up_proj.weight),
        layer.mlp.down_proj.weight,
    )
    x = F.layer_norm(x, (32,), model.model.norm.weight, model.model.norm.bias, 0.031)
    expected = F.linear(x, model.lm_head.weight)
    with torch.inference_mode():
        torch.testing.assert_close(model(ids).logits, expected, atol=5e-7, rtol=1e-5)
    assert model.config.model_type == "stablelm_epoch" and model.config.hidden_act == "silu"


def test_allowlist_and_incompatible_cache_fail_closed():
    from transformers import DynamicCache, LlamaConfig

    from gptqmodel.models.ports.llama.dispatch import LlamaAliasForCausalLM, _config_class

    with pytest.raises(ValueError, match="Unsupported Llama definition"):
        _config_class("unported_model")
    config = LlamaConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4)
    config.model_type = "longllama"
    with pytest.raises(ValueError, match="Expected package-selected"):
        LlamaAliasForCausalLM._from_config(config)
    model = local_model()
    with pytest.raises(ValueError, match="own six-tensor"):
        model(torch.tensor([[3, 7]]), past_key_values=DynamicCache())
    with pytest.raises(ValueError, match="MHA"):
        local_model(num_key_value_heads=2)


@pytest.mark.parametrize("model_type", ["stablelm_epoch", "yi"])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_legacy_native_alias_cache_and_beam_generation(model_type, heads, dtype):
    model = local_model(model_type, dtype=dtype, num_key_value_heads=heads)
    ids = torch.tensor([[3, 7, 5, 9], [0, 3, 6, 8]])
    mask = ids.ne(0).long()
    with torch.inference_mode():
        full = model(ids, attention_mask=mask).logits
        prefix = model(ids[:, :2], attention_mask=mask[:, :2])
        cached = model(ids[:, 2:], attention_mask=mask, past_key_values=prefix.past_key_values).logits
        atol, rtol = (3e-6, 3e-5) if dtype == torch.float32 else (2e-3, 2e-3)
        torch.testing.assert_close(cached, full[:, 2:], atol=atol, rtol=rtol)
        assert model.generate(ids, attention_mask=mask, max_new_tokens=2, num_beams=2, eos_token_id=None).shape == (
            2,
            6,
        )


@pytest.mark.parametrize("positionals", [False, True])
@pytest.mark.parametrize("sdpa", [False, True])
@pytest.mark.parametrize("mem_layers", [[], [0], [0, 1]])
def test_longllama_causality_padding_and_memory_is_used(positionals, sdpa, mem_layers):
    torch.manual_seed(52)
    model = local_model(mem_positionals=positionals, torch_attention=sdpa, mem_layers=mem_layers)
    ids = torch.tensor([[0, 0, 3, 9, 4, 8, 6, 12, 3, 6, 5, 9], [0, 3, 5, 9, 4, 8, 6, 12, 3, 6, 5, 9]])
    mask = ids.ne(0).long()
    with torch.inference_mode():
        reference = model(ids, attention_mask=mask).logits
        future = ids.clone()
        future[:, 10:] = 20
        observed = model(future, attention_mask=mask).logits
        torch.testing.assert_close(observed[:, :10], reference[:, :10], atol=0, rtol=0)
        padding = ids.clone()
        padding[~mask.bool()] = 18
        observed = model(padding, attention_mask=mask).logits
        torch.testing.assert_close(observed[mask.bool()], reference[mask.bool()], atol=0, rtol=0)
        different_prefix = ids.clone()
        different_prefix[:, 3:7] = 21
        observed = model(different_prefix, attention_mask=mask).logits
        if mem_layers:
            assert (observed[:, -4:] - reference[:, -4:]).abs().max() > 1e-7
        else:
            torch.testing.assert_close(observed[:, -4:], reference[:, -4:], atol=0, rtol=0)


def test_longllama_cache_batch_ownership_reset_and_crop_rejection():
    model = local_model()
    ids = torch.tensor([[3, 7, 5, 9, 4, 8, 6, 12, 3, 6, 5, 9], [4, 3, 5, 7, 6, 9, 3, 5, 6, 7, 4, 3]])
    with torch.inference_mode():
        first = model(ids).past_key_values
        cache = deepcopy(first)
        assert cache.batch_size == 2 and cache.is_initialized and not cache.is_croppable
        cache.batch_repeat_interleave(2)
        assert cache.batch_size == 4
        cache.batch_select_indices(torch.tensor([2, 0]))
        for old, new in zip(first, cache):
            for a, b in zip(old, new):
                torch.testing.assert_close(b, a.flip(0), atol=0, rtol=0)
        token = torch.tensor([[6], [5]])
        torch.testing.assert_close(
            model(token.flip(0), past_key_values=cache).logits,
            model(token, past_key_values=first).logits.flip(0),
            atol=0,
            rtol=0,
        )
        with pytest.raises(ValueError, match="cannot be rolled back"):
            cache.crop(1)
        cache.reset()
        assert cache.get_seq_length() == 0 and not cache.is_initialized
        torch.testing.assert_close(model(ids, past_key_values=cache).logits, model(ids).logits, atol=0, rtol=0)


@pytest.mark.parametrize("model_type", ["longllama", "stablelm_epoch", "yi", "llama"])
def test_subfolder_config_payload_and_save_retry(tmp_path, model_type):
    folder = tmp_path / "nested"
    hostile_config(folder, model_type)
    model = local_model(model_type)
    model.save_pretrained(folder)
    model_class = type(model)
    config = load_model_config(tmp_path, subfolder="nested", trust_remote_code=True)
    restored = LlamaQModel.loader.from_pretrained(tmp_path, subfolder="nested", config=config).eval()
    assert type(restored) is model_class
    with torch.inference_mode():
        ids = torch.tensor([[3, 7, 5]])
        torch.testing.assert_close(restored(ids).logits, model(ids).logits, atol=0, rtol=0)


def save_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(
        WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "a": 4, "b": 5, "c": 6, "d": 7}, unk_token="[UNK]")
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]"
    ).save_pretrained(path)


@pytest.mark.parametrize("model_type", NATIVE_TYPES + ["longllama", "stablelm_epoch", "yi"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_load_cannot_execute_hostile_maps(tmp_path, monkeypatch, model_type, trust, bypass):
    from gptqmodel import BACKEND, GPTQModel

    model = local_model(model_type)
    model.save_pretrained(tmp_path)
    save_tokenizer(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "hostile.Config", "AutoModelForCausalLM": "hostile.Model"}
    (tmp_path / "config.json").write_text(json.dumps(payload))
    (tmp_path / "hostile.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    wrapper = GPTQModel.load(
        str(tmp_path), backend=BACKEND.TORCH, device="cpu", dtype=torch.float32, trust_remote_code=trust
    )
    assert type(wrapper) is LlamaQModel
    assert type(wrapper.model) is type(model)
    for layer in wrapper.model.model.layers:
        for key in layer.state_dict():
            assert wrapper.module_plan().find(key.rsplit(".", 1)[0]) is not None
    with torch.inference_mode():
        ids = torch.tensor([[3, 7, 5]])
        torch.testing.assert_close(wrapper.model(ids).logits, model(ids).logits, atol=0, rtol=0)


@pytest.mark.parametrize(
    "model_type,kv_heads",
    [
        ("longllama", 4),
        ("stablelm_epoch", 4),
        ("stablelm_epoch", 2),
        ("stablelm_epoch", 1),
        ("yi", 4),
        ("cohere", 4),
        ("stablelm", 1),
        ("stablelm", 2),
        ("stablelm", 4),
    ],
)
def test_public_plain_awq_quantizes_every_linear_save_reload(tmp_path, model_type, kv_heads):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    source, output = tmp_path / "source", tmp_path / "packed"
    model = local_model(
        model_type,
        num_hidden_layers=1,
        num_key_value_heads=kv_heads,
        max_position_embeddings=16,
        last_context_length=16,
        use_qk_norm=model_type == "cohere",
        qk_layernorm=model_type == "stablelm",
    )
    model.save_pretrained(source)
    save_tokenizer(source)
    before = {k: v.clone() for k, v in model.state_dict().items() if ".weight" not in k or "norm" in k}
    wrapper = GPTQModel.load(
        str(source),
        quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
        backend=BACKEND.TORCH,
        dtype=torch.float16,
    )
    layer = wrapper.model.model.layers[0]
    assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    samples = [
        {"input_ids": torch.tensor([[4, 5, 6, 7, 5, 4, 7, 6]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)}
        for _ in range(2)
    ]
    wrapper.quantize(
        CalibrationConfig(dataset=samples, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    packed = {name: module for name, module in layer.named_modules() if isinstance(module, AwqTorchLinear)}
    assert set(packed) == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
    for key, value in before.items():
        expected_dtype = torch.float32 if key.endswith("inv_freq") else torch.float16
        torch.testing.assert_close(wrapper.model.state_dict()[key], value.to(expected_dtype), atol=0, rtol=0)
    with torch.inference_mode():
        logits = wrapper.model(**samples[0], use_cache=False).logits
    assert torch.isfinite(logits).all()
    wrapper.save(str(output))
    assert list(output.glob("*.safetensors")) and not list(output.glob("*.bin"))
    reloaded = GPTQModel.load(str(output), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert sum(isinstance(m, AwqTorchLinear) for m in reloaded.model.modules()) == 7
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(**samples[0], use_cache=False).logits, logits, atol=0, rtol=0)
    expected_state = {key: value.clone() for key, value in wrapper.model.state_dict().items()}
    for key, value in reloaded.model.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], atol=0, rtol=0)
    second = tmp_path / "resaved"
    reloaded.save(str(second))
    again = GPTQModel.load(str(second), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert set(again.model.state_dict()) == set(expected_state)
    for key, value in again.model.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], atol=0, rtol=0)
    with torch.inference_mode():
        torch.testing.assert_close(again.model(**samples[0], use_cache=False).logits, logits, atol=0, rtol=0)


@pytest.mark.parametrize("model_type", ["longllama", "stablelm_epoch", "yi"])
@pytest.mark.parametrize("assign", [False, True])
def test_state_dict_assignment_replaces_initialized_owners(model_type, assign):
    source, target = local_model(model_type), local_model(model_type)
    ids = torch.tensor([[3, 7, 5, 9, 4, 8, 6, 12, 3, 6, 5, 9]])
    with torch.inference_mode():
        expected = source(ids).logits
        assert not torch.allclose(target(ids).logits, expected)
    target.load_state_dict(source.state_dict(), assign=assign, strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(target(ids).logits, expected, atol=0, rtol=0)
