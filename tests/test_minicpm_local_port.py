# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real original MiniCPM execution, independent source equations and public lifecycle."""

import json
import math
import shutil

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, decoders, normalizers
from tokenizers.models import BPE, WordLevel
from torch.nn import functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.cache_utils import DynamicCache, StaticCache

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models.definitions.minicpm import MiniCPMGPTQ
from gptqmodel.models.ports.minicpm import MiniCPMConfig, MiniCPMForCausalLM
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig, ExpertsRoutingOverride, MoEConfig, QuantizeConfig
from gptqmodel.utils.hf import load_hf_tokenizer, load_model_config


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(**kwargs):
    values = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 16,
              "scale_emb": 3, "dim_model_base": 8, "scale_depth": 1.4, "initializer_range": 0.12,
              "tie_word_embeddings": False, "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 3,
              "attn_implementation": "eager"}
    values.update(kwargs)
    return MiniCPMConfig(**values)


def _model(**kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(912)
        model = MiniCPMForCausalLM(_config(**kwargs)).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith(".bias"):
                    parameter.copy_(torch.linspace(-0.08, 0.09, parameter.numel()))
    return model


def _checkpoint(path, **kwargs):
    path.mkdir(exist_ok=True)
    model = _model(**kwargs)
    model.save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "a": 4, "b": 5, "c": 6, "d": 7},
            unk_token="[UNK]")),
        pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]",
    )
    tokenizer.save_pretrained(path)
    marker = path / "EXECUTED"
    (path / "hostile.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint code executed')\n"
    )
    config = json.loads((path / "config.json").read_text())
    config["auto_map"] = {"AutoConfig": "hostile.Config", "AutoModelForCausalLM": "hostile.Model",
                          "AutoTokenizer": ["hostile.Tokenizer", "hostile.FastTokenizer"]}
    (path / "config.json").write_text(json.dumps(config))
    config = json.loads((path / "tokenizer_config.json").read_text())
    config["auto_map"] = {"AutoTokenizer": ["hostile.Tokenizer", "hostile.FastTokenizer"]}
    (path / "tokenizer_config.json").write_text(json.dumps(config))
    return model, marker


def _oracle(weights, c, ids, positions, padding=None, past=None, sdpa=False):
    """Functional OpenBMB equations, not port layers, RoPE, masks or cache helpers.

    Sources: 6011acb (dense/128k) and 7495286 (MoE); full pins in port NOTICE.
    The cache contains ROTATED keys; unlike InternLM1 it does not re-rotate history.
    """
    batch, length = ids.shape
    h = F.embedding(ids, weights["model.embed_tokens.weight"]) * c.scale_emb
    dim = c.hidden_size // c.num_attention_heads
    new_cache = []

    def linear(x, name, ignore_bias=False):
        return F.linear(x, weights[name + ".weight"], None if ignore_bias else weights.get(name + ".bias"))

    def norm(x, name):
        return (x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + c.rms_norm_eps)).to(x.dtype) * weights[name]

    def rotary(x, total):
        base = c.rope_theta
        params = c.rope_parameters
        kind = params["rope_type"]
        pos = positions.float()
        scale = 1.0
        if kind == "linear":
            pos = pos / params["factor"]
        elif kind == "dynamic" and total > c.max_position_embeddings:
            base = base * (params["factor"] * total / c.max_position_embeddings - params["factor"] + 1) ** (dim / (dim - 2))
        elif kind == "longrope":
            original = params["original_max_position_embeddings"]
            factors = params["long_factor"] if total > original else params["short_factor"]
            ratio = c.max_position_embeddings / original
            if ratio > 1:
                scale = math.sqrt(1 + math.log(ratio) / math.log(original))
        freq = base ** (-torch.arange(0, dim, 2).float() / dim)
        if kind == "longrope":
            freq = freq / torch.tensor(factors, dtype=freq.dtype)
        angles = pos[..., None] * freq
        trig_dtype = torch.float32 if kind == "longrope" or c._attn_implementation == "eager" else x.dtype
        cos = angles.cos()[:, None].to(trig_dtype) * scale
        sin = angles.sin()[:, None].to(trig_dtype) * scale
        left, right = x.float().chunk(2, dim=-1)
        return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1).to(x.dtype)

    def mlp(x, prefix, expert=False):
        a, b, d = ("w1", "w3", "w2") if expert else ("gate_proj", "up_proj", "down_proj")
        return linear(F.silu(linear(x, prefix + a)) * linear(x, prefix + b), prefix + d)

    for index in range(c.num_hidden_layers):
        p = f"model.layers.{index}."
        normalized = norm(h, p + "input_layernorm.weight")
        q, k, v = [linear(normalized, p + "self_attn." + name, c.pretraining_tp > 1).reshape(
            batch, length, heads, dim).transpose(1, 2)
            for name, heads in (("q_proj", c.num_attention_heads), ("k_proj", c.num_key_value_heads),
                                ("v_proj", c.num_key_value_heads))]
        old_length = 0 if past is None else past[index][0].shape[-2]
        q, k = rotary(q, old_length + length), rotary(k, old_length + length)
        if past is not None:
            k, v = torch.cat((past[index][0], k), dim=-2), torch.cat((past[index][1], v), dim=-2)
        new_cache.append((k, v))
        # Explicit head selection is independent of the port's repeat_interleave layout.
        mapping = torch.arange(c.num_attention_heads) // (c.num_attention_heads // c.num_key_value_heads)
        scores = torch.matmul(q, k[:, mapping].transpose(-1, -2)) * dim ** -0.5
        allowed = torch.arange(old_length + length)[None, :] <= torch.arange(old_length, old_length + length)[:, None]
        allowed = allowed[None, None].expand(batch, 1, -1, -1)
        if padding is not None:
            allowed = allowed & padding[:, None, None].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        if sdpa:
            out = F.scaled_dot_product_attention(q, k[:, mapping], v[:, mapping], attn_mask=allowed)
        else:
            out = torch.matmul(scores.softmax(-1, dtype=torch.float32).to(q.dtype), v[:, mapping])
        attention = linear(out.transpose(1, 2).reshape(batch, length, -1), p + "self_attn.o_proj", c.pretraining_tp > 1)
        h = h + attention * (c.scale_depth * c.num_hidden_layers ** -0.5)
        normalized = norm(h, p + "post_attention_layernorm.weight")
        if c.num_experts:
            flat = normalized.reshape(-1, c.hidden_size)
            probabilities = linear(flat, p + "mlp.gate").float().softmax(-1)
            probabilities, selected = probabilities.topk(c.num_experts_per_tok, dim=-1)
            probabilities = (probabilities / probabilities.sum(-1, keepdim=True)).to(flat.dtype)
            # Evaluate every expert then select per token, rather than scatter accumulating.
            all_experts = torch.stack([mlp(flat, p + f"mlp.experts.{i}.", True) for i in range(c.num_experts)], 1)
            chosen = all_experts[torch.arange(flat.shape[0])[:, None], selected]
            feedforward = (chosen * probabilities[..., None]).sum(1).view_as(h)
        else:
            feedforward = mlp(normalized, p + "mlp.")
        h = h + feedforward * (c.scale_depth * c.num_hidden_layers ** -0.5)
    h = norm(h, "model.norm.weight")
    if c.pretraining_tp == 1:
        h = h * (c.dim_model_base / c.hidden_size)
    return linear(h, "lm_head").float(), new_cache


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_public_loader_ignores_all_checkpoint_code(tmp_path, monkeypatch, trust, bypass, moe, legacy):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    source, marker = _checkpoint(tmp_path, num_experts=3 if moe else 0, num_experts_per_tok=2 if moe else 0)
    if legacy:
        payload = json.loads((tmp_path / "config.json").read_text())
        payload.pop("model_type")
        payload["transformers_version"] = "4.36.0"
        (tmp_path / "config.json").write_text(json.dumps(payload))
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is MiniCPMConfig
    loaded = GPTQModel.load(str(tmp_path), quantize_config=None, trust_remote_code=trust,
                            device="cpu", dtype=torch.float32, attn_implementation="eager")
    assert isinstance(loaded, MiniCPMGPTQ)
    assert type(loaded.model) is MiniCPMForCausalLM
    assert type(loaded.model.config) is MiniCPMConfig
    ids = torch.tensor([[2, 4, 7, 6]])
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(ids).logits, source(ids).logits, atol=0, rtol=0)
    for key, tensor in loaded.model.state_dict().items():
        torch.testing.assert_close(tensor, source.state_dict()[key], atol=0, rtol=0)
    assert loaded.trust_remote_code is False
    assert not marker.exists()


def test_batched_gptq_save_and_reload_ignores_checkpoint_python(tmp_path):
    source = tmp_path / "source"
    _, marker = _checkpoint(source, num_hidden_layers=1)
    model = GPTQModel.load(
        str(source),
        quantize_config=QuantizeConfig(bits=4, group_size=16, device="cpu", offload_to_disk=False),
        trust_remote_code=True,
        backend=BACKEND.TORCH,
        dtype=torch.float16,
    )
    ids = torch.tensor([[2, 4, 7, 5, 6, 4, 5, 7]])
    calibration = [
        {"input_ids": item, "attention_mask": torch.ones_like(item)}
        for item in (ids, ids.flip(-1)[:, :6])
    ]
    model.quantize(CalibrationConfig(dataset=calibration, batch_size=2, min_length=1), backend=BACKEND.TORCH)
    packed = {name: tensor.clone() for name, tensor in model.model.state_dict().items() if name.endswith(".qweight")}
    assert len(packed) == 7
    with torch.inference_mode():
        expected = model.model(ids, use_cache=False).logits
    assert torch.isfinite(expected).all()

    destination = tmp_path / "quantized"
    model.save(str(destination))
    assert list(destination.glob("*.safetensors"))
    assert not list(destination.rglob("*.py"))
    assert not json.loads((destination / "config.json").read_text()).get("auto_map")

    reloaded = GPTQModel.load(
        str(destination), device="cpu", backend=BACKEND.TORCH, dtype=torch.float16, trust_remote_code=True
    )
    assert type(reloaded.model) is MiniCPMForCausalLM
    assert reloaded.trust_remote_code is False
    state = reloaded.model.state_dict()
    for name, tensor in packed.items():
        torch.testing.assert_close(state[name], tensor, atol=0, rtol=0)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(ids, use_cache=False).logits, expected, atol=0, rtol=0)
    assert not marker.exists()


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("architectures", [["MiniCPMForCausalLM", "ForeignForCausalLM"],
                                           ["MiniCPMForCausalLM", None]])
def test_legacy_architecture_conflict_fails_before_code_and_repairs(tmp_path, monkeypatch, moe, architectures):
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    source, marker = _checkpoint(tmp_path, num_experts=3 if moe else 0, num_experts_per_tok=2 if moe else 0)
    path = tmp_path / "config.json"
    payload = json.loads(path.read_text())
    payload.pop("model_type")
    path.write_text(json.dumps(payload))
    first = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32,
                           trust_remote_code=True, attn_implementation="eager")
    broken = dict(payload, architectures=architectures)
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="architectures"):
        load_model_config(tmp_path, trust_remote_code=True)
    with pytest.raises(ValueError, match="architectures"):
        GPTQModel.load(str(tmp_path), device="cpu", trust_remote_code=True)
    assert not marker.exists()
    path.write_text(json.dumps(payload))
    repaired = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32,
                              trust_remote_code=True, attn_implementation="eager")
    assert type(repaired.model) is MiniCPMForCausalLM
    ids = torch.tensor([[2, 4, 7, 6]])
    with torch.no_grad():
        expected = source(ids).logits
        torch.testing.assert_close(first.model(ids).logits, expected, atol=0, rtol=0)
        torch.testing.assert_close(repaired.model(ids).logits, expected, atol=0, rtol=0)
    assert not marker.exists()


def test_legacy_sentencepiece_bpe_is_not_rewritten_as_mistral_bytelevel(tmp_path):
    vocab = {token: i for i, token in enumerate(["<unk>", "<s>", "</s>", "▁", "H", "e", "l", "o"])}
    for value in range(256):
        vocab[f"<0x{value:02X}>"] = len(vocab)
    for value in range(100_001 - len(vocab)):
        vocab[f"unused{value}"] = len(vocab)
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>", byte_fallback=True))
    backend.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    backend.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(),
                                         decoders.Strip(" ", 1, 0)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    tokenizer.save_pretrained(tmp_path)
    metadata = json.loads((tmp_path / "tokenizer_config.json").read_text())
    metadata.update(tokenizer_class="LlamaTokenizer", legacy=True)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    config = _config(vocab_size=len(vocab))
    source = MiniCPMForCausalLM(config).eval()
    source.save_pretrained(tmp_path, safe_serialization=True)
    payload = config.to_dict()
    payload.pop("model_type")
    payload["transformers_version"] = "4.36.0"
    (tmp_path / "config.json").write_text(json.dumps(payload))
    text = "Hello GPT-QModel! 你好，世界。"
    expected = backend.encode(text, add_special_tokens=False).ids
    native = AutoTokenizer.from_pretrained(tmp_path, config=config, trust_remote_code=False)
    assert native.encode(text, add_special_tokens=False) == expected
    loaded = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=False)
    assert loaded.encode(text, add_special_tokens=False) == expected
    public = GPTQModel.load(str(tmp_path), quantize_config=None, device="cpu", dtype=torch.float32,
                           trust_remote_code=False, attn_implementation="eager")
    assert public.tokenizer.encode(text, add_special_tokens=False) == expected
    ids = torch.tensor([expected])
    with torch.no_grad():
        torch.testing.assert_close(public.model(ids).logits, source(ids).logits, atol=0, rtol=0)


@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("rope", [None, "linear", "dynamic"])
def test_checkpoint_equations_mha_gqa_mqa_bias_scaled_dense_and_moe(kv, bias, moe, rope):
    model = _model(num_key_value_heads=kv, attention_bias=bias,
                   num_experts=3 if moe else 0, num_experts_per_tok=2 if moe else 0,
                   rope_scaling=None if rope is None else {"type": rope, "factor": 2.0}, max_position_embeddings=4)
    ids = torch.tensor([[2, 4, 7, 5, 6, 8], [2, 9, 4, 6, 7, 5]])
    padding = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    positions = padding.cumsum(-1) - 1
    positions.masked_fill_(padding == 0, 1)
    with torch.no_grad():
        expected, _ = _oracle(model.state_dict(), model.config, ids, positions, padding)
        actual = model(ids, attention_mask=padding, position_ids=positions, output_attentions=True,
                        output_hidden_states=True)
        torch.testing.assert_close(actual.logits[:, 1:], expected[:, 1:], atol=4e-7, rtol=2e-5)
        assert len(actual.hidden_states) == 3
        assert len(actual.attentions) == 2
        assert actual.attentions[0].shape == (2, 4, 6, 6)


@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("moe", [False, True])
def test_native_sdpa_matches_independent_fp32_attention_equations(kv, moe):
    model = _model(num_key_value_heads=kv, num_experts=3 if moe else 0,
                   num_experts_per_tok=2 if moe else 0, attn_implementation="sdpa")
    ids = torch.tensor([[2, 4, 7, 5, 6, 8]])
    with torch.no_grad():
        expected, _ = _oracle(model.state_dict(), model.config, ids, torch.arange(6)[None])
        actual = model(ids).logits
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=2e-5)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("rope", [None, "dynamic"])
def test_modern_cache_generation_causality_and_safetensors_roundtrip(tmp_path, monkeypatch, implementation, dtype, moe, rope):
    def removed_cache_api(_):
        raise AssertionError("legacy seen_tokens cache API was read")

    monkeypatch.setattr(DynamicCache, "seen_tokens", property(removed_cache_api), raising=False)
    model = _model(attn_implementation=implementation, num_experts=3 if moe else 0,
                   num_experts_per_tok=2 if moe else 0,
                   rope_scaling=None if rope is None else {"type": rope, "factor": 2.0}).to(dtype)
    ids = torch.tensor([[2, 4, 7, 5, 6, 8]])
    tolerance = 8e-7 if dtype == torch.float32 else 0.003
    with torch.no_grad():
        expected, _ = _oracle(model.state_dict(), model.config, ids, torch.arange(6)[None], sdpa=implementation == "sdpa")
        full = model(ids, use_cache=False).logits
        torch.testing.assert_close(full, expected, atol=tolerance, rtol=tolerance)
        cache, oracle_cache = None, None
        for start, end in ((0, 1), (1, 5), (5, 6)):
            expected, oracle_cache = _oracle(model.state_dict(), model.config, ids[:, start:end],
                                             torch.arange(start, end)[None], past=oracle_cache, sdpa=implementation == "sdpa")
            output = model(ids[:, start:end], past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            assert isinstance(cache, DynamicCache)
            assert cache.get_seq_length() == end
            torch.testing.assert_close(output.logits, expected, atol=tolerance, rtol=tolerance)
            for i, (key, value) in enumerate(oracle_cache):
                torch.testing.assert_close(cache.layers[i].keys, key, atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(cache.layers[i].values, value, atol=tolerance, rtol=tolerance)
        perturbed = ids.clone()
        perturbed[:, -1] = 10
        torch.testing.assert_close(model(perturbed).logits[:, :-1], full[:, :-1], atol=tolerance, rtol=tolerance)
        for beams in (1, 3):
            generated = model.generate(ids[:, :3], max_new_tokens=3, num_beams=beams, do_sample=False, eos_token_id=None)
            uncached = model.generate(ids[:, :3], max_new_tokens=3, num_beams=beams, do_sample=False,
                                      use_cache=False, eos_token_id=None)
            assert torch.equal(generated, uncached)
        for cycle in range(2):
            destination = tmp_path / str(cycle)
            model.save_pretrained(destination)
            assert list(destination.glob("*.safetensors"))
            assert not list(destination.glob("*.bin"))
            assert not list(destination.glob("*.py"))
            loaded = MiniCPMGPTQ.loader.from_pretrained(destination, dtype=dtype, attn_implementation=implementation)
            assert set(loaded.state_dict()) == set(model.state_dict())
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], tensor, atol=0, rtol=0)
            torch.testing.assert_close(loaded(ids).logits, full, atol=tolerance, rtol=tolerance)
            model = loaded


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dynamic_ntk_crossing_context_matches_rotated_key_source_and_resets(dtype):
    model = _model(max_position_embeddings=4, rope_scaling={"type": "dynamic", "factor": 2.0}).to(dtype)
    ids = torch.tensor([[2, 4, 7, 6, 5, 9, 8]])
    tolerance = 5e-7 if dtype == torch.float32 else 0.003
    cache, reference = None, None
    with torch.no_grad():
        for start, end in ((0, 3), (3, 5), (5, 7)):
            expected, reference = _oracle(model.state_dict(), model.config, ids[:, start:end],
                                          torch.arange(start, end)[None], past=reference)
            output = model(ids[:, start:end], past_key_values=cache)
            cache = output.past_key_values
            torch.testing.assert_close(output.logits, expected, atol=tolerance, rtol=tolerance)
        expected, _ = _oracle(model.state_dict(), model.config, ids[:, :3], torch.arange(3)[None])
        torch.testing.assert_close(model(ids[:, :3]).logits, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_minicpm4_longrope_short_long_factors_and_crossing_cache(implementation, dtype):
    model = _model(
        attn_implementation=implementation,
        max_position_embeddings=16,
        rope_scaling={
            "rope_type": "longrope",
            "short_factor": [1.0, 1.25, 1.5, 2.0],
            "long_factor": [2.0, 2.5, 3.0, 4.0],
            "original_max_position_embeddings": 8,
            "factor": 1.0,
        },
    ).to(dtype)
    ids = torch.tensor([[2, 4, 7, 5, 6, 8, 4, 7, 5, 6]])
    tolerance = 5e-7 if dtype == torch.float32 else 0.003
    with torch.no_grad():
        for length in (6, 10):
            expected, _ = _oracle(
                model.state_dict(), model.config, ids[:, :length], torch.arange(length)[None],
                sdpa=implementation == "sdpa",
            )
            torch.testing.assert_close(model(ids[:, :length]).logits, expected, atol=tolerance, rtol=tolerance)
        # The first six tokens cache short-factor keys; the continuation uses
        # long factors, matching the source cache semantics at the threshold.
        expected, past = _oracle(
            model.state_dict(), model.config, ids[:, :6], torch.arange(6)[None],
            sdpa=implementation == "sdpa",
        )
        first = model(ids[:, :6], use_cache=True)
        torch.testing.assert_close(first.logits, expected, atol=tolerance, rtol=tolerance)
        expected, _ = _oracle(
            model.state_dict(), model.config, ids[:, 6:], torch.arange(6, 10)[None], past=past,
            sdpa=implementation == "sdpa",
        )
        rest = model(ids[:, 6:], past_key_values=first.past_key_values, use_cache=True)
        torch.testing.assert_close(rest.logits, expected, atol=tolerance, rtol=tolerance)


def test_minicpm_stored_head_dim_reaches_attention_projections():
    config = _config(head_dim=8)
    assert config.head_dim == 8
    with torch.device("meta"):
        model = MiniCPMForCausalLM(config)
    attention = model.model.layers[0].self_attn
    assert attention.q_proj.out_features == config.num_attention_heads * config.head_dim
    assert attention.k_proj.out_features == config.num_key_value_heads * config.head_dim
    with pytest.raises(ValueError, match="head_dim"):
        _config(head_dim=4)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_legacy_pretraining_tp_keeps_its_distinct_head_scaling(tp):
    model = _model(pretraining_tp=tp, attention_bias=True)
    ids = torch.tensor([[2, 4, 7, 5]])
    with torch.no_grad():
        expected, _ = _oracle(model.state_dict(), model.config, ids, torch.arange(4)[None])
        # tp=4 sums four FP32 partial products in a different order than the oracle.
        torch.testing.assert_close(model(ids).logits, expected, atol=1e-6, rtol=2e-5)


def test_eager_attentions_keep_the_causal_padding_mask():
    # SDPA does not materialize attention probabilities; recorded attentions come from eager.
    model = _model(attn_implementation="eager")
    ids = torch.tensor([[2, 4, 7, 5], [0, 2, 6, 4]])
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    with torch.no_grad():
        plain = model(ids, attention_mask=mask).logits
        traced = model(ids, attention_mask=mask, output_attentions=True)
    # Fully padded query rows are undefined; compare every attended position.
    torch.testing.assert_close(traced.logits[mask.bool()], plain[mask.bool()], atol=2e-6, rtol=1e-5)
    for weights in traced.attentions:
        future = torch.ones(4, 4, dtype=torch.bool).triu(1)
        assert torch.all(weights[0][:, future] == 0)
        assert torch.all(weights[1, :, 1:][:, future[1:]] == 0)
        assert torch.all(weights[1, :, 1:, 0] == 0)


def test_inputs_embeds_are_already_scaled_and_loss_tuple_contract():
    model = _model()
    ids = torch.tensor([[2, 4, 7, 5]])
    with torch.no_grad():
        ordinary = model(ids, labels=ids)
        embedded = model(inputs_embeds=model.get_input_embeddings()(ids) * model.config.scale_emb, labels=ids)
        torch.testing.assert_close(ordinary.logits, embedded.logits, atol=0, rtol=0)
        torch.testing.assert_close(ordinary.loss, F.cross_entropy(ordinary.logits[:, :-1].reshape(-1, 64),
                                                                 ids[:, 1:].reshape(-1)))
        assert model(ids, return_dict=False)[0].shape == ordinary.logits.shape
        assert model(ids, logits_to_keep=2).logits.shape == (1, 2, 64)
        assert model(ids, logits_to_keep=torch.tensor([0, 2])).logits.shape == (1, 2, 64)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_eager_fp32_rotary_trig_reaches_first_layer_cache_without_narrowing(dtype):
    model = _model(num_hidden_layers=1).to(dtype)
    ids = torch.tensor([[2, 4, 7, 5, 6, 8, 9]])
    with torch.no_grad():
        _, expected_cache = _oracle(model.state_dict(), model.config, ids, torch.arange(7)[None])
        actual = model(ids).past_key_values.layers[0]
    torch.testing.assert_close(actual.keys, expected_cache[0][0], atol=0, rtol=0)
    torch.testing.assert_close(actual.values, expected_cache[0][1], atol=0, rtol=0)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tied_checkpoint_public_load_and_save_keep_one_exact_owner(tmp_path, moe, dtype):
    source, marker = _checkpoint(tmp_path / "source", tie_word_embeddings=True,
                                  num_experts=3 if moe else 0, num_experts_per_tok=2 if moe else 0)
    ids = torch.tensor([[2, 4, 7, 5]])
    source = source.to(dtype)
    for destination in (tmp_path / "source", tmp_path / "again"):
        if destination.name == "again":
            source.save_pretrained(destination)
            for asset in (tmp_path / "source").glob("*token*.json"):
                shutil.copyfile(asset, destination / asset.name)
        loaded = GPTQModel.load(str(destination), device="cpu", dtype=dtype, attn_implementation="eager")
        assert loaded.model.lm_head.weight is loaded.model.model.embed_tokens.weight
        for name, value in loaded.model.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[name], atol=0, rtol=0)
        with torch.no_grad():
            # Weights are bit-identical above; safetensors-mapped storage may
            # select a different CPU GEMM path and move FP32 logits by one ulp.
            tolerance = 1e-6 if dtype == torch.float32 else 0
            torch.testing.assert_close(loaded.model(ids).logits, source(ids).logits, atol=tolerance, rtol=tolerance)
        source = loaded.model
    assert not marker.exists()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_subfolder_bad_safetensors_repair_and_same_path_retry(tmp_path, dtype):
    nested = tmp_path / "nested"
    source, marker = _checkpoint(nested, rope_scaling={"type": "linear", "factor": 2.0})
    weights = load_file(nested / "model.safetensors")
    original = (nested / "model.safetensors").read_bytes()
    (nested / "model.safetensors").write_bytes(b"corrupt safetensors")
    with pytest.raises(Exception, match="[Hh]eader|[Ss]afetensor"):
        GPTQModel.load(str(tmp_path), subfolder="nested", device="cpu", dtype=dtype)
    (nested / "model.safetensors").write_bytes(original)
    loaded = GPTQModel.load(str(tmp_path), subfolder="nested", device="cpu", dtype=dtype, attn_implementation="eager")
    ids = torch.tensor([[2, 4, 5]])
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(ids).logits, source.to(dtype)(ids).logits, atol=0, rtol=0)
    for name, tensor in loaded.model.state_dict().items():
        torch.testing.assert_close(tensor, weights[name].to(dtype), atol=0, rtol=0)
    assert loaded.model.config.rope_parameters["rope_type"] == "linear"
    assert not marker.exists()


def test_invalid_cache_rejection_leaves_live_cache_usable():
    model = _model()
    ids = torch.tensor([[2, 4, 5]])
    with torch.no_grad():
        first = model(ids[:, :2])
        cache = first.past_key_values
        before = [layer.keys.clone() for layer in cache.layers]
        with pytest.raises(ValueError, match="use_cache"):
            model(ids[:, 2:], past_key_values=cache, use_cache=False)
        for i, tensor in enumerate(before):
            torch.testing.assert_close(cache.layers[i].keys, tensor, atol=0, rtol=0)
        with pytest.raises(ValueError, match="DynamicCache"):
            model(ids, past_key_values=StaticCache(config=model.config, max_cache_len=8))
        with pytest.raises(ValueError, match="DynamicCache"):
            model(ids, past_key_values=())
        continued = model(ids[:, 2:], past_key_values=cache)
        torch.testing.assert_close(continued.logits, model(ids).logits[:, 2:], atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize("moe", [False, True])
# "0.5%" is the default RTN fallback threshold. The earlier "plain" value was
# never parsed while the override emitted no scale groups, and is not a valid
# threshold.
@pytest.mark.parametrize("fallback", [None, "0.5%"])
@pytest.mark.parametrize("all_experts", [False, True])
def test_public_awq_scale_search_pack_infer_two_safetensors_cycles(tmp_path, monkeypatch, moe, fallback, all_experts):
    from gptqmodel.looper import awq_processor

    source, marker = _checkpoint(tmp_path / "source", num_hidden_layers=1,
                                  num_experts=3 if moe else 0, num_experts_per_tok=2 if moe else 0)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=fallback, offload_to_disk=False,
                     moe=MoEConfig(routing=ExpertsRoutingOverride()) if all_experts and moe else None)
    model = GPTQModel.load(str(tmp_path / "source"), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    applied = []
    original_apply_scale = awq_processor.apply_scale

    def record_apply_scale(module, scales_list, *args, **kwargs):
        applied.extend((source, tuple(targets)) for source, targets, _scales, _loss in scales_list)
        return original_apply_scale(module, scales_list, *args, **kwargs)

    monkeypatch.setattr(awq_processor, "apply_scale", record_apply_scale)
    ids = torch.tensor([[4, 5, 6, 7, 4, 7, 6, 5]])
    calibration = [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for _ in range(2)]
    dense = {name: value.to(torch.float16).clone() for name, value in source.state_dict().items()}
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    expected_linears = 4 + (9 if moe else 3)
    assert sum(isinstance(m, AwqTorchLinear) for m in model.model.modules()) == expected_linears
    # Scale search ran for every branch (GQA kv=2 leaves v_proj -> o_proj out).
    attention = [("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))]
    if moe:
        mlp = [("post_attention_layernorm", tuple(f"mlp.experts.{i}.{n}" for i in range(3) for n in ("w1", "w3")))]
        mlp += [(f"mlp.experts.{i}.w3", (f"mlp.experts.{i}.w2",)) for i in range(3)]
    else:
        mlp = [("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")), ("mlp.up_proj", ("mlp.down_proj",))]
    assert applied == attention + mlp
    state = {name: value.clone() for name, value in model.model.state_dict().items()}
    assert sum(name.endswith(".qweight") for name in state) == expected_linears
    if moe:
        assert model.model.model.layers[0].mlp.num_experts_per_tok == 2
        # The unquantized router stays dense and absorbs the inverse of the
        # post-attention norm fold, so the routed logits are unchanged.
        norm = "model.layers.0.post_attention_layernorm.weight"
        router = "model.layers.0.mlp.gate.weight"
        torch.testing.assert_close((state[router] * state[norm]).float(), (dense[router] * dense[norm]).float(),
                                   atol=2e-3, rtol=2e-3)
    for name in ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"):
        torch.testing.assert_close(state[name], dense[name], atol=0, rtol=0)
    with torch.no_grad():
        expected = model.model(ids).logits
        assert torch.isfinite(expected).all()
        for cycle in range(2):
            destination = tmp_path / f"packed-{cycle}"
            model.save(str(destination))
            assert list(destination.glob("*.safetensors"))
            assert not list(destination.glob("*.bin"))
            model = GPTQModel.load(str(destination), device="cpu", backend=BACKEND.TORCH, dtype=torch.float16)
            assert set(model.model.state_dict()) == set(state)
            for name, value in model.model.state_dict().items():
                torch.testing.assert_close(value, state[name], atol=0, rtol=0)
            torch.testing.assert_close(model.model(ids).logits, expected, atol=0, rtol=0)
            assert model.model.generate(ids[:, :3], max_new_tokens=2, do_sample=False).shape[-1] == 5
    assert not marker.exists()
