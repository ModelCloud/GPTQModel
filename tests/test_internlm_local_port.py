# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""InternLM's checkpoint Python cannot select its implementation."""

import copy
import json

import pytest
import torch
from torch.nn import functional as F

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.internlm import InternLMQModel
from gptqmodel.utils.hf import load_model_config


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_internlm_uses_package_config_despite_checkpoint_auto_map(tmp_path, monkeypatch, trust, bypass):
    monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    if bypass:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    marker = tmp_path / "checkpoint_python_executed"
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "internlm", "hidden_size": 32, "intermediate_size": 48,
        "num_attention_heads": 4, "num_hidden_layers": 2, "vocab_size": 37,
        "auto_map": {"AutoConfig": "checkpoint_config.UntrustedConfig",
                     "AutoModelForCausalLM": "checkpoint_config.UntrustedModel"},
    }))
    (tmp_path / "checkpoint_config.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python executed')\n"
    )
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config).__module__ == "gptqmodel.models.ports.internlm.configuration_internlm"
    assert not marker.exists()


def _config(**kwargs):
    from gptqmodel.models.ports.internlm.configuration_internlm import InternLMConfig
    values = {"vocab_size": 37, "hidden_size": 32, "intermediate_size": 48,
              "num_attention_heads": 4, "num_hidden_layers": 2,
              "max_position_embeddings": 16, "attn_implementation": "eager"}
    values.update(kwargs)
    return InternLMConfig(**values)


def _model(**kwargs):
    from gptqmodel.models.ports.internlm.modeling_internlm import InternLMForCausalLM
    torch.manual_seed(901)
    return InternLMForCausalLM(_config(**kwargs)).eval()


def _upstream_computation(weights, config, tokens, positions, padding=None, past=None):
    """Functional oracle of pinned upstream eager forward, including RAW keys.

    Uses checkpoint tensors only, no port modules/rope/mask/cache helpers.
    Original: internlm/internlm-7b@96e127d08d851a88cac736a9b091dd953ae1b873,
    modeling_internlm.py, Apache-2.0, InternLM/HuggingFace teams.
    """
    batch, length = tokens.shape
    hidden = F.embedding(tokens, weights["model.embed_tokens.weight"])
    heads, dim = config.num_attention_heads, config.hidden_size // config.num_attention_heads
    cache = []

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name):
        normalized = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)
        return weights[name] * normalized.to(weights[name].dtype)

    def rotate(x, indices, base, inverse=None):
        powers = torch.arange(dim // 2).float() * (2.0 / dim)
        inverse = 1.0 / torch.pow(torch.tensor(float(base)), powers) if inverse is None else inverse.float()
        theta = indices.float()[..., None] * inverse
        # Use the two half-vector equations separately, independently of the port's helper.
        cosine, sine = theta.cos()[:, None].to(x.dtype), theta.sin()[:, None].to(x.dtype)
        first, second = x[..., :dim // 2], x[..., dim // 2:]
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1)

    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        normalized = norm(hidden, prefix + "input_layernorm.weight")
        q, k, v = [linear(normalized, prefix + "self_attn." + name).reshape(batch, length, heads, dim).transpose(1, 2)
                   for name in ("q_proj", "k_proj", "v_proj")]
        full_positions = positions.expand(batch, -1)
        if past is not None:
            old_k, old_v, old_positions = past[index]
            k, v = torch.cat((old_k, k), 2), torch.cat((old_v, v), 2)
            full_positions = torch.cat((old_positions, full_positions), -1)
        cache.append((k, v, full_positions))
        seq_len = k.shape[-2]
        base = config.rotary["base"]
        if config.rotary["type"] == "dynamic" and seq_len > config.max_position_embeddings:
            factor = config.rotary.get("scaling_factor", 1.0)
            base *= (factor * seq_len / config.max_position_embeddings - (factor - 1)) ** (dim / (dim - 2))
        inverse = None if config.rotary["type"] == "dynamic" and seq_len > config.max_position_embeddings else weights[
            prefix + "self_attn.rotary_emb.inv_freq"]
        scores = torch.matmul(rotate(q, positions, base, inverse),
                              rotate(k, full_positions, base, inverse).transpose(-1, -2)) / dim ** 0.5
        allowed = torch.arange(seq_len)[None, :] <= torch.arange(seq_len - length, seq_len)[:, None]
        allowed = allowed[None, None].expand(batch, 1, -1, -1)
        if padding is not None:
            allowed = allowed & padding[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attended = torch.matmul(scores.float().softmax(-1).to(q.dtype), v)
        hidden = hidden + linear(attended.transpose(1, 2).reshape(batch, length, -1), prefix + "self_attn.o_proj")
        normalized = norm(hidden, prefix + "post_attention_layernorm.weight")
        activated = F.silu(linear(normalized, prefix + "mlp.gate_proj")) * linear(normalized, prefix + "mlp.up_proj")
        hidden = hidden + linear(activated, prefix + "mlp.down_proj")
    return linear(norm(hidden, "model.norm.weight"), "lm_head").float(), cache


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("rotary_type", ["origin", "dynamic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_internlm_matches_upstream_computation_and_raw_key_cache(bias, rotary_type, dtype):
    # Cross the trained context length: rotated-key Llama caching is incorrect here.
    model = _model(bias=bias, max_position_embeddings=4,
                   rotary={"type": rotary_type, "base": 10000, "scaling_factor": 2.0}).to(dtype)
    tokens = torch.tensor([[0, 1, 9, 7, 3, 6], [1, 4, 8, 5, 2, 9]])
    padding = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    positions = padding.cumsum(-1) - 1
    positions.masked_fill_(padding == 0, 1)
    tolerance = 3e-6 if dtype == torch.float32 else 0.004
    weights = model.state_dict()
    with torch.no_grad():
        expected, _ = _upstream_computation(weights, model.config, tokens, positions, padding)
        actual = model(tokens, position_ids=positions, attention_mask=padding)
        torch.testing.assert_close(actual.logits, expected, atol=tolerance, rtol=tolerance)
        reference_cache = None
        cache = None
        for start, end in ((0, 3), (3, 4), (4, 6)):
            expected, reference_cache = _upstream_computation(
                weights, model.config, tokens[:, start:end], positions[:, start:end], padding[:, :end], reference_cache,
            )
            actual = model(tokens[:, start:end], position_ids=positions[:, start:end],
                           attention_mask=padding[:, :end], past_key_values=cache, use_cache=True)
            cache = actual.past_key_values
            torch.testing.assert_close(actual.logits, expected, atol=tolerance, rtol=tolerance)
            assert cache.get_seq_length() == end
            for index, (raw_keys, _, key_positions) in enumerate(reference_cache):
                torch.testing.assert_close(cache.layers[index].keys, raw_keys, atol=tolerance, rtol=tolerance)
                assert torch.equal(cache.positions[index], key_positions)
        # A new short request must not retain the long request's NTK base.
        short = model(tokens[:, :2], position_ids=positions[:, :2], attention_mask=padding[:, :2])
        expected, _ = _upstream_computation(weights, model.config, tokens[:, :2], positions[:, :2], padding[:, :2])
        torch.testing.assert_close(short.logits, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_internlm_generation_cache_save_reload_and_causality(tmp_path, implementation):
    model = _model(attn_implementation=implementation)
    ids = torch.tensor([[1, 5, 7, 9, 3], [1, 4, 6, 8, 2]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        prefix = model(ids[:, :3])
        continued = model(ids[:, 3:], past_key_values=prefix.past_key_values)
        torch.testing.assert_close(continued.logits, expected[:, 3:], atol=2e-7, rtol=1e-5)
        perturbed = ids.clone()
        perturbed[:, -1] = 11
        torch.testing.assert_close(model(perturbed).logits[:, :-1], expected[:, :-1], atol=2e-7, rtol=1e-5)
        for beams in (1, 3):
            generated = model.generate(ids[:, :3], max_new_tokens=3, num_beams=beams, do_sample=False, eos_token_id=None)
            uncached = model.generate(ids[:, :3], max_new_tokens=3, num_beams=beams, do_sample=False,
                                       use_cache=False, eos_token_id=None)
            assert torch.equal(generated, uncached)
        model.save_pretrained(tmp_path)
        assert list(tmp_path.glob("*.safetensors"))
        assert not list(tmp_path.glob("*.bin"))
        reloaded = InternLMQModel.loader.from_pretrained(tmp_path, attn_implementation=implementation)
        torch.testing.assert_close(reloaded(ids, use_cache=False).logits, expected, atol=2e-7, rtol=1e-5)
        assert set(reloaded.state_dict()) == set(model.state_dict())
        assert not list(tmp_path.glob("*.py"))


def test_internlm_cache_metadata_follows_reorder_crop_reset_and_read_only_use():
    from gptqmodel.models.ports.internlm.modeling_internlm import InternLMCache
    model = _model()
    ids = torch.tensor([[1, 5, 7], [2, 6, 8]])
    positions = torch.tensor([[2, 3, 4], [8, 9, 10]])
    with torch.no_grad():
        cache = model(ids, position_ids=positions).past_key_values
        assert isinstance(cache, InternLMCache)
        original = copy.deepcopy(cache)
        model(ids[:, :1], position_ids=positions[:, :1] + 3,
              past_key_values=cache, use_cache=False)
        assert cache.get_seq_length() == 3
        for current, before in zip(cache.layers, original.layers):
            torch.testing.assert_close(current.keys, before.keys)
        cache.reorder_cache(torch.tensor([1, 0, 1]))
        assert torch.equal(cache.positions[0], positions[[1, 0, 1]])
        cache.batch_repeat_interleave(2)
        cache.batch_select_indices(torch.tensor([0, 3]))
        assert torch.equal(cache.positions[0], positions[[1, 0]])
        cache.crop(-1)
        assert cache.get_seq_length() == 2
        assert torch.equal(cache.positions[0], positions[[1, 0], :2])
        cache.reset()
        assert cache.positions == {}
        assert cache.get_seq_length() == 0
        fresh = model(ids[:, :1], past_key_values=cache)
        torch.testing.assert_close(fresh.logits, model(ids[:, :1]).logits)


def test_internlm_invalid_mask_does_not_poison_cache_retry():
    model = _model()
    ids = torch.tensor([[1, 7, 9]])
    with torch.no_grad():
        cache = model(ids[:, :2]).past_key_values
        with pytest.raises(ValueError, match="complete key sequence"):
            model(ids[:, 2:], past_key_values=cache, attention_mask=torch.ones(1, 2))
        assert cache.get_seq_length() == 2
        actual = model(ids[:, 2:], past_key_values=cache).logits
        torch.testing.assert_close(actual, model(ids, use_cache=False).logits[:, -1:], atol=2e-7, rtol=1e-5)


def _write_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
                                   **{str(i): i for i in range(3, 37)}}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                       unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.save_pretrained(path)


def _write_release_tokenizer(path):
    from tests.test_internlm_tokenizers_local_port import RELEASES, copy_release_assets
    copy_release_assets(path, RELEASES[0])


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_internlm_public_dispatch_uses_definition_owned_model(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel
    model = _model()
    model.save_pretrained(tmp_path)
    _write_release_tokenizer(tmp_path)
    monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    if bypass:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    marker = tmp_path / "checkpoint_python_executed"
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "checkpoint_code.UntrustedConfig",
                           "AutoModelForCausalLM": "checkpoint_code.UntrustedModel"}
    (tmp_path / "config.json").write_text(json.dumps(payload))
    (tmp_path / "checkpoint_code.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python executed')\n"
    )
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert isinstance(loaded, InternLMQModel)
    assert type(loaded.model).__module__.startswith(
        "gptqmodel.models.ports.internlm.")
    assert loaded.trust_remote_code is False
    assert not marker.exists()
    with torch.no_grad():
        ids = torch.tensor([[1, 7, 9]])
        torch.testing.assert_close(loaded.model(ids).logits, model(ids).logits, atol=2e-7, rtol=1e-5)


@pytest.mark.cpu
@pytest.mark.parametrize("bias", [False, True])
def test_internlm_plain_awq_quantizes_every_linear_and_reloads(tmp_path, bias):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig
    model = _model(bias=bias)
    model.save_pretrained(tmp_path)
    _write_release_tokenizer(tmp_path)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    assert wrapper.awq_get_modules_for_scaling(wrapper.model.model.layers[0], {}, {}) == []
    calibration = [{"input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == 14
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(expected).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors"))
    reloaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert sum(isinstance(module, AwqTorchLinear) for module in reloaded.model.modules()) == 14
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(**calibration[0], use_cache=False).logits, expected,
                                   atol=2e-4, rtol=2e-3)
