# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real InternLM2/2.5 inference, checkpoint and definition-owned selection."""

import json

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.cache_utils import DynamicCache, StaticCache

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.models.definitions.internlm2 import InternLM2QModel
from gptqmodel.models.ports.internlm2 import InternLM2Config, InternLM2ForCausalLM
from gptqmodel.utils.hf import load_model_config, prepare_remote_model_init_compat
from gptqmodel.utils.security import BYPASS_ENV


@pytest.fixture(autouse=True)
def bounded_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
        yield
    torch.set_num_threads(old)


def tiny_config(**kwargs):
    options = {
        "vocab_size": 48, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 32,
        "bias": False, "pad_token_id": 0, "eos_token_id": 2,
    }
    options.update(kwargs)
    return InternLM2Config(**options)


def hostile_checkpoint(path, config):
    config.save_pretrained(path)
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {
        "AutoConfig": "evil.RemoteConfig", "AutoModelForCausalLM": "evil.RemoteModel",
    }
    (path / "config.json").write_text(json.dumps(payload))
    marker = path / "EXECUTED"
    (path / "evil.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "raise RuntimeError('checkpoint Python must never execute')\n",
    )
    return marker


@pytest.mark.parametrize("trust,bypass", [(False, False), (True, True)])
def test_definition_loader_ignores_hostile_checkpoint_and_retry(tmp_path, monkeypatch, trust, bypass):
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security exception, not a behavior knob.
    marker = hostile_checkpoint(tmp_path, tiny_config())
    loaded = None
    try:
        loaded = load_model_config(tmp_path, trust_remote_code=trust)
    except (ValueError, RuntimeError):
        pass
    assert isinstance(loaded, InternLM2Config), "InternLM2 must select package config without checkpoint code"
    prepare_remote_model_init_compat(tmp_path, loaded, trust_remote_code=trust)
    loader = InternLM2QModel.loader
    with pytest.raises(OSError):
        loader.from_pretrained(tmp_path, trust_remote_code=trust, local_files_only=True)
    assert not marker.exists()
    original = InternLM2ForCausalLM(loaded).eval()
    original.save_pretrained(tmp_path)
    marker = hostile_checkpoint(tmp_path, loaded)
    restored = loader.from_pretrained(tmp_path, trust_remote_code=trust, code_revision="evil", local_files_only=True)
    assert type(restored) is InternLM2ForCausalLM
    assert type(loader.from_config(loaded, trust_remote_code=trust)) is InternLM2ForCausalLM
    ids = torch.tensor([[1, 5, 7, 9]])
    torch.testing.assert_close(restored(ids, use_cache=False).logits, original(ids, use_cache=False).logits)
    assert not marker.exists()


def reference_logits(model, ids, mask=None, positions=None):
    """Independent, direct tensor oracle for the pinned upstream eager computation."""
    c, weights = model.config, model.state_dict()
    x = F.embedding(ids, weights["model.tok_embeddings.weight"])
    batch, length, _ = x.shape
    if positions is None:
        positions = torch.arange(length).unsqueeze(0)
    params = c.rope_parameters
    kind = params["rope_type"]
    base, dim, factor = params["rope_theta"], c.hidden_size // c.num_attention_heads, params.get("factor", 1)
    if kind == "dynamic" and positions.max() + 1 > c.max_position_embeddings:
        base *= (factor * (positions.max() + 1) / c.max_position_embeddings - (factor - 1)) ** (dim / (dim - 2))
    angles = positions.float().unsqueeze(-1) * (base ** (-torch.arange(0, dim, 2).float() / dim))
    if kind == "linear":
        angles /= factor
    angles = torch.cat([angles, angles], dim=-1).unsqueeze(1)
    cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

    def norm(x, name):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + c.rms_norm_eps)
        return weights[name] * y.to(x.dtype)

    def linear(x, name, omit_bias=False):
        return F.linear(x, weights[name + ".weight"], None if omit_bias else weights.get(name + ".bias"))

    def rotate(x):
        half = dim // 2
        return x * cosine + torch.cat([-x[..., half:], x[..., :half]], -1) * sine

    for i in range(c.num_hidden_layers):
        prefix = f"model.layers.{i}."
        h = norm(x, prefix + "attention_norm.weight")
        packed = linear(h, prefix + "attention.wqkv", c.pretraining_tp > 1)
        # Independently index each KV group rather than reusing the port's reshape/split.
        groups, keys, values = [], [], []
        repeat = c.num_attention_heads // c.num_key_value_heads
        for k in range(c.num_key_value_heads):
            start = k * (repeat + 2) * dim
            groups.extend([packed[..., start + q * dim:start + (q + 1) * dim] for q in range(repeat)])
            keys.append(packed[..., start + repeat * dim:start + (repeat + 1) * dim])
            values.append(packed[..., start + (repeat + 1) * dim:start + (repeat + 2) * dim])
        q = rotate(torch.stack(groups, 1))
        k = rotate(torch.stack(keys, 1)).repeat_interleave(repeat, dim=1)
        v = torch.stack(values, 1).repeat_interleave(repeat, dim=1)
        scores = q @ k.transpose(-2, -1) / dim ** 0.5
        allowed = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
        allowed = allowed[None, None].expand(batch, 1, length, length)
        if mask is not None:
            allowed = allowed & mask[:, None, None].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        h = (F.softmax(scores.float(), -1).to(x.dtype) @ v).transpose(1, 2).reshape(batch, length, c.hidden_size)
        x = x + linear(h, prefix + "attention.wo", c.pretraining_tp > 1)
        h = norm(x, prefix + "ffn_norm.weight")
        x = x + linear(F.silu(linear(h, prefix + "feed_forward.w1")) * linear(h, prefix + "feed_forward.w3"), prefix + "feed_forward.w2")
    return linear(norm(x, "model.norm.weight"), "output").float()


@pytest.mark.parametrize(
    "heads,bias,rope",
    [(1, True, None), (2, False, {"type": "linear", "factor": 2.0}), (4, True, {"type": "dynamic", "factor": 2.0})],
    ids=["mqa", "gqa-linear", "mha-dynamic"],
)
def test_full_model_matches_independent_checkpoint_oracle(heads, bias, rope):
    model = InternLM2ForCausalLM(tiny_config(num_key_value_heads=heads, bias=bias, rope_scaling=rope, max_position_embeddings=4)).eval()
    # Bias initialization is zero; make the bias branch observable.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.copy_(torch.linspace(-0.07, 0.11, p.numel()))
    ids = torch.tensor([[1, 3, 5, 7, 9, 11], [1, 4, 6, 8, 10, 12]])
    mask = torch.tensor([[1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1]])
    expected = reference_logits(model, ids, mask)
    actual = model(ids, attention_mask=mask, use_cache=False, output_hidden_states=True, output_attentions=True)
    torch.testing.assert_close(actual.logits, expected, atol=1e-7, rtol=1e-5)
    assert len(actual.hidden_states) == 3 and len(actual.attentions) == 2
    assert actual.attentions[0][0, :, 0, 1:].count_nonzero() == 0
    changed = ids.clone()
    changed[:, -1] = 13
    torch.testing.assert_close(model(changed, attention_mask=mask, use_cache=False).logits[:, :-1], actual.logits[:, :-1])


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("cache_class", [DynamicCache, StaticCache])
def test_cache_prefill_continue_reset_and_generation(implementation, cache_class):
    c = tiny_config(attn_implementation=implementation)
    model = InternLM2ForCausalLM(c).eval()
    ids = torch.tensor([[0, 1, 3, 5, 7, 9], [1, 4, 6, 8, 10, 12]])
    mask = ids.ne(0).long()
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    full = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
    cache = cache_class(config=c, **({"max_cache_len": 12} if cache_class is StaticCache else {}))
    prefix = model(ids[:, :4], attention_mask=mask[:, :4], position_ids=positions[:, :4], past_key_values=cache)
    assert prefix.past_key_values is cache and cache.get_seq_length() == 4
    for end in (5, 6):
        out = model(ids[:, end - 1:end], attention_mask=mask[:, :end], position_ids=positions[:, end - 1:end], past_key_values=cache)
        torch.testing.assert_close(out.logits, full[:, end - 1:end], atol=1e-7, rtol=1e-5)
    assert cache.get_seq_length() == 6
    cache.reset()
    again = model(ids, attention_mask=mask, position_ids=positions, past_key_values=cache)
    torch.testing.assert_close(again.logits[mask.bool()], full[mask.bool()], atol=1e-7, rtol=1e-5)
    generation = {"attention_mask": mask, "max_new_tokens": 3, "min_new_tokens": 3, "do_sample": False}
    generated = model.generate(ids, **generation)
    torch.testing.assert_close(generated, model.generate(ids, use_cache=False, **generation))
    beam = model.generate(ids, num_beams=2, **generation)
    torch.testing.assert_close(beam, model.generate(ids, num_beams=2, use_cache=False, **generation))


def test_sdpa_matches_eager_with_attention_outputs_and_padding():
    model = InternLM2ForCausalLM(tiny_config(attn_implementation="sdpa")).eval()
    ids = torch.tensor([[1, 4, 6, 8], [1, 7, 9, 12]])
    mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]])
    expected = reference_logits(model, ids, mask)
    torch.testing.assert_close(model(ids, attention_mask=mask, use_cache=False).logits, expected, atol=1e-7, rtol=1e-5)
    # Attention probabilities come from the eager kernel (SDPA does not materialize them).
    model.set_attn_implementation("eager")
    out = model(ids, attention_mask=mask, use_cache=False, output_attentions=True)
    torch.testing.assert_close(out.logits, expected, atol=1e-7, rtol=1e-5)
    assert out.attentions[0][0, :, 0, 1:].count_nonzero() == 0


@pytest.mark.parametrize("bias", [False, True])
def test_legacy_tensor_parallel_eager_simulation(bias):
    model = InternLM2ForCausalLM(tiny_config(pretraining_tp=2, bias=bias)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.fill_(0.25)
    ids = torch.tensor([[1, 4, 7, 10]])
    torch.testing.assert_close(model(ids, use_cache=False).logits, reference_logits(model, ids), atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("dtype,tied", [(torch.float32, True), (torch.float16, False), (torch.bfloat16, True)])
def test_safetensors_save_reload_resave_equivalence(tmp_path, dtype, tied):
    model = InternLM2ForCausalLM(tiny_config(tie_word_embeddings=tied, rope_scaling={"type": "dynamic", "factor": 2.0})).to(dtype).eval()
    ids = torch.tensor([[1, 5, 9, 13]])
    expected = model(ids, use_cache=False).logits
    before = model.state_dict()
    for i in range(2):
        path = tmp_path / str(i)
        model.save_pretrained(path)
        assert (path / "model.safetensors").is_file()
        assert not list(path.glob("*.bin"))
        with safe_open(path / "model.safetensors", framework="pt") as f:
            assert "model.layers.0.attention.wqkv.weight" in f.keys()
            assert not any("inv_freq" in key for key in f.keys())
        model = InternLM2QModel.loader.from_pretrained(path, dtype=dtype).eval()
        assert all(p.dtype == dtype and not p.is_meta for p in model.parameters())
        assert (model.output.weight is model.model.tok_embeddings.weight) is tied
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, atol=0, rtol=0)


def test_dynamic_rope_long_short_request_does_not_leak_state():
    c = tiny_config(rope_scaling={"type": "dynamic", "factor": 2.0}, max_position_embeddings=4)
    model = InternLM2ForCausalLM(c).eval()
    short, long = torch.tensor([[1, 3, 5]]), torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]])
    before = model(short, use_cache=False).logits
    inv = model.model.rotary_emb.inv_freq.clone()
    model(long, use_cache=False)
    torch.testing.assert_close(model(short, use_cache=False).logits, before, atol=0, rtol=0)
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, inv, atol=0, rtol=0)


def test_invalid_inputs_fail_then_same_model_retries_with_cache():
    model = InternLM2ForCausalLM(tiny_config()).eval()
    ids = torch.tensor([[1, 5, 7]])
    with pytest.raises(ValueError, match="exactly one"):
        model(ids, inputs_embeds=model.get_input_embeddings()(ids))
    out = model(ids, labels=ids, output_hidden_states=True)
    assert out.loss.isfinite()
    out.loss.backward()
    assert model.model.layers[0].attention.wqkv.weight.grad.abs().sum() > 0
    direct = model(ids, use_cache=False).logits
    embedded = model(inputs_embeds=model.get_input_embeddings()(ids), use_cache=False).logits
    torch.testing.assert_close(direct, embedded, atol=0, rtol=0)
    assert model(ids, return_dict=False, use_cache=False)[0].shape == (1, 3, 48)


def write_tokenizer(path):
    from tests.test_internlm_tokenizers_local_port import copy_release_assets
    copy_release_assets(path)


@pytest.mark.parametrize("trust,bypass", [(False, False), (True, True)])
def test_public_dense_load_never_dispatches_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    c = tiny_config()
    original = InternLM2ForCausalLM(c).eval()
    original.save_pretrained(tmp_path)
    write_tokenizer(tmp_path)
    marker = hostile_checkpoint(tmp_path, c)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert isinstance(loaded, InternLM2QModel)
    assert type(loaded.model) is InternLM2ForCausalLM and type(loaded.model.config) is InternLM2Config
    assert loaded.lm_head == "output" and loaded.module_plan().find("attention.wqkv").quantize
    ids = torch.tensor([[1, 5, 7]])
    torch.testing.assert_close(loaded.model(ids, use_cache=False).logits, original(ids, use_cache=False).logits)
    assert not marker.exists()


def test_twenty_b_head_group_ratio_and_explicit_positions():
    c = tiny_config(hidden_size=48, num_attention_heads=6, num_key_value_heads=1, rope_scaling={"type": "dynamic", "factor": 2.5}, rope_theta=50000000)
    model = InternLM2ForCausalLM(c).eval()
    ids = torch.tensor([[1, 5, 7], [1, 4, 9]])
    positions = torch.tensor([[4, 5, 6], [7, 8, 9]])
    torch.testing.assert_close(model(ids, position_ids=positions, use_cache=False).logits, reference_logits(model, ids, positions=positions), atol=1e-7, rtol=1e-5)


@pytest.mark.cpu
@pytest.mark.parametrize("heads,bias", [(1, False), (2, True)], ids=["mqa", "gqa-bias"])
def test_public_awq_scales_quantizes_packs_and_reloads(tmp_path, heads, bias):
    from gptqmodel import BACKEND
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    original = InternLM2ForCausalLM(tiny_config(num_key_value_heads=heads, bias=bias)).eval()
    with torch.no_grad():
        for name, value in original.named_parameters():
            if "norm" in name:
                value.uniform_(0.6, 1.4)
    original.save_pretrained(tmp_path)
    norms = {k: v.half() for k, v in original.state_dict().items() if k.startswith("model.layers.") and "norm" in k}
    write_tokenizer(tmp_path)
    config = AWQConfig(device="cpu", group_size=16, fallback=None)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=config, backend=BACKEND.TORCH, dtype=torch.float16)
    calibration = [{"input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == 10
    # Scale search ran: every decoder norm absorbed its group's AWQ scale.
    assert len(norms) == 4
    assert all(not torch.equal(wrapper.model.state_dict()[key], value) for key, value in norms.items())
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(expected).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
    reloaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert type(reloaded.model) is InternLM2ForCausalLM
    assert sum(isinstance(module, AwqTorchLinear) for module in reloaded.model.modules()) == 10
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(**calibration[0], use_cache=False).logits, expected,
                                   atol=2e-4, rtol=2e-3)
