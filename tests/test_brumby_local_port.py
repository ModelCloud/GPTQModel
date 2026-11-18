# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real Brumby arithmetic, state transitions, generation and local loading."""

import copy
import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast
from transformers.cache_utils import DynamicCache

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models.auto import check_and_get_model_definition
from gptqmodel.models.definitions.brumby import BrumbyQModel
from gptqmodel.models.ports.brumby.configuration_brumby import BrumbyConfig
from gptqmodel.models.ports.brumby.modeling_brumby import (
    BrumbyForCausalLM,
    _PowerCacheLayer,
    power_retention,
)
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


def tiny_config(**kwargs):
    values = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "chunk_size": 3,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(kwargs)
    return BrumbyConfig(**values)


@pytest.fixture(autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _poison(path):
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "evil.BadConfig", "AutoModelForCausalLM": "evil.BadModel"}
    (path / "config.json").write_text(json.dumps(payload))
    marker = path / "EXECUTED"
    (path / "evil.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python executed')\n"
    )
    return marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_definition_owned_loading_ignores_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    torch.manual_seed(17)
    original = BrumbyForCausalLM(tiny_config()).eval()
    original.save_pretrained(tmp_path)
    marker = _poison(tmp_path)
    # The existing security bypass itself is the tested public surface.
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is BrumbyConfig
    assert check_and_get_model_definition(str(tmp_path), trust_remote_code=trust) is BrumbyQModel
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    model = BrumbyQModel.loader.from_pretrained(tmp_path, config=config, trust_remote_code=trust).eval()
    assert type(model) is BrumbyForCausalLM
    x = torch.tensor([[3, 7, 8, 4]])
    with torch.no_grad():
        torch.testing.assert_close(model(x).logits, original(x).logits, rtol=0, atol=0)
    with torch.device("meta"):
        shell = BrumbyQModel.loader.from_config(config, trust_remote_code=trust)
    assert next(shell.parameters()).device.type == "meta"
    assert not marker.exists()


@pytest.mark.parametrize("rope_type", ["default", "linear", "dynamic"])
def test_local_rope_configuration_builds_rotary(rope_type):
    config = tiny_config(
        rope_parameters={
            "rope_type": rope_type,
            "rope_theta": 1000000.0,
            **({"factor": 2.0} if rope_type != "default" else {}),
        }
    )
    with torch.device("meta"):
        model = BrumbyForCausalLM(config)
    assert model.model.rotary_emb.inv_freq.shape == (4,)


@torch.no_grad()
def test_generation_hf_chunked_prefill_embeds_and_legacy_rope(tmp_path):
    model = BrumbyForCausalLM(tiny_config(rope_theta=1000000, rope_scaling={"type": "linear", "factor": 2.0})).eval()
    ids = torch.tensor([[3, 4, 6, 5, 8, 7, 3]])
    expected = model.generate(ids, max_new_tokens=3, do_sample=False, eos_token_id=None)
    chunked = model.generate(ids, prefill_chunk_size=2, max_new_tokens=3, do_sample=False, eos_token_id=None)
    assert torch.equal(chunked, expected)
    embeds = model.get_input_embeddings()(ids)
    embedded = model.generate(inputs_embeds=embeds, max_new_tokens=3, do_sample=False, eos_token_id=None)
    assert torch.equal(embedded, expected[:, -3:])
    cache = model(ids[:, :4]).past_key_values
    resumed = model.generate(ids, past_key_values=cache, max_new_tokens=3, do_sample=False, eos_token_id=None)
    assert torch.equal(resumed, expected)
    model.save_pretrained(tmp_path)
    reload = BrumbyQModel.loader.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(reload(ids, use_cache=False).logits, model(ids, use_cache=False).logits, rtol=0, atol=0)


@torch.no_grad()
def test_dynamic_rope_extends_and_resets_frequencies():
    model = BrumbyForCausalLM(
        tiny_config(
            max_position_embeddings=4, rope_parameters={"rope_type": "dynamic", "factor": 2.0, "rope_theta": 10000.0}
        )
    ).eval()
    before = model.model.rotary_emb.inv_freq.clone()
    assert torch.isfinite(model(torch.tensor([[3, 4, 5, 6, 7, 8]]), use_cache=False).logits).all()
    assert not torch.equal(before, model.model.rotary_emb.inv_freq)
    model(torch.tensor([[3, 4]]), use_cache=False)
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, before, rtol=0, atol=0)


def retention_oracle(q, k, v, gates, mask, use_exp=False, window=None):
    """Independent quadratic FP64 attention form of retention 1.0.7.

    No feature map, recurrent-state helper or port intermediate is reused.
    Mirrors _attention/reference.py's log-domain stabilization and epsilon.
    """
    batch, heads, length, _ = q.shape
    result = torch.zeros_like(q, dtype=torch.float64)
    groups = heads // k.shape[1]
    for b in range(batch):
        for h in range(heads):
            kh = h // groups
            for t in range(length):
                active = mask[b, : t + 1].bool()
                if window is not None:
                    active = active & (torch.arange(t + 1) > t - window)
                scores = q[b, h, t].double() @ k[b, kh, : t + 1].double().T / q.shape[-1] ** 0.5
                if not use_exp:
                    scores = 2 * (scores.abs() + 1e-7).log()
                    cumulative = (gates[b, kh, : t + 1].double() * mask[b, : t + 1]).cumsum(-1)
                    scores = scores + cumulative[-1] - cumulative
                if active.any():
                    weights = (scores[active] - scores[active].max()).exp()
                    weights = weights / (weights.sum() + (0 if use_exp else 1e-6))
                    result[b, h, t] = weights @ v[b, kh, : t + 1][active].double()
    return result.to(q.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunk", [None, 2, 4])
def test_retention_matches_publisher_quadratic_math_across_compression(dtype, chunk):
    torch.manual_seed(19)
    q = torch.randn(2, 4, 9, 8).to(dtype)
    k = torch.randn(2, 2, 9, 8).to(dtype)
    v = torch.randn(2, 2, 9, 8).to(dtype)
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    gates = F.logsigmoid(torch.randn(2, 2, 9)) * mask[:, None]
    expected = retention_oracle(q, k, v, gates, mask)
    layer = _PowerCacheLayer()
    outputs = []
    for start, stop in [(0, 3), (3, 4), (4, 7), (7, 9)]:
        outputs.append(
            power_retention(
                q[:, :, start:stop],
                k[:, :, start:stop],
                v[:, :, start:stop],
                gates[:, :, start:stop],
                mask[:, start:stop],
                layer,
                chunk,
                8**-0.5,
            )
        )
    actual = torch.cat(outputs, 2)
    # Compare rounded results separately from the FP64 arithmetic oracle.
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    assert layer.cumulative_length == 9
    if chunk is not None:
        assert layer.state is not None
        assert layer.state.shape[-2] == 8 * 9 // 2
        assert layer.keys is None or layer.keys.shape[2] < chunk


def _model_oracle(model, ids, mask):
    """Functional checkpoint-weight oracle; never invokes a port forward."""
    c = model.config
    weights = model.state_dict()
    x = F.embedding(ids, weights["model.embed_tokens.weight"]).double()
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    inv = c.rope_parameters["rope_theta"] ** (-torch.arange(0, c.head_dim, 2).double() / c.head_dim)
    angles = positions[..., None] * inv
    cos = torch.cat((angles, angles), -1).cos()[:, None]
    sin = torch.cat((angles, angles), -1).sin()[:, None]

    def norm(z, name):
        return z * (z.square().mean(-1, keepdim=True) + c.rms_norm_eps).rsqrt() * weights[name + ".weight"].double()

    for index in range(c.num_hidden_layers):
        prefix = f"model.layers.{index}."

        def linear(z, name):
            return F.linear(
                z,
                weights[prefix + name + ".weight"].double(),
                weights[prefix + name + ".bias"].double() if prefix + name + ".bias" in weights else None,
            )

        z = norm(x, prefix + "input_layernorm")
        q = linear(z, "self_attn.q_proj").view(*ids.shape, -1, c.head_dim)
        k = linear(z, "self_attn.k_proj").view(*ids.shape, -1, c.head_dim)
        q = norm(q, prefix + "self_attn.q_norm").transpose(1, 2)
        k = norm(k, prefix + "self_attn.k_norm").transpose(1, 2)
        q = q * cos + torch.cat((-q[..., c.head_dim // 2 :], q[..., : c.head_dim // 2]), -1) * sin
        k = k * cos + torch.cat((-k[..., c.head_dim // 2 :], k[..., : c.head_dim // 2]), -1) * sin
        v = linear(z, "self_attn.v_proj").view(*ids.shape, -1, c.head_dim).transpose(1, 2)
        gates = F.logsigmoid(linear(z, "self_attn.g_proj")).transpose(1, 2)
        attn = retention_oracle(
            q,
            k,
            v,
            gates,
            mask,
            c.use_exp,
            c.sliding_window if c.layer_types[index] == "sliding_attention" and c.use_exp else None,
        )
        x = x + linear(attn.transpose(1, 2).reshape(*ids.shape, -1), "self_attn.o_proj")
        z = norm(x, prefix + "post_attention_layernorm")
        x = x + linear(F.silu(linear(z, "mlp.gate_proj")) * linear(z, "mlp.up_proj"), "mlp.down_proj")
    return F.linear(norm(x, "model.norm"), weights["lm_head.weight"].double()).float()


@pytest.mark.parametrize("use_exp", [False, True])
@pytest.mark.parametrize("layout", ["14b", "4b", "mha_bias"])
@torch.no_grad()
def test_native_forward_generation_cache_and_safetensors(tmp_path, use_exp, layout):
    torch.manual_seed(29)
    overrides = (
        {"hidden_size": 40, "num_attention_heads": 8}
        if layout == "4b"
        else {"hidden_size": 80, "num_attention_heads": 10}
    )
    if layout == "mha_bias":
        overrides.update(
            hidden_size=32, num_attention_heads=4, num_key_value_heads=4, attention_bias=True, tie_word_embeddings=True
        )
    config = tiny_config(use_exp=use_exp, **overrides)
    model = BrumbyForCausalLM(config).eval()
    ids = torch.tensor([[0, 3, 6, 5, 7, 9, 3], [4, 8, 3, 6, 9, 5, 8]])
    mask = ids.ne(0).long()
    expected = _model_oracle(model, ids, mask)
    full = model(ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(full, expected, rtol=2e-5, atol=2e-5)
    prefill = model(ids[:, :4], attention_mask=mask[:, :4])
    cache = prefill.past_key_values
    continuation = model(ids[:, 4:6], attention_mask=mask[:, :6], past_key_values=cache)
    last = model(ids[:, 6:], attention_mask=mask, past_key_values=cache)
    torch.testing.assert_close(
        torch.cat((prefill.logits, continuation.logits, last.logits), 1), full, rtol=2e-5, atol=2e-5
    )
    assert cache.get_seq_length() == 7
    if not use_exp:
        assert all(layer.state is not None for layer in cache.layers)
    greedy = model.generate(ids, attention_mask=mask, do_sample=False, max_new_tokens=3, eos_token_id=None)
    uncached = model.generate(
        ids, attention_mask=mask, do_sample=False, max_new_tokens=3, eos_token_id=None, use_cache=False
    )
    assert torch.equal(greedy, uncached)
    beam = model.generate(ids[:1], attention_mask=mask[:1], num_beams=2, max_new_tokens=3, eos_token_id=None)
    assert beam.shape == (1, 10)
    assert torch.isfinite(model(ids, attention_mask=mask, labels=ids).loss)
    model.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
    reloaded = BrumbyQModel.loader.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(reloaded(ids, attention_mask=mask, use_cache=False).logits, full, rtol=0, atol=0)
    assert set(reloaded.state_dict()) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, reloaded.state_dict()[name]), name
    reloaded.save_pretrained(tmp_path / "again")
    again = BrumbyQModel.loader.from_pretrained(tmp_path / "again").eval()
    torch.testing.assert_close(again(ids, attention_mask=mask, use_cache=False).logits, full, rtol=0, atol=0)


@torch.no_grad()
def test_cache_rollback_retry_beam_reorder_and_reset(monkeypatch):
    torch.manual_seed(31)
    model = BrumbyForCausalLM(tiny_config()).eval()
    ids = torch.tensor([[3, 4, 7, 5, 8, 9], [5, 6, 4, 3, 7, 8]])
    cache = model(ids[:, :2]).past_key_values
    before = copy.deepcopy(cache)
    forward = model.model.layers[1].forward
    fired = []

    def fail(*args, **kwargs):
        assert cache.layers[0].cumulative_length == 5  # first layer already compressed/updated
        fired.append(True)
        raise RuntimeError("injected after first layer cache mutation")

    monkeypatch.setattr(model.model.layers[1], "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model(ids[:, 2:5], past_key_values=cache)
    assert fired and cache.get_seq_length() == 2
    for actual, old in zip(cache.layers, before.layers):
        for name in ("keys", "values", "gates", "valid", "state", "normalizer"):
            a, b = getattr(actual, name), getattr(old, name)
            assert a is b is None or (a is not None and b is not None and torch.equal(a, b))
    monkeypatch.setattr(model.model.layers[1], "forward", forward)
    retry = model(ids[:, 2:5], past_key_values=cache).logits
    fresh = model(ids[:, :5], use_cache=False).logits[:, 2:]
    torch.testing.assert_close(retry, fresh, rtol=2e-5, atol=2e-5)
    order = torch.tensor([1, 0, 1])
    cache.reorder_cache(order)
    out = model(ids[order, 5:], past_key_values=cache).logits
    expected = model(ids[order], use_cache=False).logits[:, -1:]
    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-5)
    cache.reset()
    assert cache.get_seq_length() == 0 and not cache.is_initialized
    reset = model(ids, past_key_values=cache).logits
    torch.testing.assert_close(reset, model(ids, use_cache=False).logits, rtol=0, atol=0)
    with pytest.raises(TypeError, match="KV-only"):
        model(ids, past_key_values=DynamicCache())
    with pytest.raises(ValueError, match="use_cache=True"):
        model(ids, past_key_values=cache, use_cache=False)
    with pytest.raises(ValueError, match="cannot be cropped"):
        cache.crop(1)


def test_cache_is_request_owned_and_invalid_mask_does_not_mutate():
    model = BrumbyForCausalLM(tiny_config()).eval()
    with torch.no_grad():
        first = model(torch.tensor([[3, 4]])).past_key_values
        second = model(torch.tensor([[6, 7]])).past_key_values
        assert first is not second
        assert first.layers[0].keys.data_ptr() != second.layers[0].keys.data_ptr()
        with pytest.raises(ValueError, match="attention_mask"):
            model(torch.tensor([[5]]), attention_mask=torch.ones(1, 1), past_key_values=first)
        assert first.get_seq_length() == 2


@torch.no_grad()
def test_lm_head_failure_restores_compressed_cache_before_retry(monkeypatch):
    model = BrumbyForCausalLM(tiny_config()).eval()
    ids = torch.tensor([[3, 4, 6, 5, 7, 8]])
    cache = model(ids[:, :4]).past_key_values
    before = copy.deepcopy(cache)
    forward = model.lm_head.forward

    def fail(*args, **kwargs):
        assert cache.get_seq_length() == 6
        raise RuntimeError("injected LM head failure")

    monkeypatch.setattr(model.lm_head, "forward", fail)
    with pytest.raises(RuntimeError, match="LM head"):
        model(ids[:, 4:], past_key_values=cache)
    assert cache.get_seq_length() == 4
    for actual, old in zip(cache.layers, before.layers):
        for name in ("keys", "values", "gates", "valid", "state", "normalizer"):
            a, b = getattr(actual, name), getattr(old, name)
            assert a is b is None or (a is not None and b is not None and torch.equal(a, b))
    monkeypatch.setattr(model.lm_head, "forward", forward)
    actual = model(ids[:, 4:], past_key_values=cache).logits
    torch.testing.assert_close(actual, model(ids, use_cache=False).logits[:, 4:], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("use_exp", [False, True])
@torch.no_grad()
def test_model_is_causal_and_exponential_variant_keeps_sliding_window(use_exp):
    model = BrumbyForCausalLM(
        tiny_config(use_exp=use_exp, use_sliding_window=True, sliding_window=3, max_window_layers=0)
    ).eval()
    ids = torch.tensor([[3, 4, 6, 5, 7, 8]])
    baseline = model(ids, use_cache=False).logits
    changed = ids.clone()
    changed[:, 4:] = torch.tensor([[9, 10]])
    torch.testing.assert_close(model(changed, use_cache=False).logits[:, :4], baseline[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(baseline, _model_oracle(model, ids, torch.ones_like(ids)), rtol=2e-5, atol=2e-5)
    cache = model(ids[:, :4]).past_key_values
    torch.testing.assert_close(model(ids[:, 4:], past_key_values=cache).logits, baseline[:, 4:], rtol=2e-5, atol=2e-5)


@torch.no_grad()
def test_native_modules_and_init_preserve_partially_loaded_tensors():
    from transformers.models.qwen3 import modeling_qwen3

    model = BrumbyForCausalLM(tiny_config(attention_bias=True, tie_word_embeddings=True))
    layer = model.model.layers[0]
    assert type(layer.mlp) is modeling_qwen3.Qwen3MLP
    assert type(layer.self_attn.q_norm) is modeling_qwen3.Qwen3RMSNorm
    assert type(model.model.rotary_emb) is modeling_qwen3.Qwen3RotaryEmbedding
    assert model.lm_head.weight is model.model.embed_tokens.weight
    # Transformers 5 initializes only tensors that were not loaded: a module
    # with a loaded weight and a missing bias must keep the loaded weight.
    proj = layer.self_attn.q_proj
    proj.weight.fill_(0.5)
    proj.bias.fill_(7.0)
    proj.weight._is_hf_initialized = True
    model._init_weights(proj)
    assert torch.all(proj.weight == 0.5)
    assert torch.all(proj.bias == 0)
    rotary = model.model.rotary_emb
    expected = rotary.inv_freq.clone()
    rotary.inv_freq.zero_()
    rotary.original_inv_freq.zero_()
    model._init_weights(rotary)
    torch.testing.assert_close(rotary.inv_freq, expected, rtol=0, atol=0)
    torch.testing.assert_close(rotary.original_inv_freq, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"p": 4}, "degree-2"),
        ({"head_dim": 7}, "head_dim"),
        ({"num_key_value_heads": 3}, "divisible"),
        ({"chunk_size": 0}, "chunk_size"),
    ],
)
def test_invalid_config_fails_explicitly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        tiny_config(**kwargs)


@pytest.mark.cpu
@pytest.mark.parametrize("fallback", [None, {}])
@pytest.mark.parametrize("hidden_size", [80, 160])
def test_awq_quantizes_and_packs_every_supported_linear(tmp_path, hidden_size, fallback):
    torch.manual_seed(41)
    config = tiny_config(
        hidden_size=hidden_size,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=8 if hidden_size == 80 else 10,
        num_key_value_heads=2,
        head_dim=16,
        chunk_size=3,
    )
    original = BrumbyForCausalLM(config).eval()
    original.save_pretrained(tmp_path)
    tokenizer = Tokenizer(
        WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "a": 3, "b": 4, "c": 5, "d": 6, "e": 7}, unk_token="[UNK]")
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    ).save_pretrained(tmp_path)
    model = GPTQModel.load(
        str(tmp_path),
        quantize_config=AWQConfig(device="cpu", group_size=16, fallback=fallback),
        backend=BACKEND.TORCH,
        dtype=torch.float16,
        trust_remote_code=False,
    )
    assert type(model.model) is BrumbyForCausalLM
    layer = model.model.model.layers[0]
    # The input/post-attention norms absorb AWQ scales and the dense retention
    # gate is compensated for the input-norm fold; the per-head norms stay dense.
    folded_names = ["input_layernorm", "post_attention_layernorm", "self_attn.g_proj"]
    dense_names = [*folded_names, "self_attn.q_norm", "self_attn.k_norm"]
    # GPT-QModel loads a meta shell backed by LazyTurtle; compare dense tensors
    # against the actual saved checkpoint, not unmaterialized shell storage.
    dense_before = {
        name: {
            k: v.detach().half().clone() for k, v in original.model.layers[0].get_submodule(name).state_dict().items()
        }
        for name in dense_names
    }
    calibration = [
        {"input_ids": torch.tensor([[3, 4, 6, 5, 7, 3, 4, 5]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)}
        for _ in range(2)
    ]
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    expected = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
    actual = {name for name, module in layer.named_modules() if isinstance(module, AwqTorchLinear)}
    assert actual == expected
    folded_changed = False
    for name, state in dense_before.items():
        for key, value in state.items():
            current = layer.get_submodule(name).state_dict()[key]
            if name in folded_names and key == "weight":
                folded_changed |= not torch.equal(current, value)
            else:
                torch.testing.assert_close(current, value, rtol=0, atol=0)
    assert folded_changed, "AWQ scale search did not fold any scale"
    with torch.inference_mode():
        assert torch.isfinite(model.model(**calibration[0], use_cache=False).logits).all()
