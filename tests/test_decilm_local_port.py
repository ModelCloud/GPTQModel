# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real tiny-model qualification; no checkpoint Python or trained weights."""

import json

import pytest
import torch
from torch.nn import functional as F

from gptqmodel.models.definitions.decilm import DeciLMQModel
from gptqmodel.utils.hf import load_model_config
from gptqmodel.utils.security import BYPASS_ENV


def config_kwargs(alias="deci", **overrides):
    values = {
        "model_type": alias, "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads_per_layer": [2, 1, 4],
        "max_position_embeddings": 32, "rms_norm_eps": 1e-5, "pad_token_id": 0, "eos_token_id": None,
        "bos_token_id": 1, "tie_word_embeddings": False,
    }
    if alias == "nemotron-nas":
        values.update(intermediate_size=None, num_key_value_heads=None,
                      num_key_value_heads_per_layer=None, block_configs=[
            {"attention": {"no_op": True}, "ffn": {"ffn_mult": 2.0}},
            {"attention": {"n_heads_in_group": 4}, "ffn": {"no_op": True}},
            {"attention": {"replace_with_linear": True}, "ffn": {"replace_with_linear": True}},
        ])
    values.update(overrides)
    return values


def tiny(alias="deci", **overrides):
    from gptqmodel.models.ports.decilm.configuration_decilm import DeciLMConfig
    from gptqmodel.models.ports.decilm.modeling_decilm import DeciLMForCausalLM

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        model = DeciLMForCausalLM(DeciLMConfig(**config_kwargs(alias, **overrides)))
    return model.eval()


def save_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, "a": 3, "b": 4}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]",
                            unk_token="[UNK]").save_pretrained(path)


def malicious_checkpoint(path, alias):
    path.mkdir(exist_ok=True)
    marker = path / "EXECUTED"
    for name, cls in (("configuration_evil", "EvilConfig"), ("modeling_evil", "EvilModel")):
        (path / (name + ".py")).write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')\n"
            f"raise RuntimeError('checkpoint code executed: {cls}')\n"
        )
    payload = config_kwargs(alias)
    payload["auto_map"] = {
        "AutoConfig": "configuration_evil.EvilConfig", "AutoModelForCausalLM": "modeling_evil.EvilModel",
    }
    (path / "config.json").write_text(json.dumps(payload))
    return marker


@pytest.mark.parametrize("alias", ["deci", "nemotron-nas"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_config_ignores_checkpoint_code(tmp_path, monkeypatch, alias, trust, bypass):
    # Existing security-bypass flag is the subject of this test, not a new knob.
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    marker = malicious_checkpoint(tmp_path, alias)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config).__module__ == "gptqmodel.models.ports.decilm.configuration_decilm"
    assert config.model_type == alias
    assert DeciLMQModel.require_trust_remote_code is False
    with torch.device("meta"):
        model = DeciLMQModel.loader.from_config(config, trust_remote_code=trust)
    assert type(model).__module__ == "gptqmodel.models.ports.decilm.modeling_decilm"
    assert next(model.parameters()).device.type == "meta"
    assert not marker.exists()


def reference_rope(x, pos, config, seq_len):
    """Independent equations from the pinned Llama/Deci implementations."""
    dim = x.shape[-1]
    theta = config.rope_parameters["rope_theta"]
    params = config.rope_parameters
    kind = params["rope_type"]
    if kind == "dynamic" and seq_len > config.max_position_embeddings:
        factor = params["factor"]
        theta *= (factor * seq_len / config.max_position_embeddings - (factor - 1)) ** (dim / (dim - 2))
    inv = 1 / theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim)
    if kind == "linear":
        inv = inv / params["factor"]
    if kind == "llama3":
        wavelength = 2 * torch.pi / inv
        old = params["original_max_position_embeddings"]
        low, high = params["low_freq_factor"], params["high_freq_factor"]
        smooth = (old / wavelength - low) / (high - low)
        middle = (1 - smooth) * inv / params["factor"] + smooth * inv
        inv = torch.where(wavelength > old / low, inv / params["factor"],
                          torch.where(wavelength < old / high, inv, middle))
    angles = pos.float()[..., None] * inv
    angles = torch.cat((angles, angles), dim=-1)[:, None]
    first, second = x.chunk(2, -1)
    rotated = torch.cat((-second, first), -1)
    return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)


def reference_forward(model, ids, mask=None, positions=None):
    """Functional state-dict oracle; never calls a port forward/helper."""
    c = model.config
    weights = model.state_dict()
    x = F.embedding(ids, weights["model.embed_tokens.weight"])
    batch, length = ids.shape
    positions = torch.arange(length)[None] if positions is None else positions
    head_dim = c.hidden_size // c.num_attention_heads

    def norm(value, path):
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + c.rms_norm_eps)
        return weights[path + ".weight"] * normalized.to(value.dtype)

    def linear(value, path):
        # The legacy TP path omits biases.
        bias = weights.get(path + ".bias") if c.pretraining_tp == 1 else None
        return F.linear(value, weights[path + ".weight"], bias)

    for i in range(c.num_hidden_layers):
        prefix = f"model.layers.{i}."
        block = c.block_configs[i] if c.block_configs is not None else {
            "attention": {"n_heads_in_group": c.num_attention_heads // c.num_key_value_heads_per_layer[i]}, "ffn": {},
        }
        a, f = block["attention"], block["ffn"]
        if not a.get("no_op"):
            value = norm(x, prefix + "input_layernorm")
            if a.get("replace_with_linear"):
                value = linear(value, prefix + "self_attn.linear_attn")
            else:
                heads = c.num_attention_heads
                group = a["n_heads_in_group"]
                q, k, v = [linear(value, prefix + "self_attn." + name + "_proj") for name in ("q", "k", "v")]
                q = q.view(batch, length, heads, head_dim).transpose(1, 2)
                k = k.view(batch, length, heads // group, head_dim).transpose(1, 2)
                v = v.view(batch, length, heads // group, head_dim).transpose(1, 2)
                q = reference_rope(q, positions, c, max(length, int(positions.max()) + 1))
                k = reference_rope(k, positions, c, max(length, int(positions.max()) + 1))
                k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
                allowed = torch.ones(length, length, dtype=torch.bool).tril()
                window, sinks = a.get("window_length"), a.get("num_sink_tokens")
                if window is not None and (sinks is None or a.get("unshifted_sink")
                                           or a.get("use_prefill_window_in_sink_attention")):
                    allowed &= torch.arange(length)[None] > torch.arange(length)[:, None] - window
                additive = torch.zeros(batch, 1, length, length).to(x.dtype)
                additive.masked_fill_(~allowed[None, None], torch.finfo(x.dtype).min)
                if mask is not None:
                    additive.masked_fill_(~mask[:, None, None].bool(), torch.finfo(x.dtype).min)
                if a.get("unshifted_sink"):
                    additive[..., :sinks] = 0
                probs = torch.softmax(q @ k.transpose(-1, -2) / head_dim ** 0.5 + additive,
                                      dim=-1, dtype=torch.float32).to(x.dtype)
                value = (probs @ v).transpose(1, 2).reshape(batch, length, c.hidden_size)
                value = linear(value, prefix + "self_attn.o_proj")
            x = x + value
        if not f.get("no_op"):
            value = norm(x, prefix + "post_attention_layernorm")
            if f.get("replace_with_linear"):
                value = linear(value, prefix + "mlp.linear_mlp")
            else:
                value = F.silu(linear(value, prefix + "mlp.gate_proj")) * linear(value, prefix + "mlp.up_proj")
                value = linear(value, prefix + "mlp.down_proj")
            x = x + value
    return linear(norm(x, "model.norm"), "lm_head").float()


@pytest.mark.parametrize("alias", ["deci", "nemotron-nas"])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("tp", [1, 2])
@torch.inference_mode()
def test_forward_matches_independent_checkpoint_equations(alias, implementation, tp):
    model = tiny(alias, attn_implementation=implementation, pretraining_tp=tp)
    ids = torch.tensor([[0, 1, 3, 5, 7], [1, 4, 6, 8, 9]])
    mask = ids.ne(0).long()
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    expected = reference_forward(model, ids, mask, positions)
    actual = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual[:, 1:], expected[:, 1:], rtol=2e-5, atol=1e-6)
    # Nontrivial recurrent/MLP contributions, not just embedding->head parity.
    assert actual.std() > 0.02


@pytest.mark.parametrize("kind,params", [
    ("linear", {"factor": 2.0}),
    ("dynamic", {"factor": 2.0}),
    ("llama3", {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8}),
])
@torch.inference_mode()
def test_scaled_rope_matches_reference(kind, params):
    model = tiny(max_position_embeddings=4, rope_scaling={"rope_type": kind, **params})
    ids = torch.tensor([[1, 4, 5, 6, 7, 8, 9]])
    torch.testing.assert_close(model(ids, use_cache=False).logits, reference_forward(model, ids),
                               rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("alias", ["deci", "nemotron-nas"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_cache_chunk_generation_and_reset(alias, dtype):
    from gptqmodel.models.ports.decilm.modeling_decilm import DeciLMCache

    model = tiny(alias, attn_implementation="eager").to(dtype)
    ids = torch.tensor([[0, 1, 4, 7, 3, 6], [1, 9, 4, 8, 5, 7]])
    mask = ids.ne(0).long()
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    full = model(ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    prefill = model(ids[:, :3], attention_mask=mask[:, :3], position_ids=pos[:, :3])
    cache = prefill.past_key_values
    assert isinstance(cache, DeciLMCache)
    tail = model(ids[:, 3:5], attention_mask=mask[:, :5], position_ids=pos[:, 3:5], past_key_values=cache)
    last = model(ids[:, 5:], attention_mask=mask, position_ids=pos[:, 5:], past_key_values=cache)
    tolerance = 2e-3 if dtype == torch.bfloat16 else 3e-4 if dtype == torch.float16 else 1e-6
    torch.testing.assert_close(tail.logits, full[:, 3:5], rtol=2e-2, atol=tolerance)
    torch.testing.assert_close(last.logits, full[:, 5:], rtol=2e-2, atol=tolerance)
    assert cache.get_seq_length() == 6
    expected_kv = [2, 1, 4] if alias == "deci" else [None, 1, None]
    for layer, expected in zip(cache.layers, expected_kv):
        assert (layer.keys.shape[1] if layer.is_initialized else None) == expected
    cached = model.generate(ids[:, 1:], max_new_tokens=3, do_sample=False, use_cache=True)
    uncached = model.generate(ids[:, 1:], max_new_tokens=3, do_sample=False, use_cache=False)
    assert torch.equal(cached, uncached)
    # Modern beam search reorders the heterogeneous cache in place.
    beams = model.generate(ids[:, 1:], max_new_tokens=2, num_beams=2, do_sample=False)
    assert beams.shape == (2, 7)
    cache.reset()
    assert cache.get_seq_length() == 0
    again = model(ids, attention_mask=mask, position_ids=pos, past_key_values=cache)
    torch.testing.assert_close(again.logits[:, 1:], full[:, 1:], rtol=0, atol=0)


@pytest.mark.parametrize("alias", ["deci", "nemotron-nas"])
@pytest.mark.parametrize("trust,bypass", [(False, False), (True, False), (False, True), (True, True)])
@torch.inference_mode()
def test_safetensors_reload_owns_all_weights_and_ignores_remote(tmp_path, monkeypatch, alias, trust, bypass):
    model = tiny(alias)
    model.save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    marker = malicious_checkpoint(tmp_path / "evil", alias)
    payload["auto_map"] = json.loads((tmp_path / "evil/config.json").read_text())["auto_map"]
    for source in (tmp_path / "evil").glob("*.py"):
        (tmp_path / source.name).write_text(source.read_text())
    (tmp_path / "config.json").write_text(json.dumps(payload))
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    loaded = DeciLMQModel.loader.from_pretrained(tmp_path, trust_remote_code=trust)
    assert loaded.config.model_type == alias
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
    ids = torch.tensor([[1, 7, 3, 6]])
    torch.testing.assert_close(model(ids).logits, loaded(ids).logits, rtol=0, atol=0)
    # Exercise the complete public dispatcher and real tokenizer consumer too.
    from gptqmodel import GPTQModel

    save_tokenizer(tmp_path)
    public = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert isinstance(public, DeciLMQModel) and public.trust_remote_code is False
    torch.testing.assert_close(model(ids).logits, public.model(ids).logits, rtol=0, atol=0)
    loaded.save_pretrained(tmp_path / "again")
    twice = DeciLMQModel.loader.from_pretrained(tmp_path / "again")
    torch.testing.assert_close(model(ids).logits, twice(ids).logits, rtol=0, atol=0)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.rglob("*.bin"))
    assert not marker.exists()


@pytest.mark.parametrize("length", [6, 9])
@torch.inference_mode()
def test_shifted_sink_initial_prefill_can_fill_or_exceed_window(length):
    blocks = [{"attention": {"n_heads_in_group": 2, "window_length": 6, "num_sink_tokens": 1},
               "ffn": {"ffn_mult": 2.0}}]
    model = tiny("nemotron-nas", num_hidden_layers=1, block_configs=blocks)
    ids = torch.arange(1, length + 1)[None]
    actual = model(ids)
    torch.testing.assert_close(actual.logits, reference_forward(model, ids), rtol=2e-5, atol=1e-6)
    assert actual.past_key_values.layers[0].keys.shape[-2] == length


@pytest.mark.parametrize("unshifted,sinks,window", [(False, None, 3), (True, 1, 3), (True, 1, 0)])
@torch.inference_mode()
def test_window_and_unshifted_sink_prefill_decode(unshifted, sinks, window):
    blocks = [{"attention": {"n_heads_in_group": 2, "window_length": window,
                            "num_sink_tokens": sinks, "unshifted_sink": unshifted},
               "ffn": {"ffn_mult": 2.0}}]
    model = tiny("nemotron-nas", block_configs=blocks, num_hidden_layers=1)
    ids = torch.tensor([[1, 3, 5, 7, 9, 4, 6]])
    expected = reference_forward(model, ids)
    torch.testing.assert_close(model(ids, use_cache=False).logits, expected, rtol=2e-5, atol=1e-6)
    cache = None
    outputs = []
    for i in range(ids.shape[1]):
        result = model(ids[:, i:i + 1], past_key_values=cache)
        cache = result.past_key_values
        outputs.append(result.logits)
    torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=2e-5, atol=1e-6)


@torch.inference_mode()
def test_shifted_sink_cache_retention_rerotation_and_reorder():
    from gptqmodel.models.ports.decilm.configuration_decilm import DeciLMConfig
    from gptqmodel.models.ports.decilm.modeling_decilm import DeciLMCache

    cfg = DeciLMConfig(**config_kwargs("nemotron-nas", num_hidden_layers=1, block_configs=[
        {"attention": {"n_heads_in_group": 2, "window_length": 6, "num_sink_tokens": 1},
         "ffn": {"no_op": True}},
    ]))
    cache = DeciLMCache(cfg)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        raw = torch.randn(2, 2, 12, 8)
        values = torch.randn_like(raw)
    retained, effective_positions = [], []
    for start, count in [(0, 4), (4, 1), (5, 1), (6, 2), (8, 1), (9, 2)]:
        new = list(range(start, start + count))
        # Pinned SinkCache shifts at equality too (its growth condition is <).
        overflowing = bool(retained) and len(retained) + count >= 6
        retained = (retained + new if not overflowing else retained[:1] + retained[-(5 - count):] + new)
        effective_positions = (effective_positions + new if not overflowing else effective_positions[:1]
                               + [p - count for p in effective_positions[-(5 - count):]] + new)
        k = reference_rope(raw[:, :, start:start + count], torch.arange(start, start + count)[None],
                           cfg, start + count)
        frequencies = torch.arange(0, 8, 2).float() / 8
        angles = torch.arange(start, start + count).float()[:, None] / (10000.0 ** frequencies)
        angles = torch.cat((angles, angles), -1)[None]
        keys, vals = cache.update(k, values[:, :, start:start + count], 0,
                                 {"cos": angles.cos(), "sin": angles.sin()})
        cache.seen_tokens += count
        # New keys keep original positions; previously retained keys shift to
        # sink/window positions, as in pinned upstream SinkCache.
        positions = torch.tensor(effective_positions)[None]
        expected = reference_rope(raw[:, :, retained], positions, cfg, start + count)
        torch.testing.assert_close(keys, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(vals, values[:, :, retained], rtol=0, atol=0)
    original = cache.layers[0].keys.clone()
    cache.reorder_cache(torch.tensor([1, 0]))
    torch.testing.assert_close(cache.layers[0].keys, original.flip(0), rtol=0, atol=0)


@torch.inference_mode()
def test_cache_failure_after_first_layer_rolls_back_and_retry_matches_fresh(monkeypatch):
    model = tiny()
    ids = torch.tensor([[1, 4, 7, 9, 2]])
    cache = model(ids[:, :3]).past_key_values
    before = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    original = model.model.layers[1].forward
    triggered = []

    def fail(*args, **kwargs):
        triggered.append(True)
        raise RuntimeError("injected after first KV update")

    monkeypatch.setattr(model.model.layers[1], "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model(ids[:, 3:], past_key_values=cache)
    assert triggered and cache.get_seq_length() == 3
    for layer, (key, value) in zip(cache.layers, before):
        torch.testing.assert_close(layer.keys, key, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, value, rtol=0, atol=0)
    monkeypatch.setattr(model.model.layers[1], "forward", original)
    retry = model(ids[:, 3:], past_key_values=cache).logits
    fresh = model(ids, use_cache=False).logits[:, 3:]
    torch.testing.assert_close(retry, fresh, rtol=2e-5, atol=1e-6)


def test_invalid_architecture_fails_then_corrected_checkpoint_loads(tmp_path):
    from gptqmodel.models.ports.decilm.configuration_decilm import DeciLMConfig

    payload = config_kwargs(num_key_value_heads_per_layer=[3, 1, 4])
    (tmp_path / "config.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="divide"):
        load_model_config(tmp_path)
    payload["num_key_value_heads_per_layer"] = [2, 1, 4]
    (tmp_path / "config.json").write_text(json.dumps(payload))
    cfg = load_model_config(tmp_path)
    assert isinstance(cfg, DeciLMConfig)
    model = DeciLMQModel.loader.from_config(cfg)
    assert torch.isfinite(model(torch.tensor([[1, 3]])).logits).all()
    with pytest.raises(ValueError, match="dynamic heterogeneous"):
        model.generate(torch.tensor([[1, 3]]), max_new_tokens=1, cache_implementation="static")


@pytest.mark.parametrize("alias", ["deci", "nemotron-nas"])
@pytest.mark.parametrize("fallback", [None, {}])
def test_awq_plain_policy_quantizes_and_packs_every_linear(tmp_path, alias, fallback):
    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig
    from gptqmodel.utils.backend import BACKEND

    model = tiny(alias, hidden_size=64, intermediate_size=128)
    expected = sum(isinstance(module, torch.nn.Linear) for layer in model.model.layers for module in layer.modules())
    model.save_pretrained(tmp_path)
    save_tokenizer(tmp_path)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=fallback)
    loaded = GPTQModel.load(str(tmp_path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    assert loaded.awq_get_modules_for_scaling(loaded.model.model.layers[0], {}, {}) == []
    calibration = [{"input_ids": torch.tensor([[1, 3, 4, 3, 4, 3, 4, 3]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    norms = {name: tensor.half().clone() for name, tensor in model.state_dict().items() if "layernorm" in name}
    loaded.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    assert sum(isinstance(module, AwqTorchLinear) for module in loaded.model.modules()) == expected
    for name, tensor in norms.items():
        torch.testing.assert_close(tensor, loaded.model.state_dict()[name], rtol=0, atol=0)
    with torch.inference_mode():
        expected_logits = loaded.model(**calibration[0], use_cache=False).logits
        assert torch.isfinite(expected_logits).all()
    packed = tmp_path / "packed"
    loaded.save_quantized(str(packed))
    assert list(packed.glob("*.safetensors")) and not list(packed.glob("*.bin"))
    restored = GPTQModel.load(str(packed), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert sum(isinstance(module, AwqTorchLinear) for module in restored.model.modules()) == expected
    with torch.inference_mode():
        torch.testing.assert_close(restored.model(**calibration[0], use_cache=False).logits, expected_logits,
                                   rtol=2e-3, atol=2e-4)
