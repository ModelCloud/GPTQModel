# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent XVERSE equations and actual public loading/quantization cycles."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import SafetensorError
from transformers import GenerationConfig
from transformers.cache_utils import DynamicCache, StaticCache

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.definitions.xverse import XverseQModel
from gptqmodel.models.ports.xverse import XverseConfig, XverseForCausalLM, XverseTokenizer
from gptqmodel.models.ports.xverse.modeling_xverse import XverseRMSNorm
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import CalibrationConfig
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.hf import load_model_config
from tests.port_assets import has_port_asset, read_port_asset, read_port_json


PORT = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/xverse"
FIXTURES = PORT / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def asset(family, name):
    return read_port_asset(FIXTURES / f"{family}-{name}")


def tiny_config(**kwargs):
    values = {
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 32,
        "max_tokenizer_truncation": 32,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 1,
        "_attn_implementation": "eager",
    }
    values.update(kwargs)
    return XverseConfig(**values)


def tiny_model(**kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(571)
        model = XverseForCausalLM(tiny_config(**kwargs)).eval()
        with torch.no_grad():
            for name, value in model.named_parameters():
                if "layernorm" in name or name == "model.norm.weight":
                    value.uniform_(0.65, 1.45)
    return model


def source_checkpoint(path, family="XVERSE-7B-Chat", **kwargs):
    path.mkdir(parents=True, exist_ok=True)
    original = json.loads(asset(family, "config.json"))
    overrides = {
        "vocab_size": original["vocab_size"],
        "bos_token_id": original["bos_token_id"],
        "eos_token_id": original["eos_token_id"],
        "pad_token_id": original.get("pad_token_id"),
        "num_key_value_heads": 2,
    }
    overrides.update(kwargs)
    model = tiny_model(**overrides)
    gen_file = FIXTURES / f"{family}-generation_config.json"
    model.generation_config = (
        GenerationConfig.from_dict(read_port_json(gen_file))
        if has_port_asset(gen_file)
        else GenerationConfig.from_model_config(model.config)
    )
    model.save_pretrained(path)
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        (path / name).write_bytes(asset(family, name))
    marker = path / "EXECUTED"
    poison = (
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('untrusted Python executed')\n"
    )
    (path / "untrusted.py").write_text(poison)
    for name in ["config.json", "tokenizer_config.json"]:
        metadata = json.loads((path / name).read_text())
        metadata["auto_map"] = {
            "AutoConfig": "untrusted.Config",
            "AutoModelForCausalLM": "untrusted.Model",
            "AutoTokenizer": ["untrusted.Slow", "untrusted.Fast"],
        }
        if name == "tokenizer_config.json":
            metadata["tokenizer_class"] = "Untrusted"
        (path / name).write_text(json.dumps(metadata))
    (path / "custom_generate").mkdir(exist_ok=True)
    (path / "custom_generate/generate.py").write_text(poison)
    return model, marker


def linear(x, state, name):
    return F.linear(
        x,
        state[name + ".weight"].double(),
        state.get(name + ".bias", None).double() if name + ".bias" in state else None,
    )


def oracle(model, ids, mask=None, positions=None, past=None, return_cache=False):
    """Source equations in float64, with no calls to owned model helpers."""
    c, state = model.config, model.state_dict()
    x = F.embedding(ids, state["model.embed_tokens.weight"].double())
    batch, length = ids.shape
    prefix = past[0][0].shape[-2] if past is not None else 0
    total = prefix + length
    positions = torch.arange(prefix, total).expand(batch, -1) if positions is None else positions
    outputs, routers, keys, values = [], [], [], []

    def norm(x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.rms_norm_eps) * state[name + ".weight"].double()

    def mlp(x, name):
        return linear(
            F.silu(linear(x, state, name + ".gate_proj")) * linear(x, state, name + ".up_proj"),
            state,
            name + ".down_proj",
        )

    for index in range(c.num_hidden_layers):
        p = f"model.layers.{index}."
        y = norm(x, p + "input_layernorm")
        head = c.hidden_size // c.num_attention_heads
        q, k, v = [
            linear(y, state, p + "self_attn." + name).view(batch, length, heads, head).transpose(1, 2)
            for name, heads in [
                ("q_proj", c.num_attention_heads),
                ("k_proj", c.num_key_value_heads),
                ("v_proj", c.num_key_value_heads),
            ]
        ]
        theta = c.rope_theta
        pos = positions.double()
        if c.xverse_rope_type == "dynamic_256k" and total > c.max_position_embeddings:
            theta *= (total / (c.max_position_embeddings / 2) - 1) ** (head / (head - 2))
        if c.xverse_rope_scaling is not None:
            scale = c.xverse_rope_scaling
            if scale["type"] == "linear":
                pos = pos / scale["factor"]
            elif positions.max() + 1 > c.max_position_embeddings:
                theta *= (
                    scale["factor"] * (positions.max().item() + 1) / c.max_position_embeddings - (scale["factor"] - 1)
                ) ** (head / (head - 2))
        inv = theta ** (-torch.arange(0, head, 2).double() / head)
        angle = pos[..., None] * inv
        angle = torch.cat([angle, angle], -1)[:, None]

        def rotate(x):
            left, right = x.chunk(2, -1)
            return x * angle.cos() + torch.cat([-right, left], -1) * angle.sin()

        q, k = rotate(q), rotate(k)
        if past is not None:
            k = torch.cat([past[index][0], k], -2)
            v = torch.cat([past[index][1], v], -2)
        keys.append(k)
        values.append(v)
        k, v = [z.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1) for z in [k, v]]
        scores = q @ k.transpose(-1, -2) / head**0.5
        allowed = (torch.arange(total)[None, :] <= torch.arange(prefix, total)[:, None])[None, None].expand(
            batch, 1, -1, -1
        )
        if mask is not None:
            allowed = allowed & mask[:, None, None, :].bool()
        probs = scores.masked_fill(~allowed, -1e100).softmax(-1)
        y = (probs @ v).transpose(1, 2).reshape(batch, length, c.hidden_size)
        x = x + linear(y, state, p + "self_attn.o_proj")
        y = norm(x, p + "post_attention_layernorm")
        if c.num_experts:
            flat = y.reshape(-1, c.hidden_size)
            router = linear(flat, state, p + "mlp.router")
            routers.append(router)
            probs = router.softmax(-1)
            selected = torch.argsort(probs, dim=-1, descending=True)[:, : c.moe_top_k]
            top = probs.gather(1, selected)
            top = top / (top.sum(-1, keepdim=True) + 1e-6)
            terms = torch.zeros_like(flat)
            for token in range(flat.shape[0]):
                for slot in range(c.moe_top_k):
                    expert = selected[token, slot].item()
                    terms[token] += top[token, slot] * mlp(flat[token], p + f"mlp.experts.{expert}")
            y = terms.reshape_as(y)
            if c.num_shared_experts:
                y = y + mlp(norm(x, p + "post_attention_layernorm"), p + "mlp.shared_experts")
        else:
            y = mlp(y, p + "mlp")
        x = x + y
        outputs.append(x)
    hidden = norm(x, "model.norm")
    result = linear(hidden, state, "lm_head"), routers, keys
    return (*result, list(zip(keys, values))) if return_cache else result


@pytest.mark.parametrize("moe,shared", [(0, None), (4, None), (4, 2)])
@pytest.mark.parametrize("kv,bias", [(4, False), (2, True), (1, False)])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_independent_full_decoder_equations_and_rotated_kv(moe, shared, kv, bias, implementation):
    model = tiny_model(
        num_experts=moe,
        moe_top_k=2,
        num_shared_experts=shared,
        num_key_value_heads=kv,
        attention_bias=bias,
        rope_theta=500000 if moe else 10000,
        _attn_implementation=implementation,
    )
    ids = torch.tensor([[4, 9, 12, 8, 7], [7, 11, 14, 16, 18]])
    mask = torch.ones_like(ids)
    mask[1, 0] = 0
    with torch.inference_mode():
        out = model(
            ids, attention_mask=mask, output_router_logits=bool(moe), output_hidden_states=True, use_cache=True
        )
        expected, routers, keys = oracle(model, ids, mask)
        valid = mask.bool()
        torch.testing.assert_close(out.logits[valid].double(), expected[valid], rtol=2e-5, atol=2e-7)
        for index, key in enumerate(keys):
            torch.testing.assert_close(
                out.past_key_values.layers[index].keys[valid[:, None, :, None].expand_as(key)].double(),
                key[valid[:, None, :, None].expand_as(key)],
                rtol=2e-5,
                atol=2e-7,
            )
        for actual, expected in zip(out.router_logits or () if moe else (), routers):
            torch.testing.assert_close(
                actual.double()[mask.reshape(-1).bool()], expected[mask.reshape(-1).bool()], rtol=2e-5, atol=2e-7
            )
        changed = ids.clone()
        changed[:, -1] = 29
        torch.testing.assert_close(
            model(changed, attention_mask=mask).logits[:, :-1][mask[:, :-1].bool()],
            out.logits[:, :-1][mask[:, :-1].bool()],
            rtol=1e-5,
            atol=1e-7,
        )


@pytest.mark.parametrize("moe", [0, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_source_norm_cast_order(moe, dtype):
    config = tiny_config(num_experts=moe, moe_top_k=2)
    norm = XverseRMSNorm(config).to(dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.31, 1.73, config.hidden_size))
    x = torch.linspace(-3.82, 5.29, 3 * config.hidden_size).reshape(3, -1).to(dtype)
    scaled = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + config.rms_norm_eps)
    expected = norm.weight * scaled.to(dtype) if moe else (norm.weight * scaled).to(dtype)
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["dense", "long", "moe", "moe_linear", "moe_dynamic"])
def test_source_rotary_policy_independent_oracle_and_reload(tmp_path, variant):
    moe = variant.startswith("moe")
    options = {"num_experts": 4 if moe else 0, "moe_top_k": 2, "max_position_embeddings": 4}
    if variant == "long":
        options.update(xverse_rope_type="dynamic_256k", rope_theta=500000)
    if variant in {"moe_linear", "moe_dynamic"}:
        options["rope_scaling"] = {"type": variant.split("_")[1], "factor": 2.0}
    model = tiny_model(**options)
    ids = torch.tensor([[4, 11, 12, 13, 8, 7, 5]])
    with torch.inference_mode():
        expected, _, _ = oracle(model, ids)
        torch.testing.assert_close(model(ids).logits.double(), expected, rtol=3e-5, atol=3e-7)
    model.save_pretrained(tmp_path)
    loaded = XverseQModel.loader.from_pretrained(tmp_path, attn_implementation="eager")
    assert loaded.config.xverse_rope_type == model.config.xverse_rope_type
    assert loaded.config.xverse_rope_scaling == model.config.xverse_rope_scaling
    # Dynamic MoE frequency state intentionally retains its source's warm-cache base.
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, model(ids).logits, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["dense", "long", "moe_linear", "moe_dynamic"])
def test_independent_streaming_equations_cross_original_context_without_rewriting_old_keys(variant):
    moe = variant.startswith("moe")
    options = {"num_experts": 4 if moe else 0, "moe_top_k": 2, "max_position_embeddings": 4, "num_key_value_heads": 2}
    if variant == "long":
        options.update(xverse_rope_type="dynamic_256k", rope_theta=500000)
    if moe:
        options["rope_scaling"] = {"type": variant.split("_")[1], "factor": 2.0}
    model = tiny_model(**options)
    ids = torch.tensor([[4, 8, 11, 9, 7, 5, 12]])
    native_cache = reference_cache = None
    with torch.inference_mode():
        for start, end in [(0, 1), (1, 5), (5, 6), (6, 7)]:
            old_keys = [] if native_cache is None else [layer.keys.clone() for layer in native_cache.layers]
            expected, _, keys, reference_cache = oracle(
                model, ids[:, start:end], past=reference_cache, return_cache=True
            )
            actual = model(ids[:, start:end], past_key_values=native_cache, use_cache=True)
            native_cache = actual.past_key_values
            torch.testing.assert_close(actual.logits.double(), expected, rtol=3e-5, atol=3e-7)
            for index, reference in enumerate(keys):
                torch.testing.assert_close(native_cache.layers[index].keys.double(), reference, rtol=3e-5, atol=3e-7)
                if old_keys:
                    torch.testing.assert_close(
                        native_cache.layers[index].keys[..., :start, :], old_keys[index], rtol=0, atol=0
                    )


@pytest.mark.parametrize("variant", ["dense", "long", "moe_linear", "moe_dynamic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_source_float32_rotary_at_released_context_boundary_and_dtype_movement(variant, dtype):
    from gptqmodel.models.ports.xverse.modeling_xverse import XverseRotaryEmbedding

    moe = variant.startswith("moe")
    options = {
        "num_experts": 4 if moe else 0,
        "moe_top_k": 2,
        "hidden_size": 512,
        "intermediate_size": 768,
        "num_attention_heads": 4,
        "max_position_embeddings": 32768 if variant == "long" else 8192,
    }
    if variant == "long":
        options.update(xverse_rope_type="dynamic_256k", rope_theta=500000)
    if moe:
        options["rope_scaling"] = {"type": variant.split("_")[1], "factor": 2.0}
    config = tiny_config(**options)
    module = XverseRotaryEmbedding(config).to(dtype)
    original = module.inv_freq.clone()
    module = module.half().bfloat16().float().to(dtype)
    torch.testing.assert_close(module.inv_freq, original, rtol=0, atol=0)
    assert module.inv_freq.dtype == torch.float32
    length = 262144 if variant == "long" else 8192 if variant == "dense" else 16384
    positions = torch.tensor([[length - 3, length - 1]])
    dim = 128
    base = config.rope_theta
    pos = positions.float()
    if variant == "long":
        base *= (length / (config.max_position_embeddings / 2) - 1) ** (dim / (dim - 2))
    if variant == "moe_linear":
        pos /= 2
    if variant == "moe_dynamic":
        base *= (2 * length / config.max_position_embeddings - 1) ** (dim / (dim - 2))
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    expected = pos[..., None] * inv
    expected = torch.cat([expected, expected], -1)[:, None]
    cos, sin = module(torch.zeros(1, 4, 2, dim, dtype=dtype), positions, length)
    torch.testing.assert_close(cos, expected.cos().to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(sin, expected.sin().to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"num_key_value_heads": 0}, "head"),
        ({"num_key_value_heads": 3}, "head"),
        ({"hidden_size": 28}, "even"),
        ({"num_experts": True}, "num_experts"),
        ({"num_experts": 2, "moe_top_k": 3}, "moe_top_k"),
        ({"num_shared_experts": 0}, "num_shared_experts"),
        ({"pretraining_tp": 3}, "pretraining_tp"),
        ({"rope_scaling": {"type": "linear", "factor": 2.0}}, "MoE"),
        ({"num_experts": 4, "moe_top_k": 2, "rope_scaling": {"type": "linear", "factor": 1.0}}, "factor"),
        ({"xverse_rope_type": "foreign"}, "xverse_rope_type"),
        ({"num_experts": 4, "moe_top_k": 2, "xverse_rope_type": "dynamic_256k"}, "dense"),
    ],
)
def test_invalid_geometry_fails_before_model_construction_and_corrected_retry(kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        tiny_config(**kwargs)
    assert tiny_model()(torch.tensor([[4, 8]])).logits.shape == (1, 2, 48)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_cache_continuation_greedy_beam_and_two_safetensors_cycles(tmp_path, moe, dtype, implementation):
    model = tiny_model(
        num_experts=4 if moe else 0, moe_top_k=2, num_shared_experts=1 if moe else None, num_key_value_heads=2
    )
    model = (
        XverseQModel.loader.from_config(model.config, dtype=dtype, attn_implementation=implementation).eval().to(dtype)
    )
    if moe:
        for layer in model.model.layers:
            layer.mlp.router.float()
    ids = torch.tensor([[4, 8, 9, 14, 12, 10, 7], [7, 4, 8, 10, 9, 5, 6]])
    mask = torch.ones_like(ids)
    mask[1, 0] = 0
    tolerance = {torch.float32: 1e-6, torch.float16: 8e-4, torch.bfloat16: 8e-3}[dtype]
    with torch.inference_mode():
        full = model(ids, attention_mask=mask, use_cache=False).logits
        cached = None
        for start, end in [(0, 1), (1, 5), (5, 6), (6, 7)]:
            out = model(ids[:, start:end], attention_mask=mask[:, :end], past_key_values=cached, use_cache=True)
            cached = out.past_key_values
            torch.testing.assert_close(
                out.logits[mask[:, start:end].bool()],
                full[:, start:end][mask[:, start:end].bool()],
                rtol=tolerance,
                atol=tolerance,
            )
        assert cached.get_seq_length() == ids.shape[1]
        legacy = tuple((layer.keys, layer.values) for layer in cached.layers)
        next_ids = torch.tensor([[8], [9]])
        old = model(next_ids, past_key_values=legacy, attention_mask=F.pad(mask, (0, 1), value=1), use_cache=True)
        assert isinstance(old.past_key_values, tuple)
        torch.testing.assert_close(
            old.logits,
            model(torch.cat([ids, next_ids], -1), attention_mask=F.pad(mask, (0, 1), value=1)).logits[:, -1:],
            rtol=tolerance,
            atol=tolerance,
        )
        generation = {
            beams: model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, num_beams=beams)
            for beams in [1, 3]
        }
        for cycle in range(2):
            saved = tmp_path / str(cycle)
            model.save_pretrained(saved)
            assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
            state = {k: v.clone() for k, v in model.state_dict().items()}
            model = XverseQModel.loader.from_pretrained(saved, dtype=dtype, attn_implementation=implementation).eval()
            for key, value in state.items():
                assert value.dtype == model.state_dict()[key].dtype
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(model(ids, attention_mask=mask, use_cache=False).logits, full, rtol=0, atol=0)
            for beams, expected in generation.items():
                assert torch.equal(
                    model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, num_beams=beams),
                    expected,
                )


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
@pytest.mark.parametrize("family", ["XVERSE-7B-Chat", "XVERSE-MoE-A4.2B", "XVERSE-Ent-A5.7B"])
def test_real_assets_public_hostile_load_config_tokenizer_and_generation(tmp_path, monkeypatch, trust, bypass, family):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    moe = family != "XVERSE-7B-Chat"
    expected, marker = source_checkpoint(
        tmp_path, family, num_experts=4 if moe else 0, moe_top_k=2, num_shared_experts=2 if moe else None
    )
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is XverseConfig
    loaded = GPTQModel.load(
        str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager"
    )
    assert type(loaded.model) is XverseForCausalLM
    assert isinstance(loaded.tokenizer.tokenizer, XverseTokenizer)
    reference_tokenizer = json.loads((PORT / "tokenizer_oracles.json").read_text())[family]
    for prompt, tokens in reference_tokenizer["ids"].items():
        assert loaded.tokenizer.tokenizer.encode(prompt, add_special_tokens=False) == tokens
        assert (
            loaded.tokenizer.tokenizer.decode(tokens, skip_special_tokens=False)
            == reference_tokenizer["decode"][prompt]
        )
    ids = torch.tensor([[4, 8, 12]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, expected(ids).logits, rtol=0, atol=0)
        assert torch.equal(
            loaded.model.generate(ids, max_new_tokens=2, do_sample=False),
            expected.generate(ids, max_new_tokens=2, do_sample=False),
        )
    assert loaded.model.generation_config.eos_token_id == expected.generation_config.eos_token_id
    if trust and bypass == "1":
        for cycle in range(2):
            destination = tmp_path / f"dense-{cycle}"
            previous = {k: v.clone() for k, v in loaded.model.state_dict().items()}
            loaded.model.save_pretrained(destination)
            loaded.tokenizer.tokenizer.save_pretrained(destination)
            loaded = GPTQModel.load(str(destination), device="cpu", dtype=torch.float32, attn_implementation="eager")
            for key, value in previous.items():
                torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
            for prompt, tokens in reference_tokenizer["ids"].items():
                assert loaded.tokenizer.tokenizer.encode(prompt, add_special_tokens=False) == tokens
            with torch.inference_mode():
                torch.testing.assert_close(loaded.model(ids).logits, expected(ids).logits, rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("family", ["XVERSE-7B-Chat", "XVERSE-MoE-A4.2B", "XVERSE-Ent-A5.7B"])
def test_corrupt_real_tokenizer_after_warm_load_fails_and_repaired_same_path_recovers(tmp_path, family):
    original, marker = source_checkpoint(tmp_path, family)
    tokenizer_path = tmp_path / "tokenizer.json"
    saved = tokenizer_path.read_bytes()
    GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    tokenizer_path.write_bytes(b"{invalid JSON")
    with pytest.raises(Exception, match="(?i)json|expect|key must"):
        GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    tokenizer_path.write_bytes(saved)
    repaired = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, attn_implementation="eager")
    expected = json.loads((PORT / "tokenizer_oracles.json").read_text())[family]["ids"]
    for prompt, ids in expected.items():
        assert repaired.tokenizer.tokenizer.encode(prompt, add_special_tokens=False) == ids
    with torch.inference_mode():
        torch.testing.assert_close(
            repaired.model(torch.tensor([[4, 8]])).logits, original(torch.tensor([[4, 8]])).logits, rtol=0, atol=0
        )
    assert tokenizer_path.read_bytes() == saved
    assert not marker.exists()


@pytest.mark.parametrize("family", ["XVERSE-7B-Chat", "XVERSE-MoE-A4.2B", "XVERSE-Ent-A5.7B"])
def test_real_serialized_tokenizer_ids_and_two_exact_reloads(tmp_path, family):
    expected = json.loads((PORT / "tokenizer_oracles.json").read_text())[family]
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        (tmp_path / name).write_bytes(asset(family, name))
    tokenizer = XverseQModel.loader.load_tokenizer(tmp_path)
    for prompt, ids in expected["ids"].items():
        assert tokenizer.encode(prompt, add_special_tokens=False) == ids
    for cycle in range(2):
        destination = tmp_path / str(cycle)
        tokenizer.save_pretrained(destination)
        tokenizer = XverseQModel.loader.load_tokenizer(destination)
        for prompt, ids in expected["ids"].items():
            assert tokenizer.encode(prompt, add_special_tokens=False) == ids
            assert tokenizer.decode(ids, skip_special_tokens=False) == expected["decode"][prompt]


@pytest.mark.parametrize("moe,shared,kv", [(0, None, 4), (4, None, 2), (4, 2, 1)])
@pytest.mark.parametrize("offload", [False, True])
def test_real_public_plain_awq_all_linears_dense_exclusions_and_two_exact_reloads(tmp_path, moe, shared, kv, offload):
    source = tmp_path / "source"
    family = "XVERSE-Ent-A5.7B" if shared else "XVERSE-MoE-A4.2B" if moe else "XVERSE-7B-Chat"
    expected, marker = source_checkpoint(
        source, family, num_experts=moe, moe_top_k=2, num_shared_experts=shared, num_key_value_heads=kv
    )
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    wrapper = GPTQModel.load(str(source), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    assert wrapper.awq_get_modules_for_scaling(wrapper.model.model.layers[0], {}, {}) == []
    expected_packed = {
        name
        for name, module in expected.named_modules()
        if isinstance(module, torch.nn.Linear) and name.startswith("model.layers.") and not name.endswith("router")
    }
    original = expected.state_dict()
    captured = []

    def capture(module, args, kwargs):
        captured.append((args[0] if args else kwargs["hidden_states"]).detach().cpu().clone())

    samples = [
        {
            "input_ids": torch.tensor([[4, 8, 11, 9, 12, 8, 5, 7]]),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[8, 7, 11, 5, 9, 6, 4, 12]]),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        },
    ]
    handle = wrapper.model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        wrapper.quantize(
            CalibrationConfig(dataset=samples, batch_size=1, min_length=1, sort="desc"), backend=BACKEND.TORCH
        )
    finally:
        handle.remove()
    assert len(captured) >= 2
    expected_inputs = [
        F.embedding(sample["input_ids"], original["model.embed_tokens.weight"].half()) for sample in samples
    ]
    for value in captured[:2]:
        assert any(torch.equal(value, reference) for reference in expected_inputs)
    packed = {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert packed == expected_packed
    state = wrapper.model.state_dict()
    for name, value in original.items():
        if name.removesuffix(".weight") not in packed:
            wanted_dtype = torch.float32 if "router" in name or "inv_freq" in name else torch.float16
            assert state[name].dtype == wanted_dtype
            torch.testing.assert_close(state[name], value.to(wanted_dtype), rtol=0, atol=0)
    assert all(name + ".weight" not in state and state[name + ".qweight"].dtype == torch.int32 for name in packed)
    with torch.inference_mode():
        logits = wrapper.model(**samples[0], use_cache=False).logits
        assert torch.isfinite(logits).all()
        generated = wrapper.model.generate(**samples[0], do_sample=False, max_new_tokens=2)
    for cycle in range(2):
        destination = tmp_path / f"packed-{cycle}"
        before = {k: v.clone() for k, v in wrapper.model.state_dict().items()}
        wrapper.save(str(destination))
        assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
        wrapper = GPTQModel.load(str(destination), device="cpu", backend=BACKEND.TORCH, dtype=torch.float16)
        assert type(wrapper.model) is XverseForCausalLM
        assert wrapper.model.config.num_experts == moe
        for key, value in before.items():
            assert wrapper.model.state_dict()[key].dtype == value.dtype
            torch.testing.assert_close(wrapper.model.state_dict()[key], value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(wrapper.model(**samples[0], use_cache=False).logits, logits, rtol=0, atol=0)
            assert torch.equal(wrapper.model.generate(**samples[0], do_sample=False, max_new_tokens=2), generated)
    assert not marker.exists()
    del expected, original, state, wrapper


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("assign", [False, True])
def test_failed_safetensor_load_repair_state_assignment_and_subfolder(tmp_path, moe, assign):
    model = tiny_model(num_experts=4 if moe else 0, moe_top_k=2, num_shared_experts=1 if moe else None)
    nested = tmp_path / "nested"
    model.save_pretrained(nested)
    payload = nested / "model.safetensors"
    saved = payload.read_bytes()
    payload.write_bytes(b"not a safetensor")
    with pytest.raises((SafetensorError, OSError), match="header|Header|safetensor"):
        XverseQModel.loader.from_pretrained(tmp_path, subfolder="nested")
    payload.write_bytes(saved)
    loaded = XverseQModel.loader.from_pretrained(tmp_path, subfolder="nested", attn_implementation="eager").eval()
    destination = tiny_model(num_experts=4 if moe else 0, moe_top_k=2, num_shared_experts=1 if moe else None)
    with torch.no_grad():
        destination.model.embed_tokens.weight.add_(0.217)
    destination.load_state_dict(loaded.state_dict(), assign=assign)
    ids = torch.tensor([[4, 8, 12]])
    with torch.inference_mode():
        torch.testing.assert_close(destination(ids).logits, model(ids).logits, rtol=0, atol=0)
    with pytest.raises(ValueError, match="complete cached sequence"):
        loaded(ids[:, :1], attention_mask=torch.ones(1, 8), use_cache=True)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, model(ids).logits, rtol=0, atol=0)


def test_invalid_foreign_cache_rejection_then_reuse():
    model = tiny_model()
    ids = torch.tensor([[4, 8]])
    with pytest.raises(ValueError, match="DynamicCache"):
        model(ids, past_key_values=StaticCache(config=model.config, max_cache_len=16))
    cache = DynamicCache()
    assert model(ids, past_key_values=cache).past_key_values.get_seq_length() == 2


def test_generation_policy_is_loaded_saved_and_effective(tmp_path):
    model = tiny_model()
    ids = torch.tensor([[4, 8, 11]])
    with torch.inference_mode():
        eos = model(ids).logits[0, -1].argmax().item()
    model.generation_config = GenerationConfig(
        eos_token_id=[eos, 3], pad_token_id=1, bos_token_id=2, do_sample=True, top_p=0.83, temperature=0.7
    )
    initial = copy.deepcopy(model.generation_config.to_dict())
    for cycle in range(2):
        saved = tmp_path / str(cycle)
        model.save_pretrained(saved)
        model = XverseQModel.loader.from_pretrained(saved).eval()
        assert model.generation_config.eos_token_id == [eos, 3]
        assert model.generation_config.top_p == 0.83
        assert model.generate(ids, do_sample=False, max_new_tokens=4).shape[-1] == ids.shape[-1] + 1
        assert model.generate(ids, do_sample=False, eos_token_id=None, max_new_tokens=3).shape[-1] == ids.shape[-1] + 3
    assert model.generation_config.to_dict() == initial


def test_asset_sha_manifest():
    sources = json.loads((PORT / "source_manifest.json").read_text())
    for family, metadata in sources.items():
        if family == "code_license":
            assert hashlib.sha256((PORT / "LICENSE").read_bytes()).hexdigest() == metadata["sha256"]
            continue
        for name, expected in metadata.get("sha256", {}).items():
            path = FIXTURES / f"{family}-{name}"
            if has_port_asset(path):
                assert hashlib.sha256(read_port_asset(path)).hexdigest() == expected


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("index_name", ["model.safetensors.index.json", "pytorch_model.bin.index.json"])
def test_released_index_metadata_preserves_exported_rope_without_loading_pickle(tmp_path, persistent, index_name):
    config = tiny_config(num_experts=4, moe_top_k=2)
    raw = config.to_dict()
    del raw["xverse_rope_persistent"]
    raw["_attn_implementation"] = "flash_attention_2"
    (tmp_path / "config.json").write_text(json.dumps(raw))
    weight_map = {"model.embed_tokens.weight": "never-read.bin"}
    if persistent:
        weight_map["model.layers.0.self_attn.rotary_emb.inv_freq"] = "never-read.bin"
    (tmp_path / index_name).write_text(json.dumps({"weight_map": weight_map}))
    loaded = load_model_config(tmp_path)
    assert loaded.xverse_rope_persistent is persistent
    assert loaded._attn_implementation is None
    assert XverseQModel.loader.from_config(loaded).config._attn_implementation == "sdpa"
    with pytest.raises(ValueError, match="Flash Attention 2"):
        XverseQModel.loader.from_config(loaded, attn_implementation="flash_attention_2")
    loaded.save_pretrained(tmp_path / "roundtrip")
    assert XverseQModel.loader.load_config(tmp_path / "roundtrip").xverse_rope_persistent is persistent


@pytest.mark.parametrize("moe", [False, True])
def test_pretraining_tp_dense_slices_match_and_quantization_fails_before_mutation(moe):
    unsliced = tiny_model(num_experts=4 if moe else 0, moe_top_k=2, num_key_value_heads=2)
    sliced = tiny_model(num_experts=4 if moe else 0, moe_top_k=2, num_key_value_heads=2, pretraining_tp=2)
    sliced.load_state_dict(unsliced.state_dict())
    ids = torch.tensor([[4, 8, 11, 12]])
    with torch.inference_mode():
        torch.testing.assert_close(sliced(ids).logits, unsliced(ids).logits, rtol=1e-5, atol=1e-7)
    before = {k: v.clone() for k, v in sliced.state_dict().items()}
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="pretraining_tp"):
        XverseQModel.pre_quantize(SimpleNamespace(model=sliced), sliced.model.layers[0])
    for key, value in before.items():
        torch.testing.assert_close(sliced.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("tuple_output", [False, True])
def test_optional_outputs_labels_router_loss_and_four_dimensional_mask(moe, tuple_output):
    model = tiny_model(num_experts=4 if moe else 0, moe_top_k=2, num_shared_experts=1 if moe else None)
    ids = torch.tensor([[4, 8, 12, 11], [8, 9, 12, 4]])
    with torch.inference_mode():
        output = model(
            ids,
            labels=ids,
            output_attentions=True,
            output_hidden_states=True,
            output_router_logits=moe,
            return_dict=not tuple_output,
        )
        if tuple_output:
            assert torch.isfinite(output[0])
            return
        assert len(output.hidden_states) == 3
        assert len(output.attentions) == 2
        expected, routers, _ = oracle(model, ids)
        expected_loss = F.cross_entropy(expected[:, :-1].reshape(-1, expected.shape[-1]), ids[:, 1:].reshape(-1))
        if moe:
            probs = torch.cat(routers).softmax(-1)
            chosen = probs.argsort(-1, descending=True)[:, :2]
            selection_counts = torch.stack([(chosen == index).double().mean(0) for index in range(4)], -1)
            aux = (selection_counts * probs.mean(0)).sum() * 4
            torch.testing.assert_close(output.aux_loss.double(), aux, rtol=2e-5, atol=1e-6)
            expected_loss += model.config.router_aux_loss_coef * aux
        torch.testing.assert_close(output.loss.double(), expected_loss, rtol=2e-5, atol=1e-6)
        for attention in output.attentions:
            assert torch.count_nonzero(torch.triu(attention, diagonal=1)) == 0
        mask = torch.zeros(2, 1, 4, 4).masked_fill(torch.ones(4, 4).triu(1).bool(), torch.finfo(torch.float32).min)
        torch.testing.assert_close(
            model(inputs_embeds=model.model.embed_tokens(ids), attention_mask=mask).logits,
            output.logits,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("variant", ["dense", "long", "moe"])
def test_source_chat_prompt_bytes_truncation_and_native_generate(tmp_path, variant):
    family = "XVERSE-MoE-A4.2B" if variant == "moe" else "XVERSE-7B-Chat"
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        (tmp_path / name).write_bytes(asset(family, name))
    tokenizer = XverseQModel.loader.load_tokenizer(tmp_path)
    model = tiny_model(
        vocab_size=100534,
        num_experts=4 if variant == "moe" else 0,
        moe_top_k=2,
        xverse_rope_type="dynamic_256k" if variant == "long" else "default",
        max_position_embeddings=256,
        max_tokenizer_truncation=200,
    )
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "好。"},
        {"role": "user", "content": "Hello world!"},
    ]
    expected = json.loads((PORT / "tokenizer_oracles.json").read_text())[family]["chat_ids"]
    model.generation_config.max_new_tokens = 2
    model.generation_config.do_sample = True
    ids = model._build_chat_input(tokenizer, messages, 2)
    assert ids.tolist() == [expected]
    policy = copy.deepcopy(model.generation_config)
    policy.do_sample = False
    with torch.inference_mode():
        expected_response = tokenizer.decode(
            model.generate(ids, generation_config=policy)[0, ids.shape[-1] :], skip_special_tokens=True
        )
    assert model.chat(tokenizer, messages, generation_config=policy) == expected_response
    assert model.generation_config.to_dict() != policy.to_dict()
    if variant != "dense":
        extended = [
            {"role": "system", "content": "安全"},
            *messages[:2],
            {"role": "exec", "content": "2+2"},
            {"role": "assistant", "content": "4"},
            messages[-1],
        ]
        expected_extended = json.loads((PORT / "tokenizer_oracles.json").read_text())[family]["extended_chat_ids"]
        assert model._build_chat_input(tokenizer, extended, 2).tolist() == [expected_extended]
        with pytest.raises(ValueError, match="system message must be first"):
            model._build_chat_input(tokenizer, [messages[0], extended[0], messages[-1]], 2)
    with pytest.raises(NotImplementedError, match="streaming"):
        model.chat(tokenizer, messages, stream=True)
    with pytest.raises(ValueError, match="role"):
        model._build_chat_input(tokenizer, [{"role": "unknown", "content": "bad"}])
    messages[-1]["content"] *= 50
    assert model._build_chat_input(tokenizer, messages, 250).shape[-1] == 128
