# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""K2-Horizon package ownership and real dense/MoVA runtime regressions."""

import json
import math

import pytest
import torch
import torch.nn.functional as F

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.k2_horizon import K2HorizonQModel
from gptqmodel.utils.hf import load_model_config


def tiny_config(**overrides):
    from gptqmodel.models.ports.k2_horizon import K2HorizonConfig

    values = {
        "vocab_size": 43, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4, "rope_head_dim": 4,
        "num_experts": 0, "num_experts_per_tok": 0, "query_key_norm": False,
        "max_position_embeddings": 32, "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    }
    values.update(overrides)
    return K2HorizonConfig(**values)


def variant_config(variant, **overrides):
    values = {}
    if variant not in {"dense", "yarn"}:
        values.update(num_experts=3, num_experts_per_tok=2, moe_intermediate_size=12,
                      num_shared_experts=1, mlp_only_layers=[0], moe_gate_bias=True,
                      router_score_func="sigmoid", router_scaling_factor=2.5, norm_topk_prob=True)
    if variant == "mova":
        values.update(mova_num_experts=4, mova_num_experts_per_tok=2,
                      attention_gate_func="softplus", layernorm_num_groups=2)
    if variant == "moe":
        values.update(rope_head_dim=2)
    if variant == "qknorm":
        values.update(query_key_norm=True, layernorm_num_groups=2, attention_gate_func="silu")
    if variant == "yarn":
        values.update(head_dim=8, rope_head_dim=8, max_position_embeddings=128, rope_theta=1000000.0,
                      rope_parameters={"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 8,
                                       "beta_fast": 128, "beta_slow": 4, "truncate": True,
                                       "attention_factor": 1.2772588722239782})
    values.update(overrides)
    return tiny_config(**values)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(1717)
        yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("trust,bypass", [(False, False), (True, True)])
def test_checkpoint_code_never_owns_k2_config(tmp_path, monkeypatch, trust, bypass):
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")
    marker = tmp_path / "executed"
    (tmp_path / "evil.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote')\n")
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "k2_horizon", "auto_map": {"AutoConfig": "evil.Config", "AutoModelForCausalLM": "evil.Model"},
    }))
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config).__module__.startswith("gptqmodel.models.ports.k2_horizon.")
    assert not marker.exists()


@pytest.mark.parametrize("variant", ["dense", "mova", "moe", "qknorm", "yarn"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_forward_cache_generation_and_safetensors_roundtrip(tmp_path, variant, backend):
    from gptqmodel.models.ports.k2_horizon import K2HorizonForCausalLM

    config = variant_config(variant)
    config._attn_implementation = backend
    model = K2HorizonForCausalLM(config).eval()
    ids = torch.tensor([[1, 5, 9, 13, 17], [0, 0, 3, 7, 11]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
    with torch.no_grad():
        full = model(ids, attention_mask=mask, use_cache=False).logits
        first = model(ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
        assert first.past_key_values.get_seq_length() == 3
        continued = model(ids[:, 3:], attention_mask=mask, past_key_values=first.past_key_values)
        assert continued.past_key_values is first.past_key_values
        assert continued.past_key_values.get_seq_length() == 5
        torch.testing.assert_close(continued.logits, full[:, 3:], rtol=3e-5, atol=3e-6)
        assert torch.isfinite(full).all() and full.std() > 0.01
        generated = model.generate(ids[:1], max_new_tokens=3, do_sample=False, eos_token_id=None)
        no_cache = model.generate(ids[:1], max_new_tokens=3, do_sample=False, use_cache=False, eos_token_id=None)
        assert torch.equal(generated, no_cache)
        assert generated.shape == (1, 8)

    model.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors"))
    assert not list(tmp_path.glob("*.bin"))
    snapshot = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    for _ in range(2):
        model = K2HorizonQModel.loader.from_pretrained(tmp_path, dtype=torch.float32, attn_implementation=backend).eval()
        assert set(model.state_dict()) == set(snapshot)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, snapshot[name], rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(model(ids, attention_mask=mask, use_cache=False).logits, full, rtol=0, atol=0)
        model.save_pretrained(tmp_path)


@pytest.mark.parametrize(
    "variant,trust,bypass", [("dense", False, False), ("mova", True, True), ("moe", True, False), ("moe", False, True)]
)
def test_saved_weights_ignore_malicious_auto_map_on_retry(tmp_path, monkeypatch, variant, trust, bypass):
    from gptqmodel.models.ports.k2_horizon import K2HorizonConfig, K2HorizonForCausalLM

    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")
    model = K2HorizonForCausalLM(variant_config(variant)).eval()
    model.save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "evil.Config", "AutoModelForCausalLM": "evil.Model"}
    (tmp_path / "config.json").write_text(json.dumps(payload))
    marker = tmp_path / "executed"
    (tmp_path / "evil.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote')\n")
    with torch.no_grad():
        expected = model(torch.tensor([[1, 3, 8]]), use_cache=False).logits
    (tmp_path / "model.safetensors").rename(tmp_path / "missing.safetensors")
    with pytest.raises(OSError):
        K2HorizonQModel.loader.from_pretrained(tmp_path, trust_remote_code=trust)
    (tmp_path / "missing.safetensors").rename(tmp_path / "model.safetensors")
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is K2HorizonConfig
    loaded = K2HorizonQModel.loader.from_pretrained(tmp_path, trust_remote_code=trust, dtype=torch.float32).eval()
    with torch.no_grad():
        # Security/retry is the point here; the reload may select a different attention kernel.
        torch.testing.assert_close(
            loaded(torch.tensor([[1, 3, 8]]), use_cache=False).logits, expected, rtol=1e-5, atol=1e-6
        )
    assert type(loaded) is K2HorizonForCausalLM
    assert not marker.exists()


@pytest.mark.parametrize("variant", ["dense", "mova"])
def test_attention_implementation_is_switchable_through_tf5_interface(variant):
    from gptqmodel.models.ports.k2_horizon import K2HorizonForCausalLM

    config = variant_config(variant)
    config._attn_implementation = "eager"
    model = K2HorizonForCausalLM(config).eval()
    ids = torch.tensor([[1, 4, 8, 2]])
    with torch.no_grad():
        eager = model(ids, use_cache=False).logits
        model.set_attn_implementation("sdpa")
        assert model.config._attn_implementation == "sdpa"
        torch.testing.assert_close(model(ids, use_cache=False).logits, eager, rtol=1e-5, atol=1e-5)


def test_output_capture_and_cache_failure_retry():
    from gptqmodel.models.ports.k2_horizon import K2HorizonForCausalLM

    config = variant_config("mova")
    config._attn_implementation = "eager"
    model = K2HorizonForCausalLM(config).eval()
    ids = torch.tensor([[1, 4, 8]])
    with pytest.raises(ValueError, match="exactly one"):
        model(ids, inputs_embeds=model.model.embed_tokens(ids))
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True, output_router_logits=True, labels=ids)
        assert len(out.hidden_states) == 3
        assert len(out.attentions) == 2
        assert len(out.router_logits) == 1
        assert torch.isfinite(out.loss) and torch.isfinite(out.aux_loss)
        assert out.past_key_values.get_seq_length() == 3
        # An independent request must start a new cache after the warm request.
        fresh = model(ids[:, :1])
        assert fresh.past_key_values.get_seq_length() == 1
        assert fresh.past_key_values is not out.past_key_values


def _reference_logits(model, ids, position_ids):
    """Functional source-derived oracle, with token-wise routing and complex RoPE.

    No model forward, router, RMSNorm, rotation, or attention helper is called.
    IFM's pinned MoVA implementation is the semantic source (port README).
    """
    c = model.config
    weights = model.state_dict()

    def linear(x, path, bias=True):
        return F.linear(x, weights[path + ".weight"], weights.get(path + ".bias") if bias else None)

    def norm(x, path, groups):
        shape = x.shape
        grouped = x.float().reshape(*shape[:-1], groups, -1)
        grouped = grouped / torch.sqrt(grouped.square().mean(-1, keepdim=True) + c.rms_norm_eps)
        return (grouped.reshape(shape) * weights[path + ".weight"]).to(x.dtype)

    def routed(x, path, count, topk, mova=False):
        flat = x.reshape(-1, c.hidden_size)
        router = path + (".v_router" if mova else ".gate")
        logits = linear(flat, router, bias=False).float()
        scores = logits.sigmoid() if c.router_score_func == "sigmoid" else logits.softmax(-1)
        choice = scores + weights.get(router + ".bias", torch.zeros(count))
        selected = choice.argsort(dim=-1, descending=True)[:, :topk]
        rows = []
        for token, selected_row in enumerate(selected):
            mass = scores[token, selected_row]
            if (mova and topk > 1) or (not mova and c.norm_topk_prob):
                mass = mass / mass.sum()
            terms = []
            for weight, expert in zip(mass, selected_row.tolist()):
                if mova:
                    value = F.silu(linear(flat[token], f"{path}.v_experts.{expert}"))
                else:
                    value = mlp(flat[token], f"{path}.experts.{expert}")
                terms.append(value * (weight * c.router_scaling_factor).to(value.dtype))
            rows.append(torch.stack(terms).sum(0))
        return torch.stack(rows).reshape(*x.shape[:-1], -1)

    def mlp(x, path):
        return linear(F.silu(linear(x, path + ".gate_proj")) * linear(x, path + ".up_proj"), path + ".down_proj")

    def rotate(x):
        # Upstream stores split real/imaginary halves; partial RoPE rotates
        # the first pairs, leaving the remaining pairs unchanged.
        half = c.head_dim // 2
        pairs = torch.complex(x[..., :half].float(), x[..., half:].float())
        dim = c.rope_head_dim or c.head_dim
        theta = c.rope_parameters["rope_theta"]
        inv = theta ** (-torch.arange(0, dim, 2).float() / dim)
        scale = 1.0
        if c.rope_parameters["rope_type"] == "yarn":
            params = c.rope_parameters
            original = params["original_max_position_embeddings"]
            low = max(math.floor(dim * math.log(original / (params["beta_fast"] * 2 * math.pi)) / (2 * math.log(theta))), 0)
            high = min(math.ceil(dim * math.log(original / (params["beta_slow"] * 2 * math.pi)) / (2 * math.log(theta))), dim - 1)
            ramp = ((torch.arange(dim // 2).float() - low) / (high - low if high != low else 0.001)).clamp(0, 1)
            inv = inv * (1 - ramp) + inv / params["factor"] * ramp
            scale = params["attention_factor"]
        angles = position_ids[:, None, :, None] * inv[None, None, None, :]
        phase = torch.polar(torch.ones_like(angles) * scale, angles)
        pairs = torch.cat([pairs[..., :dim // 2] * phase, pairs[..., dim // 2:]], dim=-1)
        return torch.cat([pairs.real, pairs.imag], dim=-1).to(x.dtype)

    hidden = F.embedding(ids, weights["model.embed_tokens.weight"])
    length = ids.shape[1]
    allowed = torch.ones(length, length, dtype=torch.bool).tril()
    for index in range(c.num_hidden_layers):
        root = f"model.layers.{index}"
        x = norm(hidden, root + ".input_layernorm", c.layernorm_num_groups)
        path = root + ".self_attn"
        sparse = index not in c.mlp_only_layers and c.num_experts > 0 and (index + 1) % c.decoder_sparse_step == 0
        q = linear(x, path + ".q_proj")
        k = linear(x, path + ".k_proj")
        if c.query_key_norm:
            q = norm(q, path + ".q_norm", c.num_attention_heads)
            k = norm(k, path + ".k_norm", c.num_key_value_heads)
        v = routed(x, path, c.mova_num_experts, c.mova_num_experts_per_tok, mova=True) if sparse and c.mova_num_experts else linear(x, path + ".v_proj")
        q = rotate(q.view(*x.shape[:-1], c.num_attention_heads, c.head_dim).transpose(1, 2))
        k = rotate(k.view(*x.shape[:-1], c.num_key_value_heads, c.head_dim).transpose(1, 2))
        v = v.view(*x.shape[:-1], c.num_key_value_heads, c.head_dim).transpose(1, 2)
        k = k.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, dim=1)
        v = v.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, dim=1)
        score = (q @ k.transpose(-1, -2)) / math.sqrt(c.head_dim)
        score = score.masked_fill(~allowed, torch.finfo(score.dtype).min)
        attention = (score.float().softmax(-1).to(x.dtype) @ v).transpose(1, 2)
        if c.attention_gate_func:
            gate = linear(x, path + ".gate_proj").view_as(attention)
            gate = F.silu(gate) if c.attention_gate_func == "silu" else F.softplus(gate, beta=math.log(2))
            attention = attention * gate
        hidden = hidden + linear(attention.reshape(*x.shape[:-1], -1), path + ".o_proj")
        x = norm(hidden, root + ".post_attention_layernorm", c.layernorm_num_groups)
        if sparse:
            output = routed(x, root + ".mlp", c.num_experts, c.num_experts_per_tok)
            if c.num_shared_experts:
                output = output + mlp(x, root + ".mlp.shared_experts")
        else:
            output = mlp(x, root + ".mlp")
        hidden = hidden + output
    return linear(norm(hidden, "model.norm", c.layernorm_num_groups), "lm_head")


@pytest.mark.parametrize("variant", ["dense", "mova", "moe", "qknorm", "yarn"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_logits_match_independent_source_derived_oracle(variant, backend):
    from gptqmodel.models.ports.k2_horizon import K2HorizonForCausalLM

    config = variant_config(variant)
    config._attn_implementation = backend
    model = K2HorizonForCausalLM(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.uniform_(0.6, 1.4)
            else:
                parameter.normal_(0, 0.17)
        ids = torch.tensor([[1, 4, 9, 15], [3, 7, 11, 18]])
        positions = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
        expected = _reference_logits(model, ids, positions)
        actual = model(ids, position_ids=positions, use_cache=False).logits
        assert actual.std() > 0.1
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("variant", ["dense", "mova", "moe"])
def test_low_precision_cache_and_checkpoint_dtype(tmp_path, dtype, variant):
    from gptqmodel.models.ports.k2_horizon import K2HorizonForCausalLM

    model = K2HorizonForCausalLM(variant_config(variant)).to(dtype).eval()
    ids = torch.tensor([[1, 4, 9, 15]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        cached = model(ids[:, :2], use_cache=True)
        actual = model(ids[:, 2:], past_key_values=cached.past_key_values).logits
        assert expected.dtype is dtype
        torch.testing.assert_close(actual, expected[:, 2:], rtol=0.02, atol=0.002)
    model.save_pretrained(tmp_path)
    loaded = K2HorizonQModel.loader.from_pretrained(tmp_path, dtype=dtype).eval()
    assert all(t.dtype is dtype for t in loaded.parameters())
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected, rtol=0, atol=0)


def test_dense_router_output_and_loss_are_finite():
    from gptqmodel.models.ports.k2_horizon import K2HorizonForCausalLM

    model = K2HorizonForCausalLM(variant_config("dense")).eval()
    ids = torch.tensor([[1, 5, 9]])
    result = model(ids, labels=ids, output_router_logits=True, use_cache=False)
    assert torch.isfinite(result.loss)
    assert result.aux_loss.item() == 0
    assert result.router_logits == ()


@pytest.mark.parametrize("variant,trust,bypass", [("dense", False, False), ("mova", True, True)])
def test_public_gptqmodel_load_uses_local_classes_and_native_tokenizer(tmp_path, monkeypatch, variant, trust, bypass):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import GPTQModel
    from gptqmodel.models.ports.k2_horizon import K2HorizonForCausalLM

    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")
    model = K2HorizonForCausalLM(variant_config(variant)).bfloat16().eval()
    model.save_pretrained(tmp_path)
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
                                  **{str(i): i for i in range(3, 43)}}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["auto_map"] = {"AutoConfig": "evil.Config", "AutoModelForCausalLM": "evil.Model"}
    config_path.write_text(json.dumps(config))
    marker = tmp_path / "executed"
    (tmp_path / "evil.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote')\n")
    ids = torch.tensor([[1, 5, 9]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
    loaded = GPTQModel.load(str(tmp_path), device="cpu", trust_remote_code=trust, dtype=torch.bfloat16)
    assert isinstance(loaded, K2HorizonQModel)
    assert type(loaded.model) is K2HorizonForCausalLM
    assert loaded.trust_remote_code is False
    assert loaded.tokenizer is not None
    assert not marker.exists()
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(ids, use_cache=False).logits, expected, rtol=0, atol=0)
        generated = loaded.model.generate(ids, max_new_tokens=2, do_sample=False, eos_token_id=None)
        assert generated.shape == (1, 5)


@pytest.mark.cpu
@pytest.mark.parametrize("variant", ["dense", "mova"])
def test_public_awq_scaled_native_pack_save_reload(tmp_path, variant):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.models.ports.k2_horizon import K2HorizonForCausalLM
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    model = K2HorizonForCausalLM(variant_config(variant, intermediate_size=32, moe_intermediate_size=16)).half().eval()
    model.save_pretrained(tmp_path)
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
                                  **{str(i): i for i in range(3, 43)}}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                           eos_token="[EOS]").save_pretrained(tmp_path)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
                             backend=BACKEND.TORCH, dtype=torch.float16)
    calibration = [{"input_ids": torch.tensor([[1, 5, 8, 13, 20, 9, 25, 6]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    quantized_names = {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert len(quantized_names) == (14 if variant == "dense" else 28)
    # Scale search ran: at least one decoder norm absorbed a searched scale.
    state = wrapper.model.state_dict()
    assert any(
        not torch.equal(state[name], value)
        for name, value in model.state_dict().items()
        if name.endswith("layernorm.weight")
    )
    if variant == "mova":
        assert "model.layers.1.self_attn.v_experts.3" in quantized_names
        assert "model.layers.1.mlp.experts.2.down_proj" in quantized_names
        assert isinstance(wrapper.model.model.layers[1].mlp.gate, torch.nn.Linear)
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
        assert torch.isfinite(expected).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors"))
    assert not list(saved.glob("*.bin"))
    loaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert {name for name, module in loaded.model.named_modules() if isinstance(module, AwqTorchLinear)} == quantized_names
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(**calibration[0], use_cache=False).logits, expected, rtol=2e-3, atol=2e-4)
