# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pinned Raon source contracts, not a DeepSeek architecture approximation."""

import copy
import gc
import json
import weakref
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from gptqmodel.models.definitions.raon1_mini import Raon1MiniQModel
from gptqmodel.models.local import LocalModelLoader
from tests.port_assets import read_port_asset


def test_definition_owns_raon_model_instead_of_checkpoint_python():
    assert isinstance(Raon1MiniQModel.loader, LocalModelLoader)
    assert not Raon1MiniQModel.require_trust_remote_code


def _config(layout="hybrid", **kwargs):
    from gptqmodel.models.ports.raon1_mini.configuration_raon1_mini import Raon1MiniConfig

    rda = {"hybrid": [1], "reverse": [2], "rda": [1, 2], "mla": []}[layout]
    values = {
        "vocab_size": 64,
        "pad_token_id": 0,
        "eos_token_id": 63,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "head_dim": 4,
        "q_lora_rank": 16,
        "kv_lora_rank": 14,
        "qk_nope_head_dim": 6,
        "qk_rope_head_dim": 2,
        "v_head_dim": 4,
        "num_experts": 3,
        "num_experts_per_token": 2,
        "moe_intermediate_size": 32,
        "rda_decay_rank": 16,
        "attn_implementation": "eager",
        "rda_config": {
            "num_heads": 8,
            "head_dim": 4,
            "conv_kernel_size": 3,
            "rda_layers": rda,
            "mla_layers": [i for i in [1, 2] if i not in rda],
        },
    }
    values.update(kwargs)
    return Raon1MiniConfig(**values)


def _model(dtype=torch.float32, layout="hybrid", **kwargs):
    from gptqmodel.models.ports.raon1_mini.modeling_raon1_mini import Raon1MiniForCausalLM

    torch.manual_seed(5173)
    model = Raon1MiniForCausalLM._from_config(_config(layout, **kwargs), dtype=dtype).eval()
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            if tensor.dtype == torch.float8_e4m3fn:
                tensor.copy_((torch.randn(tensor.shape) * 0.5).to(tensor.dtype))
            elif (
                name.endswith("A_log")
                or name.endswith("dt_bias")
                or name.endswith("eta")
                or name.endswith("e_score_correction_bias")
            ):
                tensor.copy_(torch.linspace(0.1234567, 0.4345678, tensor.numel()).reshape(tensor.shape))
            elif name.endswith("weight") and "norm" in name:
                tensor.copy_(1 + torch.randn_like(tensor) * 0.1)
            elif ".alpha_" in name:
                tensor.fill_(1.1)
            else:
                tensor.copy_(torch.randn_like(tensor) * 0.13)
        for name, tensor in model.named_buffers():
            if tensor.dtype == torch.float8_e4m3fn:
                tensor.copy_((torch.randn(tensor.shape) * 0.5).to(tensor.dtype))
            elif name.endswith("weight_scale_inv"):
                tensor.copy_(torch.arange(tensor.numel()).reshape(tensor.shape).remainder(3).add(123).to(torch.uint8))
    return model


def _source_oracle(model, tokens):
    """Independent float64 equations of pinned Raon source; no port forward/helper calls."""
    c = model.config
    w = {k: v.double() for k, v in model.state_dict().items()}

    def linear(x, name):
        return x @ w[name + ".weight"].T

    def norm(x, name, eps):
        return x / (x.square().mean(-1, keepdim=True) + eps).sqrt() * w[name + ".weight"]

    def mlp(x, name):
        g, v = linear(x, name + ".gate_proj"), linear(x, name + ".up_proj")
        g = c.activation_situ_beta * (g / c.activation_situ_beta).tanh() * g.sigmoid()
        v = c.activation_situ_linear_beta * (v / c.activation_situ_linear_beta).tanh()
        return linear(g * v, name + ".down_proj")

    def route(stream, name):
        n = c.raon_mhc_streams
        flat = stream.flatten(-2)
        inv = (flat.square().mean(-1, keepdim=True) + c.raon_mhc_selector_eps).rsqrt()
        pre = (w[name + ".alpha_pre"] * (flat @ w[name + ".phi_pre"].T * inv) + w[name + ".b_pre"]).sigmoid()
        post = (
            c.raon_mhc_post_factor
            * (w[name + ".alpha_post"] * (flat @ w[name + ".phi_post"].T * inv) + w[name + ".b_post"]).sigmoid()
        )
        z = w[name + ".res.alpha_z"] * (flat @ w[name + ".res.phi_z"].T * inv) + w[name + ".res.b_z"]
        basis = torch.zeros(n, n - 1, dtype=torch.float64)
        for k in range(n - 1):
            basis[: k + 1, k] = 1 / ((k + 1) * (k + 2)) ** 0.5
            basis[k + 1, k] = -(k + 1) / ((k + 1) * (k + 2)) ** 0.5
        perturb = basis @ z.reshape(*flat.shape[:-1], n - 1, n - 1) @ basis.T
        eye = torch.eye(n, dtype=torch.float64)
        center = (eye + torch.ones_like(eye) / n) / 2
        rho = torch.maximum(torch.zeros_like(perturb[..., 0, 0]), (-perturb / center).flatten(-2).amax(-1))
        chart = center + perturb / (1 + rho[..., None, None])
        gamma = w[name + ".res.eta"].sigmoid()
        return pre, post, (1 - gamma) * eye + gamma * chart

    b, t = tokens.shape
    streams = F.embedding(tokens, w["model.embed_tokens.weight"]).unsqueeze(2).expand(-1, -1, c.raon_mhc_streams, -1)
    for idx in range(c.num_hidden_layers):
        p = f"model.layers.{idx}"
        for site, norm_name in [("self_attn", "input_layernorm"), ("mlp", "post_attention_layernorm")]:
            pre, post, mix = route(streams, p + "." + site + "_raon_mhc")
            x = norm((pre[..., None] * streams).sum(2), p + "." + norm_name, c.rms_norm_eps)
            if site == "mlp":
                if not c.is_moe_layer(idx):
                    out = mlp(x, p + ".mlp")
                else:
                    mp = p + ".moe"
                    scores = linear(x, mp + ".gate").sigmoid()
                    top = (
                        (scores + w[mp + ".gate.e_score_correction_bias"])
                        .topk(c.num_experts_per_token, dim=-1)
                        .indices
                    )
                    probs = scores.gather(-1, top)
                    if c.moe_renormalize:
                        probs = probs / (probs.sum(-1, keepdim=True) + 1e-20)
                    out = torch.zeros_like(x)
                    for ei in range(c.num_experts):
                        gain = ((top == ei) * probs).sum(-1, keepdim=True) * c.routed_scaling_factor
                        out += gain * mlp(x, mp + f".experts.{ei}")
                    if c.num_shared_experts:
                        out += mlp(x, mp + ".shared_experts")
            elif not c.is_rda_layer(idx):
                ap = p + ".self_attn"
                q = linear(
                    norm(linear(x, ap + ".q_a_proj"), ap + ".q_a_layernorm", c.mla_latent_norm_eps), ap + ".q_b_proj"
                )
                q = q.reshape(b, t, c.num_attention_heads, c.qk_nope_head_dim + c.qk_rope_head_dim).transpose(1, 2)
                latent = linear(x, ap + ".kv_a_proj_with_mqa")
                low, extra = latent[..., : c.kv_lora_rank], latent[..., c.kv_lora_rank :]
                kv = linear(norm(low, ap + ".kv_a_layernorm", c.mla_latent_norm_eps), ap + ".kv_b_proj")
                kv = kv.reshape(b, t, c.num_attention_heads, c.qk_nope_head_dim + c.v_head_dim)
                k = torch.cat(
                    (kv[..., : c.qk_nope_head_dim], extra.unsqueeze(2).expand(-1, -1, c.num_attention_heads, -1)), -1
                ).transpose(1, 2)
                v = kv[..., c.qk_nope_head_dim :].transpose(1, 2)
                scores = (q @ k.transpose(-1, -2)) / (c.qk_nope_head_dim + c.qk_rope_head_dim) ** 0.5
                scores = scores.masked_fill(torch.ones(t, t, dtype=torch.bool).triu(1), float("-inf"))
                context = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, -1)
                if c.mla_use_output_gate:
                    context *= linear(x, ap + ".g_proj").sigmoid()
                out = linear(context, ap + ".o_proj")
            else:
                ap = p + ".self_attn"
                h, d = c.rda_num_heads, c.rda_head_dim

                def conv(name):
                    raw = linear(x, ap + "." + name + "_proj").transpose(1, 2)
                    padded = F.pad(raw, (c.rda_conv_kernel_size - 1, 0))
                    weight = w[ap + "." + name + "_conv1d.weight"]
                    value = sum(
                        padded[..., j : j + t] * weight[:, j][None, :, None] for j in range(c.rda_conv_kernel_size)
                    )
                    return F.silu(value).transpose(1, 2).reshape(b, t, h, d)

                q, k, v = conv("q"), conv("k"), conv("v")
                q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6) / d**0.5
                k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                raw_beta = linear(x, ap + ".b_proj")
                beta = raw_beta.sigmoid()
                ratio = (raw_beta + linear(x, ap + ".rda_w_proj")).sigmoid() / beta
                v *= ratio[..., None]
                decay = linear(linear(x, ap + ".f_a_proj"), ap + ".f_b_proj").reshape(b, t, h, d) + w[
                    ap + ".dt_bias"
                ].reshape(h, d)
                decay *= w[ap + ".A_log"].exp()[None, None, :, None]
                logs = (
                    -F.softplus(decay) if c.rda_gate_lower_bound is None else c.rda_gate_lower_bound * decay.sigmoid()
                )
                state = torch.zeros(b, h, d, d, dtype=torch.float64)
                outputs = []
                for ti in range(t):
                    state *= logs[:, ti].exp().unsqueeze(-2)
                    prediction = (state * k[:, ti].unsqueeze(-2)).sum(-1)
                    update = (v[:, ti] - prediction).unsqueeze(-1) * k[:, ti].unsqueeze(-2)
                    state += beta[:, ti, :, None, None] * update
                    outputs.append((state * q[:, ti].unsqueeze(-2)).sum(-1))
                output = norm(torch.stack(outputs, 1), ap + ".o_norm", c.rms_norm_eps)
                output *= linear(x, ap + ".g_proj").reshape(b, t, h, d).sigmoid()
                out = linear(output.flatten(2), ap + ".o_proj")
            streams = mix @ streams + post[..., None] * out.unsqueeze(2)
    hidden = norm(streams.mean(2), "model.norm", c.rms_norm_eps)
    return linear(hidden, "lm_head")


@pytest.mark.parametrize("layout", ["hybrid", "reverse", "rda", "mla"])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"num_experts": None},
        {"num_shared_experts": 0, "moe_renormalize": False, "routed_scaling_factor": 1.7},
        {"mla_use_output_gate": False, "raon_mhc_streams": 2},
    ],
)
def test_actual_hybrid_decoder_matches_independent_float64_equations(layout, settings):
    model = _model(layout=layout, **settings)
    tokens = torch.tensor([[1, 5, 12, 8], [2, 7, 9, 4]])
    with torch.no_grad():
        expected = _source_oracle(model, tokens)
        actual = model(tokens, use_cache=False).logits
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("layout", ["hybrid", "reverse", "rda", "mla"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_cache_chunks_causality_padding_and_greedy_beam(layout, dtype, implementation):
    model = _model(dtype=dtype, layout=layout, attn_implementation=implementation)
    tokens = torch.tensor([[1, 2, 9, 4, 6, 7], [2, 3, 11, 8, 4, 5]])
    tol = {torch.float32: 3e-6, torch.float16: 0.002, torch.bfloat16: 0.012}[dtype]
    with torch.no_grad():
        expected = model(tokens, use_cache=False).logits
        cache = None
        for start, end in [(0, 1), (1, 5), (5, 6)]:
            actual = model(tokens[:, start:end], past_key_values=cache)
            cache = actual.past_key_values
            assert cache.get_seq_length() == end
            torch.testing.assert_close(actual.logits, expected[:, start:end], atol=tol, rtol=tol)
            for state in cache.recurrent_states:
                if state is not None:
                    assert state.dtype == torch.float32
        changed = tokens.clone()
        changed[:, -1] = 16
        torch.testing.assert_close(model(changed).logits[:, :-1], expected[:, :-1], atol=tol, rtol=tol)
        padded = F.pad(tokens[:1], (2, 0))
        mask = padded.ne(0).long()
        torch.testing.assert_close(model(padded, attention_mask=mask).logits[:, 2:], expected[:1], atol=tol, rtol=tol)
        for beams in (1, 3):
            generated = model.generate(
                tokens[:, :2], max_new_tokens=3, num_beams=beams, do_sample=False, eos_token_id=None
            )
            dense = model.generate(
                tokens[:, :2], max_new_tokens=3, num_beams=beams, do_sample=False, eos_token_id=None, use_cache=False
            )
            assert torch.equal(generated, dense)


def _assert_tensors(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        assert value.dtype == expected[name].dtype, name
        if value.dtype == torch.float8_e4m3fn:
            assert torch.equal(value.view(torch.uint8), expected[name].view(torch.uint8)), name
        else:
            assert torch.equal(value, expected[name]), name


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("assign", [False, True])
def test_dense_two_safetensors_cycles_preserve_every_tensor_and_fp32_policy(tmp_path, dtype, assign):
    model = _model(dtype=dtype)
    ids = torch.tensor([[1, 7, 12, 5]])
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        for cycle in range(2):
            path = tmp_path / str(cycle)
            model.save_pretrained(path)
            assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
            model = Raon1MiniQModel.loader.from_pretrained(path, dtype=dtype, attn_implementation="eager")
            _assert_tensors(model.state_dict(), state)
            torch.testing.assert_close(model(ids, use_cache=False).logits, expected)
            for name, value in model.named_parameters():
                if name.endswith(("A_log", "dt_bias", "e_score_correction_bias", "eta")):
                    assert value.dtype == torch.float32
                else:
                    assert value.dtype == dtype
        destination = _model(dtype=dtype)
        destination.load_state_dict(state, assign=assign)
        _assert_tensors(destination.state_dict(), state)
        torch.testing.assert_close(destination(ids).logits, expected)


def test_invalid_mask_and_foreign_cache_fail_before_mutation_then_retry():
    from transformers.cache_utils import DynamicCache

    model = _model()
    ids = torch.tensor([[1, 7, 5]])
    with torch.no_grad():
        cache = model(ids[:, :2]).past_key_values
        before = copy.deepcopy(cache)
        with pytest.raises(ValueError, match="complete cached key sequence"):
            model(ids[:, 2:], past_key_values=cache, attention_mask=torch.ones(1, 1))
        assert cache.get_seq_length() == 2
        for field in ("recurrent_states", "conv_states", "key_cache", "value_cache"):
            for a, b in zip(getattr(cache, field), getattr(before, field)):
                if a is not None:
                    assert torch.equal(a, b)
        with pytest.raises(TypeError, match="hybrid"):
            model(ids, past_key_values=DynamicCache())
        readonly = model(ids[:, 2:], past_key_values=cache, use_cache=False)
        assert cache.get_seq_length() == 2
        retried = model(ids[:, 2:], past_key_values=cache)
        torch.testing.assert_close(retried.logits, readonly.logits)
        torch.testing.assert_close(retried.logits, model(ids).logits[:, 2:], atol=3e-6, rtol=1e-5)


def test_kernel_one_recurrent_only_layout_has_empty_conv_history():
    config = _config("rda").rda_config
    config["conv_kernel_size"] = 1
    model = _model(layout="rda", rda_config=config)
    ids = torch.tensor([[1, 7, 5]])
    with torch.no_grad():
        prefix = model(ids[:, :1])
        assert prefix.past_key_values.conv_states[0].shape[-1] == 0
        actual = model(ids[:, 1:], past_key_values=prefix.past_key_values)
        torch.testing.assert_close(actual.logits, model(ids).logits[:, 1:], atol=3e-6, rtol=1e-5)


def _write_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel({"<pad>": 0, "<unk>": 1, **{f"t{i}": i for i in range(2, 64)}}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>").save_pretrained(path)


def _hostile(path):
    marker = path / "checkpoint_executed"
    code = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint code executed')\n"
    for filename in ["remote_config.py", "remote_model.py", "remote_tokenizer.py", "custom_generate/generate.py"]:
        target = path / filename
        target.parent.mkdir(exist_ok=True)
        target.write_text(code)
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {
        "AutoConfig": "remote_config.RemoteConfig",
        "AutoModelForCausalLM": "remote_model.RemoteModel",
    }
    (path / "config.json").write_text(json.dumps(payload))
    tok = json.loads((path / "tokenizer_config.json").read_text())
    tok.update(
        tokenizer_class="RemoteTokenizer",
        auto_map={"AutoTokenizer": ["remote_tokenizer.RemoteTokenizer", "remote_tokenizer.RemoteTokenizer"]},
    )
    (path / "tokenizer_config.json").write_text(json.dumps(tok))
    return marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_model_tokenizer_hostile_maps_fail_repair_retry_and_two_reloads(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel
    from gptqmodel.models.ports.raon1_mini.tokenization_raon1_mini import Raon1MiniTokenizer
    from gptqmodel.utils.hf import load_model_config

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    model = _model()
    model.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = _hostile(tmp_path)
    assert type(load_model_config(tmp_path, trust_remote_code=trust)) is type(model.config)
    raw = (tmp_path / "tokenizer.json").read_bytes()
    (tmp_path / "tokenizer.json").write_text("broken")
    with pytest.raises(Exception, match="expected|parse|tokenizer|value|key"):
        GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert not marker.exists()
    (tmp_path / "tokenizer.json").write_bytes(raw)
    ids = torch.tensor([[1, 4, 6, 9]])
    with torch.no_grad():
        expected = model(ids).logits
        for cycle in range(3):
            loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
            assert isinstance(loaded, Raon1MiniQModel)
            assert isinstance(loaded.tokenizer.tokenizer, Raon1MiniTokenizer)
            assert loaded.trust_remote_code is False
            torch.testing.assert_close(loaded.model(ids).logits, expected)
            assert not marker.exists()
            if cycle < 2:
                tmp_path = tmp_path / f"reload{cycle}"
                loaded.model.save_pretrained(tmp_path)
                loaded.tokenizer.save_pretrained(tmp_path)


MX = {"quant_method": "mxfp8", "weight_block_size": [1, 32], "activation_scheme": "dynamic", "dequantize": False}


def test_all_u8_exponent_bytes_match_independent_e8m0_encoding_including_nan():
    from gptqmodel.models.ports.raon1_mini.modeling_raon1_mini import dequantize_mxfp8

    exponents = torch.arange(256, dtype=torch.uint8).reshape(256, 1)
    weights = torch.ones(256, 32, dtype=torch.float8_e4m3fn)
    expected = torch.tensor([2.0 ** (e - 127) if e < 255 else float("nan") for e in range(256)])
    actual = dequantize_mxfp8(weights, exponents, torch.float32)
    torch.testing.assert_close(actual, expected[:, None].expand(-1, 32), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_every_e4m3_e8m0_byte_pair_and_two_safetensors_cycles_match_bit_oracle(tmp_path, dtype):
    from safetensors.torch import load_file, save_file

    from gptqmodel.models.ports.raon1_mini.modeling_raon1_mini import (
        Raon1MiniRoutedLinear,
        dequantize_mxfp8,
    )

    decoded = []
    for byte in range(256):
        sign = -1.0 if byte & 128 else 1.0
        exponent, fraction = (byte >> 3) & 15, byte & 7
        if exponent == 15 and fraction == 7:
            value = float("nan")
        elif exponent == 0:
            value = sign * fraction * 2.0**-9
        else:
            value = sign * (1.0 + fraction / 8.0) * 2.0 ** (exponent - 7)
        decoded.append(value)
    expected = (
        torch.tensor(
            [
                [value * 2.0 ** (exponent - 127) if exponent < 255 else float("nan") for value in decoded]
                for exponent in range(256)
            ],
            dtype=torch.float64,
        )
        .reshape(-1, 32)
        .to(dtype)
    )
    payload = torch.arange(256, dtype=torch.uint8).reshape(8, 32).repeat(256, 1)
    scales = torch.arange(256, dtype=torch.uint8).repeat_interleave(8).reshape(-1, 1)
    projection = Raon1MiniRoutedLinear(32, 2048, compressed=True)
    projection.weight.view(torch.uint8).copy_(payload)
    projection.weight_scale_inv.copy_(scales)
    projection.to(dtype=dtype)
    for cycle in range(3):
        assert projection.weight.dtype == torch.float8_e4m3fn
        assert projection.weight_scale_inv.dtype == torch.uint8
        assert projection.weight.element_size() == projection.weight_scale_inv.element_size() == 1
        assert torch.equal(projection.weight.view(torch.uint8), payload)
        assert torch.equal(projection.weight_scale_inv, scales)
        actual = dequantize_mxfp8(projection.weight, projection.weight_scale_inv, dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(torch.signbit(actual[expected == 0]), torch.signbit(expected[expected == 0]))
        assert set(projection.state_dict()) == {"weight", "weight_scale_inv"}
        if cycle < 2:
            checkpoint = tmp_path / f"cycle{cycle}.safetensors"
            save_file(projection.state_dict(), str(checkpoint))
            state = load_file(str(checkpoint))
            projection = Raon1MiniRoutedLinear(32, 2048, compressed=True)
            projection.load_state_dict(state, assign=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mxfp8_byte_exponents_independent_oracle_mutation_and_no_retained_dense_storage(dtype, monkeypatch):
    import gptqmodel.models.ports.raon1_mini.modeling_raon1_mini as port

    projection = port.Raon1MiniRoutedLinear(64, 16, compressed=True)
    values = torch.linspace(-3, 3, 16 * 64).reshape(16, 64).to(torch.float8_e4m3fn)
    exponents = torch.arange(32).reshape(16, 2).remainder(9).add(119).to(torch.uint8)
    projection.weight.data.copy_(values)
    projection.weight_scale_inv.copy_(exponents)
    x = torch.linspace(-0.8, 0.6, 3 * 64).reshape(3, 64).to(dtype)
    refs = []
    original = port.dequantize_mxfp8

    def capture(*args):
        result = original(*args)
        refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(port, "dequantize_mxfp8", capture)
    for cycle in range(3):
        if cycle == 1:
            projection.weight_scale_inv[3, 1] += 2
            projection.weight.data[3, 35] = 2.0
        factor = torch.tensor(
            [2.0 ** (int(e) - 127) for e in projection.weight_scale_inv.flatten()], dtype=torch.float64
        ).reshape(16, 2)
        dense = (projection.weight.double().reshape(16, 2, 32) * factor[..., None]).reshape(16, 64).to(dtype)
        expected = F.linear(x, dense)
        with torch.no_grad():
            actual = projection(x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert set(projection.state_dict()) == {"weight", "weight_scale_inv"}
        stored = [t for t in vars(projection).values() if isinstance(t, torch.Tensor)]
        assert stored == []
        assert projection.weight.numel() + projection.weight_scale_inv.numel() == 1056
        assert projection.weight.dtype == torch.float8_e4m3fn
        assert projection.weight_scale_inv.dtype == torch.uint8
        projection.to(dtype=dtype, device="cpu")
        assert projection.weight.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_compressed_model_two_exact_safetensors_cycles_and_source_quantize_rejection(tmp_path, dtype):
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig

    model = _model(dtype=dtype, quantization_config=MX)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    ids = torch.tensor([[1, 3, 7, 9]])
    with torch.no_grad():
        expected = model(ids).logits
        for cycle in range(2):
            path = tmp_path / str(cycle)
            model.save_pretrained(path)
            _write_tokenizer(path)
            loaded = GPTQModel.load(str(path), device="cpu", dtype=dtype, attn_implementation="eager")
            model = loaded.model
            _assert_tensors(model.state_dict(), state)
            torch.testing.assert_close(model(ids).logits, expected)
            loaded.quantize_config = AWQConfig(device="cpu", fallback=None)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            with pytest.raises(NotImplementedError, match="source re-quantization is not qualified"):
                loaded.quantize(CalibrationConfig(dataset=[{"input_ids": ids}]))
            _assert_tensors(model.state_dict(), before)
            assert loaded.quantize_config.fallback is None
            candidate = GPTQModel.load(
                str(path), dtype=dtype, quantize_config=AWQConfig(device="cpu", fallback=None, offload_to_disk=False)
            )
            with pytest.raises(NotImplementedError, match="source re-quantization is not qualified"):
                candidate.quantize(CalibrationConfig(dataset=[{"input_ids": ids}]))
            _assert_tensors(candidate.model.state_dict(), state)


def test_presence_penalty_excludes_prompt_and_is_not_frequency_penalty():
    from gptqmodel.models.ports.raon1_mini.modeling_raon1_mini import Raon1MiniPresencePenaltyLogitsProcessor

    processor = Raon1MiniPresencePenaltyLogitsProcessor(1.5, 3)
    scores = torch.zeros(1, 16)
    actual = processor(torch.tensor([[1, 2, 3, 4, 4, 5]]), scores)
    expected = scores.clone()
    expected[0, [4, 5]] = -1.5
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "settings",
    [
        {"raon_mhc_streams": 1},
        {"hidden_act": "silu"},
        {"mla_use_nope": False},
        {"num_experts_per_token": 5},
        {"rda_config": {"rda_layers": [1], "mla_layers": [1]}},
        {"quantization_config": {**MX, "weight_block_size": [128, 128]}},
        {"quantization_config": {**MX, "dequantize": True}},
    ],
)
def test_invalid_or_unimplemented_source_policies_fail_closed(settings):
    with pytest.raises(ValueError):
        _config(**settings)


@pytest.mark.parametrize("layout", ["hybrid", "reverse", "rda", "mla"])
@pytest.mark.parametrize("offload", [False, True])
def test_public_awq_all_projections_pack_infer_and_two_exact_reloads(tmp_path, layout, offload):
    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig
    from gptqmodel.utils.backend import BACKEND

    model = _model(layout=layout, dtype=torch.float16)
    source = tmp_path / "source"
    model.save_pretrained(source)
    _write_tokenizer(source)
    cfg = AWQConfig(
        bits=4,
        group_size=-1,
        sym=False,
        device="cpu",
        fallback=None,
        offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    wrapped = GPTQModel.load(str(source), quantize_config=cfg, dtype=torch.float16, attn_implementation="eager")
    ids = torch.tensor([[1, 4, 7, 12, 9, 6, 13, 17]])
    linear_names = {
        name
        for name, module in wrapped.model.named_modules()
        if isinstance(module, torch.nn.Linear) and name != "lm_head"
    }
    # AWQ scale search folds into the two decoder norms and compensates the
    # router that reads the post-attention norm; every other non-linear tensor
    # must be preserved exactly.
    folded = {
        f"model.layers.{i}.{norm}.weight"
        for i in range(len(model.model.layers))
        for norm in ("input_layernorm", "post_attention_layernorm", "moe.gate")
    }
    dense = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if name.rsplit(".", 1)[0] not in linear_names and name not in folded
    }
    captured = []

    def capture(layer, args):
        if not captured:
            captured.append(args[0].detach().clone())

    handle = wrapped.model.model.layers[0].register_forward_pre_hook(capture)
    wrapped.quantize(
        CalibrationConfig(
            dataset=[{"input_ids": ids, "attention_mask": torch.ones_like(ids)}],
            concat_size=None,
            min_length=1,
            batch_size=1,
        ),
        backend=BACKEND.AWQ_TORCH,
    )
    handle.remove()
    torch.testing.assert_close(
        captured[0], F.embedding(ids, model.model.embed_tokens.weight).unsqueeze(2).expand(-1, -1, 4, -1)
    )
    assert cfg.fallback is None
    packed = {name for name, module in wrapped.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert packed == linear_names
    with torch.no_grad():
        expected = wrapped.model(ids).logits
        assert torch.isfinite(expected).all()
        state = {name: value.clone() for name, value in wrapped.model.state_dict().items() if not value.is_meta}
        # Scale search is active: at least one decoder norm absorbed a scale.
        assert any(not torch.equal(state[name], model.state_dict()[name]) for name in folded if name in state)
        state.update(dense)
        for name, value in dense.items():
            if not wrapped.model.state_dict()[name].is_meta:
                assert torch.equal(wrapped.model.state_dict()[name], value), name
        for cycle in range(2):
            target = tmp_path / f"packed{cycle}"
            wrapped.save(str(target))
            wrapped = GPTQModel.load(
                str(target), device="cpu", backend=BACKEND.AWQ_TORCH, dtype=torch.float16, attn_implementation="eager"
            )
            _assert_tensors(wrapped.model.state_dict(), state)
            torch.testing.assert_close(wrapped.model(ids).logits, expected, atol=0, rtol=0)
            assert list(target.glob("*.safetensors")) and not list(target.glob("*.bin"))


@pytest.mark.parametrize("layout", ["hybrid", "reverse", "rda", "mla"])
def test_cache_reorder_and_reset_are_equivalent_to_fresh_sequences(layout):
    model = _model(layout=layout)
    ids = torch.tensor([[1, 7, 3], [2, 9, 4]])
    indices = torch.tensor([1, 1, 0])
    with torch.no_grad():
        cache = model(ids[:, :2]).past_key_values
        cache.reorder_cache(indices)
        actual = model(ids[indices, 2:], past_key_values=cache)
        expected = model(ids[indices])
        torch.testing.assert_close(actual.logits, expected.logits[:, 2:], atol=3e-6, rtol=1e-5)
        cache.reset()
        assert not cache.has_previous_state and cache.get_seq_length() == 0
        for store in (cache.key_cache, cache.value_cache, cache.recurrent_states, cache.conv_states):
            assert all(value is None for value in store)
        torch.testing.assert_close(model(ids, past_key_values=cache).logits, model(ids).logits)


def test_optional_outputs_embeddings_loss_and_selected_logits():
    model = _model()
    ids = torch.tensor([[1, 4, 8, 6]])
    with torch.no_grad():
        out = model(ids, labels=ids, output_hidden_states=True, use_cache=False)
        assert len(out.hidden_states) == 3
        torch.testing.assert_close(
            out.loss, F.cross_entropy(out.logits[:, :-1].float().reshape(-1, 64), ids[:, 1:].flatten())
        )
        torch.testing.assert_close(model(inputs_embeds=model.get_input_embeddings()(ids)).logits, out.logits)
        assert len(model(ids, use_cache=False, return_dict=False)) == 1
        torch.testing.assert_close(model(ids, logits_to_keep=1).logits, out.logits[:, -1:])
        torch.testing.assert_close(model(ids, logits_to_keep=torch.tensor([0, 2])).logits, out.logits[:, [0, 2]])


def test_saved_generation_sampling_presence_and_multiple_eos_policies(tmp_path):
    from transformers import GenerationConfig

    model = _model()
    ids = torch.tensor([[1, 7, 5]])
    policy = GenerationConfig(
        do_sample=True,
        temperature=0.9,
        top_k=9,
        top_p=0.88,
        presence_penalty=1.5,
        max_new_tokens=4,
        pad_token_id=0,
        eos_token_id=None,
    )
    model.generation_config = policy
    before = copy.deepcopy(policy.to_dict())
    with torch.no_grad():
        torch.manual_seed(317)
        generated = model.generate(ids, return_dict_in_generate=True, output_scores=True)
        reference = ids.clone()
        torch.manual_seed(317)
        for step, actual_scores in enumerate(generated.scores):
            logits = _source_oracle(model, reference).float()[:, -1]
            seen = set(reference[0, ids.shape[1] :].tolist())
            for token in seen:
                logits[0, token] -= 1.5
            logits /= 0.9
            logits[logits < logits.topk(9).values[:, -1:]] = float("-inf")
            ordered, order = logits.sort(dim=-1)
            remove = ordered.softmax(-1).cumsum(-1) <= 1 - 0.88
            remove[:, -1] = False
            logits = logits.scatter(1, order, ordered.masked_fill(remove, float("-inf")))
            torch.testing.assert_close(actual_scores, logits, atol=5e-6, rtol=3e-5)
            token = torch.multinomial(logits.softmax(-1), 1)
            reference = torch.cat((reference, token), -1)
            assert torch.equal(reference, generated.sequences[:, : ids.shape[1] + step + 1])
        assert policy.to_dict() == before
        greedy = model.generate(ids, do_sample=False, presence_penalty=0, max_new_tokens=2)
        stop = int(greedy[0, ids.shape[1]])
        stopped = model.generate(
            ids, do_sample=False, presence_penalty=0, eos_token_id=[62 if stop != 62 else 61, stop], max_new_tokens=4
        )
        assert stopped.shape[1] == ids.shape[1] + 1
        for cycle in range(2):
            path = tmp_path / str(cycle)
            model.save_pretrained(path)
            model = Raon1MiniQModel.loader.from_pretrained(path, attn_implementation="eager")
            assert model.generation_config.presence_penalty == 1.5
            torch.manual_seed(317)
            assert torch.equal(model.generate(ids), generated.sequences)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_pinned_actual_release_tokenizer_ids_chat_security_and_reload(tmp_path, monkeypatch, trust, bypass):
    import hashlib

    from gptqmodel.utils.hf import load_hf_tokenizer

    port = Path(__file__).parents[1] / "gptqmodel/models/ports/raon1_mini"
    reference = json.loads(read_port_asset(port / "tokenizer_reference.json"))
    raw = read_port_asset(port / "tokenizer.json")
    assert hashlib.sha256(raw).hexdigest() == reference["tokenizer_sha256"]
    (tmp_path / "tokenizer.json").write_bytes(raw)
    (tmp_path / "tokenizer_config.json").write_bytes(read_port_asset(port / "released_tokenizer_config.json"))
    (tmp_path / "chat_template.jinja").write_bytes(read_port_asset(port / "released_chat_template.jinja"))
    config = _config(vocab_size=259840, pad_token_id=259200, eos_token_id=[259202, 259200])
    config.save_pretrained(tmp_path)
    marker = _hostile(tmp_path)
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    tok = load_hf_tokenizer(str(tmp_path), model_config=config, trust_remote_code=trust)
    for row in reference["cases"]:
        assert tok.encode(row["text"], add_special_tokens=False) == row["ids"]
        assert tok.decode(row["ids"], skip_special_tokens=False) == row["decoded"]
    assert tok.eos_token_id == 259202 and tok.pad_token_id == 259200
    messages = [{"role": "user", "content": "안녕하세요?"}]
    expected = "<|im_start|>user\n안녕하세요?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) == expected
    assert tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ).endswith("<think></think>\n\n")
    for cycle in range(2):
        target = tmp_path / f"tokenizer{cycle}"
        tok.save_pretrained(target)
        tok = load_hf_tokenizer(str(target), model_config=config, trust_remote_code=trust)
        for row in reference["cases"]:
            assert tok.encode(row["text"], add_special_tokens=False) == row["ids"]
    assert not marker.exists()
