# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import math

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.cache_utils import StaticCache

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.models.definitions.nanbeige import NanbeigeQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.nanbeige.configuration_nanbeige import NanbeigeConfig
from gptqmodel.models.ports.nanbeige.modeling_nanbeige import (
    NanbeigeForCausalLM,
    NanbeigeHyperConnectionModule,
    NanbeigeRMSNorm,
)
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.hf import load_model_config


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_loops": 2,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_position_embeddings": 32,
    }
    values.update(kwargs)
    return NanbeigeConfig(**values)


def model(**kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(918)
        result = NanbeigeForCausalLM(config(**kwargs)).eval()
        with torch.no_grad():
            for module in result.modules():
                if isinstance(module, NanbeigeRMSNorm):
                    module.weight.copy_(torch.linspace(0.73, 1.19, module.weight.numel()))
                if isinstance(module, NanbeigeHyperConnectionModule):
                    module.mapping_proj.weight.normal_(std=0.13)
                    if module.bias is not None:
                        module.bias.uniform_(-0.3, 0.4)
                    else:
                        module.mapping_proj.bias.uniform_(-0.3, 0.4)
                    for name in ("alpha_pre", "alpha_post", "alpha_res"):
                        getattr(module, name).fill_(0.17)
    return result


IDS = torch.tensor([[1, 7, 4, 11, 3, 9], [1, 8, 5, 6, 10, 12]])


def tokenizer(path):
    raw = Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t3"))
    result = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="t0", bos_token="t1", eos_token="t2", unk_token="t3", model_max_length=32
    )
    result.save_pretrained(path)


def hostile_checkpoint(path, **kwargs):
    original = model(**kwargs)
    original.save_pretrained(path)
    tokenizer(path)
    marker = path / "EXECUTED"
    for filename in ("configuration_evil.py", "modeling_evil.py", "tokenization_evil.py"):
        (path / filename).write_text(
            f'from pathlib import Path\nPath({str(marker)!r}).write_text("executed")\nraise AssertionError("checkpoint code ran")\n'
        )
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {
        "AutoConfig": "configuration_evil.EvilConfig",
        "AutoModel": "modeling_evil.EvilModel",
        "AutoModelForCausalLM": "modeling_evil.EvilModel",
    }
    (path / "config.json").write_text(json.dumps(payload))
    tok = json.loads((path / "tokenizer_config.json").read_text())
    tok["auto_map"] = {"AutoTokenizer": ["tokenization_evil.EvilTokenizer", None]}
    (path / "tokenizer_config.json").write_text(json.dumps(tok))
    return original, marker


def linear(x, module):
    return F.linear(x, module.weight, module.bias)


def norm(x, module):
    value = x.float()
    return (value / torch.sqrt(value.square().mean(-1, keepdim=True) + module.variance_epsilon)).to(
        x.dtype
    ) * module.weight


def hc_input(x, owner):
    n = owner.num_residual_streams
    streams = x.view(*x.shape[:2], n, -1)
    if owner.enable_mhc:
        projected = F.linear(x, owner.mapping_proj.weight)
        r = 1 / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) / math.sqrt(x.shape[-1]) + 1e-6)
        alpha = torch.cat([owner.alpha_pre.expand(n), owner.alpha_post.expand(n), owner.alpha_res.expand(n * n)])
        if owner.mhc_identity_nohresparam:
            alpha = alpha[: 2 * n]
        logits = r * projected * alpha + owner.bias
        pre, post = logits[..., :n].sigmoid(), logits[..., n : 2 * n].sigmoid() * 2
        if owner.enable_h_res_identity:
            residual_map = torch.eye(n, dtype=x.dtype).expand(*x.shape[:2], n, n)
        else:
            logits = logits[..., 2 * n :].view(*x.shape[:2], n, n)
            residual_map = (logits - logits.amax(-1, keepdim=True)).exp()
            for _ in range(owner.sinkhorn_iterations):
                residual_map = residual_map / residual_map.sum(-1, keepdim=True).clamp_min(1e-6)
                residual_map = residual_map / residual_map.sum(-2, keepdim=True).clamp_min(1e-6)
    else:
        logits = linear(x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6), owner.mapping_proj).tanh()
        static_pre = torch.zeros(n, dtype=x.dtype)
        static_pre[owner.layer_idx % n] = 1
        pre = logits[..., :n] * owner.alpha_pre + static_pre
        post = logits[..., n : 2 * n] * owner.alpha_post + 1
        residual_map = logits[..., 2 * n :].view(*x.shape[:2], n, n) * owner.alpha_res + torch.eye(n, dtype=x.dtype)
    residual = streams if owner.enable_h_res_identity else residual_map @ streams
    return (streams * pre.unsqueeze(-1)).sum(-2), residual, post


def convert_streams(x, hidden, target):
    current = x.shape[-1] // hidden
    streams = x.view(*x.shape[:2], current, hidden)
    if target % current == 0:
        return streams.repeat_interleave(target // current, -2).flatten(-2)
    if current % target == 0:
        return streams.view(*x.shape[:2], target, current // target, hidden).mean(-2).flatten(-2)
    return streams.mean(-2, keepdim=True).expand(*x.shape[:2], target, hidden).flatten(-2)


def reference_ngram(m, ids):
    c = m.config
    words = F.embedding(ids, m.model.embed_tokens.weight)
    ngram = m.model.ngram_embeddings
    context = ids if m.model.lookup_table is None else m.model.lookup_table[ids]
    base = c.vocab_size
    if c.ngram_mod_force_prime:
        base += 1
        while any(base % d == 0 for d in range(2, math.isqrt(base) + 1)):
            base += 1
    parts = []
    previous = int(c.ngram_vocab_size_ratio * c.vocab_size)
    for table, embedder in enumerate(ngram.embedders):
        order = table // c.emb_split_num + 2
        if c.ngram_mod_force_prime:
            previous += 1
            while any(previous % d == 0 for d in range(2, math.isqrt(previous) + 1)):
                previous += 1
            modulus = previous
        else:
            modulus = int(c.ngram_vocab_size_ratio * c.vocab_size + table * 2 + 1)
        hashed = torch.empty_like(ids)
        for b in range(ids.shape[0]):
            start = 0
            for t in range(ids.shape[1]):
                value = int(context[b, t])
                for lag in range(1, order):
                    prior = int(context[b, t - lag]) if t - lag >= start else 0
                    value += prior * pow(base, lag, modulus)
                hashed[b, t] = value % modulus
                if int(ids[b, t]) == c.eos_token_id:
                    start = t + 1
        parts.append(F.embedding(hashed, embedder.weight))
    if c.ngram_fused_mode == "concat":
        features = linear(torch.cat(parts, -1), ngram.concat_proj)
        combined = words + features
    else:
        total = sum(linear(part, proj) for part, proj in zip(parts, ngram.post_projs))
        features = total / len(parts)
        combined = (words + total) / (1 + len(parts))
    return (words if c.skip_ngram_for_input else combined), features


def reference_logits(m, ids, mask=None, trace=None):
    """Independent equations, without invoking model forward/attention/cache helpers."""
    c = m.config
    x = F.embedding(ids, m.model.embed_tokens.weight)
    features = None
    if m.model.ngram_embeddings is not None:
        x, features = reference_ngram(m, ids)
    batch, seq, _ = x.shape
    logical_loops = len(c.loop_loss_weights) + 1 if c.loop_loss_weights else c.num_loops
    if c.enable_double_loop_split:
        middle = c.loop_middle_layers
        first = (c.num_hidden_layers - middle) // 2
        order = [(i, None) for i in range(first)]
        order += [
            (i, repeat)
            for repeat in range((c.num_hidden_layers + middle) // middle)
            for i in range(first, first + middle)
        ]
        order += [(i, None) for i in range(first + middle, c.num_hidden_layers)]
        logical_loops = 1
    else:
        order = [(i, None) for i in range(c.num_hidden_layers)]
    shared, anchors = {}, []
    for _ in range(logical_loops):
        for index, repeat in order:
            layer = m.model.layers[index]
            if str(index) in m.model.ngram_layer_fusion:
                fusion = m.model.ngram_layer_fusion[str(index)]
                gate_hidden = linear(x, fusion.hidden_down_proj) if fusion.hidden_down_proj else x
                gate_hidden = norm(gate_hidden, fusion.hidden_norm)
                key = norm(linear(features, fusion.key_proj), fusion.ngram_norm)
                gate = (gate_hidden * key).sum(-1, keepdim=True) / math.sqrt(fusion.fusion_size)
                gate = (gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid()
                added = gate * linear(features, fusion.value_proj)
                x = x + (linear(added, fusion.output_proj) if fusion.output_proj else added)
            if c.enable_hyper_connection:
                x = convert_streams(x, c.hidden_size, layer.num_residual_streams)
                attn_hc = (
                    layer.self_attn_hc
                    if repeat in (None, 0) or not c.mhc_diff_for_loop
                    else layer.self_attn_mhc_loop_hcs[repeat - 1]
                )
                mlp_hc = (
                    layer.mlp_hc
                    if repeat in (None, 0) or not c.mhc_diff_for_loop
                    else layer.mlp_mhc_loop_hcs[repeat - 1]
                )
                attn_input, residual, post = hc_input(x, attn_hc)
            else:
                attn_input = x
            a = layer.self_attn
            z = norm(attn_input, layer.input_layernorm)
            if trace is not None:
                trace.setdefault(index, []).append(z.detach().clone())
            q = linear(z, a.q_proj).view(batch, seq, c.num_attention_heads, c.head_dim).transpose(1, 2)
            k = linear(z, a.k_proj).view(batch, seq, c.num_key_value_heads, c.head_dim).transpose(1, 2)
            v = linear(z, a.v_proj).view(batch, seq, c.num_key_value_heads, c.head_dim).transpose(1, 2)
            if c.qk_layernorm:
                q, k = norm(q, a.q_layernorm), norm(k, a.k_layernorm)
            positions = torch.arange(seq, dtype=torch.float32)
            theta = c.rope_theta
            rope = c.rope_parameters
            if rope["rope_type"] == "linear":
                positions = positions / rope["factor"]
            if rope["rope_type"] == "dynamic" and seq > c.max_position_embeddings:
                theta *= (rope["factor"] * seq / c.max_position_embeddings - (rope["factor"] - 1)) ** (
                    c.head_dim / (c.head_dim - 2)
                )
            inv = 1.0 / (theta ** (torch.arange(0, c.head_dim, 2).float() / c.head_dim))
            phases = positions[:, None] * inv[None, :]
            phases = torch.cat([phases, phases], -1)
            cos, sin = phases.cos().to(x.dtype), phases.sin().to(x.dtype)
            q = q * cos + torch.cat([-q[..., c.head_dim // 2 :], q[..., : c.head_dim // 2]], -1) * sin
            k = k * cos + torch.cat([-k[..., c.head_dim // 2 :], k[..., : c.head_dim // 2]], -1) * sin
            if c.loop_share_kv and repeat is not None:
                if repeat == 0:
                    shared[index] = (k, v)
                else:
                    k, v = shared[index]
            if c.enable_depth_attention:
                grouped_q = q.view(batch, c.num_key_value_heads, -1, seq, c.head_dim).mean(2)
                keys, values = [pair[0] for pair in anchors] + [k], [pair[1] for pair in anchors] + [v]
                scores = torch.stack([(grouped_q.float() * key.float()).sum(-1) for key in keys]) / math.sqrt(
                    c.head_dim
                )
                coefficients = scores.softmax(0).to(x.dtype)
                v = sum(coefficients[i].unsqueeze(-1) * value for i, value in enumerate(values))
                if index % c.depth_attention_stride == 0:
                    anchors.append((k, v))
            k = k.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1)
            v = v.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1)
            scores = q @ k.transpose(-1, -2) / math.sqrt(c.head_dim)
            forbidden = torch.ones(seq, seq, dtype=torch.bool).triu(1)[None, None].expand(batch, 1, seq, seq)
            if mask is not None:
                forbidden = forbidden | ~mask[:, None, None, :].bool()
            scores = scores.masked_fill(forbidden, torch.finfo(x.dtype).min)
            values = scores.float().softmax(-1).to(x.dtype) @ v
            y = linear(values.transpose(1, 2).reshape(batch, seq, -1), a.o_proj)
            if c.enable_hyper_connection:
                x = (residual + post.unsqueeze(-1) * y.unsqueeze(-2)).flatten(-2)
                mlp_input, residual, post = hc_input(x, mlp_hc)
            else:
                x = x + y
                mlp_input = x
            z = norm(mlp_input, layer.post_attention_layernorm)
            y = linear(F.silu(linear(z, layer.mlp.gate_proj)) * linear(z, layer.mlp.up_proj), layer.mlp.down_proj)
            x = (residual + post.unsqueeze(-1) * y.unsqueeze(-2)).flatten(-2) if c.enable_hyper_connection else x + y
        if c.enable_hyper_connection:
            x = x.view(batch, seq, -1, c.hidden_size).mean(-2)
        if not c.skip_loop_final_norm:
            x = norm(x, m.model.norm)
    if c.skip_loop_final_norm:
        x = norm(x, m.model.norm)
    return linear(x, m.lm_head).float()


def test_definition_owns_model_and_ignores_checkpoint_trust():
    assert isinstance(NanbeigeQModel.loader, LocalModelLoader)
    assert not NanbeigeQModel.require_trust_remote_code
    assert NanbeigeQModel.fusable is None


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("kv_heads,bias", [(1, True), (2, False), (4, True)])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"qk_layernorm": True},
        {"loop_loss_weights": [0.1, 0.3]},
        {"rope_scaling": {"type": "linear", "factor": 2.0}},
        {"enable_depth_attention": True},
        {"enable_double_loop_split": True, "num_hidden_layers": 4, "loop_middle_layers": 2},
        {"enable_double_loop_split": True, "num_hidden_layers": 4, "loop_middle_layers": 2, "loop_share_kv": True},
        {"enable_hyper_connection": True},
        {"enable_hyper_connection": True, "enable_mhc": True},
        {"enable_hyper_connection": True, "enable_mhc": True, "enable_h_res_identity": True},
        {
            "enable_hyper_connection": True,
            "enable_mhc": True,
            "enable_h_res_identity": True,
            "mhc_identity_nohresparam": True,
        },
        {
            "enable_double_loop_split": True,
            "num_hidden_layers": 4,
            "loop_middle_layers": 2,
            "enable_hyper_connection": True,
            "enable_mhc": True,
            "mhc_diff_for_loop": True,
            "num_residual_streams": 2,
            "mhc_double_stream_position_for_loop": "mid",
        },
    ],
)
def test_independent_float64_source_equations(kv_heads, bias, options, attention):
    m = model(
        attn_implementation=attention, num_key_value_heads=kv_heads, attention_bias=bias, mlp_bias=bias, **options
    ).double()
    mask = torch.tensor([[1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    with torch.inference_mode():
        expected = reference_logits(m, IDS, mask)
        got = m(IDS, attention_mask=mask, use_cache=False)
    torch.testing.assert_close(got.logits, expected, atol=4e-7, rtol=4e-6)
    assert got.logits.float().std() > 0.04


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"loop_loss_weights": [0.2, 0.3]},
        {"qk_layernorm": True},
        {"enable_depth_attention": True},
        {"enable_double_loop_split": True, "num_hidden_layers": 4, "loop_middle_layers": 2},
        {"enable_double_loop_split": True, "num_hidden_layers": 4, "loop_middle_layers": 2, "loop_share_kv": True},
        {
            "enable_double_loop_split": True,
            "num_hidden_layers": 4,
            "loop_middle_layers": 2,
            "enable_hyper_connection": True,
            "enable_mhc": True,
            "num_residual_streams": 2,
            "mhc_double_stream_position_for_loop": "mid",
        },
        {"emb_neighbor_num": 3, "emb_split_num": 2, "ngram_vocab_size_ratio": 1, "emb_tp_num": 1},
    ],
)
def test_cache_prefill_single_chunk_generation_and_safetensors(tmp_path, dtype, attention, options):
    m = model(attn_implementation=attention, **options).to(dtype)
    atol = 8e-3 if dtype == torch.bfloat16 else 1e-3 if dtype == torch.float16 else 8e-7
    with torch.inference_mode():
        full = m(IDS, use_cache=False, output_hidden_states=True).logits
        prefix = m(IDS[:, :1], use_cache=True)
        continuation = m(IDS[:, 1:5], past_key_values=prefix.past_key_values, use_cache=True)
        final = m(IDS[:, 5:], past_key_values=continuation.past_key_values, use_cache=True)
        torch.testing.assert_close(full[:, 1:5], continuation.logits, atol=atol, rtol=atol)
        torch.testing.assert_close(full[:, 5:], final.logits, atol=atol, rtol=atol)
        assert final.past_key_values.get_seq_length() == IDS.shape[1]
        expected_slots = (
            m.config.num_hidden_layers
            if m.config.loop_share_kv
            else len(m.model._get_layer_execution_order()) * m.model._get_num_loops()
        )
        assert len(final.past_key_values.layers) == expected_slots
        assert all(layer.keys.shape[-2] == IDS.shape[1] for layer in final.past_key_values.layers)
        greedy = m.generate(IDS[:1, :3], max_new_tokens=3, do_sample=False, eos_token_id=None)
        manual = IDS[:1, :3].clone()
        for _ in range(3):
            token = m(manual, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            manual = torch.cat((manual, token), 1)
        torch.testing.assert_close(greedy, manual)
        beam = m.generate(IDS[:1, :3], max_new_tokens=2, num_beams=2, eos_token_id=None)
        assert beam.shape == (1, 5)
    m.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
    loaded = NanbeigeForCausalLM.from_pretrained(tmp_path, dtype=dtype, attn_implementation=attention)
    assert loaded.config._attn_implementation == attention
    for key, value in m.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(IDS, use_cache=False).logits, full, atol=0, rtol=0)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_hostile_config_model_tokenizer_and_repaired_safetensors_retry(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    source, marker = hostile_checkpoint(tmp_path)
    with pytest.raises((OSError, ValueError)):
        GPTQModel.load(
            str(tmp_path),
            device="cpu",
            backend=BACKEND.TORCH,
            dtype=torch.float32,
            trust_remote_code=trust,
            code_revision="nonexistent",
            subfolder="not-present",
        )
    selected = load_model_config(str(tmp_path), trust_remote_code=trust, code_revision="nonexistent")
    assert type(selected) is NanbeigeConfig
    kwargs = {
        "device": "cpu",
        "backend": BACKEND.TORCH,
        "dtype": torch.float32,
        "trust_remote_code": trust,
        "code_revision": "nonexistent",
    }
    wrapped = GPTQModel.load(str(tmp_path), **kwargs)
    assert type(wrapped) is NanbeigeQModel
    assert type(wrapped.model) is NanbeigeForCausalLM
    assert wrapped.tokenizer.encode("t7") == [7]
    with torch.inference_mode():
        torch.testing.assert_close(
            wrapped.model(IDS, use_cache=False).logits, source(IDS, use_cache=False).logits, atol=0, rtol=0
        )
    payload = next(tmp_path.glob("*.safetensors"))
    saved = payload.read_bytes()
    payload.write_bytes(b"not safetensors")
    with pytest.raises(Exception, match="[Hh]eader|[Ss]afetensor|[Dd]eserializ"):
        GPTQModel.load(str(tmp_path), **kwargs)
    payload.write_bytes(saved)
    repaired = GPTQModel.load(str(tmp_path), **kwargs)
    with torch.inference_mode():
        torch.testing.assert_close(
            repaired.model(IDS, use_cache=False).logits, source(IDS, use_cache=False).logits, atol=0, rtol=0
        )
    assert not marker.exists()


def test_cache_and_input_failures_are_transactional_and_retryable():
    m = model(emb_neighbor_num=3, emb_split_num=2, ngram_vocab_size_ratio=1, emb_tp_num=1)
    with torch.inference_mode():
        out = m(IDS[:, :3], use_cache=True)
    cache = out.past_key_values
    before = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    ngram = cache.ngram_context.clone()
    with pytest.raises(ValueError, match="cache_position"):
        m(IDS[:, 3:4], past_key_values=cache, cache_position=torch.tensor([99]), use_cache=True)
    with pytest.raises(ValueError, match="mask"):
        m(IDS[:, 3:4], past_key_values=cache, attention_mask=torch.ones(2, 99), use_cache=True)
    with pytest.raises(ValueError, match="hidden-state.*history"):
        m(inputs_embeds=m.model.embed_tokens(IDS[:, 3:4]), past_key_values=cache, use_cache=True)
    for i, pair in enumerate(before):
        torch.testing.assert_close(cache.layers[i].keys, pair[0], atol=0, rtol=0)
        torch.testing.assert_close(cache.layers[i].values, pair[1], atol=0, rtol=0)
    torch.testing.assert_close(cache.ngram_context, ngram, atol=0, rtol=0)
    with torch.inference_mode():
        retry = m(IDS[:, 3:4], past_key_values=cache, use_cache=True)
        expected = m(IDS[:, :4], use_cache=False)
    torch.testing.assert_close(retry.logits, expected.logits[:, -1:], atol=5e-7, rtol=1e-5)
    with pytest.raises(ValueError, match="Dynamic"):
        m(IDS, past_key_values=StaticCache(config=m.config, max_cache_len=32), use_cache=True)
    with pytest.raises(ValueError, match="Dynamic"):
        m(IDS, past_key_values=(), use_cache=True)
    with pytest.raises(ValueError, match="must be enabled"):
        m(IDS[:, :1], past_key_values=cache, use_cache=False)
    with pytest.raises(ValueError, match="no key/value history"):
        m(IDS[:, :1], past_key_values=model()(IDS[:, :3], use_cache=True).past_key_values, use_cache=True)
    assert m(IDS, use_cache=False).logits.shape == (2, 6, 64)


@pytest.mark.parametrize(
    "options",
    [
        {"num_loops": 0},
        {"num_key_value_heads": 3},
        {"head_dim": 3},
        {"enable_mhc": True},
        {"ngram_fused_mode": "mystery"},
        {"emb_neighbor_num": 3, "emb_split_num": 0, "ngram_vocab_size_ratio": 1},
        {"enable_double_loop_split": True, "num_hidden_layers": 3, "loop_middle_layers": 2},
        {"mhc_diff_for_loop": True, "enable_double_loop_split": True},
        {"loop_share_kv": True},
        {"enable_hyper_connection": True, "num_residual_streams": [2]},
    ],
)
def test_invalid_layout_metadata_fails_explicitly(options):
    with pytest.raises((ValueError, TypeError)):
        config(**options)


def test_causality_logits_slice_loss_attention_and_generate_left_padding():
    m = model(attn_implementation="eager")
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    ids = IDS.masked_fill(mask == 0, 0)
    with torch.inference_mode():
        full = m(ids, attention_mask=mask, use_cache=False, output_attentions=True, output_hidden_states=True)
        changed = ids.clone()
        changed[:, -2:] = 23
        altered = m(changed, attention_mask=mask, use_cache=False)
        valid = mask[:, :-2].bool()
        torch.testing.assert_close(full.logits[:, :-2][valid], altered.logits[:, :-2][valid], atol=0, rtol=0)
        assert len(full.attentions) == 2
        assert len(full.hidden_states) == 3
        assert all(weights.shape == (2, 4, 6, 6) for weights in full.attentions)
        torch.testing.assert_close(
            m(ids, attention_mask=mask, use_cache=False, logits_to_keep=2).logits, full.logits[:, -2:], atol=0, rtol=0
        )
        torch.testing.assert_close(
            m(ids, attention_mask=mask, use_cache=False, logits_to_keep=torch.tensor([0, 3])).logits,
            full.logits[:, [0, 3]],
            atol=0,
            rtol=0,
        )
        labels = ids.clone()
        labels[:, 1] = -100
        loss = m(ids, attention_mask=mask, use_cache=False, labels=labels).loss
        expected = F.cross_entropy(full.logits[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(loss, expected, atol=0, rtol=0)
        batch = m.generate(ids, attention_mask=mask, max_new_tokens=2, eos_token_id=None)
        for i in range(2):
            single = m.generate(ids[i : i + 1, mask[i].bool()], max_new_tokens=2, eos_token_id=None)
            torch.testing.assert_close(batch[i, -2:], single[0, -2:])


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("variant", ["gqa", "mha_bias", "mqa", "split_shared", "ngram_fusion", "hc"])
def test_public_awq_all_physical_owners_capture_pack_two_safetensors_cycles(tmp_path, offload, variant):
    options = {
        "gqa": {"head_dim": 16},
        "mha_bias": {"num_key_value_heads": 4, "attention_bias": True, "mlp_bias": True},
        "mqa": {"num_key_value_heads": 1, "enable_depth_attention": True},
        "split_shared": {
            "num_hidden_layers": 4,
            "enable_double_loop_split": True,
            "loop_middle_layers": 2,
            "loop_share_kv": True,
        },
        "ngram_fusion": {
            "emb_neighbor_num": 3,
            "emb_split_num": 2,
            "ngram_vocab_size_ratio": 1,
            "emb_tp_num": 1,
            "ngram_insert_all_layers": True,
            "ngram_layer_downproject_size": 16,
        },
        "hc": {"enable_hyper_connection": True, "enable_mhc": True, "num_residual_streams": 2},
    }[variant]
    path = tmp_path / "source"
    source = model(**options).half()
    source.save_pretrained(path)
    tokenizer(path)
    original = {key: value.clone() for key, value in source.state_dict().items()}
    qconfig = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    wrapped = GPTQModel.load(str(path), quantize_config=qconfig, backend=BACKEND.TORCH, dtype=torch.float16)
    assert type(wrapped) is NanbeigeQModel
    assert wrapped.quantize_config.fallback is None
    captured = {i: [] for i in range(source.config.num_hidden_layers)}
    handles = []
    for i, layer in enumerate(wrapped.model.model.layers):

        def capture(module, args, index=i):
            if args[0].shape == (1, 6, 32):
                captured[index].append(args[0].detach().clone())

        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(capture))
    trace = {}
    with torch.inference_mode():
        reference_logits(source, IDS[:1], trace=trace)
    calibration = [
        {"input_ids": IDS[i : i + 1], "attention_mask": torch.ones(1, 6, dtype=torch.long)} for i in range(2)
    ]
    try:
        wrapped.quantize(
            CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
            backend=BACKEND.TORCH,
        )
    finally:
        for handle in handles:
            handle.remove()
    for index, executions in trace.items():
        assert len(captured[index]) >= len(executions), index
        for got, reference in zip(captured[index], executions):
            torch.testing.assert_close(got, reference, atol=0.012, rtol=0.025)
    expected = {
        f"model.layers.{i}.{suffix}"
        for i in range(source.config.num_hidden_layers)
        for suffix in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }
    observed = {name for name, module in wrapped.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert observed == expected
    assert wrapped._looped_per_layer is None
    # AWQ scale search folds its scales into the two decoder RMSNorm gains;
    # every other non-quantized owner must be untouched.
    folded = {
        f"model.layers.{i}.{norm}.weight"
        for i in range(source.config.num_hidden_layers)
        for norm in ("input_layernorm", "post_attention_layernorm")
    }
    for key, value in original.items():
        if key.rsplit(".", 1)[0] not in expected and key not in folded:
            torch.testing.assert_close(wrapped.model.state_dict()[key], value, atol=0, rtol=0)
    # Scale search is active: at least one norm absorbed a non-unit scale.
    assert any(not torch.equal(wrapped.model.state_dict()[key], original[key]) for key in folded)
    with torch.inference_mode():
        logits = wrapped.model(IDS, use_cache=False).logits
        assert torch.isfinite(logits).all() and logits.std() > 0.04
        full_precision = source(IDS, use_cache=False).logits
        assert (logits - full_precision).abs().max() < 0.11
        generated = wrapped.generate(IDS[:1, :3], max_new_tokens=2, eos_token_id=None)
    state = {key: value.clone() for key, value in wrapped.model.state_dict().items()}
    for cycle in range(2):
        destination = tmp_path / f"awq-{cycle}"
        wrapped.save(str(destination))
        assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
        wrapped = GPTQModel.load(str(destination), backend=BACKEND.TORCH, dtype=torch.float16, device="cpu")
        assert {
            name for name, module in wrapped.model.named_modules() if isinstance(module, AwqTorchLinear)
        } == expected
        assert state.keys() == wrapped.model.state_dict().keys()
        for key, value in state.items():
            torch.testing.assert_close(wrapped.model.state_dict()[key], value, atol=0, rtol=0)
        frequency = 1.0 / (
            source.config.rope_theta ** (torch.arange(0, source.config.head_dim, 2).float() / source.config.head_dim)
        )
        for layer in wrapped.model.model.layers:
            assert layer.self_attn.rotary_emb.inv_freq.dtype == torch.float32
            torch.testing.assert_close(layer.self_attn.rotary_emb.inv_freq, frequency, atol=0, rtol=0)
        with torch.inference_mode():
            torch.testing.assert_close(wrapped.model(IDS, use_cache=False).logits, logits, atol=0, rtol=0)
            torch.testing.assert_close(wrapped.generate(IDS[:1, :3], max_new_tokens=2, eos_token_id=None), generated)


@pytest.mark.parametrize("mode", ["average", "concat"])
@pytest.mark.parametrize("prime", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("fusion", ["input", "layers", "both"])
def test_independent_ngram_hash_eos_embedding_and_layer_fusion(mode, prime, compressed, fusion):
    m = model(
        emb_neighbor_num=3,
        emb_split_num=2,
        ngram_vocab_size_ratio=1,
        emb_tp_num=8,
        ngram_fused_mode=mode,
        ngram_mod_force_prime=prime,
        ngram_compressed_tokenizer=compressed,
        skip_ngram_for_input=fusion == "layers",
        ngram_insert_all_layers=fusion != "input",
        ngram_layer_downproject_size=16,
    ).double()
    if compressed:
        m.model.lookup_table.copy_((torch.arange(64) * 11 + 3) % 64)
    ids = torch.tensor([[1, 7, 2, 4, 11, 2, 8], [1, 5, 6, 7, 2, 3, 4]])
    with torch.inference_mode():
        embeddings, features = reference_ngram(m, ids)
        actual, actual_features = m.model.ngram_embeddings(
            ids, lookup_table=m.model.lookup_table, return_ngram_embeddings=True
        )
        torch.testing.assert_close(actual, embeddings, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual_features, features, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(m(ids, use_cache=False).logits, reference_logits(m, ids), atol=5e-7, rtol=5e-6)
        first = m(ids[:, :4], use_cache=True)
        last = m(ids[:, 4:], past_key_values=first.past_key_values, use_cache=True)
        torch.testing.assert_close(last.logits, m(ids, use_cache=False).logits[:, 4:], atol=5e-7, rtol=5e-6)


def test_low_precision_rope_buffers_keep_exact_fp32_frequency_and_meta_materialization():
    m = model(rope_theta=70000000, head_dim=16)
    freq = 1.0 / (70000000 ** (torch.arange(0, 16, 2).float() / 16))
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        m = m.to(dtype)
        for layer in m.model.layers:
            assert layer.self_attn.rotary_emb.inv_freq.dtype == torch.float32
            torch.testing.assert_close(layer.self_attn.rotary_emb.inv_freq, freq, atol=0, rtol=0)
    with torch.device("meta"):
        empty = NanbeigeForCausalLM(config(head_dim=16, rope_theta=70000000))
    empty.to_empty(device="cpu")
    for layer in empty.model.layers:
        torch.testing.assert_close(layer.self_attn.rotary_emb.inv_freq, freq, atol=0, rtol=0)


@pytest.mark.parametrize("expanded", [False, True])
def test_source_physical_depth_metadata_preserves_checkpoint_owners_after_legacy_expansion(tmp_path, expanded):
    original = model(num_hidden_layers=2, num_loops=3)
    metadata = original.config.to_dict()
    metadata.update(num_physical_layers=2, num_hidden_layers=6 if expanded else 2)
    normalized = NanbeigeForCausalLM(NanbeigeConfig(**metadata)).eval()
    assert normalized.config.num_hidden_layers == 2
    assert len(normalized.model.layers) == 2
    assert normalized.state_dict().keys() == original.state_dict().keys()
    normalized.load_state_dict(original.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(normalized(IDS).logits, original(IDS).logits, atol=0, rtol=0)
        cached = normalized(IDS[:, :3], use_cache=True)
        assert len(cached.past_key_values.layers) == 6
    normalized.save_pretrained(tmp_path)
    reloaded = NanbeigeForCausalLM.from_pretrained(tmp_path)
    assert reloaded.config.num_physical_layers == 2
    assert len(reloaded.model.layers) == 2
    with pytest.raises(ValueError, match="num_physical_layers"):
        config(num_physical_layers=3, num_hidden_layers=5)


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_released_non_llama_query_width_contract_has_independent_logits_and_cache(attention, kv_heads):
    m = model(
        attn_implementation=attention, head_dim=16, num_key_value_heads=kv_heads, attention_bias=True, mlp_bias=True
    ).double()
    assert m.model.layers[0].self_attn.q_proj.weight.shape == (64, 32)
    assert m.model.layers[0].self_attn.o_proj.weight.shape == (32, 64)
    with torch.inference_mode():
        expected = reference_logits(m, IDS)
        torch.testing.assert_close(m(IDS, use_cache=False).logits, expected, atol=4e-7, rtol=4e-6)
        prefix = m(IDS[:, :2], use_cache=True)
        actual = m(IDS[:, 2:], use_cache=True, past_key_values=prefix.past_key_values).logits
        torch.testing.assert_close(actual, expected[:, 2:], atol=4e-7, rtol=4e-6)


def test_eager_external_boolean_and_additive_4d_masks_share_independent_causal_contract():
    m = model(attn_implementation="eager").double()
    padding = torch.tensor([[1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    keep = torch.ones(6, 6, dtype=torch.bool).tril()[None, None] & padding[:, None, None, :].bool()
    additive = torch.zeros(keep.shape, dtype=torch.float64).masked_fill(~keep, torch.finfo(torch.float64).min)
    with torch.inference_mode():
        expected = reference_logits(m, IDS, padding)
        torch.testing.assert_close(m(IDS, attention_mask=keep, use_cache=False).logits, expected, atol=4e-7, rtol=4e-6)
        torch.testing.assert_close(
            m(IDS, attention_mask=additive, use_cache=False).logits, expected, atol=4e-7, rtol=4e-6
        )


@pytest.mark.parametrize("trust", [False, True])
def test_public_local_port_loads_selected_subfolder_not_hostile_root(tmp_path, monkeypatch, trust):
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    source, marker = hostile_checkpoint(tmp_path / "owned")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "auto_map": {"AutoConfig": "evil.ShouldNeverRun", "AutoModelForCausalLM": "evil.ShouldNeverRun"},
            }
        )
    )
    (tmp_path / "evil.py").write_text('raise RuntimeError("root checkpoint Python executed")')
    wrapped = GPTQModel.load(
        str(tmp_path),
        subfolder="owned",
        device="cpu",
        backend=BACKEND.TORCH,
        dtype=torch.float32,
        trust_remote_code=trust,
        code_revision="untrusted",
    )
    assert type(wrapped.model) is NanbeigeForCausalLM
    assert not marker.exists()
    with torch.inference_mode():
        torch.testing.assert_close(
            wrapped.model(IDS, use_cache=False).logits, source(IDS, use_cache=False).logits, atol=0, rtol=0
        )
