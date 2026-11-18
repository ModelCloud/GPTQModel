# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest
import torch
from safetensors.torch import load_file
from transformers.cache_utils import DynamicCache, LinearAttentionLayer

from gptqmodel.models.definitions.bailing_moe import BailingMoeQModel
from gptqmodel.models.ports.bailing_moe.configuration_bailing import BailingConfig
from gptqmodel.models.ports.bailing_moe.modeling_bailing import (
    BailingCache,
    BailingForCausalLM,
    LinearAttention,
    RotaryEmbedding,
)
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code


def recurrent_states(cache):
    return {
        idx: layer.recurrent_states[0] for idx, layer in enumerate(cache.layers)
        if isinstance(layer, LinearAttentionLayer) and layer.has_previous_state[0]
    }


def tiny_config(layout="dense", **kwargs):
    values = {
        "model_type": "bailing_moe" if layout == "dense" else "bailing_hybrid",
        "architectures": ["BailingMoeV2ForCausalLM" if layout == "dense" else "BailingMoeV2_5ForCausalLM"],
        "vocab_size": 37, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "pad_token_id": 0, "eos_token_id": 2,
        "num_experts": 4, "n_group": 1, "topk_group": 1, "num_experts_per_tok": 2, "num_shared_experts": 1,
        "moe_intermediate_size": 12, "first_k_dense_replace": 1, "routed_scaling_factor": 2.5,
        "layer_group_size": 2 if layout == "hybrid" else 1, "group_norm_size": 2,
        "q_lora_rank": None if layout in ("dense", "mla_dense_q") else 8,
        "kv_lora_rank": 4, "qk_rope_head_dim": 4, "qk_nope_head_dim": 4, "v_head_dim": 6,
    }
    values.update(kwargs)
    if layout == "legacy_gqa":
        values.update(model_type="bailing_moe", architectures=["BailingMoeForCausalLM"], q_lora_rank=None, norm_head=True)
    elif layout == "mla_alias":
        values["model_type"] = "bailing_moe"
    return BailingConfig(**values)


@pytest.mark.parametrize("layout", ["dense", "mla", "mla_dense_q", "hybrid", "legacy_gqa", "mla_alias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_forward_cache_generation_and_safetensors(layout, dtype, tmp_path):
    torch.manual_seed(731)
    model = BailingForCausalLM(tiny_config(layout)).to(dtype).eval()
    if layout == "mla_alias":
        assert hasattr(model.model.layers[0].attention, "q_a_proj")
    if layout == "legacy_gqa":
        assert not hasattr(model.model.layers[1].mlp.gate, "expert_bias")
        assert not hasattr(model.model.layers[0].attention, "query_layernorm")
    ids = torch.tensor([[0, 5, 11, 7, 3], [4, 9, 6, 8, 5]])
    mask = (ids != 0).long()
    with torch.no_grad():
        full = model(ids, attention_mask=mask, use_cache=False).logits
        prefill = model(ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
        cache = prefill.past_key_values
        assert cache.get_seq_length() == 3
        continuation = model(ids[:, 3:], attention_mask=mask, past_key_values=cache)
        assert cache.get_seq_length() == ids.shape[1]
        tolerance = 2e-3 if dtype == torch.bfloat16 else 2e-6
        torch.testing.assert_close(continuation.logits, full[:, 3:], rtol=tolerance, atol=tolerance)
        assert torch.isfinite(full).all() and full.std() > 0.01
        changed = ids.clone()
        changed[:, -1] = 17
        valid = mask[:, :-1].bool()
        torch.testing.assert_close(
            model(changed, attention_mask=mask, use_cache=False).logits[:, :-1][valid], full[:, :-1][valid],
        )
        generated = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, eos_token_id=None)
        uncached = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=False, eos_token_id=None)
        assert torch.equal(generated, uncached)
        model.save_pretrained(tmp_path)
        assert not list(tmp_path.glob("*.bin"))
        payload = load_file(str(tmp_path / "model.safetensors"))
        assert payload.keys() == model.state_dict().keys()
        reloaded = BailingMoeQModel.loader.from_pretrained(tmp_path, trust_remote_code=False).eval()
        assert reloaded.config.model_type == model.config.model_type
        for key, value in model.state_dict().items():
            torch.testing.assert_close(reloaded.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(reloaded(ids, attention_mask=mask, use_cache=False).logits, full)
        reloaded.save_pretrained(tmp_path / "again")
        again = BailingMoeQModel.loader.from_pretrained(tmp_path / "again").eval()
        torch.testing.assert_close(again(ids, attention_mask=mask, use_cache=False).logits, full)


@pytest.mark.parametrize("layout", ["dense", "mla", "hybrid", "mla_dense_q", "legacy_gqa", "mla_alias"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_config_and_definition_ignore_malicious_auto_map(layout, trust, bypass, tmp_path, monkeypatch):
    config = tiny_config(layout)
    config.auto_map = {"AutoConfig": "evil.Config", "AutoModelForCausalLM": "evil.Model"}
    BailingForCausalLM(config).save_pretrained(tmp_path)
    marker = tmp_path / "executed"
    (tmp_path / "evil.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('executed')\n")
    if bypass:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")  # Existing security API, not a modeling switch.
    else:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    loaded = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(loaded) is BailingConfig
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    with torch.device("meta"):
        model = BailingMoeQModel.loader.from_config(loaded, trust_remote_code=trust)
    assert type(model) is BailingForCausalLM
    assert next(model.parameters()).is_meta
    restored = BailingMoeQModel.loader.from_pretrained(tmp_path, trust_remote_code=trust).eval()
    assert type(restored) is BailingForCausalLM
    assert torch.isfinite(restored(torch.tensor([[3, 7]]), use_cache=False).logits).all()
    assert not marker.exists()


def test_genuine_native_hybrid_is_not_replaced_by_the_port(tmp_path):
    from transformers import BailingHybridConfig, BailingHybridForCausalLM

    config = BailingHybridConfig(
        vocab_size=37, pad_token_id=0, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, qk_nope_head_dim=4,
        qk_rope_head_dim=4, v_head_dim=8, kv_lora_rank=4, num_local_experts=4,
        num_experts_per_tok=2, n_group=1, topk_group=1, moe_intermediate_size=12,
        layer_types=["full_attention", "full_attention"], mlp_layer_types=["sparse", "sparse"],
        num_mtp_layers=0, architectures=["BailingHybridForCausalLM"],
        auto_map={"AutoConfig": "evil.Config", "AutoModelForCausalLM": "evil.Model"},
    )
    config.save_pretrained(tmp_path)
    loaded = load_model_config(tmp_path, trust_remote_code=False)
    assert type(loaded) is BailingHybridConfig
    with torch.device("meta"):
        model = BailingMoeQModel.loader.from_config(loaded)
    assert type(model) is BailingHybridForCausalLM
    assert hasattr(model.model.layers[0], "self_attn")
    native = BailingMoeQModel.loader.from_config(loaded).eval()
    ids = torch.tensor([[3, 7, 9]])
    with torch.no_grad():
        expected = native(ids, use_cache=False).logits
        assert torch.isfinite(expected).all()
        native.save_pretrained(tmp_path)
        restored = BailingMoeQModel.loader.from_pretrained(tmp_path, trust_remote_code=True).eval()
        assert type(restored) is BailingHybridForCausalLM
        torch.testing.assert_close(restored(ids, use_cache=False).logits, expected)


@pytest.mark.parametrize("layout", ["dense", "hybrid", "legacy_gqa", "mla_alias"])
def test_failure_invalidates_partial_cache_then_reset_replays_exactly(layout):
    torch.manual_seed(23)
    model = BailingForCausalLM(tiny_config(layout)).eval()
    ids = torch.tensor([[3, 7, 5, 9]])
    with torch.no_grad():
        cache = model(ids[:, :2]).past_key_values
        handle = model.model.layers[1].register_forward_pre_hook(
            lambda *args: (_ for _ in ()).throw(RuntimeError("injected second-layer failure"))
        )
        with pytest.raises(RuntimeError, match="injected"):
            model(ids[:, 2:], past_key_values=cache)
        handle.remove()
        assert cache.incomplete
        if layout == "hybrid":
            assert recurrent_states(cache)
        else:
            assert cache.layers[0].get_seq_length() == 4
        with pytest.raises(ValueError, match="failed partial forward"):
            model(ids[:, 2:], past_key_values=cache)
        cache.reset()
        assert not cache.incomplete and cache.get_seq_length() == 0 and not recurrent_states(cache)
        replay = model(ids, past_key_values=cache).logits
        fresh = model(ids, use_cache=False).logits
        torch.testing.assert_close(replay, fresh, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["dense", "mla", "mla_dense_q", "hybrid", "legacy_gqa", "mla_alias"])
@pytest.mark.parametrize("boundary", ["lm_head", "loss", "mtp_head", "mtp_loss"])
def test_public_forward_failure_invalidates_advanced_cache_until_reset(layout, boundary, monkeypatch):
    from torch.nn import functional as functional


    torch.manual_seed(93)
    model = BailingForCausalLM(tiny_config(layout, num_nextn_predict_layers=1)).eval()
    ids = torch.tensor([[0, 3, 7, 5, 9, 11]])
    mask = (ids != 0).long()
    calls = 0
    fired = False
    original_linear, original_loss = functional.linear, model.loss_function

    def fail():
        nonlocal fired
        fired = True
        raise RuntimeError(f"injected public {boundary} failure")

    def linear(x, weight, bias=None):
        nonlocal calls
        if weight.shape == model.lm_head.weight.shape:
            calls += 1
            if calls == (2 if boundary == "mtp_head" else 1):
                fail()
        return original_linear(x, weight, bias)

    def loss(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == (2 if boundary == "mtp_loss" else 1):
            fail()
        return original_loss(*args, **kwargs)

    with torch.no_grad():
        cache = model(ids[:, :3], attention_mask=mask[:, :3]).past_key_values
        assert not cache.incomplete and cache.get_seq_length() == 3
        previous_recurrent = {idx: state.clone() for idx, state in recurrent_states(cache).items()}
        with monkeypatch.context() as patch:
            if boundary in ("lm_head", "mtp_head"):
                patch.setattr(functional, "linear", linear)
            elif boundary in ("loss", "mtp_loss"):
                patch.setattr(model, "loss_function", loss)
            with pytest.raises(RuntimeError, match=f"injected public {boundary}"):
                model(
                    ids[:, 3:], attention_mask=mask, labels=ids[:, 3:], past_key_values=cache,
                )
            assert fired
            assert cache.get_seq_length() == ids.shape[1]
            if previous_recurrent:
                assert any(not torch.equal(state, recurrent_states(cache)[idx]) for idx, state in previous_recurrent.items())
            else:
                assert cache.layers[0].get_seq_length() == ids.shape[1]
            assert cache.incomplete
            with pytest.raises(ValueError, match="failed partial forward"):
                model(ids[:, 3:], attention_mask=mask, past_key_values=cache)
        cache.reset()
        assert not cache.incomplete and cache.get_seq_length() == 0 and not recurrent_states(cache)
        fresh = model(ids, attention_mask=mask, labels=ids, use_cache=False)
        replay = model(ids, attention_mask=mask, labels=ids, past_key_values=cache)
        torch.testing.assert_close(replay.logits, fresh.logits, rtol=0, atol=0)
        torch.testing.assert_close(replay.loss, fresh.loss, rtol=0, atol=0)
        for actual, expected in zip(replay.mtp_logits, fresh.mtp_logits, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not cache.incomplete and cache.get_seq_length() == ids.shape[1]
        extended = torch.cat((ids, torch.tensor([[13]])), dim=-1)
        continued = model(extended[:, -1:], attention_mask=(extended != 0).long(), past_key_values=cache)
        reference = model(extended, attention_mask=(extended != 0).long(), use_cache=False)
        torch.testing.assert_close(continued.logits, reference.logits[:, -1:], rtol=1e-5, atol=1e-6)
        assert not cache.incomplete


def test_recurrent_cache_reorder_repeat_select_and_crop_rejection():
    model = BailingForCausalLM(tiny_config("hybrid")).eval()
    ids = torch.tensor([[3, 7, 9], [8, 4, 5]])
    with torch.no_grad():
        cache = model(ids[:, :2]).past_key_values
        original_state = recurrent_states(cache)[0].clone()
        cache.reorder_cache(torch.tensor([1, 0]))
        torch.testing.assert_close(recurrent_states(cache)[0], original_state.flip(0))
        out = model(ids.flip(0)[:, 2:], past_key_values=cache).logits
        torch.testing.assert_close(out, model(ids.flip(0), use_cache=False).logits[:, 2:], atol=2e-6, rtol=2e-6)
        cache.batch_repeat_interleave(2)
        assert recurrent_states(cache)[0].shape[0] == 4
        cache.batch_select_indices(torch.tensor([0, 3]))
        assert recurrent_states(cache)[0].shape[0] == 2
        with pytest.raises(NotImplementedError, match="cannot be cropped"):
            cache.crop(-1)
        assert cache.get_seq_length() == 3
        cache.reset()
        assert cache.get_seq_length() == 0 and not recurrent_states(cache)
    with pytest.raises(TypeError, match="BailingCache"):
        model(ids, past_key_values=DynamicCache())


def test_mtp_owned_weights_and_loss_survive_round_trip(tmp_path):
    model = BailingForCausalLM(tiny_config("hybrid", num_nextn_predict_layers=1)).eval()
    ids = torch.tensor([[3, 7, 9, 8]])
    with torch.no_grad():
        output = model(ids, labels=ids, use_cache=False)
        assert len(output.mtp_logits) == len(output.mtp_loss) == 1
        assert torch.isfinite(output.loss)
        assert model.model.layers[2].is_mtp
        model.save_pretrained(tmp_path)
        restored = BailingMoeQModel.loader.from_pretrained(tmp_path).eval()
        torch.testing.assert_close(restored(ids, labels=ids, use_cache=False).mtp_logits[0], output.mtp_logits[0])


def test_validation_fails_before_cache_mutation_and_retry_works():
    model = BailingForCausalLM(tiny_config("hybrid")).eval()
    ids = torch.tensor([[3, 5]])
    cache = BailingCache(model.config)
    with pytest.raises(ValueError, match="attention_mask"):
        model(ids, past_key_values=cache, attention_mask=torch.ones(1, 3))
    assert not cache.incomplete and not recurrent_states(cache) and cache.get_seq_length() == 0
    assert model(ids, past_key_values=cache).past_key_values.get_seq_length() == 2
    with pytest.raises(ValueError, match="group-limited"):
        tiny_config("dense", n_group=4)


def test_no_optional_cuda_import_is_needed_for_linear_forward():
    model = BailingForCausalLM(tiny_config("hybrid")).eval()
    assert isinstance(model.model.layers[0].attention, LinearAttention)
    assert model(torch.tensor([[3, 5]]), use_cache=False).logits.shape == (1, 2, 37)


@pytest.mark.parametrize("rope_type", ["linear", "dynamic", "yarn"])
def test_scaled_rope_survives_safetensors_reload(rope_type, tmp_path):
    config = tiny_config("hybrid", rope_scaling={"rope_type": rope_type, "factor": 2.0})
    torch.manual_seed(61)
    model = BailingForCausalLM(config).eval()
    ids = torch.tensor([[3, 5, 9, 12]])
    with torch.no_grad():
        reference = model(ids, use_cache=False).logits
        assert torch.isfinite(reference).all()
        model.save_pretrained(tmp_path)
        restored = BailingMoeQModel.loader.from_pretrained(tmp_path).eval()
        torch.testing.assert_close(restored(ids, use_cache=False).logits, reference)


def test_yarn_rope_uses_standardized_parameters_and_partial_dim_once():
    """`rope_scaling` is a TF5 alias of `rope_parameters`; YaRN must not reapply partial rotary."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.llama.configuration_llama import LlamaConfig

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        config = tiny_config("hybrid", rope_theta=1000.0, max_position_embeddings=64,
                             rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16})
    assert config.rope_parameters["rope_type"] == "yarn" and config.rope_theta == 1000.0
    for mla in (False, True):
        rotary = RotaryEmbedding(config, mla=mla)
        dim = config.qk_rope_head_dim if mla else int(config.head_dim * config.partial_rotary_factor)
        reference = LlamaConfig(head_dim=dim, hidden_size=16, num_attention_heads=2, max_position_embeddings=64,
                                rope_parameters={"rope_type": "yarn", "rope_theta": 1000.0, "factor": 4.0,
                                                 "original_max_position_embeddings": 16})
        inv_freq, scale = ROPE_INIT_FUNCTIONS["yarn"](reference)
        torch.testing.assert_close(rotary.inv_freq, inv_freq)
        assert rotary.attention_scaling == scale
    reloaded = BailingConfig.from_dict(config.to_dict())
    assert reloaded.rope_parameters == config.rope_parameters and reloaded.rope_theta == 1000.0


def _checkpoint_oracle(config, weights, ids, mask):
    """Independent full-prefix transcription of the pinned Antgroup equations.

    Uses checkpoint tensors only, never model modules or production forward
    helpers. Linear attention uses the quadratic decay matrix, not recurrence.
    """
    from torch.nn import functional as functional

    def linear(x, name):
        return functional.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name, groups=1, eps=None):
        shape, dtype = x.shape, x.dtype
        grouped = x.reshape(*shape[:-1], groups, shape[-1] // groups).float()
        grouped = grouped / torch.sqrt(grouped.square().mean(-1, keepdim=True) + (eps or config.rms_norm_eps))
        return grouped.to(dtype).reshape(shape) * weights[name + ".weight"]

    def rotary(x, dim, interleave=False):
        rot = x[..., :dim]
        if interleave:
            rot = torch.cat((rot[..., 0::2], rot[..., 1::2]), dim=-1)
        freq = config.rope_theta ** (-torch.arange(0, dim, 2).float() / dim)
        angle = torch.arange(ids.shape[1])[:, None] * freq[None, :]
        angle = torch.cat((angle, angle), dim=-1)[None, None].to(x.dtype)
        a, b = rot.chunk(2, dim=-1)
        return torch.cat((rot * angle.cos() + torch.cat((-b, a), -1) * angle.sin(), x[..., dim:]), -1)

    def mlp(x, prefix):
        return linear(functional.silu(linear(x, prefix + ".gate_proj")) * linear(x, prefix + ".up_proj"), prefix + ".down_proj")

    batch, length = ids.shape
    heads, dim = config.num_attention_heads, config.head_dim
    hidden = functional.embedding(ids, weights["model.word_embeddings.weight"])
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        x = norm(hidden, prefix + ".input_layernorm")
        attn = prefix + ".attention"
        if attn + ".query_key_value.weight" in weights:
            linear_layer = attn + ".g_proj.weight" in weights
            kvheads = heads if linear_layer else config.num_key_value_heads
            qkv = linear(x, attn + ".query_key_value").reshape(batch, length, heads + 2 * kvheads, dim)
            q, k, value = (t.transpose(1, 2) for t in qkv.split((heads, kvheads, kvheads), -2))
            if config.use_qk_norm:
                q, k = norm(q, attn + ".query_layernorm"), norm(k, attn + ".key_layernorm")
            q = rotary(q, int(dim * config.partial_rotary_factor))
            k = rotary(k, int(dim * config.partial_rotary_factor))
            k, value = (t.repeat_interleave(heads // kvheads, dim=1) for t in (k, value))
            scale = dim ** -0.5
        else:
            linear_layer = False
            q = linear(x, attn + ".q_proj") if config.q_lora_rank is None else linear(
                norm(linear(x, attn + ".q_a_proj"), attn + ".q_a_layernorm", eps=1e-6), attn + ".q_b_proj",
            )
            q = q.reshape(batch, length, heads, config.qk_head_dim).transpose(1, 2)
            q_plain, q_rot = q.split((config.qk_nope_head_dim, config.qk_rope_head_dim), -1)
            compressed, k_rot = linear(x, attn + ".kv_a_proj_with_mqa").split((config.kv_lora_rank, config.qk_rope_head_dim), -1)
            kv = linear(norm(compressed, attn + ".kv_a_layernorm", eps=1e-6), attn + ".kv_b_proj").reshape(
                batch, length, heads, config.qk_nope_head_dim + config.v_head_dim,
            ).transpose(1, 2)
            k_plain, value = kv.split((config.qk_nope_head_dim, config.v_head_dim), -1)
            q_rot = rotary(q_rot, config.qk_rope_head_dim, config.rope_interleave)
            k_rot = rotary(k_rot[:, None], config.qk_rope_head_dim, config.rope_interleave)
            q = torch.cat((q_plain, q_rot), -1)
            k = torch.cat((k_plain, k_rot.expand(batch, heads, length, -1)), -1)
            scale = config.qk_head_dim ** -0.5
        scores = (q @ k.transpose(-1, -2)) * scale
        causal = torch.ones(length, length, dtype=torch.bool).tril()
        if linear_layer:
            # Tiny oracle uses two heads: ALiBi slopes are 1/16 and 1/256.
            assert heads == 2
            slopes = -torch.tensor([1 / 16, 1 / 256]) * (1 - (layer - 1) / (config.num_hidden_layers - 1) + 1e-5)
            distance = torch.arange(length)[:, None] - torch.arange(length)[None, :]
            decay = (slopes[:, None, None] * distance[None]).exp() * causal
            attention = (scores * decay[None]) @ (value * mask[:, None, :, None])
            attention = attention.transpose(1, 2).reshape(batch, length, -1)
            attention = norm(attention, attn + ".g_norm", config.group_norm_size) * linear(x, attn + ".g_proj").sigmoid()
        else:
            scores = scores.masked_fill(~(causal[None, None] & mask[:, None, None].bool()), torch.finfo(scores.dtype).min)
            attention = (scores.float().softmax(-1).to(value.dtype) @ value).transpose(1, 2).reshape(batch, length, -1)
        hidden = hidden + linear(attention, attn + ".dense")
        x = norm(hidden, prefix + ".post_attention_layernorm")
        if layer < config.first_k_dense_replace or config.num_experts is None:
            out = mlp(x, prefix + ".mlp")
        else:
            legacy = prefix + ".mlp.gate.expert_bias" not in weights
            logits = linear(x if legacy else x.float(), prefix + ".mlp.gate")
            scores = logits.float().softmax(-1) if legacy else logits.sigmoid()
            selection = scores
            if not legacy:
                selection = selection + weights[prefix + ".mlp.gate.expert_bias"]
                groups = selection.reshape(batch, length, config.n_group, -1)
                chosen_groups = groups.topk(2, -1).values.sum(-1).topk(config.topk_group, -1).indices
                selected_mask = torch.zeros_like(groups[..., 0]).scatter_(-1, chosen_groups, 1).bool()
                selection = selection.masked_fill(~selected_mask[..., None].expand_as(groups).reshape_as(selection), -torch.inf)
            selected = selection.topk(config.num_experts_per_tok, -1).indices
            coeff = scores.gather(-1, selected)
            if config.num_experts_per_tok > 1 and (not legacy or config.norm_topk_prob):
                coeff = coeff / (coeff.sum(-1, keepdim=True) + 1e-20)
            if not legacy:
                coeff = coeff * config.routed_scaling_factor
            out = torch.zeros_like(x, dtype=coeff.dtype)
            for expert in range(config.num_experts):
                expert_weight = ((selected == expert) * coeff).sum(-1)
                out = out + mlp(x, prefix + f".mlp.experts.{expert}").to(coeff.dtype) * expert_weight[..., None]
            out = out.to(x.dtype)
            if config.num_shared_experts is not None:
                out = out + mlp(x, prefix + ".mlp.shared_experts")
        hidden = hidden + out
    hidden = norm(hidden, "model.norm")
    if config.norm_head:
        weight = weights["lm_head.weight"].float()
        weight = weight / torch.sqrt(weight.square().sum(0, keepdim=True)).clamp(min=1e-7)
        return functional.linear(hidden, weight.to(hidden.dtype)).float()
    return linear(hidden, "lm_head").float()


@pytest.mark.parametrize("layout", ["dense", "mla", "mla_dense_q", "hybrid", "legacy_gqa", "mla_alias"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("interleave", [False, True])
def test_complete_forward_matches_independent_checkpoint_equations(layout, bias, interleave):
    torch.manual_seed(44)
    config = tiny_config(layout, use_bias=bias, use_qkv_bias=bias, rope_interleave=interleave, n_group=2)
    model = BailingForCausalLM(config).eval()
    ids = torch.tensor([[3, 8, 5, 11, 13], [0, 4, 8, 9, 7]])
    mask = (ids != 0).long()
    with torch.no_grad():
        reference = _checkpoint_oracle(config, model.state_dict(), ids, mask)
        output = model(ids, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("normalize", [False, True])
def test_legacy_softmax_routing_respects_norm_topk_probability(normalize):
    torch.manual_seed(39)
    model = BailingForCausalLM(tiny_config("legacy_gqa", norm_topk_prob=normalize)).eval()
    ids = torch.tensor([[3, 5, 8, 9]])
    with torch.no_grad():
        output = model(ids, use_cache=False).logits
        reference = _checkpoint_oracle(model.config, model.state_dict(), ids, torch.ones_like(ids))
        torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-5)


def test_native_config_with_retained_checkpoint_metadata_uses_mla(tmp_path):
    from transformers.models.bailing_hybrid.configuration_bailing_hybrid import BailingHybridConfig

    values = tiny_config("mla").to_dict()
    values["architectures"] = ["BailingHybridForCausalLM"]
    native_config = BailingHybridConfig.from_dict(values)
    for key in ("q_lora_rank", "num_experts_per_tok", "n_group", "head_dim", "partial_rotary_factor"):
        setattr(native_config, key, values[key])
    model = BailingMoeQModel.loader.from_config(native_config).eval()
    assert type(model) is BailingForCausalLM
    assert hasattr(model.model.layers[0].attention, "q_a_proj")
    ids = torch.tensor([[3, 5, 8]])
    with torch.no_grad():
        expected = _checkpoint_oracle(model.config, model.state_dict(), ids, torch.ones_like(ids))
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, atol=1e-6, rtol=1e-5)
        model.save_pretrained(tmp_path)
        restored = BailingMoeQModel.loader.from_pretrained(tmp_path).eval()
        assert type(restored) is BailingForCausalLM
        torch.testing.assert_close(restored(ids, use_cache=False).logits, expected, atol=1e-6, rtol=1e-5)
