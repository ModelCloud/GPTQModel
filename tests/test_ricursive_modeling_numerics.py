# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Independent full-forward oracle and cache/precision boundary coverage."""

import math
import os

import pytest
import torch
import torch.nn.functional as F
from ricursive_test_utils import compatibility_module, make_model
from transformers.cache_utils import DynamicCache, StaticCache


def norm(x, module):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + module.eps)).to(
        x.dtype
    ) * module.weight


@torch.inference_mode()
def oracle(model, ids, mask):
    x = F.embedding(ids, model.model.embed_tokens.weight)
    if model.config.embed_scale is not None:
        x = x * model.config.embed_scale
    seed = x
    batch, length = ids.shape
    pos = torch.arange(length)
    for layer in model.model.layers:
        h = norm(x, layer.input_layernorm)
        att = layer.self_attn
        q = norm(
            F.linear(h, att.q_proj.weight).reshape(batch, length, att.num_heads, att.head_dim), att.q_norm
        ).transpose(1, 2)
        k = norm(
            F.linear(h, att.k_proj.weight).reshape(batch, length, att.num_kv_heads, att.head_dim), att.k_norm
        ).transpose(1, 2)
        value_input = h
        if model.config.use_valres:
            blend = att.value_res_lambda[0].sigmoid()
            value_input = seed + blend * (h - seed)
        v = (
            F.linear(value_input, att.v_proj.weight)
            .reshape(batch, length, att.num_kv_heads, att.head_dim)
            .transpose(1, 2)
        )
        if att.layer_type == "sliding_attention":
            inv = model.config.rope_theta ** (-torch.arange(0, att.head_dim, 2).float() / att.head_dim)
            angles = pos[:, None] * inv[None, :]
            cos = torch.cat([angles.cos(), angles.cos()], dim=-1)[None, None]
            sin = torch.cat([angles.sin(), angles.sin()], dim=-1)[None, None]

            def rotate(z):
                a, b = z.chunk(2, dim=-1)
                return z * cos + torch.cat([-b, a], dim=-1) * sin

            q, k = rotate(q), rotate(k)
        k = torch.repeat_interleave(k, att.num_kv_groups, dim=1)
        v = torch.repeat_interleave(v, att.num_kv_groups, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(att.head_dim)
        allowed = pos[:, None] >= pos[None, :]
        if att.layer_type == "sliding_attention" and model.config.sliding_window is not None:
            allowed &= (pos[:, None] - pos[None, :]) < model.config.sliding_window
        allowed = allowed[None, None] & mask[:, None, None, :].bool()
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1).nan_to_num()
        values = (weights @ v).transpose(1, 2).reshape(batch, length, -1)
        if model.config.use_gated_attention:
            values *= F.linear(h, att.g_proj.weight).sigmoid()
        x = x + norm(F.linear(values, att.o_proj.weight), layer.post_attention_layernorm)
        h = norm(x, layer.pre_feedforward_layernorm)
        gate = F.linear(h, layer.mlp.gate_proj.weight)
        up = F.linear(h, layer.mlp.up_proj.weight)
        clamp = model.config.ffn_clamp_threshold
        if clamp:
            gate = gate.clamp(max=clamp)
            up = up.clamp(min=-clamp, max=clamp)
        output = F.linear(F.silu(gate) * up, layer.mlp.down_proj.weight)
        x += norm(output, layer.post_feedforward_layernorm)
    return F.linear(norm(x, model.model.norm), model.lm_head.weight)


@pytest.mark.parametrize("window", [1, 4, 32, None])
@pytest.mark.parametrize("query_heads", [4, 8])
@pytest.mark.parametrize("valres,gated,scale,clamp", [(True, True, 5.0, 0.05), (False, False, None, None)])
@pytest.mark.parametrize("padding", [False, True])
@torch.inference_mode()
def test_independent_oracle(window, query_heads, valres, gated, scale, clamp, padding):
    model = make_model(
        sliding_window=window,
        use_valres=valres,
        use_gated_attention=gated,
        embed_scale=scale,
        ffn_clamp_threshold=clamp,
        num_attention_heads=query_heads,
    )
    ids = torch.arange(3, 24).unsqueeze(0).repeat(2, 1)
    mask = torch.ones_like(ids)
    if padding:
        mask[0, :3] = 0
        mask[1, -3:] = 0
    actual = model(ids, attention_mask=mask, use_cache=False).logits
    expected = oracle(model, ids, mask)
    torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("cache_kind", ["full", "hybrid", "static"])
@pytest.mark.parametrize("window", [1, 4, 8])
@pytest.mark.parametrize("mask_kind", ["none", "left", "capacity"])
@pytest.mark.parametrize("chunks", [(1, 1, 1, 8, 1, 5), (7, 2, 6, 2), (3, 4, 4, 6)])
@torch.inference_mode()
def test_eviction_and_blocks(cache_kind, window, mask_kind, chunks):
    model = make_model(sliding_window=window)
    ids = torch.arange(3, 20).unsqueeze(0).repeat(2, 1)
    mask = None if mask_kind == "none" else torch.ones(2, 32 if mask_kind == "capacity" else 17, dtype=torch.long)
    if mask is not None:
        mask[0, :2] = 0
        if mask_kind == "capacity":
            mask[:, 17:] = 0
    full_mask = None if mask is None else mask[:, :17]
    expected = model(ids, attention_mask=full_mask, use_cache=False).logits
    cache = {
        "full": lambda: DynamicCache(),
        "hybrid": lambda: DynamicCache(config=model.config),
        "static": lambda: StaticCache(config=model.config, max_cache_len=32),
    }[cache_kind]()
    outputs = []
    start = 0
    for length in chunks:
        end = start + length
        block_mask = None if mask is None else mask if mask_kind == "capacity" else mask[:, :end]
        out = model(
            ids[:, start:end],
            past_key_values=cache,
            attention_mask=block_mask,
            cache_position=torch.arange(start, end),
            use_cache=True,
        )
        outputs.append(out.logits)
        start = end
    actual = torch.cat(outputs, dim=1)
    valid = torch.ones_like(ids, dtype=torch.bool) if full_mask is None else full_mask.bool()
    torch.testing.assert_close(actual[valid], expected[valid], atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_chunked_mlp(dtype):
    model = make_model(ffn_clamp_threshold=0.05).to(dtype)
    mlp = model.model.layers[0].mlp
    x = torch.randn(2, 13, 32, dtype=dtype)
    original = os.environ.get("RIPPLE_MLP_CHUNK")
    try:
        os.environ["RIPPLE_MLP_CHUNK"] = "0"
        expected = mlp(x)
        os.environ["RIPPLE_MLP_CHUNK"] = "3"
        actual = mlp(x)
    finally:
        if original is None:
            os.environ.pop("RIPPLE_MLP_CHUNK", None)
        else:
            os.environ["RIPPLE_MLP_CHUNK"] = original
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cache_implementation", ["dynamic", "static"])
@pytest.mark.parametrize("chunk_size", [None, 4])
@pytest.mark.parametrize("beams", [1, 3])
@torch.inference_mode()
def test_generation_modes(cache_implementation, chunk_size, beams):
    model = make_model()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9]])
    options = {"max_new_tokens": 5, "do_sample": False, "num_beams": beams}
    expected = model.generate(ids, use_cache=False, **options)
    actual = model.generate(ids, cache_implementation=cache_implementation, prefill_chunk_size=chunk_size, **options)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("runtime", ["native", "gptqmodel"])
@torch.inference_mode()
def test_default_outputs_and_loss(runtime):
    model = make_model()
    if runtime == "gptqmodel":
        apply_ricursive_generate_patch = compatibility_module().apply_ricursive_generate_patch
        apply_ricursive_generate_patch(model)
    ids = torch.tensor([[3, 4, 5, 6]])
    base = model(ids, labels=ids, use_cache=False)
    out = model(ids, labels=ids, use_cache=False, logits_to_keep=1, output_hidden_states=True, return_dict=False)
    expected_loss = F.cross_entropy(base.logits[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1))
    torch.testing.assert_close(out[0], expected_loss)
    assert out[1].shape == (1, 1, 64)
    assert len(out[2]) == 3
