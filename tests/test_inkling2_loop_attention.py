# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from weakref import ref

import pytest
import torch
from torch.nn import functional as F

from gptqmodel.models.inkling2_loop import modeling
from gptqmodel.models.inkling2_loop.configuration import Inkling2LoopVisionConfig
from gptqmodel.models.inkling2_loop.vision import Inkling2LoopVisionAttention, Inkling2LoopVisionRotary


def _inputs(dtype, repeats):
    generator = torch.Generator().manual_seed(711)
    query = torch.randn(2, 6, 7, 8, generator=generator, dtype=dtype)
    key = torch.randn(2, 6 // repeats, 11, 8, generator=generator, dtype=dtype)
    value = torch.randn(2, 6 // repeats, 11, 8, generator=generator, dtype=dtype)
    bias = torch.randn(2, 6, 7, 11, generator=generator, dtype=dtype)
    sinks = torch.randn(2, 6, 7, 1, generator=generator, dtype=dtype)
    return query, key, value, bias, sinks


def _reference_attention(query, key, value, mask, scaling, position_bias, sinks, num_key_value_groups, dropout):
    keys = key.repeat_interleave(num_key_value_groups, dim=1)
    values = value.repeat_interleave(num_key_value_groups, dim=1)
    scores = (query @ keys.transpose(2, 3)) * scaling + position_bias + mask
    if sinks is not None:
        scores = torch.cat((scores, sinks), dim=-1)
    probabilities = F.softmax(scores.float(), dim=-1)
    if sinks is not None:
        probabilities = probabilities[..., :-1]
    probabilities = F.dropout(probabilities.to(query.dtype), p=dropout, training=dropout > 0.0)
    return (probabilities @ values).transpose(1, 2).contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("repeats", [1, 3])
@pytest.mark.parametrize("has_sinks", [False, True])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_eager_attention_matches_reference_forward_and_gradients(dtype, repeats, has_sinks, dropout):
    query, key, value, bias, sinks = _inputs(dtype, repeats)
    mask = torch.zeros((2, 1, 7, 11), dtype=dtype)
    mask[..., 9:] = float("-inf")
    differentiable = (query, key, value, bias, sinks) if has_sinks else (query, key, value, bias)

    def run(attend):
        inputs = tuple(tensor.clone().requires_grad_() for tensor in differentiable)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(991)
            output = attend(*inputs[:3], mask, 8**-0.5, inputs[3], inputs[4] if has_sinks else None, repeats, dropout)
        gradients = torch.autograd.grad(output.float().square().sum(), inputs)
        return (output, *gradients)

    bits = torch.int32 if dtype == torch.float32 else torch.int16
    for expected, actual in zip(run(_reference_attention), run(modeling._attend), strict=True):
        assert torch.equal(expected.contiguous().view(bits), actual.contiguous().view(bits))


def test_eager_attention_releases_scores_and_repeated_keys_before_softmax(monkeypatch):
    query, key, value, bias, sinks = _inputs(torch.bfloat16, 3)
    cat, softmax, repeat = torch.cat, F.softmax, modeling._repeat_key_value
    pre_sink_scores = []
    repeated = []

    def record_cat(tensors, *, dim):
        pre_sink_scores.append(ref(tensors[0]))
        return cat(tensors, dim=dim)

    def record_repeat(tensor, repeats):
        result = repeat(tensor, repeats)
        repeated.append(ref(result))
        return result

    def inspect_softmax(scores, *, dim):
        assert len(pre_sink_scores) == 1
        assert pre_sink_scores[0]() is None, "BF16 scores retained during FP32 softmax"
        assert len(repeated) == 1 and repeated[0]() is None
        return softmax(scores, dim=dim)

    monkeypatch.setattr(torch, "cat", record_cat)
    monkeypatch.setattr(F, "softmax", inspect_softmax)
    monkeypatch.setattr(modeling, "_repeat_key_value", record_repeat)
    with torch.inference_mode():
        output = modeling._attend(query, key, value, None, 8**-0.5, bias, sinks, num_key_value_groups=3)
    assert output.shape == (2, 7, 6, 8)
    assert len(repeated) == 2


def _vision_reference(attention, hidden, boundaries, cos, sin):
    images = []
    for start, end in zip(boundaries, boundaries[1:]):
        project = (attention.q_proj, attention.k_proj, attention.v_proj)
        q, k, v = (
            F.linear(hidden[start:end], layer.weight, layer.bias)
            .unflatten(-1, (attention.num_heads, attention.head_dim))
            .unbind(dim=1)
            for layer in project
        )
        c, s = cos[start:end, ::2], sin[start:end, ::2]

        def rotate(head):
            real, imag = head[:, ::2].float(), head[:, 1::2].float()
            return torch.stack((real * c - imag * s, real * s + imag * c), dim=-1).flatten(1).to(head.dtype)

        heads = []
        for query, key, value in zip(q, k, v, strict=True):
            scores = (rotate(query) @ rotate(key).T) / attention.head_dim**0.5
            probabilities = scores.float().softmax(dim=-1).to(query.dtype)
            heads.append(probabilities @ value)
        images.append(torch.cat(heads, dim=1))
    return F.linear(torch.cat(images), attention.out_proj.weight, attention.out_proj.bias)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "boundaries",
    [(0, 2, 5), (0, 3, 16)],
    ids=["rotary-across-tokens", "output-head-order"],
)
def test_vision_attention_matches_per_image_and_head_reference(dtype, boundaries):
    config = Inkling2LoopVisionConfig(hidden_size=128, num_attention_heads=16)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(711)
        attention = Inkling2LoopVisionAttention(config).to(dtype=dtype).eval()
        hidden = torch.randn(boundaries[-1], config.hidden_size, dtype=dtype)
    positions = torch.arange(boundaries[-1])
    if len(positions) == config.num_attention_heads:
        # Isolate output head ordering with identity rotations and equal axis lengths.
        positions = torch.zeros_like(positions)
    position_ids = torch.stack((positions // 3, positions % 3), dim=-1)
    cos, sin = Inkling2LoopVisionRotary(config)(position_ids)
    with torch.inference_mode():
        expected = _vision_reference(attention, hidden, boundaries, cos, sin)
        actual = attention(hidden, torch.tensor(boundaries, dtype=torch.int32), (cos, sin))
    assert actual.shape == hidden.shape
    assert actual.dtype == dtype
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (2e-3, 2e-3)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
