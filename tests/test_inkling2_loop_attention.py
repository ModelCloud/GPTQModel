# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from weakref import ref

import pytest
import torch
from torch.nn import functional as F

from gptqmodel.models.ports.inkling2_loop import modeling


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
