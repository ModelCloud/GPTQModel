# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Small-budget regressions for local GSQ candidate banks."""

import pytest
import torch

from gptqmodel.quantization import GSQConfig
from gptqmodel.quantization.gsq_fp8 import refine_fp8_weight
from gptqmodel.quantization.gsq_qqq import refine_qqq_codes
from gptqmodel.quantization.gsq_scalar import refine_affine_scalar


def test_affine_candidate_chunks_reach_later_rows_and_preserve_hard_score():
    rows, width = 5, 4
    weight = torch.zeros((rows, width))
    scales = torch.full((rows, 1), 0.25)
    zeros = torch.full_like(scales, 8)
    groups = torch.zeros(width, dtype=torch.int32)
    target = torch.tensor([[0.35, -0.15, 0.50, -0.20]]).repeat(rows, 1)
    cross = torch.eye(width) * 0.1
    config = GSQConfig(enabled=True, steps=4, candidates=4,
                       max_candidate_bytes=2 * width * 4 * 4)

    with torch.inference_mode():
        result = refine_affine_scalar(weight, scales, zeros, groups, target=target,
                                      bits=4, config=config, hessian=torch.eye(width),
                                      cross_moment=cross)

    codes = ((result.weight + zeros[:, groups] * scales[:, groups]) /
             scales[:, groups]).round().clamp(0, 15)
    decoded = scales[:, groups] * (codes - zeros[:, groups])
    error = decoded - target
    expected = (error.square().sum() - 2 * (error * (target @ cross)).sum()) / target.square().sum()
    assert result.weight.shape == weight.shape
    assert len(result.history) == config.steps + 1
    assert result.after <= result.before
    torch.testing.assert_close(torch.tensor(result.after), expected, rtol=1e-6, atol=1e-6)


def test_qqq_candidate_chunks_preserve_codes_and_hard_score():
    rows, width = 5, 4
    codes = torch.full((rows, width), 8, dtype=torch.int32)
    scales = torch.full((rows, 1), 0.25)
    target = torch.tensor([[0.35, -0.15, 0.50, -0.20]]).repeat(rows, 1)
    cross = torch.eye(width) * 0.1
    config = GSQConfig(enabled=True, steps=4, candidates=4,
                       max_candidate_bytes=2 * width * 4 * 4)

    with torch.inference_mode():
        selected, before, after, history = refine_qqq_codes(
            codes, scales, target=target, group_size=-1, hessian=torch.eye(width),
            cross_moment=cross, config=config)

    signed = torch.where(selected >= 8, selected - 16, selected).float()
    decoded = signed * scales
    error = decoded - target
    expected = (error.square().sum() - 2 * (error * (target @ cross)).sum()) / target.square().sum()
    assert selected.shape == codes.shape
    assert len(history) == config.steps + 1
    assert after <= before
    torch.testing.assert_close(torch.tensor(after), expected, rtol=1e-6, atol=1e-6)


def test_fp8_block_candidate_chunks_can_split_a_scale_block():
    rows, width = 6, 4
    weight = torch.zeros((rows, width)).to(torch.float8_e4m3fn)
    scale_inv = torch.full((3, 2), 2.0)
    target = torch.tensor([[0.35, -0.15, 0.50, -0.20]]).repeat(rows, 1)
    config = GSQConfig(enabled=True, steps=4, candidates=4,
                       max_candidate_bytes=3 * width * 4 * 4)

    with torch.inference_mode():
        result = refine_fp8_weight(weight, scale_inv, target=target, config=config,
                                   hessian=torch.eye(width), method='block', block_size=(2, 2))

    expanded_scales = scale_inv.repeat_interleave(2, 0).repeat_interleave(2, 1)
    decoded = result['weight'].float() / expanded_scales
    expected = (decoded - target).square().sum() / target.square().sum()
    assert result['weight'].shape == weight.shape
    assert len(result['history']) == config.steps + 1
    assert result['after'] <= result['before']
    torch.testing.assert_close(torch.tensor(result['after']), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('method', ['affine', 'qqq', 'fp8'])
def test_candidate_budget_still_rejects_one_row(method):
    config = GSQConfig(enabled=True, steps=1, candidates=4, max_candidate_bytes=63)
    target = torch.zeros((2, 4))
    with pytest.raises(ValueError, match='one output row'):
        if method == 'affine':
            refine_affine_scalar(target, torch.ones((2, 1)), torch.zeros((2, 1)),
                                 torch.zeros(4, dtype=torch.int32), target=target, bits=4,
                                 config=config)
        elif method == 'qqq':
            refine_qqq_codes(torch.zeros((2, 4), dtype=torch.int32), torch.ones((2, 1)),
                             target=target, group_size=-1, hessian=torch.eye(4),
                             cross_moment=torch.zeros((4, 4)), config=config)
        else:
            refine_fp8_weight(target.to(torch.float8_e4m3fn), torch.ones(2),
                              target=target, config=config)
