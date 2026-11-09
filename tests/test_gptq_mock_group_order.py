# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def test_gptq_mock_mode_uses_each_columns_own_group():
    weight = torch.tensor(
        [[1.0, 1.1, -3.0, -2.8], [0.5, 0.7, 4.0, 4.2]]
    )
    layer = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    config = QuantizeConfig(
        bits=4,
        group_size=2,
        sym=False,
        static_groups=False,
        act_group_aware=False,
        mock_quantization=True,
        fallback=None,
    )
    task = GPTQ(layer, config)
    task.quantizer.configure(perchannel=True)
    task.H = torch.eye(4)
    task.nsamples = 1

    actual, *_ = task.quantize(blocksize=4)
    expected = torch.empty_like(weight)
    for start in (0, 2):
        group = weight[:, start : start + 2]
        low = group.min(dim=1, keepdim=True).values
        high = group.max(dim=1, keepdim=True).values
        low = torch.minimum(low, torch.zeros_like(low))
        high = torch.maximum(high, torch.zeros_like(high))
        scale = (high - low).clamp(min=1e-5) / 15
        zero = torch.round(-low / scale)
        codes = torch.clamp(torch.round(group / scale) + zero, 0, 15)
        expected[:, start : start + 2] = scale * (codes - zero)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_gptq_mock_mode_handles_groups_crossing_requested_blocks():
    weight = torch.linspace(-3, 3, 256).reshape(1, 256)
    layer = torch.nn.Linear(256, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    config = QuantizeConfig(
        bits=4,
        group_size=48,
        sym=False,
        static_groups=False,
        act_group_aware=False,
        mock_quantization=True,
        fallback=None,
    )
    task = GPTQ(layer, config)
    task.quantizer.configure(perchannel=True)
    task.H = torch.eye(256)
    task.nsamples = 1

    actual, scales, zeros, g_idx, *_ = task.quantize(blocksize=128)

    expected = torch.empty_like(weight)
    for start in range(0, 256, 48):
        group = weight[:, start : start + 48]
        low = torch.minimum(group.amin(dim=1, keepdim=True), torch.zeros((1, 1)))
        high = torch.maximum(group.amax(dim=1, keepdim=True), torch.zeros((1, 1)))
        scale = (high - low).clamp(min=1e-5) / 15
        zero = torch.round(-low / scale)
        codes = torch.clamp(torch.round(group / scale) + zero, 0, 15)
        expected[:, start : start + 48] = scale * (codes - zero)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    assert scales.shape == zeros.shape == (1, 6)
    assert torch.equal(g_idx, torch.arange(256, dtype=torch.int32) // 48)


def test_gptq_mock_mode_static_groups_follow_group_aware_permutation():
    weight = torch.tensor([[1.0, 5.0, -0.1, 0.1], [-2.0, 4.0, -0.2, 0.2]])
    layer = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    config = QuantizeConfig(
        bits=4,
        group_size=2,
        sym=False,
        static_groups=True,
        act_group_aware=True,
        mock_quantization=True,
        fallback=None,
    )
    task = GPTQ(layer, config)
    task.quantizer.configure(perchannel=True)
    task.H = torch.diag(torch.tensor([1.0, 2.0, 10.0, 20.0]))
    task.nsamples = 1

    actual, scales, zeros, g_idx, *_ = task.quantize(blocksize=4)

    expected = torch.empty_like(weight)
    for start in (0, 2):
        group = weight[:, start : start + 2]
        low = torch.minimum(group.amin(dim=1, keepdim=True), torch.zeros((2, 1)))
        high = torch.maximum(group.amax(dim=1, keepdim=True), torch.zeros((2, 1)))
        scale = (high - low).clamp(min=1e-5) / 15
        zero = torch.round(-low / scale)
        codes = torch.clamp(torch.round(group / scale) + zero, 0, 15)
        expected[:, start : start + 2] = scale * (codes - zero)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    assert scales.shape == zeros.shape == (2, 2)
    assert torch.equal(g_idx, torch.tensor([0, 0, 1, 1], dtype=torch.int32))
