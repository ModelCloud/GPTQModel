# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def test_gptq_static_groups_follow_group_aware_column_permutation():
    weight = torch.tensor(
        [[1.0, 5.0, -0.1, 0.1], [-2.0, 4.0, -0.2, 0.2]]
    )
    layer = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)

    config = QuantizeConfig(
        bits=4,
        group_size=2,
        sym=False,
        static_groups=True,
        desc_act=False,
        fallback=None,
    )
    task = GPTQ(layer, config)
    task.quantizer.configure(perchannel=True)
    task.H = torch.diag(torch.tensor([1.0, 2.0, 10.0, 20.0]))
    task.nsamples = 4

    actual, *_ = task.quantize(blocksize=4)
    expected = torch.empty_like(weight)
    for start in (0, 2):
        group = weight[:, start : start + 2]
        low = torch.minimum(
            group.min(dim=1, keepdim=True).values,
            torch.zeros(2, 1),
        )
        high = torch.maximum(
            group.max(dim=1, keepdim=True).values,
            torch.zeros(2, 1),
        )
        scale = (high - low) / 15
        zero = torch.round(-low / scale)
        codes = torch.clamp(torch.round(group / scale) + zero, 0, 15)
        expected[:, start : start + 2] = scale * (codes - zero)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("mock_quantization", [False, True])
def test_static_group_aware_pack_roundtrip_matches_independent_affine_oracle(
    monkeypatch, mock_quantization
):
    # The second group has the larger Hessian diagonal, so GAR moves it first;
    # the increasing first group is also reversed within its group.
    original = torch.cat((2 * torch.arange(16), torch.arange(16) - 15)).float()
    weight = original.repeat(32, 1)
    diagonal = torch.cat((torch.arange(1, 17), torch.arange(64, 48, -1))).float()
    layer = torch.nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)

    config = QuantizeConfig(
        bits=4,
        group_size=16,
        sym=False,
        static_groups=True,
        act_group_aware=True,
        mock_quantization=mock_quantization,
        fallback=None,
    )
    task = GPTQ(layer, config)
    task.quantizer.configure(perchannel=True)
    task.H = torch.diag(diagonal)
    task.nsamples = 32
    task.hessian_inverse = lambda h: (torch.eye(h.shape[0]), config.damp_percent)

    observed_group_orders = []
    original_compute_global_perm = gptq_mod.compute_global_perm

    def record_global_perm(*args, **kwargs):
        permutation = original_compute_global_perm(*args, **kwargs)
        observed_group_orders.append(permutation.clone())
        return permutation

    monkeypatch.setattr(gptq_mod, "compute_global_perm", record_global_perm)
    actual_weight, scales, zeros, g_idx, *_ = task.quantize(blocksize=32)
    assert len(observed_group_orders) == 1
    assert torch.equal(observed_group_orders[0], torch.tensor([1, 0]))

    expected_weight = torch.empty_like(weight)
    expected_codes = torch.empty_like(weight, dtype=torch.int32)
    expected_scales = torch.empty((32, 2))
    expected_zeros = torch.empty((32, 2))
    for group_index, start in enumerate((0, 16)):
        source = weight[:, start : start + 16]
        minimum = torch.minimum(source.amin(dim=1, keepdim=True), torch.zeros(32, 1))
        maximum = torch.maximum(source.amax(dim=1, keepdim=True), torch.zeros(32, 1))
        scale = (maximum - minimum) / 15
        zero = torch.round(-minimum / scale)
        codes = torch.clamp(torch.round(source / scale) + zero, 0, 15)
        expected_weight[:, start : start + 16] = scale * (codes - zero)
        expected_codes[:, start : start + 16] = codes.int()
        expected_scales[:, group_index] = scale[:, 0]
        expected_zeros[:, group_index] = zero[:, 0]
    expected_g_idx = torch.arange(32, dtype=torch.int32) // 16

    torch.testing.assert_close(actual_weight, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(zeros, expected_zeros, rtol=0, atol=0)
    torch.testing.assert_close(g_idx, expected_g_idx, rtol=0, atol=0)

    with torch.no_grad():
        layer.weight.copy_(actual_weight)
    monkeypatch.setenv("GPTQMODEL_DISABLE_PACK_EXT", "1")
    packed = TorchLinear(
        bits=4,
        group_size=16,
        sym=False,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
    )
    packed.pack_block(layer, scales, zeros, g_idx, workers=1)
    packed.post_init()

    shifts = (4 * torch.arange(8, dtype=torch.int64)).view(1, 8, 1)
    expected_qweight = (
        (expected_codes.T.long().reshape(4, 8, 32) << shifts).sum(dim=1) & 0xFFFFFFFF
    ).int()
    expected_qzeros = (
        (expected_zeros.T.long().reshape(2, 4, 8) << shifts.transpose(1, 2)).sum(dim=2)
        & 0xFFFFFFFF
    ).int()
    assert torch.equal(packed.qweight, expected_qweight)
    assert torch.equal(packed.qzeros, expected_qzeros)
    torch.testing.assert_close(
        packed.dequantize_weight().T.float(), expected_weight, rtol=0, atol=0
    )
