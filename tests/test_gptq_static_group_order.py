# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import torch

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
