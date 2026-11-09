# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import torch

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear


def test_fp8_packing_preserves_fp32_bias_range():
    linear = torch.nn.Linear(4, 1, bias=True, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(70000.0)
    module = TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=4,
        out_features=1,
        bias=True,
        register_buffers=False,
        format="float8_e4m3fn",
    )
    module.pack_calibrated(
        linear,
        torch.zeros((1, 4), dtype=torch.float8_e4m3fn),
        torch.ones((1,), dtype=torch.float32),
    )

    output = module(torch.zeros((2, 4), dtype=torch.float32))

    assert module.bias.dtype == torch.float32
    assert torch.equal(output, torch.full((2, 1), 70000.0))
