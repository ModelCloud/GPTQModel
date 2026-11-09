# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


def test_gguf_packing_preserves_fp32_bias_range():
    linear = torch.nn.Linear(32, 32, bias=True, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(100000.0)
    module = GGUFTorchLinear(
        bits="q8_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=True,
        register_buffers=False,
    )
    module.pack_original(linear, scales=None, zeros=None)

    output = module(torch.zeros((2, 32), dtype=torch.float32))

    assert module.bias.dtype == torch.float32
    assert torch.equal(output, torch.full((2, 32), 100000.0))
