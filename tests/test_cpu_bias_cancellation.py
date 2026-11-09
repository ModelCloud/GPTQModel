# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


def test_cpu_fp16_bias_cancels_before_output_rounding():
    module = GGUFTorchLinear(
        bits="q8_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=512,
        out_features=1,
        bias=True,
        register_buffers=True,
    )
    module.qweight.zero_()
    blocks = module.qweight.view(1, -1, 34)
    blocks[:, :, :2] = torch.tensor(
        [2.0], dtype=torch.float16
    ).view(torch.uint8)
    blocks[:, :, 2:] = 64
    module.bias.fill_(-65504.0)
    x = torch.ones((1, 512), dtype=torch.float16)

    output = module._forward_dequant_matmul(x, bias=module.bias)

    assert output.dtype == torch.float16
    assert output[0, 0].item() == 32.0
