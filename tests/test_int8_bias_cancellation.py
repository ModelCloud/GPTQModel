# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import torch

from gptqmodel.nn_modules.qlinear.torch_int8 import Int8PackedModule


def test_torch_int8_bias_is_added_to_wide_product():
    # A single group of 512 channels whose scale is 1.0 and whose residual
    # codes are all +1.
    module = Int8PackedModule(
        torch.ones((1, 1, 512), dtype=torch.int8),
        torch.ones((1, 1)),
        group_lengths=(512,),
    )
    x = torch.full((1, 512), 128.0, dtype=torch.float16)
    bias = torch.tensor([-65504.0], dtype=torch.float16)

    output = module(x, bias=bias)

    assert output.dtype == torch.float16
    assert output[0, 0].item() == 32.0
