# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear


def _quant_tensors(out_features, in_features, group_size):
    groups = in_features // group_size
    scales = torch.rand((out_features, groups), dtype=torch.float16) + 0.5
    zeros = torch.randint(0, 15, (out_features, groups), dtype=torch.int32)
    return scales, zeros


def test_gemv_awq_pack_registers_bias_when_created_without_buffers():
    """`pack()` must be able to register the bias buffer.

    Regression: ``__init__`` used to set ``self.bias = None`` unconditionally,
    which made ``hasattr(module, "bias")`` true and turned the later
    ``register_buffer("bias", ...)`` inside ``pack()`` into a
    ``KeyError: "attribute 'bias' already exists"``.  That aborted every
    ``FORMAT.GEMV`` quantization of a model whose linear layers carry a bias
    (e.g. Qwen2 q/k/v projections).
    """

    in_features, out_features, group_size = 256, 64, 128
    module = AwqGEMVLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=False,
    )
    linear = nn.Linear(in_features, out_features, bias=True)
    scales, zeros = _quant_tensors(out_features, in_features, group_size)

    module.pack(linear, scales, zeros)

    assert "bias" in module._buffers
    assert module.bias is not None
    assert module.bias.shape == (out_features,)
    assert torch.equal(module.bias, linear.bias.detach().half())


def test_gemv_awq_pack_sets_bias_none_when_layer_has_no_bias():
    in_features, out_features, group_size = 256, 64, 128
    module = AwqGEMVLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=False,
    )
    linear = nn.Linear(in_features, out_features, bias=False)
    scales, zeros = _quant_tensors(out_features, in_features, group_size)

    module.pack(linear, scales, zeros)

    assert module.bias is None
