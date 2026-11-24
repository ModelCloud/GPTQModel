# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A wide activation decodes in its own dtype, like the fused kernels.

The dense fallback used to narrow an FP32/FP64 activation to the stored FP16
scale dtype before the matmul, which adds a rounding the fused kernels never
perform: they form each weight in FP32 and round once to the activation dtype.
Decoding in the activation dtype also keeps the fallback bitwise aligned with
the shared int4pack decode (``decode_dtype_for_scales``).
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT


pytestmark = [pytest.mark.cpu]


def _layer(dtype=torch.float16, scale=1.0, weight=None):
    source = torch.nn.Linear(512, 32, bias=False, dtype=dtype)
    with torch.no_grad():
        source.weight.fill_(scale if weight is None else weight)
    module = TorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=512,
        out_features=32,
        bias=False,
        format=FORMAT.GPTQ_V2,
    )
    scales = torch.full((32, 4), scale, dtype=dtype)
    zeros = torch.zeros((32, 4), dtype=dtype)
    module.pack_block(
        source,
        scales,
        zeros,
        torch.arange(512, dtype=torch.int32) // 128,
        workers=1,
    )
    module.eval()
    return module


def test_a_wide_activation_decodes_in_its_own_dtype():
    module = _layer()

    # Control: the half-dtype rule from #5740 is unchanged.
    assert module._forward_decode_dtype(torch.float16) is torch.float16
    # Wide activations decode in their own dtype instead of the stored FP16.
    assert module._forward_decode_dtype(torch.float32) is torch.float32
    assert module._forward_decode_dtype(torch.float64) is torch.float64


def test_the_wide_fallback_matches_a_float64_oracle():
    # ``0.1 * 3`` is a code*scale product that FP16 cannot hold exactly, so a
    # FP16 decode and a FP32 decode differ in the last bits.
    module = _layer(scale=0.1, weight=0.3)
    weight = module.dequantize_weight(dtype=torch.float32)
    # The Torch decoder returns the reconstructed weight as ``[in, out]``.
    assert weight.shape == (512, 32)
    assert not torch.equal(
        weight, module.dequantize_weight(dtype=torch.float16).to(torch.float32)
    ), "the fixture must decode differently in FP16 and FP32"
    x = torch.randn(3, 512, dtype=torch.float32)

    with torch.no_grad():
        actual = module(x)

    expected = (x.double() @ weight.double()).float()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
