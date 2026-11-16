# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MacaqLinear's Torch fallback must follow the activation dtype (#4217).

``compute_dtype`` is captured in ``__init__`` while ``.to(dtype)`` moves the
bias/pre-scale/Hadamard buffers.  The fallback therefore has to derive its
compute dtype from the input at forward time; otherwise a biased layer raises
a dtype mismatch and an unbiased one silently computes in the stale dtype.
"""

import itertools

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear
from gptqmodel.quantization import macaq as macaq_core


def _make_layer(*, precision, rotation, pre_mode, bias, dtype):
    layer = MacaqLinear(
        bits=4,
        group_size=32,
        in_features=128,
        out_features=64,
        bias=bias,
        activation_precision=precision,
        rotation=rotation,
        pre_mode=pre_mode,
        dtype=dtype,
    )
    layer.W_packed.random_(0, 255)
    layer.W_scales.fill_(122)
    layer.post_init()
    layer.eval()
    return layer


def _expected(layer, x, dtype):
    """Recompute the Torch fallback in ``dtype`` from the module's live buffers."""

    weight = macaq_core.dequantize_mxfp4(layer.W_packed, layer.W_scales, 128, dtype=dtype)
    if layer.activation_precision == "w4a4":
        xb = macaq_core.macaq_quantize_activations(
            x, layer.pre_scale, layer.BH, layer.block_size
        ).to(dtype)
    else:
        xb = x.to(dtype)
        if layer.pre_scale is not None:
            xb = xb * layer.pre_scale.to(dtype)
        if layer.BH is not None:
            xb = macaq_core.rotate_blocks(xb, layer.BH)
    bias = None if layer.bias is None else layer.bias.to(dtype)
    return F.linear(xb, weight, bias)


@pytest.mark.parametrize("target_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    ("precision", "rotation", "pre_mode", "bias"),
    list(itertools.product(("w4a4", "w4a16"), ("hadamard", "none"), ("balance", "none"), (True, False))),
    ids=lambda value: str(value),
)
def test_macaq_torch_forward_after_to_follows_input_dtype(
    target_dtype, precision, rotation, pre_mode, bias
):
    """A ``.to(dtype)`` after ``__init__`` must not leave the fallback stale."""

    layer = _make_layer(
        precision=precision, rotation=rotation, pre_mode=pre_mode, bias=bias, dtype=torch.bfloat16
    ).to(target_dtype)
    x = torch.randn(3, 128, dtype=target_dtype)

    out = layer(x)

    assert out.dtype == target_dtype
    torch.testing.assert_close(out, _expected(layer, x, target_dtype), rtol=0, atol=0)


def test_macaq_torch_forward_follows_fp16_input_built_as_bf16():
    """Without ``.to()`` the input dtype alone decides the fallback dtype."""

    layer = _make_layer(
        precision="w4a16", rotation="none", pre_mode="none", bias=True, dtype=torch.bfloat16
    )
    x = torch.randn(2, 128, dtype=torch.float16)

    out = layer(x)

    assert out.dtype == torch.float16
    torch.testing.assert_close(out, _expected(layer, x, torch.float16), rtol=0, atol=0)
