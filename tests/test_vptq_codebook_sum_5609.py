# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""VPTQ must round the codebook sum once, not back to the FP16 codebook dtype (#5609).

Both selected centroids are finite FP16 values (32768), so their sum (65536)
is finite in FP32 and representable in BF16 but not in FP16. ``dequantize``
used to narrow the FP32 reduction to the codebook storage dtype before the
forward widened it to the activation dtype: the weight became FP16 inf and the
BF16 forward then returned NaN on the zero entries of the activation. The
reconstruction is now requested in the activation dtype and rounded once.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear._numerics import codebook_rounding_dtype
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear


def _build(centroid: float, *, enable_norm: bool = False) -> VPTQLinear:
    module = VPTQLinear(
        bits=4,
        in_features=32,
        out_features=8,
        bias=False,
        vector_lens=(-1, 4),
        num_centroids=(-1, 16),
        num_res_centroids=(-1, 2),
        dtype=torch.float16,
        group_size=-1,
        enable_norm=enable_norm,
    )
    with torch.no_grad():
        module.indices.zero_()
        module.centroids.weight.fill_(centroid)
        module.res_centroids.weight.fill_(centroid)
    return module


def test_vptq_bf16_forward_survives_a_finite_fp16_codebook_sum():
    """The exact #5609 repro: 32768 + 32768 must decode to 65536 in BF16."""

    module = _build(32768.0)
    with torch.no_grad():
        x = torch.zeros(1, 32, dtype=torch.bfloat16)
        x[:, 0] = 0.25

        weight = module.dequantize(dtype=torch.bfloat16)
        assert weight.dtype is torch.bfloat16
        assert torch.isfinite(weight).all()
        assert torch.equal(weight, torch.full_like(weight, 65536.0))

        out = module(x)
        assert out.dtype is torch.bfloat16
        assert torch.isfinite(out).all()
        assert out[0, 0].item() == 16384.0
        assert torch.equal(out, torch.full_like(out, 16384.0))


@pytest.mark.parametrize(
    ("centroid", "expected"),
    [(32752.0, 65504.0), (32768.0, 65536.0)],
)
def test_vptq_codebook_sum_boundary_is_rounded_once(centroid, expected):
    """65504 is the largest FP16 sum; the next step already needs the wide dtype."""

    module = _build(centroid)
    fp16_weight = module.dequantize()
    assert fp16_weight.dtype is torch.float16
    if expected <= torch.finfo(torch.float16).max:
        assert fp16_weight[0, 0].item() == expected
    else:
        # The codebook storage dtype cannot hold the sum, so a BF16 forward
        # must not consume this narrowed tensor.
        assert not torch.isfinite(fp16_weight).all()

    weight = module.dequantize(dtype=torch.bfloat16)
    assert torch.isfinite(weight).all()
    assert torch.equal(weight, torch.full_like(weight, expected))


def test_vptq_bf16_normalization_product_keeps_the_wide_range():
    """The norm epilogue multiplies the sum before the single rounding."""

    module = _build(32768.0, enable_norm=True)
    with torch.no_grad():
        module.weight_scale.fill_(2.0)
        module.weight_bias.zero_()
        x = torch.zeros(1, 32, dtype=torch.bfloat16)
        x[:, 0] = 0.25

        weight = module.dequantize(dtype=torch.bfloat16)
        assert torch.isfinite(weight).all()
        assert torch.equal(weight, torch.full_like(weight, 131072.0))

        out = module(x)
        assert torch.isfinite(out).all()
        assert out[0, 0].item() == 32768.0
        assert torch.equal(out, torch.full_like(out, 32768.0))


@pytest.mark.parametrize(
    ("activation", "codebook", "expected"),
    [
        (torch.float16, torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        # The activation dtype holds every codebook value: keep the historical
        # codebook-dtype rounding so FP32/FP64 numerics stay bitwise identical.
        (torch.float32, torch.float16, torch.float16),
        (torch.float64, torch.float16, torch.float16),
        (torch.float32, torch.bfloat16, torch.bfloat16),
        # BF16 has FP16's range beat but a coarser mantissa: rounding to FP16
        # first can overflow, so round once to the activation dtype.
        (torch.bfloat16, torch.float16, torch.bfloat16),
        # FP16 cannot hold BF16's range; the codebook dtype stays the target.
        (torch.float16, torch.bfloat16, torch.bfloat16),
    ],
)
def test_codebook_rounding_dtype_policy(activation, codebook, expected):
    assert codebook_rounding_dtype(activation, codebook) is expected


def test_vptq_fp32_forward_keeps_the_codebook_dtype_rounding():
    """The fix must not silently change the established FP32 numerics."""

    module = _build(1000.0)
    with torch.no_grad():
        x = torch.zeros(1, 32, dtype=torch.float32)
        x[:, 0] = 0.25

        expected = torch.nn.functional.linear(x, module.dequantize().float())
        out = module(x)

    assert torch.equal(out, expected)
    assert out[0, 0].item() == 500.0
