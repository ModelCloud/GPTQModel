# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5694: the INT8 FP32 epilogue must stay finite for extreme absmax ratios.

A single active input of ``2**120`` and a packed weight column of ``2**-120``
are both the row and the column maximum, so every nonzero code is exactly
``127`` and the integer dot product is exactly ``127**2 = 16129``: the exact
projection is ``1.0`` with no quantization error on either side. The pre-fix
FP32 epilogue evaluated ``mm_out * row_absmax * col_absmax / 127**2`` left to
right, so it multiplied by the ``2**120`` activation absmax first and overflowed
to ``+inf`` before the decreasing ``2**-120`` weight absmax could be applied.
``2**100`` stayed finite and agreed with FP64, which shows the ordering, not the
integer dot, is the defect.

The tests below use the FP64 equation of the packed tensors as the oracle, not
the pre-fix FP32 expression, and cover the ordinary end-to-end path plus the
chunked helper that repeats the same ordering.
"""

import sys

import pytest
import torch

from gptqmodel.nn_modules.qlinear.bitsandbytes import (
    BitsAndBytesLinear,
    _int8_chunked_fp32_epilogue,
    _int8_vectorwise_quant_fp32,
)


pytestmark = [pytest.mark.cpu]

_IN_FEATURES = 16
_OUT_FEATURES = 4


def _packed_layer(
    in_features: int = _IN_FEATURES,
    out_features: int = _OUT_FEATURES,
    *,
    bias: bool = False,
) -> tuple[BitsAndBytesLinear, torch.nn.Linear]:
    """Pack an FP32 ``nn.Linear`` into an INT8 ``BitsAndBytesLinear``."""

    linear = torch.nn.Linear(in_features, out_features, bias=bias)
    layer = BitsAndBytesLinear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        dtype=torch.float32,
    )
    layer.pack_original(linear, None, None)
    layer.post_init()
    return layer, linear


def test_extreme_absmax_ratio_projects_exactly_unit():
    """A 2**120 input times a 2**-120 weight is 1.0, not ``+inf`` (#5694)."""

    layer, linear = _packed_layer()
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[:, 0] = 2.0**-120
    layer.pack_original(linear, None, None)
    layer.post_init()

    # ``2**100`` overflowed no intermediate, so it is the control: the pre-fix
    # epilogue already answered it correctly and the fix must not disturb it.
    for exponent, expected in ((100, 2.0**-20), (120, 1.0)):
        x = torch.zeros(1, _IN_FEATURES, dtype=torch.float32)
        x[0, 0] = 2.0**exponent

        reference = (x.double() @ linear.weight.double().t())[0, 0].item()
        assert reference == expected, "the FP64 equation of the fixture changed"

        out = layer(x)

        assert torch.isfinite(out).all(), f"2**{exponent} produced a non-finite output"
        assert out[0, 0].item() == expected, (exponent, out[0, 0].item())


def test_extreme_absmax_ratio_survives_chunked_epilogue(monkeypatch):
    """The chunked helper repeats the ordering, so it needs the same fix.

    The two rows carry opposite extreme ratios, so a single global multiply
    order cannot keep both finite: each element has to rescale its own
    exponents. Cells whose exact product leaves the FP32 range must still
    saturate to ``+inf``/``0`` instead of wrapping.
    """

    bnb_module = sys.modules[BitsAndBytesLinear.__module__]
    mm_out = torch.full((2, 4), 127 * 127, dtype=torch.int32)
    x_stats = torch.tensor([2.0**120, 2.0**-120], dtype=torch.float32)
    weight_scb = torch.tensor(
        [2.0**-120, 2.0**-100, 2.0**100, 2.0**120],
        dtype=torch.float32,
    )

    # One row per chunk so the chunk boundary is exercised too.
    monkeypatch.setattr(bnb_module, "_INT8_EPILOGUE_CHUNK_ELEMENTS", 4)
    actual = _int8_chunked_fp32_epilogue(
        mm_out,
        x_stats,
        weight_scb,
        out_dtype=torch.float32,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0**20, float("inf"), float("inf")],
            [0.0, 0.0, 2.0**-20, 1.0],
        ],
        dtype=torch.float32,
    )
    assert actual[0, 0].item() == 1.0
    assert actual[1, 3].item() == 1.0
    assert torch.isfinite(actual[0, 1]) and actual[0, 1].item() == 2.0**20
    assert torch.isfinite(actual[1, 2]) and actual[1, 2].item() == 2.0**-20
    assert actual[0, 2].item() == float("inf")
    assert actual[1, 0].item() == 0.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ordinary_projection_keeps_pre_fix_accuracy():
    """A moderate absmax ratio must keep the pre-fix FP32 accuracy."""

    torch.manual_seed(5694)
    layer, linear = _packed_layer(32, 8, bias=True)
    with torch.no_grad():
        linear.weight.copy_(torch.randn_like(linear.weight) * 0.05)
        linear.bias.copy_(torch.randn_like(linear.bias))
    layer.pack_original(linear, None, None)
    layer.post_init()

    # Integer codes scaled by a power of two: every row maximum is exactly
    # ``127 * 2**-3``, so the row quantizer represents the fixture exactly and
    # the oracle needs no activation quantization error term.
    codes_in = torch.randint(-127, 128, (5, 32), dtype=torch.int32).to(torch.float32)
    for row in range(5):
        codes_in[row, 0] = 127.0 if row % 2 == 0 else -127.0
    x = codes_in * 2.0**-3

    codes, stats, _ = _int8_vectorwise_quant_fp32(x)
    assert torch.equal(
        codes.float() * (stats / 127.0).reshape(-1, 1),
        x,
    ), "the fixture must not carry activation quantization error"

    out = layer(x)

    w_int8 = layer.weight
    scb = layer.weight_scb
    # ``|mm| <= 127 * 127 * 32 < 2**24``, so the FP32 integer dot is exact.
    mm_out = codes.float() @ w_int8.float().t()
    pre_fix = (
        mm_out * stats.reshape(-1, 1) * scb.reshape(1, -1) / (127.0 * 127.0)
        + layer.bias
    )
    oracle = (
        x.double()
        @ (w_int8.double() * (scb.double() / 127.0).reshape(-1, 1)).t()
        + layer.bias.double()
    )

    torch.testing.assert_close(out.double(), oracle, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out.double(), pre_fix.double(), rtol=1e-6, atol=1e-6)
