# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #5654: CPU FP8 decode must not take a reciprocal.

``TorchFP8Linear._dequantize_cpu`` used to convert a stored inverse scale into
a multiplier with ``torch.reciprocal``.  For a finite scale below ``2**-128``
that reciprocal is not representable in FP32 even though every decoded
quotient is, so the decoded weight became ``inf``/``NaN`` and a finite forward
returned ``NaN``.  The CPU path now divides by the stored inverse scale.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear


#: Every ``(code, inverse_scale)`` pair decodes to exactly ``2**122``.  The
#: first two inverse scales have no finite FP32 reciprocal; the third is the
#: control just inside the reciprocal range (``1 / 2**-127 == 2**127``).
ENCODINGS = (
    (2.0**-7, 2.0**-129),
    (2.0**-6, 2.0**-128),
    (2.0**-5, 2.0**-127),
)

DECODED = 2.0**122


def _module(*, semantics: str = "inverse") -> TorchFP8Linear:
    return TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=2,
        out_features=1,
        format="float8_e4m3fn",
        weight_scale_method="row",
        weight_scale_semantics=semantics,
    ).eval()


def _load(module: TorchFP8Linear, code: float, scale: float) -> None:
    module.load_state_dict(
        {
            "weight": torch.tensor([[code, 0.0]]).to(torch.float8_e4m3fn),
            "weight_scale_inv": torch.tensor([scale], dtype=torch.float32),
        }
    )


@pytest.mark.parametrize(("code", "scale"), ENCODINGS)
def test_cpu_inverse_scale_decode_stays_finite(code, scale):
    module = _module()
    _load(module, code, scale)

    decoded = module.dequantize_weight(dtype=torch.float32)

    assert decoded.shape == (2, 1)
    assert torch.isfinite(decoded).all()
    torch.testing.assert_close(
        decoded,
        torch.tensor([[DECODED], [0.0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )

    x = torch.tensor([[2.0**-122, 0.0]])
    out = module(x)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)


def test_fp32_reciprocal_boundary_that_overflowed():
    """``2**-128`` is finite in FP32, but its reciprocal is not."""

    boundary = torch.tensor(2.0**-128, dtype=torch.float32)
    assert torch.isfinite(boundary)
    assert not torch.isfinite(torch.reciprocal(boundary))
    # The control one binade up still has a finite reciprocal.
    assert torch.isfinite(torch.reciprocal(torch.tensor(2.0**-127, dtype=torch.float32)))


def test_multiplier_scale_semantics_are_unchanged():
    module = _module(semantics="multiplier")
    _load(module, 2.0**-7, 2.0**7)

    decoded = module.dequantize_weight(dtype=torch.float32)

    torch.testing.assert_close(
        decoded,
        torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
