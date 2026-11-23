# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The QQQ INT8 activation grid, shared by inference and calibration.

Every QQQ activation row is quantized on a symmetric INT8 grid: one scale per
row, ``amax(row) / 127``, nearest codes clamped to ``[-128, 127]``. The portable
kernel widens the row to FP32 before taking the maximum and dividing, and keeps
the scale in FP32, so the deployed grid does not depend on the activation's
storage dtype. The Triton and Metal kernels evaluate the same expression from
the stored operands; their tests pin them to this function's codes and scales.

A positive row maximum whose FP32 quotient by ``127`` rounds to zero is a
scale-representation limit, not an all-zero row: the row is still finitely
encodable because the scale may be a subnormal. Such a quotient is floored at
the least positive FP32 subnormal instead of taking the all-zero convention,
which is reserved for a maximum of exactly zero.

A calibration statistic describes the deployed operand only if it uses that
same arithmetic. Keeping one implementation here means the kernel and the
calibration moments cannot drift: ``QQQLinear.dynamic_quant`` and the QQQ GSQ
calibration both call :func:`qqq_activation_quantize`.
"""

import torch

from .activation_floatx import FP32_MIN_SUBNORMAL


#: The symmetric INT8 grid bound. QQQ divides by ``127`` (not ``128``), so
#: ``-128`` is reachable through rounding, not through the scale definition.
QQQ_ACTIVATION_MAX = 127

#: Activation dtypes the QQQ inference kernels accept. Other floating widths
#: evaluate the same expression, but are outside the supported contract.
QQQ_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def qqq_activation_scale(row_max: torch.Tensor) -> torch.Tensor:
    """Return the deployed FP32 row scale for ``row_max`` values.

    ``row_max`` is ``amax(|row|)`` in FP32. The all-zero convention is selected
    from that maximum, not from the quotient: a subnormal maximum divides to
    zero above while the row is still finitely encodable, so such a quotient is
    floored at the least positive FP32 subnormal. This is the single definition
    the portable quantizer uses; the Triton kernel mirrors both rules, and the
    Metal kernel selects its zero row from the maximum but has no qualified
    subnormal floor (see ``docs/qqq-activation-grid.md``).
    """
    scale = row_max.div(float(QQQ_ACTIVATION_MAX))
    zero_row = row_max == 0
    scale = torch.where(zero_row, torch.ones_like(scale), scale)
    return torch.where(
        ~zero_row & (scale == 0),
        torch.full_like(scale, FP32_MIN_SUBNORMAL),
        scale,
    )


def qqq_activation_quantize(x: torch.Tensor):
    """Quantize the last axis of ``x`` on the deployed QQQ INT8 grid.

    Returns ``(codes, scale)``: INT8 contiguous codes and an FP32 scale of shape
    ``(*x.shape[:-1], 1)``. Only a row whose maximum is exactly zero receives
    the runtime's unit scale, so it decodes to exactly zero and never divides by
    zero. A finite nonzero row whose ``amax / 127`` quotient rounds to zero keeps
    a representable positive scale of ``2**-149``, so its codes are not erased.
    """
    if not x.is_floating_point():
        raise ValueError("QQQ activation quantization requires a floating tensor")
    if x.shape[-1] == 0:
        raise ValueError("QQQ activation quantization requires a nonempty last axis")
    # Widen before the reduction and the division: the deployed scale is FP32,
    # and computing it in the storage dtype loses both range and precision.
    wide = x.float()
    scale = qqq_activation_scale(wide.abs().amax(dim=-1, keepdim=True))
    codes = (wide / scale).round().clamp(-128, 127).to(torch.int8)
    return codes.contiguous(), scale
