# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The QQQ INT8 activation grid, shared by inference and calibration.

Every QQQ activation row is quantized on a symmetric INT8 grid: one scale per
row, ``amax(row) / 127``, nearest codes clamped to ``[-128, 127]``. The portable
kernel widens the row to FP32 before taking the maximum and dividing, and keeps
the scale in FP32, so the deployed grid does not depend on the activation's
storage dtype. The Triton and Metal kernels evaluate the same expression from
the stored operands; their tests pin them to this function's codes and scales.

A calibration statistic describes the deployed operand only if it uses that
same arithmetic. Keeping one implementation here means the kernel and the
calibration moments cannot drift: ``QQQLinear.dynamic_quant`` and the QQQ GSQ
calibration both call :func:`qqq_activation_quantize`.
"""

import torch


#: The symmetric INT8 grid bound. QQQ divides by ``127`` (not ``128``), so
#: ``-128`` is reachable through rounding, not through the scale definition.
QQQ_ACTIVATION_MAX = 127

#: Activation dtypes the QQQ inference kernels accept. Other floating widths
#: evaluate the same expression, but are outside the supported contract.
QQQ_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def qqq_activation_quantize(x: torch.Tensor):
    """Quantize the last axis of ``x`` on the deployed QQQ INT8 grid.

    Returns ``(codes, scale)``: INT8 contiguous codes and an FP32 scale of shape
    ``(*x.shape[:-1], 1)``. A row whose maximum is zero receives the runtime's
    unit scale, so it decodes to exactly zero and never divides by zero.
    """
    if not x.is_floating_point():
        raise ValueError("QQQ activation quantization requires a floating tensor")
    if x.shape[-1] == 0:
        raise ValueError("QQQ activation quantization requires a nonempty last axis")
    # Widen before the reduction and the division: the deployed scale is FP32,
    # and computing it in the storage dtype loses both range and precision.
    wide = x.float()
    scale = wide.abs().amax(dim=-1, keepdim=True).div(float(QQQ_ACTIVATION_MAX))
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = (wide / scale).round().clamp(-128, 127).to(torch.int8)
    return codes.contiguous(), scale
