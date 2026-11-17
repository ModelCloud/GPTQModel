# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#4563: Marlin must accept a column-strided prefill activation.

Mamba-family mixers feed ``out_proj``/``x_proj`` a transposed view with
``stride(-1) > 1``. Marlin's GEMM requires a unit contraction stride, so the
kernel boundary materializes the operand only for that case. The test drives
both GPTQ-Marlin and AWQ-Marlin and checks the strided result against the
contiguous call and an independent FP64 Torch oracle.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear
from gptqmodel.utils.marlin import (
    marlin_import_exception,
    marlin_runtime_available,
    marlin_runtime_error,
)


pytestmark = pytest.mark.cuda

SHAPES = (
    (256, 128),  # tile aligned: AUTO resolves to the Marlin class
    (384, 224),  # N is padded to the Marlin tile
)


def _build_marlin(in_features, out_features, method, dtype, backend):
    torch.manual_seed(4563)
    codes = torch.randint(0, 16, (in_features, out_features))
    zeros = torch.full((in_features // 128, out_features), 8, dtype=torch.int32)
    scales = torch.full(zeros.shape, 1 / 512, dtype=dtype)
    indices = torch.arange(in_features, dtype=torch.int32) // 128
    weight = ((codes - zeros[indices]) * scales[indices]).to(dtype)

    linear = torch.nn.Linear(in_features, out_features, bias=True).cpu().to(dtype)
    linear.weight.data.copy_(weight.t())

    kwargs = {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "in_features": in_features,
        "out_features": out_features,
        "bias": True,
        "register_buffers": True,
        "dtype": dtype,
    }

    if method == "gptq":
        source = TorchLinear(**kwargs, format=FORMAT.GPTQ_V2)
        source.pack_block(linear, scales.t(), zeros.t().float(), indices)
        module = MarlinLinear(**kwargs, backend=backend).cuda()
        for name in ("qweight", "qzeros", "scales", "bias", "g_idx"):
            src, dst = getattr(source, name, None), getattr(module, name, None)
            if (
                isinstance(src, torch.Tensor)
                and isinstance(dst, torch.Tensor)
                and src.shape == dst.shape
            ):
                dst.data.copy_(src.cuda())
        module.qzero_format(format=2)
    else:
        module = AwqMarlinLinear(**kwargs, backend=backend).cuda()
        qweight, qzeros = pack_awq(codes, zeros, 4)
        module.qweight.data.copy_(qweight.cuda())
        module.qzeros.data.copy_(qzeros.cuda())
        module.scales.data.copy_(scales.cuda())
        module.bias.data.copy_(linear.bias.detach().cuda())

    module.post_init()
    return module.eval(), weight.cuda(), linear.bias.detach().cuda().clone()


def _require_marlin(dtype):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("Marlin requires an NVIDIA CUDA device")
    if marlin_import_exception is not None:
        pytest.skip(f"Marlin extension unavailable: {marlin_import_exception}")
    if not marlin_runtime_available(dtype):
        pytest.skip(marlin_runtime_error(dtype))
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Marlin BF16 requires compute capability >= 8.0")


@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("in_features,out_features", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_marlin_prefill_accepts_a_column_strided_activation(
    method, in_features, out_features, dtype
):
    _require_marlin(dtype)

    backend = BACKEND.AUTO if (in_features, out_features) == SHAPES[0] else BACKEND.MARLIN
    module, weight, bias = _build_marlin(
        in_features, out_features, method, dtype, backend
    )

    # A real Mamba-style view: transpose [B, K, L] to [B, L, K]. Batch 1 lets
    # ``reshape(-1, K)`` keep stride(1) == L, which is the failing case.
    base = torch.randn(1, in_features, 5, device="cuda", dtype=dtype)
    x_strided = base.transpose(1, 2)
    assert x_strided.shape == (1, 5, in_features)
    assert x_strided.stride(-1) != 1
    x_contiguous = x_strided.contiguous()

    with torch.inference_mode():
        strided = module(x_strided)
        contiguous = module(x_contiguous)
        repeated = module(x_strided)

    torch.cuda.synchronize(x_strided.device)

    assert strided.shape == (1, 5, out_features)
    assert strided.dtype == dtype
    torch.testing.assert_close(strided, contiguous, rtol=0, atol=0)
    torch.testing.assert_close(repeated, strided, rtol=0, atol=0)

    expected = (
        x_contiguous.reshape(-1, in_features).double() @ weight.double()
        + bias.double()
    ).reshape(strided.shape)
    torch.testing.assert_close(strided.double(), expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="AUTO resolves Marlin only on NVIDIA CUDA",
)
def test_auto_selects_marlin_for_gptq_and_awq():
    def select(method, fmt):
        return select_quant_linear(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            backend=BACKEND.AUTO,
            format=fmt,
            quant_method=method,
            device=DEVICE.CUDA,
            pack=False,
            pack_dtype=torch.int32,
        )

    assert select(METHOD.GPTQ, FORMAT.GPTQ) is MarlinLinear
    assert select(METHOD.AWQ, FORMAT.GEMM) is AwqMarlinLinear
