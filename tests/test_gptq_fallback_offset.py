# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTQ/QQQ fallback strategies must keep the group offset.

MIDPOINT/MEAN/MEDIAN/STDCLIP derive an affine zero point and clamp it to
``[0, maxq]``. A group that does not cross zero clamped to the boundary and
shifted every dequantized value by roughly the group minimum (about four code
steps for ``[0.01, 0.05]`` at 4 bits). Extending the fitted range to include
zero keeps the zero point representable and the error within half a step.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.config import (
    Fallback,
    FallbackStrategy,
    QuantizeConfig,
)
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ


STRATEGIES = [
    FallbackStrategy.MIDPOINT,
    FallbackStrategy.MEAN,
    FallbackStrategy.MEDIAN,
    FallbackStrategy.STDCLIP,
]


def _quantize(cls, strategy, values):
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        fallback=Fallback(strategy=strategy, threshold=False),
    )
    linear = nn.Linear(128, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(values, dtype=torch.float32).reshape(1, 128))
    quantizer = cls(linear, qcfg)
    if cls is GPTQ:
        result = quantizer._fallback_quantize(strategy, 128)
    else:
        result = quantizer._fallback_quantize(strategy)
    dequant, scale, zero = result[0], result[1], result[2]
    return linear.weight, dequant, scale, zero


@pytest.mark.parametrize("cls", [GPTQ, QQQ], ids=["gptq", "qqq"])
@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
@pytest.mark.parametrize(
    "values",
    [
        [0.01 + 0.04 * i / 127 for i in range(128)],   # positive-only
        [-0.05 + 0.04 * i / 127 for i in range(128)],  # negative-only
        [-0.02 + 0.05 * i / 127 for i in range(128)],  # crosses zero
    ],
    ids=["positive_only", "negative_only", "crosses_zero"],
)
def test_fallback_error_stays_within_half_a_step(cls, strategy, values):
    weight, dequant, scale, zero = _quantize(cls, strategy, values)
    step = float(scale.flatten()[0])
    assert step > 0
    error = (dequant - weight).abs().max().item()
    assert error <= 0.5 * step + 1e-6, (
        f"{cls.__name__}/{strategy.value}: max error {error:.6f} exceeds "
        f"half a step {0.5 * step:.6f} (zero point {float(zero.flatten()[0]):.2f})"
    )


@pytest.mark.parametrize("cls", [GPTQ, QQQ], ids=["gptq", "qqq"])
@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
@pytest.mark.parametrize(
    "values",
    [
        [0.01 + 0.04 * i / 127 for i in range(128)],   # positive-only
        [-0.05 + 0.04 * i / 127 for i in range(128)],  # negative-only
        [-0.02 + 0.05 * i / 127 for i in range(128)],  # crosses zero
    ],
    ids=["positive_only", "negative_only", "crosses_zero"],
)
def test_symmetric_fallback_stores_the_fixed_kernel_zero_point(cls, strategy, values):
    """``sym=True`` must store zero = 2 ** (bits - 1) for every strategy.

    Torch/Triton/ExLlamaV2 read the stored qzeros, but Marlin ignores them for
    symmetric GPTQ and assumes 8 at 4 bits. A positive-only group used to store
    0/6/7 and made Marlin disagree with the other kernels.
    """

    _, dequant, scale, zero = _quantize(cls, strategy, values)
    weight = torch.tensor(values, dtype=torch.float32).reshape(1, 128)
    step = float(scale.flatten()[0])

    assert float(zero.flatten()[0]) == float(2 ** (4 - 1))
    assert (dequant - weight).abs().max().item() <= 0.5 * step + 1e-6


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="GPTQ Marlin needs a CUDA device",
)
def test_sym_fallback_marlin_matches_torch_on_a_positive_only_group():
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization.config import FORMAT
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.marlin import marlin_runtime_available

    if not marlin_runtime_available(torch.float16):
        pytest.skip("Marlin torch.ops kernels are not available")

    out_features, in_features = 64, 128
    values = 0.01 + 0.04 * torch.arange(in_features).float() / (in_features - 1)
    linear = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        linear.weight.copy_(values.repeat(out_features, 1))

    qcfg = QuantizeConfig(
        bits=4,
        group_size=in_features,
        sym=True,
        fallback=Fallback(strategy=FallbackStrategy.MIDPOINT, threshold=False),
    )
    quantizer = GPTQ(linear, qcfg)
    _, scale, zero, g_idx = quantizer._fallback_quantize(
        FallbackStrategy.MIDPOINT, in_features
    )[:4]
    assert float(zero.flatten()[0]) == 8.0

    torch_module = TorchLinear(
        bits=4,
        group_size=in_features,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        format=FORMAT.GPTQ_V2,
        register_buffers=False,
    )
    torch_module.pack(linear, scale, zero, g_idx)
    torch_module.post_init()

    device = torch.device("cuda")
    marlin = MarlinLinear(
        bits=4,
        group_size=in_features,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        backend=BACKEND.MARLIN,
    )
    marlin.qweight.data.copy_(torch_module.qweight.data)
    marlin.scales.data.copy_(torch_module.scales.data.to(marlin.scales.dtype))
    marlin.g_idx.data.copy_(torch_module.g_idx.data)
    marlin = marlin.to(device).eval()
    marlin.post_init()
    torch_module = torch_module.to(device).eval()

    x = torch.randn(8, in_features, device=device, dtype=torch.float16)
    with torch.no_grad():
        torch_out = torch_module(x)
        marlin_out = marlin(x)

    torch.testing.assert_close(marlin_out, torch_out, rtol=2e-3, atol=2e-3)
