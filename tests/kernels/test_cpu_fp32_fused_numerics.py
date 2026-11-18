# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CPU FP32 TorchFused kernels must not round activations or outputs to BF16.

``TorchFusedLinear``/``TorchFusedAwqLinear`` declare ``torch.float32`` support
and AUTO selects them for every FP32 GPTQ/AWQ checkpoint on CPU. The int4pack
op computes in FP32, but the forward used to narrow the activation to BF16
before the op and round the result back, which made FP32 less accurate than
BF16 (#5087). Each case below compares the forward against an FP64 oracle at
FP32 tolerance.
"""

import pytest
import torch

from gptqmodel.looper.linear_mode import LinearMode
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.utils.torch import TORCH_HAS_FUSED_OPS


pytestmark = [pytest.mark.cpu]

IN_FEATURES = 512
OUT_FEATURES = 256
GROUP_SIZE = 128
BITS = 4
ROWS = 8
# The direct int4pack path is ~1e-6 on this shape; the BF16 round trip that
# #5087 reported is ~2e-3.
FP32_TOL = 1e-5


def _dense(*, bias: bool) -> torch.nn.Linear:
    torch.manual_seed(0)
    return torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=bias, dtype=torch.float32)


def _gptq_kwargs(*, bias: bool):
    return {
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "sym": True,
        "desc_act": False,
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "bias": bias,
        "format": FORMAT.GPTQ_V2,
    }


def _pack_gptq(klass, *, bias: bool):
    module = klass(**_gptq_kwargs(bias=bias))
    groups = IN_FEATURES // GROUP_SIZE
    module.pack(
        _dense(bias=bias),
        torch.full((OUT_FEATURES, groups), 1 / 128, dtype=torch.float32),
        torch.full((OUT_FEATURES, groups), 1 << (BITS - 1), dtype=torch.int32),
        torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE,
    )
    if hasattr(module, "post_init"):
        module.post_init()
    return module.eval()


def _awq_kwargs(*, bias: bool):
    return {
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "sym": True,
        "desc_act": False,
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "bias": bias,
        "register_buffers": True,
    }


def _pack_awq(klass, *, bias: bool):
    groups = IN_FEATURES // GROUP_SIZE
    scales = torch.full((OUT_FEATURES, groups), 1 / 128, dtype=torch.float32)
    zeros = torch.full((OUT_FEATURES, groups), 1 << (BITS - 1), dtype=torch.int32)

    source = AwqTorchLinear(**_awq_kwargs(bias=bias))
    source.pack(_dense(bias=bias), scales, zeros)
    if hasattr(source, "post_init"):
        source.post_init()
    source.eval()

    module = klass(**_awq_kwargs(bias=bias))
    for name in ("qweight", "qzeros", "scales", "bias"):
        src = getattr(source, name, None)
        dst = getattr(module, name, None)
        if (
            isinstance(src, torch.Tensor)
            and isinstance(dst, torch.Tensor)
            and src.shape == dst.shape
        ):
            dst.data.copy_(src.to(device=dst.device, dtype=dst.dtype))
    if hasattr(module, "post_init"):
        module.post_init()
    return module.eval(), source


def _gptq_reference(source, x):
    weight = source.dequantize_weight().double()
    reference = x.double() @ weight
    if source.bias is not None:
        reference = reference + source.bias.double()
    return reference


def _awq_reference(source, x):
    weight = dequantize_gemm(
        qweight=source.qweight,
        qzeros=source.qzeros,
        scales=source.scales,
        bits=BITS,
        group_size=GROUP_SIZE,
    ).double()
    reference = x.double() @ weight
    if source.bias is not None:
        reference = reference + source.bias.double()
    return reference


def _rel_error(actual, reference):
    return ((actual.double() - reference).norm() / reference.norm()).item()


@pytest.mark.skipif(not TORCH_HAS_FUSED_OPS, reason="Torch fused ops require PyTorch>=2.8")
@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
def test_cpu_fp32_torch_fused_matches_fp64_oracle(bias: bool):
    reference_source = _pack_gptq(TorchLinear, bias=bias)
    module = _pack_gptq(TorchFusedLinear, bias=bias)

    torch.manual_seed(1)
    x = torch.randn(ROWS, IN_FEATURES, dtype=torch.float32)
    reference = _gptq_reference(reference_source, x)

    with torch.inference_mode():
        out = module(x)

    assert out.dtype == torch.float32
    assert module.linear_mode == LinearMode.INFERENCE, (
        "the FP32 case must exercise the fused int4pack path, not dense dequantization"
    )
    assert module.scales_and_zeros.dtype == torch.float32
    assert _rel_error(out, reference) <= FP32_TOL


@pytest.mark.skipif(not TORCH_HAS_FUSED_OPS, reason="Torch fused ops require PyTorch>=2.8")
@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
def test_cpu_fp32_torch_fused_awq_matches_fp64_oracle(bias: bool):
    module, source = _pack_awq(TorchFusedAwqLinear, bias=bias)

    torch.manual_seed(1)
    x = torch.randn(ROWS, IN_FEATURES, dtype=torch.float32)
    reference = _awq_reference(source, x)

    with torch.inference_mode():
        out = module(x)

    assert out.dtype == torch.float32
    assert module.linear_mode == LinearMode.INFERENCE, (
        "the FP32 case must exercise the fused int4pack path, not dense dequantization"
    )
    assert module.scales_and_zeros.dtype == torch.float32
    assert _rel_error(out, reference) <= FP32_TOL
