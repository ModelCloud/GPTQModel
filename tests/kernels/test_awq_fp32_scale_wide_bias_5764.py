# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""An FP32 checkpoint scale must survive a wide bias epilogue (#5764).

Every AWQ GEMM checkpoint below stores one FP32 scale per group and an FP32
bias. The fixture enables a single input lane with the exactly representable
FP16 value 2048, so the exact output is ``2048 * scale - 2048`` and is itself
representable in FP16. A decoder that narrows ``1 + 2**-12`` to the activation
dtype first rounds the scale to 1.0 and returns 0 instead of 0.5; widening the
later product cannot recover those bits.

The half-scale control uses ``1 + 2**-10``, which FP16 holds exactly, so the
same fixture is lossless on every path. Both cases also prove the stored scale
keeps its checkpoint dtype and value after a forward.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.torch import TORCH_HAS_FUSED_OPS


IN_FEATURES = 128
OUT_FEATURES = 32
GROUP_SIZE = 128
BITS = 4
BIAS = -2048.0
ACTIVATION = 2048.0
# FP16 spacing at 1.0 is 2**-10, so 1 + 2**-12 is not representable: a half
# decode rounds it to 1.0 and the exact 0.5 output collapses to 0.
LOSSY_SCALE = 1.0 + 2**-12
# Exactly representable in FP16, so narrowing it is lossless.
EXACT_SCALE = 1.0 + 2**-10

CPU_KERNELS = (
    AwqTorchLinear,
    TorchAtenAwqLinear,
    TorchFusedAwqLinear,
    TorchAtenLinear,
    TorchFusedLinear,
)
FUSED_KERNELS = (TorchFusedAwqLinear, TorchFusedLinear)


def _is_gptq(cls) -> bool:
    return cls in (TorchAtenLinear, TorchFusedLinear)


def _module(
    cls,
    *,
    scale: float,
    device: torch.device | str = "cpu",
    out_features: int = OUT_FEATURES,
):
    """Build the packed fixture from the issue: code 9, zero point 8."""

    module = cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=False,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=out_features,
        bias=True,
        dtype=torch.float16,
        register_buffers=True,
        **({"format": FORMAT.GPTQ_V2} if _is_gptq(cls) else {}),
    )
    module = module.to(device).eval()
    module.qweight.fill_(0x99999999 - (1 << 32))
    module.qzeros.fill_(0x88888888 - (1 << 32))
    module.scales = torch.full_like(module.scales, scale, dtype=torch.float32)
    module.bias = torch.full_like(module.bias, BIAS, dtype=torch.float32)
    if _is_gptq(cls):
        module.g_idx.zero_()
    module.post_init()
    return module


def _fixture(scale: float, out_features: int = OUT_FEATURES):
    """Return the activation and the FP64 oracle for the single-lane input."""

    x = torch.zeros((1, IN_FEATURES), dtype=torch.float16)
    x[0, 0] = ACTIVATION
    weight64 = torch.full((IN_FEATURES, out_features), scale, dtype=torch.float64)
    bias64 = torch.full((out_features,), BIAS, dtype=torch.float64)
    expected = (x.double() @ weight64 + bias64).to(x.dtype)
    return x, expected


def _assert_scale_preserved(module, scale: float) -> None:
    """The checkpoint scale keeps its FP32 dtype and value across a forward."""

    assert module.scales.dtype == torch.float32
    torch.testing.assert_close(
        module.scales,
        torch.full_like(module.scales, scale),
        rtol=0,
        atol=0,
    )
    saved = module.state_dict().get("scales")
    assert saved is not None and saved.dtype == torch.float32
    assert torch.equal(saved.to(device=module.scales.device), module.scales)


@pytest.mark.cpu
@pytest.mark.parametrize("cls", CPU_KERNELS, ids=lambda cls: cls.__name__)
def test_fp32_scale_survives_the_wide_bias_epilogue(cls):
    if cls in FUSED_KERNELS and not TORCH_HAS_FUSED_OPS:
        pytest.skip("Torch fused ops require PyTorch>=2.8")

    module = _module(cls, scale=LOSSY_SCALE)
    x, expected = _fixture(LOSSY_SCALE)

    with torch.inference_mode():
        cold = module(x)
        warm = module(x)

    assert cold.dtype == torch.float16 and warm.dtype == torch.float16
    # 2048 * (1 + 2**-12) - 2048 == 0.5 exactly.
    assert cold[0, 0].item() == 0.5, f"{cls.__name__} lost the FP32 scale: {cold[0, 0].item()}"
    assert torch.equal(cold, expected), f"{cls.__name__} cold forward differs from the FP64 oracle"
    assert torch.equal(warm, expected), f"{cls.__name__} warm forward differs from the FP64 oracle"
    _assert_scale_preserved(module, LOSSY_SCALE)


@pytest.mark.cpu
@pytest.mark.parametrize("cls", CPU_KERNELS, ids=lambda cls: cls.__name__)
def test_fp16_exact_scale_control(cls):
    if cls in FUSED_KERNELS and not TORCH_HAS_FUSED_OPS:
        pytest.skip("Torch fused ops require PyTorch>=2.8")

    module = _module(cls, scale=EXACT_SCALE)
    x, expected = _fixture(EXACT_SCALE)

    with torch.inference_mode():
        out = module(x)

    # 2048 * (1 + 2**-10) - 2048 == 2.0 exactly.
    assert out[0, 0].item() == 2.0
    assert torch.equal(out, expected), f"{cls.__name__} differs from the FP64 oracle"
    _assert_scale_preserved(module, EXACT_SCALE)


@pytest.mark.cpu
def test_portable_decode_stays_half_for_half_scale_checkpoints():
    """The common FP16-scale checkpoint keeps the half decode (no regression)."""

    module = _module(AwqTorchLinear, scale=EXACT_SCALE)
    module.scales = module.scales.to(torch.float16)
    module.clear_weight_cache()

    assert module._decode_dtype_for_activation(torch.float16) == torch.float16


@pytest.mark.cpu
def test_portable_decode_widens_only_for_a_wide_bias_epilogue():
    """A half bias keeps the half decode; a wide bias asks for the FP32 decode."""

    module = _module(AwqTorchLinear, scale=LOSSY_SCALE)

    assert module._decode_dtype_for_activation(torch.float16) == torch.float32
    module.bias = module.bias.to(torch.float16)
    assert module._decode_dtype_for_activation(torch.float16) == torch.float16


def _awq_gemm_available() -> bool:
    if not torch.cuda.is_available():
        return False
    from gptqmodel.utils.awq import awq_runtime_available

    return awq_runtime_available()


@pytest.mark.skipif(not _awq_gemm_available(), reason="CUDA AWQ GEMM extension is unavailable")
@pytest.mark.parametrize("scale", [LOSSY_SCALE, EXACT_SCALE], ids=["lossy", "exact"])
def test_cuda_awq_gemm_keeps_fp32_scale(scale):
    device = torch.device("cuda")
    # The split-K kernel packs out_features in 64-wide blocks.
    out_features = 128
    module = _module(
        AwqGEMMLinear, scale=scale, device=device, out_features=out_features
    )
    x, expected = _fixture(scale, out_features=out_features)
    x = x.to(device)
    expected = expected.to(device)

    with torch.inference_mode():
        out = module(x)

    assert out.dtype == torch.float16
    assert torch.equal(out, expected), f"AWQ GEMM lost the FP32 scale: {out[0, 0].item()}"
    _assert_scale_preserved(module, scale)
