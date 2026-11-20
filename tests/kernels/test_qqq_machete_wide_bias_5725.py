# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#5725: add the bias to the complete FP32 reduction, not the narrowed image.

QQQ and Machete (GPTQ and AWQ) compute in FP32 and narrow the accumulator to
the activation dtype in the epilogue. Both Python wrappers then added the bias
to that narrowed tensor, so a pre-bias partial sum above 65504 became ``inf``
even when the biased result is finite, and every other row was rounded twice.

The contract is: add the bias once to the complete reduction, then round once.
For QQQ the native kernel now owns that epilogue (the GPU tests drive the real
``QQQLinear`` forward). Machete's kernels are SM90-only, so its tests run on
CPU and pin the Python-side launch contract; native SM90 execution is not
measured here and the PR records that gap.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import gptqmodel.nn_modules.qlinear.machete as machete_module
import gptqmodel.nn_modules.qlinear.machete_awq as machete_awq_module
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear
from gptqmodel.utils.machete import scalar_types
from gptqmodel.utils.qqq import qqq_runtime_available, qqq_runtime_error


IN_FEATURES = 128
OUT_FEATURES = 128
GROUP_SIZE = 128

# The weight equals its scale, so 128 codes with x=1 sum to about 1.0003. The
# bias pushes the true result to about 1.0007, whose only FP16 neighbour in the
# interval is 1.0009765625; narrowing the partial sum first gives 1.0 and the
# second rounding keeps it there.
DOUBLE_ROUNDING_WEIGHT = 0.00781484375
DOUBLE_ROUNDING_BIAS = 0.0004
DOUBLE_ROUNDING_SINGLE_CAST = 1.0009765625


def _require_qqq():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("QQQ CUDA requires compute capability >= 8.0")
    if not qqq_runtime_available():
        pytest.skip(qqq_runtime_error() or "QQQ CUDA runtime unavailable")


def _build_qqq(weight: float, scale: float, bias: float | None) -> QQQLinear:
    linear = torch.nn.Linear(
        IN_FEATURES, OUT_FEATURES, bias=bias is not None, dtype=torch.float16
    )
    with torch.no_grad():
        linear.weight.fill_(weight)
        if bias is not None:
            linear.bias.fill_(bias)
    module = QQQLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias is not None,
    )
    scales = torch.full((OUT_FEATURES, 1), scale, dtype=torch.float32)
    module.pack(linear, scales)
    module.post_init()
    return module.cuda().eval()


def _qqq_oracle(x: torch.Tensor, weight: float, bias: float | None) -> torch.Tensor:
    matrix = torch.full(
        (OUT_FEATURES, IN_FEATURES), weight, dtype=torch.float32, device=x.device
    )
    acc = x.float() @ matrix.t()
    if bias is not None:
        acc = acc + bias
    return acc.to(torch.float16)


@pytest.mark.cuda
def test_qqq_bias_joins_the_fp32_accumulator_before_the_output_cast():
    """A 71680 pre-bias sum with a -20000 bias is 51680, not inf."""

    _require_qqq()
    module = _build_qqq(weight=70.0, scale=10.0, bias=-20000.0)
    x = torch.full((8, IN_FEATURES), 8.0, dtype=torch.float16, device="cuda")

    with torch.inference_mode():
        out = module(x)

    assert out.dtype == torch.float16
    assert out.shape == (8, OUT_FEATURES)
    assert torch.isfinite(out).all(), f"{int((~torch.isfinite(out)).sum())} non-finite"
    assert torch.equal(out, _qqq_oracle(x, 70.0, -20000.0))
    assert out[0, 0].item() == 51680.0


@pytest.mark.cuda
def test_qqq_bias_is_not_double_rounded_through_the_activation_dtype():
    """The visible cast must be the only rounding after the reduction."""

    _require_qqq()
    module = _build_qqq(
        weight=DOUBLE_ROUNDING_WEIGHT,
        scale=DOUBLE_ROUNDING_WEIGHT,
        bias=DOUBLE_ROUNDING_BIAS,
    )
    x = torch.ones((4, IN_FEATURES), dtype=torch.float16, device="cuda")

    with torch.inference_mode():
        out = module(x)

    expected = _qqq_oracle(x, DOUBLE_ROUNDING_WEIGHT, DOUBLE_ROUNDING_BIAS)
    assert out.dtype == torch.float16
    assert torch.equal(out, expected)
    assert out[0, 0].item() == DOUBLE_ROUNDING_SINGLE_CAST
    # The pre-fix order narrowed 1.0003 to 1.0 first, and 1.0 + 0.0004 rounds
    # back to 1.0.
    narrowed_first = (
        (x.float() @ torch.full(
            (OUT_FEATURES, IN_FEATURES),
            DOUBLE_ROUNDING_WEIGHT,
            dtype=torch.float32,
            device="cuda",
        ).t()).to(torch.float16).float()
        + DOUBLE_ROUNDING_BIAS
    ).to(torch.float16)
    assert narrowed_first[0, 0].item() == 1.0
    assert not torch.equal(out, narrowed_first)


@pytest.mark.cuda
def test_qqq_bias_free_forward_keeps_the_unchanged_path():
    """Without a bias the kernel still narrows the accumulator once."""

    _require_qqq()
    module = _build_qqq(
        weight=DOUBLE_ROUNDING_WEIGHT, scale=DOUBLE_ROUNDING_WEIGHT, bias=None
    )
    assert module.bias is None
    x = torch.ones((4, IN_FEATURES), dtype=torch.float16, device="cuda")

    with torch.inference_mode():
        out = module(x)

    assert out.dtype == torch.float16
    assert torch.equal(out, _qqq_oracle(x, DOUBLE_ROUNDING_WEIGHT, None))
    assert out[0, 0].item() == 1.0


@pytest.mark.cuda
def test_qqq_kernel_bias_operand_is_fp32_and_contiguous():
    """The launch operand is promoted to FP32; checkpoint storage is not."""

    _require_qqq()
    module = _build_qqq(weight=1.0, scale=1.0, bias=0.5)
    assert module.bias.dtype == torch.float16
    operand = module._kernel_bias(torch.device("cuda"))
    assert operand is not None
    assert operand.dtype == torch.float32
    assert operand.is_contiguous()
    assert operand.device.type == "cuda"
    assert module.bias.dtype == torch.float16


def _bare_machete(cls):
    """Build an uninitialized module so the test can observe the native kwargs."""

    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    module.group_size = module.in_features = module.out_features = 128
    module.scales = torch.full((1, 128), 1 / 16, dtype=torch.float16)
    module.qzeros = torch.full((1, 128), 0.0, dtype=torch.float16)
    module.qweight = torch.empty(16, 128, dtype=torch.int32)
    module.weight_type = scalar_types.uint4b8
    module.has_zero_points = False
    module.input_perm = torch.empty(0, dtype=torch.int32)
    module.adapter = None
    return module


MACHETE_CASES = (
    (machete_module, machete_module.MacheteLinear),
    (machete_awq_module, machete_awq_module.AwqMacheteLinear),
)


@pytest.mark.parametrize(("module_module", "cls"), MACHETE_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_machete_bias_uses_the_fp32_image_and_rounds_once(
    module_module, cls, dtype
):
    """Bias is added to the FP32 image and the result keeps the input dtype."""

    module = _bare_machete(cls)
    module.bias = torch.full((128,), DOUBLE_ROUNDING_BIAS, dtype=dtype)
    x = torch.ones((4, 128), dtype=dtype)
    fp32_image = torch.full((4, 128), 1.0003, dtype=torch.float32)

    with patch.object(
        module_module, "machete_mm", return_value=fp32_image.clone()
    ) as native:
        out = module(x)

    assert native.call_args.kwargs["out_type"] is torch.float32
    assert out.dtype == dtype
    expected = (fp32_image + module.bias.float()).to(dtype)
    assert torch.equal(out, expected)
    # A narrowed-first epilogue would have produced 1.0 here. BF16's wider
    # spacing cannot represent the difference at this magnitude, so pin it
    # only for FP16; the dtype and value checks above still hold for both.
    if dtype == torch.float16:
        narrowed_first = (fp32_image.to(dtype).float() + module.bias.float()).to(dtype)
        assert out[0, 0].item() == DOUBLE_ROUNDING_SINGLE_CAST
        assert not torch.equal(out, narrowed_first)


@pytest.mark.parametrize(("module_module", "cls"), MACHETE_CASES)
def test_machete_without_bias_keeps_the_native_image_dtype(module_module, cls):
    """A bias-free layer must not pay for the FP32 image or an extra copy."""

    module = _bare_machete(cls)
    module.bias = None
    x = torch.ones((4, 128), dtype=torch.float16)
    native_output = torch.full((4, 128), 1.0003, dtype=torch.float16)

    with patch.object(
        module_module, "machete_mm", return_value=native_output
    ) as native:
        out = module(x)

    assert "out_type" not in native.call_args.kwargs
    assert torch.equal(out, native_output)
    assert out.dtype == torch.float16
