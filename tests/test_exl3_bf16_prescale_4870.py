# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for #4870's BF16 -> FP16 pre-scale.

#4938 made the pre-scale per row, but still normalized a scaled row to
``(0.5, 1]`` and still triggered it at 64. A row that carries a massive
activation (the BOS token on a prefill) therefore lost its *own* small elements
to FP16 subnormals, and rows between 64 and FP16's Hadamard bound were divided
for nothing. The pre-scale now triggers only near that bound and brings a row
just inside it, so those rows are passed through untouched.
"""

import math

import pytest
import torch

from gptqmodel.nn_modules import exllamav3 as exl3_bridge
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the native EXL3 dispatch needs a CUDA device",
)


def _bridge(out_features: int = 64) -> ExllamaV3Linear:
    module = object.__new__(ExllamaV3Linear)
    module.out_features = out_features
    module.bias = None
    return module


def _prepared(module, x):
    prepared = module._native_bf16_input(x)
    assert prepared is not None
    return prepared


def _scale_rows(scale, rows: int) -> torch.Tensor:
    """The per-row scale, with ``None`` (nothing scaled) read as all ones."""
    if scale is None:
        return torch.ones(rows, 1)
    return scale.reshape(rows, -1)


def test_threshold_sits_near_the_fp16_limit():
    """64 was far below what FP16 needs; the Hadamard bound is the real limit."""
    assert exl3_bridge._FP16_SAFE_AMAX == pytest.approx(
        torch.finfo(torch.float16).max / math.sqrt(128.0)
    )
    assert exl3_bridge._FP16_SAFE_AMAX > 64.0 * 10


def test_in_range_rows_are_passed_through_untouched():
    """Rows FP16 can hold must not be divided, even well above the old 64."""
    torch.manual_seed(4870)
    x = (torch.randn(8, 512) * 32.0).to(torch.bfloat16)
    assert 64.0 < float(x.abs().max()) <= exl3_bridge._FP16_SAFE_AMAX

    native_input, output_scale = _prepared(_bridge(), x)

    # `None` and an all-ones tensor are the same decision, "scale 1" (#4955).
    assert torch.equal(_scale_rows(output_scale, x.shape[0]), torch.ones(8, 1))
    assert torch.equal(native_input, x.to(torch.float16))
    assert torch.equal(native_input.float(), x.float())


@pytest.mark.parametrize("amax", [260.0, 1176.0, 5000.0, 20000.0])
def test_massive_activation_row_keeps_its_small_elements(amax):
    """The outlier row itself must not lose its small values (#4870).

    Normalizing the row to ``(0.5, 1]`` divided it by its whole amax, so its
    ordinary elements fell into FP16 subnormals: at amax 20000 the small
    elements' norm error was 5.6e-2, at 1176 it was 3.7e-3. Scaling only as far
    as the bound keeps them exact.
    """
    torch.manual_seed(4870)
    x = (torch.randn(4, 4096) * 0.01).to(torch.bfloat16)
    x[0, 0] = amax

    native_input, output_scale = _prepared(_bridge(), x)
    scale = _scale_rows(output_scale, x.shape[0])
    restored = native_input.float()[0] * scale[0, 0]
    small = (x[0].abs() > 1e-4) & (x[0].abs() < 0.1)
    reference = x[0].float()[small]
    relative = float((restored[small] - reference).norm() / reference.norm())

    assert relative <= 1e-4, f"amax={amax}: small elements lost, rel err {relative:.3e}"


def test_wide_row_is_scaled_just_inside_the_bound():
    """A genuinely wide row is scaled, but only just inside the bound."""
    x = torch.full((4, 128), 1e5, dtype=torch.bfloat16)

    native_input, output_scale = _prepared(_bridge(), x)
    scale = float(_scale_rows(output_scale, x.shape[0])[0, 0])

    assert scale > 1.0
    assert math.log2(scale).is_integer()
    assert bool(torch.isfinite(native_input).all())
    assert float(native_input.abs().max()) <= exl3_bridge._FP16_SAFE_AMAX
    # The bound is a ceiling, not a target of 1: the row keeps its headroom.
    assert float(native_input.abs().max()) > exl3_bridge._FP16_SAFE_AMAX / 2.0
    torch.testing.assert_close(
        native_input.float() * scale, x.float(), rtol=1e-6, atol=0
    )


def test_scale_is_per_row_when_rows_differ():
    """Each row gets its own power-of-two scale; narrow rows are untouched."""
    x = torch.zeros(3, 256, dtype=torch.bfloat16)
    x[0, 0] = 1e5
    x[1, 0] = 10.0
    x[2, 0] = 2.0

    native_input, output_scale = _prepared(_bridge(), x)
    scale = _scale_rows(output_scale, x.shape[0])[:, 0]

    assert float(scale[0]) > 1.0
    assert float(scale[1]) == 1.0
    assert float(scale[2]) == 1.0
    assert math.log2(float(scale[0])).is_integer()
    assert torch.equal(native_input[1:].float(), x[1:].float())


def _mixed_magnitude_batch(outlier: float, device: str):
    """64 rows of N(0, 0.3) with one large outlier element in row 0."""
    torch.manual_seed(4870)
    rows, in_features, out_features = 64, 8192, 64
    x = (torch.randn(rows, in_features, device=device) * 0.3).to(torch.bfloat16)
    x[0, 5] = outlier
    # A small FP16 weight keeps the ordinary rows' kernel output near FP16's
    # smallest normal, which is where a per-tensor scale used to lose them.
    weight = (torch.randn(in_features, out_features, device=device) * 0.0002).to(
        torch.float16
    )
    return x, weight


@requires_cuda
@pytest.mark.parametrize("outlier", [500.0, 2000.0, 10000.0])
def test_mixed_magnitude_fp16_gemm_keeps_ordinary_rows(outlier):
    """#4870's core case: one outlier row must not degrade the other rows."""
    x, weight = _mixed_magnitude_batch(outlier, "cuda")
    module = _bridge(weight.shape[1])

    native_input, output_scale = _prepared(module, x)
    # The native EXL3 kernel is FP16: it consumes FP16 and returns FP16.
    kernel_output = (native_input @ weight).to(torch.float16)
    if output_scale is not None:
        kernel_output = (kernel_output.float() * output_scale).to(torch.float16)
    reference = x.double() @ weight.double()

    actual = kernel_output[1:].double()
    relative = (actual - reference[1:]).norm(dim=1) / reference[1:].norm(dim=1)
    assert float(relative.max()) <= 5e-3, (
        f"outlier={outlier}: ordinary rows lost precision, "
        f"max rel err {float(relative.max()):.3e}"
    )
