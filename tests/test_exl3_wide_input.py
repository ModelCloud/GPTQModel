# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from gptqmodel.nn_modules import exllamav3 as exl3_bridge
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from tests.accelerator_gates import requires_nvidia_cuda


requires_cuda = pytest.mark.skipif(
    not requires_nvidia_cuda(),
    reason="the native EXL3 dispatch needs a CUDA device",
)


def _bridge(out_features: int = 4) -> ExllamaV3Linear:
    """An ``ExllamaV3Linear`` without ``nn.Module.__init__`` for dispatch tests."""
    module = object.__new__(ExllamaV3Linear)
    module.out_features = out_features
    module.bias = None
    return module


class _FakeNative:
    """Stand-in for ``LinearEXL3``; records the FP16 input it receives."""

    def __init__(self, weight=None, bias=None, out_features=4):
        self.weight = weight
        self.bias = bias
        self.out_features = out_features
        self.calls = []

    def forward(self, x, params, out_dtype):
        assert x.dtype == torch.float16
        amax = float(x.abs().amax()) if x.numel() else 0.0
        self.calls.append((tuple(x.shape), amax))
        if self.weight is None:
            return torch.zeros(
                (*x.shape[:-1], self.out_features), dtype=out_dtype, device=x.device
            )
        # Mimic the native FP16 GEMM (FP16 inputs, FP16 output, bias included).
        result = x @ self.weight
        if self.bias is not None:
            result = result + self.bias
        return result.to(out_dtype)


def _install(module, native, torch_path=None):
    module._ensure_inner = lambda: native
    if torch_path is None:
        module._ensure_torch_inner = lambda: pytest.fail(
            "BF16 must use the native FP16 kernel"
        )
    else:
        module._ensure_torch_inner = lambda: torch_path


@pytest.mark.parametrize("rows", [1, 32, 33, 64])
def test_float32_inputs_use_torch_reconstruction(rows):
    module = _bridge()
    calls = []

    class TorchPath:
        def forward(self, x):
            calls.append(("torch", x.dtype))
            return torch.ones((*x.shape[:-1], 4), dtype=x.dtype)

    _install(module, _FakeNative(), TorchPath())
    output = module.forward(torch.ones((rows, 3), dtype=torch.float32))

    assert output.dtype == torch.float32
    assert output.shape == (rows, 4)
    assert calls == [("torch", torch.float32)]


@requires_cuda
@pytest.mark.parametrize("rows", [1, 32, 33, 64])
def test_bfloat16_inputs_keep_native_dispatch(rows):
    """Regression for #4767: BF16 must not fall back to the Torch path."""
    module = _bridge()
    native = _FakeNative()
    _install(module, native)

    output = module.forward(torch.ones((rows, 3), device="cuda", dtype=torch.bfloat16))

    assert output.dtype == torch.bfloat16
    assert output.shape == (rows, 4)
    assert native.calls == [((rows, 3), 1.0)]


@requires_cuda
def test_bfloat16_zero_rows_keep_native_dispatch():
    module = _bridge()
    native = _FakeNative()
    _install(module, native)

    output = module.forward(torch.empty(0, 3, device="cuda", dtype=torch.bfloat16))

    assert output.dtype == torch.bfloat16 and output.shape == (0, 4)
    assert native.calls == [((0, 3), 0.0)]


@pytest.mark.parametrize("rows", [32, 33])
def test_exllamav3_half_inputs_keep_native_dispatch(rows):
    module = object.__new__(ExllamaV3Linear)
    module.out_features = 4
    calls = []

    class Native:
        def forward(self, x, params, out_dtype):
            calls.append((x.dtype, out_dtype, x.shape[0]))
            return torch.ones((rows, 4), dtype=out_dtype)

    module._ensure_inner = lambda: Native()
    module._ensure_torch_inner = lambda: pytest.fail("FP16 must keep the native path")

    output = module.forward(torch.ones((rows, 3), dtype=torch.float16))

    assert output.shape == (rows, 4)
    assert output.dtype == torch.float16
    assert calls == [(torch.float16, torch.float16, rows)]


@requires_cuda
@pytest.mark.parametrize("rows, branch", [(32, "bc"), (33, "hgemm")])
def test_native_exl3_row_boundary_uses_expected_half_kernel(monkeypatch, rows, branch):
    from gptqmodel.exllamav3.modules.quant import exl3 as native_exl3

    calls = []

    class FakeBC:
        def run(self, x, y):
            calls.append("bc")
            y.copy_(x)

    class FakeExt:
        @staticmethod
        def had_r_128(x, y, suh, svh, scale):
            y.copy_(x)

        @staticmethod
        def hgemm(x, weight, y):
            calls.append("hgemm")
            y.copy_(x @ weight)

    monkeypatch.setattr(native_exl3, "ext", FakeExt)
    inner = object.__new__(native_exl3.LinearEXL3)
    inner.in_features = inner.out_features = 128
    inner.out_dtype = torch.float16
    inner.suh = inner.svh = inner.bias = None
    inner.bc = FakeBC()
    inner.get_inner_weight_tensor = lambda: torch.eye(128, dtype=torch.float16)

    x = torch.ones((rows, 128), dtype=torch.float16)
    actual = inner.forward(x, {})

    assert calls == [branch]
    torch.testing.assert_close(actual, x, rtol=0, atol=0)


def test_bfloat16_normal_range_cast_is_exact():
    module = _bridge()
    x = torch.linspace(-32.0, 32.0, 257, dtype=torch.bfloat16).reshape(1, 257)

    prepared = module._native_bf16_input(x)

    assert prepared is not None
    native_input, output_scale = prepared
    # `None` and an all-ones tensor are the same decision, "scale 1" (#4955).
    if output_scale is not None:
        assert torch.equal(output_scale, torch.ones_like(output_scale))
    assert native_input.dtype == torch.float16
    assert torch.equal(native_input.float(), x.float())


@pytest.mark.parametrize("magnitude", [1e4, 1e5, 1e10, 1e30])
def test_bfloat16_wide_input_is_power_of_two_prescaled(magnitude):
    module = _bridge()
    x = (torch.randn(1, 257) * magnitude).to(torch.bfloat16)

    prepared = module._native_bf16_input(x)

    assert prepared is not None
    native_input, output_scale = prepared
    assert native_input.dtype == torch.float16
    assert float(native_input.abs().amax()) <= exl3_bridge._FP16_SAFE_AMAX
    assert math.log2(float(output_scale.flatten()[0])).is_integer()
    assert float(output_scale.flatten()[0]) > 1.0
    # Power-of-two scaling is exact, so only the BF16 rounding remains.
    torch.testing.assert_close(
        native_input.float() * output_scale, x.float(), rtol=1e-6, atol=0
    )


def test_bfloat16_wide_input_prescale_handles_amax_above_fp16():
    """The pre-scale keeps a 1e5 input out of FP16's inf range."""
    module = _bridge()
    x = torch.full((2, 128), 1e5, dtype=torch.bfloat16)

    native_input, output_scale = module._native_bf16_input(x)

    assert bool(torch.isfinite(native_input).all())
    torch.testing.assert_close(
        native_input.float() * output_scale, x.float(), rtol=1e-6, atol=0
    )


def test_bfloat16_prescale_is_per_row_not_per_tensor():
    """One outlier row must not scale the ordinary rows (#4870).

    A per-tensor scale divided every row by ``2**ceil(log2(amax))``, so a
    single massive activation pushed the rest of the batch into FP16
    subnormals (smallest normal 6.1e-5) and lost most of their precision.
    """

    module = _bridge()
    torch.manual_seed(4870)
    x = (torch.randn(64, 257) * 0.3).to(torch.bfloat16)
    x[0, 5] = 1e4

    prepared = module._native_bf16_input(x)

    assert prepared is not None
    native_input, output_scale = prepared
    # Only the row that needs it is scaled.
    assert float(output_scale[0, 0]) > 1.0
    assert torch.equal(output_scale[1:], torch.ones_like(output_scale[1:]))
    # BF16 -> FP16 is exact inside the safe range, so the ordinary rows are
    # bit-identical to the input: no shared divisor, no subnormal loss.
    torch.testing.assert_close(native_input[1:].float(), x[1:].float(), rtol=0, atol=0)
    # The wide row stays finite. Its own small values do go subnormal, which is
    # inherent to a row that mixes 1e4 with 0.3; the point is that it can no
    # longer drag the other rows down with it.
    assert bool(torch.isfinite(native_input[0]).all())


@requires_cuda
@pytest.mark.parametrize("magnitude", [1.0, 1e4, 1e5])
@pytest.mark.parametrize("rows", [1, 33])
def test_wide_bfloat16_native_numerics_stay_finite_and_close(magnitude, rows):
    """Mocked native kernel: no overflow and BF16/FP64-close for wide inputs."""
    torch.manual_seed(4767)
    out_features, in_features = 32, 128
    weight = (torch.randn(in_features, out_features) * 0.02).to(
        device="cuda", dtype=torch.float16
    )
    bias = (torch.randn(out_features) * 0.01).to(device="cuda", dtype=torch.float16)
    native = _FakeNative(weight=weight, bias=bias, out_features=out_features)
    module = _bridge(out_features)
    module.bias = bias
    _install(module, native)

    x = (torch.randn(rows, in_features, device="cuda") * magnitude).to(torch.bfloat16)
    source_amax = float(x.abs().amax())
    actual = module.forward(x)

    assert actual.dtype == torch.bfloat16 and actual.shape == (rows, out_features)
    assert bool(torch.isfinite(actual).all())
    # The kernel only ever sees normalized FP16 activations: untouched below
    # the threshold, divided by a power of two above it.
    if source_amax <= exl3_bridge._FP16_SAFE_AMAX:
        assert native.calls[0][1] == pytest.approx(source_amax)
    else:
        assert native.calls[0][1] <= exl3_bridge._FP16_SAFE_AMAX

    reference = x.double() @ weight.double() + bias.double()
    difference = actual.double() - reference
    # Dot-product cancellation makes per-element relative error meaningless
    # near zero; the FP16 accumulation's scale-relative error is the bound.
    assert float(difference.norm() / reference.norm()) <= 1e-2
    assert float(difference.abs().max()) <= 2e-2 * float(reference.abs().max())


@requires_cuda
def test_nonfinite_bfloat16_keeps_torch_reconstruction():
    module = _bridge()
    calls = []

    class TorchPath:
        def forward(self, x):
            calls.append(("torch", x.dtype))
            return torch.zeros((*x.shape[:-1], 4), dtype=x.dtype, device=x.device)

    native = _FakeNative()
    _install(module, native, TorchPath())
    x = torch.ones((2, 3), device="cuda", dtype=torch.bfloat16)
    x[0, 0] = float("inf")

    output = module.forward(x)

    assert native.calls == []
    assert calls == [("torch", torch.bfloat16)]
    assert output.dtype == torch.bfloat16


def test_fp16_safe_amax_is_inside_fp16_range():
    assert 0 < exl3_bridge._FP16_SAFE_AMAX < torch.finfo(torch.float16).max
