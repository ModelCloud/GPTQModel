# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import gptqmodel.nn_modules.qlinear.fp4 as fp4_module
import gptqmodel.quantization.dtype as dtype_module


try:
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
except Exception:
    NVFP4Tensor = None


pytestmark = pytest.mark.skipif(NVFP4Tensor is None, reason="torchao NVFP4 support required")


def _make_fp4_linear(
    *,
    in_features: int = 128,
    out_features: int = 64,
    packed_dtype: torch.dtype = torch.uint8,
) -> fp4_module.TorchFP4Linear:
    torch.manual_seed(17)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.04
    nv_weight = NVFP4Tensor.to_nvfp4(weight, block_size=16)
    packed = nv_weight.qdata if packed_dtype is torch.uint8 else nv_weight.qdata.view(packed_dtype)
    return fp4_module.TorchFP4Linear(
        in_features=in_features,
        out_features=out_features,
        weight=packed,
        weight_scale=nv_weight.scale,
        weight_block_size=16,
        orig_dtype=torch.bfloat16,
        bias=torch.randn(out_features, dtype=torch.bfloat16) * 0.01,
    ).eval()


def _oracle(module: fp4_module.TorchFP4Linear, x: torch.Tensor) -> torch.Tensor:
    native_weight = NVFP4Tensor(
        module.weight,
        module.weight_scale,
        block_size=module.weight_block_size,
        orig_dtype=module.orig_dtype,
    )
    dense_weight = native_weight.dequantize(x.dtype)
    bias = module.bias.to(dtype=x.dtype) if isinstance(module.bias, torch.Tensor) else None
    return F.linear(x, dense_weight, bias)


@pytest.mark.parametrize(
    "packed_dtype",
    [
        torch.uint8,
        pytest.param(
            getattr(torch, "float4_e2m1fn_x2", None),
            marks=pytest.mark.skipif(
                not hasattr(torch, "float4_e2m1fn_x2"),
                reason="float4_e2m1fn_x2 unavailable",
            ),
        ),
    ],
    ids=["uint8", "float4"],
)
@pytest.mark.parametrize("activation_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fp4_cpu_forward_matches_torchao_oracle(
    packed_dtype: torch.dtype,
    activation_dtype: torch.dtype,
):
    module = _make_fp4_linear(packed_dtype=packed_dtype)
    x = torch.randn(5, module.in_features, dtype=activation_dtype)

    with torch.inference_mode():
        output = module(x)
        expected = _oracle(module, x)

    assert output.dtype is activation_dtype
    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)


def test_fp4_cpu_forward_decodes_per_call_without_dense_cache(monkeypatch):
    module = _make_fp4_linear()
    x = torch.randn(5, module.in_features, dtype=torch.bfloat16)
    calls = 0
    dequantize = fp4_module.dequantize_f4_e2m1

    def counted_dequantize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return dequantize(*args, **kwargs)

    monkeypatch.setattr(fp4_module, "dequantize_f4_e2m1", counted_dequantize)
    monkeypatch.setattr(
        module,
        "_native_weight",
        lambda: pytest.fail("CPU forward must not construct the torchao wrapper"),
    )

    with torch.inference_mode():
        first = module(x)
        second = module(x)

    assert calls == 2
    assert not any("cache" in name for name in module.__dict__)
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_fp4_cpu_forward_tracks_live_weight_and_scale_mutation():
    module = _make_fp4_linear()
    x = torch.randn(7, module.in_features, dtype=torch.bfloat16)
    with torch.inference_mode():
        before = module(x)

    torch.manual_seed(29)
    replacement = NVFP4Tensor.to_nvfp4(
        torch.randn(module.out_features, module.in_features, dtype=torch.bfloat16) * 0.04,
        block_size=module.weight_block_size,
    )
    with torch.no_grad():
        module.weight.copy_(replacement.qdata.view_as(module.weight))
        module.weight_scale.copy_(replacement.scale.view_as(module.weight_scale))

    with torch.inference_mode():
        after = module(x)
        expected = _oracle(module, x)

    assert not torch.equal(after, before)
    torch.testing.assert_close(after, expected, rtol=2e-3, atol=2e-3)


def test_fp4_cpu_forward_uses_reference_fallback(monkeypatch):
    module = _make_fp4_linear()
    x = torch.randn(5, module.in_features, dtype=torch.bfloat16)
    monkeypatch.setattr(dtype_module, "_load_floatx_cpu_ops", lambda: None)

    with torch.inference_mode():
        output = module(x)
        expected = _oracle(module, x)

    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)



# One E2M1 code with a unit block scale decodes to exactly 33/32. Sixteen
# 4096-valued FP16 activations therefore produce 16 * 4096 * 33/32 = 67584,
# which is above the largest finite FP16 value (65504). A stored FP32
# ``-67584`` bias must cancel it to zero instead of being narrowed to FP16
# ``-inf`` first.
_FP4_WEIGHT_VALUE = 33 / 32
_FP4_CANCELLING_BIAS = -16 * 4096 * _FP4_WEIGHT_VALUE


def _make_cancellation_linear(*, bias_dtype: torch.dtype, bias_value: float) -> fp4_module.TorchFP4Linear:
    """Build a one-output FP4 layer whose finite dot is cancelled by the bias."""

    nv_weight = NVFP4Tensor.to_nvfp4(torch.ones(1, 16, dtype=torch.bfloat16), block_size=16)
    return fp4_module.TorchFP4Linear(
        in_features=16,
        out_features=1,
        weight=nv_weight.qdata,
        weight_scale=nv_weight.scale,
        weight_block_size=16,
        orig_dtype=torch.bfloat16,
        bias=torch.tensor([bias_value], dtype=bias_dtype),
    ).eval()


def _fp4_dense_weight(module: fp4_module.TorchFP4Linear, dtype: torch.dtype) -> torch.Tensor:
    return NVFP4Tensor(
        module.weight,
        module.weight_scale,
        block_size=module.weight_block_size,
        orig_dtype=module.orig_dtype,
    ).dequantize(dtype)


@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_fp4_cpu_forward_keeps_fp32_bias_range(activation_dtype):
    """An FP32 bias must be added before the output is rounded to the activation dtype."""

    module = _make_cancellation_linear(
        bias_dtype=torch.float32,
        bias_value=_FP4_CANCELLING_BIAS,
    )
    assert module.bias.dtype is torch.float32
    x = torch.full((1, 16), 4096.0, dtype=activation_dtype)
    dense_weight = _fp4_dense_weight(module, torch.float64)
    assert float(dense_weight[0, 0]) == pytest.approx(_FP4_WEIGHT_VALUE, abs=0)
    oracle = (x.double() @ dense_weight.t() + module.bias.double()).to(activation_dtype)
    # The low-precision dot alone already overflows; only the wide epilogue
    # recovers 0.
    dot_narrow = (x.to(activation_dtype) @ dense_weight.to(activation_dtype).t()).to(
        activation_dtype
    )
    bias_narrow = module.bias.to(activation_dtype)

    with torch.inference_mode():
        output = module(x)

    assert output.dtype is activation_dtype
    assert torch.isfinite(output).all()
    assert float(output.item()) == float(oracle.item()) == 0.0
    if activation_dtype is torch.float16:
        assert not torch.isfinite(dot_narrow).all()
        assert not torch.isfinite(bias_narrow).all()
    else:
        # BF16 can hold both operands, so it only exercises the exact
        # cancellation; the FP16 case above is the overflow regression.
        assert torch.isfinite(dot_narrow).all()
        assert float(bias_narrow.item()) == -67584.0
        assert float(dot_narrow.item()) == 67584.0


@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_fp4_cpu_forward_promotes_narrow_bias(activation_dtype):
    """A bias narrower than the activation is widened losslessly, never narrowed."""

    bias_value = -1.03125
    module = _make_cancellation_linear(bias_dtype=torch.float16, bias_value=bias_value)
    x = torch.full((1, 16), 1.0, dtype=activation_dtype)
    dense_weight = _fp4_dense_weight(module, torch.float64)
    oracle = (x.double() @ dense_weight.t() + module.bias.double()).to(activation_dtype)

    with torch.inference_mode():
        output = module(x)

    assert output.dtype is activation_dtype
    torch.testing.assert_close(output, oracle, rtol=2e-3, atol=2e-3)


def test_fp4_representable_fp32_bias_uses_narrow_epilogue():
    """A wide bias whose values fit the activation dtype must not force the FP32 path.

    The checkpoint bias is kept at its stored width, which is normally FP32, so
    keying the epilogue off the buffer dtype alone would copy the whole weight
    to FP32 on every CPU forward. An exact cast returns the same bias.
    """

    module = _make_cancellation_linear(
        bias_dtype=torch.float32,
        bias_value=0.5,
    )
    assert module.bias.dtype is torch.float32
    x = torch.full((1, 16), 4096.0, dtype=torch.float16)

    bias, wide = module._epilogue_bias(x, module.bias)
    assert not wide
    assert bias.dtype is x.dtype
    assert bias.item() == 0.5

    # An inexact value still has to keep the wide epilogue.
    module.bias.data.fill_(-65536.0)
    bias, wide = module._epilogue_bias(x, module.bias)
    assert wide
    assert bias.dtype is torch.float32


def test_fp4_narrow_epilogue_rounds_once_for_representable_bias():
    """The exact-cast path must still match the FP64 oracle bitwise."""

    module = _make_cancellation_linear(bias_dtype=torch.float32, bias_value=-1.03125)
    x = torch.full((1, 16), 1.0, dtype=torch.float16)
    dense_weight = _fp4_dense_weight(module, torch.float64)
    oracle = (x.double() @ dense_weight.t() + module.bias.double()).to(torch.float16)

    with torch.inference_mode():
        output = module(x)

    assert output.dtype is torch.float16
    torch.testing.assert_close(output, oracle, rtol=0, atol=0)


def test_fp4_cpu_forward_uses_the_exact_cast_gate(monkeypatch):
    """The forward must use the exact-cast decision, not the buffer dtype.

    ``_epilogue_bias`` returning the narrow bias is only useful if the ephemeral
    CPU path consumes it; this pins the call site so removing the call cannot
    silently reopen the per-forward FP32 weight copy.
    """

    module = _make_cancellation_linear(bias_dtype=torch.float32, bias_value=0.5)
    assert module.bias.dtype is torch.float32
    seen = []
    original = fp4_module.F.linear

    def tracked(x, weight, bias=None):
        seen.append(None if bias is None else bias.dtype)
        return original(x, weight, bias)

    monkeypatch.setattr(fp4_module.F, "linear", tracked)
    x = torch.full((1, 16), 1.0, dtype=torch.float16)

    with torch.inference_mode():
        module(x)

    assert seen == [torch.float16]

    # An out-of-range value must still reach the wide helper with its FP32 bias.
    module.bias.data.fill_(-65536.0)
    seen.clear()
    wide_bias = []
    original_wide = fp4_module.linear_wide_bias

    def tracked_wide(x, weight, bias=None):
        wide_bias.append(None if bias is None else bias.dtype)
        return original_wide(x, weight, bias)

    monkeypatch.setattr(fp4_module, "linear_wide_bias", tracked_wide)
    with torch.inference_mode():
        module(x)

    assert seen == []
    assert wide_bias == [torch.float32]
