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
