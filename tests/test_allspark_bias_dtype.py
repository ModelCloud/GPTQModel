# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AllSpark bias storage width.

An AllSpark layer adds its bias inside the kernel's FP32 epilogue, so the
buffer only has to *store* the value exactly. A bias that is exact in the
activation dtype therefore stays FP16/BF16 and keeps the layer on the
half-precision GEMM path; anything else is kept FP32 and takes the fallback.

These tests are CPU-only: they check dtype selection and the HIP prefill route,
never a compiled kernel.
"""

import math

import pytest
import torch

from gptqmodel.nn_modules.qlinear.allspark import AllSparkLinear, _bias_storage_dtype
from gptqmodel.utils.allspark_hip import _prefill_bias_path, _prepare_bias


_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AllSparkLinear construction needs a CUDA runtime"
)


def _make_module(dtype: torch.dtype, *, k: int = 32, n: int = 128) -> AllSparkLinear:
    return AllSparkLinear(
        bits=8, group_size=-1, desc_act=False, sym=True,
        in_features=k, out_features=n, bias=True, dtype=dtype,
    )


def _state(k: int, n: int, dtype: torch.dtype, bias: torch.Tensor) -> dict:
    return {
        "qweight": torch.zeros((k // 4, n), dtype=torch.int32),
        "scales": torch.ones((1, n), dtype=dtype),
        "bias": bias,
    }


def _exact_bias(n: int = 128) -> torch.Tensor:
    # Multiples of 0.25 are exact in both FP16 and BF16.
    return torch.arange(n, dtype=torch.float32) * 0.25 - 8.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bias_storage_dtype_keeps_exact_values(dtype):
    assert _bias_storage_dtype(_exact_bias(), dtype) == dtype
    assert _bias_storage_dtype(torch.tensor([0.0, -0.0], dtype=torch.float32), dtype) == dtype


def test_bias_storage_dtype_falls_back_for_fp16_range():
    # -65536 is finite FP32 but -inf as FP16; BF16 still holds it exactly.
    values = torch.tensor([-65536.0], dtype=torch.float32)
    assert _bias_storage_dtype(values, torch.float16) == torch.float32
    assert _bias_storage_dtype(values, torch.bfloat16) == torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bias_storage_dtype_falls_back_for_mantissa(dtype):
    values = torch.tensor([math.pi, 1.0000001], dtype=torch.float32)
    assert _bias_storage_dtype(values, dtype) == torch.float32


@_requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bias_buffer_starts_in_activation_dtype(dtype):
    assert _make_module(dtype).bias.dtype == dtype


@_requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_load_keeps_exact_bias_in_activation_dtype(dtype):
    module = _make_module(dtype)
    bias = _exact_bias()
    module.load_state_dict(_state(32, 128, dtype, bias), strict=True)
    assert module.bias.dtype == dtype
    torch.testing.assert_close(module.bias.float(), bias)


@_requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_load_keeps_inexact_bias_fp32(dtype):
    module = _make_module(dtype)
    bias = torch.full((128,), math.pi, dtype=torch.float32)
    module.load_state_dict(_state(32, 128, dtype, bias), strict=True)
    assert module.bias.dtype == torch.float32
    assert module.bias[0].item() == bias[0].item()


@_requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_post_init_narrows_exact_fp32_bias(dtype):
    # The loader keeps checkpoint biases FP32 while casting the model dtype;
    # post_init() is what moves an exactly-representable one back to half.
    module = _make_module(dtype)
    bias = _exact_bias()
    module.register_buffer("bias", bias.clone())
    module._narrow_bias_if_lossless()
    assert module.bias.dtype == dtype
    torch.testing.assert_close(module.bias.float(), bias)


@_requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_post_init_keeps_inexact_fp32_bias(dtype):
    module = _make_module(dtype)
    bias = torch.full((128,), math.pi, dtype=torch.float32)
    module.register_buffer("bias", bias.clone())
    module._narrow_bias_if_lossless()
    assert module.bias.dtype == torch.float32
    assert module.bias[0].item() == bias[0].item()


@_requires_cuda
def test_fp32_fallback_bias_survives_dtype_move():
    module = _make_module(torch.float16)
    bias = torch.full((128,), math.pi, dtype=torch.float32)
    module.register_buffer("bias", bias.clone())
    module.to(torch.bfloat16)
    assert module.bias.dtype == torch.float32
    assert module.bias[0].item() == bias[0].item()


@_requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias_value, expect_dtype", [
    (0.25, "half"),
    (math.pi, torch.float32),
])
def test_checkpoint_loader_then_post_init_picks_bias_width(tmp_path, dtype, bias_value, expect_dtype):
    """The model loader keeps biases FP32; post_init() re-narrows an exact one."""
    from safetensors.torch import save_file

    from gptqmodel.utils.model import load_checkpoint_in_model_then_tie_weights

    k, n = 32, 128
    bias = torch.full((n,), bias_value, dtype=torch.float32)
    state = _state(k, n, dtype, bias)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _make_module(dtype, k=k, n=n)

        def tie_weights(self, **_kwargs):
            pass

    checkpoint = tmp_path / "model.safetensors"
    save_file({f"proj.{name}": value for name, value in state.items()}, checkpoint,
              metadata={"format": "pt"})
    model = Wrapper()
    model.proj.bias.zero_()
    load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(checkpoint),
        device_map={"": "cpu"},
        dtype=dtype,
        keep_in_fp32_modules=["proj.bias"],
        fp32_bf16_buffer_names=["proj.bias"],
    )
    # The loader deliberately keeps the checkpoint bias wide.
    assert model.proj.bias.dtype == torch.float32
    model.proj._narrow_bias_if_lossless()
    if expect_dtype == "half":
        assert model.proj.bias.dtype == dtype
        torch.testing.assert_close(model.proj.bias.float(), bias)
    else:
        assert model.proj.bias.dtype == torch.float32
        assert model.proj.bias[0].item() == bias[0].item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prepared_half_bias_selects_half_prefill_path(dtype):
    a = torch.zeros((32, 64), dtype=dtype)
    bias = torch.zeros((128,), dtype=dtype)
    prepared = _prepare_bias(a, bias, 128)
    assert prepared is bias
    assert _prefill_bias_path(prepared, a.dtype) == "addmm"


def test_prepared_fp32_bias_selects_fp32_prefill_fallback():
    a = torch.zeros((32, 64), dtype=torch.float16)
    bias = torch.full((128,), -100000.0, dtype=torch.float32)
    prepared = _prepare_bias(a, bias, 128)
    assert prepared.dtype == torch.float32
    assert _prefill_bias_path(prepared, a.dtype) == "fp32"


def test_other_half_bias_is_widened_to_fp32_prefill_fallback():
    a = torch.zeros((32, 64), dtype=torch.float16)
    bias = torch.zeros((128,), dtype=torch.bfloat16)
    prepared = _prepare_bias(a, bias, 128)
    assert prepared.dtype == torch.float32
    assert _prefill_bias_path(prepared, a.dtype) == "fp32"


def test_no_bias_selects_plain_prefill_path():
    assert _prefill_bias_path(None, torch.float16) == "plain"


class _FakeProps:
    multi_processor_count = 1
    major = 9
    minor = 0


def _patch_hip_prefill(monkeypatch, weight: torch.Tensor) -> None:
    from gptqmodel.utils import allspark_hip

    monkeypatch.setattr(allspark_hip, "_validate_activation", lambda a, group_size: a.shape[1])
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: _FakeProps())
    monkeypatch.setattr(
        torch.ops.gptqmodel_allspark_hip, "allspark_hip_unpack",
        lambda *args, **kwargs: weight, raising=False,
    )


def test_hip_gemm_half_bias_uses_addmm(monkeypatch):
    from gptqmodel.utils import allspark_hip

    a = torch.zeros((32, 64), dtype=torch.float16)
    weight = torch.zeros((64, 128), dtype=torch.float16)
    bias = torch.zeros((128,), dtype=torch.float16)
    _patch_hip_prefill(monkeypatch, weight)
    seen = {}

    def fake_addmm(b, m1, m2, **kwargs):
        seen["dtypes"] = (b.dtype, m1.dtype, m2.dtype)
        return m1 @ m2 + b

    monkeypatch.setattr(torch, "addmm", fake_addmm)
    out = allspark_hip.allspark_hip_gemm(
        a, torch.empty((0,), dtype=torch.uint8), torch.empty((0,), dtype=torch.float16),
        128, bias=bias,
    )
    assert seen["dtypes"] == (torch.float16, torch.float16, torch.float16)
    assert out.dtype == torch.float16


def test_hip_gemm_fp32_bias_uses_fp32_fallback(monkeypatch):
    from gptqmodel.utils import allspark_hip

    a = torch.zeros((32, 64), dtype=torch.float16)
    weight = torch.zeros((64, 128), dtype=torch.float16)
    bias = torch.full((128,), -100000.0, dtype=torch.float32)
    _patch_hip_prefill(monkeypatch, weight)
    seen = {}

    def fake_matmul(m1, m2, **kwargs):
        seen["dtypes"] = (m1.dtype, m2.dtype)
        return m1 @ m2

    monkeypatch.setattr(torch, "matmul", fake_matmul)
    out = allspark_hip.allspark_hip_gemm(
        a, torch.empty((0,), dtype=torch.uint8), torch.empty((0,), dtype=torch.float16),
        128, bias=bias,
    )
    assert seen["dtypes"] == (torch.float32, torch.float32)
    assert out.dtype == torch.float16
