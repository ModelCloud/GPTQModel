# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import weakref

import pytest
import torch

from gptqmodel.nn_modules.qlinear._numerics import linear_fp32_reduce
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear
from tests.test_torch_weight_storage import state_tensors


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["matrix", "batched", "transposed", "sliced"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_rounds_after_bias(dtype, layout, with_bias):
    value = 300.0 if with_bias else 3.0
    x = torch.full((2, 3, 8 if layout == "sliced" else 4), value, device="cuda", dtype=dtype)
    if layout == "matrix":
        x = x.reshape(-1, 4)
    elif layout == "transposed":
        x = x.transpose(0, 1)
    elif layout == "sliced":
        x = x[..., ::2]
    weight = torch.full((8, 4), 100.0, device="cuda", dtype=dtype)
    bias = torch.full((8,), -60000.0, device="cuda", dtype=dtype) if with_bias else None
    oracle = x.double() @ weight.double().t()
    if bias is not None:
        oracle += bias.double()

    for _ in range(2):
        actual = linear_fp32_reduce(x, weight, bias)
        assert actual.dtype == dtype
        assert actual.shape == x.shape[:-1] + (8,)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, oracle.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vptq_noncontiguous_bias_and_weight_storage(dtype, monkeypatch):
    module = (
        VPTQLinear(
            bits=1,
            in_features=128,
            out_features=64,
            bias=True,
            vector_lens=[-1, 4],
            num_centroids=[-1, 16],
            num_res_centroids=[-1, -1],
            group_size=128,
            group_num=-1,
            enable_norm=False,
            enable_perm=False,
            dtype=dtype,
        )
        .to("cuda")
        .eval()
    )
    module.indices.zero_()
    module.bias.data.fill_(-60000.0)
    x = torch.full((2, 3, 128), 30.0, device="cuda", dtype=dtype).transpose(0, 1)
    references = []
    original = module.dequantize

    def tracked(*args, **kwargs):
        weight = original(*args, **kwargs)
        references.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(module, "dequantize", tracked)
    allowed = {tensor.untyped_storage().data_ptr() for tensor in state_tensors(module)}
    for centroid in (32.0, 16.0):
        module.centroids.weight.data.fill_(centroid)
        oracle = x.double().sum(-1, keepdim=True) * centroid + module.bias.double()
        for _ in range(2):
            with torch.inference_mode():
                actual = module(x)
            assert actual.dtype == dtype
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, oracle.to(dtype), rtol=0, atol=0)
            assert references and all(reference() is None for reference in references)
            assert all(tensor.untyped_storage().data_ptr() in allowed for tensor in state_tensors(module))


# The user-visible FP16 regression from the issue: a unit dot of sixteen 4096
# activations is exactly 65536 (legal in FP32/BF16, `inf` in FP16) and a stored
# FP32 -65536 bias must cancel it to zero.
_CANCELLATION_ROWS = 16
_CANCELLATION_VALUE = 4096.0
_CANCELLATION_BIAS = -(_CANCELLATION_ROWS * _CANCELLATION_VALUE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_fp32_reduce_never_narrows_wide_bias(dtype):
    """A bias wider than the activation must be added after a wide reduction."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")
    x = torch.full((1, _CANCELLATION_ROWS), _CANCELLATION_VALUE, device=device, dtype=dtype)
    weight = torch.ones((1, _CANCELLATION_ROWS), device=device, dtype=dtype)
    bias = torch.full((1,), _CANCELLATION_BIAS, device=device, dtype=torch.float32)
    oracle = (x.double() @ weight.double().t() + bias.double()).to(dtype)

    for _ in range(2):
        actual = linear_fp32_reduce(x, weight, bias)
        assert actual.dtype is dtype
        assert actual.shape == (1, 1)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["matrix", "batched", "transposed", "sliced"])
def test_linear_fp32_reduce_wide_bias_layouts(dtype, layout):
    """Wide-bias cancellation must hold for every input layout the module accepts."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")
    width = 8 if layout == "sliced" else _CANCELLATION_ROWS
    x = torch.full((2, 3, width * 2 if layout == "sliced" else width), _CANCELLATION_VALUE, device=device, dtype=dtype)
    if layout == "matrix":
        x = x.reshape(-1, width)
    elif layout == "transposed":
        x = x.transpose(0, 1)
    elif layout == "sliced":
        x = x[..., ::2]
    assert x.shape[-1] == width
    weight = torch.ones((1, width), device=device, dtype=dtype)
    bias = torch.full((1,), -(width * _CANCELLATION_VALUE), device=device, dtype=torch.float32)
    oracle = (x.double() @ weight.double().t() + bias.double()).to(dtype)

    actual = linear_fp32_reduce(x, weight, bias)

    assert actual.dtype is dtype
    assert actual.shape == x.shape[:-1] + (1,)
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vptq_wide_bias_cancels_without_dense_cache(dtype):
    """VPTQ must add its wide bias after a wide reduction and retain no dense weight."""

    module = (
        VPTQLinear(
            bits=1,
            in_features=_CANCELLATION_ROWS,
            out_features=1,
            bias=True,
            vector_lens=[-1, 4],
            num_centroids=[-1, 16],
            num_res_centroids=[-1, -1],
            group_size=_CANCELLATION_ROWS,
            group_num=-1,
            enable_norm=False,
            enable_perm=False,
            dtype=dtype,
        )
        .eval()
    )
    module.indices.zero_()
    module.centroids.weight.data.fill_(1.0)
    # The checkpoint stores the bias in FP32; a FP16 buffer cannot hold it.
    module.register_buffer("bias", torch.tensor([_CANCELLATION_BIAS], dtype=torch.float32))
    assert module.bias.dtype is torch.float32
    x = torch.full((1, _CANCELLATION_ROWS), _CANCELLATION_VALUE, dtype=dtype)

    references = []
    original = module.dequantize
    allowed = {tensor.untyped_storage().data_ptr() for tensor in state_tensors(module)}

    def tracked(*args, **kwargs):
        weight = original(*args, **kwargs)
        references.append(weakref.ref(weight))
        return weight

    module.dequantize = tracked
    for _ in range(2):
        with torch.inference_mode():
            actual = module(x)
        assert actual.dtype is dtype
        assert torch.isfinite(actual).all()
        assert float(actual.item()) == 0.0

    # Only the packed checkpoint buffers may persist after forward.
    assert references and all(reference() is None for reference in references)
    assert {tensor.untyped_storage().data_ptr() for tensor in state_tensors(module)} <= allowed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vptq_cuda_wide_bias_skips_fused_kernel(dtype, monkeypatch):
    """A wide stored bias must not reach the native kernel, which requires its dtype."""

    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("VPTQ_CUDA is CUDA-only")

    from gptqmodel.nn_modules.qlinear.vptq_cuda import VPTQCudaLinear

    module = (
        VPTQCudaLinear(
            bits=1,
            in_features=_CANCELLATION_ROWS,
            out_features=1,
            bias=True,
            vector_lens=[-1, 4],
            num_centroids=[-1, 16],
            num_res_centroids=[-1, -1],
            group_size=_CANCELLATION_ROWS,
            group_num=-1,
            enable_norm=False,
            enable_perm=False,
            dtype=dtype,
        )
        .to("cuda")
        .eval()
    )
    module.indices.zero_()
    module.centroids.weight.data.fill_(1.0)
    module.register_buffer("bias", torch.tensor([_CANCELLATION_BIAS], dtype=torch.float32, device="cuda"))
    # One row is inside the fused-kernel row limit; only the wide bias skips it.
    x = torch.full((1, _CANCELLATION_ROWS), _CANCELLATION_VALUE, device="cuda", dtype=dtype)

    fused_calls = []

    def fail_fused(*args, **kwargs):
        fused_calls.append(True)
        raise AssertionError("wide bias must not reach the native VPTQ kernel")

    monkeypatch.setattr("gptqmodel.utils.vptq_cuda.vptq_cuda_linear", fail_fused)

    with torch.inference_mode():
        actual = module(x)

    assert not fused_calls
    assert actual.dtype is dtype
    assert float(actual.item()) == 0.0
