# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf_cpp import (
    GGUFCppKernel,
    GGUFCudaKernel,
)


def _packed_cancellation_case(kernel_type):
    if kernel_type is GGUFCppKernel:
        ok, error = kernel_type.cached_validate_once()
        if not ok:
            pytest.skip(f"GGML CPU bridge unavailable: {error}")
    module = kernel_type(
        bits="q8_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=1,
        bias=True,
    ).eval()
    module.qweight[:, :2] = torch.tensor([0.125], dtype=torch.float16).view(torch.uint8)
    module.qweight[:, 2:] = 25
    module.bias.fill_(-100000.0)
    x = torch.full((1, 32), 1000.0, dtype=torch.float16)
    # Q8_0 stores one FP16 scale and 32 INT8 codes per block.
    decoded = torch.full((32, 1), 25 * 0.125, dtype=torch.float64)
    expected = (x.double() @ decoded + module.bias.double()).to(x.dtype)
    assert expected.item() == 0.0
    return module, x, expected


@pytest.mark.parametrize("native_dtype", [torch.float32, torch.float16])
def test_gguf_cpp_cpu_bias_is_added_before_output_rounding(monkeypatch, native_dtype):
    module, x, expected = _packed_cancellation_case(GGUFCppKernel)
    packed = module.qweight.clone()
    bias = module.bias.clone()
    monkeypatch.setattr(module, "_use_ggml_cpu_matmul", lambda _x: True)
    native = torch.tensor([[100000.0]], dtype=torch.float32).to(native_dtype)
    monkeypatch.setattr(module, "_forward_ggml_cpu", lambda _x: native)

    with torch.inference_mode():
        first = module(x)
        second = module(x)

    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    assert torch.equal(module.qweight, packed)
    assert torch.equal(module.bias, bias)
    assert set(module.state_dict()) == {"qweight", "bias"}


def test_gguf_cpp_torch_fallback_passes_bias_into_wide_matmul(monkeypatch):
    module, x, expected = _packed_cancellation_case(GGUFCppKernel)
    monkeypatch.setattr(module, "_use_ggml_cpu_matmul", lambda _x: False)
    with torch.inference_mode():
        actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
def test_gguf_cpp_cuda_native_bias_matches_independent_packed_oracle():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    ok, error = GGUFCudaKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"GGML CUDA backend unavailable: {error}")
    module, x, expected = _packed_cancellation_case(GGUFCudaKernel)
    module = module.to("cuda")
    with torch.inference_mode():
        actual = module(x.cuda())
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-3)
