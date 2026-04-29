# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel, GGUFCudaKernel, _get_ggml_bridge
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear


def _build_quant_modules(
    bits: str,
    *,
    include_cuda: bool = False,
) -> tuple[GGUFTorchLinear, GGUFCppKernel, GGUFCudaKernel | None]:
    torch.manual_seed(7)
    linear = torch.nn.Linear(64, 48, bias=True, dtype=torch.float16).cpu().eval()

    torch_kernel = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=48,
        bias=True,
        register_buffers=True,
    )
    torch_kernel.pack_original(linear, scales=None, zeros=None)

    cpp_kernel = GGUFCppKernel(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=48,
        bias=True,
        register_buffers=True,
    )
    cpp_kernel.load_state_dict(torch_kernel.state_dict(), strict=True)

    cuda_kernel = None
    if include_cuda:
        cuda_kernel = GGUFCudaKernel(
            bits=bits,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=64,
            out_features=48,
            bias=True,
            register_buffers=True,
        )
        cuda_kernel.load_state_dict(torch_kernel.state_dict(), strict=True)
        cuda_kernel = cuda_kernel.eval()
    return torch_kernel.eval(), cpp_kernel.eval(), cuda_kernel


def test_gguf_cpp_kernel_validate_once_uses_llama_cpp():
    GGUFCppKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCppKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python unavailable: {err}")
    assert ok, err
    assert _get_ggml_bridge() is not None


@pytest.mark.parametrize("bits", ["q4_k_s", "q4_k_m", "q5_k_m", "q6_k"])
def test_gguf_cpp_kernel_forward_matches_torch_kernel(bits: str):
    GGUFCppKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCppKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python unavailable: {err}")

    torch_kernel, cpp_kernel, _ = _build_quant_modules(bits)
    x = torch.randn(9, 64, dtype=torch.float32)

    with torch.inference_mode():
        torch_out = torch_kernel(x)
        cpp_out = cpp_kernel(x)

    assert cpp_out.dtype == x.dtype
    assert cpp_out.shape == torch_out.shape
    assert torch.allclose(cpp_out, torch_out, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("bits", ["q4_k_s", "q4_k_m", "q5_k_m", "q6_k"])
def test_gguf_cuda_kernel_forward_matches_torch_kernel(bits: str):
    GGUFCudaKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCudaKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python CUDA unavailable: {err}")

    torch_kernel, _, cuda_kernel = _build_quant_modules(bits, include_cuda=True)
    torch_kernel = torch_kernel.to(device="cuda")
    assert cuda_kernel is not None
    cuda_kernel = cuda_kernel.to(device="cuda")
    x = torch.randn(9, 64, dtype=torch.float16, device="cuda")

    with torch.inference_mode():
        torch_out = torch_kernel(x)
        cuda_out = cuda_kernel(x)

    assert cuda_out.dtype == x.dtype
    assert cuda_out.device.type == "cuda"
    assert cuda_out.shape == torch_out.shape
    assert torch.allclose(cuda_out, torch_out, rtol=8e-2, atol=8e-2)


def test_gguf_cuda_kernel_reuses_cached_plan():
    GGUFCudaKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCudaKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python CUDA unavailable: {err}")

    _, _, cuda_kernel = _build_quant_modules("q4_k_m", include_cuda=True)
    assert cuda_kernel is not None
    cuda_kernel = cuda_kernel.to(device="cuda")
    x = torch.randn(9, 64, dtype=torch.float16, device="cuda")

    assert cuda_kernel._ggml_cuda_plans == {}
    with torch.inference_mode():
        first = cuda_kernel(x)
        assert len(cuda_kernel._ggml_cuda_plans) == 1
        first_plan = next(iter(cuda_kernel._ggml_cuda_plans.values()))
        second = cuda_kernel(x)
        second_plan = next(iter(cuda_kernel._ggml_cuda_plans.values()))

    assert first_plan is second_plan
    assert torch.allclose(first, second, rtol=0, atol=0)


def test_gguf_cuda_kernel_fp32_preserves_output_dtype():
    GGUFCudaKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCudaKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python CUDA unavailable: {err}")

    torch_kernel, _, cuda_kernel = _build_quant_modules("q4_k_m", include_cuda=True)
    torch_kernel = torch_kernel.to(device="cuda")
    assert cuda_kernel is not None
    cuda_kernel = cuda_kernel.to(device="cuda")
    x = torch.randn(9, 64, dtype=torch.float32, device="cuda")

    with torch.inference_mode():
        torch_out = torch_kernel(x)
        cuda_out = cuda_kernel(x)

    assert cuda_out.dtype == torch.float32
    assert cuda_out.device.type == "cuda"
    assert torch.allclose(cuda_out, torch_out, rtol=8e-2, atol=8e-2)


def test_gguf_cpp_kernel_explicit_backend_selection():
    GGUFCppKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCppKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python unavailable: {err}")

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.GGUF_CPP_CPU,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFCppKernel


def test_gguf_cuda_kernel_explicit_backend_selection():
    GGUFCudaKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCudaKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python CUDA unavailable: {err}")

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.GGUF_CPP_CUDA,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFCudaKernel
