# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchQuantLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel, _get_ggml_bridge
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear
from gptqmodel.models._const import DEVICE


def _build_quant_modules(bits: str) -> tuple[GGUFTorchQuantLinear, GGUFCppKernel]:
    torch.manual_seed(7)
    linear = torch.nn.Linear(64, 48, bias=True, dtype=torch.float16).cpu().eval()

    torch_kernel = GGUFTorchQuantLinear(
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
    return torch_kernel.eval(), cpp_kernel.eval()


def test_gguf_cpp_kernel_validate_once_uses_llama_cpp():
    GGUFCppKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCppKernel.cached_validate_once()
    assert ok, err
    assert _get_ggml_bridge() is not None


@pytest.mark.parametrize("bits", ["q4_k_m", "q5_k_m", "q6_k"])
def test_gguf_cpp_kernel_forward_matches_torch_kernel(bits: str):
    GGUFCppKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCppKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python unavailable: {err}")

    torch_kernel, cpp_kernel = _build_quant_modules(bits)
    x = torch.randn(9, 64, dtype=torch.float32)

    with torch.inference_mode():
        torch_out = torch_kernel(x)
        cpp_out = cpp_kernel(x)

    assert cpp_out.dtype == x.dtype
    assert cpp_out.shape == torch_out.shape
    assert torch.allclose(cpp_out, torch_out, rtol=3e-2, atol=3e-2)


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
        backend=BACKEND.GGUF_CPP,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFCppKernel
