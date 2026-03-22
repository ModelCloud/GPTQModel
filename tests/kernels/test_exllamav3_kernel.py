# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear


def _get_quantize_exl3():
    if not torch.cuda.is_available():
        pytest.skip("EXL3 kernel verification requires CUDA/HIP.")

    try:
        from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"EXL3 quantizer unavailable: {exc}")

    return quantize_exl3


def _clone_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
        for name, tensor in tensors.items()
    }


def _build_kernels(bits: int, codebook: str) -> tuple[torch.Tensor, ExllamaV3TorchLinear, ExllamaV3Linear]:
    quantize_exl3 = _get_quantize_exl3()

    torch.manual_seed(17)
    in_features = {
        "3inst": 128,
        "mcg": 256,
        "mul1": 384,
    }[codebook]
    out_features = 128
    weight = torch.randn(in_features, out_features, device="cuda", dtype=torch.float32)
    quant_device = weight.device
    hessian = torch.eye(in_features, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_features, device="cuda", dtype=torch.float16)

    quant_args: dict[str, object] = {
        "K": bits,
        "devices": [quant_device],
        "apply_out_scales": True,
        "sigma_reg": 0.025,
        "seed": 787,
    }
    if codebook == "mcg":
        quant_args["mcg"] = True
    elif codebook == "mul1":
        quant_args["mul1"] = True

    weight_q, _, out_tensors = quantize_exl3(
        weight=weight,
        H_data={"H": hessian, "count": in_features, "finalized": False},
        quant_args=quant_args,
        return_weight_q=True,
    )

    base_tensors = dict(out_tensors)
    base_tensors["bias"] = bias

    torch_kernel = ExllamaV3TorchLinear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name=f"kernel_{codebook}_{bits}",
        tensors=_clone_tensors(base_tensors),
    ).eval()

    cuda_kernel = ExllamaV3Linear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name=f"kernel_{codebook}_{bits}",
        tensors=_clone_tensors(base_tensors),
    ).eval()

    try:
        cuda_kernel.post_init()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"EXL3 CUDA runtime unavailable: {exc}")

    return weight_q, torch_kernel, cuda_kernel


@pytest.mark.parametrize(
    ("bits", "codebook"),
    [
        (2, "3inst"),
        (2, "mcg"),
        (4, "mul1"),
    ],
)
def test_exllamav3_torch_weight_matches_quantized_reference(bits: int, codebook: str):
    weight_q, torch_kernel, _ = _build_kernels(bits, codebook)

    with torch.inference_mode():
        dense_weight = torch_kernel.get_weight_tensor(dtype=torch.float32)

    assert dense_weight.dtype == torch.float32
    assert dense_weight.shape == weight_q.shape
    assert torch.allclose(dense_weight, weight_q, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize(
    ("bits", "codebook"),
    [
        (2, "3inst"),
        (2, "mcg"),
        (4, "mul1"),
    ],
)
def test_exllamav3_cuda_small_batch_matches_torch_reference(bits: int, codebook: str):
    _, torch_kernel, cuda_kernel = _build_kernels(bits, codebook)
    x = torch.randn(9, torch_kernel.in_features, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        torch_out = torch_kernel(x)
        cuda_out = cuda_kernel(x)

    assert torch_out.dtype == x.dtype
    assert cuda_out.dtype == x.dtype
    assert cuda_out.shape == torch_out.shape
    assert torch.allclose(cuda_out, torch_out, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    ("bits", "codebook"),
    [
        (2, "3inst"),
        (2, "mcg"),
        (4, "mul1"),
    ],
)
def test_exllamav3_cuda_large_batch_matches_torch_reference(bits: int, codebook: str):
    _, torch_kernel, cuda_kernel = _build_kernels(bits, codebook)
    x = torch.randn(40, torch_kernel.in_features, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        torch_out = torch_kernel(x)
        cuda_out = cuda_kernel(x)

    assert torch_out.dtype == x.dtype
    assert cuda_out.dtype == x.dtype
    assert cuda_out.shape == torch_out.shape
    assert torch.allclose(cuda_out, torch_out, rtol=8e-2, atol=8e-2)
