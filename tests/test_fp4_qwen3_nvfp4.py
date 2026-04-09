# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from gptqmodel.quantization.dtype import dequantize_f4_e2m1


try:
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
except Exception:
    NVFP4Tensor = None


MODEL_DIR = Path("/mnt/SFS-6CFyUykx/models/Qwen3-8B-NVFP4")
WEIGHT_KEY = "model.layers.0.mlp.down_proj.weight"
SCALE_KEY = "model.layers.0.mlp.down_proj.weight_scale"


def _nvfp4_to_dtype(nv_tensor, dtype: torch.dtype) -> torch.Tensor:
    to_dtype = getattr(nv_tensor, "to_dtype", None)
    if callable(to_dtype):
        return to_dtype(dtype)
    return nv_tensor.dequantize(dtype)


def _load_first_layer_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][WEIGHT_KEY]

    with safe_open(MODEL_DIR / shard, framework="pt", device="cpu") as f:
        weight = f.get_tensor(WEIGHT_KEY)
        scales = f.get_tensor(SCALE_KEY)

    return weight, scales


@pytest.mark.skipif(NVFP4Tensor is None, reason="torchao NVFP4 support required")
@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Qwen3 8B NVFP4 model not available")
def test_fp4_qwen3_module_dequant_matches_nvfp4_tensor():
    weight, scales = _load_first_layer_tensors()

    dequant = dequantize_f4_e2m1(weight, scale=scales, axis=None, target_dtype=torch.bfloat16)

    nv_tensor = NVFP4Tensor(weight, scales, block_size=16, orig_dtype=torch.bfloat16)
    expected = _nvfp4_to_dtype(nv_tensor, torch.bfloat16)

    diff = torch.max(torch.abs(dequant - expected)).item()
    assert torch.allclose(dequant, expected, atol=1e-3, rtol=1e-3), diff


@pytest.mark.skipif(NVFP4Tensor is None, reason="torchao NVFP4 support required")
@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Qwen3 8B NVFP4 model not available")
def test_fp4_qwen3_module_gpu_consistency():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device("cuda:0")
    weight, scales = _load_first_layer_tensors()

    cpu = dequantize_f4_e2m1(weight, scale=scales, axis=None, target_dtype=torch.bfloat16)

    torch.cuda.set_device(dev)
    gpu_weight = weight.to(dev)
    gpu_scales = scales.to(dev)
    gpu = dequantize_f4_e2m1(gpu_weight, scale=gpu_scales, axis=None, target_dtype=torch.bfloat16)

    assert torch.allclose(cpu, gpu.cpu(), atol=1e-3, rtol=1e-3)
