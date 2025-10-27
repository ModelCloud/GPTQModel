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


MODEL_DIR = Path("/monster/data/model/Llama-3.3-70B-Instruct-FP4")


@pytest.mark.skipif(NVFP4Tensor is None, reason="torchao NVFP4 support required")
@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Llama-3.3 FP4 model not available")
def test_fp4_llama3_module_dequant_matches_nvfp4_tensor():
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    shard = sorted(set(index["weight_map"].values()))[0]

    with safe_open(MODEL_DIR / shard, framework="pt", device="cpu") as f:
        weight = f.get_tensor("model.layers.0.mlp.down_proj.weight")
        scales = f.get_tensor("model.layers.0.mlp.down_proj.weight_scale")

    dequant = dequantize_f4_e2m1(weight, scale=scales, axis=None, target_dtype=torch.bfloat16)

    nv_tensor = NVFP4Tensor(weight, scales, block_size=16, orig_dtype=torch.bfloat16)
    expected = nv_tensor.to_dtype(torch.bfloat16)

    diff = torch.max(torch.abs(dequant - expected)).item()
    assert torch.allclose(dequant, expected, atol=1e-3, rtol=1e-3), diff


@pytest.mark.skipif(NVFP4Tensor is None, reason="torchao NVFP4 support required")
@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Llama-3.3 FP4 model not available")
@pytest.mark.parametrize("device", ["cuda:7", "cuda:8"], ids=["A100", "RTX5090"])
def test_fp4_llama3_module_gpu_consistency(device):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    if dev.index is not None and dev.index >= torch.cuda.device_count():
        pytest.skip(f"CUDA device {device} not accessible")

    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    shard = sorted(set(index["weight_map"].values()))[0]

    with safe_open(MODEL_DIR / shard, framework="pt", device="cpu") as f:
        weight = f.get_tensor("model.layers.0.mlp.down_proj.weight")
        scales = f.get_tensor("model.layers.0.mlp.down_proj.weight_scale")

    cpu = dequantize_f4_e2m1(weight, scale=scales, axis=None, target_dtype=torch.bfloat16)

    torch.cuda.set_device(dev)
    gpu_weight = weight.to(dev)
    gpu_scales = scales.to(dev)
    gpu = dequantize_f4_e2m1(gpu_weight, scale=gpu_scales, axis=None, target_dtype=torch.bfloat16)

    assert torch.allclose(cpu, gpu.cpu(), atol=1e-3, rtol=1e-3)
