import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from gptqmodel.quantization.dtype import dequantize_f8_e4m3


MODEL_DIR = Path("/monster/data/model/MiniMax-M2")


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="PyTorch build lacks float8_e4m3fn support",
)
@pytest.mark.skipif(not MODEL_DIR.exists(), reason="MiniMax-M2 model not found")
def test_fp8_weight_dequant_matches_scaled_matmul():
    index_path = MODEL_DIR / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    config = json.loads((MODEL_DIR / "config.json").read_text())
    weight_block_size = config.get("quantization_config", {}).get("weight_block_size", [128, 128])
    block_rows, block_cols = weight_block_size

    scale_inv_name = None
    weight_tensor = None
    scale_inv_tensor = None

    visited = set()
    for tensor_name, shard in index["weight_map"].items():
        shard_path = MODEL_DIR / shard
        if shard_path in visited:
            continue
        visited.add(shard_path)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype is torch.float8_e4m3fn:
                    candidate = key + "_scale_inv"
                    if candidate not in f.keys():
                        continue
                    scale_inv_name = candidate
                    weight_tensor = tensor
                    scale_inv_tensor = f.get_tensor(scale_inv_name)
                    break
        if weight_tensor is not None:
            break

    assert weight_tensor is not None, "No FP8 weight with matching scale_inv found"

    input_dim = weight_tensor.shape[1]
    batch = 4

    torch.manual_seed(0)
    inputs = torch.randn(batch, input_dim, dtype=torch.bfloat16)

    dequant_weight = dequantize_f8_e4m3(
        weight_tensor,
        scale_inv=scale_inv_tensor,
        axis=0,
        target_dtype=torch.bfloat16,
    )

    out_dequant = (inputs @ dequant_weight.transpose(0, 1)).to(torch.float32)

    rows, cols = weight_tensor.shape
    blocks_r = rows // block_rows
    blocks_c = cols // block_cols
    scale_inv_blocks = scale_inv_tensor.reshape(blocks_r, blocks_c)
    expanded_scale_inv = scale_inv_blocks.repeat_interleave(block_rows, dim=0).repeat_interleave(block_cols, dim=1)

    expanded_scale_inv_bf16 = expanded_scale_inv.to(torch.bfloat16)
    if torch.max(torch.abs(expanded_scale_inv_bf16)) <= 1:
        weight_native_bf16 = weight_tensor.to(torch.bfloat16) * expanded_scale_inv_bf16
    else:
        weight_native_bf16 = weight_tensor.to(torch.bfloat16) / expanded_scale_inv_bf16
    out_native = (inputs @ weight_native_bf16.transpose(0, 1)).to(torch.float32)

    assert torch.allclose(out_dequant, out_native, atol=5e-3, rtol=5e-3)
