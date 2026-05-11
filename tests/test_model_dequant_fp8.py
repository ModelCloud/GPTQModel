# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.utils.model_dequant import dequantize_model

def _write_index(model_dir, shard_name: str, keys: list[str]) -> None:
    weight_map = dict.fromkeys(keys, shard_name)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_model_fp8_resolves_scale_inv_from_other_shard(tmp_path):
    model_dir = tmp_path / "fp8_cross_shard"
    output_dir = tmp_path / "fp8_cross_shard_out"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "format": "float8_e4m3fn",
            "quant_method": "fp8",
            "weight_block_size": [2, 4],
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    torch.manual_seed(0)
    weight = torch.randn(4, 8, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale_inv = torch.linspace(2.0, 3.5, steps=4, dtype=torch.float32).view(2, 2)

    weight_shard = "model-00001-of-00002.safetensors"
    scale_shard = "model-00002-of-00002.safetensors"
    save_file({"linear.weight": weight}, str(model_dir / weight_shard))
    save_file({"linear.weight_scale_inv": scale_inv}, str(model_dir / scale_shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "linear.weight": weight_shard,
                    "linear.weight_scale_inv": scale_shard,
                }
            }
        ),
        encoding="utf-8",
    )

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / weight_shard, framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == {"linear.weight"}
        weight_out = reader.get_tensor("linear.weight")

    expanded_scale_inv = scale_inv.repeat_interleave(2, dim=0).repeat_interleave(4, dim=1)
    expected = weight.to(torch.bfloat16) / expanded_scale_inv.to(torch.bfloat16)
    torch.testing.assert_close(weight_out, expected)

    output_index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert output_index["weight_map"] == {"linear.weight": weight_shard}
    assert not (output_dir / scale_shard).exists()


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_model_fp8_allows_partial_edge_blocks(tmp_path):
    model_dir = tmp_path / "fp8_partial_blocks"
    output_dir = tmp_path / "fp8_partial_blocks_out"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "format": "float8_e4m3fn",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    rows, cols = 576, 256
    block_rows, block_cols = 128, 128

    torch.manual_seed(0)
    weight = torch.randn(rows, cols, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale_inv = torch.linspace(2.0, 3.0, steps=10, dtype=torch.float32).view(5, 2)

    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": weight,
            "linear.weight_scale_inv": scale_inv,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv"])

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        weight_out = reader.get_tensor("linear.weight")
        assert "linear.weight_scale_inv" not in reader.keys()

    expanded_scale_inv = scale_inv.repeat_interleave(block_rows, dim=0)
    expanded_scale_inv = expanded_scale_inv.repeat_interleave(block_cols, dim=1)
    expanded_scale_inv = expanded_scale_inv[:rows, :cols].to(torch.bfloat16)
    expected = weight.to(torch.bfloat16) / expanded_scale_inv

    torch.testing.assert_close(weight_out, expected)
