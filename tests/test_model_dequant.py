# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.quantization.dtype import dequantize_f8_e4m3
from gptqmodel.utils.model_dequant import dequantize_model


def pack_cols(values: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Pack per-column low-bit values into int32 words."""

    if values.dtype != torch.int32:
        values = values.to(torch.int32)

    rows, cols = values.shape
    pack_factor = 32 // bits
    if cols % pack_factor != 0:
        raise ValueError("columns must be divisible by pack factor")

    packed_cols = cols // pack_factor
    packed = torch.zeros(rows, packed_cols, dtype=torch.int32)
    mask = (1 << bits) - 1
    for col in range(cols):
        group = col // pack_factor
        shift = (col % pack_factor) * bits
        packed[:, group] |= (values[:, col] & mask) << shift
    return packed


def write_index(path: Path, shard: str, keys: list[str]) -> None:
    weight_map = dict.fromkeys(keys, shard)
    payload = {"weight_map": weight_map}
    (path / "model.safetensors.index.json").write_text(json.dumps(payload))


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_model_fp8_infers_block_size(tmp_path):
    model_dir = tmp_path / "fp8_model_infer"
    output_dir = tmp_path / "fp8_output_infer"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "fmt": "float8_e4m3fn",
            "quant_method": "fp8",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    weight = torch.randn(4, 8, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale_inv = torch.ones(2, 2, dtype=torch.float32)
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": weight,
            "linear.weight_scale_inv": scale_inv,
        },
        str(model_dir / shard_name),
    )
    write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv"])

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        weight_out = reader.get_tensor("linear.weight")
        assert weight_out.dtype is torch.bfloat16

    expected = dequantize_f8_e4m3(weight, scale_inv=scale_inv, axis=None, target_dtype=torch.bfloat16)
    assert torch.equal(weight_out, expected)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_model_fp8(tmp_path):
    model_dir = tmp_path / "fp8_model"
    output_dir = tmp_path / "fp8_output"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "fmt": "float8_e4m3fn",
            "quant_method": "fp8",
            "weight_block_size": [2, 4],
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    weight = torch.randn(2, 4, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale_inv = torch.ones(1, 1, dtype=torch.float32)
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": weight,
            "linear.weight_scale_inv": scale_inv,
            "linear.bias": torch.randn(4, dtype=torch.float32),
        },
        str(model_dir / shard_name),
    )
    write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv", "linear.bias"])

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        assert "linear.weight" in reader.keys()
        assert "linear.weight_scale_inv" not in reader.keys()
        weight_out = reader.get_tensor("linear.weight")
        bias_out = reader.get_tensor("linear.bias")

    expected = dequantize_f8_e4m3(weight, scale_inv=scale_inv, axis=None, target_dtype=torch.bfloat16)
    assert torch.equal(weight_out, expected)
    assert bias_out.dtype is torch.bfloat16

    updated_config = json.loads((output_dir / "config.json").read_text())
    assert "quantization_config" not in updated_config
    assert updated_config.get("torch_dtype") == "bfloat16"

    new_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert "linear.weight" in new_index["weight_map"]
    assert "linear.weight_scale_inv" not in new_index["weight_map"]


def test_dequantize_model_awq(tmp_path):
    model_dir = tmp_path / "awq_model"
    output_dir = tmp_path / "awq_output"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "quant_method": "awq",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    rows, cols = 8, 16
    weight_values = torch.randint(0, 16, (rows, cols), dtype=torch.int32)
    zero_values = torch.randint(0, 16, (rows, cols), dtype=torch.int32)
    scales = torch.rand(rows, cols, dtype=torch.float32) * 0.5 + 0.5
    bias = torch.randn(cols, dtype=torch.float32)

    packed_weight = pack_cols(weight_values)
    packed_zero = pack_cols(zero_values)

    shard_name = "awq.safetensors"
    save_file(
        {
            "layer.qweight": packed_weight,
            "layer.qzeros": packed_zero,
            "layer.scales": scales,
            "layer.bias": bias,
        },
        str(model_dir / shard_name),
    )
    write_index(
        model_dir,
        shard_name,
        ["layer.qweight", "layer.qzeros", "layer.scales", "layer.bias"],
    )

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        assert "layer.weight" in keys
        assert "layer.qweight" not in keys
        assert "layer.qzeros" not in keys
        weight_out = reader.get_tensor("layer.weight")
        bias_out = reader.get_tensor("layer.bias")

    expected = ((weight_values.float() - zero_values.float()) * scales).t().contiguous().to(torch.bfloat16)
    assert torch.equal(weight_out, expected)
    assert bias_out.dtype is torch.bfloat16

    updated_config = json.loads((output_dir / "config.json").read_text())
    assert "quantization_config" not in updated_config

    new_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert "layer.weight" in new_index["weight_map"]
    assert "layer.qweight" not in new_index["weight_map"]
