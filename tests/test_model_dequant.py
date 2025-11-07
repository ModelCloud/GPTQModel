# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def test_dequantize_model_compressed_tensors_pack(tmp_path):
    pytest.importorskip("compressed_tensors")
    pytest.importorskip("transformers")

    from compressed_tensors.compressors.quantized_compressors.pack_quantized import pack_to_int32
    from compressed_tensors.quantization.quant_args import QuantizationArgs
    from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
    from compressed_tensors.quantization.utils import calculate_qparams
    from transformers import LlamaConfig

    model_dir = tmp_path / "compressed_model"
    output_dir = tmp_path / "compressed_output"
    model_dir.mkdir()

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
    )

    quant_cfg = {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": 32,
                    "num_bits": 4,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }

    config_dict = config.to_dict()
    config_dict["quantization_config"] = quant_cfg
    (model_dir / "config.json").write_text(json.dumps(config_dict))

    weight_cfg = quant_cfg["config_groups"]["group_0"]["weights"]
    quant_args = QuantizationArgs(**weight_cfg)

    def compress_weight(prefix: str, weight: torch.Tensor) -> tuple[dict, torch.Tensor]:
        rows, cols = weight.shape
        group_size = quant_args.group_size or cols
        groups = cols // group_size
        reshaped = weight.view(rows, groups, group_size)
        min_vals = reshaped.amin(dim=-1)
        max_vals = reshaped.amax(dim=-1)
        scales, zero_points = calculate_qparams(min_vals, max_vals, quant_args)

        quantized = quantize(
            weight,
            scale=scales,
            zero_point=zero_points,
            args=quant_args,
            dtype=torch.int8,
        )
        packed = pack_to_int32(quantized, quant_args.num_bits)
        expected = dequantize(
            quantized,
            scale=scales,
            zero_point=zero_points,
            args=quant_args,
            dtype=torch.float32,
        )
        payload = {
            f"{prefix}.weight_packed": packed,
            f"{prefix}.weight_scale": scales,
            f"{prefix}.weight_shape": torch.tensor(weight.shape, dtype=torch.int32),
        }
        return payload, expected

    prefix_q = "model.layers.0.self_attn.q_proj"
    prefix_k = "model.layers.0.self_attn.k_proj"

    hidden = config.hidden_size
    base_weight = torch.linspace(-0.75, 0.75, steps=hidden * hidden, dtype=torch.float32).view(
        hidden, hidden
    )
    payload_q, expected_q = compress_weight(prefix_q, base_weight)
    payload_k, expected_k = compress_weight(prefix_k, base_weight.neg())

    shard_name = "model.safetensors"
    tensors = {**payload_q, **payload_k}
    save_file(tensors, str(model_dir / shard_name))
    write_index(model_dir, shard_name, list(tensors.keys()))

    from gptqmodel.utils.model_dequant import detect_format, load_json
    import gptqmodel.utils.model_dequant as model_dequant_module

    module_path = Path(model_dequant_module.__file__).resolve()
    assert REPO_ROOT in module_path.parents
    detected = detect_format(model_dir, load_json(model_dir / "config.json"))
    assert detected == "compressed-pack"

    dequantize_model(model_dir, output_dir, target_dtype=torch.float32, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        assert f"{prefix_q}.weight" in keys
        assert f"{prefix_k}.weight" in keys
        assert all(not key.endswith(("_packed", "_scale", "_shape")) for key in keys)

        weight_q = reader.get_tensor(f"{prefix_q}.weight")
        weight_k = reader.get_tensor(f"{prefix_k}.weight")

        torch.testing.assert_close(weight_q, expected_q)
        torch.testing.assert_close(weight_k, expected_k)

    updated_config = json.loads((output_dir / "config.json").read_text())
    assert "quantization_config" not in updated_config

    new_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert f"{prefix_q}.weight" in new_index["weight_map"]
    assert f"{prefix_k}.weight" in new_index["weight_map"]
    assert f"{prefix_q}.weight_scale" not in new_index["weight_map"]
