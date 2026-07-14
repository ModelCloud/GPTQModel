# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.quantization.dtype import dequantize_f4_e2m1
from gptqmodel.utils.model_dequant import (
    convert_awq_file,
    convert_bitsandbytes_shard,
    convert_compressed_pack_file,
    convert_gptq_file,
    convert_nvfp4_shard,
    dequantize_model,
    detect_format,
    finalize_for_save,
)


try:
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize
except Exception:
    nvfp4_quantize = None


def _write_index(model_dir, shard_name: str, keys: list[str]) -> None:
    weight_map = dict.fromkeys(keys, shard_name)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


def test_finalize_for_save_keeps_non_4d_tensors_contiguous():
    tensor = torch.randn(1, 2, 3, 4, 5)

    out = finalize_for_save(tensor, torch.bfloat16)

    assert out.dtype is torch.bfloat16
    assert out.device.type == "cpu"
    assert out.is_contiguous()


def test_ignored_layers_are_honored_by_non_fp8_converters(tmp_path):
    ignored_weight = torch.randn(2, 2, dtype=torch.bfloat16)
    shard_path = tmp_path / "ignored.safetensors"
    save_file(
        {
            "ignored.weight": ignored_weight,
            "ignored.weight_scale": torch.ones(1, dtype=torch.float32),
            "ignored.weight_scale_inv": torch.ones(1, dtype=torch.float32),
            "ignored.qweight": torch.ones(1, 1, dtype=torch.int32),
            "ignored.qzeros": torch.ones(1, 1, dtype=torch.int32),
            "ignored.scales": torch.ones(1, 1, dtype=torch.float32),
            "ignored.g_idx": torch.zeros(1, dtype=torch.int32),
            "ignored.weight_packed": torch.ones(1, 1, dtype=torch.int32),
            "ignored.weight_zero_point": torch.zeros(1, 1, dtype=torch.int32),
            "ignored.weight_g_idx": torch.zeros(1, dtype=torch.int32),
            "ignored.weight_shape": torch.tensor([2, 2], dtype=torch.int32),
        },
        str(shard_path),
    )

    with safe_open(shard_path, framework="pt", device="cpu") as reader:
        nvfp4_out = convert_nvfp4_shard(reader, torch.bfloat16, ignored_layers={"ignored"})
    with safe_open(shard_path, framework="pt", device="cpu") as reader:
        bnb_out = convert_bitsandbytes_shard(reader, torch.bfloat16, quant_cfg={}, ignored_layers={"ignored"})

    converter_outputs = [
        nvfp4_out,
        bnb_out,
        convert_awq_file(shard_path, torch.bfloat16, "cpu", ignored_layers={"ignored"}),
        convert_gptq_file(shard_path, torch.bfloat16, {}, "cpu", ignored_layers={"ignored"}),
        convert_compressed_pack_file(
            shard_path,
            torch.bfloat16,
            device="cpu",
            module_to_scheme={},
            compressor=object(),
            ignored_layers={"ignored"},
        ),
    ]

    for tensors in converter_outputs:
        assert set(tensors) == {"ignored.weight"}
        torch.testing.assert_close(tensors["ignored.weight"], ignored_weight)


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


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_model_fp8_crops_overpadded_block_scale_grid(tmp_path):
    model_dir = tmp_path / "fp8_overpadded_blocks"
    output_dir = tmp_path / "fp8_overpadded_blocks_out"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "format": "float8_e4m3fn",
            "quant_method": "fp8",
            "weight_block_size": [2, 2],
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    rows, cols = 5, 4
    block_rows, block_cols = 2, 2

    torch.manual_seed(0)
    weight = torch.randn(rows, cols, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale_inv = torch.linspace(2.0, 5.5, steps=8, dtype=torch.float32).view(4, 2)

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


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_model_fp8_honors_ignored_layers(tmp_path):
    model_dir = tmp_path / "fp8_ignored_layers"
    output_dir = tmp_path / "fp8_ignored_layers_out"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "format": "float8_e4m3fn",
            "quant_method": "fp8",
            "weight_block_size": [2, 2],
            "ignored_layers": ["ignored"],
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    torch.manual_seed(0)
    quant_weight = torch.randn(2, 2, dtype=torch.float32).to(torch.float8_e4m3fn)
    quant_scale_inv = torch.ones(1, 1, dtype=torch.float32) * 2
    ignored_weight = torch.randn(2, 2, dtype=torch.bfloat16)

    shard_name = "model.safetensors"
    save_file(
        {
            "quant.weight": quant_weight,
            "quant.weight_scale_inv": quant_scale_inv,
            "ignored.weight": ignored_weight,
            "ignored.weight_scale_inv": torch.ones(1, 1, dtype=torch.float32),
        },
        str(model_dir / shard_name),
    )
    _write_index(
        model_dir,
        shard_name,
        ["quant.weight", "quant.weight_scale_inv", "ignored.weight", "ignored.weight_scale_inv"],
    )

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        assert "quant.weight" in reader.keys()
        assert "ignored.weight" in reader.keys()
        assert "quant.weight_scale_inv" not in reader.keys()
        assert "ignored.weight_scale_inv" not in reader.keys()
        quant_out = reader.get_tensor("quant.weight")
        ignored_out = reader.get_tensor("ignored.weight")

    expected_quant = quant_weight.to(torch.bfloat16) / quant_scale_inv.to(torch.bfloat16)
    torch.testing.assert_close(quant_out, expected_quant)
    torch.testing.assert_close(ignored_out, ignored_weight)


def test_detect_format_modelopt_nvfp4_uses_quant_algo_config(tmp_path):
    model_dir = tmp_path / "modelopt_nvfp4_config_detect"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file({"dense.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, str(model_dir / "model.safetensors"))

    assert detect_format(model_dir, config) == "nvfp4"


@pytest.mark.skipif(nvfp4_quantize is None, reason="torchao NVFP4 support required")
def test_dequantize_model_modelopt_nvfp4_resolves_scales_from_other_shard(tmp_path):
    model_dir = tmp_path / "modelopt_nvfp4_cross_shard"
    output_dir = tmp_path / "modelopt_nvfp4_cross_shard_out"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    torch.manual_seed(0)
    dense = torch.randn(4, 16, dtype=torch.float32)
    scales, packed = nvfp4_quantize(dense, block_size=16)
    global_scale = torch.tensor(2.0, dtype=torch.float32)
    bias = torch.randn(4, dtype=torch.float32)

    weight_shard = "model-00001-of-00002.safetensors"
    scale_shard = "model-00002-of-00002.safetensors"
    save_file({"linear.weight": packed.cpu(), "linear.bias": bias}, str(model_dir / weight_shard))
    save_file(
        {
            "linear.weight_scale": scales.cpu(),
            "linear.weight_scale_2": global_scale,
        },
        str(model_dir / scale_shard),
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "linear.weight": weight_shard,
                    "linear.bias": weight_shard,
                    "linear.weight_scale": scale_shard,
                    "linear.weight_scale_2": scale_shard,
                }
            }
        ),
        encoding="utf-8",
    )

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / weight_shard, framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == {"linear.weight", "linear.bias"}
        weight_out = reader.get_tensor("linear.weight")
        bias_out = reader.get_tensor("linear.bias")

    expected_scale = scales.cpu().to(torch.float32) * global_scale
    expected = dequantize_f4_e2m1(
        packed.cpu(),
        scale=expected_scale,
        axis=None,
        target_dtype=torch.bfloat16,
    )
    assert weight_out.dtype is torch.bfloat16
    torch.testing.assert_close(weight_out, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(bias_out, bias.to(torch.bfloat16))
    assert not (output_dir / scale_shard).exists()

    output_index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert output_index["weight_map"] == {
        "linear.weight": weight_shard,
        "linear.bias": weight_shard,
    }


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="float8_e8m0fnu dtype not available",
)
def test_dequantize_model_fp8_dequantizes_deepseek_v4_packed_experts(tmp_path):
    model_dir = tmp_path / "deepseek_v4_fp4_experts"
    output_dir = tmp_path / "deepseek_v4_fp4_experts_out"
    model_dir.mkdir()

    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "expert_dtype": "fp4",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    # Logical FP4 codes 0..15 repeated once, packed as low/high nibbles.
    packed_bytes = torch.tensor(
        [[lo | (hi << 4) for lo, hi in zip(range(0, 16, 2), range(1, 16, 2))] * 2],
        dtype=torch.uint8,
    )
    weight = packed_bytes.view(torch.int8)
    scale = torch.tensor([[2.0]], dtype=torch.float32).to(torch.float8_e8m0fnu)

    weight_key = "layers.0.ffn.experts.0.w1.weight"
    scale_key = "layers.0.ffn.experts.0.w1.scale"
    shard_name = "model.safetensors"
    save_file({weight_key: weight, scale_key: scale}, str(model_dir / shard_name))
    _write_index(model_dir, shard_name, [weight_key, scale_key])

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == {weight_key}
        output = reader.get_tensor(weight_key)

    fp4_table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
    )
    expected = (fp4_table.repeat(2).view(1, 32) * 2.0).to(torch.bfloat16)
    assert output.dtype is torch.bfloat16
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="float8_e8m0fnu dtype not available",
)
def test_dequantize_model_fp8_does_not_treat_other_models_as_deepseek_v4(tmp_path):
    model_dir = tmp_path / "non_deepseek_v4_fp8"
    output_dir = tmp_path / "non_deepseek_v4_fp8_out"
    model_dir.mkdir()

    config = {
        "architectures": ["TestModel"],
        "model_type": "not_deepseek_v4",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    weight_key = "layers.0.ffn.experts.0.w1.weight"
    scale_key = "layers.0.ffn.experts.0.w1.scale"
    weight = torch.zeros((1, 16), dtype=torch.int8)
    scale = torch.ones((1, 1), dtype=torch.float32).to(torch.float8_e8m0fnu)
    shard_name = "model.safetensors"
    save_file({weight_key: weight, scale_key: scale}, str(model_dir / shard_name))
    _write_index(model_dir, shard_name, [weight_key, scale_key])

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / shard_name, framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == {weight_key, scale_key}
        assert reader.get_tensor(weight_key).dtype is torch.int8
        assert reader.get_tensor(scale_key).dtype is torch.bfloat16
