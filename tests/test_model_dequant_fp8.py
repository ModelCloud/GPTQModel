# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import math

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.quantization.dtype import decode_e8m0_scale, dequantize_f4_e2m1
from gptqmodel.utils.model_dequant import (
    convert_awq_file,
    convert_bitsandbytes_shard,
    convert_compressed_pack_file,
    convert_gptq_file,
    convert_nvfp4_shard,
    convert_mx_shard,
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


@pytest.mark.parametrize(
    "storage_dtype",
    [torch.uint8, *([torch.float8_e8m0fnu] if hasattr(torch, "float8_e8m0fnu") else [])],
)
def test_e8m0_scale_matches_independent_oracle_for_every_encoding(storage_dtype):
    encoded = torch.arange(256, dtype=torch.uint8)
    stored = encoded if storage_dtype == torch.uint8 else encoded.view(storage_dtype)

    decoded = decode_e8m0_scale(stored)

    oracle = torch.tensor([math.ldexp(1.0, code - 127) for code in range(255)], dtype=torch.float32)
    torch.testing.assert_close(decoded[:255], oracle, rtol=0, atol=0)
    assert torch.isnan(decoded[255])


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mxfp8_dequantizes_e8m0_blocks(fp8_dtype, tmp_path):
    weight = torch.ones((2, 64), dtype=torch.float32).to(fp8_dtype)
    weight[1] = 2
    scale = torch.tensor([[127, 128], [129, 127]], dtype=torch.uint8)
    path = tmp_path / "mx.safetensors"
    save_file({"linear.weight": weight, "linear.weight_scale": scale}, str(path))

    with safe_open(path, framework="pt", device="cpu") as reader:
        result = convert_mx_shard(reader, torch.bfloat16)

    expected = torch.tensor([[1.0] * 32 + [2.0] * 32, [8.0] * 32 + [2.0] * 32], dtype=torch.bfloat16)
    assert set(result) == {"linear.weight"}
    torch.testing.assert_close(result["linear.weight"], expected)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mxfp8_float32_matches_torch_oracle_across_blocks(fp8_dtype, tmp_path):
    unscaled = torch.tensor([-6.0, -1.5, -0.5, 0.0, 0.5, 1.0, 3.0, 6.0] * 8)
    weight = unscaled.repeat(2, 1).to(fp8_dtype)
    encoded_scale = torch.tensor([[117, 137], [129, 126]], dtype=torch.uint8)
    global_scale = torch.tensor(0.3, dtype=torch.float32)
    path = tmp_path / "fp8_oracle.safetensors"
    save_file({
        "linear.weight": weight,
        "linear.weight_scale": encoded_scale,
        "linear.weight_scale_2": global_scale,
    }, str(path))

    with safe_open(path, framework="pt", device="cpu") as reader:
        actual = convert_mx_shard(reader, torch.float32)["linear.weight"]

    scale = torch.tensor([
        [math.ldexp(1.0, -10), math.ldexp(1.0, 10)],
        [math.ldexp(1.0, 2), math.ldexp(1.0, -1)],
    ], dtype=torch.float64).repeat_interleave(32, dim=1)
    oracle = (weight.to(torch.float64) * scale * global_scale.to(torch.float64)).to(torch.float32)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, oracle, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mxfp8_every_code_matches_torch_oracle(fp8_dtype, tmp_path):
    weight = torch.arange(256, dtype=torch.uint8).view(fp8_dtype).reshape(4, 64)
    encoded_scale = torch.tensor([
        [117, 127], [128, 137], [126, 129], [127, 117],
    ], dtype=torch.uint8)
    path = tmp_path / "fp8_all_codes.safetensors"
    save_file({"linear.weight": weight, "linear.weight_scale": encoded_scale}, str(path))

    with safe_open(path, framework="pt", device="cpu") as reader:
        actual = convert_mx_shard(reader, torch.float32)["linear.weight"]

    scale = torch.tensor([
        [math.ldexp(1.0, -10), 1.0], [2.0, math.ldexp(1.0, 10)],
        [0.5, 4.0], [1.0, math.ldexp(1.0, -10)],
    ], dtype=torch.float64).repeat_interleave(32, dim=1)
    oracle = (weight.to(torch.float64) * scale).to(torch.float32)
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mxfp8_clips_padded_primary_and_secondary_scale_grids(fp8_dtype, tmp_path):
    weight = torch.tensor([-1.5, 0.5, 2.0, 4.0] * 16, dtype=torch.float32).repeat(2, 1).to(fp8_dtype)
    scale = torch.full((4, 4), 255, dtype=torch.uint8)
    scale[:2, :2] = torch.tensor([[127, 128], [129, 126]], dtype=torch.uint8)
    scale_2 = torch.full((4, 4), 99.0)
    scale_2[:2, :2] = torch.tensor([[0.3, 1.25], [0.5, 0.7]])
    path = tmp_path / "fp8_padded.safetensors"
    save_file({
        "linear.weight": weight,
        "linear.weight_scale": scale,
        "linear.weight_scale_2": scale_2,
    }, str(path))

    with safe_open(path, framework="pt", device="cpu") as reader:
        actual = convert_mx_shard(reader, torch.float32)["linear.weight"]

    primary = torch.tensor([[1.0, 2.0], [4.0, 0.5]], dtype=torch.float64)
    factors = (primary * scale_2[:2, :2].to(torch.float64)).repeat_interleave(32, dim=1)
    oracle = (weight.to(torch.float64) * factors).to(torch.float32)
    torch.testing.assert_close(actual, oracle, rtol=1e-6, atol=1e-6)


def test_mxfp4_float32_matches_independent_codebook_oracle(tmp_path):
    codebook = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
    packed = torch.tensor(
        [code | ((15 - code) << 4) for code in range(16)] * 2,
        dtype=torch.uint8,
    ).repeat(2, 1)
    encoded_scale = torch.tensor([[117, 137, 255], [129, 126, 255]], dtype=torch.uint8)
    global_scale = torch.tensor(0.7, dtype=torch.float32)
    path = tmp_path / "fp4_oracle.safetensors"
    save_file({
        "linear.weight": packed,
        "linear.weight_scale": encoded_scale,
        "linear.weight_scale_2": global_scale,
    }, str(path))

    with safe_open(path, framework="pt", device="cpu") as reader:
        actual = convert_mx_shard(reader, torch.float32)["linear.weight"]

    unscaled = torch.tensor(
        [value for code in range(16) for value in (codebook[code], codebook[15 - code])] * 2,
        dtype=torch.float64,
    ).repeat(2, 1)
    scale = torch.tensor([
        [math.ldexp(1.0, -10), math.ldexp(1.0, 10)],
        [math.ldexp(1.0, 2), math.ldexp(1.0, -1)],
    ], dtype=torch.float64).repeat_interleave(32, dim=1)
    oracle = (unscaled * scale * global_scale.to(torch.float64)).to(torch.float32)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, oracle, rtol=1e-6, atol=1e-6)


def test_mxfp4_nan_scale_marks_entire_block_nan(tmp_path):
    packed = torch.zeros((1, 16), dtype=torch.uint8)
    path = tmp_path / "nan_scale.safetensors"
    save_file({
        "linear.weight": packed,
        "linear.weight_scale": torch.tensor([[255]], dtype=torch.uint8),
    }, str(path))

    with safe_open(path, framework="pt", device="cpu") as reader:
        actual = convert_mx_shard(reader, torch.float32)["linear.weight"]

    assert torch.isnan(actual).all()


def test_mxfp4_dequantizes_packed_nibbles_and_expert_dimension(tmp_path):
    packed = torch.full((2, 2, 16), 0x21, dtype=torch.uint8)
    scale = torch.tensor([[[127], [128]], [[129], [126]]], dtype=torch.uint8)
    path = tmp_path / "mx_experts.safetensors"
    save_file({"experts.weight": packed, "experts.weight_scale": scale}, str(path))

    with safe_open(path, framework="pt", device="cpu") as reader:
        result = convert_mx_shard(reader, torch.bfloat16)

    expected = torch.tensor([0.5, 1.0] * 16, dtype=torch.bfloat16)
    torch.testing.assert_close(result["experts.weight"][0, 0], expected)
    torch.testing.assert_close(result["experts.weight"][0, 1], expected * 2)
    torch.testing.assert_close(result["experts.weight"][1, 0], expected * 4)
    torch.testing.assert_close(result["experts.weight"][1, 1], expected * 0.5)


@pytest.mark.parametrize("secondary_layout", ["padded", "global"])
def test_mxfp4_clips_padded_expert_scale_grid(tmp_path, secondary_layout):
    packed = torch.full((2, 2, 16), 0x21, dtype=torch.uint8)
    scale = torch.full((3, 3, 2), 255, dtype=torch.uint8)
    scale[:2, :2, 0] = torch.tensor([[127, 128], [129, 126]], dtype=torch.uint8)
    if secondary_layout == "padded":
        scale_2 = torch.full((3, 3, 2), 99.0)
        scale_2[:2, :2, 0] = torch.tensor([[1.0, 0.5], [1.0, 2.0]])
        factors = ((1.0, 1.0), (4.0, 1.0))
    else:
        scale_2 = torch.tensor([[[2.0]]])
        factors = ((2.0, 4.0), (8.0, 1.0))
    path = tmp_path / "padded_experts.safetensors"
    save_file({
        "experts.weight": packed,
        "experts.weight_scale": scale,
        "experts.weight_scale_2": scale_2,
    }, str(path))

    with safe_open(path, framework="pt", device="cpu") as reader:
        result = convert_mx_shard(reader, torch.bfloat16)

    # Independent E2M1 oracle for packed 0x21 (low nibble first), then E8M0 and global scales.
    unscaled = torch.tensor([0.5, 1.0] * 16, dtype=torch.float32)
    expected = torch.stack([torch.stack([unscaled * factor for factor in row]) for row in factors])
    torch.testing.assert_close(result["experts.weight"], expected.to(torch.bfloat16), rtol=0, atol=0)


def test_mxfp4_accepts_quark_sibling_scale_convention(tmp_path):
    path = tmp_path / "mx_sibling.safetensors"
    save_file({
        "expert.weight": torch.full((1, 16), 0x21, dtype=torch.int8),
        "expert.scale": torch.tensor([[128]], dtype=torch.uint8),
    }, str(path))

    with safe_open(path, framework="pt", device="cpu") as reader:
        result = convert_mx_shard(reader, torch.bfloat16)

    assert set(result) == {"expert.weight"}
    torch.testing.assert_close(result["expert.weight"][0], torch.tensor([1.0, 2.0] * 16, dtype=torch.bfloat16))


def test_auto_dequantizes_mixed_mx_with_cross_shard_scales(tmp_path):
    model_dir = tmp_path / "mixed_mx"
    output_dir = tmp_path / "dense"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["TestModel"],
        "quantization_config": {"format": "mixed-mxfp4-mxfp8"},
    }), encoding="utf-8")
    weight_shard = "model-00001-of-00002.safetensors"
    scale_shard = "model-00002-of-00002.safetensors"
    save_file({
        "fp4.weight": torch.full((1, 16), 0x21, dtype=torch.uint8),
        "fp8.weight": torch.ones((1, 32), dtype=torch.float32).to(torch.float8_e4m3fn),
    }, str(model_dir / weight_shard))
    save_file({
        "fp4.weight_scale": torch.tensor([[128]], dtype=torch.uint8),
        "fp8.weight_scale": torch.tensor([[129]], dtype=torch.uint8),
    }, str(model_dir / scale_shard))
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "fp4.weight": weight_shard, "fp8.weight": weight_shard,
        "fp4.weight_scale": scale_shard, "fp8.weight_scale": scale_shard,
    }}), encoding="utf-8")

    assert detect_format(model_dir, json.loads((model_dir / "config.json").read_text())) == "mixed-mx"
    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / weight_shard, framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == {"fp4.weight", "fp8.weight"}
        torch.testing.assert_close(
            reader.get_tensor("fp4.weight")[0],
            torch.tensor([1.0, 2.0] * 16, dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            reader.get_tensor("fp8.weight"),
            torch.full((1, 32), 4.0, dtype=torch.bfloat16),
        )
    assert not (output_dir / scale_shard).exists()


@pytest.mark.parametrize("format_label,scale,expected_value", [
    ("fp8", torch.tensor(0.5, dtype=torch.float32), 1.0),
    ("nvfp8", torch.tensor([[128]], dtype=torch.uint8), 4.0),
])
def test_auto_dequantizes_direct_scale_fp8_labels(tmp_path, format_label, scale, expected_value):
    model_dir = tmp_path / format_label
    output_dir = tmp_path / f"{format_label}_dense"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["TestModel"],
        "quantization_config": {"format": format_label},
    }), encoding="utf-8")
    weight = torch.full((1, 32), 2.0, dtype=torch.float32).to(torch.float8_e5m2)
    save_file({"linear.weight": weight, "linear.weight_scale": scale}, str(model_dir / "model.safetensors"))

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")

    with safe_open(output_dir / "model.safetensors", framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == {"linear.weight"}
        torch.testing.assert_close(
            reader.get_tensor("linear.weight"),
            torch.full((1, 32), expected_value, dtype=torch.bfloat16),
        )


def test_quark_metadata_auto_detects_mxfp4_with_plain_fp8_override(tmp_path):
    model_dir = tmp_path / "quark_mixed"
    output_dir = tmp_path / "quark_dense"
    model_dir.mkdir()
    config = {"quantization_config": {
        "quant_method": "quark_online",
        "global_quant_config": {"weight": {
            "dtype": "fp4", "group_size": 32, "scale_format": "e8m0",
        }},
        "layer_quant_config": {"*attn*": {"weight": {
            "dtype": "fp8_e4m3", "qscheme": "per_channel", "scale_format": "float",
        }}},
    }}
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file({
        "mlp.weight": torch.full((1, 16), 0x21, dtype=torch.uint8),
        "mlp.weight_scale": torch.tensor([[128]], dtype=torch.uint8),
        "attn.weight": torch.full((1, 32), 2.0, dtype=torch.float32).to(torch.float8_e4m3fn),
        "attn.weight_scale": torch.tensor([0.5], dtype=torch.float32),
    }, str(model_dir / "model.safetensors"))

    assert detect_format(model_dir, config) == "mxfp4"
    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")
    with safe_open(output_dir / "model.safetensors", framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == {"mlp.weight", "attn.weight"}
        torch.testing.assert_close(
            reader.get_tensor("mlp.weight")[0],
            torch.tensor([1.0, 2.0] * 16, dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            reader.get_tensor("attn.weight"),
            torch.ones((1, 32), dtype=torch.bfloat16),
        )


def test_auto_detects_e8m0_scale_before_weight_shard_without_config(tmp_path):
    model_dir = tmp_path / "scale_first"
    output_dir = tmp_path / "scale_first_dense"
    model_dir.mkdir()
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    save_file({"linear.weight_scale": torch.tensor([[128]], dtype=torch.uint8)}, str(model_dir / first))
    save_file({"linear.weight": torch.full((1, 16), 0x21, dtype=torch.uint8)}, str(model_dir / second))
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "linear.weight_scale": first, "linear.weight": second,
    }}), encoding="utf-8")

    assert detect_format(model_dir, {}) == "mxfp4"
    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")
    assert not (output_dir / first).exists()
    with safe_open(output_dir / second, framework="pt", device="cpu") as reader:
        torch.testing.assert_close(
            reader.get_tensor("linear.weight")[0],
            torch.tensor([1.0, 2.0] * 16, dtype=torch.bfloat16),
        )


def test_finalize_for_save_keeps_non_4d_tensors_contiguous():
    tensor = torch.randn(1, 2, 3, 4, 5)

    out = finalize_for_save(tensor, torch.bfloat16)

    assert out.dtype is torch.bfloat16
    assert out.device.type == "cpu"
    assert out.is_contiguous()


def test_finalize_for_save_converts_channels_last_to_default_contiguous():
    tensor = torch.randn(2, 3, 4, 5).contiguous(memory_format=torch.channels_last)
    assert not tensor.is_contiguous()

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
    aux_dir = model_dir / "aux"
    aux_dir.mkdir()
    (aux_dir / "metadata.json").write_text("{}", encoding="utf-8")

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16, device="cpu")
    dequantize_model(
        model_dir,
        output_dir,
        target_dtype=torch.bfloat16,
        device="cpu",
        resume=True,
    )
    assert (output_dir / "aux" / "metadata.json").read_text(encoding="utf-8") == "{}"

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
