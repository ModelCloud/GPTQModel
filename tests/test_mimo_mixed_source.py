# SPDX-License-Identifier: Apache-2.0
# GPU=-1

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

import gptqmodel.models.loader as loader_module
from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.mimo_v2 import MimoV2QModel, _compatible_mask
from gptqmodel.models.writer import (
    _merge_prefix_tensors_into_state_dict,
    _normalize_out_of_model_tensors_entries,
)
from gptqmodel.quantization.config import (
    AutoModuleDecoderConfig,
    AWQConfig,
    QuantizeConfig,
)
from gptqmodel.utils.mimo import decode_mimo_weight, decode_mxfp4
from gptqmodel.utils.model_dequant import (
    convert_fp8_shard,
    dequantize_model,
    detect_format,
)
from gptqmodel.utils.structure import LazyTurtle


def source_config(tp: int = 4) -> dict:
    return {
        "model_type": "mimo_v2",
        "attention_projection_layout": "fused_qkv",
        "num_key_value_heads": tp,
        "num_attention_heads": 2 * tp,
        "head_dim": 6,
        "v_head_dim": 4,
        "swa_num_key_value_heads": tp,
        "swa_num_attention_heads": 2 * tp,
        "swa_head_dim": 6,
        "swa_v_head_dim": 4,
        "hybrid_layer_pattern": [0, 1],
        "quantization_config": {
            "quant_method": "fp8",
            "store_dtype": "mxfp4",
            "mxfp4_block_size": 32,
            "weight_block_size": [8, 32],
        },
    }


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mimo_loader_uses_dense_decoding_on_native_capable_device(
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(**source_config())
    quantize_config = QuantizeConfig(bits=4, offload_to_disk=False)
    monkeypatch.setattr(
        loader_module,
        "device_supports_dtype",
        lambda device, value, **k: value == dtype,
    )
    monkeypatch.setattr(
        loader_module, "device_supports_native_fp4", lambda *a, **k: True
    )
    assert loader_module.native_floatx_source_format(config) == "mimo_mixed"
    result = loader_module.configure_native_floatx_source_quantization(
        config, quantize_config, device=torch.device("cuda:0"), model_dtype=dtype
    )
    assert result == "mimo_mixed"
    assert quantize_config.offload_to_disk is True
    plan = quantize_config._native_floatx_forward_plan
    assert plan["mode"] == "decode"
    assert plan["native_validated"] is False
    assert plan["hessian_accumulation_dtype"] == "float32"
    decoders = [
        item
        for item in quantize_config.preprocessors
        if isinstance(item, AutoModuleDecoderConfig)
    ]
    assert len(decoders) == 1
    assert decoders[0].target_dtype is dtype


@pytest.mark.parametrize("source_format", ["fp8", "nvfp4"])
@pytest.mark.parametrize("config_class", [QuantizeConfig, AWQConfig])
def test_non_mimo_decoder_default_remains_bf16(
    source_format: str,
    config_class: type[QuantizeConfig] | type[AWQConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(quantization_config={"quant_algo": source_format})
    quantize_config = config_class(bits=4)
    monkeypatch.setattr(loader_module, "device_supports_dtype", lambda *a, **k: True)
    monkeypatch.setattr(
        loader_module, "device_supports_native_fp4", lambda *a, **k: False
    )
    loader_module.configure_native_floatx_source_quantization(
        config, quantize_config, device=torch.device("cpu"), model_dtype=torch.float16
    )
    decoders = [
        item
        for item in quantize_config.preprocessors
        if isinstance(item, AutoModuleDecoderConfig)
    ]
    assert len(decoders) == 1
    assert decoders[0].target_dtype == torch.bfloat16


def qkv_fixture(tp: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Each TP shard is [12 Q, 6 K, 4 V] rows with independent padding.
    weights = torch.ones(tp * 22, 64).to(torch.float8_e4m3fn)
    scales = torch.arange(1, tp * 6 + 1, dtype=torch.float32).reshape(tp * 3, 2)
    parts: list[list[torch.Tensor]] = [[], [], []]
    for rank in range(tp):
        shard = torch.tensor(
            [
                [scales[rank * 3 + row // 8, col // 32] for col in range(64)]
                for row in range(22)
            ]
        )
        for destination, part in zip(parts, shard.split((12, 6, 4))):
            destination.append(part)
    return weights, scales, torch.cat([torch.cat(part) for part in parts])


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("layer", [0, 1])
def test_qkv_shard_scales_and_order(tp: int, layer: int) -> None:
    weight, scales, expected = qkv_fixture(tp)
    key = f"model.layers.{layer}.self_attn.qkv_proj.weight"
    decoded = decode_mimo_weight(
        source_config(tp),
        key,
        weight,
        {key + "_scale_inv": scales}.get,
        target_dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [8, 255, 256, 257])
def test_all_nibbles_and_exponent_scales(dtype: torch.dtype, rows: int) -> None:
    packed = torch.arange(rows * 32).remainder(256).to(torch.uint8).reshape(rows, 32)
    scales = torch.arange(rows * 2).remainder(16).add(120).to(torch.uint8)
    scales = scales.reshape(rows, 2)
    actual = decode_mxfp4(packed, scales, target_dtype=dtype)
    expected = torch.empty(rows, 64, dtype=dtype)
    for row in range(rows):
        for col in range(64):
            nibble = (int(packed[row, col // 2]) >> (4 * (col % 2))) & 15
            exponent, mantissa = (nibble & 7) >> 1, nibble & 1
            value = (
                mantissa / 2
                if exponent == 0
                else (1 + mantissa / 2) * 2 ** (exponent - 1)
            )
            value *= -1 if nibble & 8 else 1
            value *= 2 ** (int(scales[row, col // 32]) - 127)
            expected[row, col] = value
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale_byte", [0, 254])
def test_mxfp4_extreme_scales_and_noncontiguous_storage(
    dtype: torch.dtype, scale_byte: int
) -> None:
    packed = torch.tensor([0x80, 0x21, 0xA9], dtype=torch.uint8).repeat(32, 86)
    packed = packed[:, :257].T
    assert not packed.is_contiguous()
    scales = torch.full((2, 257), scale_byte, dtype=torch.uint8).T
    decoded = decode_mxfp4(packed, scales, target_dtype=dtype)
    values = ([0.0, -0.0], [0.5, 1.0], [-0.5, -1.0])
    expected = torch.tensor(
        [
            [value * 2.0 ** (scale_byte - 127) for value in values[row % 3]] * 32
            for row in range(257)
        ],
        dtype=dtype,
    )
    assert torch.equal(decoded.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mxfp4_rejects_output_overflow(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="overflow"):
        decode_mxfp4(
            torch.full((1, 16), 0x77, dtype=torch.uint8),
            torch.full((1, 1), 254, dtype=torch.uint8),
            target_dtype=dtype,
        )


@pytest.mark.parametrize("error", ["missing", "shape", "reserved", "block"])
def test_reject_bad_expert_source(error: str) -> None:
    config = source_config()
    name = "model.layers.1.mlp.experts.0.down_proj.weight"
    weight = torch.zeros(2, 32, dtype=torch.uint8)
    scale = torch.full((2, 2), 127, dtype=torch.uint8)
    if error == "missing":
        scale = None
    elif error == "shape":
        scale = scale[:, :1]
    elif error == "reserved":
        scale[0, 0] = 255
    else:
        config["quantization_config"]["mxfp4_block_size"] = 16
    with pytest.raises(ValueError):
        decode_mimo_weight(
            config, name, weight, lambda _: scale, target_dtype=torch.bfloat16
        )


@pytest.mark.parametrize("method", ["gptq", "nvfp4"])
def test_other_formats_are_not_mimo_source(method: str) -> None:
    config = source_config()
    config["quantization_config"]["quant_method"] = method
    assert (
        decode_mimo_weight(
            config,
            "model.layers.1.mlp.experts.0.up_proj.weight",
            torch.zeros(2, 32, dtype=torch.uint8),
            lambda _: None,
            target_dtype=torch.bfloat16,
        )
        is None
    )


@pytest.mark.parametrize(
    "name,dtype",
    [
        ("self_attn.qkv_proj", torch.float8_e4m3fnuz),
        ("mlp.experts.0.up_proj", torch.int8),
    ],
)
def test_unsupported_source_storage_fails_closed(name: str, dtype: torch.dtype) -> None:
    name = "model.layers.0." + name
    weight = torch.zeros(8, 32).to(dtype)
    with pytest.raises(ValueError, match="Unsupported MiMo source storage"):
        decode_mimo_weight(
            source_config(),
            name + ".weight",
            weight,
            {}.get,
            target_dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="Unsupported MiMo source storage"):
        SourceHarness(source_config())._decoder_weight_format(
            weight=weight, checkpoint_tensors={"weight": weight}, module_name=name
        )
    tensors = {name + ".weight": weight}
    reader = SimpleNamespace(keys=tensors.keys, get_tensor=tensors.__getitem__)
    with pytest.raises(ValueError, match="Unsupported MiMo source storage"):
        convert_fp8_shard(
            reader, torch.bfloat16, block_shape=None, source_config=source_config()
        )


def test_gptq_with_preserved_mtp_offline_dequantization(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    name = "model.layers.0.self_attn.o_proj"
    auxiliary = {
        "model.mtp.layers.0.proj.weight": torch.ones(8, 8).to(torch.float8_e4m3fn),
        "model.mtp.layers.0.proj.weight_scale_inv": torch.full((1, 1), 0.25),
    }
    save_file(
        {
            name + ".qweight": torch.full((2, 16), -1717986919, dtype=torch.int32),
            name + ".qzeros": torch.full((1, 2), 0x77777777, dtype=torch.int32),
            name + ".scales": torch.ones(1, 16).bfloat16(),
            name + ".g_idx": torch.zeros(16, dtype=torch.int32),
            **auxiliary,
        },
        source / "model.safetensors",
    )
    config = {
        "model_type": "mimo_v2",
        "quantization_config": {"quant_method": "gptq", "bits": 4},
        "mimo_mtp_source_quantization_config": source_config()["quantization_config"],
    }
    (source / "config.json").write_text(json.dumps(config))
    assert detect_format(source, config) == "gptq"
    dequantize_model(source, output)
    with safe_open(output / "model.safetensors", framework="pt") as reader:
        assert name + ".qweight" not in reader.keys()
        torch.testing.assert_close(
            reader.get_tensor(name + ".weight"),
            torch.ones(16, 16).bfloat16(),
            atol=0,
            rtol=0,
        )
        for key, value in auxiliary.items():
            assert reader.get_tensor(key).dtype == value.dtype
            assert torch.equal(
                reader.get_tensor(key).view(torch.uint8), value.view(torch.uint8)
            )


class SourceHarness:
    _decoder_source_config = BaseQModel._decoder_source_config
    _decoder_weight_format = BaseQModel._decoder_weight_format
    _build_decoder_quant_source_module = BaseQModel._build_decoder_quant_source_module

    def __init__(self, config: dict) -> None:
        self.model = SimpleNamespace(config=SimpleNamespace(**config))


def test_input_capture_recognizes_expert_only_mixed_source(tmp_path: Path) -> None:
    name = "model.layers.0.mlp.experts.0.gate_proj"
    packed = torch.full((2, 32), 0x62, dtype=torch.uint8)
    save_file(
        {
            name + ".weight": packed,
            name + ".weight_scale": torch.full((2, 2), 127, dtype=torch.uint8),
        },
        tmp_path / "model.safetensors",
    )
    shell = nn.Module()
    shell.config = SimpleNamespace(**source_config())
    shell.model = nn.Module()
    shell.model.layers = nn.ModuleList([nn.Module()])
    layer = shell.model.layers[0]
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.ModuleList([nn.Module()])
    projection = nn.Linear(64, 2, bias=False, device="meta", dtype=torch.bfloat16)
    layer.mlp.experts[0].gate_proj = projection
    harness = SourceHarness(source_config())
    harness.model = shell
    harness.turtle_model = LazyTurtle.maybe_create(
        model_local_path=str(tmp_path),
        config=shell.config,
        model_init_kwargs={"device_map": {"": "cpu"}},
        target_model=shell,
    )
    harness._active_auto_module_decoder_config = AutoModuleDecoderConfig
    classify = harness._decoder_weight_format
    classified_names = []

    def record_classify(**kwargs: object) -> str | None:
        classified_names.append(kwargs.get("module_name"))
        return classify(**kwargs)

    harness._decoder_weight_format = record_classify
    capture = StageInputsCapture(SimpleNamespace(gptq_model=harness))
    assert capture._first_layer_has_deferred_floatx_source(
        layer, module_path="model.layers.0"
    )
    assert classified_names == [name]
    assert projection.weight.is_meta


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn]
)
@pytest.mark.parametrize("materialization", ["submodule", "direct"])
def test_module_decoder_and_lazy_materialization_agree(
    tmp_path: Path, dtype: torch.dtype, materialization: str
) -> None:
    name = "model.layers.0.self_attn.qkv_proj"
    weight, scales, expected = qkv_fixture(4)
    tensors = {name + ".weight": weight, name + ".weight_scale_inv": scales}
    save_file({name + ".weight": weight}, tmp_path / "weights.safetensors")
    save_file({name + ".weight_scale_inv": scales}, tmp_path / "scales.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    key: "scales.safetensors"
                    if "scale" in key
                    else "weights.safetensors"
                    for key in tensors
                },
            }
        )
    )
    shell = nn.Module()
    shell.config = SimpleNamespace(**source_config())
    shell.model = nn.Module()
    shell.model.layers = nn.ModuleList([nn.Module()])
    layer = shell.model.layers[0]
    layer.self_attn = nn.Module()
    layer.self_attn.qkv_proj = nn.Linear(64, 88, bias=False, device="meta", dtype=dtype)
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(tmp_path),
        config=shell.config,
        model_init_kwargs={"device_map": {"": "cpu"}},
        target_model=shell,
    )

    def materialize() -> None:
        if materialization == "direct":
            turtle.sync_all_meta(shell_model=shell)
        else:
            turtle.materialize_submodule(
                target_model=shell, target_submodule=layer, device=torch.device("cpu")
            )

    if dtype == torch.float8_e4m3fn:
        with pytest.raises(
            ValueError, match="MiMo decoding requires a floating-point target dtype"
        ):
            materialize()
        assert layer.self_attn.qkv_proj.weight.is_meta
        return
    materialize()
    torch.testing.assert_close(
        layer.self_attn.qkv_proj.weight, expected.to(dtype), atol=0, rtol=0
    )
    harness = SourceHarness(source_config())
    harness.turtle_model = turtle
    # The writer replaces the runtime config before gathering passthrough weights.
    harness.model.config = SimpleNamespace(
        model_type="mimo_v2", quantization_config={"quant_method": "gptq"}
    )
    built = harness._build_decoder_quant_source_module(
        layer.self_attn.qkv_proj,
        checkpoint_tensors={"weight": weight, "weight_scale_inv": scales},
        target_dtype=dtype,
        module_name=name,
    )
    torch.testing.assert_close(built.weight, expected.to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float32])
def test_offline_decoder_preserves_mtp_and_non_targets(
    tmp_path: Path, target_dtype: torch.dtype
) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    packed = torch.full((2, 32), 0x62, dtype=torch.uint8)
    scale = torch.full((2, 2), 127, dtype=torch.uint8)
    qkv_name = "model.layers.0.self_attn.qkv_proj.weight"
    qkv_weight, qkv_scale, expected_qkv = qkv_fixture(4)
    dense = {
        key: torch.randn(4, 4).bfloat16()
        for key in (
            "model.layers.0.self_attn.o_proj.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        )
    }
    auxiliary = {
        "model.mtp.layers.0.self_attn.qkv_proj.weight": torch.ones(3, 2).to(
            torch.float8_e4m3fn
        ),
        "model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv": torch.ones(1, 1),
        "visual.proj.weight": torch.randn(4, 4).bfloat16(),
        "audio_encoder.proj.weight": torch.randn(4, 4).bfloat16(),
        "speech_embeddings.weight": torch.randn(4, 4).bfloat16(),
    }
    weights = {name: packed, qkv_name: qkv_weight, **dense, **auxiliary}
    scales = {name + "_scale": scale, qkv_name + "_scale_inv": qkv_scale}
    save_file(weights, source / "weights.safetensors")
    save_file(scales, source / "scales.safetensors")
    (source / "config.json").write_text(json.dumps(source_config()))
    mapping = {
        **dict.fromkeys(weights, "weights.safetensors"),
        **dict.fromkeys(scales, "scales.safetensors"),
    }
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    dequantize_model(source, output, target_dtype=target_dtype)
    with safe_open(output / "weights.safetensors", framework="pt") as reader:
        expected = torch.tensor([1.0, 4.0] * 32).repeat(2, 1).to(target_dtype)
        torch.testing.assert_close(reader.get_tensor(name), expected, atol=0, rtol=0)
        torch.testing.assert_close(
            reader.get_tensor(qkv_name), expected_qkv.to(target_dtype), atol=0, rtol=0
        )
        for key, value in dense.items():
            torch.testing.assert_close(
                reader.get_tensor(key), value.to(target_dtype), atol=0, rtol=0
            )
        for key, value in auxiliary.items():
            assert reader.get_tensor(key).dtype == value.dtype
            assert torch.equal(
                reader.get_tensor(key).view(torch.uint8), value.view(torch.uint8)
            )
    config = json.loads((output / "config.json").read_text())
    assert config["dtype"] == str(target_dtype).split(".")[-1]
    assert "quantization_config" not in config
    assert (
        config["mimo_mtp_source_quantization_config"]
        == source_config()["quantization_config"]
    )


def test_mtp_prefix_preservation_allows_absent_auxiliary(tmp_path: Path) -> None:
    save_file({"model.norm.weight": torch.ones(8)}, tmp_path / "model.safetensors")
    files, prefixes = _normalize_out_of_model_tensors_entries(
        MimoV2QModel.out_of_model_tensors
    )
    assert files == []
    assert prefixes == ["model.mtp"]
    state_dict = {}
    _merge_prefix_tensors_into_state_dict(prefixes, str(tmp_path), state_dict)
    assert state_dict == {}


def test_mask_compatibility_is_local_and_idempotent() -> None:
    def modern(*, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> tuple:
        return inputs_embeds, position_ids

    def legacy(*, input_embeds: torch.Tensor, cache_position: torch.Tensor) -> tuple:
        return input_embeds, cache_position

    value = torch.ones(1)
    adapted = _compatible_mask(modern)
    assert adapted(input_embeds=value, cache_position=value, position_ids=value) == (
        value,
        value,
    )
    assert _compatible_mask(adapted) is adapted
    assert _compatible_mask(legacy) is legacy
    assert "input_embeds" not in __import__("inspect").signature(modern).parameters
