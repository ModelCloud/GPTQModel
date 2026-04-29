# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.quantization.config import METHOD, FP8Config, QuantizeConfig
from gptqmodel.quantization.dtype import available_float8_dtype_names
from gptqmodel.utils.model_dequant import detect_format


@pytest.mark.parametrize("format_name", available_float8_dtype_names())
def test_fp8_quantize_config_round_trip(format_name: str):
    cfg = QuantizeConfig(
        quant_method=METHOD.FP8,
        format=format_name,
        weight_scale_method="block",
        weight_block_size=[128, 128],
    )

    assert isinstance(cfg, FP8Config)
    assert cfg.uses_weight_only_lifecycle() is True

    payload = cfg.to_dict()
    assert payload["method"] == METHOD.FP8
    assert payload["quant_method"] == METHOD.FP8
    assert payload["format"] == format_name
    assert payload["checkpoint_format"] == format_name
    assert payload["weight_scale_method"] == "block"
    assert payload["weight_block_size"] == [128, 128]
    assert payload["weight_scale_semantics"] == "inverse"

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, FP8Config)
    assert reloaded.format == format_name
    assert reloaded.weight_scale_method == "block"
    assert reloaded.weight_block_size == [128, 128]


@pytest.mark.parametrize("format_name", available_float8_dtype_names())
def test_detect_format_identifies_fp8_from_checkpoint_or_config(tmp_path, format_name: str):
    shard_path = tmp_path / "model.safetensors"
    if format_name.endswith("fnuz"):
        save_file({}, str(shard_path))
    else:
        save_file(
            {
                "model.layers.0.self_attn.q_proj.weight": torch.zeros((16, 16), dtype=getattr(torch, format_name)),
                "model.layers.0.self_attn.q_proj.weight_scale_inv": torch.ones((16,), dtype=torch.float32),
            },
            str(shard_path),
        )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "quantization_config": {
                    "method": "fp8",
                    "format": format_name,
                    "weight_scale_method": "row",
                    "weight_scale_semantics": "inverse",
                }
            }
        ),
        encoding="utf-8",
    )

    detected = detect_format(tmp_path, json.loads(config_path.read_text(encoding="utf-8")))
    assert detected == "fp8"
