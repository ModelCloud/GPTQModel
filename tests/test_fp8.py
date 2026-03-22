# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.quantization.config import METHOD, FP8Config, QuantizeConfig
from gptqmodel.utils.model_dequant import detect_format


@pytest.mark.skipif(not hasattr(torch, "float8_e5m2"), reason="float8_e5m2 unavailable")
def test_fp8_quantize_config_round_trip():
    cfg = QuantizeConfig(
        quant_method=METHOD.FP8,
        format="float8_e5m2",
        weight_scale_method="block",
        weight_block_size=[128, 128],
    )

    assert isinstance(cfg, FP8Config)
    assert cfg.uses_weight_only_lifecycle() is True

    payload = cfg.to_dict()
    assert payload["method"] == METHOD.FP8
    assert payload["quant_method"] == METHOD.FP8
    assert payload["format"] == "float8_e5m2"
    assert payload["checkpoint_format"] == "float8_e5m2"
    assert payload["weight_scale_method"] == "block"
    assert payload["weight_block_size"] == [128, 128]
    assert payload["weight_scale_semantics"] == "inverse"

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, FP8Config)
    assert reloaded.format == "float8_e5m2"
    assert reloaded.weight_scale_method == "block"
    assert reloaded.weight_block_size == [128, 128]


@pytest.mark.skipif(not hasattr(torch, "float8_e5m2"), reason="float8_e5m2 unavailable")
def test_detect_format_identifies_fp8_from_e5m2_checkpoint(tmp_path):
    shard_path = tmp_path / "model.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros((16, 16), dtype=torch.float8_e5m2),
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
                    "format": "float8_e5m2",
                    "weight_scale_method": "row",
                    "weight_scale_semantics": "inverse",
                }
            }
        ),
        encoding="utf-8",
    )

    detected = detect_format(tmp_path, json.loads(config_path.read_text(encoding="utf-8")))
    assert detected == "fp8"
