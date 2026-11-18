# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.quantization.config import METHOD, FP8Config, QuantizeConfig, WeightOnlyConfig
from gptqmodel.quantization.dtype import available_float8_dtype_names
from gptqmodel.utils.model_dequant import detect_format


# FP8 is load-only. GPT-QModel no longer invents an FP8 checkpoint: the config
# only describes a production checkpoint so the loader can decode it. The
# invented `gptqmodel_fp8` alias and the row/tensor inverse-scale export are
# gone; a production checkpoint publishes the standard `fp8` name.


@pytest.mark.parametrize("format_name", available_float8_dtype_names())
def test_fp8_config_round_trip_is_load_only(format_name: str):
    cfg = QuantizeConfig(
        quant_method=METHOD.FP8,
        format=format_name,
        weight_scale_method="block",
        weight_block_size=[128, 128],
        weight_scale_semantics="multiplier",
    )

    assert isinstance(cfg, FP8Config)
    assert cfg.is_load_only() is True
    assert cfg.uses_weight_only_lifecycle() is True

    payload = cfg.to_dict()
    assert payload["method"] == METHOD.FP8
    # Production FP8 keeps the standard name; there is no `gptqmodel_fp8`.
    assert payload["quant_method"] == METHOD.FP8
    assert payload["format"] == format_name
    assert payload["checkpoint_format"] == format_name
    assert payload["weight_scale_method"] == "block"
    assert payload["weight_block_size"] == [128, 128]
    assert payload["weight_scale_semantics"] == "multiplier"

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, FP8Config)
    assert reloaded.format == format_name
    assert reloaded.weight_scale_method == "block"
    assert reloaded.weight_block_size == [128, 128]
    assert reloaded.is_load_only() is True


def test_fp8_default_config_describes_a_production_layout():
    cfg = FP8Config()
    payload = cfg.to_dict()

    assert payload["quant_method"] == "fp8"
    assert payload["weight_scale_method"] == "tensor"
    assert payload["weight_scale_semantics"] == "multiplier"


def test_legacy_inverse_layouts_still_load_but_are_not_outputs():
    """Old GPT-QModel row/inverse checkpoints remain readable."""

    payload = {
        "quant_method": "fp8",
        "method": "fp8",
        "format": "float8_e4m3fn",
        "weight_scale_method": "row",
        "weight_scale_semantics": "inverse",
    }
    cfg = QuantizeConfig.from_quant_config(payload)
    assert isinstance(cfg, FP8Config)
    assert cfg.method is METHOD.FP8
    assert cfg.weight_scale_method == "row"
    assert cfg.weight_scale_semantics == "inverse"


def test_weight_only_fp8_export_is_rejected():
    """The invented weight-only FP8 export entry point is gone."""

    with pytest.raises(ValueError, match="weight-only FP8 export was removed"):
        QuantizeConfig(weight_only={"method": "fp8"})

    with pytest.raises(ValueError, match="weight-only FP8 export was removed"):
        QuantizeConfig(weight_only=WeightOnlyConfig(method="fp8"))


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
                    "weight_scale_method": "block",
                    "weight_block_size": [16, 16],
                }
            }
        ),
        encoding="utf-8",
    )

    detected = detect_format(tmp_path, json.loads(config_path.read_text(encoding="utf-8")))
    assert detected == "fp8"


def test_quantize_refuses_load_only_fp8():
    """`BaseQModel.quantize()` fails closed before any calibration work."""

    from gptqmodel.models.base import BaseQModel

    class _Stub:
        quantize_config = FP8Config()

    with pytest.raises(ValueError, match="FP8 is load-only"):
        BaseQModel.quantize(_Stub())


def test_torch_fp8_pack_entry_points_refuse():
    """The encoder entry points are gone; each raises instead of writing."""

    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear

    module = TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=8,
        out_features=8,
        bias=False,
        register_buffers=True,
        format="float8_e4m3fn",
        weight_scale_method="row",
    )
    for name in ("pack", "pack_block", "pack_gpu", "pack_original"):
        with pytest.raises(ValueError, match="FP8 is load-only"):
            getattr(module, name)()
