# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.quantization.config import METHOD, FP8Config, QuantizeConfig, WeightOnlyConfig
from gptqmodel.quantization.dtype import available_float8_dtype_names
from gptqmodel.utils.model_dequant import detect_format
from tests.fp8_test_utils import quantize_fp8_weight


# FP8 is load-only. GPT-QModel no longer invents an FP8 checkpoint: the config
# only describes a production checkpoint so the loader can decode it. The
# invented `gptqmodel_fp8` export and the row/tensor inverse-scale writer are
# gone; a production checkpoint publishes the standard `fp8` name. Legacy
# checkpoints that carry the old alias remain readable (see the #5272 section).


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


# --- #5272: read the legacy `gptqmodel_fp8` checkpoints -------------------
#
# #5242 removed the invented FP8 exporter, but checkpoints written between
# #5181 and #5242 still carry `quant_method: "gptqmodel_fp8"` (all layouts
# since #5214). The name is load-only compatibility: it must normalize to
# `METHOD.FP8` on read and must be refused by plain transformers, exactly like
# the other GPT-QModel-only names. Released checkpoints up to v7.11.2 publish
# the standard `fp8` name with the same inverse scale, so that combination is
# refused too.

_TINY_QWEN2_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "max_position_embeddings": 128,
    "tie_word_embeddings": True,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}

_FP8_PROJECTIONS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


def _build_legacy_fp8_checkpoint(
    target_dir,
    *,
    quant_method: str = "gptqmodel_fp8",
    include_method: bool = True,
    weight_scale_method: str = "row",
    weight_block_size=None,
):
    """Write a checkpoint in the layout the pre-#5242 writer produced.

    The test-only reference encoder in ``fp8_test_utils`` computes the same
    inverse scale the removed exporter did; production code has no writer.
    """

    from transformers import Qwen2Config, Qwen2ForCausalLM

    target_dir.mkdir(parents=True, exist_ok=True)
    config = Qwen2Config(**_TINY_QWEN2_CONFIG)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)

    tensors = {}
    for key, tensor in model.state_dict().items():
        if tensor.ndim == 2 and key.endswith(_FP8_PROJECTIONS):
            qweight, scale_inv = quantize_fp8_weight(
                tensor,
                weight_scale_method=weight_scale_method,
                weight_block_size=weight_block_size,
            )
            tensors[key] = qweight
            tensors[key + "_scale_inv"] = scale_inv
        elif key == "lm_head.weight":
            # Tied to the embedding; safetensors rejects the shared storage.
            continue
        else:
            tensors[key] = tensor
    save_file(
        {key: value.contiguous() for key, value in tensors.items()},
        str(target_dir / "model.safetensors"),
    )

    quantization_config = {
        "quant_method": quant_method,
        "format": "float8_e4m3fn",
        "checkpoint_format": "float8_e4m3fn",
        "weight_scale_method": weight_scale_method,
        "weight_scale_semantics": "inverse",
        "bits": 8,
    }
    if include_method:
        quantization_config["method"] = "fp8"
    if weight_block_size is not None:
        quantization_config["weight_block_size"] = list(weight_block_size)

    model_config = config.to_dict()
    model_config["quantization_config"] = quantization_config
    (target_dir / "config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )
    (target_dir / "quantize_config.json").write_text(
        json.dumps(quantization_config, indent=2), encoding="utf-8"
    )


def _legacy_fp8_reference_logits(checkpoint_dir, input_ids):
    """Dense logits from the checkpoint's own FP8 payload and inverse scales."""

    from transformers import Qwen2Config, Qwen2ForCausalLM

    tensors = load_file(str(checkpoint_dir / "model.safetensors"))
    model_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    model_config.pop("quantization_config", None)
    model = Qwen2ForCausalLM(Qwen2Config(**model_config)).float().eval()

    dense = {}
    for key, tensor in tensors.items():
        if key.endswith("_scale_inv"):
            continue
        scale = tensors.get(key + "_scale_inv")
        if scale is None:
            dense[key] = tensor.float()
            continue
        weight = tensor.float()
        if scale.ndim == 1:
            weight = weight / scale.float()[:, None]
        else:
            block_rows, block_cols = scale.shape
            rows, cols = weight.shape
            weight = weight.reshape(
                block_rows, rows // block_rows, block_cols, cols // block_cols
            )
            weight = weight / scale.float()[:, None, :, None]
            weight = weight.reshape(rows, cols)
        dense[key] = weight

    missing, unexpected = model.load_state_dict(dense, strict=False)
    assert set(missing) == {"lm_head.weight"}
    assert not unexpected
    with torch.no_grad():
        return model(input_ids=input_ids).logits.float()


def test_legacy_gptqmodel_fp8_quant_method_is_accepted_on_load():
    """The removed alias is read-only compatibility, never a new export."""

    payload = {
        "quant_method": "gptqmodel_fp8",
        "format": "float8_e4m3fn",
        "weight_scale_method": "row",
        "weight_scale_semantics": "inverse",
    }
    cfg = QuantizeConfig.from_quant_config(payload)

    assert isinstance(cfg, FP8Config)
    assert cfg.method is METHOD.FP8
    assert cfg.weight_scale_method == "row"
    assert cfg.weight_scale_semantics == "inverse"
    # A re-save publishes the standard name; the alias never comes back.
    assert cfg.to_dict()["quant_method"] == "fp8"


@pytest.mark.parametrize(
    "include_method,weight_scale_method,weight_block_size",
    [
        (True, "row", None),
        (True, "block", [32, 32]),
        (False, "row", None),
    ],
    ids=["published-row", "published-block", "alias-only-row"],
)
def test_legacy_gptqmodel_fp8_checkpoint_loads_and_forwards(
    tmp_path, include_method, weight_scale_method, weight_block_size
):
    """A legacy checkpoint loads natively and matches a dense Torch oracle."""

    from gptqmodel import GPTQModel

    checkpoint = tmp_path / "legacy_fp8"
    _build_legacy_fp8_checkpoint(
        checkpoint,
        include_method=include_method,
        weight_scale_method=weight_scale_method,
        weight_block_size=weight_block_size,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    model = GPTQModel.load(str(checkpoint), device="cpu", dtype=torch.float32)

    assert isinstance(model.quantize_config, FP8Config)
    assert model.quantize_config.method is METHOD.FP8
    assert model.quantize_config.weight_scale_method == weight_scale_method
    assert model.quantize_config.weight_scale_semantics == "inverse"
    with torch.no_grad():
        actual = model.model(input_ids=input_ids).logits.float()
    expected = _legacy_fp8_reference_logits(checkpoint, input_ids)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_production_fp8_multiplier_config_still_uses_hf_quantizer():
    """The legacy inverse guard must not refuse production FP8 checkpoints."""

    from transformers.quantizers import auto as hf_auto
    from transformers.utils.quantization_config import FineGrainedFP8Config

    from gptqmodel._quant_methods import is_legacy_inverse_fp8_config

    config = {
        "quant_method": "fp8",
        "weight_scale_method": "block",
        "weight_block_size": [128, 128],
        "weight_scale_semantics": "multiplier",
    }
    assert is_legacy_inverse_fp8_config(config) is False

    quantizer = hf_auto.AutoHfQuantizer.from_config(config)
    assert isinstance(quantizer.quantization_config, FineGrainedFP8Config)
