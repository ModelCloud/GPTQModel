# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel._quant_methods import FP8_EXPORT_QUANT_METHOD
from gptqmodel.quantization.config import FORMAT, METHOD, FP8Config, QuantizeConfig
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
    assert payload["quant_method"] == FP8_EXPORT_QUANT_METHOD
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


# --- #5128: non-block FP8 must not claim transformers' `fp8` quant_method ---
#
# transformers registers `fp8` for `FineGrainedFP8`, which reads a 2-D
# `weight_scale_inv` block grid as a multiplier. GPT-QModel writes inverse
# scales for every layout (`row`, `tensor`, and `block`), so no GPT-QModel FP8
# checkpoint is readable by plain transformers. Publishing `gptqmodel_fp8`
# lets the import-time guard refuse the load immediately; `fp8` is only kept
# for a third-party block/multiplier/e4m3 payload.

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_fp8_config_publishes_gptqmodel_only_quant_method():
    cfg = FP8Config()

    payload = cfg.to_dict()
    assert payload["method"] == METHOD.FP8
    assert payload["quant_method"] == FP8_EXPORT_QUANT_METHOD
    assert payload["quant_method"] != METHOD.FP8
    assert payload["weight_scale_method"] == "row"
    assert payload.get("weight_block_size") is None

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, FP8Config)
    assert reloaded.method is METHOD.FP8
    assert reloaded.weight_scale_method == "row"
    # Re-saving the loaded config keeps the non-colliding name.
    assert reloaded.to_dict()["quant_method"] == FP8_EXPORT_QUANT_METHOD


@pytest.mark.parametrize("scale_method", ["row", "tensor"])
def test_non_block_fp8_layouts_publish_gptqmodel_only_quant_method(scale_method):
    cfg = FP8Config(weight_scale_method=scale_method)

    payload = cfg.to_dict()
    assert payload["weight_scale_method"] == scale_method
    assert payload["quant_method"] == FP8_EXPORT_QUANT_METHOD


def test_block_fp8_layout_publishes_gptqmodel_only_quant_method():
    """`block` still stores an inverse scale, which `FineGrainedFP8` misreads."""

    cfg = FP8Config(weight_scale_method="block", weight_block_size=[128, 128])

    payload = cfg.to_dict()
    assert payload["weight_scale_method"] == "block"
    assert payload["weight_scale_semantics"] == "inverse"
    assert payload["quant_method"] == FP8_EXPORT_QUANT_METHOD


def test_fp8_layout_is_hf_readable_requires_block_multiplier_and_e4m3():
    """Only a third-party block/multiplier/e4m3 payload keeps the `fp8` name."""

    from types import SimpleNamespace

    from gptqmodel.quantization.config import _fp8_layout_is_hf_readable

    readable = SimpleNamespace(
        weight_scale_method="block",
        weight_block_size=[128, 128],
        weight_scale_semantics="multiplier",
        format="float8_e4m3fn",
        dynamic=None,
    )
    assert _fp8_layout_is_hf_readable(readable) is True

    for field, value in (
        ("weight_scale_method", "row"),
        ("weight_block_size", None),
        ("weight_scale_semantics", "inverse"),
        ("format", "float8_e5m2"),
    ):
        assert _fp8_layout_is_hf_readable(
            SimpleNamespace(**{**vars(readable), field: value})
        ) is False

    for override in (
        {"weight_scale_method": "row"},
        {"weight_block_size": None},
        {"weight_scale_semantics": "inverse"},
        {"format": "float8_e5m2"},
    ):
        assert _fp8_layout_is_hf_readable(
            SimpleNamespace(**{**vars(readable), "dynamic": {"layer": override}})
        ) is False

    assert _fp8_layout_is_hf_readable(
        SimpleNamespace(
            **{
                **vars(readable),
                "dynamic": {
                    "layer": {"weight_scale_method": "block", "weight_block_size": [32, 32]}
                },
            }
        )
    ) is True


def test_fp8_config_rejects_multiplier_scale_semantics():
    """GPT-QModel cannot write the multiplier convention `FineGrainedFP8` reads."""

    with pytest.raises(ValueError, match="weight_scale_semantics"):
        FP8Config(weight_scale_semantics="multiplier")


@pytest.mark.parametrize(
    "base_kwargs,dynamic_override",
    [
        (
            {"weight_scale_method": "block", "weight_block_size": [128, 128]},
            {"weight_scale_method": "row"},
        ),
        (
            {"weight_scale_method": "row"},
            {"weight_scale_method": "block", "weight_block_size": [32, 32]},
        ),
    ],
)
def test_mixed_dynamic_fp8_layouts_publish_gptqmodel_only_quant_method(
    base_kwargs, dynamic_override
):
    """One non-block module makes the whole checkpoint unreadable to HF."""

    cfg = FP8Config(
        dynamic={"model.layers.0.mlp.down_proj": dynamic_override},
        **base_kwargs,
    )

    assert cfg.to_dict()["quant_method"] == FP8_EXPORT_QUANT_METHOD


def test_legacy_fp8_quant_method_still_loads():
    """Checkpoints already published with `quant_method: "fp8"` keep loading."""

    payload = FP8Config().to_dict()
    payload["quant_method"] = "fp8"
    assert QuantizeConfig.from_quant_config(payload).method is METHOD.FP8

    # The deprecated alias alone also resolves, even without `method`.
    alias_only = dict(payload)
    alias_only.pop("method")
    reloaded = QuantizeConfig.from_quant_config(alias_only)
    assert isinstance(reloaded, FP8Config)
    assert reloaded.method is METHOD.FP8
    assert reloaded.weight_scale_method == "row"


def test_hf_compat_warning_for_gptqmodel_only_fp8():
    from gptqmodel.models.writer import hf_compat_warnings

    warnings = hf_compat_warnings(FP8Config(), FORMAT.FP8)

    assert len(warnings) == 1
    assert FP8_EXPORT_QUANT_METHOD in warnings[0]
    assert "FineGrainedFP8" in warnings[0]
    assert "GPTQModel.load" in warnings[0]

    # A block layout stores the same inverse scale, so it warns too.
    block = FP8Config(weight_scale_method="block", weight_block_size=[128, 128])
    block_warnings = hf_compat_warnings(block, FORMAT.FP8)
    assert len(block_warnings) == 1
    assert FP8_EXPORT_QUANT_METHOD in block_warnings[0]
    assert "weight_scale_semantics" in block_warnings[0]


def _build_tiny_qwen2(target_dir: Path) -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    target_dir.mkdir(parents=True, exist_ok=True)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(target_dir, safe_serialization=True)


def _read_checkpoint_config(directory: Path) -> dict:
    return json.loads((directory / "config.json").read_text(encoding="utf-8"))


def test_default_fp8_save_publishes_gptqmodel_quant_method_and_refuses_hf(tmp_path):
    """Save a real default FP8 checkpoint, reload it, and refuse it in plain HF."""

    from gptqmodel import GPTQModel

    source = tmp_path / "source"
    quantized = tmp_path / "quantized"
    _build_tiny_qwen2(source)

    model = GPTQModel.load(str(source), quantize_config=FP8Config(device="cpu"))
    model.quantize(backend="torch")
    model.save(str(quantized))

    saved = _read_checkpoint_config(quantized)["quantization_config"]
    assert saved["method"] == METHOD.FP8
    assert saved["quant_method"] == FP8_EXPORT_QUANT_METHOD
    assert saved["weight_scale_method"] == "row"
    assert saved.get("weight_block_size") is None

    weights = load_file(str(quantized / "model.safetensors"))
    weight = weights["model.layers.0.self_attn.q_proj.weight"]
    scale = weights["model.layers.0.self_attn.q_proj.weight_scale_inv"]
    assert weight.dtype == torch.float8_e4m3fn
    assert tuple(scale.shape) == (weight.shape[0],)
    assert scale.dtype == torch.float32

    # `GPTQModel.load` reads the published name back as the same FP8 layout.
    reloaded = GPTQModel.load(str(quantized))
    assert isinstance(reloaded.quantize_config, FP8Config)
    assert reloaded.quantize_config.method is METHOD.FP8
    assert reloaded.quantize_config.weight_scale_method == "row"

    # Re-saving keeps the non-colliding name.
    resaved = tmp_path / "resaved"
    reloaded.save(str(resaved))
    assert (
        _read_checkpoint_config(resaved)["quantization_config"]["quant_method"]
        == FP8_EXPORT_QUANT_METHOD
    )

    # Plain transformers, with GPT-QModel imported (so the refusing quantizer is
    # installed), must fail fast and name both the method and the loader.
    script = """
import time
import gptqmodel  # noqa: F401  installs the refusing quantizer
from transformers import AutoModelForCausalLM

started = time.time()
try:
    AutoModelForCausalLM.from_pretrained({path!r}, device_map="cpu", dtype="bfloat16")
except Exception as error:
    print(type(error).__name__, ":", error)
    print("elapsed", round(time.time() - started, 2))
else:
    raise SystemExit("transformers loaded a GPT-QModel-only FP8 checkpoint")
""".format(path=str(quantized))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    elapsed = time.time() - started

    assert proc.returncode == 0, proc.stderr
    assert "cannot be loaded by `transformers`" in proc.stdout, proc.stdout
    assert FP8_EXPORT_QUANT_METHOD in proc.stdout, proc.stdout
    assert "GPTQModel.load" in proc.stdout, proc.stdout
    # The old path spent minutes broadcasting a multi-billion-element temporary
    # on a real checkpoint; this tiny one must still fail immediately.
    assert elapsed < 60, f"refusal took {elapsed:.1f}s"


def test_block_fp8_save_publishes_gptqmodel_quant_method_and_round_trips(tmp_path):
    """A block save stores inverse scales too, so it must not claim `fp8`."""

    from gptqmodel import GPTQModel

    source = tmp_path / "source"
    quantized = tmp_path / "quantized"
    _build_tiny_qwen2(source)

    cfg = FP8Config(
        weight_scale_method="block",
        weight_block_size=[32, 32],
        device="cpu",
    )
    model = GPTQModel.load(str(source), quantize_config=cfg)
    model.quantize(backend="torch")
    model.save(str(quantized))

    saved = _read_checkpoint_config(quantized)["quantization_config"]
    assert saved["method"] == METHOD.FP8
    assert saved["quant_method"] == FP8_EXPORT_QUANT_METHOD
    assert saved["weight_scale_method"] == "block"
    assert saved["weight_scale_semantics"] == "inverse"

    # The block layout still loads back through GPT-QModel unchanged.
    reloaded = GPTQModel.load(str(quantized))
    assert isinstance(reloaded.quantize_config, FP8Config)
    assert reloaded.quantize_config.method is METHOD.FP8
    assert reloaded.quantize_config.weight_scale_method == "block"
    assert reloaded.quantize_config.weight_block_size == [32, 32]
