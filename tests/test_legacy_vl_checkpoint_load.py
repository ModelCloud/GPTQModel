# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Legacy Transformers-4.x key layouts must load through the quantized loader.

`Qwen2-VL-2B-Instruct-GPTQ-Int8` stores `model.layers.*` / `visual.*`, while
Transformers 5 builds `model.language_model.layers.*` / `model.visual.*`.
`GPTQModel.load` previously read the checkpoint keys verbatim and rejected every
quantized weight as missing (or, with validation disabled, silently left them
unused). The loader now applies the model's HF checkpoint conversion map.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.base import BaseQModel
from gptqmodel.models.loader import _checkpoint_key_converter


LEGACY_QWEN2_VL_CHECKPOINT = Path(
    "/monster/data/model/Qwen2-VL-2B-Instruct-GPTQ-Int8"
)


def _fake_qwen2_vl_model():
    return SimpleNamespace(
        config=SimpleNamespace(
            model_type="qwen2_vl",
            architectures=["Qwen2VLForConditionalGeneration"],
        )
    )


def test_checkpoint_key_converter_uses_the_hf_conversion_map():
    model = _fake_qwen2_vl_model()
    to_runtime = _checkpoint_key_converter(BaseQModel, model, to_runtime=True)
    to_checkpoint = _checkpoint_key_converter(BaseQModel, model, to_runtime=False)

    assert to_runtime("model.layers.0.mlp.down_proj.qweight") == (
        "model.language_model.layers.0.mlp.down_proj.qweight"
    )
    assert to_runtime("visual.blocks.0.attn.qkv.weight") == (
        "model.visual.blocks.0.attn.qkv.weight"
    )
    # Already-runtime keys pass through unchanged, so re-saving a native
    # checkpoint does not rewrite its layout.
    assert to_runtime("model.language_model.layers.0.mlp.down_proj.qweight") == (
        "model.language_model.layers.0.mlp.down_proj.qweight"
    )
    assert to_runtime("model.visual.blocks.0.attn.qkv.weight") == (
        "model.visual.blocks.0.attn.qkv.weight"
    )
    # Validation runs in the opposite direction.
    assert to_checkpoint("model.language_model.layers.0.mlp.down_proj.qweight") == (
        "model.layers.0.mlp.down_proj.qweight"
    )


def _checkpoint_uses_legacy_layout() -> bool:
    config_path = LEGACY_QWEN2_VL_CHECKPOINT / "config.json"
    if not config_path.is_file():
        return False
    from safetensors import safe_open

    checkpoint = LEGACY_QWEN2_VL_CHECKPOINT / "model.safetensors"
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    return (
        "model.layers.0.mlp.down_proj.qweight" in keys
        and "model.language_model.layers.0.mlp.down_proj.qweight" not in keys
    )


@pytest.mark.skipif(
    not LEGACY_QWEN2_VL_CHECKPOINT.is_dir(),
    reason="legacy-layout Qwen2-VL GPTQ checkpoint is not available",
)
def test_loads_legacy_layout_quantized_qwen2_vl_checkpoint(monkeypatch):
    if not _checkpoint_uses_legacy_layout():
        pytest.skip("checkpoint does not use the legacy Transformers 4.x layout")

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    # The processor is unrelated to the key-layout bug and needs torchvision,
    # which is not installed in every test environment.
    monkeypatch.setattr(Qwen2VLQModel, "require_load_processor", False, raising=False)

    model = GPTQModel.load(
        str(LEGACY_QWEN2_VL_CHECKPOINT),
        device="cpu",
        backend=BACKEND.GPTQ_TORCH,
    )

    quantized = [
        name
        for name, module in model.model.named_modules()
        if isinstance(module, BaseQuantLinear)
    ]
    assert len(quantized) == 196
    assert "model.language_model.layers.0.mlp.down_proj" in quantized

    input_ids = model.tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.inference_mode():
        logits = model.model(input_ids=input_ids).logits
    assert logits.shape[-1] > 0
    assert torch.isfinite(logits).all()


def test_legacy_checkpoint_config_names_the_legacy_layout():
    config_path = LEGACY_QWEN2_VL_CHECKPOINT / "config.json"
    if not config_path.is_file():
        pytest.skip("legacy-layout Qwen2-VL GPTQ checkpoint is not available")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["model_type"] == "qwen2_vl"
    assert "Qwen2VLForConditionalGeneration" in config["architectures"]
