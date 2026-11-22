# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Legacy-layout VL checkpoints must survive a re-save (#4802).

Transformers 4.x Gemma-3 VL checkpoints store the text decoder under
``language_model.model.layers.*`` while Transformers 5 builds
``model.language_model.layers.*``. ``GPTQModel.load`` already converts the keys
for inference, but the save-time export shell matched checkpoint names
verbatim, so ``save()`` refused (or zero-filled) every quantized tensor of the
rebuilt shell.

These tests build tiny CPU models for Gemma-3 VL, InternVL and Qwen2-VL,
quantize them, rewrite the checkpoint into the legacy layout, and then re-save
and reload them through ``GPTQModel``.
"""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.models.auto import MODEL_MAP_CASEFOLD
from gptqmodel.models.loader import _checkpoint_key_converter
from gptqmodel.models.writer import (
    _export_shell_key_converter,
    _validate_export_shell_quantized_slots,
)
from gptqmodel.nn_modules.qlinear import BaseQuantLinear


# --------------------------------------------------------------------------- #
# Tiny configs.  They only need enough geometry for the quantizer and kernel
# construction; no image inputs are exercised.
# --------------------------------------------------------------------------- #

_TEXT_CONFIG = {
    "vocab_size": 16,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "max_position_embeddings": 16,
}

_GEMMA3_VISION_CONFIG = {
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "image_size": 4,
    "patch_size": 2,
}


def _gemma3_config():
    from transformers import Gemma3Config

    return Gemma3Config(
        text_config=dict(_TEXT_CONFIG, head_dim=4),
        vision_config=dict(_GEMMA3_VISION_CONFIG),
        mm_tokens_per_image=4,
    )


def _internvl_config():
    from transformers.models.internvl.configuration_internvl import InternVLConfig

    return InternVLConfig(
        text_config=dict(_TEXT_CONFIG),
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 4,
            "patch_size": 2,
        },
        image_token_id=15,
        image_seq_length=4,
        downsample_ratio=0.5,
    )


def _qwen2_vl_config():
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig

    return Qwen2VLConfig(
        text_config=dict(
            _TEXT_CONFIG,
            # head_dim = 4, so the mrope sections must sum to head_dim // 2.
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
        ),
        vision_config={
            "depth": 1,
            "embed_dim": 8,
            "hidden_size": 8,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
        },
        image_token_id=15,
        vision_start_token_id=13,
        vision_end_token_id=14,
    )


# model_type, config factory, runtime prefix of a quantized decoder linear,
# legacy prefix the checkpoint uses for the same tensor.
VL_CASES = {
    "gemma3": (
        _gemma3_config,
        "model.language_model.layers.0.mlp.down_proj",
        "language_model.model.layers.0.mlp.down_proj",
    ),
    "internvl": (
        _internvl_config,
        "model.language_model.layers.0.mlp.down_proj",
        "language_model.model.layers.0.mlp.down_proj",
    ),
    "qwen2_vl": (
        _qwen2_vl_config,
        "model.language_model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.down_proj",
    ),
}


def _case_config(model_type):
    return VL_CASES[model_type][0]()


def _runtime_architectures(model_type):
    return {
        "gemma3": ["Gemma3ForConditionalGeneration"],
        "internvl": ["InternVLForConditionalGeneration"],
        "qwen2_vl": ["Qwen2VLForConditionalGeneration"],
    }[model_type]


def _tiny_model(model_type):
    torch.manual_seed(0)
    return AutoModelForImageTextToText.from_config(_case_config(model_type))


def _quantize_and_save(tmp_path, model_type, monkeypatch):
    """Quantize a tiny model and return the saved checkpoint directory."""

    definition = MODEL_MAP_CASEFOLD[model_type]
    # The multimodal processor needs torchvision assets that are unrelated to
    # key layout and absent from some CPU test images.
    monkeypatch.setattr(definition, "require_load_processor", False, raising=False)

    dense_dir = tmp_path / "dense"
    quantized_dir = tmp_path / "quantized"
    model = _tiny_model(model_type)
    model.save_pretrained(dense_dir)

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device="cpu",
    )
    wrapped = GPTQModel.load(
        str(dense_dir),
        quantize_config=quantize_config,
        backend=BACKEND.GPTQ_TORCH,
    )
    wrapped.quantize(
        CalibrationConfig(
            dataset=[
                {
                    "input_ids": torch.tensor(
                        [[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long
                    ),
                    "attention_mask": torch.ones((1, 8), dtype=torch.long),
                }
            ],
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.GPTQ_TORCH,
    )
    wrapped.save(quantized_dir)
    return quantized_dir, wrapped


def _rewrite_to_legacy_layout(checkpoint_dir, wrapped, model_type):
    """Rewrite the saved runtime keys into the Transformers-4.x layout."""

    definition = MODEL_MAP_CASEFOLD[model_type]
    to_checkpoint = _checkpoint_key_converter(definition, wrapped.model, to_runtime=False)
    assert callable(to_checkpoint)

    source = checkpoint_dir / "model.safetensors"
    tensors = load_file(source)
    legacy = {to_checkpoint(name): tensor for name, tensor in tensors.items()}
    assert len(legacy) == len(tensors)
    save_file(legacy, source)
    return set(legacy)


def _normalized_state_dict(checkpoint_dir, wrapped, model_type):
    """Load a checkpoint and return its tensors under runtime names."""

    definition = MODEL_MAP_CASEFOLD[model_type]
    to_runtime = _checkpoint_key_converter(definition, wrapped.model, to_runtime=True)
    assert callable(to_runtime)
    return {
        to_runtime(name): tensor
        for name, tensor in load_file(checkpoint_dir / "model.safetensors").items()
    }


def _assert_state_dicts_equal(left, right):
    assert set(left) == set(right)
    for name, tensor in left.items():
        other = right[name]
        assert tensor.shape == other.shape, name
        assert tensor.dtype == other.dtype, name
        assert torch.equal(tensor, other), name


# --------------------------------------------------------------------------- #
# Unit coverage: the export-shell guard must compare runtime names, not the
# checkpoint's legacy names, and must still refuse a genuinely unmapped tensor.
# --------------------------------------------------------------------------- #


def _nested_shell(runtime_keys, config):
    shell = torch.nn.Module()
    shell.config = config
    for key in runtime_keys:
        parts = key.split(".")
        parent = shell
        for part in parts[:-1]:
            child = getattr(parent, part, None)
            if child is None:
                child = torch.nn.Module()
                setattr(parent, part, child)
            parent = child
        if not hasattr(parent, parts[-1]):
            parent.register_buffer(parts[-1], torch.zeros(1))
    # ``load_checkpoint_in_model_then_tie_weights`` calls this after loading.
    shell.tie_weights = lambda **kwargs: None
    return shell


@pytest.mark.parametrize("model_type", sorted(VL_CASES))
def test_export_shell_validation_accepts_legacy_vl_keys(tmp_path, model_type):
    config = SimpleNamespace(
        model_type=model_type,
        architectures=_runtime_architectures(model_type),
        dtype=torch.float16,
    )
    runtime_prefix = VL_CASES[model_type][1]
    legacy_prefix = VL_CASES[model_type][2]
    runtime_keys = [
        f"{runtime_prefix}.qweight",
        f"{runtime_prefix}.qzeros",
        f"{runtime_prefix}.scales",
        f"{runtime_prefix}.g_idx",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    shell = _nested_shell(runtime_keys, config)
    definition = MODEL_MAP_CASEFOLD[model_type]

    to_checkpoint = _checkpoint_key_converter(definition, shell, to_runtime=False)
    checkpoint_keys = [to_checkpoint(key) for key in runtime_keys]
    assert f"{legacy_prefix}.qweight" in checkpoint_keys

    checkpoint = tmp_path / "model.safetensors"
    save_file({key: torch.zeros(1) for key in checkpoint_keys}, checkpoint)

    converter = _export_shell_key_converter(object.__new__(definition), shell)
    _validate_export_shell_quantized_slots(shell, checkpoint, key_converter=converter)

    # Without the conversion the legacy names have no destination module; the
    # pre-#4802 behavior must remain a loud failure.
    with pytest.raises(ValueError, match="export shell has no module"):
        _validate_export_shell_quantized_slots(shell, checkpoint)

    # A truly unmapped quantized module must still fail closed.
    unmapped = to_checkpoint(f"{runtime_prefix}.missing_proj.qweight")
    save_file(
        {**{key: torch.zeros(1) for key in checkpoint_keys}, unmapped: torch.zeros(1)},
        checkpoint,
    )
    with pytest.raises(ValueError, match="export shell has no module"):
        _validate_export_shell_quantized_slots(shell, checkpoint, key_converter=converter)


# --------------------------------------------------------------------------- #
# End to end: quantize -> save -> legacy layout -> load -> save -> load.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_type", sorted(VL_CASES))
def test_vl_legacy_checkpoint_resave_roundtrip(tmp_path, model_type, monkeypatch):
    quantized_dir, wrapped = _quantize_and_save(tmp_path, model_type, monkeypatch)
    legacy_keys = _rewrite_to_legacy_layout(quantized_dir, wrapped, model_type)

    runtime_prefix = VL_CASES[model_type][1]
    legacy_prefix = VL_CASES[model_type][2]
    assert f"{legacy_prefix}.qweight" in legacy_keys
    assert f"{runtime_prefix}.qweight" not in legacy_keys

    reloaded = GPTQModel.load(
        str(quantized_dir), backend=BACKEND.GPTQ_TORCH, device="cpu"
    )
    quantized = [
        module
        for module in reloaded.model.modules()
        if isinstance(module, BaseQuantLinear)
    ]
    assert quantized, "the legacy checkpoint must load quantized modules"

    resaved_dir = tmp_path / "resaved"
    reloaded.save(resaved_dir)
    GPTQModel.load(str(resaved_dir), backend=BACKEND.GPTQ_TORCH, device="cpu")

    # Compare in runtime names so families that keep writing the runtime
    # layout (InternVL/Qwen2-VL) and Gemma-3's legacy layout both round-trip.
    before = _normalized_state_dict(quantized_dir, wrapped, model_type)
    after = _normalized_state_dict(resaved_dir, reloaded, model_type)
    _assert_state_dicts_equal(before, after)
