# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the ERNIE 4.5 VL MoE generate-hook cleanup.

`pre_quantize_generate_hook_end()` restores the captured modules after input
capture. With `offload_to_disk=False` it used to call
`setattr(core_model, vision_module, ...)`, passing the module *object* as the
attribute name:

    TypeError: attribute name must be string, not 'Linear'

The cleanup must therefore address `vision_tower` and `resampler_model` by
their string attribute names, and both disk-offload configurations must work.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models import definitions
from gptqmodel.models.definitions.ernie4_5_vl_moe import Ernie4_5_VLMoeQModel


_HIDDEN = 64  # 64x64 float32 == 16 KiB, above the 4 KiB disk-offload floor


class _CoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(_HIDDEN, _HIDDEN)
        self.language_model.rotary_emb = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.vision_tower = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.resampler_model = nn.Linear(_HIDDEN, _HIDDEN, bias=False)


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _CoreModel()


def _definition(model=None, *, offload_to_disk: bool):
    instance = object.__new__(Ernie4_5_VLMoeQModel)
    nn.Module.__init__(instance)
    instance.model = model if model is not None else _Wrapper()
    instance.quantized = True
    instance.load_quantized_model = False
    instance.quantize_config = SimpleNamespace(
        device=torch.device("cpu"),
        offload_to_disk=offload_to_disk,
        offload_to_disk_path="",
    )
    return instance


def test_hook_start_materializes_named_modules():
    definition = _definition(offload_to_disk=False)

    definition.pre_quantize_generate_hook_start()

    core_model = definition.model.model
    # The names must still resolve after the materialize step; a module object
    # used as an attribute name would have raised before this point.
    assert isinstance(core_model.vision_tower, nn.Linear)
    assert isinstance(core_model.resampler_model, nn.Linear)
    assert isinstance(core_model.language_model.embed_tokens, nn.Embedding)
    assert isinstance(core_model.language_model.rotary_emb, nn.Linear)


def test_hook_end_without_disk_offload_uses_string_attribute_names(monkeypatch, tmp_path):
    definition = _definition(offload_to_disk=False)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)

    moved = []

    def _recording_move_to(obj, device, dtype=None):
        moved.append(obj)
        return obj

    monkeypatch.setattr(definitions.ernie4_5_vl_moe, "move_to", _recording_move_to)
    core_model = definition.model.model
    vision_tower = core_model.vision_tower
    resampler_model = core_model.resampler_model

    definition.pre_quantize_generate_hook_end()

    assert moved == [
        core_model.language_model.embed_tokens,
        core_model.language_model.rotary_emb,
        vision_tower,
        resampler_model,
    ]
    # Both modules must remain registered under their real names.
    assert core_model.vision_tower is vision_tower
    assert core_model.resampler_model is resampler_model
    assert hasattr(core_model, "vision_tower")
    assert hasattr(core_model, "resampler_model")


def test_hook_end_without_disk_offload_releases_real_modules(tmp_path):
    definition = _definition(offload_to_disk=False)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)
    core_model = definition.model.model
    vision_tower = core_model.vision_tower
    resampler_model = core_model.resampler_model

    definition.pre_quantize_generate_hook_end()

    assert core_model.vision_tower is vision_tower
    assert core_model.resampler_model is resampler_model
    # The non-offload path moves real (materialized) tensors to CPU.
    assert not vision_tower.weight.is_meta
    assert not resampler_model.weight.is_meta
    assert vision_tower.weight.device.type == "cpu"
    assert resampler_model.weight.device.type == "cpu"


def test_hook_end_with_disk_offload_keeps_modules_registered(tmp_path):
    definition = _definition(offload_to_disk=True)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)
    core_model = definition.model.model
    vision_tower = core_model.vision_tower
    resampler_model = core_model.resampler_model

    definition.pre_quantize_generate_hook_end()

    assert core_model.vision_tower is vision_tower
    assert core_model.resampler_model is resampler_model
    # Disk offload leaves meta placeholders plus restore hooks in place.
    assert vision_tower.weight.is_meta
    assert resampler_model.weight.is_meta
    assert core_model.language_model.embed_tokens.weight.is_meta
    assert core_model.language_model.rotary_emb.weight.is_meta
    assert hasattr(vision_tower, "_hf_hook")
    assert hasattr(resampler_model, "_hf_hook")


@pytest.mark.parametrize("offload_to_disk", [True, False])
def test_hook_end_skips_absent_optional_modules(tmp_path, offload_to_disk):
    definition = _definition(offload_to_disk=offload_to_disk)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)
    core_model = definition.model.model
    del core_model.resampler_model

    definition.pre_quantize_generate_hook_end()

    assert not hasattr(core_model, "resampler_model")


def _tiny_ernie4_5_vl_moe():
    from transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeConfig,
    )
    from transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeForConditionalGeneration,
    )

    config = Ernie4_5_VLMoeConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 64,
            "max_position_embeddings": 64,
            "moe_intermediate_size": [40, 32],
            "moe_k": 2,
            "moe_num_experts": 2,
            "moe_num_shared_experts": 1,
            "mlp_layer_types": ["dense", "sparse"],
            "layer_types": ["full_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [4, 4, 8],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "patch_size": 16,
            "image_size": 32,
            "spatial_merge_size": 2,
        },
    )
    config._attn_implementation = "eager"
    return Ernie4_5_VLMoeForConditionalGeneration(config)


@pytest.mark.parametrize("offload_to_disk", [True, False])
def test_hook_round_trip_on_real_ernie4_5_vl_moe(tmp_path, offload_to_disk):
    model = _tiny_ernie4_5_vl_moe()
    core_model = model.model
    vision_tower = core_model.vision_tower
    resampler_model = core_model.resampler_model
    owned_keys_before = set(model.state_dict().keys())

    definition = _definition(model=model, offload_to_disk=offload_to_disk)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)

    definition.pre_quantize_generate_hook_start()
    definition.pre_quantize_generate_hook_end()

    # The layout resolver and the register names must agree after cleanup; this
    # is the exact contract the input-capture finally-block depends on.
    assert Ernie4_5_VLMoeQModel._resolve_multimodal_layout(model) == ("model", core_model)
    assert core_model.vision_tower is vision_tower
    assert core_model.resampler_model is resampler_model
    # Cleanup only changes storage placement, never checkpoint ownership.
    assert set(model.state_dict().keys()) == owned_keys_before
