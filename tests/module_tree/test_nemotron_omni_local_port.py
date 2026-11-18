# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real released owners, including the entire RADIO and Parakeet checkpoint."""

import re

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.nemotron_h import NEMOTRON_H_LAYER_MODULES, NemotronHQModel
from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.nemotron_omni.configuration import NemotronOmniConfig
from gptqmodel.models.ports.nemotron_omni.modeling import NemotronH_Omni_Reasoning_V3
from tests.port_assets import read_port_json
from tests.test_nemotron_omni_local_port import PORT, tiny_model


def source_to_runtime_shapes(source):
    """Independent pinned-source equations, not the port's lazy-load map."""
    result = {}
    for name, metadata in source.items():
        if name.startswith(("vision_model.radio_model.input_conditioner.",
                            "sound_encoder.encoder.feature_extractor.")):
            continue  # Preprocessing is external; preserved by the quantized writer.
        shape = tuple(metadata["shape"])
        name = name.replace("language_model.backbone.", "language_model.model.")
        for old, new in (("mlp1.0.", "multi_modal_projector.layer_norm."),
                         ("mlp1.1.", "multi_modal_projector.linear_1."),
                         ("mlp1.3.", "multi_modal_projector.linear_2."),
                         ("sound_encoder.encoder.", "audio_tower."),
                         ("sound_projection.norm.", "embed_audio.layer_norm."),
                         ("sound_projection.linear1.", "embed_audio.linear_1."),
                         ("sound_projection.linear2.", "embed_audio.linear_2."),
                         ("vision_model.radio_model.model.patch_generator.video_embedder", "vision_model.embeddings.video_patch_projection"),
                         ("vision_model.radio_model.model.patch_generator.embedder", "vision_model.embeddings.patch_projection"),
                         ("vision_model.radio_model.model.patch_generator.pos_embed", "vision_model.embeddings.position_embedding"),
                         ("vision_model.radio_model.model.patch_generator.cls_token.token", "vision_model.embeddings.cls_register_token"),
                         ("vision_model.radio_model.model.blocks.", "vision_model.encoder.layer."),
                         (".attn.proj.", ".attention.o_proj.")):
            name = name.replace(old, new)
        if ".attn.qkv." in name:
            for projection in ("q_proj", "k_proj", "v_proj"):
                result[name.replace(".attn.qkv.", f".attention.{projection}.")] = (shape[0] // 3, *shape[1:])
        else:
            result[name] = shape
    return result


def test_all_released_tensor_shapes_and_defused_owners_match_exactly():
    source = read_port_json(PORT / "checkpoint_inventory.json")
    assert len(source) == 7349
    config = NemotronOmniConfig(**read_port_json(PORT / "source_config.json"))
    with torch.device("meta"):
        model = NemotronH_Omni_Reasoning_V3(config)
        _convert_model_with_defuser(NemotronOmniQModel, model, cleanup_original=False)
    expected = source_to_runtime_shapes(source)
    actual = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    summary_shape = actual.pop("vision_model.summary_idxs")
    assert summary_shape == (2,)
    assert actual == expected


@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_every_live_decoder_linear_and_checkpoint_tensor_has_a_plan(kv, bias):
    model = tiny_model(kv=kv, bias=bias)
    _convert_model_with_defuser(NemotronOmniQModel, model, cleanup_original=False)
    assert NemotronOmniQModel.module_tree[-1] is NEMOTRON_H_LAYER_MODULES
    tree = NemotronOmniQModel._resolve_effective_module_tree(model, None)
    plan = ModulePlan.from_tree(tree)
    assert NemotronOmniQModel.extract_layers_node() == ["language_model.model.layers"]
    assert NemotronOmniQModel.pre_lm_head_norm_module == "language_model.model.norm_f"
    assert NemotronHQModel.extract_layers_node() == ["model.layers"]
    covered = set()
    for index, layer in enumerate(model.language_model.model.layers):
        for name, value in layer.state_dict().items():
            if name.endswith((".weight", ".bias")):
                owner = name.rsplit(".", 1)[0]
            else:
                owner = name
            if owner == "mixer.gate.e_score_correction_bias":
                owner = "mixer.gate"
            spec = plan.find(owner)
            assert spec is not None, (index, name)
            if isinstance(layer.get_submodule(owner) if owner in dict(layer.named_modules()) else None, nn.Linear):
                excluded = owner == "mixer.gate"
                assert spec.quantize is not excluded, owner
                if not excluded:
                    covered.add((index, owner))
            elif re.search(r"mixer\.(A_log|D|dt_bias|norm|conv1d|gate)|^norm", owner):
                assert not spec.quantize, owner
    assert len(covered) == 18
    assert NemotronOmniQModel.fusable is None
    assert NemotronOmniQModel.awq_get_modules_for_scaling(None, None, None, None) == []
