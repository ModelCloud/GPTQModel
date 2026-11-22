# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real released owners, including the entire RADIO and Parakeet checkpoint."""

import re

import pytest
import torch
from awq_neutrality import _wrapper, assert_awq_folds_neutral
from torch import nn

from gptqmodel.models.definitions.nemotron_h import NEMOTRON_H_LAYER_MODULES, NemotronHQModel
from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.nemotron_omni.configuration import NemotronOmniConfig
from gptqmodel.models.ports.nemotron_omni.modeling import NemotronH_Omni_Reasoning_V3
from tests.port_assets import read_port_json
from tests.test_nemotron_omni_local_port import PORT, tiny_config, tiny_model


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
    model.gptqmodel_effective_module_tree = tree
    assert NemotronOmniQModel._resolve_effective_module_tree(model, None) == tree
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


def test_ssm_parameters_stay_in_the_in_proj_stage():
    model = tiny_model()
    tree = NemotronOmniQModel._resolve_effective_module_tree(model, None)
    wrapper = _wrapper(NemotronOmniQModel, model, tree)
    blocks = wrapper.full_layer_modules(model.config, is_awq_quantize=True, include_capture_only=True)
    # Untagged, the SSM parameters joined the attention q/k/v stage and hid
    # `v_proj` as the `o_proj` source on attention layers.
    assert ["mixer.q_proj", "mixer.k_proj", "mixer.v_proj"] in blocks
    assert ["mixer.in_proj", "mixer.A_log:!", "mixer.D:!", "mixer.dt_bias:!"] in blocks


@pytest.mark.parametrize("kv", [2, 4])
@pytest.mark.parametrize("latent", [None, 16, 32])
def test_awq_folds_are_neutral_on_every_mixer(kv, latent):
    from test_awq_scaling_groups import _mamba_ssm_cpu_unavailable_reason

    reason = _mamba_ssm_cpu_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    config = tiny_config(kv=kv)
    config.text_config.moe_latent_size = latent
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4123)
        model = NemotronH_Omni_Reasoning_V3(config).eval()
    _convert_model_with_defuser(NemotronOmniQModel, model, cleanup_original=False)
    tree = NemotronOmniQModel._resolve_effective_module_tree(model, None)
    ids = torch.tensor([[1, 4, 7, 9, 3, 5, 6, 8]])

    def forward():
        return model(input_ids=ids, use_cache=False).logits

    layers = model.language_model.model.layers
    # tiny_config layer order: mamba, attention, dense MLP, MoE.
    assert [type(layer.mixer).__name__ for layer in layers] == [
        "NemotronHMamba2Mixer", "NemotronHAttention", "NemotronHMLP", "NemotronHMoE",
    ]
    groups = []
    for layer in layers:
        checked = assert_awq_folds_neutral(NemotronOmniQModel, model, layer, forward, tree=tree)
        groups.append({source: (targets, consumers) for source, targets, consumers in checked})

    mamba, attention, mlp, moe = groups
    # The gated SSM norm and the scan sit between `in_proj` and `out_proj`.
    assert mamba == {"norm": (["mixer.in_proj"], [])}
    # `o_proj` reads the attention output, never the input norm; with GQA
    # (kv=2) `v_proj`'s width differs from `o_proj`'s input and the group is
    # filtered.
    assert attention["norm"] == (["mixer.q_proj", "mixer.k_proj", "mixer.v_proj"], [])
    assert ("mixer.v_proj" in attention) is (kv == 4)
    if kv == 4:
        assert attention["mixer.v_proj"] == (["mixer.o_proj"], [])
    # relu2 is not homogeneous, so `up_proj` never folds into `down_proj`.
    assert mlp == {"norm": (["mixer.up_proj"], [])}
    routed = [f"mixer.experts.{index}.up_proj" for index in range(4)]
    if latent is None:
        assert moe == {"norm": (["mixer.shared_experts.up_proj", *routed], ["mixer.gate"])}
        # The search input is the MoE block input, not a member's own capture,
        # so the processor may score only the experts this sample routes to.
        wrapper = _wrapper(NemotronOmniQModel, model, tree)
        features = {"mixer.shared_experts.up_proj": torch.randn(1, 8, 32)}
        features.update({name: torch.randn(1, 2, 32) for name in routed})
        (node,) = wrapper.awq_get_modules_for_scaling(layers[3], features, {})
        assert node["_input_feature_name"] == "mixer"
    else:
        # Routed experts read `fc1_latent_proj`'s output; the norm feeds the
        # router, the shared expert and `fc1_latent_proj`.
        assert moe == {"norm": (["mixer.shared_experts.up_proj", "mixer.fc1_latent_proj"], ["mixer.gate"])}
