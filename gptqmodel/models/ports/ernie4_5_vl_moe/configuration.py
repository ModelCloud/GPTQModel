# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Translate the pinned Baidu flat schema into the exact native architecture."""

from copy import deepcopy

from transformers import Ernie4_5_VLMoeConfig, Ernie4_5_VLMoeTextConfig


def _legacy_text_config(values):
    text = {key: deepcopy(value) for key, value in values.items() if key in Ernie4_5_VLMoeTextConfig.__dataclass_fields__}
    experts = values.get("moe_num_experts", [64, 64])
    if isinstance(experts, (list, tuple)):
        if len(experts) != 2 or experts[0] != experts[1]:
            raise ValueError("ERNIE VL requires equal text and vision expert counts, as does the upstream checkpoint")
        experts = experts[0]
    text["moe_num_experts"] = experts
    layers = values.get("num_hidden_layers", 28)
    starts = values.get("moe_layer_start_index", [1, 1])
    ends = values.get("moe_layer_end_index", [layers - 1, layers - 1])
    starts = list(starts) if isinstance(starts, (list, tuple)) else [starts] * 2
    ends = list(ends) if isinstance(ends, (list, tuple)) else [ends] * 2
    ends = [layers - 1 if end == -1 else end for end in ends]
    interval = values.get("moe_layer_interval", 1)
    if len(starts) != 2 or len(ends) != 2 or interval < 1:
        raise ValueError("Invalid ERNIE VL modality layer schedule")
    schedules = [[start <= index <= end and (index + 1) % interval == 0 for index in range(layers)]
                 for start, end in zip(starts, ends)]
    if schedules[0] != schedules[1]:
        raise ValueError("Native ERNIE VL requires the same sparse layers for both modalities")
    text["mlp_layer_types"] = ["sparse" if sparse else "dense" for sparse in schedules[0]]
    if "rope_parameters" not in text:
        rope = deepcopy(values.get("rope_scaling") or {})
        rope["rope_type"] = rope.pop("type", rope.get("rope_type", "default"))
        rope["rope_theta"] = values.get("rope_theta", 500000.0)
        text["rope_parameters"] = rope
    return text


class ErnieVLConfig(Ernie4_5_VLMoeConfig):
    """Load JSON only and return a canonical native config (including conversions)."""

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        values, remaining = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        values["model_type"] = Ernie4_5_VLMoeConfig.model_type
        return values, remaining

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        values = deepcopy(config_dict)
        if not values.get("text_config"):
            text = _legacy_text_config(values)
            vision = deepcopy(values.get("vision_config") or {})
            vision["hidden_size"] = vision.get("embed_dim", vision.get("hidden_size", values.get("pixel_hidden_size", 1280)))
            vision.setdefault("intermediate_size", int(vision["hidden_size"] * vision.get("mlp_ratio", 4)))
            vision.setdefault("temporal_merge_size", values.get("temporal_conv_size", 2))
            vision.setdefault("spatial_merge_size", values.get("spatial_conv_size", 2))
            values["text_config"] = text
            values["vision_config"] = vision
            values.setdefault("image_token_id", values.get("im_patch_id", 100295))
            fields = set(Ernie4_5_VLMoeConfig.__dataclass_fields__)
            fields.update(("architectures", "dtype", "_name_or_path"))
            # Released checkpoints spell the dtype field ``torch_dtype``; canonicalize at parse time.
            if "torch_dtype" in values:
                values.setdefault("dtype", values.pop("torch_dtype"))
            values = {key: value for key, value in values.items() if key in fields}
        values["model_type"] = Ernie4_5_VLMoeConfig.model_type
        return Ernie4_5_VLMoeConfig.from_dict(values, **kwargs)
