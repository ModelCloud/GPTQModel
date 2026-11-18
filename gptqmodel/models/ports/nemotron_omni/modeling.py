# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
"""Native parameter owners with source-compatible multimodal merging.

The native 5.23 video forward flattens the temporal grid before EVS and
passes a Tensor to torch.cat. Encode and prune once, before LM generation.
"""

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3 as NativeOmni,
)
from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import compute_retention_mask
from transformers.utils import can_return_tuple

from .configuration import NemotronOmniConfig


class NemotronH_Omni_Reasoning_V3(NativeOmni):
    config_class = NemotronOmniConfig

    def __init__(self, config):
        super().__init__(config)
        # Native RADIO introduced this deterministic, tiny metadata buffer.
        # Keep it real even when the large parameter owners are born on meta.
        self.vision_model.summary_idxs = torch.tensor(config.vision_config.summary_idxs, device="cpu")
        if config.legacy_radio_layout:
            if config.vision_config.layerscale_value != 1.0:
                raise ValueError("Legacy RADIO checkpoints have no learned LayerScale")
            for layer in self.vision_model.encoder.layer:
                layer.layer_scale1 = torch.nn.Identity()
                layer.layer_scale2 = torch.nn.Identity()

    @classmethod
    def is_custom_code(cls):
        # Native HF checkpoint conversions also apply to this metadata adapter.
        return False

    def pixel_shuffle(self, vision_features, scale_factor=0.5):
        shuffled = super().pixel_shuffle(vision_features, scale_factor)
        return shuffled.transpose(1, 2).contiguous() if self.config.ps_version == "v1" else shuffled

    @can_return_tuple
    def get_image_features(self, pixel_values, image_grid_hw, **kwargs):
        if int(image_grid_hw.prod(-1).sum()) != pixel_values.shape[0]:
            raise ValueError("Image patch count does not match image_grid_hw")
        outputs = self.vision_model(pixel_values.to(self.vision_model.dtype), image_grid_hw=image_grid_hw,
                                    return_dict=True, **kwargs)
        features = outputs.features.split(image_grid_hw.prod(-1).tolist())
        projected = tuple(self.project_vision_features(x.unsqueeze(0), h, w).squeeze(0)
                          for x, (h, w) in zip(features, image_grid_hw.tolist()))
        return BaseModelOutputWithPooling(last_hidden_state=outputs.last_hidden_state, pooler_output=projected,
                                          hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    @can_return_tuple
    def get_video_features(self, pixel_values_videos, **kwargs):
        frames = pixel_values_videos.to(self.vision_model.dtype)
        temporal = self.video_temporal_patch_dim
        if frames.ndim != 4 or frames.shape[0] == 0:
            raise ValueError("Video must contain nonempty (frames, channels, height, width)")
        if frames.shape[0] % temporal:
            frames = torch.cat((frames, frames[-1:].expand(temporal - frames.shape[0] % temporal, -1, -1, -1)))
        count, channels, height, width = frames.shape
        outputs = self.vision_model(frames.reshape(count // temporal, temporal * channels, height, width),
                                    return_dict=True, **kwargs)
        patch = self.vision_model.patch_size
        features = self.project_vision_features(outputs.features, height // patch, width // patch)
        return BaseModelOutputWithPooling(last_hidden_state=outputs.last_hidden_state, pooler_output=features,
                                          hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def _merge_media(self, input_ids, inputs_embeds, *, pixel_values=None, image_grid_hw=None,
                     pixel_values_videos=None, input_features=None, input_features_mask=None,
                     attention_mask=None, position_ids=None, labels=None):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("Image and video placeholders must be supplied in separate calls")
        if pixel_values is not None:
            if image_grid_hw is None:
                raise ValueError("image_grid_hw is required with packed image patches")
            features = torch.cat(self.get_image_features(pixel_values, image_grid_hw, return_dict=True).pooler_output)
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, features, self.image_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(mask, features.to(inputs_embeds))
        if pixel_values_videos is not None:
            if input_ids is None or input_ids.shape[0] != 1:
                raise ValueError("Nemotron Omni video requires one input_ids row")
            features = self.get_video_features(pixel_values_videos, return_dict=True).pooler_output
            if self.video_pruning_rate > 0:
                merge = round(1 / self.config.downsample_ratio)
                patch = self.vision_model.patch_size
                height = pixel_values_videos.shape[-2] // patch // merge
                width = pixel_values_videos.shape[-1] // patch // merge
                keep = compute_retention_mask(video_embeds=features, thw=(features.shape[0], height, width),
                                              spatial_merge_size=1, q=self.video_pruning_rate)
                placeholders = input_ids == self.image_token_id
                if int(placeholders.sum()) != features.shape[0] * features.shape[1]:
                    raise ValueError("Video placeholder count does not match temporal features")
                retention = torch.ones_like(input_ids, dtype=torch.bool)
                retention[placeholders] = keep
                input_ids = input_ids[retention].unsqueeze(0)
                inputs_embeds = inputs_embeds[retention].unsqueeze(0)
                if attention_mask is not None:
                    attention_mask = attention_mask[retention].unsqueeze(0)
                if labels is not None:
                    labels = labels[retention].unsqueeze(0)
                if position_ids is not None:
                    position_ids = position_ids[retention].unsqueeze(0)
                features = features.reshape(-1, features.shape[-1])[keep]
            else:
                features = features.reshape(-1, features.shape[-1])
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, features, self.image_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(mask, features.to(inputs_embeds))
        if input_features is not None:
            if self.audio_tower is None:
                raise ValueError("This Nemotron Omni configuration has no audio tower")
            features = self.get_audio_features(
                input_features.to(self.audio_tower.dtype), input_features_mask, return_dict=True
            ).pooler_output
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, features, self.audio_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(mask, features.to(inputs_embeds))
        return input_ids, inputs_embeds, attention_mask, position_ids, labels

    @can_return_tuple
    def forward(self, input_ids=None, pixel_values=None, image_grid_hw=None, pixel_values_videos=None,
                input_features=None, input_features_mask=None, attention_mask=None, position_ids=None,
                past_key_values=None, labels=None, inputs_embeds=None, use_cache=None, **kwargs):
        _, inputs_embeds, attention_mask, position_ids, labels = self._merge_media(
            input_ids, inputs_embeds, pixel_values=pixel_values, image_grid_hw=image_grid_hw,
            pixel_values_videos=pixel_values_videos, input_features=input_features,
            input_features_mask=input_features_mask, attention_mask=attention_mask,
            position_ids=position_ids, labels=labels,
        )
        return self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                   position_ids=position_ids, past_key_values=past_key_values,
                                   labels=labels, use_cache=use_cache, return_dict=True, **kwargs)

    @torch.no_grad()
    def generate(self, input_ids=None, *, inputs_embeds=None, attention_mask=None, position_ids=None,
                 pixel_values=None, image_grid_hw=None, pixel_values_videos=None,
                 input_features=None, input_features_mask=None, **kwargs):
        input_ids, inputs_embeds, attention_mask, position_ids, _ = self._merge_media(
            input_ids, inputs_embeds, pixel_values=pixel_values, image_grid_hw=image_grid_hw,
            pixel_values_videos=pixel_values_videos, input_features=input_features,
            input_features_mask=input_features_mask, attention_mask=attention_mask, position_ids=position_ids,
        )
        return super().generate(input_ids=input_ids, inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask, position_ids=position_ids, **kwargs)

    def save_pretrained(self, *args, save_original_format=False, **kwargs):
        return super().save_pretrained(*args, save_original_format=save_original_format, **kwargs)
