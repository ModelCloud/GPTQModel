# SPDX-FileCopyrightText: 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: MIT
"""OpenVLA vision-language-action model.

Adapted from github.com/openvla/openvla@c8f03f48af692657d3060c19588038c7220e9af9
(prismatic/extern/hf) for checkpoint openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig


class _ScaleFactor(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.scale_factor = source.gamma
        self.inplace = source.inplace

    def forward(self, inputs):
        return inputs.mul_(self.scale_factor) if self.inplace else inputs * self.scale_factor


class PrismaticVisionBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        try:
            import timm
            from timm.layers import LayerScale
        except ImportError as exc:
            raise ImportError(
                "OpenVLA vision requires timm; install with `pip install 'gptqmodel[vision]'` "
                "using a torchvision build compatible with your PyTorch installation."
            ) from exc

        self.use_fused_vision_backbone = config.use_fused_vision_backbone
        self.embed_dim = 0
        for i, name in enumerate(("featurizer", "fused_featurizer")[:len(config.timm_model_ids)]):
            model = timm.create_model(
                config.timm_model_ids[i], pretrained=False, num_classes=0,
                img_size=config.image_sizes[i], act_layer=config.timm_override_act_layers[i],
                **config.vision_backbone_kwargs[i],
            )
            # The released checkpoint calls LayerScale's parameter scale_factor, not gamma.
            for module in list(model.modules()):
                for child_name, child in list(module.named_children()):
                    if isinstance(child, LayerScale):
                        setattr(module, child_name, _ScaleFactor(child))
            setattr(self, name, model)
            self.embed_dim += model.embed_dim

    @staticmethod
    def _patches(model, pixels):
        # timm 1.x returns a list; the original tuple-only unwrapping is insufficient.
        return model.get_intermediate_layers(pixels, n={len(model.blocks) - 2})[0]

    def forward(self, pixel_values):
        channels = 6 if self.use_fused_vision_backbone else 3
        if pixel_values.ndim != 4 or pixel_values.shape[1] != channels:
            raise ValueError(f"OpenVLA expects BCHW images with {channels} channels")
        if not self.use_fused_vision_backbone:
            return self._patches(self.featurizer, pixel_values)
        first, second = pixel_values.split(3, dim=1)
        return torch.cat((self._patches(self.featurizer, first),
                          self._patches(self.fused_featurizer, second)), dim=-1)


class PrismaticProjector(nn.Module):
    def __init__(self, fused, vision_dim, llm_dim):
        super().__init__()
        self.use_fused_vision_backbone = fused
        self.fc1 = nn.Linear(vision_dim, 4 * vision_dim if fused else llm_dim, bias=True)
        self.fc2 = nn.Linear(4 * vision_dim if fused else llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        if fused:
            self.fc3 = nn.Linear(llm_dim, llm_dim, bias=True)
            self.act_fn2 = nn.GELU()

    def forward(self, patches):
        hidden = self.fc2(self.act_fn1(self.fc1(patches)))
        return self.fc3(self.act_fn2(hidden)) if self.use_fused_vision_backbone else hidden


@dataclass
class PrismaticCausalLMOutputWithPast(CausalLMOutputWithPast):
    projector_features: torch.Tensor | None = None


class PrismaticForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = PrismaticConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    _supports_sdpa = True
    _no_split_modules = ["PrismaticProjector", "LlamaDecoderLayer", "MistralDecoderLayer", "PhiDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def __init__(self, config):
        super().__init__(config)
        from transformers import LlamaForCausalLM, MistralForCausalLM, PhiForCausalLM

        self.vision_backbone = PrismaticVisionBackbone(config)
        self.projector = PrismaticProjector(config.use_fused_vision_backbone,
                                            self.vision_backbone.embed_dim, config.text_config.hidden_size)
        classes = {"llama": LlamaForCausalLM, "mistral": MistralForCausalLM, "phi": PhiForCausalLM}
        self.language_model = classes[config.text_config.model_type]._from_config(
            config.text_config, attn_implementation=config._attn_implementation,
        )
        if config.text_config.model_type == "phi":
            attention = ("q_proj:0", "k_proj:0", "v_proj:0", "dense:1")
            if config.text_config.qk_layernorm:
                attention += ("q_layernorm:!", "k_layernorm:!")
            self.gptqmodel_effective_module_tree = [
                "language_model", "model", "layers", "#", {
                    "input_layernorm": ("input_layernorm:!",),
                    "self_attn": attention,
                    "mlp": ("fc1:0", "fc2:1"),
                },
            ]
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True, **kwargs):
        embedding = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of, mean_resizing=mean_resizing, **kwargs,
        )
        self.config.text_config.vocab_size = embedding.num_embeddings
        self.vocab_size = embedding.num_embeddings
        return embedding

    @staticmethod
    def _insert(values, patches):
        return torch.cat((values[:, :1], patches, values[:, 1:]), dim=1)

    def _multimodal_inputs(self, input_ids, inputs_embeds, pixel_values, attention_mask, labels, position_ids):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("OpenVLA requires exactly one of input_ids and inputs_embeds")
        embeddings = self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        features = self.projector(self.vision_backbone(pixel_values))
        if embeddings.shape[0] != features.shape[0]:
            raise ValueError("OpenVLA requires the same number of images and text rows")
        if attention_mask is not None:
            if attention_mask.shape != embeddings.shape[:2]:
                raise ValueError("OpenVLA prefill attention mask must match the text tokens")
            attention_mask = self._insert(attention_mask, attention_mask.new_ones(features.shape[:2]))
        if labels is not None:
            labels = self._insert(labels, labels.new_full(features.shape[:2], -100))
        if position_ids is not None:
            patches = position_ids[:, :1] + torch.arange(1, features.shape[1] + 1, device=position_ids.device)
            position_ids = torch.cat((position_ids[:, :1], patches, position_ids[:, 1:] + features.shape[1]), 1)
        return self._insert(embeddings, features), attention_mask, labels, position_ids, features

    @staticmethod
    def _cached_mask(attention_mask, past_key_values, length):
        if attention_mask is None:
            return None
        expected = past_key_values.get_seq_length() + length
        if attention_mask.shape[1] == expected:
            return attention_mask
        patches = getattr(past_key_values, "prismatic_patch_count", 0)
        if patches and attention_mask.shape[1] + patches == expected:
            return PrismaticForConditionalGeneration._insert(
                attention_mask, attention_mask.new_ones((attention_mask.shape[0], patches)),
            )
        raise ValueError("OpenVLA cached attention mask does not match the cached sequence")

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None,
                inputs_embeds=None, past_key_values=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, output_projector_features=None, return_dict=None,
                position_ids=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("OpenVLA requires exactly one of input_ids and inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise ValueError("OpenVLA currently supports native DynamicCache only")
        cached = past_key_values is not None and past_key_values.get_seq_length() > 0
        features = None
        if cached:
            length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            attention_mask = self._cached_mask(attention_mask, past_key_values, length)
        elif pixel_values is not None:
            inputs_embeds, attention_mask, labels, position_ids, features = self._multimodal_inputs(
                input_ids, inputs_embeds, pixel_values, attention_mask, labels, position_ids,
            )
            input_ids = None
        result = self.language_model(
            input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            labels=labels, past_key_values=past_key_values, position_ids=position_ids,
            use_cache=(self.config.text_config.use_cache if use_cache is None else use_cache) and not self.training,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True,
            **kwargs,
        )
        if features is not None and result.past_key_values is not None:
            result.past_key_values.prismatic_patch_count = features.shape[1]
        output = PrismaticCausalLMOutputWithPast(
            loss=result.loss, logits=result.logits, past_key_values=result.past_key_values,
            hidden_states=result.hidden_states, attentions=result.attentions,
            projector_features=features if output_projector_features else None,
        )
        return output if (self.config.return_dict if return_dict is None else return_dict) else output.to_tuple()

    @torch.no_grad()
    def generate(self, input_ids=None, pixel_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        past = kwargs.get("past_key_values")
        patches = getattr(past, "prismatic_patch_count", 0)
        if past is not None and not isinstance(past, DynamicCache):
            raise ValueError("OpenVLA currently supports native DynamicCache only")
        if past is not None and past.get_seq_length():
            length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            attention_mask = self._cached_mask(attention_mask, past, length)
        elif pixel_values is not None:
            if attention_mask is None:
                shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                attention_mask = torch.ones(shape, dtype=torch.long, device=device)
            inputs_embeds, attention_mask, _, _, features = self._multimodal_inputs(
                input_ids, inputs_embeds, pixel_values, attention_mask, None, None,
            )
            patches = features.shape[1]
        # Native generation owns positions, image-expanded masks and per-request caches.
        generation_config = deepcopy(kwargs.pop("generation_config", None) or self.generation_config)
        kwargs = generation_config.update(**kwargs)
        result = self.language_model.generate(
            input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            generation_config=generation_config, **kwargs,
        )
        cache = getattr(result, "past_key_values", past)
        if cache is not None and patches:
            cache.prismatic_patch_count = patches
        return result


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class = OpenVLAConfig

    def __init__(self, config):
        super().__init__(config)
        self.norm_stats = config.norm_stats
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        self.vocab_size = config.text_config.vocab_size - config.pad_to_multiple_of

    def _check_unnorm_key(self, unnorm_key):
        if not self.norm_stats:
            raise ValueError("OpenVLA checkpoint has no action normalization statistics")
        if unnorm_key is None:
            if len(self.norm_stats) != 1:
                raise ValueError(f"Choose an unnorm_key from {list(self.norm_stats)}")
            unnorm_key = next(iter(self.norm_stats))
        if unnorm_key not in self.norm_stats:
            raise ValueError(f"Unknown unnorm_key {unnorm_key!r}; choose from {list(self.norm_stats)}")
        return unnorm_key

    def get_action_stats(self, unnorm_key=None):
        return self.norm_stats[self._check_unnorm_key(unnorm_key)]["action"]

    def get_action_dim(self, unnorm_key=None):
        return len(self.get_action_stats(unnorm_key)["q01"])

    @torch.no_grad()
    def predict_action(self, input_ids=None, unnorm_key=None, **kwargs):
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("OpenVLA predict_action requires one nonempty token row")
        stats = self.get_action_stats(unnorm_key)
        low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
        if low.ndim != 1 or not low.size or high.shape != low.shape or mask.shape != low.shape:
            raise ValueError("OpenVLA action normalization statistics have incompatible shapes")
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high < low):
            raise ValueError("OpenVLA action normalization statistics must be finite and ordered")
        if self.config.text_config.vocab_size <= 29871:
            raise ValueError("OpenVLA action prompts require Llama's empty token 29871")
        if input_ids[0, -1] != 29871:
            input_ids = torch.cat((input_ids, input_ids.new_full((1, 1), 29871)), dim=1)
            if kwargs.get("attention_mask") is not None:
                old_mask = kwargs["attention_mask"]
                kwargs["attention_mask"] = torch.cat((old_mask, old_mask.new_ones((1, 1))), dim=1)
        generated = self.generate(input_ids, max_new_tokens=low.size, **kwargs)
        if not torch.is_tensor(generated) or generated.shape[1] != input_ids.shape[1] + low.size:
            raise ValueError("OpenVLA generation ended before producing a complete action")
        tokens = generated[0, -low.size:].cpu().numpy()
        vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of
        indices = np.clip(vocab_size - tokens - 1, 0, len(self.bin_centers) - 1)
        normalized = self.bin_centers[indices]
        return np.where(mask, (normalized + 1) * (high - low) / 2 + low, normalized)
