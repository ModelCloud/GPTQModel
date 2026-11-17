# Copyright 2024 AIDC-AI
# SPDX-License-Identifier: Apache-2.0
"""Ovis 1.x visual-vocabulary wrapper adapted to native Transformers 5 owners."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    CLIPVisionModel,
    Gemma2ForCausalLM,
    LlamaForCausalLM,
    PreTrainedModel,
    Qwen2ForCausalLM,
    SiglipVisionModel,
)
from transformers.generation import GenerationConfig, GenerationMixin

from .configuration_ovis import IGNORE_ID, IMAGE_ATOM_ID, IMAGE_INDICATOR_IDS, IMAGE_TOKEN_ID, OvisConfig
from .processing_ovis import (
    ConversationFormatter,
    OvisProcessor,
    image_placeholders,
    image_processor_for_config,
    process_image,
)


class VisualEmbedding(nn.Embedding):
    def forward(self, tokens):
        return super().forward(tokens) if not tokens.is_floating_point() else tokens @ self.weight


class VisualTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ovis_config = config
        self.config = config.visual_tokenizer_config
        visual = self.config
        self.image_processor = image_processor_for_config(config)
        self.backbone = nn.Module()
        cls = CLIPVisionModel if visual.model_type == "clip_visual_tokenizer" else SiglipVisionModel
        # Transformers 5 flattened these native classes; retain the released prefix.
        self.backbone.vision_model = cls(visual.backbone_config)
        reserve = 5 if config.ovis_version == "1.6" else (2 if visual.use_indicators else 0)
        self.head = nn.Sequential(nn.Linear(visual.backbone_config.hidden_size * visual.hidden_stride**2 *
                                            (2 if visual.hd_booster else 1), visual.vocab_size - reserve, bias=False),
                                  nn.LayerNorm(visual.vocab_size - reserve))

    @property
    def dtype(self):
        return self.head[0].weight.dtype

    @property
    def device(self):
        return self.head[0].weight.device

    def get_backbone(self):
        return self.backbone

    def get_image_processor(self):
        return self.image_processor

    def get_image_size(self):
        return (self.config.backbone_config.image_size,) * 2

    def get_head(self):
        return self.head

    construct_image_placeholders = staticmethod(image_placeholders)

    def preprocess_image(self, image, **kwargs):
        pixels, placeholders = process_image(image, self.image_processor, self.ovis_config, **kwargs)
        return (pixels, placeholders) if self.ovis_config.ovis_version == "1.6" else pixels

    def mock_input(self):
        size = self.config.backbone_config.image_size
        channels = 15 if self.config.hd_booster else 3
        return torch.zeros(1, channels, size, size), image_placeholders((1, 1)) if self.ovis_config.ovis_version == "1.6" else [IMAGE_TOKEN_ID]

    def get_zero_pixel_values(self, n=1):
        return self.mock_input()[0].expand(n, -1, -1, -1).clone()

    def tokenize(self, logits):
        if self.config.tokenize_function == "gumbel_argmax":
            return F.gumbel_softmax(logits, tau=self.config.tau, hard=True)
        probs = logits.softmax(-1)
        if self.config.tokenize_function == "st_argmax":
            hard = F.one_hot(probs.argmax(-1), probs.shape[-1]).to(probs)
            return hard - probs.detach() + probs
        return probs

    def encode(self, pixels):
        visual = self.config
        batch = pixels.shape[0]
        adaptive_mask = None
        if visual.hd_booster:
            if pixels.shape[1] != 15:
                raise ValueError("Ovis S2 expects overall + four crops in 15 channels")
            if visual.hd_booster == "s2wrapper-adaptive":
                adaptive_mask = torch.isinf(pixels[:, -1, -1, -1])
                pixels = pixels.masked_fill(torch.isinf(pixels), 0)
            pixels = pixels.reshape(batch * 5, 3, *pixels.shape[-2:])
        output = self.backbone.vision_model(pixels, output_hidden_states=True, return_dict=True)
        features = output.last_hidden_state if self.ovis_config.ovis_version == "1.0" else output.hidden_states[-1]
        if visual.drop_cls_token:
            features = features[:, 1:]
        if visual.hd_booster:
            length, width = features.shape[1:]
            side = math.isqrt(length)
            if side**2 != length:
                raise ValueError("Ovis S2 requires square patch features")
            features = features.reshape(batch, 5, length, width)
            parts = features[:, 1:].reshape(batch, 2, 2, side, side, width)
            parts = parts.permute(0, 1, 3, 2, 4, 5).reshape(batch, 2 * side, 2 * side, width)
            pooled = F.interpolate(parts.permute(0, 3, 1, 2).float(), size=side, mode="area").flatten(2).transpose(1, 2).to(features.dtype)
            if adaptive_mask is not None:
                pooled = pooled.masked_fill(adaptive_mask[:, None, None], 0)
            features = torch.cat((features[:, 0], pooled), dim=-1)
        stride = visual.hidden_stride
        if stride > 1:
            batch, length, width = features.shape
            side = math.isqrt(length)
            if side**2 != length:
                raise ValueError("Ovis hidden stride requires square patch features")
            features = features.reshape(batch, side, side, width)
            padding = -side % stride
            if padding and self.ovis_config.ovis_version != "1.6":
                raise ValueError("Legacy Ovis stride requires divisible patch grid")
            features = F.pad(features, (0, 0, 0, padding, 0, padding))
            side += padding
            features = features.reshape(batch, side // stride, stride, side // stride, stride, width)
            features = features.permute(0, 1, 3, 2, 4, 5).reshape(batch, -1, stride * stride * width)
        return features

    def forward(self, pixel_values):
        tokens = self.tokenize(self.head(self.encode(pixel_values)))
        if self.ovis_config.ovis_version == "1.6":
            return F.pad(tokens, (0, 5))
        if not self.config.use_indicators:
            return tokens
        tokens = F.pad(tokens, (0, 2))
        begin = tokens.new_zeros(tokens.shape[0], 1, self.config.vocab_size)
        end = torch.zeros_like(begin)
        begin[..., -2], end[..., -1] = 1, 1
        return torch.cat((begin, tokens, end), dim=1)


class Ovis(PreTrainedModel, GenerationMixin):
    config_class = OvisConfig
    base_model_prefix = "ovis"
    main_input_name = "input_ids"
    _no_split_modules = ["LlamaDecoderLayer", "Qwen2DecoderLayer", "Gemma2DecoderLayer", "SiglipEncoderLayer", "CLIPEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _tied_weights_keys = {"llm.lm_head.weight": "llm.model.embed_tokens.weight"}

    @property
    def _supports_sdpa(self):
        return self.config.llm_config.model_type != "gemma2"

    def __init__(self, config, **kwargs):
        super().__init__(config)
        text = config.llm_config
        if config.disable_tie_weight:
            text.tie_word_embeddings = False
        # A serialized FA2 preference is not an instruction to import optional CUDA code.
        selected = config._attn_implementation
        text._attn_implementation = selected or "eager"
        config.visual_tokenizer_config.backbone_config._attn_implementation = selected or "eager"
        self.llm = {"llama": LlamaForCausalLM, "qwen2": Qwen2ForCausalLM, "gemma2": Gemma2ForCausalLM}[text.model_type](text)
        self.visual_tokenizer = VisualTokenizer(config)
        self.vte = VisualEmbedding(config.visual_tokenizer_config.vocab_size, config.hidden_size)
        self.text_tokenizer = None
        self.processor = None
        self.generation_config = GenerationConfig.from_model_config(text)
        self.post_init()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.llm.set_output_embeddings(value)

    def get_text_tokenizer(self):
        return self.text_tokenizer if self.text_tokenizer is not None else getattr(self, "tokenizer", None)

    def get_visual_tokenizer(self):
        return self.visual_tokenizer

    def get_llm(self):
        return self.llm

    def get_vte(self):
        return self.vte

    def get_wte(self):
        return self.get_input_embeddings()

    def get_conversation_formatter(self):
        tokenizer = self.get_text_tokenizer()
        if tokenizer is None:
            raise ValueError("Ovis preprocessing requires the caller's normalized text tokenizer")
        return ConversationFormatter(tokenizer, self.config.conversation_formatter_class, self.config.ovis_version)

    def preprocess_inputs(self, *args, **kwargs):
        processor = self.processor
        if processor is None:
            processor = OvisProcessor(self.visual_tokenizer.image_processor, self.get_conversation_formatter().tokenizer, self.config)
        return processor.preprocess_inputs(*args, **kwargs)

    def pad_truncate_sequence(self, sequences, batch_first=True, padding_value=0.0, left_padding=False):
        if left_padding:
            return nn.utils.rnn.pad_sequence([x.flip(0) for x in sequences], batch_first=True,
                                             padding_value=padding_value).flip(1)[:, -self.config.multimodal_max_length:]
        return nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first,
                                         padding_value=padding_value)[:, :self.config.multimodal_max_length]

    def _raw_text_embeddings(self, ids):
        embedding = self.get_wte()
        return F.embedding(ids, embedding.weight, padding_idx=embedding.padding_idx)

    def _decoder_embeddings(self, embeds):
        # TF4 Gemma2 scaled *after* merging; TF5 moved that scale into embed_tokens.
        if self.config.llm_config.model_type == "gemma2":
            embeds = embeds * embeds.new_tensor(self.config.hidden_size**0.5)
        return embeds

    def merge_multimodal(self, text_input_ids, text_attention_masks=None, text_labels=None, pixel_values=None,
                         left_padding=False, with_kv_cache=False):
        if text_input_ids.ndim != 2:
            raise ValueError("Ovis input_ids must be batched")
        if text_attention_masks is None:
            text_attention_masks = torch.ones_like(text_input_ids, dtype=torch.bool)
        if with_kv_cache:
            return None, self._decoder_embeddings(self._raw_text_embeddings(text_input_ids)), text_labels, text_attention_masks
        if pixel_values is None:
            pixel_values = [None] * text_input_ids.shape[0]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = list(pixel_values) if pixel_values.ndim == 5 else [pixel_values]
        if len(pixel_values) != text_input_ids.shape[0]:
            raise ValueError("Ovis requires one pixel_values group per text row")
        counts = [0 if pixels is None else pixels.shape[0] for pixels in pixel_values]
        visual_embeds = [None] * len(counts)
        visual_ids = [None] * len(counts)
        if sum(counts):
            pixels = torch.cat([value for value in pixel_values if value is not None]).to(
                device=self.visual_tokenizer.device, dtype=self.visual_tokenizer.dtype)
            tokens = self.visual_tokenizer(pixels)
            visual_embeds = self.vte(tokens.to(self.vte.weight)).to(self.get_wte().weight)
            visual_embeds = visual_embeds.split(counts)
            visual_ids = tokens.argmax(-1).split(counts)
        if text_labels is None:
            text_labels = torch.full_like(text_input_ids, IGNORE_ID)
        embeddings, masks, labels = [], [], []
        atom = IMAGE_ATOM_ID if self.config.ovis_version == "1.6" else IMAGE_TOKEN_ID
        allowed = [atom, *IMAGE_INDICATOR_IDS] if self.config.ovis_version == "1.6" else [atom]
        for row, targets, mask, images, count in zip(text_input_ids, text_labels, text_attention_masks, visual_embeds, counts):
            if ((row < 0) & ~torch.isin(row, row.new_tensor(allowed))).any():
                raise ValueError("Unknown Ovis negative placeholder")
            positions = (row == atom).nonzero().flatten().tolist()
            if len(positions) != count and positions:
                raise ValueError("Ovis image atoms and visual crops must match")
            text = self._raw_text_embeddings(row.clamp_min(0))
            if self.config.ovis_version == "1.6":
                for index, token in enumerate(IMAGE_INDICATOR_IDS):
                    text[row == token] = self.vte.weight[-5 + index].to(text)
            parts, target_parts, mask_parts = [], [], []
            start = 0
            for index, position in enumerate(positions):
                value = images[index].to(text)
                parts.extend((text[start:position], value))
                target_parts.extend((targets[start:position], targets.new_full((len(value),), IGNORE_ID)))
                mask_parts.extend((mask[start:position], mask.new_ones(len(value))))
                start = position + 1
            parts.append(text[start:])
            target_parts.append(targets[start:])
            mask_parts.append(mask[start:])
            embeddings.append(torch.cat(parts))
            labels.append(torch.cat(target_parts))
            masks.append(torch.cat(mask_parts))
        return (visual_ids, self._decoder_embeddings(self.pad_truncate_sequence(embeddings, left_padding=left_padding)),
                self.pad_truncate_sequence(labels, padding_value=IGNORE_ID, left_padding=left_padding),
                self.pad_truncate_sequence(masks, padding_value=False, left_padding=left_padding))

    def forward(self, input_ids=None, attention_mask=None, labels=None, pixel_values=None, inputs_embeds=None,
                past_key_values=None, **kwargs):
        if inputs_embeds is None:
            cached = past_key_values is not None and past_key_values.get_seq_length() > 0
            _, inputs_embeds, merged_labels, attention_mask = self.merge_multimodal(
                input_ids, attention_mask, labels, pixel_values, with_kv_cache=cached)
            labels = merged_labels if labels is not None else None
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels,
                        past_key_values=past_key_values, **kwargs)

    @torch.no_grad()
    def generate(self, inputs=None, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
        inputs = input_ids if inputs is None else inputs
        _, embeds, _, mask = self.merge_multimodal(inputs, attention_mask, None, pixel_values, left_padding=True)
        generation_config = kwargs.pop("generation_config", self.generation_config)
        generation_config = GenerationConfig.from_dict(generation_config.to_dict())
        # HybridCache was removed in TF5; native Gemma2 DynamicCache owns both layer types.
        if generation_config.cache_implementation == "hybrid":
            generation_config.cache_implementation = None
        if kwargs.get("cache_implementation") == "hybrid":
            kwargs["cache_implementation"] = None
        return self.llm.generate(inputs_embeds=embeds, attention_mask=mask, generation_config=generation_config, **kwargs)

    def save_pretrained(self, path, **kwargs):
        if kwargs.pop("safe_serialization", True) is False:
            raise ValueError("Ovis publishes safetensors payloads only")
        result = super().save_pretrained(path, **kwargs)
        self.visual_tokenizer.image_processor.save_pretrained(path)
        tokenizer = self.get_text_tokenizer()
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        return result
