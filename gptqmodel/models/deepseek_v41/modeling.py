# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.generic import can_return_tuple

from .attention import DeepSeekV41FlashCache, V41Attention, V41AttentionState
from .configuration import DeepSeekV41FlashConfig, attention_schedule
from .layers import V41HyperConnection, V41MoE, V41RMSNorm, collapse_streams, expand_streams
from .vision import V41Aligner, V41VisionTower


class V41DecoderLayer(nn.Module):
    def __init__(self, config, dtype, layer_id, schedule, *, vision):
        super().__init__()
        self.attn_norm = V41RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.ffn_norm = V41RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.attn_hc = V41HyperConnection(config)
        self.ffn_hc = V41HyperConnection(config)
        self.attn = V41Attention(config, dtype, layer_id, schedule)
        self.ffn = V41MoE(config, dtype, vision=vision)

    def forward(self, hidden, pre_mix, positions, history, shared, image_mask):
        next_pre, post, mix = self.attn_hc(hidden)
        attn_input = self.attn_norm(collapse_streams(hidden, pre_mix))
        hidden = expand_streams(self.attn(attn_input, positions, history, shared), hidden, post, mix)

        pre_mix, post, mix = self.ffn_hc(hidden)
        ffn_input = self.ffn_norm(collapse_streams(hidden, next_pre))
        return expand_streams(self.ffn(ffn_input, image_mask), hidden, post, mix), pre_mix


class DeepSeekV41FlashForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = DeepSeekV41FlashConfig
    _no_split_modules = ["V41DecoderLayer", "V41VisionTower", "V41Aligner"]
    _is_stateful = True

    def __init__(self, config, *, dtype=None):
        super().__init__(config)
        text = config.text_config
        if text is None:
            raise ValueError("DeepSeek V4.1 Flash requires text_config")
        dtype = dtype or config.dtype or torch.bfloat16
        if (
            text.num_key_value_heads != 1
            or text.sliding_window <= 0
            or not 0 < text.qk_rope_head_dim <= min(text.head_dim, text.index_head_dim)
            or text.qk_rope_head_dim % 2
        ):
            raise ValueError("Invalid DeepSeek V4.1 attention geometry")
        schedule = attention_schedule(text)
        vision = config.vision_config is not None and config.vision_config.num_hidden_layers > 0
        self.embed = nn.Embedding(text.vocab_size, text.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            V41DecoderLayer(text, dtype, index, entry, vision=vision) for index, entry in enumerate(schedule)
        )
        self.norm = V41RMSNorm(text.hidden_size, text.rms_norm_eps, dtype)
        self.head = nn.Linear(text.hidden_size, text.vocab_size, bias=False, dtype=torch.float32)
        self.vision = V41VisionTower(config.vision_config, dtype) if vision else None
        if vision:
            self.aligner = V41Aligner(config.vision_config, text.hidden_size, dtype)
            self.image_start = nn.Parameter(torch.empty(text.hidden_size, dtype=dtype))
            self.image_end = nn.Parameter(torch.empty(text.hidden_size, dtype=dtype))
            self.image_newline = nn.Parameter(torch.empty(text.hidden_size, dtype=dtype))

    @property
    def device(self):
        return self.embed.weight.device

    @property
    def dtype(self):
        return self.embed.weight.dtype

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    @classmethod
    def _supports_default_dynamic_cache(cls):
        # Hugging Face generation lets forward create and return the architecture's cache.
        return False

    def get_image_features(self, pixel_values, image_grid_thw):
        if self.vision is None or image_grid_thw is None:
            raise ValueError("DeepSeek V4.1 images require a vision checkpoint and image_grid_thw")
        grids = image_grid_thw.tolist()
        if not grids or any(frames != 1 or height <= 0 or width <= 0 for frames, height, width in grids):
            raise ValueError("DeepSeek V4.1 image grids must have shape (images, 3) with one frame per image")
        result = []
        sizes = [height * width for _, height, width in grids]
        for patches, (_, height, width) in zip(pixel_values.split(sizes), grids):
            features = self.aligner(self.vision(patches, height, width), height, width)
            endings = self.image_newline.to(features).expand(features.size(0), 1, -1)
            rows = torch.cat((features, endings), dim=1).flatten(0, 1)
            result.append(torch.cat((self.image_start.to(rows)[None], rows, self.image_end.to(rows)[None])))
        return result

    @can_return_tuple
    @torch.no_grad()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        pixel_values=None,
        image_grid_thw=None,
        logits_to_keep=0,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids and inputs_embeds")
        if attention_mask is not None and (attention_mask.ndim != 2 or not attention_mask.bool().all()):
            raise ValueError("DeepSeek V4.1 Flash currently requires unpadded sequences")
        use_cache = self.config.text_config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and (not use_cache or not isinstance(past_key_values, DeepSeekV41FlashCache)):
            raise ValueError("DeepSeek V4.1 requires its own cache and use_cache=True")
        cache = past_key_values or DeepSeekV41FlashCache(len(self.layers))
        if len(cache.layers) != len(self.layers):
            raise ValueError("DeepSeek V4.1 cache depth differs from the decoder")
        start = cache.length

        hidden = self.embed(input_ids.to(self.device)) if input_ids is not None else inputs_embeds.to(self.embed.weight)
        batch, length, _ = hidden.shape
        if cache.batch_size is not None and cache.batch_size != batch:
            raise ValueError("DeepSeek V4.1 cache batch size differs from the input")
        if length == 0 or start + length > self.config.text_config.max_position_embeddings:
            raise ValueError("Invalid DeepSeek V4.1 input length")
        image_mask = None
        if pixel_values is not None:
            if input_ids is None or start:
                raise ValueError("DeepSeek V4.1 images must be supplied with input_ids on prefill")
            image_mask = input_ids.to(self.device) == self.config.image_token_id
            images = torch.cat(self.get_image_features(pixel_values, image_grid_thw), dim=0).to(hidden)
            if images.size(0) != image_mask.sum().item():
                raise ValueError("DeepSeek V4.1 image features and placeholder count differ")
            hidden = hidden.clone()
            hidden[image_mask] = images

        streams = self.config.text_config.hc_mult
        state = V41AttentionState()
        pre = hidden.new_zeros((batch, length, streams), dtype=torch.float32)
        pre[..., 0] = 1
        hidden = hidden.unsqueeze(-2).expand(-1, -1, streams, -1)
        positions = torch.arange(start, start + length, device=hidden.device)
        histories = cache.stage() if past_key_values is not None else cache.layers
        for layer, history in zip(self.layers, histories):
            device = layer.attn_hc.fn.device
            positions = positions.to(device)
            image_mask = None if image_mask is None else image_mask.to(device)
            hidden, pre = layer(hidden.to(device), pre.to(device), positions, history, state, image_mask)

        hidden = collapse_streams(hidden, pre)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        hidden = self.norm(hidden.to(self.norm.weight.device))
        logits = F.linear(hidden.to(self.head.weight), self.head.weight).to(self.device)
        cache.layers = histories
        cache.length = start + length
        cache.batch_size = batch
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache if use_cache else None)
