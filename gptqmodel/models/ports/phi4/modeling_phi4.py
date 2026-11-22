# SPDX-FileCopyrightText: Microsoft Corporation
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: MIT
"""Legacy phi4mm checkpoint owners with current Transformers execution APIs (see NOTICE)."""

import math
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn import functional as F

from ..native.phi4_multimodal import modeling_phi4_multimodal as native
from .configuration_phi4 import Phi4MMConfig


_INPUT_MODE = ContextVar("phi4mm_input_mode", default=0)
_SEQUENCE_LENGTH = ContextVar("phi4mm_sequence_length", default=None)


class RotaryEmbedding(nn.Module):
    """Source-length LongRoPE with call-local, FP32-derived frequencies."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(self, x, position_ids):
        config, rope = self.config, self.config.rope_parameters
        dim = int(config.hidden_size // config.num_attention_heads * rope["partial_rotary_factor"])
        power = torch.arange(0, dim, 2, device=x.device).float() / dim
        frequency = rope["rope_theta"] ** power
        scale = 1.0
        if config.phi4mm_rope_type != "default":
            length = _SEQUENCE_LENGTH.get() or position_ids.max() + 1
            if config.phi4mm_rope_type in ("su", "yarn"):
                length = position_ids.max() + 1
            factors = rope["long_factor"] if length > config.original_max_position_embeddings else rope["short_factor"]
            frequency = frequency * torch.tensor(factors, device=x.device, dtype=torch.float32)
            ratio = config.max_position_embeddings / config.original_max_position_embeddings
            if ratio > 1:
                scale = (
                    1 + 0.1 * math.log(ratio)
                    if config.phi4mm_rope_type == "yarn"
                    else math.sqrt(1 + math.log(ratio) / math.log(config.original_max_position_embeddings))
                )
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            angles = position_ids.float()[..., None] / frequency
            angles = torch.cat((angles, angles), -1)
            return (angles.cos() * scale).to(x.dtype), (angles.sin() * scale).to(x.dtype)


def _mode(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or not (value == value.flatten()[0]).all():
            raise ValueError("phi4mm requires a homogeneous input_mode batch")
        value = value.flatten()[0].item()
    value = getattr(value, "value", value)
    if value is None:
        value = 0
    if value not in (0, 1, 2, 3):
        raise ValueError(f"Invalid phi4mm input_mode: {value}")
    return int(value)


class ModalityLinear(nn.Module):
    """The released PEFT layout, without mutable model-global adapter state."""

    def __init__(self, base_layer, config):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout = nn.ModuleDict()
        self.scaling = {}
        for name in ("vision", "speech"):
            spec = getattr(config, name + "_lora")
            rank = spec["r"]
            self.lora_A[name] = nn.Linear(base_layer.in_features, rank, bias=False)
            self.lora_B[name] = nn.Linear(rank, base_layer.out_features, bias=False)
            nn.init.zeros_(self.lora_B[name].weight)
            self.lora_dropout[name] = nn.Dropout(spec.get("dp", 0))
            self.scaling[name] = spec["lora_alpha"] / rank

    def forward(self, x):
        result = self.base_layer(x)
        mode = _INPUT_MODE.get()
        if mode:
            name = "speech" if mode == 2 else "vision"
            a, b = self.lora_A[name], self.lora_B[name]
            delta = b(a(self.lora_dropout[name](x.to(a.weight.dtype))))
            result = result + delta.to(result.dtype) * self.scaling[name]
        return result


class Phi4MMDecoderLayer(native.Phi4MultimodalDecoderLayer):
    def __init__(self, source, config):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        for name, module in source.named_children():
            self.add_module(name, module)
        for parent, names in ((self.self_attn, ("qkv_proj", "o_proj")), (self.mlp, ("gate_up_proj", "down_proj"))):
            for name in names:
                setattr(parent, name, ModalityLinear(getattr(parent, name), config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        input_mode=0,
        **kwargs,
    ):
        token = _INPUT_MODE.set(_mode(input_mode))
        try:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        finally:
            _INPUT_MODE.reset(token)


class ImageEmbedding(native.Phi4MultimodalImageEmbedding):
    def __init__(self, source):
        nn.Module.__init__(self)
        for name in ("config", "layer_idx", "crop_size", "image_dim_out", "num_img_tokens"):
            setattr(self, name, getattr(source, name))
        for name in ("drop", "img_processor", "image_token_compression", "img_processor_padding"):
            if hasattr(source, name):
                self.add_module(name, getattr(source, name))
        self.img_projection = nn.Sequential(source.img_projection_up, nn.GELU(), source.img_projection_down)
        self.glb_GN = source.global_img_feature_extensor
        self.sub_GN = source.sub_img_feature_extensor
        for layer in self.img_processor.encoder.layers:
            layer.self_attn.is_causal = False

    @property
    def img_projection_up(self):
        return self.img_projection[0]

    @property
    def img_projection_down(self):
        return self.img_projection[2]

    @property
    def global_img_feature_extensor(self):
        return self.glb_GN

    @property
    def sub_img_feature_extensor(self):
        return self.sub_GN


class AudioMLP(native.Phi4MultimodalAudioMLP):
    def __init__(self, source):
        nn.Module.__init__(self)
        self.layer_norm = source.layer_norm
        gate = nn.Module()
        gate.linear = source.gate_up_proj
        self.net = nn.ModuleList([gate, nn.Identity(), source.down_proj])
        self.act_fn = source.act_fn
        self.dropout = source.dropout

    @property
    def gate_up_proj(self):
        return self.net[0].linear

    @property
    def down_proj(self):
        return self.net[2]


class AudioAttention(native.Phi4MultimodalAudioAttention):
    def __init__(self, source):
        nn.Module.__init__(self)
        for name in ("config", "head_dim", "scaling", "attention_dropout"):
            setattr(self, name, getattr(source, name))
        self.is_causal = False
        for name in ("q", "k", "v"):
            setattr(self, "linear_" + name, getattr(source, name + "_proj"))
        self.linear_out = source.o_proj

    @property
    def q_proj(self):
        return self.linear_q

    @property
    def k_proj(self):
        return self.linear_k

    @property
    def v_proj(self):
        return self.linear_v

    @property
    def o_proj(self):
        return self.linear_out

    def forward(self, hidden_states, attention_mask, **kwargs):
        batch, length, width = hidden_states.shape
        heads = self.config.num_attention_heads
        q, k, v = [
            projection(hidden_states).reshape(batch, length, heads, self.head_dim).transpose(1, 2)
            for projection in (self.linear_q, self.linear_k, self.linear_v)
        ]
        scores = (q * self.scaling) @ k.transpose(-1, -2) + attention_mask
        blocked = torch.isneginf(attention_mask)
        # Source masked_softmax zeros fully masked rows after softmax.
        probabilities = scores.softmax(-1).masked_fill(blocked, 0.0)
        probabilities = F.dropout(probabilities, self.attention_dropout, self.training)
        output = (probabilities.to(v.dtype) @ v).transpose(1, 2).reshape(batch, length, width)
        return self.linear_out(output)


class Phi4MMAudioLayer(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.feed_forward_in = AudioMLP(source.feed_forward_in)
        self.feed_forward_out = AudioMLP(source.feed_forward_out)
        self.self_attn = AudioAttention(source.self_attn)
        self.conv = source.conv
        self.layer_norm_att = source.layer_norm_att
        self.layer_norm = source.layer_norm

    def forward(self, hidden_states, attention_mask):
        residual = hidden_states + 0.5 * self.feed_forward_in(hidden_states)
        hidden_states = residual + self.self_attn(self.layer_norm_att(residual), attention_mask)
        hidden_states = hidden_states + self.conv(hidden_states)
        hidden_states = hidden_states + 0.5 * self.feed_forward_out(hidden_states)
        return self.layer_norm(hidden_states)


class AudioEncoder(native.Phi4MultimodalAudioModel):
    def __init__(self, source):
        nn.Module.__init__(self)
        self.config = source.config
        self.encoder_embedding = source.encoder_embedding
        self.embed = source.embed
        self.relative_attention_bias_layer = source.relative_attention_bias_layer
        self.encoders = nn.ModuleList([Phi4MMAudioLayer(layer) for layer in source.encoders])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, mask=None, **kwargs):
        hidden_states = self.encoder_embedding(hidden_states)
        hidden_states, hs_mask, mask = self.forward_embeddings(hidden_states, mask)
        bs, seq_len, dim = hidden_states.shape
        padding = (-seq_len) % 500 if seq_len > 500 else 0
        if seq_len > 500:
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding))
            hidden_states = hidden_states.reshape(-1, 500, dim)
            masks_unfold = None
            if mask is not None:
                masks_unfold = F.pad(mask.squeeze(1), (0, padding)).reshape(-1, 500)
            hs_mask = self.calculate_hs_mask(hidden_states, hidden_states.device, masks_unfold)
        bias = self.relative_attention_bias_layer(hidden_states)
        # The source uses masked_softmax: false entries must be excluded, not added as 0/1.
        attention_mask = bias.expand(
            bs if seq_len <= 500 else hidden_states.shape[0], -1, hidden_states.shape[1], -1
        ).clone()
        attention_mask = attention_mask.masked_fill(~hs_mask.unsqueeze(1), float("-inf"))
        for layer in self.encoders:
            hidden_states = layer(hidden_states, attention_mask)
        if seq_len > 500:
            hidden_states = hidden_states.reshape(bs, -1, dim)[:, :seq_len]
        return hidden_states


class AudioEmbedding(native.Phi4MultimodalAudioEmbedding):
    def __init__(self, source):
        nn.Module.__init__(self)
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.drop = source.drop
        self.encoder = AudioEncoder(source.encoder)
        self.audio_projection = nn.ModuleDict(
            {
                "speech": nn.Sequential(source.up_proj_for_speech, nn.GELU(), source.down_proj_for_speech),
                "vision": nn.Sequential(
                    source.up_proj_for_vision_speech, nn.GELU(), source.down_proj_for_vision_speech
                ),
            }
        )

    @property
    def up_proj_for_speech(self):
        return self.audio_projection["speech"][0]

    @property
    def down_proj_for_speech(self):
        return self.audio_projection["speech"][2]

    @property
    def up_proj_for_vision_speech(self):
        return self.audio_projection["vision"][0]

    @property
    def down_proj_for_vision_speech(self):
        return self.audio_projection["vision"][2]


class FeatureEmbedding(native.Phi4MultimodalFeatureEmbedding):
    def __init__(self, source):
        nn.Module.__init__(self)
        self.config = source.config
        self.image_token_id = source.image_token_id
        self.audio_token_id = source.audio_token_id
        self.image_embed = ImageEmbedding(source.image_embed)
        self.audio_embed = AudioEmbedding(source.audio_embed)

    def forward(
        self,
        input_ids,
        inputs_embeds,
        image_pixel_values=None,
        audio_input_features=None,
        image_sizes=None,
        image_attention_mask=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
    ):
        image_embeds = audio_embeds = None
        if image_pixel_values is not None and (input_ids == self.image_token_id).any():
            image_embeds = self.image_embed(
                input_ids, inputs_embeds, image_pixel_values, image_sizes, image_attention_mask
            )
        if audio_input_features is not None and (input_ids == self.audio_token_id).any():
            audio_embeds = self.audio_embed(
                input_ids,
                inputs_embeds,
                audio_input_features,
                audio_embed_sizes,
                audio_attention_mask,
                "vision" if _INPUT_MODE.get() in (1, 3) else "speech",
            )
        if image_embeds is not None and audio_embeds is not None:
            return torch.where((input_ids == self.image_token_id).unsqueeze(-1), image_embeds, audio_embeds)
        return (
            image_embeds if image_embeds is not None else audio_embeds if audio_embeds is not None else inputs_embeds
        )


class Phi4MMForCausalLM(native.Phi4MultimodalForCausalLM):
    config_class = Phi4MMConfig
    # Device-map splitting matches class names; the wrapped legacy layers replace the native ones.
    _no_split_modules = ["Phi4MMDecoderLayer", "Phi4MMAudioLayer", "Phi4MultimodalVisionEncoderLayer"]
    _supports_flash_attn = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _tp_plan = {}
    _pp_plan = {}

    def __init__(self, config):
        super().__init__(config)
        self.model.layers = nn.ModuleList([Phi4MMDecoderLayer(layer, config) for layer in self.model.layers])
        self.model.embed_tokens_extend = FeatureEmbedding(self.model.embed_tokens_extend)
        self.model.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        input_image_embeds=None,
        image_sizes=None,
        image_attention_mask=None,
        input_audio_embeds=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=0,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        mode = _mode(input_mode)
        native_images = kwargs.pop("image_pixel_values", None)
        native_audio = kwargs.pop("audio_input_features", None)
        if input_image_embeds is None:
            input_image_embeds = native_images
        elif native_images is not None:
            raise ValueError("Specify only one image input spelling")
        if input_audio_embeds is None:
            input_audio_embeds = native_audio
        elif native_audio is not None:
            raise ValueError("Specify only one audio input spelling")
        if input_ids is not None:
            input_ids = input_ids.clone()
            input_ids[(input_ids < 0) & (input_ids > -10000)] = self.config.vision_config.image_token_id
            input_ids[(input_ids <= -10000) & (input_ids > -1000000000)] = self.config.audio_config.audio_token_id
        token = _INPUT_MODE.set(mode)
        length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        length += past_key_values.get_seq_length() if past_key_values is not None else 0
        sequence_token = _SEQUENCE_LENGTH.set(length)
        try:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                image_pixel_values=input_image_embeds,
                image_sizes=image_sizes,
                image_attention_mask=image_attention_mask,
                audio_input_features=input_audio_embeds,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                labels=labels,
                use_cache=use_cache,
                input_mode=mode,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        finally:
            _INPUT_MODE.reset(token)
            _SEQUENCE_LENGTH.reset(sequence_token)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, input_mode=0, input_image_embeds=None, input_audio_embeds=None, **kwargs
    ):
        result = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, input_mode=input_mode, **kwargs
        )
        result["input_image_embeds"] = input_image_embeds
        result["input_audio_embeds"] = input_audio_embeds
        return result

    def _expand_inputs_for_generation(self, expand_size=1, is_encoder_decoder=False, input_ids=None, **kwargs):
        if expand_size == 1 or input_ids is None:
            return super()._expand_inputs_for_generation(expand_size, is_encoder_decoder, input_ids, **kwargs)
        grouped = {}
        for kind, names, token_id in (
            (
                "image",
                ("input_image_embeds", "image_pixel_values", "image_sizes", "image_attention_mask"),
                self.config.vision_config.image_token_id,
            ),
            (
                "audio",
                ("input_audio_embeds", "audio_input_features", "audio_embed_sizes", "audio_attention_mask"),
                self.config.audio_config.audio_token_id,
            ),
        ):
            values = {name: kwargs.pop(name) for name in names if kwargs.get(name) is not None}
            media = next((values[name] for name in names[:2] if name in values), None)
            if media is None or media.numel() == 0:
                grouped.update(values)
                continue
            if kind == "audio":
                lengths = values["audio_embed_sizes"].tolist()
                placeholders = (input_ids == token_id) | ((input_ids <= -10000) & (input_ids > -1000000000))
            else:
                lengths = []
                crop = self.config.vision_config.image_size
                pooled = crop // self.config.vision_config.patch_size // 2
                for i, (height, width) in enumerate(values["image_sizes"].tolist()):
                    h, w = height // crop, width // crop
                    mask = values.get("image_attention_mask")
                    local = (
                        mask[i, 1 : 1 + h * w, ::2, ::2] if mask is not None else media.new_ones(h * w, pooled, pooled)
                    )
                    local = local.reshape(h, w, pooled, pooled).permute(0, 2, 1, 3).reshape(h * pooled, w * pooled)
                    lengths.append(pooled * (pooled + 1) + 1 + int(local.sum()) + int(local[:, 0].sum()))
                placeholders = (input_ids == token_id) | ((input_ids < 0) & (input_ids > -10000))
            cursor, indexes = 0, []
            for count in placeholders.sum(-1).tolist():
                start = cursor
                while count > 0 and cursor < len(lengths):
                    count -= lengths[cursor]
                    cursor += 1
                if count != 0:
                    raise ValueError(f"phi4mm {kind} placeholders do not match media lengths")
                indexes.extend(list(range(start, cursor)) * expand_size)
            if cursor != len(lengths):
                raise ValueError(f"phi4mm unused {kind} inputs")
            for name, value in values.items():
                grouped[name] = value.index_select(0, torch.tensor(indexes, device=value.device))
        ids, expanded = super()._expand_inputs_for_generation(expand_size, is_encoder_decoder, input_ids, **kwargs)
        return ids, dict(expanded, **grouped)
