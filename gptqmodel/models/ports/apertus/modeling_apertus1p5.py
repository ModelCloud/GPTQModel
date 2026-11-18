# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel, slice_logits_to_keep
from ..native.apertus.modeling_apertus import ApertusModel, ApertusPreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    torch_compilable_check,
)

from .configuration_apertus1p5 import Apertus1p5Config, Apertus1p5TextConfig, Apertus1p5VisionTokenizerConfig
from .modeling_wavtokenizer import WavTokenizerEncoderModel


@auto_docstring
@dataclass
class Apertus1p5VisionTokenizerEncoderOutput(ModelOutput):
    r"""
    image_tokens (`torch.LongTensor` of shape `(batch_size, height // factor, width // factor)`):
        Discrete codebook indices in `[0, config.codebook_size)`, with one index per spatial cell.
        `height` and `width` are the input pixel dimensions and `factor` is `config.spatial_scale_factor`
        (16 by default). The indices have no language-vocabulary offset applied.
    """

    image_tokens: torch.LongTensor | None = None


@auto_docstring
@dataclass
class Apertus1p5VisionTokenizerModelOutput(BaseModelOutputWithPooling):
    r"""
    pooler_output (`tuple[torch.Tensor]`):
        One `(num_codes, hidden_size)` tensor of embedded image codes per input image.
    image_tokens (`list[torch.LongTensor]`):
        One grid of discrete codebook indices per input image, of shape `(height // factor, width // factor)`
        where `factor` is `config.spatial_scale_factor` (16 by default).
    """

    pooler_output: tuple[torch.Tensor] | None = None
    image_tokens: list[torch.LongTensor] | None = None


@auto_docstring
@dataclass
class Apertus1p5AudioTokenizerModelOutput(BaseModelOutputWithPooling):
    r"""
    pooler_output (`tuple[torch.Tensor]`):
        One `(num_codes, hidden_size)` tensor of embedded audio codes per input clip.
    audio_tokens (`list[torch.LongTensor]`):
        One 1D tensor of `ceil(length / hop)` discrete codebook indices per input clip.
    """

    pooler_output: tuple[torch.Tensor] | None = None
    audio_tokens: list[torch.LongTensor] | None = None


class Apertus1p5VisionTokenizerResnetBlock(nn.Module):
    def __init__(
        self,
        config,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return residual + hidden_states


class Apertus1p5VisionTokenizerAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)

        # compute attention
        batch_size, channels, height, width = query_states.shape
        query_states = query_states.reshape(batch_size, channels, height * width).permute(0, 2, 1)
        key_states = key_states.reshape(batch_size, channels, height * width)
        attn_weights = torch.bmm(query_states, key_states)
        attn_weights = attn_weights * (int(channels) ** (-0.5))
        attn_weights = F.softmax(attn_weights, dim=2)

        # attend to values
        value_states = value_states.reshape(batch_size, channels, height * width)
        attn_weights = attn_weights.permute(0, 2, 1)
        attn_output = torch.bmm(value_states, attn_weights).reshape(batch_size, channels, height, width)

        attn_output = self.proj_out(attn_output)
        return residual + attn_output


class Apertus1p5VisionTokenizerDownsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, hidden_states):
        # no asymmetric padding in torch conv, must do it ourselves
        hidden_states = F.pad(hidden_states, pad=(0, 1, 0, 1), mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Apertus1p5VisionTokenizerEncoderBlock(nn.Module):
    def __init__(
        self,
        config: Apertus1p5VisionTokenizerConfig,
        in_channels: int,
        out_channels: int,
        use_attention: bool,
    ):
        super().__init__()
        self.resnet = Apertus1p5VisionTokenizerResnetBlock(config, in_channels, out_channels, conv_shortcut=False)
        self.attention = Apertus1p5VisionTokenizerAttnBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnet(hidden_states)
        return self.attention(hidden_states)


class Apertus1p5VisionTokenizerEncoderStage(nn.Module):
    def __init__(self, config: Apertus1p5VisionTokenizerConfig, stage_idx: int):
        super().__init__()
        previous_multiplier = 1 if stage_idx == 0 else config.channel_multiplier[stage_idx - 1]
        in_channels = config.base_channels * previous_multiplier
        out_channels = config.base_channels * config.channel_multiplier[stage_idx]
        resolution = config.resolution // 2**stage_idx
        use_attention = resolution in config.attn_resolutions

        layers = []
        for _ in range(config.num_res_blocks):
            layers.append(Apertus1p5VisionTokenizerEncoderBlock(config, in_channels, out_channels, use_attention))
            in_channels = out_channels

        self.layers = nn.ModuleList(layers)
        self.downsample = (
            # CODEPATH: channel_multiplier sets the stage count; all but the final stage downsample.
            Apertus1p5VisionTokenizerDownsample(out_channels)
            if stage_idx < len(config.channel_multiplier) - 1
            else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.downsample(hidden_states)


class Apertus1p5VisionTokenizerMiddleBlock(nn.Module):
    def __init__(self, config: Apertus1p5VisionTokenizerConfig, channels: int):
        super().__init__()
        self.block_1 = Apertus1p5VisionTokenizerResnetBlock(config, channels, channels, conv_shortcut=False)
        self.attn_1 = Apertus1p5VisionTokenizerAttnBlock(channels)
        self.block_2 = Apertus1p5VisionTokenizerResnetBlock(config, channels, channels, conv_shortcut=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.block_1(hidden_states)
        hidden_states = self.attn_1(hidden_states)
        return self.block_2(hidden_states)


class Apertus1p5VisionTokenizerEncoder(nn.Module):
    def __init__(self, config: Apertus1p5VisionTokenizerConfig):
        super().__init__()
        output_channels = config.base_channels * config.channel_multiplier[-1]

        self.conv_in = nn.Conv2d(config.in_channels, config.base_channels, kernel_size=3, stride=1, padding=1)
        self.stages = nn.ModuleList(
            [
                Apertus1p5VisionTokenizerEncoderStage(config, stage_idx)
                for stage_idx in range(len(config.channel_multiplier))
            ]
        )
        self.mid = Apertus1p5VisionTokenizerMiddleBlock(config, output_channels)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=output_channels, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(output_channels, config.latent_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(pixel_values)
        for stage in self.stages:
            hidden_states = stage(hidden_states)

        hidden_states = self.mid(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * torch.sigmoid(hidden_states)
        return self.conv_out(hidden_states)


class Apertus1p5VisionTokenizerVectorQuantizer(nn.Module):
    """Inference-only IBQ codebook scoring using dot-product similarity and `argmax`."""

    def __init__(self, config: Apertus1p5VisionTokenizerConfig):
        super().__init__()
        self.embedding = nn.Linear(config.embed_dim, config.codebook_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.LongTensor:
        batch_size, _, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        logits = self.embedding(hidden_states)
        return logits.argmax(dim=-1).reshape(batch_size, height, width)


@auto_docstring
class Apertus1p5VisionTokenizerPreTrainedModel(PreTrainedModel):
    config: Apertus1p5VisionTokenizerConfig
    base_model_prefix = "vision_tokenizer"
    input_modalities = ("image",)
    _no_split_modules = ["Apertus1p5VisionTokenizerResnetBlock", "Apertus1p5VisionTokenizerAttnBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.GroupNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight)


@auto_docstring(
    custom_intro="""
    Encode-only EMU3.5-derived vision tokenizer used by Apertus 1.5. It returns hard codebook indices; run it in
    `float32` for stable code assignment (Apertus 1.5 does this automatically).
    """
)
class Apertus1p5VisionTokenizerModel(Apertus1p5VisionTokenizerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Apertus1p5VisionTokenizerConfig):
        super().__init__(config)
        self.encoder = Apertus1p5VisionTokenizerEncoder(config)
        self.quantize = Apertus1p5VisionTokenizerVectorQuantizer(config)
        self.quant_conv = nn.Conv2d(config.latent_channels, config.embed_dim, kernel_size=1)
        self.post_init()

    @auto_docstring(custom_intro="Tokenizes images into a grid of discrete codebook indices.")
    def encode(self, pixel_values: torch.Tensor) -> Apertus1p5VisionTokenizerEncoderOutput:
        r"""
        pixel_values (`torch.Tensor` of shape `(batch_size, channels, height, width)`):
            RGB images normalized to `[-1, 1]`.

        Returns:
            `Apertus1p5VisionTokenizerEncoderOutput`: Discrete image-code grids in `image_tokens`.
        """
        # match the encoder dtype (fp32 inside Apertus 1.5, the requested dtype when used standalone)
        pixel_values = pixel_values.to(self.encoder.conv_in.weight.dtype)
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        return Apertus1p5VisionTokenizerEncoderOutput(image_tokens=self.quantize(hidden_states))


@auto_docstring
class Apertus1p5PreTrainedModel(PreTrainedModel):
    config: Apertus1p5Config
    base_model_prefix = "model"
    input_modalities = ("image", "audio", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    # per-image / per-audio-clip tokenizer loops are data-dependent control flow
    _can_compile_fullgraph = False
    _supports_flex_attn = True
    _supports_attention_backend = True
    _no_split_modules = [
        "ApertusDecoderLayer",
        "Apertus1p5VisionTokenizerResnetBlock",
        "Apertus1p5VisionTokenizerAttnBlock",
    ]


# The text backbone is the native Transformers Apertus decoder (same math and state-dict keys); only
# the extended-vocabulary config type differs.
class Apertus1p5TextPreTrainedModel(ApertusPreTrainedModel):
    config: Apertus1p5TextConfig


class Apertus1p5TextModel(ApertusModel):
    config: Apertus1p5TextConfig


@auto_docstring
class Apertus1p5TextForCausalLM(Apertus1p5TextPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}
    config: Apertus1p5TextConfig
    input_modalities = ("text",)
    _can_compile_fullgraph = True
    _keys_to_ignore_on_load_unexpected = [r"^model\.vision_tokenizer\.", r"^model\.audio_tokenizer\."]

    def __init__(self, config: Apertus1p5TextConfig):
        config.validate_output_vocab_size()
        super().__init__(config)
        self.model = Apertus1p5TextModel(config)
        self.vocab_size = config.vocab_size
        # the LM head may be pruned to the text-only prefix of the vocabulary; input embeddings keep the
        # full vocabulary, and retained output ids equal their token ids
        self.lm_head = nn.Linear(config.hidden_size, config.output_vocab_size or config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Target token ids. `-100` ignores a position; other values must be within the physical LM-head range.
            With a pruned head, mask multimodal/input-only ids with `-100`.

        Returns:
            [`~modeling_outputs.CausalLMOutputWithPast`] or `tuple(torch.Tensor)`:
                Logits use the physical LM-head width (`config.output_vocab_size` when set, otherwise
                `config.vocab_size`), with or without `labels`.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.lm_head.out_features, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class Apertus1p5Model(Apertus1p5PreTrainedModel):
    # code assignment is an argmax over codebook logits: half precision flips ~10% of image codes (bf16, 131k
    # codebook), so both tokenizers are kept in fp32 even when the model is loaded in fp16/bf16
    _keep_in_fp32_modules_strict = ["vision_tokenizer", "audio_tokenizer"]

    def __init__(self, config: Apertus1p5Config):
        super().__init__(config)
        self.language_model = Apertus1p5TextModel(config.text_config)
        self.vision_tokenizer = Apertus1p5VisionTokenizerModel(config.vision_config)
        self.audio_tokenizer = WavTokenizerEncoderModel(config.audio_config)
        self.post_init()

    def _encode_images(self, pixel_values: torch.Tensor, image_sizes: torch.LongTensor) -> list[torch.LongTensor]:
        """Crop batch padding and retain the raw code grid from each individually encoded image."""
        image_tokens = []
        for image, size in zip(pixel_values, image_sizes):
            image = image[None, :, : int(size[0]), : int(size[1])]
            image_tokens.append(self.vision_tokenizer.encode(image).image_tokens[0])
        return image_tokens

    def get_image_tokens(self, pixel_values: torch.Tensor, image_sizes: torch.LongTensor) -> torch.LongTensor:
        """
        Tokenizes each unpadded image and returns flattened shared-vocabulary ids.

        Args:
            pixel_values (`torch.Tensor` of shape `(num_images, num_channels, height, width)`):
                The tensors corresponding to the input images, padded to a common size.
            image_sizes (`torch.LongTensor` of shape `(num_images, 2)`):
                The true size of each image, being (height, width).

        Returns:
            `torch.LongTensor`: Image code ids offset by `config.image_token_offset`.
        """
        image_tokens = self._encode_images(pixel_values, image_sizes)
        return torch.cat([codes.flatten() for codes in image_tokens]) + self.config.image_token_offset

    @can_return_tuple
    @auto_docstring(custom_intro="Tokenizes images and embeds their discrete codes with the text embedding layer.")
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Apertus1p5VisionTokenizerModelOutput:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_images, num_channels, height, width)`):
            The tensors corresponding to the input images, padded to a common size.
        image_sizes (`torch.LongTensor` of shape `(num_images, 2)`):
            The true size of each image, being (height, width).
        """
        image_tokens = self._encode_images(pixel_values, image_sizes)
        vocab_ids = torch.cat([codes.flatten() for codes in image_tokens]) + self.config.image_token_offset
        image_embeddings = self.get_input_embeddings()(vocab_ids)
        image_features = torch.split(image_embeddings, [codes.numel() for codes in image_tokens])
        return Apertus1p5VisionTokenizerModelOutput(image_tokens=image_tokens, pooler_output=image_features)

    def get_audio_tokens(self, input_features: torch.Tensor, feature_attention_mask: torch.Tensor) -> torch.LongTensor:
        """
        Tokenizes each unpadded audio clip and returns flattened shared-vocabulary ids.

        Args:
            input_features (`torch.Tensor` of shape `(num_clips, 1, max_length)`):
                Mono 24 kHz waveforms, padded to a common length.
            feature_attention_mask (`torch.Tensor` of shape `(num_clips, max_length)`):
                Mask with 1 for valid samples and 0 for padding, as returned by the feature extractor.

        Returns:
            `torch.LongTensor`: Audio code ids offset by `config.audio_token_offset`.
        """
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(1)
        audio_lengths = feature_attention_mask.sum(dim=-1)
        if (audio_lengths == 0).any():
            raise ValueError("`feature_attention_mask` marks an audio clip with zero valid samples.")
        # a 0 -> 1 transition along a row means the mask is not right-padded
        if (feature_attention_mask.long().diff(dim=-1) > 0).any():
            raise ValueError(
                "`feature_attention_mask` must be right-padded (valid samples first), as returned by the feature "
                "extractor."
            )
        vocab_ids_list = []
        for clip, length in zip(input_features, audio_lengths):
            clip = clip[None, :, : int(length)].to(self.audio_tokenizer.dtype)
            codes = self.audio_tokenizer.encode(clip, return_dict=True).audio_codes
            vocab_ids_list.append(codes.flatten() + self.config.audio_token_offset)
        return torch.cat(vocab_ids_list)

    def _get_audio_token_counts(self, feature_attention_mask: torch.Tensor) -> list[int]:
        """Number of audio codes per clip: one per started `hop_length` window of valid samples."""
        hop_length = self.audio_tokenizer.config.hop_length
        return ((feature_attention_mask.sum(dim=-1) + hop_length - 1) // hop_length).tolist()

    @can_return_tuple
    @auto_docstring(custom_intro="Tokenizes audio and embeds its discrete codes with the text embedding layer.")
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Apertus1p5AudioTokenizerModelOutput:
        r"""
        input_features (`torch.Tensor` of shape `(num_clips, 1, max_length)`):
            Mono 24 kHz waveforms, padded to a common length.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, max_length)`):
            Mask with 1 for valid samples and 0 for padding, as returned by the feature extractor.
        """
        vocab_ids = self.get_audio_tokens(input_features, feature_attention_mask)
        split_sizes = self._get_audio_token_counts(feature_attention_mask)
        audio_embeddings = self.get_input_embeddings()(vocab_ids)
        audio_features = torch.split(audio_embeddings, split_sizes)
        audio_tokens = [ids - self.config.audio_token_offset for ids in torch.split(vocab_ids, split_sizes)]
        return Apertus1p5AudioTokenizerModelOutput(audio_tokens=audio_tokens, pooler_output=audio_features)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
    ):
        """Build placeholder masks and verify that their lengths match the supplied media features."""
        if input_ids is None:
            embedder = self.get_input_embeddings()
            image_token_embed = embedder(
                torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = (inputs_embeds == image_token_embed).all(-1)
            audio_token_embed = embedder(
                torch.full((), self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_audio_mask = (inputs_embeds == audio_token_embed).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        if image_features is not None:
            n_image_tokens = special_image_mask.sum()
            torch_compilable_check(
                n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, "
                f"features: {image_features.shape[0]}",
            )
        if audio_features is not None:
            n_audio_tokens = special_audio_mask.sum()
            torch_compilable_check(
                n_audio_tokens * inputs_embeds.shape[-1] == audio_features.numel(),
                f"Audio features and audio tokens do not match, tokens: {n_audio_tokens}, "
                f"features: {audio_features.shape[0]}",
            )

        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        special_audio_mask = special_audio_mask.unsqueeze(-1).to(inputs_embeds.device)
        return special_image_mask, special_audio_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_images, num_channels, max_height, max_width)`, *optional*):
            Processor-produced RGB image tensors, normalized to `[-1, 1]` and padded. Requires `image_sizes`;
            rows must follow expanded image-placeholder order.
        image_sizes (`torch.LongTensor` of shape `(num_images, 2)`, *optional*):
            True `(height, width)` per image, required with `pixel_values` to remove batch padding.
        input_features (`torch.Tensor` of shape `(num_clips, 1, max_length)`, *optional*):
            Processor-produced mono 24 kHz waveforms, peak-normalized and padded. Requires
            `feature_attention_mask`; rows must follow expanded audio-placeholder order.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, max_length)`, *optional*):
            Right-padded mask with 1 for valid samples and 0 for padding, required with `input_features`.
        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Input embeddings instead of `input_ids`. With modality inputs, placeholder positions must contain
            their exact token embeddings so the model can detect them.
        mm_encoder_outputs (`dict[str, BaseModelOutputWithPooling]`, *optional*):
            Precomputed image features under the `"image"` key. Its `pooler_output` contains one
            `(num_image_tokens, hidden_size)` tensor per image, in placeholder order. Replaces `pixel_values`
            and `image_sizes`; audio still uses `input_features` and `feature_attention_mask`.
        """
        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values and mm_encoder_outputs at the same time")
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(pixel_values, image_sizes, return_dict=True)

        image_features = None
        if mm_encoder_outputs.get("image") is not None:
            image_features = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
        audio_features = None
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features, feature_attention_mask, return_dict=True
            ).pooler_output
            audio_features = torch.cat(audio_features, dim=0)

        if image_features is not None or audio_features is not None:
            special_image_mask, special_audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features, audio_features=audio_features
            )
            if image_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            if audio_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        return self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )


@auto_docstring
class Apertus1p5ForConditionalGeneration(Apertus1p5PreTrainedModel, GenerationMixin):
    output_modalities = ("text",)
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}

    def __init__(self, config: Apertus1p5Config):
        config.validate_architecture()
        super().__init__(config)
        self.model = Apertus1p5Model(config)
        # the LM head may be pruned to the text-only prefix of the vocabulary (see Apertus1p5TextConfig)
        text_config = config.text_config
        self.lm_head = nn.Linear(
            text_config.hidden_size, text_config.output_vocab_size or text_config.vocab_size, bias=False
        )

        self.post_init()

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Apertus1p5VisionTokenizerModelOutput:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_images, num_channels, height, width)`):
            The tensors corresponding to the input images, padded to a common size.
        image_sizes (`torch.LongTensor` of shape `(num_images, 2)`):
            The true size of each image, being (height, width).
        """
        return self.model.get_image_features(pixel_values, image_sizes, **kwargs)

    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Apertus1p5AudioTokenizerModelOutput:
        r"""
        input_features (`torch.Tensor` of shape `(num_clips, 1, max_length)`):
            Mono 24 kHz waveforms, padded to a common length.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, max_length)`):
            Mask with 1 for valid samples and 0 for padding, as returned by the feature extractor.
        """
        return self.model.get_audio_features(input_features, feature_attention_mask, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_images, num_channels, max_height, max_width)`, *optional*):
            Padded RGB images in `[-1, 1]`, in image-placeholder order. Requires `image_sizes`.
        image_sizes (`torch.LongTensor` of shape `(num_images, 2)`, *optional*):
            Unpadded `(height, width)` per image, required with `pixel_values`.
        input_features (`torch.Tensor` of shape `(num_clips, 1, max_length)`, *optional*):
            Padded, peak-normalized mono 24 kHz waveforms in audio-placeholder order. Requires `feature_attention_mask`.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, max_length)`, *optional*):
            Audio sample mask: 1 for valid samples, 0 for right padding. Required with `input_features`.
        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Embeddings instead of `input_ids`. With media inputs, placeholders must retain their exact token embeddings.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Next-token labels in `[0, output_vocab_size)` (or `[0, text_config.vocab_size)` when unset).
            Use `-100` for input-only multimodal tokens and other ignored positions.
        mm_encoder_outputs (`dict[str, BaseModelOutputWithPooling]`, *optional*):
            Precomputed image features under the `"image"` key. Its `pooler_output` contains one
            `(num_image_tokens, hidden_size)` tensor per image, in placeholder order. Replaces `pixel_values`
            and `image_sizes`; audio still uses `input_features` and `feature_attention_mask`.

        Returns:
            [`~modeling_outputs.CausalLMOutputWithPast`] or `tuple(torch.Tensor)`:
                Logits have width `text_config.output_vocab_size`, or `text_config.vocab_size` when unset,
                with or without `labels`.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        hidden_states = outputs[0]
        hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.lm_head.out_features, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor | None, dict[str, Any]]:
        """Expand audio clips by prompt; shared generation handles precomputed image features."""
        if expand_size == 1 or model_kwargs.get("input_features") is None:
            return super()._expand_inputs_for_generation(expand_size, is_encoder_decoder, input_ids, **model_kwargs)

        inputs_embeds = model_kwargs.get("inputs_embeds")
        if inputs_embeds is not None:
            token_embed = self.get_input_embeddings()(
                torch.full((), self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            placeholder_mask = (inputs_embeds == token_embed).all(dim=-1)
        elif input_ids is not None:
            placeholder_mask = input_ids == self.config.audio_token_id
        else:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be provided for multimodal expansion.")

        audio_token_counts = self.model._get_audio_token_counts(model_kwargs["feature_attention_mask"])
        audio_nums = []
        audio_idx = 0
        for expected_tokens in placeholder_mask.sum(dim=-1).tolist():
            sample_audio_idx = audio_idx
            actual_tokens = 0
            while actual_tokens < expected_tokens and audio_idx < len(audio_token_counts):
                actual_tokens += audio_token_counts[audio_idx]
                audio_idx += 1
            audio_nums.append(audio_idx - sample_audio_idx)

        # Keep flattened audio tensors out of the shared per-text-row expansion.
        audio_kwargs = {}
        for key in ("input_features", "feature_attention_mask"):
            value = model_kwargs.pop(key)
            repeats = (expand_size,) + (1,) * (value.ndim - 1)
            audio_kwargs[key] = torch.cat([group.repeat(repeats) for group in value.split(audio_nums)], dim=0)

        input_ids, model_kwargs = super()._expand_inputs_for_generation(
            expand_size, is_encoder_decoder, input_ids, **model_kwargs
        )
        model_kwargs.update(audio_kwargs)
        return input_ids, model_kwargs


__all__ = [
    "Apertus1p5ForConditionalGeneration",
    "Apertus1p5PreTrainedModel",
    "Apertus1p5TextForCausalLM",
    "Apertus1p5TextModel",
    "Apertus1p5TextPreTrainedModel",
    "Apertus1p5VisionTokenizerModel",
    "Apertus1p5VisionTokenizerPreTrainedModel",
    "Apertus1p5Model",
]
