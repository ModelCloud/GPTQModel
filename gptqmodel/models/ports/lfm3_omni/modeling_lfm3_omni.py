# Copyright 2026 The HuggingFace Team. All rights reserved.
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
# GPT-QModel adaptation: package-owned loading and Transformers 5 execution.
# Pinned source, modifications and qualification limits: README.md and NOTICE.
import copy
import warnings
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.streamers import BaseStreamer
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.lfm2.modeling_lfm2 import Lfm2Model
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    torch_compilable_check,
)
from transformers.utils.import_utils import is_tracing

from .configuration_lfm3_omni import Lfm3OmniConfig, Lfm3OmniEncoderConfig
from .modeling_lfm3 import Lfm3Model
from .modeling_lfm3_vl import Lfm3VlVisionModel


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Omni outputs, with hidden states and attentions.
    """
)
@dataclass
class Lfm3OmniModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_audio_tokens, hidden_size)`.
        audio_hidden_states of the model produced by the speech encoder and after projecting the last hidden state.
    """

    audio_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Omni causal language model (or autoregressive) outputs.
    """
)
@dataclass
class Lfm3OmniCausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_audio_tokens, hidden_size)`.
        audio_hidden_states of the model produced by the speech encoder and after projecting the last hidden state.
    text_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        The language modeling loss of the text tokens.
    audio_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `audio_labels` is provided):
        The depth decoder loss of the speech frames.
    """

    audio_hidden_states: torch.FloatTensor | None = None
    text_loss: torch.FloatTensor | None = None
    audio_loss: torch.FloatTensor | None = None


class Lfm3OmniEncoderStreamingCache:
    """Bounded downsampler context and causal encoder state for synchronized audio chunks."""

    def __init__(self, config: Lfm3OmniEncoderConfig):
        cache_config = copy.deepcopy(config.text_config)
        cache_config.sliding_window = config.sliding_window
        cache_config.layer_types = [
            "sliding_attention" if kind == "full_attention" else kind for kind in cache_config.layer_types
        ]
        self.past_key_values = DynamicCache(config=cache_config)
        self.features: torch.Tensor | None = None
        self.offset = 0
        self.num_frames = 0
        self.num_tokens = 0
        self.finished = False


@auto_docstring(
    custom_intro="""
    Outputs of [`Lfm3OmniForConditionalGeneration.generate`].
    """
)
@dataclass
class Lfm3OmniGenerateOutput(ModelOutput):
    r"""
    sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        The prompt followed by the generated tokens: text tokens as their ids, and one `audio_out_token_id` per
        generated speech frame.
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
        The codes of the generated speech frames in generation order, without the end-of-audio frames unless
        `return_audio_eos` was set. Rows are padded with -1; slice to `audio_lengths` before codec decoding.
    sequence_lengths (`torch.LongTensor` of shape `(batch_size,)`):
        The prompt width (including input padding) plus the number of returned generated slots for each row.
    audio_lengths (`torch.LongTensor` of shape `(batch_size,)`):
        The number of returned speech frames per row.
    """

    sequences: torch.LongTensor = None
    audio_codes: torch.LongTensor | None = None
    sequence_lengths: torch.LongTensor | None = None
    audio_lengths: torch.LongTensor | None = None


@auto_docstring(custom_intro="State returned by [`Lfm3OmniForConditionalGeneration.prefill_audio`].")
@dataclass
class Lfm3OmniStreamingOutput(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*):
        Language-model cache including all completed audio tokens. Updated in place by subsequent calls.
    encoder_cache (`Lfm3OmniEncoderStreamingCache`, *optional*):
        Streaming encoder state for the current utterance. Start a new encoder cache for each utterance.
    attention_mask (`torch.LongTensor` of shape `(batch_size, cached_sequence_length)`, *optional*):
        Full language-model mask, including the text prefix and completed audio tokens.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_new_tokens, hidden_size)`, *optional*):
        Language-model states for this chunk, or `None` when no audio token was completed.
    """

    past_key_values: Cache | None = None
    encoder_cache: Lfm3OmniEncoderStreamingCache | None = None
    attention_mask: torch.LongTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None


class Lfm3OmniLiveTranscriptionState:
    """State of one live transcription: the encoder follows the whole stream, the language model one context."""

    def __init__(self):
        self.encoder_cache: Lfm3OmniEncoderStreamingCache | None = None
        self.past_key_values: Cache | None = None
        self.attention_mask: torch.LongTensor | None = None
        # fed before the next chunk's audio: the prompt of a new context, or the `<|text_end|>` closing the last block
        self.pending: torch.FloatTensor | None = None
        self.finished = False


@auto_docstring(custom_intro="Output of [`Lfm3OmniForConditionalGeneration.transcribe_live`].")
@dataclass
class Lfm3OmniLiveTranscriptionOutput(ModelOutput):
    r"""
    sequences (`torch.LongTensor` of shape `(1, num_generated)`):
        The text tokens of this chunk, without the `<|text_end|>` closing the block or the final stop token.
    num_audio_tokens (`int`):
        Audio tokens the chunk completed.
    closed (`bool`):
        Whether the model ended the block itself, with `<|text_end|>` or a stop token on the final chunk, instead of
        exhausting `max_new_tokens`.
    state ([`Lfm3OmniLiveTranscriptionState`]):
        The stream state for the next call.
    """

    sequences: torch.LongTensor | None = None
    num_audio_tokens: int | None = None
    closed: bool | None = None
    state: Lfm3OmniLiveTranscriptionState | None = None


class Lfm3OmniRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The heads were trained with a norm that scales in fp32 and rounds once, not weight * bf16(x) like the backbone.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * self.weight).to(input_dtype)


class Lfm3OmniCodebookEmbedding(nn.Module):
    """Input embedding and tied output head of one codebook of the depth decoder."""

    def __init__(self, config: Lfm3OmniConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.codebook_size, config.depth_config.hidden_size)
        self.embedding_norm = Lfm3OmniRMSNorm(config.depth_config.hidden_size, eps=config.depth_config.norm_eps)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.embedding(codes)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(self.embedding_norm(hidden_states), self.embedding.weight)


class Lfm3OmniDepthDecoder(nn.Module):
    """
    Predicts the codebooks of one speech frame from the language model's hidden state. Every codebook gets its own
    slot of the projected hidden state plus either its known code embedding or a learned mask embedding; a
    bidirectional LFM3 model over the slots then scores the masked codebooks.
    """

    def __init__(self, config: Lfm3OmniConfig):
        super().__init__()
        self.config = config
        depth_config = config.depth_config
        self.num_codebooks = config.num_codebooks
        self.hidden_size = depth_config.hidden_size
        self.depth_linear = nn.Linear(config.text_config.hidden_size, self.num_codebooks * self.hidden_size)
        self.mask_embedding = nn.Parameter(torch.empty(self.hidden_size))
        self.model = Lfm3Model(depth_config)
        self.model.register_module("embed_tokens", None)
        self.depth_embeddings = nn.ModuleList(Lfm3OmniCodebookEmbedding(config) for _ in range(self.num_codebooks))

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [embedding(codes[..., index]) for index, embedding in enumerate(self.depth_embeddings)], dim=-2
        )

    def forward(self, hidden_states: torch.Tensor, codes: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        r"""
        hidden_states (`torch.FloatTensor` of shape `(num_frames, hidden_size)`):
            The language model's hidden state at the position preceding each frame.
        codes (`torch.LongTensor` of shape `(num_frames, num_codebooks)`):
            The codes of each frame.
        masked (`torch.BoolTensor` of shape `(num_frames, num_codebooks)`):
            The codebooks hidden from the decoder, which it is asked to predict.

        Returns:
            `torch.FloatTensor` of shape `(num_frames, num_codebooks, codebook_size)`: the logits of every codebook.
        """
        depth_in = self.depth_linear(hidden_states).unflatten(-1, (self.num_codebooks, self.hidden_size))
        code_embeds = self.embed_codes(codes)
        depth_in = depth_in + torch.where(masked[..., None], self.mask_embedding.to(code_embeds.dtype), code_embeds)
        depth_out = self.trunk(depth_in)
        return torch.stack(
            [embedding.logits(depth_out[:, index]) for index, embedding in enumerate(self.depth_embeddings)], dim=1
        )

    def trunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # all codebooks of a frame are predicted jointly, so the attention is bidirectional
        return self.model(
            inputs_embeds=hidden_states,
            attention_mask={"full_attention": None},
            is_causal=False,
            use_cache=False,
        ).last_hidden_state


def create_masks_by_layer_type(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
) -> dict:
    """The cache sizes each mask from the layer it is asked about, and the generic mask utilities ask about the first
    non-sliding layer, which for a leading `conv` layer redirects to the first attention layer of any type: name the
    first layer of each mask's own type instead."""
    config = config.get_text_config()
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    layer_types, causal_mask_mapping = config.layer_types, {}
    if "full_attention" in layer_types:
        layer_idx = layer_types.index("full_attention")
        causal_mask_mapping["full_attention"] = create_causal_mask(**mask_kwargs, layer_idx=layer_idx)
    if "sliding_attention" in layer_types:
        layer_idx = layer_types.index("sliding_attention")
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
            **mask_kwargs, layer_idx=layer_idx
        )
    if "conv" in layer_types:
        causal_mask_mapping["conv"] = create_recurrent_attention_mask(**mask_kwargs)
    return causal_mask_mapping


@auto_docstring
class Lfm3OmniPreTrainedModel(PreTrainedModel):
    config: Lfm3OmniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Lfm3OmniDepthDecoder"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    input_modalities = ("image", "video", "audio", "text")

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Lfm3OmniDepthDecoder):
            init.normal_(module.mask_embedding, std=0.02)

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        return create_masks_by_layer_type(config, inputs_embeds, attention_mask, past_key_values, position_ids)


class Lfm3OmniMultiModalProjector(nn.Module):
    """Merges each block of `downsample_factor**2` vision tokens into one language token."""

    def __init__(self, config: Lfm3OmniConfig):
        super().__init__()
        self.downsample_factor = config.downsample_factor
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.projector_layer_norm_eps)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * config.downsample_factor**2,
            config.projector_hidden_size,
            bias=config.projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.projector_hidden_size, config.text_config.hidden_size, bias=config.projector_bias
        )

    def forward(self, image_features: torch.Tensor):
        hidden_states = self.layer_norm(image_features).reshape(-1, self.linear_1.in_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Lfm3OmniDownsampler(nn.Module):
    """Three stride-2 convolutions over the (mel, time) plane: 128 mel bins and 8 frames become one vector."""

    def __init__(self, config: Lfm3OmniEncoderConfig):
        super().__init__()
        channels = config.conv_channels
        conv = {"kernel_size": 3, "stride": 2, "padding": 1}
        self.conv_in = nn.Conv2d(1, channels, **conv)
        self.conv_mid = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_mid = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_end = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_end = nn.Conv2d(channels, channels, kernel_size=1)
        self.lin_out = nn.Conv1d(config.num_mel_bins * channels // 8, config.hidden_size, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, input_features: torch.Tensor, feature_lengths: torch.Tensor | None = None) -> torch.Tensor:
        def mask_padding(hidden_states, lengths):
            if lengths is None:
                return hidden_states
            valid = torch.arange(hidden_states.shape[-1], device=hidden_states.device)[None] < lengths[:, None]
            return hidden_states * valid[:, None, None].to(hidden_states.dtype)

        hidden_states = mask_padding(input_features.unsqueeze(1), feature_lengths)
        hidden_states = self.act(self.conv_in(hidden_states))
        lengths = None if feature_lengths is None else (feature_lengths + 1) // 2
        hidden_states = mask_padding(hidden_states, lengths)
        hidden_states = self.act(self.lin_mid(self.conv_mid(hidden_states)))
        lengths = None if lengths is None else (lengths + 1) // 2
        hidden_states = mask_padding(hidden_states, lengths)
        hidden_states = self.act(self.lin_end(self.conv_end(hidden_states)))
        return self.lin_out(hidden_states.flatten(1, 2))

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return -(lengths // -8)


class Lfm3OmniEncoder(nn.Module):
    def __init__(self, config: Lfm3OmniEncoderConfig):
        super().__init__()
        self.config = config
        self.downsample = Lfm3OmniDownsampler(config)
        self.lfm = Lfm2Model(config.text_config)
        # the encoder only ever consumes spectrogram features
        self.lfm.register_module("embed_tokens", None)

    def forward(
        self, input_features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.downsample(input_features, feature_lengths).transpose(1, 2)
        # The mask bounds the window for mask-based backends; flash attention uses the kernel kwarg.
        self.lfm.config.sliding_window = self.config.sliding_window
        attention_mask = create_sliding_window_causal_mask(
            config=self.lfm.config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=None,
            position_ids=None,
        )
        outputs = self.lfm(
            inputs_embeds=hidden_states,
            attention_mask={"full_attention": attention_mask, "conv": None},
            use_cache=False,
            sliding_window=self.config.sliding_window,
        )
        return outputs.last_hidden_state, self.downsample.output_lengths(feature_lengths)

    @torch.no_grad()
    def stream(
        self,
        input_features: torch.Tensor,
        cache: Lfm3OmniEncoderStreamingCache | None = None,
        is_final: bool = False,
    ) -> tuple[torch.Tensor, Lfm3OmniEncoderStreamingCache]:
        """Encode complete stride-8 windows, with seven frames of lookahead and bounded left context."""
        if self.training:
            raise ValueError("Streaming audio encoding requires eval mode")
        if cache is None:
            cache = Lfm3OmniEncoderStreamingCache(self.config)
        if cache.finished:
            raise ValueError("This stream is finalized; start with a new cache")
        features = input_features.to(self.downsample.conv_in.weight.dtype)
        if features.ndim != 3 or features.shape[1] != self.config.num_mel_bins:
            raise ValueError("Expected features of shape (batch_size, num_mel_bins, num_frames)")
        if cache.features is not None:
            features = torch.cat((cache.features, features), dim=-1)
        num_frames = cache.num_frames + input_features.shape[-1]
        num_tokens = (num_frames + (7 if is_final else 0)) // 8
        count = num_tokens - cache.num_tokens
        if count:
            start = cache.num_tokens - cache.offset // 8
            hidden_states = self.downsample(features).transpose(1, 2)[:, start : start + count]
            positions = torch.arange(cache.num_tokens, num_tokens, device=features.device)[None]
            self.lfm.config.sliding_window = self.config.sliding_window
            mask = create_sliding_window_causal_mask(
                config=self.lfm.config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                past_key_values=cache.past_key_values,
                position_ids=positions,
                layer_idx=self.lfm.config.layer_types.index("full_attention"),
            )
            try:
                outputs = self.lfm(
                    inputs_embeds=hidden_states,
                    attention_mask={"full_attention": mask, "conv": None},
                    past_key_values=cache.past_key_values,
                    position_ids=positions,
                    use_cache=True,
                    sliding_window=self.config.sliding_window,
                )
            except Exception:
                # the layers already run have advanced the key/value cache: the stream cannot be resumed
                cache.finished = True
                raise
            encoded = outputs.last_hidden_state
        else:
            encoded = features.new_empty((features.shape[0], 0, self.config.hidden_size))
        offset = max(0, (num_tokens - 1) * 8)
        cache.features = features[..., offset - cache.offset :].clone() if not is_final else None
        cache.num_frames, cache.offset, cache.num_tokens, cache.finished = num_frames, offset, num_tokens, is_final
        return encoded, cache


class Lfm3OmniProjector(nn.Module):
    def __init__(self, config: Lfm3OmniConfig):
        super().__init__()
        in_features = config.audio_encoder_config.hidden_size
        out_features = config.text_config.hidden_size
        self.layer_norm = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(out_features, out_features)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(self.layer_norm(audio_features))))


@auto_docstring
class Lfm3OmniModel(Lfm3OmniPreTrainedModel):
    def __init__(self, config: Lfm3OmniConfig):
        super().__init__(config)
        self.language_model = Lfm3Model(config.text_config)
        self.audio_encoder = Lfm3OmniEncoder(config.audio_encoder_config)
        self.audio_projector = Lfm3OmniProjector(config)
        self.codebook_embedding = nn.Embedding(
            config.num_codebooks * config.codebook_size, config.text_config.hidden_size
        )
        self.depth_decoder = Lfm3OmniDepthDecoder(config)
        self.vision_tower = Lfm3VlVisionModel(config.vision_config)
        self.multi_modal_projector = Lfm3OmniMultiModalProjector(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @auto_docstring
    def get_audio_features(
        self, input_features: torch.FloatTensor, feature_attention_mask: torch.Tensor
    ) -> torch.FloatTensor:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_mel_bins, num_frames)`):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, num_frames)`):
            The mask of the valid spectrogram frames of each clip.

        Returns:
            `torch.FloatTensor` of shape `(num_audio_tokens, hidden_size)`: the projected speech features of all clips
            in order, one per eight spectrogram frames.
        """
        # the streaming paths refuse these shapes too; offline, a few missing mel bins would encode silently
        num_mel_bins = self.config.audio_encoder_config.num_mel_bins
        if input_features.ndim != 3 or input_features.shape[1] != num_mel_bins:
            raise ValueError(
                f"Expected input_features of shape (num_clips, {num_mel_bins}, num_frames), got "
                f"{tuple(input_features.shape)}"
            )
        if feature_attention_mask.shape != (input_features.shape[0], input_features.shape[-1]):
            raise ValueError(
                f"Expected feature_attention_mask of shape {(input_features.shape[0], input_features.shape[-1])}, "
                f"got {tuple(feature_attention_mask.shape)}"
            )
        feature_lengths = feature_attention_mask.sum(-1)
        input_features = input_features.to(self.audio_encoder.downsample.conv_in.weight.dtype)
        encoded, encoded_lengths = self.audio_encoder(input_features, feature_lengths)
        valid = torch.arange(encoded.shape[1], device=encoded.device)[None, :] < encoded_lengths[:, None]
        return self.audio_projector(encoded[valid])

    @torch.no_grad()
    def get_audio_features_stream(
        self,
        input_features: torch.FloatTensor,
        cache: Lfm3OmniEncoderStreamingCache | None = None,
        is_final: bool = False,
    ) -> tuple[torch.Tensor, Lfm3OmniEncoderStreamingCache]:
        """Project newly completed audio tokens, returning `(batch, tokens, hidden)` and the updated cache."""
        encoded, cache = self.audio_encoder.stream(input_features, cache, is_final)
        return self.audio_projector(encoded), cache

    def get_audio_out_embeds(self, audio_out_codes: torch.LongTensor) -> torch.FloatTensor:
        r"""
        audio_out_codes (`torch.LongTensor` of shape `(num_frames, num_codebooks)`):
            The codes of the speech frames in the prompt.

        Returns:
            `torch.FloatTensor` of shape `(num_frames, hidden_size)`: the sum of the codebook embeddings of each frame.
        """
        num_codebooks, codebook_size = self.config.num_codebooks, self.config.codebook_size
        if audio_out_codes.ndim != 2 or audio_out_codes.shape[-1] != num_codebooks:
            raise ValueError(
                f"Expected audio_out_codes of shape (num_frames, {num_codebooks}), got {tuple(audio_out_codes.shape)}"
            )
        # the codebooks share one table: an out-of-range code would silently read a neighbouring codebook. The check
        # is data-dependent, so a traced or captured forward skips it (`generate` prefills eagerly)
        if not is_tracing(audio_out_codes):
            torch_compilable_check(
                (audio_out_codes >= 0) & (audio_out_codes < codebook_size),
                f"audio_out_codes must be in [0, {codebook_size})",
            )
        offsets = torch.arange(num_codebooks, device=audio_out_codes.device) * codebook_size
        return self.codebook_embedding(audio_out_codes + offsets).sum(-2)

    def get_placeholder_positions(self, input_ids, inputs_embeds, token_id: int) -> torch.Tensor:
        if input_ids is not None:
            return input_ids == token_id
        token_embed = self.get_input_embeddings()(
            torch.full((), token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        return (inputs_embeds == token_embed).all(-1)

    def check_speech_placeholders(
        self, input_ids, inputs_embeds, input_features=None, audio_out_codes=None, live: bool = False
    ) -> None:
        """A speech placeholder with nothing behind it would be read as text and switch `generate` into speech mode.
        The check is data-dependent, so a compiled, exported or CUDA-graph-captured forward skips it; `generate` and
        `transcribe_live` run eagerly."""
        if is_tracing(input_ids if input_ids is not None else inputs_embeds):
            return
        for field, supplied, name in (
            ("audio_token_id", input_features is not None, "input_features"),
            ("audio_out_token_id", audio_out_codes is not None, "audio_out_codes"),
        ):
            if not supplied:
                torch_compilable_check(
                    ~self.get_placeholder_positions(input_ids, inputs_embeds, getattr(self.config, field)),
                    "Live transcription prompts cannot hold speech placeholders: the audio arrives as `input_features`"
                    if live
                    else f"{field} placeholders require `{name}`",
                )

    def get_placeholder_mask(self, input_ids, inputs_embeds, token_id: int, num_features: int) -> torch.Tensor:
        mask = self.get_placeholder_positions(input_ids, inputs_embeds, token_id)
        # data-dependent, so a traced or captured forward skips it, like `check_speech_placeholders`
        if not is_tracing(mask):
            torch_compilable_check(
                mask.sum() == num_features,
                lambda: f"Placeholder tokens and features do not match: {int(mask.sum())} tokens {token_id}, "
                f"{num_features} features",
            )
        return mask.unsqueeze(-1).to(inputs_embeds.device)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_out_codes: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3OmniModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, patch_dim)`, *optional*):
            Flattened image patches. `patch_dim` is channels times temporal patch size times squared spatial patch size.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height, and width patch counts before spatial merging, in image placeholder order.
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, patch_dim)`, *optional*):
            Flattened video patches; the tower encodes every temporal patch as its own spatial sequence.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal, height, and width patch counts before spatial merging, in video placeholder order.
        input_features (`torch.FloatTensor` of shape `(num_clips, num_mel_bins, num_frames)`, *optional*):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            The mask of the valid spectrogram frames of each clip.
        audio_out_codes (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames in the prompt, one row per `audio_out_token_id` placeholder. A speech
            placeholder without its features or codes is refused, except in a compiled or captured forward, which
            skips the check.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        # vision placeholders follow the Qwen3-VL convention and are left to the count check
        self.check_speech_placeholders(input_ids, inputs_embeds, input_features, audio_out_codes)

        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be supplied together")
        if pixel_values is not None:
            features = self.get_image_features(pixel_values, image_grid_thw).to(inputs_embeds)
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, self.config.image_token_id, features.shape[0])
            inputs_embeds = inputs_embeds.masked_scatter(mask, features)
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("pixel_values_videos and video_grid_thw must be supplied together")
        if pixel_values_videos is not None:
            features = self.get_video_features(pixel_values_videos, video_grid_thw).to(inputs_embeds)
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, self.config.video_token_id, features.shape[0])
            inputs_embeds = inputs_embeds.masked_scatter(mask, features)

        audio_features = None
        if input_features is not None:
            if feature_attention_mask is None:
                feature_attention_mask = torch.ones(
                    input_features.shape[0], input_features.shape[-1], dtype=torch.bool, device=input_features.device
                )
            audio_features = self.get_audio_features(input_features, feature_attention_mask).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, self.config.audio_token_id, audio_features.shape[0]
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
        if audio_out_codes is not None:
            audio_out_embeds = self.get_audio_out_embeds(audio_out_codes).to(inputs_embeds.device, inputs_embeds.dtype)
            audio_out_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, self.config.audio_out_token_id, audio_out_embeds.shape[0]
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_out_mask, audio_out_embeds)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return Lfm3OmniModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_features,
        )

    @auto_docstring
    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor) -> torch.Tensor:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, patch_dim)`, *optional*):
            Flattened image patches. `patch_dim` is channels times temporal patch size times squared spatial patch size.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height, and width patch counts before spatial merging, in image placeholder order.

        Returns:
            `torch.Tensor` of shape `(num_image_tokens, hidden_size)`: Projected image features in placeholder order.
        """
        vision = self.vision_tower(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        return self.multi_modal_projector(vision.last_hidden_state)

    @auto_docstring
    def get_video_features(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.LongTensor) -> torch.Tensor:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, patch_dim)`, *optional*):
            Flattened video patches; the tower encodes every temporal patch as its own spatial sequence.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal, height, and width patch counts before spatial merging, in video placeholder order.

        Returns:
            `torch.Tensor` of shape `(num_video_tokens, hidden_size)`: Projected video features in placeholder order.
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw)


class Lfm3OmniGenerationMode(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    INTERLEAVED = "interleaved"


@auto_docstring(
    custom_intro="""
    The LFM3-Omni model: an LFM3 language model that reads speech through the Liquid speech encoder, reads images
    through the LFM3-VL vision tower and writes speech as codebook frames through a depth decoder, alongside text.
    """
)
class Lfm3OmniForConditionalGeneration(Lfm3OmniPreTrainedModel, GenerationMixin):
    # Trainer counts text labels only; the joint loss also supervises speech frames.
    accepts_loss_kwargs = False
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _checkpoint_conversion_mapping = {}

    # `GenerationConfig` fields the text/speech mode machine does not implement, with their inert values
    _generation_fields_unsupported = {
        "min_length": 0,
        "min_new_tokens": None,
        "max_time": None,
        "stop_strings": None,
        "early_stopping": False,
        "penalty_alpha": None,
        "dola_layers": None,
        "use_cache": True,
        "cache_implementation": None,
        "cache_config": None,
        "return_legacy_cache": None,
        "prefill_chunk_size": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "diversity_penalty": 0.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "force_words_ids": None,
        "renormalize_logits": False,
        "remove_invalid_values": False,
        "top_h": None,
        "encoder_repetition_penalty": 1.0,
        "use_mtp": None,
        "constraints": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "exponential_decay_length_penalty": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "sequence_bias": None,
        "token_healing": False,
        "guidance_scale": None,
        "watermarking_config": None,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "output_logits": None,
        "encoder_no_repeat_ngram_size": 0,
        "prompt_lookup_num_tokens": None,
        "assistant_early_exit": None,
        "compile_config": None,
    }

    def __init__(self, config: Lfm3OmniConfig):
        super().__init__(config)
        self.model = Lfm3OmniModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    @auto_docstring
    def get_audio_features(self, input_features: torch.FloatTensor, feature_attention_mask: torch.Tensor):
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_mel_bins, num_frames)`):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, num_frames)`):
            The mask of the valid spectrogram frames of each clip.
        """
        return self.model.get_audio_features(input_features, feature_attention_mask)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def _placeholder_token_ids(self) -> list[int]:
        return [
            self.config.audio_token_id,
            self.config.audio_out_token_id,
            self.config.image_token_id,
            self.config.video_token_id,
        ]

    @property
    def depth_decoder(self):
        return self.model.depth_decoder

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_out_codes: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        audio_labels: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3OmniCausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, patch_dim)`, *optional*):
            Flattened image patches. `patch_dim` is channels times temporal patch size times squared spatial patch size.
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, patch_dim)`, *optional*):
            Flattened video patches; the tower encodes every temporal patch as its own spatial sequence.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal, height, and width patch counts before spatial merging, in video placeholder order.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height, and width patch counts before spatial merging, in image placeholder order.
        input_features (`torch.FloatTensor` of shape `(num_clips, num_mel_bins, num_frames)`, *optional*):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            The mask of the valid spectrogram frames of each clip.
        audio_out_codes (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames in the sequence, one row per `audio_out_token_id` placeholder. A speech
            placeholder without its features or codes is refused, except in a compiled or captured forward, which
            skips the check.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        audio_labels (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames to learn, one row per `audio_out_token_id` placeholder; rows of `-100` are
            ignored. The depth decoder predicts a random subset of each frame's codebooks from the others and the
            hidden state preceding the frame.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_out_codes=audio_out_codes,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = text_loss = audio_loss = None
        if labels is not None:
            text_loss = (
                self.loss_function(
                    logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
                )
                if (labels[..., 1:] != -100).any()
                else logits.sum() * 0.0
            )
        if audio_labels is not None:
            if input_ids is None:
                raise ValueError("`audio_labels` requires `input_ids` to locate the speech frames")
            audio_loss = self.audio_loss(input_ids, hidden_states, audio_labels)

        if text_loss is not None or audio_loss is not None:
            loss = self.combined_loss(input_ids, labels, audio_labels, text_loss, audio_loss)

        return Lfm3OmniCausalLMOutputWithPast(
            loss=loss,
            text_loss=text_loss,
            audio_loss=audio_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=outputs.audio_hidden_states,
        )

    def audio_loss(
        self, input_ids: torch.LongTensor, hidden_states: torch.FloatTensor, audio_labels: torch.LongTensor
    ) -> torch.FloatTensor:
        config = self.config
        # under `device_map` sharding the last hidden states sit on the last layer's device, not the inputs'
        input_ids, audio_labels = input_ids.to(hidden_states.device), audio_labels.to(hidden_states.device)
        placeholders = input_ids == config.audio_out_token_id
        if audio_labels.shape != (int(placeholders.sum()), config.num_codebooks):
            raise ValueError(
                f"Expected audio_labels of shape (num_frames, {config.num_codebooks}), one row per "
                f"`audio_out_token_id` placeholder, got {tuple(audio_labels.shape)}"
            )
        # rows of -100 are ignored; any other code indexes the codebook embeddings and the loss
        ignored = (audio_labels == -100).all(-1, keepdim=True)
        torch_compilable_check(
            ignored | ((audio_labels >= 0) & (audio_labels < config.codebook_size)),
            f"audio_labels must be in [0, {config.codebook_size}) or rows of -100",
        )
        # a frame is predicted from the position before it, so the first position of a row has no frame
        preceding = placeholders.clone()
        preceding[:, 0] = False
        supervised = (audio_labels != -100).all(-1)[preceding[placeholders]]
        latents = hidden_states[preceding.roll(-1, dims=1)][supervised]
        codes = audio_labels[preceding[placeholders]][supervised]
        if codes.shape[0] == 0:
            return hidden_states.sum() * 0.0
        ratio = torch.rand(codes.shape[0], 1, device=codes.device)
        masked = torch.rand(codes.shape, device=codes.device) < ratio
        masked |= ~masked.any(-1, keepdim=True)
        masked |= (codes == config.audio_eos_code).all(-1, keepdim=True)
        logits = self.model.depth_decoder(latents, codes, masked)
        # under `device_map` sharding the depth decoder answers on its own device
        codes, masked = codes.to(logits.device), masked.to(logits.device)
        losses = F.cross_entropy(logits.transpose(1, 2).float(), codes, reduction="none")
        return (losses * masked).sum() / masked.sum().clamp_min(1)

    def combined_loss(self, input_ids, labels, audio_labels, text_loss, audio_loss) -> torch.Tensor:
        """Pooled mean over the supervised text tokens and speech frames, each modality's mean weighted by its
        count and its configured weight, as in training."""
        # under `device_map` sharding the labels and the two losses can sit on three devices
        device = (text_loss if text_loss is not None else audio_loss).device
        text_count = 0 if labels is None else (labels[..., 1:] != -100).sum().to(device)
        audio_count = 0
        if audio_labels is not None:
            placeholders = input_ids.to(audio_labels.device) == self.config.audio_out_token_id
            preceding = placeholders.clone()
            preceding[:, 0] = False
            audio_count = (audio_labels != -100).all(-1)[preceding[placeholders]].sum().to(device)
        text_weight = self.config.text_loss_weight * text_count
        audio_weight = self.config.audio_loss_weight * audio_count
        return (
            text_weight * (text_loss if text_loss is not None else 0.0)
            + audio_weight * (audio_loss.to(device) if audio_loss is not None else 0.0)
        ) / (text_weight + audio_weight + 1e-6)

    @staticmethod
    def _build_warpers(
        temperature: float | None, top_k: int | None, top_p: float | None = None, min_p: float | None = None
    ) -> list:
        """The logits warpers of one sampling setting, empty for greedy decoding; building them validates the values."""
        # an explicit zero temperature is greedy; without a temperature, top_k alone samples at unit temperature
        if temperature == 0 or (temperature is None and not top_k):
            return []
        warpers = []
        if temperature:
            warpers.append(TemperatureLogitsWarper(float(temperature)))
        if top_k:
            warpers.append(TopKLogitsWarper(int(top_k)))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(float(top_p)))
        if min_p:
            warpers.append(MinPLogitsWarper(float(min_p)))
        return warpers

    @classmethod
    def _sample(
        cls,
        logits: torch.Tensor,
        temperature: float | None,
        top_k: int | None,
        repetition_penalty: float | None = None,
        history: torch.Tensor | None = None,
        suppress: list[int] = (),
        top_p: float | None = None,
        min_p: float | None = None,
        warpers: list | None = None,
    ) -> torch.Tensor:
        scores = logits.float()[None].clone()
        if repetition_penalty is not None and repetition_penalty != 1.0 and history is not None:
            penalty = RepetitionPenaltyLogitsProcessor(float(repetition_penalty))
            scores = penalty(history.reshape(1, -1).to(scores.device), scores)
        if suppress:
            scores[:, list(suppress)] = -float("inf")
        if warpers is None:
            warpers = cls._build_warpers(temperature, top_k, top_p, min_p)
        if not warpers:
            return scores.argmax(-1)
        for warper in warpers:
            scores = warper(None, scores)
        return torch.multinomial(scores.softmax(-1), 1)[0]

    def sample_audio_frame(
        self,
        hidden_state: torch.Tensor,
        temperature: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        history: torch.Tensor | list[torch.Tensor | None] | None = None,
        warpers: list | None = None,
    ) -> torch.LongTensor:
        r"""
        Samples the codes of one speech frame from a language model hidden state of shape `(hidden_size,)` or `(batch_size, hidden_size)`: the
        codebooks are revealed in a random order over the passes of `config.maskgit_schedule`, each pass scoring the
        codebooks still hidden from the depth decoder. `history` holds the previous frames, `(num_frames,
        num_codebooks)`, for the repetition penalty, or a list of per-row histories for a batch. `warpers` are the
        logits warpers of `temperature` and `top_k`, prebuilt by `_build_warpers`.
        """
        single = hidden_state.ndim == 1
        if single:
            hidden_state = hidden_state[None]
            history = [history]
        batch_size = hidden_state.shape[0]
        if history is None:
            history = [None] * batch_size
        num_codebooks = self.config.num_codebooks
        codes = hidden_state.new_zeros((batch_size, num_codebooks), dtype=torch.long)
        revealed = torch.zeros_like(codes, dtype=torch.bool)
        order = torch.stack([torch.randperm(num_codebooks, device=hidden_state.device) for _ in range(batch_size)])
        for pick in order.split(self.config.maskgit_schedule, dim=1):
            # the depth decoder's forward keeps every pass on its own device under `device_map` sharding or offload
            logits = self.model.depth_decoder(hidden_state, codes, ~revealed)
            for row in range(batch_size):
                for index in pick[row].tolist():
                    codebook_history = None if history[row] is None else history[row][:, index]
                    codes[row, index] = self._sample(
                        logits[row, index], temperature, top_k, repetition_penalty, codebook_history, warpers=warpers
                    ).to(codes.device)
            revealed.scatter_(1, pick, True)
        return codes[0] if single else codes

    @torch.no_grad()
    def prefill_audio(
        self,
        input_features: torch.FloatTensor,
        past_key_values: Cache | None = None,
        encoder_cache: Lfm3OmniEncoderStreamingCache | None = None,
        attention_mask: torch.Tensor | None = None,
        is_final: bool = False,
    ) -> Lfm3OmniStreamingOutput:
        """
        Encode a mel chunk and append completed audio tokens to the language-model cache.

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                New mel frames, without padding. Batched streams must have synchronized chunk lengths.
            past_key_values (`Cache`, *optional*):
                Cache from a text-prefix forward pass or the preceding `prefill_audio` call. Updated in place.
            encoder_cache (`Lfm3OmniEncoderStreamingCache`, *optional*):
                Encoder state from the preceding call. Omit at the start of each new utterance.
            attention_mask (`torch.Tensor` of shape `(batch_size, cached_sequence_length)`, *optional*):
                Mask of the tokens already in `past_key_values`, required for a nonempty cache. Pass the returned
                mask to the next call. Audio tokens are appended as valid positions.
            is_final (`bool`, *optional*, defaults to `False`):
                Flush the encoder's trailing token and finalize this utterance, including for an empty final chunk.

        Returns:
            [`Lfm3OmniStreamingOutput`]: Updated encoder state, language-model cache and full attention mask.
            Chunks that complete no token leave the language-model cache unchanged. After finalization, pass the
            closing audio/user delimiters and assistant prefix to `generate` with this cache and a mask extended
            for those new tokens. Returned generation sequences include only that suffix and the generated tokens.
        """
        if self.training:
            raise ValueError("Streaming audio prefill requires eval mode")
        if input_features.ndim != 3 or input_features.shape[1] != self.config.audio_encoder_config.num_mel_bins:
            raise ValueError("Expected features of shape (batch_size, num_mel_bins, num_frames)")
        batch_size = input_features.shape[0]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if attention_mask is None:
            if past_length:
                raise ValueError("Pass attention_mask covering the tokens already in past_key_values")
            attention_mask = torch.ones((batch_size, 0), dtype=torch.long, device=input_features.device)
        if attention_mask.shape != (batch_size, past_length):
            raise ValueError("attention_mask must cover exactly the tokens already in past_key_values")
        not_binary = ((attention_mask != 0) & (attention_mask != 1)).any()
        not_monotonic = ((attention_mask[:, 1:].long() - attention_mask[:, :-1].long()) < 0).any()
        if bool(not_binary | not_monotonic):
            raise ValueError("attention_mask must describe contiguous left-padded prompts")
        if past_length and not bool(attention_mask[:, -1].all()):
            raise ValueError("Each cached prompt must end in a valid token")
        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config.text_config)
        embeddings, encoder_cache = self.model.get_audio_features_stream(input_features, encoder_cache, is_final)
        num_tokens = embeddings.shape[1]
        hidden_states = None
        if num_tokens:
            attention_mask = torch.cat((attention_mask, attention_mask.new_ones((batch_size, num_tokens))), dim=-1)
            position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)[:, -num_tokens:]
            outputs = self.model.language_model(
                inputs_embeds=embeddings,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            hidden_states, past_key_values = outputs.last_hidden_state, outputs.past_key_values
        return Lfm3OmniStreamingOutput(
            past_key_values=past_key_values,
            encoder_cache=encoder_cache,
            attention_mask=attention_mask,
            last_hidden_state=hidden_states,
        )

    def _live_forward(self, state: Lfm3OmniLiveTranscriptionState, inputs_embeds: torch.FloatTensor) -> torch.Tensor:
        past_length = state.attention_mask.shape[1]
        state.attention_mask = torch.cat(
            (state.attention_mask, state.attention_mask.new_ones((1, inputs_embeds.shape[1]))), dim=-1
        )
        position_ids = torch.arange(past_length, state.attention_mask.shape[1], device=inputs_embeds.device)[None]
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            past_key_values=state.past_key_values,
            attention_mask=state.attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        return outputs.last_hidden_state[:, -1]

    @torch.no_grad()
    def transcribe_live(
        self,
        input_features: torch.FloatTensor,
        state: Lfm3OmniLiveTranscriptionState | None = None,
        input_ids: torch.LongTensor | None = None,
        is_final: bool = False,
        max_new_tokens: int = 48,
        suppress_token_ids: list[int] | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> Lfm3OmniLiveTranscriptionOutput:
        """
        Transcribe one chunk of a live stream with a checkpoint trained for live ASR.

        The audio goes inside the assistant turn in chunks of `config.interleaved_n_audio` encoder tokens, each
        followed by the text it completes:

            <prompt> [A x 12] text<|text_end|> [A x 12] text<|text_end|> ... [A x r] <|mixed_end|> text<|text_end|><|im_end|>

        Args:
            input_features (`torch.FloatTensor` of shape `(1, num_mel_bins, num_frames)`):
                The chunk's new mel frames, normally `8 * config.interleaved_n_audio`. A chunk that completes no
                audio token decodes no text.
            state ([`Lfm3OmniLiveTranscriptionState`], *optional*):
                The state returned by the previous call. Omit for a new stream.
            input_ids (`torch.LongTensor` of shape `(1, prompt_length)`, *optional*):
                The prompt of a new language-model context (the system turn and the assistant header), required
                for a new stream. With a `state`, it starts the context over while the encoder keeps following the
                stream, the way training samples of at most 30 seconds are chained.
            is_final (`bool`, *optional*, defaults to `False`):
                Flush the encoder and decode the closing block, which ends at `<|text_end|>` or a stop token.
            max_new_tokens (`int`, *optional*, defaults to 48):
                Text budget of the block; a block that exhausts it is still closed with `<|text_end|>`.
            suppress_token_ids (`list[int]`, *optional*):
                Tokens never valid before the final chunk, on top of `<|audio_start|>`, `<|mixed_start|>` and the
                end-of-turn token: pass `<|mixed_end|>` and `<|endoftext|>`.
            stop_token_ids (`list[int]`, *optional*):
                Tokens ending the final block on top of the end-of-turn token: pass `<|endoftext|>` and `<|im_start|>`.

        Returns:
            [`Lfm3OmniLiveTranscriptionOutput`]: the chunk's text tokens (the final block starts with
            `<|mixed_end|>`), the number of audio tokens it completed, and the updated state.
        """
        if self.training:
            raise ValueError("Live transcription requires eval mode")
        num_mel_bins = self.config.audio_encoder_config.num_mel_bins
        if input_features.ndim != 3 or input_features.shape[0] != 1 or input_features.shape[1] != num_mel_bins:
            raise ValueError(
                f"Live transcription takes one stream: features of shape (1, {num_mel_bins}, num_frames), got "
                f"{tuple(input_features.shape)}"
            )
        if input_ids is not None:
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise ValueError("Live transcription takes one prompt of shape (1, prompt_length)")
            self.model.check_speech_placeholders(input_ids, None, live=True)
        if state is None:
            if input_ids is None:
                raise ValueError("A new stream needs the prompt `input_ids`")
            state = Lfm3OmniLiveTranscriptionState()
        if state.finished:
            raise ValueError("This stream is finalized; start with a new state")
        eos_token_ids = self.config.text_config.eos_token_id
        eos_token_ids = (
            [] if eos_token_ids is None else [eos_token_ids] if isinstance(eos_token_ids, int) else list(eos_token_ids)
        )
        text_end_token_id = self.config.text_end_token_id
        suppress = [
            token_id
            for token_id in (
                self.config.audio_start_token_id,
                self.config.mixed_start_token_id,
                *eos_token_ids,
                *(suppress_token_ids or []),
            )
            # `<|text_end|>` closes every block, so it is never suppressed even when it is listed as an end-of-turn id
            if token_id is not None and token_id != text_end_token_id
        ]
        # the ids index the logits; refused before the encoder cache advances
        vocab_size = self.config.text_config.vocab_size
        if any(not 0 <= token_id < vocab_size for token_id in suppress):
            raise ValueError(f"Suppressed token ids must lie in [0, {vocab_size}), got {suppress}")
        stop = {*eos_token_ids, *(stop_token_ids or [])}
        # the encoder runs before a restart replaces the context, so a rejected chunk leaves the context intact
        embeddings, state.encoder_cache = self.model.get_audio_features_stream(
            input_features, state.encoder_cache, is_final
        )
        embed_tokens = self.get_input_embeddings()
        # an offloaded table sits on `meta`: the device comes from the embedding output it is joined with
        embeddings = embeddings.to(dtype=embed_tokens.weight.dtype)
        if input_ids is not None:
            state.past_key_values = DynamicCache(config=self.config.text_config)
            state.attention_mask = torch.zeros((1, 0), dtype=torch.long, device=input_features.device)
            state.pending = embed_tokens(input_ids)
        num_audio_tokens = embeddings.shape[1]
        generated, closed = [], False
        if num_audio_tokens or is_final:
            # the pending text goes in with the audio it precedes, so a chunk completing no token leaves it pending
            inputs_embeds = (
                embeddings
                if state.pending is None
                else torch.cat((state.pending, embeddings.to(state.pending.device)), dim=1)
            )
            logits = self.lm_head(self._live_forward(state, inputs_embeds))
            state.pending = None
            # the placeholders are masked on every chunk, the final one included
            placeholder_ids = torch.tensor(self._placeholder_token_ids, device=logits.device)
            suppress = torch.tensor(suppress, dtype=torch.long, device=logits.device)
            for _ in range(max_new_tokens):
                logits[:, placeholder_ids] = float("-inf")
                if not is_final:
                    logits[:, suppress] = float("-inf")
                next_token = logits.argmax(-1, keepdim=True)
                token = int(next_token)
                if token == text_end_token_id or (is_final and token in stop):
                    closed = True
                    break
                generated.append(token)
                logits = self.lm_head(self._live_forward(state, embed_tokens(next_token)))
            if is_final:
                state.finished = True
            else:
                # the block closes with `<|text_end|>`, fed together with the next chunk's audio
                state.pending = embed_tokens(torch.tensor([[text_end_token_id]], device=input_features.device))
        return Lfm3OmniLiveTranscriptionOutput(
            sequences=torch.tensor([generated], dtype=torch.long, device=input_features.device),
            num_audio_tokens=num_audio_tokens,
            closed=closed,
            state=state,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_out_codes: torch.LongTensor | None = None,
        generation_config: GenerationConfig | None = None,
        generation_mode: Lfm3OmniGenerationMode | str | list[str] | None = None,
        audio_temperature: float | None = None,
        audio_top_k: int | None = None,
        audio_repetition_penalty: float | None = None,
        audio_repetition_window: int | None = None,
        text_repetition_window: int | None = None,
        suppress_mixed: bool = False,
        suppress_audio_out: bool = False,
        return_audio_eos: bool = False,
        streamer: BaseStreamer | list[BaseStreamer | None] | None = None,
        audio_streamer: BaseStreamer | list[BaseStreamer | None] | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> Lfm3OmniGenerateOutput | torch.LongTensor:
        r"""
        Generates text and speech for a batch of left-padded prompts.

        Text is the default output. `<|audio_start|>` switches to speech until the depth decoder emits an
        end-of-audio frame, `<|mixed_start|>` to blocks of `config.interleaved_n_text` text tokens and
        `config.interleaved_n_audio` speech frames until `<|text_end|>` closes the text and the remaining speech
        ends with an end-of-audio frame. A prompt ending on `<|audio_start|>`, a speech frame or `<|mixed_start|>`
        starts in that mode. Generation stops at `<|im_end|>` emitted in text mode or after `max_new_tokens`
        generated slots, a speech frame counting as one slot.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The uncached prompt tokens, with one `audio_token_id` per encoded frame of each input clip and one
                `audio_out_token_id` per speech frame. With `past_key_values`, supply only the new suffix; returned
                sequences contain this suffix and generated tokens, excluding the cached prefix.
            input_features (`torch.FloatTensor` of shape `(num_clips, num_mel_bins, num_frames)`, *optional*):
                The log-mel spectrograms of the input clips.
            feature_attention_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
                The mask of the valid spectrogram frames of each clip.
            audio_out_codes (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
                The codes of the speech frames in the prompt.
            generation_config ([`GenerationConfig`], *optional*):
                `max_new_tokens`, `do_sample`, `temperature`, `top_k`, `top_p`, `min_p` and `repetition_penalty`
                apply to the text tokens; the penalty covers the last `text_repetition_window` generated tokens.
                Without `max_new_tokens`, an explicit `max_length` bounds the prompt plus the generated slots and
                must exceed the prompt; the default `max_length` counts generated slots alone, as in
                [`GenerationMixin`].
                Fields the text/speech mode machine does not implement (beam search, contrastive search,
                `stop_strings`, `output_*`, `use_cache=False`, cache implementations, ...) raise instead of being
                ignored. The bare call returns [`Lfm3OmniGenerateOutput`]. `return_dict_in_generate` is read from
                the call, then a passed configuration, then the model's `generation_config`: `False` returns the
                sequences alone, and so does a passed configuration when none of them sets it, like
                [`GenerationMixin`], as pipelines expect.
            generation_mode (`str` or `list[str]`, *optional*):
                `"text"`, `"audio"` or `"interleaved"`, or one mode per row; inferred per row when omitted. An
                explicit `"text"` keeps the reply in text by forbidding `<|mixed_start|>` and `<|audio_start|>`; an
                inferred text mode lets the model switch to speech.
            audio_temperature (`float`, *optional*):
                The sampling temperature of the speech codes; `0` picks the most likely code. This and the four
                arguments below are read from `generation_config` when the call omits them, so a checkpoint's
                `generation_config.json` can carry the speech sampling settings.
            audio_top_k (`int`, *optional*):
                The top-k filtering of the speech codes.
            audio_repetition_penalty (`float`, *optional*):
                The penalty on the codes each codebook emitted in the last `audio_repetition_window` frames.
            audio_repetition_window (`int`, *optional*):
                The number of past frames the audio repetition penalty covers; 64 when nothing sets it.
            text_repetition_window (`int`, *optional*):
                The number of past tokens the text repetition penalty covers; 32 when nothing sets it.
            suppress_mixed (`bool`, *optional*, defaults to `False`):
                Whether to forbid `<|mixed_start|>`.
            suppress_audio_out (`bool`, *optional*, defaults to `False`):
                Whether to forbid `<|audio_start|>`.
            return_audio_eos (`bool`, *optional*, defaults to `False`):
                Whether to keep the end-of-audio frames in `audio_codes` and their placeholders in `sequences`, so
                the output can be fed back as a prompt; they must be dropped before decoding with the codec.
            streamer ([`~generation.BaseStreamer`], *optional*):
                Receives the prompt and then every generated text token. For batches, pass one streamer per row.
            audio_streamer ([`~generation.BaseStreamer`], *optional*):
                Receives every generated speech frame, `(num_codebooks,)`, as soon as it is sampled. For batches,
                pass one streamer per row. An entry may be `None`.
            past_key_values (`Cache`, *optional*):
                Cache from a prefix forward pass or `prefill_audio`, updated in place to cover the prompt and every
                generated slot, end-of-audio frames included (`return_audio_eos=True` returns them); a batch whose rows
                finished at different steps holds masked slots and cannot be continued. Pass `attention_mask`
                covering both the cached prefix and `input_ids`. Positions are computed from that full mask unless
                supplied.
        """
        # a consumer blocked on a streamer must see the end whichever statement raises, argument checks included
        streams = list(
            {
                id(stream): stream
                for value in (streamer, audio_streamer)
                for stream in (value if isinstance(value, (list, tuple)) else (value,))
                if stream is not None
            }.values()
        )
        try:
            output = self._generate(
                input_ids,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                audio_out_codes=audio_out_codes,
                generation_config=generation_config,
                generation_mode=generation_mode,
                audio_temperature=audio_temperature,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
                audio_repetition_window=audio_repetition_window,
                text_repetition_window=text_repetition_window,
                suppress_mixed=suppress_mixed,
                suppress_audio_out=suppress_audio_out,
                return_audio_eos=return_audio_eos,
                streamer=streamer,
                audio_streamer=audio_streamer,
                past_key_values=past_key_values,
                **kwargs,
            )
        except BaseException:
            self._end_streamers(streams, raise_errors=False)
            raise
        self._end_streamers(streams)
        return output

    @staticmethod
    def _end_streamers(streams: list, raise_errors: bool = True) -> None:
        # every consumer is released before an `end()` failure surfaces
        errors = []
        for stream in streams:
            try:
                stream.end()
            except Exception as error:
                errors.append(error)
        if errors and raise_errors:
            raise errors[0]

    def _generate(
        self,
        input_ids: torch.LongTensor,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_out_codes: torch.LongTensor | None = None,
        generation_config: GenerationConfig | None = None,
        generation_mode: Lfm3OmniGenerationMode | str | list[str] | None = None,
        audio_temperature: float | None = None,
        audio_top_k: int | None = None,
        audio_repetition_penalty: float | None = None,
        audio_repetition_window: int | None = None,
        text_repetition_window: int | None = None,
        suppress_mixed: bool = False,
        suppress_audio_out: bool = False,
        return_audio_eos: bool = False,
        streamer: BaseStreamer | list[BaseStreamer | None] | None = None,
        audio_streamer: BaseStreamer | list[BaseStreamer | None] | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> Lfm3OmniGenerateOutput | torch.LongTensor:
        if input_ids.ndim != 2 or not input_ids.shape[0] or not input_ids.shape[1]:
            raise ValueError("input_ids must be a nonempty batch of prompts")
        batch_size = input_ids.shape[0]
        for name in (
            "logits_processor",
            "stopping_criteria",
            "prefix_allowed_tokens_fn",
            "assistant_model",
            "negative_prompt_ids",
            "negative_prompt_attention_mask",
            "custom_generate",
        ):
            if kwargs.get(name) is not None:
                raise ValueError(f"{type(self).__name__}.generate does not support `{name}`")
        # rows stop independently, so ZeRO-3 / FSDP ranks would leave the collective forwards at different steps
        if kwargs.pop("synced_gpus", None):
            raise ValueError(f"{type(self).__name__}.generate does not support `synced_gpus=True`")
        # call over passed configuration over `self.generation_config`; only a bare call defaults to the output object
        return_dict = kwargs.get("return_dict_in_generate")
        if return_dict is None and generation_config is not None:
            return_dict = generation_config.return_dict_in_generate
        if return_dict is None:
            return_dict = self.generation_config.return_dict_in_generate
        plain_sequences = return_dict is False or (return_dict is None and generation_config is not None)
        has_default_max_length = (
            kwargs.get("max_length") is None
            and (generation_config is None or generation_config.max_length is None)
            and self.generation_config.max_length is None
        )
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        unsupported = sorted(
            name
            for name, inert in self._generation_fields_unsupported.items()
            if getattr(generation_config, name, inert) not in (inert, None)
        )
        if unsupported:
            raise ValueError(
                f"{type(self).__name__}.generate does not implement {unsupported}; see its docstring for the "
                "supported fields"
            )

        def speech_setting(name, value, default=None):
            # the call, then the generation configuration, like the text sampling fields
            if value is None:
                value = getattr(generation_config, name, None)
            return default if value is None else value

        audio_temperature = speech_setting("audio_temperature", audio_temperature)
        audio_top_k = speech_setting("audio_top_k", audio_top_k)
        audio_repetition_penalty = speech_setting("audio_repetition_penalty", audio_repetition_penalty)
        audio_repetition_window = speech_setting("audio_repetition_window", audio_repetition_window, 64)
        text_repetition_window = speech_setting("text_repetition_window", text_repetition_window, 32)
        attention_mask = model_kwargs.pop("attention_mask", None)
        # only a caller that keeps the cache pays the decode step of the last sampled slot
        caller_cache = past_key_values is not None
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if attention_mask is None and past_length:
            raise ValueError("Cached generation requires attention_mask covering the cached prefix and input_ids")
        if attention_mask is None:
            pad_id = generation_config.pad_token_id
            if pad_id is None:
                pad_id = self.config.text_config.pad_token_id
            eos_id = generation_config.eos_token_id
            if eos_id is None:
                eos_id = self.config.text_config.eos_token_id
            eos_ids = list(eos_id) if isinstance(eos_id, (list, tuple)) else [eos_id]
            attention_mask = (
                input_ids.ne(pad_id).long()
                if pad_id is not None and pad_id not in eos_ids
                else torch.ones_like(input_ids)
            )
        position_ids = model_kwargs.pop("position_ids", None)
        self._validate_model_kwargs(model_kwargs.copy())
        if generation_config.num_beams != 1 or generation_config.num_return_sequences != 1:
            raise ValueError(f"{type(self).__name__} supports one greedy or sampled continuation per input row")
        if attention_mask.shape != (batch_size, past_length + input_ids.shape[1]) or not bool(
            attention_mask[:, -1].all()
        ):
            raise ValueError("Use left-padded prompts and an attention_mask covering the cached prefix and input_ids")
        not_binary = ((attention_mask != 0) & (attention_mask != 1)).any()
        not_monotonic = ((attention_mask[:, 1:].long() - attention_mask[:, :-1].long()) < 0).any()
        if bool(not_binary | not_monotonic):
            raise ValueError("attention_mask must describe contiguous left-padded prompts of zeros and ones")
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)[:, -input_ids.shape[1] :]
        if position_ids.shape not in ((1, input_ids.shape[1]), input_ids.shape):
            raise ValueError("position_ids must cover the uncached input_ids, with one row or one row per prompt")
        if audio_repetition_window < 0 or text_repetition_window < 0:
            raise ValueError("Repetition windows must be nonnegative")
        text_temperature = generation_config.temperature if generation_config.do_sample else None
        text_top_k = generation_config.top_k if generation_config.do_sample else None
        text_top_p = generation_config.top_p if generation_config.do_sample else None
        text_min_p = generation_config.min_p if generation_config.do_sample else None
        config = self.config
        pad_token_id = generation_config.pad_token_id
        if pad_token_id is None:
            pad_token_id = config.text_config.pad_token_id
        if pad_token_id is None:
            pad_token_id = config.text_config.eos_token_id
        if isinstance(pad_token_id, (list, tuple)):
            pad_token_id = pad_token_id[0]
        if pad_token_id is None:
            raise ValueError("Set pad_token_id to assemble generated sequences")
        eos_ids = generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = config.text_config.eos_token_id
        eos_ids = list(eos_ids) if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        suppress = self._placeholder_token_ids + [
            token_id
            for flag, token_id in (
                (suppress_mixed, config.mixed_start_token_id),
                (suppress_audio_out, config.audio_start_token_id),
            )
            if flag
        ]
        text_only_suppress = suppress + [config.mixed_start_token_id, config.audio_start_token_id]
        text_only = [False] * batch_size
        if generation_mode is None:
            modes = [
                Lfm3OmniGenerationMode.AUDIO
                if token in (config.audio_start_token_id, config.audio_out_token_id)
                else Lfm3OmniGenerationMode.INTERLEAVED
                if token == config.mixed_start_token_id
                else Lfm3OmniGenerationMode.TEXT
                for token in input_ids[:, -1].tolist()
            ]
        else:
            modes = generation_mode if isinstance(generation_mode, list) else [generation_mode] * batch_size
            if len(modes) != batch_size:
                raise ValueError("generation_mode must have one entry per input row")
            modes = [Lfm3OmniGenerationMode(mode) for mode in modes]
            text_only = [mode is Lfm3OmniGenerationMode.TEXT for mode in modes]
            # speech always follows its marker in training; append the one the requested mode needs
            markers = {
                Lfm3OmniGenerationMode.AUDIO: (config.audio_start_token_id, config.audio_out_token_id),
                Lfm3OmniGenerationMode.INTERLEAVED: (config.mixed_start_token_id,),
            }
            needed = [
                markers[mode][0] if mode in markers and token not in markers[mode] else None
                for mode, token in zip(modes, input_ids[:, -1].tolist())
            ]
            if len(set(needed)) > 1:
                raise ValueError("Rows need different generation markers; end each prompt with its marker instead")
            if needed[0] is not None:
                marker = input_ids.new_full((batch_size, 1), needed[0])
                input_ids = torch.cat((input_ids, marker), 1)
                attention_mask = torch.cat((attention_mask, attention_mask.new_ones((batch_size, 1))), 1)
                position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), 1)

        def row_streamers(value):
            if value is None:
                return [None] * batch_size
            if isinstance(value, (list, tuple)) and len(value) == batch_size:
                return list(value)
            if batch_size == 1 and not isinstance(value, (list, tuple)):
                return [value]
            raise ValueError("Pass one streamer per input row")

        text_streams, audio_streams = row_streamers(streamer), row_streamers(audio_streamer)
        # the appended marker is part of the prompt, so `max_length` counts it
        max_new_tokens = generation_config.max_new_tokens
        if max_new_tokens is None and has_default_max_length:
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
            max_new_tokens = generation_config.max_length
        elif max_new_tokens is None:
            prompt_length = past_length + input_ids.shape[1]
            if prompt_length >= generation_config.max_length:
                raise ValueError(
                    f"Input length of input_ids is {prompt_length}, but `max_length` is set to "
                    f"{generation_config.max_length}. Increase `max_length` or set `max_new_tokens`."
                )
            max_new_tokens = generation_config.max_length - prompt_length
        # the warper constructors validate the sampling values: build them before the prefill mutates a passed
        # cache
        text_warpers = self._build_warpers(text_temperature, text_top_k, text_top_p, text_min_p)
        audio_warpers = self._build_warpers(audio_temperature, audio_top_k)
        for penalty in (generation_config.repetition_penalty, audio_repetition_penalty):
            if penalty is not None and penalty != 1.0:
                RepetitionPenaltyLogitsProcessor(float(penalty))
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_out_codes=audio_out_codes,
            use_cache=True,
            past_key_values=past_key_values
            if past_key_values is not None
            else DynamicCache(config=config.text_config),
            **model_kwargs,
        )
        # Prefill has validated the flattened frame/placeholder mapping. A replayed terminal frame resumes
        # the text phase, just as it does when sampled inside this call; explicit mode overrides still win.
        if generation_mode is None and audio_out_codes is not None:
            last_frames = (input_ids == config.audio_out_token_id).sum(-1).cumsum(0) - 1
            for row, last_frame in enumerate(last_frames.tolist()):
                if input_ids[row, -1] == config.audio_out_token_id:
                    if bool((audio_out_codes[last_frame] == config.audio_eos_code).any()):
                        modes[row] = Lfm3OmniGenerationMode.TEXT
        hidden, cache = outputs.last_hidden_state[:, -1], outputs.past_key_values
        embed_tokens = self.get_input_embeddings()
        for row, stream in enumerate(text_streams):
            if stream is not None:
                stream.put(input_ids[row : row + 1])
        text_out = [[] for _ in range(batch_size)]
        frames = [[] for _ in range(batch_size)]
        text_history = [deque(maxlen=text_repetition_window) for _ in range(batch_size)]
        audio_history = [deque(maxlen=audio_repetition_window) for _ in range(batch_size)]
        slots_left = [config.interleaved_n_text] * batch_size
        speaking, text_done, finished = [False] * batch_size, [False] * batch_size, [False] * batch_size
        for step in range(max_new_tokens):
            interleaving = [mode is Lfm3OmniGenerationMode.INTERLEAVED for mode in modes]
            audio_rows = [
                row
                for row in range(batch_size)
                if not finished[row]
                and (modes[row] is Lfm3OmniGenerationMode.AUDIO or (interleaving[row] and speaking[row]))
            ]
            sampled_frames = {}
            if audio_rows:
                histories = [
                    torch.stack(tuple(audio_history[row])) if audio_history[row] else None for row in audio_rows
                ]
                if len(audio_rows) == 1:
                    sampled = self.sample_audio_frame(
                        hidden[audio_rows[0]],
                        audio_temperature,
                        audio_top_k,
                        audio_repetition_penalty,
                        histories[0],
                        audio_warpers,
                    )[None]
                else:
                    sampled = self.sample_audio_frame(
                        hidden[audio_rows],
                        audio_temperature,
                        audio_top_k,
                        audio_repetition_penalty,
                        histories,
                        audio_warpers,
                    )
                sampled_frames = dict(zip(audio_rows, sampled))
            # the rows still running feed this step's slot, so a caller's cache covers every returned slot
            active = input_ids.new_tensor([not done for done in finished])[:, None]
            next_embeds = hidden.new_zeros((batch_size, 1, hidden.shape[-1]))
            for row in range(batch_size):
                if finished[row]:
                    continue
                if row in sampled_frames:
                    codes = sampled_frames[row]
                    is_eos = bool((codes == config.audio_eos_code).any())
                    if is_eos:
                        codes = torch.full_like(codes, config.audio_eos_code)
                        modes[row] = Lfm3OmniGenerationMode.TEXT
                    else:
                        audio_history[row].append(codes)
                        if audio_streams[row] is not None:
                            audio_streams[row].put(codes)
                    if not is_eos or return_audio_eos:
                        frames[row].append(codes)
                        text_out[row].append(config.audio_out_token_id)
                    next_embeds[row] = self.model.get_audio_out_embeds(codes[None])
                else:
                    history = torch.stack(tuple(text_history[row])) if text_history[row] else None
                    token = self._sample(
                        self.lm_head(hidden[row]),
                        text_temperature,
                        text_top_k,
                        generation_config.repetition_penalty,
                        history,
                        text_only_suppress if text_only[row] else suppress,
                        top_p=text_top_p,
                        min_p=text_min_p,
                        warpers=text_warpers,
                    )
                    token_id = int(token)
                    text_history[row].append(token)
                    text_out[row].append(token_id)
                    if text_streams[row] is not None:
                        text_streams[row].put(token.view(1, 1))
                    if token_id == config.text_end_token_id:
                        # inside an interleaved block it ends the text and starts the audio tail; otherwise it ends
                        # the turn, so listing it in `eos_token_id` for text-only serving changes nothing here
                        if interleaving[row]:
                            text_done[row], speaking[row] = True, True
                        else:
                            finished[row] = True
                    elif token_id in eos_ids:
                        # `<|im_end|>` ends the turn in any text slot, interleaved blocks included
                        finished[row] = True
                    elif interleaving[row]:
                        pass  # the mode markers only open a block; inside one they are ordinary text
                    elif token_id == config.audio_start_token_id:
                        modes[row] = Lfm3OmniGenerationMode.AUDIO
                    elif token_id == config.mixed_start_token_id:
                        modes[row] = Lfm3OmniGenerationMode.INTERLEAVED
                        slots_left[row], speaking[row], text_done[row] = config.interleaved_n_text, False, False
                    next_embeds[row] = embed_tokens(token.view(1))
                if interleaving[row] and not text_done[row]:
                    slots_left[row] -= 1
                    if slots_left[row] == 0:
                        speaking[row] = not speaking[row]
                        slots_left[row] = config.interleaved_n_audio if speaking[row] else config.interleaved_n_text
            last_step = all(finished) or step == max_new_tokens - 1
            if last_step and not caller_cache:
                # the last sampled token needs no hidden state: skip the decode step nothing would consume
                break
            attention_mask = torch.cat((attention_mask, active), dim=1)
            position_ids = position_ids[:, -1:] + active
            outputs = self.model.language_model(
                inputs_embeds=next_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            if last_step:
                break
            hidden, cache = outputs.last_hidden_state[:, -1], outputs.past_key_values
        generated = input_ids.new_full((batch_size, max(map(len, text_out))), pad_token_id)
        audio_codes = input_ids.new_full((batch_size, config.num_codebooks, max(map(len, frames))), -1)
        for row in range(batch_size):
            generated[row, : len(text_out[row])] = input_ids.new_tensor(text_out[row])
            if frames[row]:
                audio_codes[row, :, : len(frames[row])] = torch.stack(frames[row], dim=-1)
        output = Lfm3OmniGenerateOutput(
            sequences=torch.cat((input_ids, generated), dim=1),
            audio_codes=audio_codes,
            sequence_lengths=input_ids.new_tensor([input_ids.shape[1] + len(tokens) for tokens in text_out]),
            audio_lengths=input_ids.new_tensor([len(row_frames) for row_frames in frames]),
        )
        return output.sequences if plain_sequences else output

    @auto_docstring
    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor):
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, patch_dim)`, *optional*):
            Flattened image patches. `patch_dim` is channels times temporal patch size times squared spatial patch size.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height, and width patch counts before spatial merging, in image placeholder order.

        Returns:
            `torch.Tensor` of shape `(num_image_tokens, hidden_size)`: Projected image features in placeholder order.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @auto_docstring
    def get_video_features(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.LongTensor):
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, patch_dim)`, *optional*):
            Flattened video patches; the tower encodes every temporal patch as its own spatial sequence.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal, height, and width patch counts before spatial merging, in video placeholder order.

        Returns:
            `torch.Tensor` of shape `(num_video_tokens, hidden_size)`: Projected video features in placeholder order.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)


@dataclass
class Lfm3OmniFeatureStreamingCache:
    """Unconsumed mono samples and frame count for one input stream."""

    samples: torch.Tensor | None = None
    num_samples: int = 0
    num_frames: int = 0
    finished: bool = False


class Lfm3OmniFeatureExtractorKwargs(TypedDict, total=False):
    """
    sampling_rate (`int`, *optional*):
        The sampling rate of the clips, which must be the extractor's own; it is checked, never resampled.
    return_tensors (`str` or [`~utils.TensorType`], *optional*):
        The tensor type of the returned features.
    load_audio_backend (`str`, *optional*):
        The decoding backend `apply_chat_template` loads audio with; the extractor itself ignores it.
    """

    # the extractor always right-pads to whole frames and returns the mask, so the padding, truncation and
    # attention-mask options of `AudioKwargs` are not offered rather than silently dropped
    sampling_rate: int | None
    return_tensors: str | TensorType | None
    load_audio_backend: str | None


__all__ = [
    "Lfm3OmniFeatureStreamingCache",
    "Lfm3OmniEncoderStreamingCache",
    "Lfm3OmniLiveTranscriptionState",
    "Lfm3OmniLiveTranscriptionOutput",
    "Lfm3OmniModel",
    "Lfm3OmniPreTrainedModel",
    "Lfm3OmniForConditionalGeneration",
    "Lfm3OmniGenerateOutput",
    "Lfm3OmniStreamingOutput",
]
