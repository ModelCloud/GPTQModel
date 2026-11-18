# Adapted for package-owned Transformers 5 loading by ModelCloud.ai, 2026.
# Pinned upstream: see NOTICE. The hybrid DeltaNet/MoE text stack and shared vision
# layers are the native Transformers Qwen3.5-MoE classes (identical parameter names).
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check

from ..native.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoePreTrainedModel,
    eager_attention_forward,
    load_balancing_loss_func,
    torch_chunk_gated_delta_rule,  # noqa: F401 - re-exported kernel reference
    torch_recurrent_gated_delta_rule,  # noqa: F401 - re-exported kernel reference
)
from ..native.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel as InternS2PreviewTextModel
from ..native.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeVisionModel as InternS2PreviewVisionModel,
)
from .configuration_interns2_preview import (
    InternS2PreviewConfig,
    InternS2PreviewTextConfig,  # noqa: F401 - re-exported with InternS2PreviewTextModel
    InternS2PreviewTimeSeriesConfig,
)


logger = logging.get_logger(__name__)


# Adapted from transformers.models.glm.modular_glm.apply_rotary_pos_emb


class InternS2PreviewPreTrainedModel(Qwen3_5MoePreTrainedModel):
    """Native Qwen3.5-MoE init, hybrid-cache and output-capture contract plus the time-series encoder."""

    config: InternS2PreviewConfig
    _no_split_modules = [
        "Qwen3_5MoeDecoderLayer",
        "Qwen3_5MoeVisionBlock",
        "InternS2PreviewTimeSeriesEncoderLayer",
    ]


class InternS2PreviewTimeSeriesAttention(nn.Module):
    """Multi-headed self-attention of the (uncached) causal time-series encoder."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        config: InternS2PreviewTimeSeriesConfig | None = None,
        num_key_value_groups: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        # The source encoder always applies a causal mask (see InternS2PreviewTimeSeriesEncoder.forward).
        self.is_causal = True

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.num_key_value_groups = num_key_value_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len = hidden_states.shape[:-1]
        hidden_shape = (bsz, tgt_len, -1, self.head_dim)

        # The source scales the query projection before attention (and calls attention with
        # scaling=1.0); keep that order for bit-compatible numerics.
        query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class InternS2PreviewTimeSeriesEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: InternS2PreviewTimeSeriesConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = InternS2PreviewTimeSeriesAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


class InternS2PreviewTimeSeriesEncoder(PreTrainedModel):
    config: InternS2PreviewTimeSeriesConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternS2PreviewTimeSeriesEncoderLayer", "WhisperDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True

    def __init__(self, config: InternS2PreviewTimeSeriesConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        # self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, self.embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, self.embed_dim)

        self.layers = nn.ModuleList(
            [InternS2PreviewTimeSeriesEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        self.post_init()

        self.mask_type = None
        self.chunk_length = None

        self.adapt_in = nn.Linear(config.ts_adapt_in_dim, 80)
        self.adapt_out = nn.Linear(self.embed_dim, config.ts_adapt_out_dim)

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def define_masktype(self, masktype, chunk_length=None):
        self.mask_type = masktype
        self.chunk_length = chunk_length

    def prepare_chunk_attention_mask(self, attention_mask, input_shape, inputs_embeds):

        block_size = round(self.chunk_length / 4 * 2)
        matrix_size = input_shape[1]

        matrix = torch.ones(matrix_size, matrix_size)

        num_full_blocks = round(matrix_size // block_size)
        remainder = matrix_size % block_size
        for i in range(num_full_blocks):
            row_start = i * block_size
            col_start = i * block_size
            matrix[row_start : row_start + block_size, col_start : col_start + block_size] = torch.zeros(
                block_size, block_size
            )

        if remainder > 0:
            last_row_start = num_full_blocks * block_size
            last_col_start = num_full_blocks * block_size
            matrix[last_row_start : last_row_start + remainder, last_col_start : last_col_start + remainder] = (
                torch.zeros(remainder, remainder)
            )

        matrix = matrix * -65504
        matrix = matrix.unsqueeze(0).unsqueeze(0).repeat(input_shape[0], 1, 1, 1)
        attention_mask = matrix.to(inputs_embeds.device)
        return attention_mask

    def forward(self, input_features, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        # (N, T, C) -> (T, N, C) -> (N, C, T)
        input_features = input_features.permute(1, 0, 2)
        input_features = self.adapt_in(input_features)
        input_features = input_features.permute(1, 2, 0)

        # (N, C, T) -> (N, C, T//2)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        # (N, C, T) -> (N, T, C)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # torch.Size([1, 100, 768])
        embed_pos = self.embed_positions.weight  # torch.Size([1500, 768])

        if inputs_embeds.shape[1] > embed_pos.shape[0]:
            target_len = inputs_embeds.shape[1]
            padding = [0, 0, 0, target_len - embed_pos.shape[0]]

            embed_pos = nn.functional.pad(embed_pos, pad=padding, mode="constant", value=0)
            hidden_states = inputs_embeds[:, : embed_pos.shape[0], :] + embed_pos
        else:
            hidden_states = inputs_embeds + embed_pos[: inputs_embeds.shape[1], :]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        input_shape = inputs_embeds.size()[:-1]
        if self.mask_type == "chunk":
            attention_mask = self.prepare_chunk_attention_mask(None, input_shape, inputs_embeds)
        else:
            # The source encoder is causal over the (unpadded) time-series frames.
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                past_key_values=None,
                position_ids=torch.arange(input_shape[1], device=inputs_embeds.device).unsqueeze(0),
            )

        for encoder_layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.training and torch.rand([]) < self.layerdrop:
                continue
            hidden_states = encoder_layer(hidden_states, attention_mask, **kwargs)

        # (N, T, C) -> (T, N, C)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.adapt_out(hidden_states)

        # (T, N, C) -> (N, T, C)
        hidden_states = hidden_states.permute(1, 0, 2)
        return BaseModelOutput(last_hidden_state=hidden_states)


class InternS2PreviewTimeSeriesConcatSubsampling(nn.Module):
    def __init__(self, in_channels: int, concat_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * concat_size

    def forward(self, ts_signals: torch.Tensor, ts_lens: torch.Tensor):
        if ts_signals.shape[1] % 2 != 0:
            ts_signals = ts_signals[:, :-1, :]
        even_frames = ts_signals[:, ::2, :]
        odd_frames = ts_signals[:, 1::2, :]
        ts_signals = torch.cat((even_frames, odd_frames), dim=2)
        ts_lens = ts_lens // 2
        return ts_signals, ts_lens


class InternS2PreviewTimeSeriesFixPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).to(torch.float32)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        # Checkpoint positional buffers may remain FP32 when the encoder runs
        # in FP16/BF16. Round the sum once into the activation dtype, matching
        # the released BF16 path without forcing FP32 callers to BF16.
        return (x + self.pe[: x.size(0), :]).to(dtype=x.dtype).clone()


class InternS2PreviewTimeSeriesMultiChannelAdaptiveSubsampling(nn.Module):
    def __init__(self, hidden_dim: int = 128, nhead: int = 8, num_encoder_layers: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.pos_encoder = InternS2PreviewTimeSeriesFixPositionalEncoding(d_model=hidden_dim)
        self.subsampling = InternS2PreviewTimeSeriesConcatSubsampling(128, 2)

    def forward(
        self, inputs: torch.Tensor, input_lens: torch.Tensor, sr: torch.Tensor, channels: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, feature_lens = self.forward_patch(inputs, input_lens, sr, channels)
        outputs = features
        output_lens = feature_lens
        return outputs, output_lens

    def forward_patch(
        self, inputs: torch.Tensor, input_lens: torch.Tensor, sr: torch.Tensor, channels: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sr = sr.float()
        strides = torch.floor(160 / ((1 + torch.exp(-sr / 100)) ** 6))
        patch_sizes = strides * 2
        patched_outputs = []
        output_lens = []

        for i in range(len(inputs)):
            le = input_lens[i]
            channel = channels[i]
            seq = inputs[i, :le, :channel]  # [seq_len, channel]
            ps = patch_sizes[i].item()
            st = strides[i].item()

            output_len = torch.ceil((le - ps) / st) + 1
            pad_len = ((output_len - 1) * st + ps - le).long().item()
            if seq.ndim == 1:
                seq = seq.unsqueeze(-1)
            seq = nn.functional.pad(seq, (0, 0, 0, pad_len), "constant", 0)
            assert output_len > 0, (seq.shape, ps, st, le, output_len)
            output_lens.append(output_len)
            indices = (torch.arange(0, output_len * st, st).unsqueeze(1) + torch.arange(ps)).long()
            patched = seq[indices]

            output = self.forward_encoder(patched)  # [num_patch, D]
            patched_outputs.append(output)

        outputs = nn.utils.rnn.pad_sequence(patched_outputs, batch_first=True)
        output_lens = torch.tensor(output_lens).squeeze().to(outputs.device).long()
        if output_lens.ndim == 0:
            output_lens = output_lens.unsqueeze(0)

        outputs, output_lens = self.subsampling(outputs.clone(), output_lens.clone())
        return outputs, output_lens

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        num_patch, patch_len, C = x.shape
        # conv1
        x = x.reshape(num_patch * C, 1, patch_len)  # each channel is treated as an independent sample into conv1
        x = nn.functional.relu(self.conv(x))  # [B*C, D1, L]
        x = x.permute(2, 0, 1)  # [L, B*C, D1]

        x = self.pos_encoder(x)  # [L, B*C, D1]
        x = self.transformer_encoder(x)
        x = x.mean(0)

        x = x.reshape(num_patch, C, -1)

        return x.mean(1)


class InternS2PreviewTimeSeriesProjector(nn.Module):
    def __init__(self, config: InternS2PreviewTimeSeriesConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.ts_hidden_dim)
        self.linear_1 = nn.Linear(config.ts_hidden_dim, config.out_hidden_size)
        self.act = ACT2FN[config.activation_function]
        self.linear_2 = nn.Linear(config.out_hidden_size, config.out_hidden_size)

    def forward(self, ts_features):
        hidden_states = self.layer_norm(ts_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class InternS2PreviewTimeSeriesModel(PreTrainedModel):
    main_input_name = "time_series_signals"
    config_class = InternS2PreviewTimeSeriesConfig
    _no_split_modules = ["InternS2PreviewTimeSeriesEncoderLayer"]

    def __init__(self, config: InternS2PreviewTimeSeriesConfig):
        super().__init__(config)
        self.config = config
        self.encoder_embed = InternS2PreviewTimeSeriesMultiChannelAdaptiveSubsampling()
        self.encoder = InternS2PreviewTimeSeriesEncoder(config)
        self.projector = InternS2PreviewTimeSeriesProjector(config)

    def get_input_embeddings(self):
        return self.encoder_embed

    def make_pad_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        r"""
        lengths (`torch.Tensor` of shape `(batch_size,)`):
            A 1-D tensor containing sentence lengths.

        Returns:
        A 2-D bool tensor, where masked positions are filled with `True` and non-masked positions are filled with `False`.
        Example:
        ```python
        >>> lengths = torch.tensor([1, 3, 2, 5])
        >>> self.make_pad_mask(lengths)
        tensor([[False,  True,  True,  True,  True],
                [False, False, False,  True,  True],
                [False, False,  True,  True,  True],
                [False, False, False, False, False]])
        ```
        """
        assert lengths.ndim == 1, lengths.ndim
        max_len = lengths.max()
        n = lengths.size(0)
        seq_range = torch.arange(0, max_len, device=lengths.device)
        expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
        return expaned_lengths >= lengths.unsqueeze(-1)

    def forward(
        self,
        time_series_signals: torch.FloatTensor = None,
        ts_lens: torch.Tensor = None,
        sr: torch.Tensor = None,
        channels: torch.LongTensor | None = None,
        time_series_embeds: torch.FloatTensor = None,
    ):
        if time_series_signals is None and time_series_embeds is None:
            raise ValueError("You have to specify time_series_signals or time_series_embeds")

        if (
            time_series_embeds is not None
            and len(time_series_embeds.shape) == 3
            and time_series_embeds.shape[-1] == self.config.ts_adapt_in_dim
        ):
            time_series_embeds = time_series_embeds
        else:
            if time_series_signals.ndim == 3:
                time_series_embeds, ts_lens = self.encoder_embed(time_series_signals, ts_lens, sr, channels)
            else:
                raise ValueError(f"wrong time_series_signals size: {time_series_signals[0].shape}")

        # [B, 64000, 1] -> [B, 200, 256] -> [B, 100, 1024]
        encoder_outputs = self.encoder(input_features=time_series_embeds)

        # ts_lens after encoder
        ts_lens = (ts_lens + 1) // 2
        assert torch.all(ts_lens > 0), f"The length of time_series_embeds is so small. ts_lens: {ts_lens}"

        src_key_padding_mask = self.make_pad_mask(ts_lens)
        last_hidden_state = encoder_outputs.last_hidden_state

        ts_pad_mask = src_key_padding_mask
        ts_embeds = self.projector(last_hidden_state)

        return ts_embeds, ts_pad_mask


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class InternS2PreviewModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    router_logits: tuple[torch.FloatTensor] | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for InternS2Preview causal language model (or autoregressive) outputs.
    """
)
class InternS2PreviewCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    router_logits: tuple[torch.FloatTensor] | None = None
    aux_loss: torch.FloatTensor | None = None


class InternS2PreviewModel(InternS2PreviewPreTrainedModel):
    base_model_prefix = "model"
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    config: InternS2PreviewConfig

    def __init__(self, config):
        super().__init__(config)
        self.visual = InternS2PreviewVisionModel._from_config(config.vision_config)
        self.language_model = InternS2PreviewTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here
        self.time_series = InternS2PreviewTimeSeriesModel._from_config(config.ts_config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, InternS2Preview use timestamps rather than absolute time position ids."""

        # Since we use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        image_grid_thw_list = image_grid_thw.tolist() if image_grid_thw is not None else None
        video_grid_thw_list = video_grid_thw.tolist() if video_grid_thw is not None else None

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = image_grid_thw_list[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw_list[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                )
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    @can_return_tuple
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output: BaseModelOutputWithPooling = self.visual(
            pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds

        return vision_output

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                inputs_embeds[special_video_mask].numel() == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )
        return special_image_mask, special_video_mask

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        can_compute_mrope = input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None)

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        # Use pre-calculated rope-deltas to infer correct 3D position ids
        elif self.rope_deltas is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=position_ids.device)
        else:
            # Can't build correct 3D positions; the text model derives them from the cache length.
            position_ids = None
        return position_ids

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        ts_values: torch.FloatTensor | list[torch.FloatTensor] = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        ts_lens: torch.Tensor | list[torch.Tensor] = None,
        ts_sr: torch.FloatTensor | list[torch.FloatTensor] = None,
        ts_channels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | InternS2PreviewModelOutputWithPast:
        r"""
        ts_values (`torch.FloatTensor` of shape `(batch_size, seq_len, num_channels)`, *optional*):
            The tensors corresponding to the input time series signals.
        ts_lens (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            The valid lengths of each time series signal in the batch.
        ts_sr (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            The sampling rates of each time series signal in the batch.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithPooling = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if ts_values is not None:
            ts_features, ts_pad_mask = self.get_ts_feature(ts_values, ts_lens, ts_sr, ts_channels)  # [B, T, C], [B, T]
            ts_features = ts_features[~ts_pad_mask].to(
                inputs_embeds.device, inputs_embeds.dtype
            )  # [num_valid_ts_tokens, C]
            B, N, C = inputs_embeds.shape
            input_ids = input_ids.reshape(B * N)
            inputs_embeds = inputs_embeds.reshape(B * N, C)
            # replace ts_token in inputs_embeds and attention_mask
            ts_placeholder = input_ids == self.config.ts_token_id
            n_ts_placeholders = ts_placeholder.sum().item()
            n_ts_tokens = ts_features.size(0)
            assert n_ts_placeholders == n_ts_tokens, (
                f"[ERROR]: Mismatch: <TS_CONTEXT> tokens={n_ts_placeholders}, ts_embeds_valid={n_ts_tokens}"
            )

            try:
                # TODO why not scatter?
                inputs_embeds[ts_placeholder] = inputs_embeds[ts_placeholder] * 0.0 + ts_features
            except Exception as e:
                print(
                    f"warning: {e}, inputs_embeds[selected].shape={inputs_embeds[ts_placeholder].shape}, ts_embeds_valid.shape={ts_features.shape}"
                )
                inputs_embeds[ts_placeholder] = inputs_embeds[ts_placeholder] * 0.0 + n_ts_tokens[:n_ts_placeholders]

            inputs_embeds = inputs_embeds.reshape(B, N, C)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return InternS2PreviewModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )

    @can_return_tuple
    def get_ts_feature(
        self,
        ts_values: torch.FloatTensor | list[torch.FloatTensor],
        ts_lens: torch.Tensor | list[torch.Tensor],
        sr: torch.FloatTensor | list[torch.FloatTensor],
        ts_channels: torch.LongTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.Tensor]:
        r"""
        ts_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`):
            The time series values to be fed to a model.
        ts_lens (`torch.Tensor` of shape `(batch_size,)`):
            The length of the time series.
        sr (`torch.FloatTensor` of shape `(batch_size,)`):
            The sampling rate of the time series.
        """
        ts_embeds, ts_pad_mask = self.time_series(
            time_series_signals=ts_values,
            ts_lens=ts_lens,
            sr=sr,
            channels=ts_channels,
        )
        return ts_embeds, ts_pad_mask


# NOTE: Cannot inherit from `Qwen3_5MoeForCausalLM` here due to a converter limitation: when the modular
# `__init__` contains statements before `super().__init__()`, `_fix_init_location` unconditionally moves
# `super().__init__()` to the top, which prevents pre-processing `config` (e.g. extracting `config.text_config`)
# before the parent call. The full body is therefore written out explicitly.


class InternS2PreviewForConditionalGeneration(InternS2PreviewPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    config: InternS2PreviewConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = InternS2PreviewModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        return self.model.get_video_features(
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs
        )

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        return self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        ts_values: torch.FloatTensor | list[torch.FloatTensor] = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        ts_lens: torch.Tensor | list[torch.Tensor] = None,
        ts_sr: torch.FloatTensor | list[torch.FloatTensor] = None,
        ts_channels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | InternS2PreviewCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

        >>> model = Qwen3_5MoeForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-35B-A3B-Instruct", dtype="auto", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-35B-A3B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image in short."},
                ],
            }
        ]

        >>> # Preparation for inference
        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        >>> inputs = inputs.to(model.device)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=128)
        >>> generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        >>> processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "A woman in a plaid shirt sits on a sandy beach at sunset, smiling as she gives a high-five to a yellow Labrador Retriever wearing a harness. The ocean waves roll in the background."
        ```"""

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            ts_values=ts_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            ts_lens=ts_lens,
            ts_sr=ts_sr,
            ts_channels=ts_channels,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        return InternS2PreviewCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(self, input_ids, *args, is_first_iteration=False, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, *args, is_first_iteration=is_first_iteration, **kwargs
        )
        # Time-series features are merged during prefill only; cached decode steps never see them again.
        if not is_first_iteration and kwargs.get("use_cache", True):
            for key in ("ts_values", "ts_lens", "ts_sr", "ts_channels"):
                model_inputs.pop(key, None)
        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # Overwritten -- requires 3D position ids

        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        # Early exit in case we are continuing generation from past kv
        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            text_positions += self.model.rope_deltas
            return text_positions

        # Otherwise compute 3d position ids for vision tokens and concat with text position ids
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if is_input_ids and (
            model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None
        ):
            model_kwargs = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
            vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)
            self.model.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
            )

        # Concatenate "text + vision" positions into [4, bs, seq-len]
        text_positions = text_positions[None, ...]
        position_ids = torch.cat([text_positions, vision_positions], dim=0)

        return position_ids

    def _get_image_nums_and_video_nums(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- InternS2Preview use timestamps and remove second_per_grid_ts
        # Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            # video_nums: (batch_size,)
            # since video_nums is the number of videos in the input dependent on the input_ids(vision_start),
            # but InternS2Preview append vision_start to each frame of each video, so we need to recover the real video_nums according to video_grid_thw
            if video_grid_thw is not None:
                cumulative_frame_counts = torch.cumsum(video_grid_thw[:, 0], dim=0)
                cumulative_token_video_counts = torch.cumsum(video_nums, dim=0)
                # Find video boundaries in cumulative_frame_counts
                video_boundary_indices = torch.searchsorted(cumulative_frame_counts, cumulative_token_video_counts)
                # example: video_boundary_indices = [3, 5] means video_nums = [4, 2]
                video_nums = torch.diff(torch.cat([-video_boundary_indices.new_ones(1), video_boundary_indices]))

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if key == "position_ids" and dict_to_expand[key].ndim == 3:
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=1)
                elif (
                    dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    @property
    def time_series(self):
        return self.model.time_series

    def get_ts_feature(
        self,
        ts_values: torch.FloatTensor | list[torch.FloatTensor],
        ts_lens: torch.Tensor | list[torch.Tensor],
        sr: torch.FloatTensor | list[torch.FloatTensor],
    ):
        r"""
        ts_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`):
            The time series values to be fed to a model.
        ts_lens (`torch.Tensor` of shape `(batch_size,)`):
            The length of the time series.
        sr (`torch.FloatTensor` of shape `(batch_size,)`):
            The sampling rate of the time series.
        """
        return self.model.get_ts_feature(ts_values, ts_lens, sr)


__all__ = [
    "InternS2PreviewVisionModel",
    "InternS2PreviewTextModel",
    "InternS2PreviewModel",
    "InternS2PreviewForConditionalGeneration",
    "InternS2PreviewPreTrainedModel",
]
