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
# Pinned source and modifications: NOTICE.
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.configuration_utils import PreTrainedConfig
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
)
from transformers.utils.generic import (
    is_flash_attention_requested,
    maybe_autocast,
    merge_with_config_defaults,
)
from transformers.utils.output_capturing import OutputRecorder, capture_outputs
from transformers.vision_utils import get_max_seqlen, get_vision_attention_seqlens, get_vision_position_ids

from .configuration_lfm3_vl import Lfm3VlVisionConfig


class Lfm3VlRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Lfm3VlRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Lfm3VlMLP(nn.Module):
    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Lfm3VlVisionRotaryEmbedding(nn.Module):
    """
    Simple axial 2D rope with same freqs used for H and W grids. The freqs are
    pre-computed using `head-dim//4` which is later used to concat H and W positions.
    The final angles rotate over the whole head dim, no partial rotation involved.
    """

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__()
        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_axial_rope_parameters
        if self.rope_type != "axial":
            raise ValueError(f"{self.__class__.__name__} supports only axial rope, but requested {self.rope_type}")
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_axial_rope_parameters(config: Lfm3VlVisionConfig, **kwargs) -> tuple[torch.Tensor, float]:
        """Inverse frequencies of the axial RoPE over half the head dimension per axis, with an attention scaling of 1."""
        base = config.rope_parameters["rope_theta"]
        spatial_dim = config.hidden_size // config.num_attention_heads // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (2, N) — row 0 = h coords, row 1 = w coords
        position_ids_expanded = position_ids[..., None].float()
        device_type = x.device.type if isinstance(x.device.type, str) else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = position_ids_expanded * self.inv_freq.float()
            cos = freqs.cos() * self.attention_scaling
            sin = freqs.sin() * self.attention_scaling

        cos = self.recomposition_frequencies(cos)
        sin = self.recomposition_frequencies(sin)
        return cos, sin

    def recomposition_frequencies(self, freq):
        """
        Recompose the frequencies into the final spatial layout used per each grid.
        """
        freq_h, freq_w = freq[:, 0], freq[:, 1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Lfm3VlVisionAttention(nn.Module):
    """Bidirectional grouped-query attention over the patches of one frame, with RMSNorm on the query and key heads."""

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_layernorm = Lfm3VlRMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = Lfm3VlRMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int | None = None,
        frame_shapes: list[list[int]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = self.q_layernorm(self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim))
        key_states = self.k_layernorm(
            self.k_proj(hidden_states).view(seq_length, self.num_key_value_heads, self.head_dim)
        )
        value_states = self.v_proj(hidden_states).view(seq_length, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: frames of one shape (every video, equal-size images) attend as one batch, mixed
            # shapes frame by frame. `frame_shapes` gives the lengths without a device sync.
            if frame_shapes is not None:
                lengths = [height * width for height, width in frame_shapes]
            else:
                lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            if len(set(lengths)) == 1:
                query_states, key_states, value_states = (
                    tensor.reshape(-1, len(lengths), lengths[0], self.head_dim).transpose(0, 1)
                    for tensor in (query_states, key_states, value_states)
                )
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0,
                    is_causal=False,
                    **kwargs,
                )
            else:
                splits = [torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)]
                attn_outputs, attn_weights = [], []
                for q, k, v in zip(*splits):
                    output, weights = attention_interface(
                        self,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0,
                        is_causal=False,
                        **kwargs,
                    )
                    attn_outputs.append(output)
                    attn_weights.append(weights)
                attn_output = torch.cat(attn_outputs, dim=1)
                # frames of different shapes have differently sized maps: one entry per frame, or None under sdpa
                attn_weights = None if attn_weights[0] is None else tuple(attn_weights)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class Lfm3VlVisionConv2d(nn.Module):
    """Gated depthwise 2D convolution over the patch grid of each frame."""

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.spatial_conv = nn.Conv2d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=config.hidden_size,
            bias=False,
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, frame_shapes: list[list[int]]):
        B, C, x = self.in_proj(hidden_states).chunk(3, dim=-1)
        gated = B * x
        merge = self.spatial_merge_size
        num_frames = len(frame_shapes)
        if len(set(map(tuple, frame_shapes))) == 1:
            # frames of one shape (every video, equal-size images) convolve as one batch: merge-block order -> raster
            # grid, convolve, back
            height, width = frame_shapes[0]
            frames = gated.view(num_frames, height // merge, width // merge, merge, merge, -1)
            frames = frames.permute(0, 5, 1, 3, 2, 4).reshape(num_frames, -1, height, width)
            frames = self.spatial_conv(frames)
            frames = frames.view(num_frames, -1, height // merge, merge, width // merge, merge).permute(
                0, 2, 4, 3, 5, 1
            )
            conv_output = frames.reshape(gated.shape[0], -1)
        else:
            conv_outputs = []
            lengths = [height * width for height, width in frame_shapes]
            for frame, (height, width) in zip(torch.split(gated, lengths, dim=0), frame_shapes):
                frame = frame.view(height // merge, width // merge, merge, merge, -1)
                frame = frame.permute(4, 0, 2, 1, 3).reshape(1, -1, height, width)
                frame = self.spatial_conv(frame)
                frame = frame.view(-1, height // merge, merge, width // merge, merge).permute(1, 3, 2, 4, 0)
                conv_outputs.append(frame.reshape(height * width, -1))
            conv_output = torch.cat(conv_outputs, dim=0)
        return self.out_proj(C * conv_output)


class Lfm3VlVisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: Lfm3VlVisionConfig, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm3VlVisionAttention(config)
        else:
            self.conv = Lfm3VlVisionConv2d(config)
        self.feed_forward = Lfm3VlMLP(config)
        self.operator_norm = Lfm3VlRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm3VlRMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        frame_shapes: list[list[int]],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.is_attention_layer:
            operator_output, _ = self.self_attn(
                self.operator_norm(hidden_states),
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                frame_shapes=frame_shapes,
                **kwargs,
            )
        else:
            operator_output = self.conv(self.operator_norm(hidden_states), frame_shapes)
        hidden_states = hidden_states + operator_output
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


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
class Lfm3VlPreTrainedModel(PreTrainedModel):
    config: Lfm3VlVisionConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]

    _supports_flash_attn = True
    _supports_sdpa = True
    # the cached decode step traces without graph breaks, like the text model
    _can_compile_fullgraph = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _no_split_modules = ["Lfm3VlVisionLayer"]

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        # the static-cache masks of the text model are sized by layer type, not by the generic first attention layer
        return create_masks_by_layer_type(config, inputs_embeds, attention_mask, past_key_values, position_ids)


def get_vision_frame_shapes(grid_thw: torch.Tensor) -> list[list[int]]:
    return grid_thw[:, 1:].repeat_interleave(grid_thw[:, 0], dim=0).tolist()


@auto_docstring(
    custom_intro="""
    The LFM3-VL vision encoder: a stack of gated convolution and bidirectional attention layers over the patches of
    each image or video frame, embedded by a linear patchifier.
    """
)
class Lfm3VlVisionModel(Lfm3VlPreTrainedModel):
    config: Lfm3VlVisionConfig
    input_modalities = ("image", "video")
    _input_embed_layer = "patchifier"
    _can_record_outputs = {
        "hidden_states": Lfm3VlVisionLayer,
        "attentions": OutputRecorder(Lfm3VlVisionAttention, index=1),
    }

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.patchifier = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.embedding_norm = Lfm3VlRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.rotary_pos_emb = Lfm3VlVisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [Lfm3VlVisionLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every image or video, grouped by merge block.
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width dimensions of the patch grid of each image or video.
        """
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        frame_shapes = get_vision_frame_shapes(grid_thw)

        hidden_states = self.patchifier(pixel_values.to(self.patchifier.weight.dtype))
        hidden_states = self.embedding_norm(hidden_states)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                frame_shapes=frame_shapes,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)



__all__ = ["Lfm3VlVisionModel"]
