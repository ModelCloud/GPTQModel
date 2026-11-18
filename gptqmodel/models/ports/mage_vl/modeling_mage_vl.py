# SPDX-FileCopyrightText: 2026 Microsoft
# SPDX-License-Identifier: Apache-2.0
# Adapted from microsoft/Mage-VL@d88b153285f1633a61b2f693c59c8576693af185.
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from transformers import AutoModel
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.utils.auto_docstring import auto_docstring
from transformers.utils.generic import (
    ModelOutput,
    can_return_tuple,
    is_flash_attention_requested,
    merge_with_config_defaults,
)
from transformers.utils.output_capturing import capture_outputs

from .configuration_mage_vl import MageVLConfig, MageVLVisionConfig


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Mage-VL outputs, with hidden states and attentions.
    """
)
class MageVLModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    last_hidden_state: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.Tensor]] = None
    attentions: Optional[tuple[torch.Tensor]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Mage-VL causal language model (or autoregressive) outputs.
    """
)
class MageVLCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.Tensor]] = None
    attentions: Optional[tuple[torch.Tensor]] = None


# ---------------------------------------------------------------------------
# Vision Rotary Embedding
# ---------------------------------------------------------------------------


class VisionRotaryEmbedding(nn.Module):
    """
    3D (T,H,W) Rotary frequency constructor with 4:6:6 split.
    Supports both grid_thw-based and explicit position-based RoPE computation.
    """

    inv_freq_t: torch.Tensor
    inv_freq_h: torch.Tensor
    inv_freq_w: torch.Tensor

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16."
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must also be divisible by 16 to split into 4:6:6."

        self.head_dim = head_dim
        self.base = base

        # 4:6:6 split for T:H:W
        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

    def _frequencies(self, device, dtype=None):
        frequencies = []
        for buffer, size in ((self.inv_freq_t, self.t_size), (self.inv_freq_h, self.h_size),
                             (self.inv_freq_w, self.w_size)):
            if buffer.is_meta:
                # Lazy checkpoint shells omit these non-persistent buffers.
                buffer = (1.0 / (self.base ** (torch.arange(size, device=device, dtype=torch.float32) / size))).to(buffer.dtype)
            frequencies.append(buffer.to(device=device, dtype=dtype))
        return frequencies

    def forward(self, grid_thw: torch.Tensor, rope_dtype=None) -> torch.Tensor:
        """
        Compute rotary position embeddings from grid_thw (Qwen2VL style).

        Args:
            grid_thw: [num_samples, 3] tensor with [t, h, w] for each sample

        Returns:
            freqs: [total_seq_len, half] tensor of position frequencies
        """
        device = grid_thw.device
        inv_t, inv_h, inv_w = self._frequencies(device, rope_dtype)

        all_freqs = []
        for sample_thw in grid_thw:
            t, h, w = (int(v) for v in sample_thw.tolist())

            # Compute frequency tables
            ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
            fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
            fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

            # Build position indices for this sample
            t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
            h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
            w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

            # Concatenate frequencies: [seq_len, half]
            sample_freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
            all_freqs.append(sample_freqs)

        return torch.cat(all_freqs, dim=0)

    def forward_from_positions(self, patch_positions: torch.Tensor, rope_dtype=None) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit patch positions.

        Args:
            patch_positions: [seq_len, 3] tensor with [t, h, w] positions for each patch

        Returns:
            freqs: [seq_len, half] tensor of position frequencies
        """
        device = patch_positions.device
        inv_t, inv_h, inv_w = self._frequencies(device, rope_dtype)

        t_pos = patch_positions[:, 0].float()
        h_pos = patch_positions[:, 1].float()
        w_pos = patch_positions[:, 2].float()

        ft = torch.outer(t_pos, inv_t)
        fh = torch.outer(h_pos, inv_h)
        fw = torch.outer(w_pos, inv_w)

        return torch.cat([ft, fh, fw], dim=-1)

    def forward_with_thw(self, t: int, h: int, w: int, device=None) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit t, h, w dimensions.

        Args:
            t: Number of temporal frames
            h: Number of height patches
            w: Number of width patches
            device: Target device

        Returns:
            freqs: [t*h*w, half] tensor of position frequencies
        """
        if device is None:
            device = self.inv_freq_t.device

        inv_t, inv_h, inv_w = self._frequencies(device)

        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return freqs


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


class MageVLVisionEmbeddings(nn.Module):
    """
    Patch embedding layer that converts pre-processed patches to embeddings.

    This module is designed to receive patches that have already been extracted
    and arranged by the Qwen2VL image processor in 2x2 block spatial order.

    Input format: [total_patches, num_channels, patch_size, patch_size]
    Output format: [total_patches, embed_dim]
    """

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.in_channels = config.num_channels

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        hidden_states = self.patch_embedding(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)

        return hidden_states


# ---------------------------------------------------------------------------
# Patch Merger
# ---------------------------------------------------------------------------


class MageVLVisionPatchMerger(nn.Module):
    """
    Patch merger that merges spatial_merge_size x spatial_merge_size patches into one.

    This module is designed to work with Qwen2VL-style patch processing where patches
    are already arranged in 2x2 block order by the image processor.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-05,
        use_patch_position_encoding: bool = False,
        patch_position_encoding_type: str = "absolute",
        max_position_embeddings: int = 8192,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )
        self.spatial_merge_size = spatial_merge_size
        self.use_patch_position_encoding = use_patch_position_encoding
        self.patch_position_encoding_type = patch_position_encoding_type

        if self.use_patch_position_encoding:
            if self.patch_position_encoding_type != "absolute":
                raise ValueError(
                    f"Unknown patch_position_encoding_type: {self.patch_position_encoding_type}. "
                    "Only 'absolute' is supported."
                )
            self.pos_emb_h = nn.Embedding(max_position_embeddings, dim)
            self.pos_emb_w = nn.Embedding(max_position_embeddings, dim)

    def forward(self, x: torch.Tensor, patch_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Merge patches from Qwen2VL-style input.

        The input patches are already arranged in 2x2 block order by the image processor,
        so we simply need to apply LayerNorm, reshape, and project through MLP.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
               where seq_len = t * h * w (patches in 2x2 block order)

        Returns:
            Merged tensor of shape [batch_size, seq_len // spatial_merge_size^2, dim]
            or [seq_len // spatial_merge_size^2, dim]
        """
        if patch_positions is not None and patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)

        x = self.ln_q(x).view(-1, self.hidden_size)
        x = self.mlp(x)

        if self.use_patch_position_encoding and patch_positions is not None:
            pp = patch_positions.view(-1, self.spatial_merge_size**2, 3)
            pp = pp[:, 0, :]
            pp = (pp // self.spatial_merge_size).long()

            x = x + self.pos_emb_h(pp[:, 1]) + self.pos_emb_w(pp[:, 2])

        return x


def rotate_half(x):
    """
    Interleaved rotation to match Source model's implementation.
    (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D)
    # freqs: (B, L, D)
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    # We need to broadcast freqs to match heads
    # (B, L, D) -> (B, 1, L, D)
    # Keep the same dtype as q, k to avoid memory doubling from float32 promotion
    cos = freqs.cos().unsqueeze(1).float()
    sin = freqs.sin().unsqueeze(1).float()

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager attention; query/key/value are expected as ``(B, H, L, D)``."""
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()  # (B, L, H, D)
    return attn_output, attn_weights


class MageVLVisionAttention(nn.Module):
    """
    Multi-headed attention with RoPE support, dispatched through
    :data:`ALL_ATTENTION_FUNCTIONS` (``eager`` / ``sdpa`` / ``flash_attention_2``)
    based on ``config._attn_implementation``.
    """

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.num_key_value_groups = 1  # required by repeat_kv-aware eager paths
        self.scale = self.head_dim**-0.5
        self.scaling = self.scale  # alias expected by some attention interfaces
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = hidden_states.size()
        # (B, L, 3*H*D) -> (B, L, 3, H, D) -> 3 x (B, L, H, D) -> 3 x (B, H, L, D)
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(batch_size, q_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 1, 3, 4)
            .unbind(0)
        )
        query_states = q.transpose(1, 2)
        key_states = k.transpose(1, 2)
        value_states = v.transpose(1, 2)

        if rotary_pos_emb is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation or "eager", eager_attention_forward
        )
        dropout = 0.0 if not self.training else self.attention_dropout
        attn_weights = None

        if cu_seqlens is not None and is_flash_attention_requested(self.config):
            # Flash Attention varlen path: pass cu_seq_lens / max_length kwargs.
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scale,
                dropout=dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        elif cu_seqlens is not None:
            # Non-FA implementations do not understand cu_seqlens directly; mirror
            # Qwen3-VL by splitting the packed sequence into per-sample chunks
            # along the L dim of (B, H, L, D) and running attention per chunk.
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            splits = [torch.split(t, lengths, dim=2) for t in (query_states, key_states, value_states)]
            attn_outputs = [
                attention_interface(
                    self,
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    attention_mask=None,
                    scaling=self.scale,
                    dropout=dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q_chunk, k_chunk, v_chunk in zip(*splits)
            ]
            # interface output is (B, l_i, H, D); concat along the L axis
            attn_output = torch.cat(attn_outputs, dim=1)
        else:
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                if attn_mask.shape[0] == 1 and batch_size > 1:
                    attn_mask = attn_mask.expand(batch_size, -1, -1)
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, L, L)
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=attn_mask,
                scaling=self.scale,
                dropout=dropout,
                is_causal=False,
                **kwargs,
            )

        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
        attn_output = self.proj(attn_output)

        return attn_output, attn_weights


class MageVLVisionEncoderLayer(GradientCheckpointingLayer):
    """Pre-norm vision encoder layer."""

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MageVLVisionAttention(config)
        self.layer_norm1 = get_norm_layer(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = get_norm_layer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MageVLVisionEncoder(nn.Module):
    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MageVLVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> BaseModelOutput:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                **kwargs,
            )
        return BaseModelOutput(last_hidden_state=hidden_states)


class MageVLPreTrainedModel(PreTrainedModel):
    config_class = MageVLConfig
    base_model_prefix = "model"
    input_modalities = ["image", "video", "text"]
    supports_gradient_checkpointing = True
    _no_split_modules = ["MageVLVisionEncoderLayer", "Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        # Re-initialize VisionRotaryEmbedding inv_freq buffers.
        # These are registered with persistent=False, so they are not in the checkpoint
        # state_dict. When ``from_pretrained`` materializes the model from meta tensors,
        # the values in these buffers end up uninitialized. Mirror Qwen3-VL by explicitly
        # filling them here so RoPE produces the correct frequencies post-load.
        if isinstance(module, VisionRotaryEmbedding):
            base = module.base
            with torch.no_grad():
                inv_t = 1.0 / (base ** (torch.arange(module.t_size, dtype=torch.float32) / module.t_size))
                inv_h = 1.0 / (base ** (torch.arange(module.h_size, dtype=torch.float32) / module.h_size))
                inv_w = 1.0 / (base ** (torch.arange(module.w_size, dtype=torch.float32) / module.w_size))
                module.inv_freq_t.copy_(inv_t.to(module.inv_freq_t.device))
                module.inv_freq_h.copy_(inv_h.to(module.inv_freq_h.device))
                module.inv_freq_w.copy_(inv_w.to(module.inv_freq_w.device))


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """
    Multi-Head Attention Pooling with a learned probe (PMA-style).
    """

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        attn_output, _ = self.attention(probe, hidden_states, hidden_states)

        residual = attn_output
        attn_output = self.norm(attn_output)
        attn_output = residual + self.mlp(attn_output)

        return attn_output[:, 0]


# ---------------------------------------------------------------------------
# Vision Model
# ---------------------------------------------------------------------------


class MageVLVisionPretrainedModel(MageVLPreTrainedModel):
    """
    Mage-VL Vision Model.

    This vision model is designed to work with Qwen2VL-style image processing:
        - Receives pre-processed patches in 2x2 block spatial order
        - Applies RoPE with matching 2x2 block layout conversion
        - Accepts explicit patch_positions for RoPE computation

    Input format:
        hidden_state: [total_patches, num_channels, patch_size, patch_size]
        grid_thw: [num_samples, 3] with [t, h, w] for each sample
    """

    _can_record_outputs = {
        "hidden_states": MageVLVisionEncoderLayer,
        "attentions": MageVLVisionAttention,
    }

    def __init__(self, config: MageVLVisionConfig):
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        # Vision components
        self.embeddings = MageVLVisionEmbeddings(config)
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = MageVLVisionEncoder(config)
        self.video_rope = VisionRotaryEmbedding(config)

        if config.use_head:
            self.layernorm_post = get_norm_layer(config)
            self.head = Siglip2MultiheadAttentionPoolingHead(config)
        else:
            self.layernorm_post = None
            self.head = None

        self.merger = MageVLVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
            use_patch_position_encoding=getattr(config, "use_patch_position_encoding", False),
            patch_position_encoding_type=getattr(config, "patch_position_encoding_type", "absolute"),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
        )

        self.post_init()

    def _build_cu_seqlens(
        self,
        grid_thw: Optional[torch.Tensor],
        total_patches: int,
        fixed_t: Optional[int] = 4,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, int]:
        if grid_thw is None or grid_thw.numel() == 0:
            # Fallback for no grid_thw: treat as single sequence
            return torch.tensor([0, total_patches], dtype=torch.int32, device=device), total_patches

        if device is None:
            device = grid_thw.device

        cu_seqlens = [0]
        max_seqlen = 0
        total_entries = grid_thw.shape[0]
        current_len = 0

        # Calculate cumulative lengths: split sequences based on fixed_t if provided
        for idx in range(total_entries):
            t_val = int(grid_thw[idx, 0].item())
            h_val = int(grid_thw[idx, 1].item())
            w_val = int(grid_thw[idx, 2].item())

            if fixed_t is not None and fixed_t > 0 and t_val > fixed_t:
                # Split large t into chunks of fixed_t
                num_full_windows = t_val // fixed_t
                remainder = t_val % fixed_t

                # Add full windows
                for _ in range(num_full_windows):
                    chunk_patches = fixed_t * int(h_val) * int(w_val)
                    current_len += chunk_patches
                    max_seqlen = max(max_seqlen, chunk_patches)
                    cu_seqlens.append(current_len)

                # Add remainder if any
                if remainder > 0:
                    chunk_patches = remainder * int(h_val) * int(w_val)
                    current_len += chunk_patches
                    max_seqlen = max(max_seqlen, chunk_patches)
                    cu_seqlens.append(current_len)
            else:
                # Standard case: add as one chunk
                chunk_patches = t_val * int(h_val) * int(w_val)
                current_len += chunk_patches
                max_seqlen = max(max_seqlen, chunk_patches)
                cu_seqlens.append(current_len)

        last_len = cu_seqlens[-1]
        if last_len != total_patches:
            raise ValueError(
                "cu_seqlens calculation mismatch:\n"
                f"- total_patches: {total_patches}\n"
                f"- calculated total: {last_len}\n"
                f"- grid_thw: {grid_thw}"
            )

        return torch.tensor(cu_seqlens, dtype=torch.int32, device=device), max_seqlen

    def _build_block_attention_mask(
        self,
        grid_thw: torch.Tensor,
        total_patches: int,
        fixed_t: Optional[int] = 4,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        if grid_thw is None or grid_thw.numel() == 0:
            return None

        if device is None:
            device = grid_thw.device

        lengths = []
        total_entries = grid_thw.shape[0]

        for idx in range(total_entries):
            t_val = int(grid_thw[idx, 0].item())
            h_val = int(grid_thw[idx, 1].item())
            w_val = int(grid_thw[idx, 2].item())

            if fixed_t is not None and fixed_t > 0 and t_val > fixed_t:
                # Split large t into chunks of fixed_t
                num_full_windows = t_val // fixed_t
                remainder = t_val % fixed_t

                # Add full windows
                for _ in range(num_full_windows):
                    lengths.append(fixed_t * int(h_val) * int(w_val))

                # Add remainder if any
                if remainder > 0:
                    lengths.append(remainder * int(h_val) * int(w_val))
            else:
                lengths.append(t_val * int(h_val) * int(w_val))

        total_len = sum(lengths)
        if total_len != total_patches:
            raise ValueError(
                "Block attention mask length mismatch:\n"
                f"- total_patches: {total_patches}\n"
                f"- total_len: {total_len}\n"
                f"- grid_thw: {grid_thw}"
            )

        attn_mask = torch.ones((total_len, total_len), dtype=torch.bool, device=device)
        start = 0
        for size in lengths:
            end = start + size
            attn_mask[start:end, start:end] = False
            start = end

        return attn_mask

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self,
        hidden_state: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        skip_merger: Optional[bool] = False,
        rope_dtype=None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        r"""
        Forward pass for vision model.

        This method accepts pre-processed patches from Qwen2VL image processor and applies
        RoPE (Rotary Position Embedding) in 2x2 block layout to match the spatial arrangement
        of patches.

        Args:
            hidden_state: Pre-processed patches from Qwen2VL processor.
                Shape: [total_patches, num_channels, patch_size, patch_size]
            grid_thw: Grid sizes tensor of shape [num_samples, 3] with [t, h, w] for each sample.
                Required for computing RoPE and handling visible indices.
            patch_positions: Optional explicit patch positions for RoPE computation.
            skip_merger: If True, skip patch merger (useful for consistency checking).

        Returns:
            BaseModelOutputWithPooling with last_hidden_state containing merged features.
            Encoder-layer hidden states / attentions are recorded via ``output_hidden_states`` /
            ``output_attentions``.
        """
        # 1. Embeddings
        # Note: embeddings returns [total_patches, embed_dim], we need to add batch dimension
        hidden_states = self.embeddings(hidden_state)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)  # [1, total_patches, embed_dim]
        batch_size, total_patches, _ = hidden_states.shape

        # 2. RoPE Construction
        if patch_positions is not None and patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)
        if patch_positions is not None:
            freqs_visible = self.video_rope.forward_from_positions(patch_positions, rope_dtype=rope_dtype)
        elif grid_thw is not None:
            freqs_visible = self.video_rope(grid_thw, rope_dtype=rope_dtype)
        else:
            raise ValueError("Mage-VL vision needs patch_positions or grid_thw for 3D RoPE.")

        # Concatenate D/2 + D/2 -> D for applying rope
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)
        if freqs_visible.dim() == 2:
            freqs_visible = freqs_visible.unsqueeze(0)

        # 3. Pre-Norm & Encoder
        hidden_states = self.layernorm_pre(hidden_states)

        cu_seqlens, max_seqlen = self._build_cu_seqlens(
            grid_thw=grid_thw,
            total_patches=total_patches,
            fixed_t=getattr(self.config, "frame_windows_size", 4),
            device=hidden_states.device,
        )

        sequence_output = self.encoder(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=freqs_visible,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **kwargs,
        ).last_hidden_state

        # Post-Norm
        if self.layernorm_post is not None:
            sequence_output = self.layernorm_post(sequence_output)

        # Skip merger for consistency check with original ViT
        if skip_merger:
            pooled_output = self.head(sequence_output) if self.head is not None else None
            return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output)

        # Patch merger: input patches are already in 2x2 block order from Qwen2VL processor
        merged_output = self.merger(sequence_output, patch_positions=patch_positions)
        return BaseModelOutputWithPooling(last_hidden_state=merged_output, pooler_output=None)


@auto_docstring
class MageVLModel(MageVLPreTrainedModel):
    base_model_prefix = ""
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: MageVLConfig
    _no_split_modules = ["MageVLVisionEncoderLayer", "Qwen3DecoderLayer"]

    def __init__(self, config: MageVLConfig):
        super().__init__(config)
        self.visual = MageVLVisionPretrainedModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config, trust_remote_code=False)
        self.streammind_gate = None
        self._streammind_model_path: str | None = None

        # Initialize weights and apply final processing
        self.post_init()

    def _load_streammind_gate(self):
        if self.streammind_gate is not None:
            return self.streammind_gate
        from safetensors.torch import load_file

        from .streammind_gate import StreamMindGate

        if not self._streammind_model_path:
            raise RuntimeError(
                "StreamMind gate path is unavailable. Load the model with "
                "MageVLForConditionalGeneration.from_pretrained() with a local directory."
            )
        gate_path = Path(self._streammind_model_path) / "streammind_gate.safetensors"
        if not gate_path.is_file():
            raise FileNotFoundError(f"Mage-VL StreamMind local safetensors asset is missing: {gate_path}")
        try:
            from mamba_ssm.models.mixer_seq_simple import create_block  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Mage-VL StreamMind requires installed mamba-ssm and its device kernels.") from exc
        del create_block
        gate = StreamMindGate(self.language_model.config.hidden_size)
        state = load_file(str(gate_path))
        gate.load_state_dict(state, strict=True)
        gate.to(
            device=next(self.visual.parameters()).device,
            dtype=next(self.visual.parameters()).dtype,
        ).eval()
        self.streammind_gate = gate
        return gate

    def _streammind_vision_tokens(self, pixel_values, image_grid_thw, patch_positions=None):
        pixel_values = pixel_values.type(self.visual.embeddings.patch_embedding.weight.dtype)
        # Gate training rounded RoPE buffers to BF16; keep that rounding local.
        vision_output = self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            patch_positions=patch_positions,
            rope_dtype=pixel_values.dtype,
        )
        merged = vision_output.last_hidden_state.reshape(-1, vision_output.last_hidden_state.shape[-1])
        merge = self.visual.spatial_merge_size
        time = int(image_grid_thw[:, 0].sum().item())
        patches_per_time = int(
            (image_grid_thw[0, 1].item() // merge)
            * (image_grid_thw[0, 2].item() // merge)
        )
        vision_tokens = merged.reshape(1, time, patches_per_time, -1)
        return vision_tokens

    def streammind_gate_forward(self, pixel_values, image_grid_thw, patch_positions=None):
        """Run the gate on one segment without changing the base LLM visual path."""
        gate = self._load_streammind_gate()
        vision_tokens = self._streammind_vision_tokens(
            pixel_values, image_grid_thw, patch_positions=patch_positions
        )
        return gate(vision_tokens)

    def streammind_gate_forward_segments(self, segments):
        """Run EPFE continuously over a list of codec segments from one stream."""
        gate = self._load_streammind_gate()
        tokens = [
            self._streammind_vision_tokens(
                segment["pixel_values"],
                segment["image_grid_thw"],
                patch_positions=segment.get("patch_positions"),
            )
            for segment in segments
        ]
        lengths = [token.shape[1] for token in tokens]
        boundaries = torch.tensor(lengths).cumsum(0).tolist()
        return gate(
            torch.cat(tokens, dim=1), response_positions=boundaries
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: Optional[torch.Tensor] = None,
        patch_positions=None,
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos: Pre-processed patches from Qwen2VL processor.
                `torch.Tensor` of shape `(total_patches, num_channels, patch_size, patch_size)`
            video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Convert to correct dtype
        pixel_values_videos = pixel_values_videos.type(self.visual.embeddings.patch_embedding.weight.dtype)

        # Forward through vision model with grid_thw
        vision_output = self.visual(pixel_values_videos, grid_thw=video_grid_thw, patch_positions=patch_positions)

        video_embeds = vision_output.last_hidden_state

        # Compute split sizes from video_grid_thw or from input shape
        if video_grid_thw is not None:
            split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        else:
            # Compute from input shape
            batch_size = pixel_values_videos.shape[0]
            split_sizes = [video_embeds.shape[1]] * batch_size

        # Split embeddings per video
        if len(split_sizes) > 1:
            video_embeds = torch.split(video_embeds.view(-1, video_embeds.shape[-1]), split_sizes)
        else:
            video_embeds = [video_embeds.view(-1, video_embeds.shape[-1])]

        return video_embeds

    def get_image_features(
        self, pixel_values, image_grid_thw: Optional[torch.Tensor] = None, patch_positions=None
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values: Pre-processed patches from Qwen2VL processor.
                - `torch.Tensor` of shape `(total_patches, num_channels, patch_size, patch_size)`
            image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        # Standard format from Qwen2VL processor
        if pixel_values.dim() == 2:
            # Convert to correct dtype
            pixel_values = pixel_values.type(self.visual.embeddings.patch_embedding.weight.dtype)

            # Forward through vision model with grid_thw
            vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, patch_positions=patch_positions)

            image_embeds = vision_output.last_hidden_state

            # Compute split sizes from grid_thw
            if image_grid_thw is not None:
                split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            else:
                # Fallback: assume single image
                split_sizes = [image_embeds.shape[0] if image_embeds.dim() == 2 else image_embeds.shape[1]]

            # Split embeddings per image
            image_embeds_flat = image_embeds.view(-1, image_embeds.shape[-1])
            if len(split_sizes) > 1:
                image_embeds = list(torch.split(image_embeds_flat, split_sizes))
            else:
                image_embeds = [image_embeds_flat]

            return image_embeds
        else:
            raise ValueError(
                f"Unsupported pixel_values shape: expected 4D tensor [total_patches, C, H, W], "
                f"got {pixel_values.shape if hasattr(pixel_values, 'shape') else type(pixel_values)}"
            )

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
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
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> MageVLModelOutputWithPast:
        r"""
        image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        patch_positions (`torch.Tensor` of shape `(total_patches, 3)` or `(1, total_patches, 3)`, *optional*):
            Explicit per-patch `(t, h, w)` position indices used by the vision tower to compute 3D rotary
            position embeddings (and the optional absolute position embedding inside the patch merger).
            `total_patches` is the sum of `t * h * w` across all images and videos in the batch, matching
            the layout produced by the Qwen2VL-style image processor.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Note: see the top-level ``MageVLForConditionalGeneration.forward``
        docstring; currently video flows in via the ``image_grid_thw`` / ``pixel_values``
        alias, so ``pixel_values_videos`` / ``video_grid_thw`` /
        ``second_per_grid_ts`` are unused at this layer.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if not isinstance(inputs_embeds, torch.Tensor):
            raise ValueError("Mage-VL needs input_ids or inputs_embeds.")

        image_embeds = None

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, patch_positions=patch_positions)

        if image_embeds is not None:
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw, patch_positions=patch_positions
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Simple 1D position_ids; a 2D padding mask yields per-row positions (left padding).
        if position_ids is None and attention_mask is not None and attention_mask.ndim == 2:
            all_positions = attention_mask.long().cumsum(-1) - 1
            all_positions.masked_fill_(attention_mask == 0, 1)
            position_ids = all_positions[:, -inputs_embeds.shape[1]:]

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return MageVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class MageVLForConditionalGeneration(MageVLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = MageVLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model.model._streammind_model_path = str(pretrained_model_name_or_path)
        return model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: Optional[torch.Tensor] = None,
        patch_positions=None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, patch_positions=patch_positions)

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def streammind_gate_forward(self, pixel_values, image_grid_thw, patch_positions=None):
        return self.model.streammind_gate_forward(
            pixel_values,
            image_grid_thw,
            patch_positions=patch_positions,
        )

    def streammind_gate_forward_segments(self, segments):
        return self.model.streammind_gate_forward_segments(segments)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ) -> MageVLCausalLMOutputWithPast:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        patch_positions (`torch.Tensor` of shape `(total_patches, 3)` or `(1, total_patches, 3)`, *optional*):
            Explicit per-patch `(t, h, w)` position indices used by the vision tower to compute 3D rotary
            position embeddings (and the optional absolute position embedding inside the patch merger).
            `total_patches` is the sum of `t * h * w` across all images and videos in the batch, matching
            the layout produced by the Qwen2VL-style image processor.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Note (native-video alias):
            The companion ``MageVLProcessor.__call__(videos=...)`` does NOT
            pass ``pixel_values_videos`` / ``video_grid_thw`` / ``second_per_grid_ts``
            to this forward. Instead it aliases the video patch tensor as
            ``pixel_values=`` and ``image_grid_thw=``, so video inputs share the
            same code path as multi-image inputs, including explicit patch
            positions and per-frame ``<X.X seconds>`` text tags. Direct video
            tensors are also accepted with their video grid metadata.
            ``second_per_grid_ts`` is retained for API compatibility, not
            consumed by the vision encoder.

        Example:

        ```python
        >>> from PIL import Image
        >>> from gptqmodel.models.ports.mage_vl.modeling_mage_vl import MageVLForConditionalGeneration
        >>> from gptqmodel.models.ports.mage_vl.processing_mage_vl import MageVLProcessor

        >>> model = MageVLForConditionalGeneration.from_pretrained("/path/to/Mage-VL")
        >>> processor = MageVLProcessor.from_pretrained("/path/to/Mage-VL")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> image = Image.open("/path/to/image.jpg").convert("RGB")

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True)
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            patch_positions=patch_positions,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep))

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return MageVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.Tensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.Tensor` of shape `(batch_size, num_videos_sample)`)
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

    # Transformers' return annotation uses the legacy LongTensor subclass.
    def _expand_inputs_for_generation(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
            "patch_positions",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    if image_grid_thw is None:
                        raise ValueError("Mage-VL image generation needs image_grid_thw.")
                    # split images into samples
                    samples = torch.split(image_grid_thw, image_nums.tolist())
                    # compute the sequence length of images for each sample
                    lengths = [int(torch.prod(sample, dim=1).sum().item()) for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = image_nums.tolist()
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    if video_grid_thw is None:
                        raise ValueError("Mage-VL video generation needs video_grid_thw.")
                    samples = torch.split(video_grid_thw, video_nums.tolist())
                    lengths = [int(torch.prod(sample, dim=1).sum().item()) for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = video_nums.tolist()
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=video_nums.tolist(), repeat_times=expand_size
                    )
                elif key == "patch_positions":
                    if image_grid_thw is not None and image_grid_thw.numel() > 0 and image_nums.sum() > 0:
                        samples = torch.split(image_grid_thw, image_nums.tolist())
                        lengths = [int(torch.prod(sample, dim=1).sum().item()) for sample in samples]
                    elif video_grid_thw is not None and video_grid_thw.numel() > 0 and video_nums.sum() > 0:
                        samples = torch.split(video_grid_thw, video_nums.tolist())
                        lengths = [int(torch.prod(sample, dim=1).sum().item()) for sample in samples]
                    else:
                        continue
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
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


__all__ = [
    "MageVLForConditionalGeneration",
    "MageVLModel",
    "MageVLPreTrainedModel",
]
