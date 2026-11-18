# SPDX-License-Identifier: Apache-2.0
# Adapted from ricursiveAI/Ripple-65B-Preview, revision
# 2f007aabfcedd0f8763d86d08f9aa0afbc344cea. See NOTICE.
"""Package-owned Ripple inference, preserving released parameter names and equations."""

from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_ricursive import RicursiveConfig


def _new_default_cache(config: RicursiveConfig) -> DynamicCache:
    """Each request owns its configured native hybrid cache."""
    return DynamicCache(config=config)


class RicursiveRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class RicursiveRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # NOTE: inv_freq is computed on-the-fly in forward() rather than stored
        # as a buffer, because non-persistent buffers get corrupted during
        # from_pretrained() loading.

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim)
        )
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)  # (B, 1, T, dim)
    sin = sin.unsqueeze(1)  # (B, 1, T, dim)
    # OLMo-core applies RoPE in full precision (float32) for numerical stability
    orig_dtype = q.dtype
    q_float, k_float = q.float(), k.float()
    cos_float, sin_float = cos.float(), sin.float()
    q_embed = (q_float * cos_float) + (rotate_half(q_float) * sin_float)
    k_embed = (k_float * cos_float) + (rotate_half(k_float) * sin_float)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class RicursiveAttention(nn.Module):
    def __init__(self, config: RicursiveConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        if config.use_valres:
            self.value_res_lambda = nn.Parameter(torch.zeros(1))

        # Per-head Q/K norms (RMSNorm on head_dim)
        self.q_norm = RicursiveRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RicursiveRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Gated attention
        if config.use_gated_attention:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        # RoPE only for local (sliding_attention) layers
        if self.layer_type == "sliding_attention":
            self.rotary_emb = RicursiveRotaryEmbedding(self.head_dim, theta=config.rope_theta)

    def _apply_sliding_window_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        q_len: int,
        kv_len: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: Optional[torch.LongTensor],
        total_kv: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if self.layer_type != "sliding_attention":
            return attention_mask

        # transformers >= 5 builds a hybrid cache: a sliding layer holds at most
        # `sliding_window` keys while a full-attention layer keeps growing, so the
        # shared mask produced by `_update_causal_mask` can be longer than this
        # layer's key tensor. Keep only the keys this layer actually holds.
        if attention_mask is not None and attention_mask.shape[-1] > kv_len:
            mask_len = attention_mask.shape[-1]
            if total_kv is not None:
                # The retained keys sit at `[total_kv - kv_len, total_kv)`.
                start = max(0, min(total_kv - kv_len, mask_len - kv_len))
                attention_mask = attention_mask[..., start : start + kv_len]
            else:
                attention_mask = attention_mask[..., -kv_len:]

        if cache_position is not None:
            query_positions = cache_position.to(device=device).reshape(-1)
        else:
            query_positions = torch.arange(kv_len - q_len, kv_len, device=device)

        window = self.config.sliding_window
        if window is None:
            return attention_mask
        if kv_len <= window:
            # The window cannot exclude any key, so the result is either the
            # unchanged (already causal) mask or, without one, plain causal
            # attention -- which SDPA applies through `is_causal`.
            if attention_mask is None:
                # SDPA's implicit causal mask is top-left aligned. It only
                # matches this query block when it is the whole sequence
                # (`q_len == kv_len`) or a single decode position
                # (`q_len == 1`). A multi-token block verified against a warm
                # cache sits at the *end* of the key span, so fall through and
                # build the end-anchored mask below.
                if q_len == 1 or q_len == kv_len:
                    return None
            if q_len == 1:
                return attention_mask

        # Keys are contiguous and end at the newest query position. After the cache
        # evicts from the left they no longer start at position 0, so anchor the
        # range at the first key the layer holds instead of at zero.
        key_positions = torch.arange(kv_len, device=device) + (
            query_positions[-1] - kv_len + 1
        )

        allowed = (
            (key_positions[None, :] <= query_positions[:, None])
            & (key_positions[None, :] >= query_positions[:, None] - (window - 1))
        )[None, None, :, :]

        if attention_mask is None:
            return allowed
        if attention_mask.dtype == torch.bool:
            return attention_mask & allowed
        # Finite minima would reopen an existing -inf exclusion on an empty row.
        return attention_mask.masked_fill(~allowed, float("-inf"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        value_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        value_states = hidden_states
        if value_residual is not None:
            if not self.config.use_valres:
                raise RuntimeError("value_residual was supplied but config.use_valres is false")
            blend = torch.sigmoid(self.value_res_lambda[0]).to(dtype=hidden_states.dtype)
            value_states = torch.lerp(
                value_residual.to(dtype=hidden_states.dtype), hidden_states, blend
            )
        v = self.v_proj(value_states)

        # Infer rank-local head counts from projections after tensor parallel sharding.
        q = q.view(bsz, q_len, -1, self.head_dim)
        k = k.view(bsz, q_len, -1, self.head_dim)
        v = v.view(bsz, q_len, -1, self.head_dim)

        # Per-head Q/K norm (applied before transpose, on head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (B, n_heads, T, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE only for local layers
        if self.layer_type == "sliding_attention":
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache
        total_kv = None
        if past_key_value is not None:
            # Native Transformers 5 caches own their write cursor.
            k, v = past_key_value.update(k, v, self.layer_idx)
            # Number of keys this layer has actually written. A preallocated
            # (StaticCache) layer returns its full capacity from `update()`,
            # including slots that have not been filled yet; an evicting
            # (sliding-window) layer reports more than the trailing window it
            # returns. The mask must use the real positions, not the tensor size.
            total_kv = _cache_layer_seq_length(past_key_value, self.layer_idx)

        # Drop the unwritten tail of a preallocated cache. Without this a
        # six-token prefill into a sixteen-slot StaticCache looks like a
        # ten-token prefix, and the end-anchored verify mask below would let
        # each query attend slots belonging to a different absolute position.
        if total_kv is not None and 0 < total_kv < k.shape[2]:
            k = k[:, :, :total_kv]
            v = v[:, :, :total_kv]

        # Expand only written slots, not the entire capacity of a static cache.
        num_kv_groups = q.shape[1] // k.shape[1]
        k = repeat_kv(k, num_kv_groups)
        v = repeat_kv(v, num_kv_groups)

        # Align an incoming mask that is longer than the keys this layer holds.
        # The shared mask is built over absolute key positions
        # `0..target_length-1`; keep the slice that matches the keys present.
        #
        # The keys returned by `update()` occupy the absolute positions
        # `[total_kv - kv_len, total_kv)`. Anchoring on the tensor tail
        # (`[-kv_len:]`) is only correct when the mask length equals
        # `total_kv`; with a mask sized to the *allocated cache capacity* it
        # selects the unwritten tail instead of the retained window and
        # silently produces wrong attention.
        if attention_mask is not None and attention_mask.shape[-1] > k.shape[2]:
            kv_len_now = k.shape[2]
            mask_len = attention_mask.shape[-1]
            if total_kv is not None:
                start = max(0, min(total_kv - kv_len_now, mask_len - kv_len_now))
                attention_mask = attention_mask[..., start : start + kv_len_now]
            else:
                # Unknown cache: the keys are the trailing window.
                attention_mask = attention_mask[..., -kv_len_now:]

        attention_mask = self._apply_sliding_window_mask(
            attention_mask=attention_mask,
            q_len=q.shape[2],
            kv_len=k.shape[2],
            dtype=q.dtype,
            device=q.device,
            cache_position=cache_position,
            total_kv=total_kv,
        )

        if attention_mask is None and q.shape[2] > 1 and k.shape[2] > q.shape[2]:
            # Multi-token verify block against a warm cache. SDPA's implicit
            # causal mask is top-left aligned, so when `q_len < kv_len` it hides
            # every key before the block. Build the mask from the actual query
            # positions (and the absolute positions of the returned keys) rather
            # than assuming the key tensor starts at position 0.
            if cache_position is not None and cache_position.numel() == q.shape[2]:
                query_positions = cache_position.to(
                    device=q.device, dtype=torch.long
                ).reshape(-1)
            else:
                query_positions = torch.arange(
                    k.shape[2] - q.shape[2], k.shape[2], device=q.device
                )
            if total_kv is not None and total_kv > k.shape[2]:
                # Evicting cache: the returned keys are the trailing window that
                # ends at the newest query, not positions 0..kv_len-1.
                last_query = int(query_positions[-1].item())
                key_positions = torch.arange(
                    last_query - k.shape[2] + 1, last_query + 1, device=q.device
                )
            else:
                key_positions = torch.arange(k.shape[2], device=q.device)
            attention_mask = (
                key_positions[None, :] <= query_positions[:, None]
            )[None, None]

        # Attention (use is_causal when no explicit mask for correct causal behavior)
        # Cast to plain bool to avoid SymBool issues with torch.compile
        is_causal = bool(attention_mask is None and q.shape[2] > 1)
        # Keep the SDPA result even when returning diagnostic weights. Rebuilding
        # the output from rounded BF16 probabilities changes logits and token IDs.
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_causal
        )
        attn_weights = None
        if output_attentions:
            # SDPA does not expose weights; compute them separately only on request.
            weight_keys = k
            with torch.autocast(device_type=q.device.type, enabled=False):
                scores = torch.matmul(q.float(), weight_keys.float().transpose(-1, -2)) * self.head_dim**-0.5
                if is_causal:
                    causal = torch.ones(q_len, weight_keys.shape[2], dtype=torch.bool, device=q.device).tril()
                    scores = scores.masked_fill(~causal, float("-inf"))
                if attention_mask is not None:
                    if attention_mask.dtype == torch.bool:
                        scores = scores.masked_fill(~attention_mask, float("-inf"))
                    else:
                        scores = scores + attention_mask.float()
                attn_weights = F.softmax(scores, dim=-1).nan_to_num().to(q.dtype)

        # (B, n_heads, T, head_dim) -> (B, T, n_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Gated attention: output * sigmoid(g_proj(input))
        if self.config.use_gated_attention:
            gate = self.g_proj(hidden_states)
            if self.config.gate_full_precision:
                gate = gate.float()
            gate = torch.sigmoid(gate).to(attn_output.dtype)
            attn_output = attn_output.view(bsz, q_len, -1) * gate
        else:
            attn_output = attn_output.view(bsz, q_len, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class RicursiveMLP(nn.Module):
    def __init__(self, config: RicursiveConfig):
        super().__init__()
        self.clamp_threshold = config.ffn_clamp_threshold
        self.chunk_size = config.mlp_chunk_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def _forward_rows(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.clamp_threshold:
            gate = gate.clamp(min=None, max=self.clamp_threshold)
            up = up.clamp(min=-self.clamp_threshold, max=self.clamp_threshold)
        return self.down_proj(F.silu(gate) * up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ripple's MLP expands hidden_size -> intermediate_size (5120 -> 40960
        # for the 65B checkpoint), so a long prefill materialises `gate`, `up`
        # and `silu(gate) * up` at once. Row chunking preserves the equations;
        # backend GEMM selection can still introduce low-precision rounding
        # differences. Concatenating the pieces bounds the transient peak.
        # The config owns the row budget; zero disables chunking.
        chunk = self.chunk_size
        if chunk <= 0 or not torch.is_tensor(x) or x.dim() not in (2, 3):
            return self._forward_rows(x)

        shape = x.shape
        hidden = shape[-1]
        rows = 1
        for size in shape[:-1]:
            rows *= size
        if rows <= chunk:
            return self._forward_rows(x)

        flat = x.reshape(rows, hidden)
        pieces = [
            self._forward_rows(flat[start : start + chunk])
            for start in range(0, rows, chunk)
        ]
        out = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        return out.reshape(*shape[:-1], out.shape[-1])


class RicursiveDecoderLayer(nn.Module):
    """PeriNorm decoder layer with 4 layer norms."""

    def __init__(self, config: RicursiveConfig, layer_idx: int):
        super().__init__()
        self.self_attn = RicursiveAttention(config, layer_idx)
        self.mlp = RicursiveMLP(config)

        # PeriNorm: 4 norms
        self.input_layernorm = RicursiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RicursiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RicursiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RicursiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        value_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        # Attention sub-block (PeriNorm: pre-norm -> attn -> post-norm -> residual)
        normed = self.input_layernorm(hidden_states)
        attn_output, attn_weights, present_key_value = self.self_attn(
            normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            value_residual=value_residual,
            **kwargs,
        )
        hidden_states = hidden_states + self.post_attention_layernorm(attn_output)

        # FFN sub-block (PeriNorm: pre-norm -> ffn -> post-norm -> residual)
        normed = self.pre_feedforward_layernorm(hidden_states)
        ff_output = self.mlp(normed)
        hidden_states = hidden_states + self.post_feedforward_layernorm(ff_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class RicursivePreTrainedModel(PreTrainedModel):
    config_class = RicursiveConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RicursiveDecoderLayer"]
    _supports_sdpa = True


class RicursiveModel(RicursivePreTrainedModel):
    def __init__(self, config: RicursiveConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [RicursiveDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RicursiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        checkpointing = self.gradient_checkpointing and self.training
        if checkpointing:
            if past_key_values is not None:
                raise ValueError("Gradient checkpointing does not support a supplied KV cache")
            # Recomputed layers must not append to a cache for a second time.
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed_scale: non-folded checkpoints carry the scale (e.g. sqrt(hidden_size)) in the
        # config and apply it here at runtime; folded checkpoints set embed_scale=None and bake
        # the scale into embed_tokens.weight, so this is a no-op. This mirrors the SGLang native
        # port (sglang_ricursive) so served and trained forwards agree. It must run BEFORE the
        # value-residual seed below, whose source is the (scaled) layer-0 embeddings.
        embed_scale = getattr(self.config, "embed_scale", None)
        if embed_scale is not None:
            inputs_embeds = inputs_embeds * embed_scale

        if use_cache and past_key_values is None:
            past_key_values = _new_default_cache(self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Build causal mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values
        )

        hidden_states = inputs_embeds
        value_residual = hidden_states if self.config.use_valres else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if checkpointing:
                # Pass the value-residual seed as an input too, so gradients
                # through the embedding path survive reentrant checkpointing.
                layer_outputs = self._gradient_checkpointing_func(
                    partial(layer.__call__, **kwargs), hidden_states, causal_mask,
                    position_ids, None, output_attentions, False, cache_position,
                    value_residual,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    value_residual=value_residual,
                    **kwargs,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
    ) -> Optional[torch.Tensor]:
        # When there's no explicit attention_mask (no padding) and we're doing
        # standard prefill or cached decoding, return None and let SDPA use
        # its efficient is_causal path. This also avoids numerical issues
        # from float min mask values with large activations.
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            # A caller-provided 4D mask already defines visibility (or additive
            # bias). Preserve it; each local layer still applies its window.
            dtype = torch.bool if attention_mask.dtype == torch.bool else input_tensor.dtype
            return attention_mask.to(device=input_tensor.device, dtype=dtype)
        if torch.all(attention_mask == 1):
            return None

        # Only build an explicit mask when we have a padding mask
        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]

        if cache_position is None:
            query_positions = torch.arange(sequence_length, device=device)
        else:
            query_positions = cache_position.to(device=device).reshape(-1)

        # Size the key axis from `cache_position`, which is exactly what the
        # attention layer uses to size its key tensor.
        # `past_key_values.get_max_cache_shape()` was removed in transformers 5,
        # and its replacement reports the *capacity* (e.g. `sliding_window`),
        # which disagrees with the live key count whenever the cache is not full.
        target_length = int(query_positions[-1].item()) + 1
        target_length = max(target_length, attention_mask.shape[-1])

        # A shorter padding mask leaves the trailing keys unspecified; treat them as
        # visible -- the causal term below still applies.
        if attention_mask.shape[-1] < target_length:
            pad = attention_mask.new_ones(
                attention_mask.shape[0], target_length - attention_mask.shape[-1]
            )
            attention_mask = torch.cat([pad, attention_mask], dim=-1)

        key_positions = torch.arange(target_length, device=device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        allowed = allowed[None, None, :, :] & attention_mask[:, None, None, :].to(torch.bool)

        causal_mask = torch.zeros(
            (attention_mask.shape[0], 1, sequence_length, target_length),
            dtype=dtype,
            device=device,
        )
        # `0.0` for visible keys (not `1.0`, which would bias every score) and the
        # dtype minimum elsewhere. Building the mask from `allowed` also keeps a
        # single-token decode step from coming out fully masked.
        return causal_mask.masked_fill(~allowed, torch.finfo(dtype).min)


def _cache_seq_length(past_key_values: Optional[Cache]) -> int:
    """Number of written keys in the caller-owned native cache."""
    if past_key_values is None:
        return 0
    return int(past_key_values.get_seq_length())


def _cache_layer_seq_length(
    past_key_values: Optional[Cache], layer_idx: int
) -> Optional[int]:
    """Actual number of keys written for `layer_idx`, or `None` if unknown.

    `Cache.get_seq_length(layer_idx)` counts *written* tokens. For a
    preallocated static layer that is smaller than the tensor `update()`
    returns until the cache fills; for an evicting sliding-window layer it is
    larger, because the layer only returns a trailing window.
    """
    if past_key_values is None:
        return None
    return int(past_key_values.get_seq_length(layer_idx))


class RicursiveForCausalLM(RicursivePreTrainedModel, GenerationMixin):
    # Transformers 5 target-to-source tied-weight ownership.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    # Transformers 5 no longer injects `GenerationMixin`; it is inherited explicitly.

    def __init__(self, config: RicursiveConfig):
        super().__init__(config)
        self.model = RicursiveModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        # Honour `logits_to_keep` by slicing the LM head *input*. Projecting the
        # full `[batch, seq, vocab]` tensor only to discard all but the last row
        # costs several GiB of transient memory at prefill batch sizes.
        #
        # When `labels` are supplied the full sequence must be kept: the causal-LM
        # loss shifts logits and labels by one position, so trimming both to the
        # same short window leaves the shift empty (and the loss NaN). Labeled
        # calls therefore project the full sequence to get a defined loss, then
        # trim only the *returned* logits so the output shape still honours
        # `logits_to_keep`. The generation path, which never passes labels, keeps
        # the memory win from slicing the LM head input.
        keep = (
            int(logits_to_keep)
            if logits_to_keep is not None and logits_to_keep > 0
            else 0
        )
        if keep and labels is None and hidden_states.shape[1] > keep:
            hidden_states = hidden_states[:, -keep:, :]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].to(logits.device).contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum" if num_items_in_batch is not None else "mean",
            )
            if num_items_in_batch is not None:
                # Trainer supplies the token count across all accumulated microbatches.
                loss = loss / torch.as_tensor(num_items_in_batch, device=loss.device)
            if keep:
                # Loss used the full sequence; only the returned tensor is trimmed.
                logits = logits[..., -keep:, :]

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
    ):
        """Advance explicit positions when the generation backend leaves them unchanged."""
        position_ids = model_kwargs.get("position_ids")
        cache_position = model_kwargs.get("cache_position")
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        # Preserve caller-supplied positions if a generation strategy retains
        # their original tensor rather than extending it.
        if position_ids is not None and model_kwargs.get("position_ids") is position_ids:
            offsets = torch.arange(
                1, num_new_tokens + 1, device=position_ids.device, dtype=position_ids.dtype
            )
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, position_ids[:, -1:] + offsets], dim=-1
            )
        # Transformers 5 no longer maintains cache_position. Keep an explicitly
        # supplied vector in sync if a generation strategy retains it.
        if cache_position is not None and model_kwargs.get("cache_position") is cache_position:
            if model_kwargs.get("use_cache", True):
                model_kwargs["cache_position"] = cache_position[-1:] + num_new_tokens
            else:
                offsets = torch.arange(
                    1, num_new_tokens + 1, device=cache_position.device, dtype=cache_position.dtype
                )
                model_kwargs["cache_position"] = torch.cat([cache_position, cache_position[-1:] + offsets])
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        next_sequence_length: Optional[int] = None,
        is_first_iteration: Optional[bool] = None,
        **kwargs,
    ):
        """Prepare a full prompt or a new-only block for cached generation.

        With a warm cache and no explicit ``cache_position``, inputs longer
        than the cached prefix are full prompts; shorter or equal-length inputs
        are new-only blocks. For a longer new-only block, pass
        ``torch.arange(past_length, past_length + input_ids.shape[1])`` as
        ``cache_position``. Attention masks may be padded to cache capacity and
        therefore cannot identify which input layout the caller supplied.
        """
        # transformers 5.x does not move remote-code inputs onto the model device for
        # this model, so a plain CPU `input_ids` reaches the embedding lookup and
        # raises. Its own generation helper performs this move, so mirror it here.
        device = getattr(self, "device", None)
        if device is not None and getattr(device, "type", None) != "meta":
            if input_ids is not None and input_ids.device != device:
                input_ids = input_ids.to(device)
            if attention_mask is not None and attention_mask.device != device:
                attention_mask = attention_mask.to(device)
            if inputs_embeds is not None and inputs_embeds.device != device:
                inputs_embeds = inputs_embeds.to(device)
            # transformers derives `position_ids` from the *caller's* tensors before
            # this hook runs, so a CPU prompt yields CPU positions even after the
            # move above. RoPE then multiplies them against CUDA frequencies and fails.
            if position_ids is not None and position_ids.device != device:
                position_ids = position_ids.to(device)

        past_length = _cache_seq_length(past_key_values)

        use_embeds = inputs_embeds is not None and (
            is_first_iteration or past_length == 0 or input_ids is None or input_ids.numel() == 0
        )
        input_tensor = inputs_embeds if use_embeds else input_ids
        input_length = input_tensor.shape[1]

        if cache_position is None:
            # Inputs no longer than the cache are treated as new-only blocks.
            # Longer inputs default to full prompts containing the cached prefix.
            # Mask width/content cannot distinguish a longer new-only block from
            # a full prompt with a capacity-sized mask (even an all-visible one).
            # Callers supplying longer new-only blocks must pass cache_position.
            full_prompt = past_length > 0 and input_length > past_length
            new_len = (
                next_sequence_length if next_sequence_length is not None
                else input_length - past_length if full_prompt else input_length
            )
            cache_position = torch.arange(
                past_length, past_length + new_len, device=input_tensor.device
            )
        else:
            cache_position = cache_position.to(input_tensor.device)

        # Only slice when the cache is warm and `input_ids` still carries cached
        # tokens. Slicing unconditionally turned a `None` cache position into
        # `input_ids[:, None]` and injected a bogus axis.
        new_len = cache_position.numel()
        sliced = past_length > 0 and input_length > new_len
        if sliced:
            # During continuation, input_ids may contain only the new prompt
            # and generated tokens. Absolute cache offsets cannot index that
            # shorter buffer. Both layouts end with the unprocessed tokens.
            input_tensor = input_tensor[:, -new_len:] if new_len else input_tensor[:, :0]

        if use_embeds:
            model_inputs = {"inputs_embeds": input_tensor}
        else:
            model_inputs = {"input_ids": input_tensor}

        sequence_length = input_tensor.shape[1]
        if attention_mask is not None and attention_mask.dim() == 2 and new_len:
            required_length = int(cache_position[-1]) + 1
            if attention_mask.shape[-1] < required_length:
                # A current-block mask omits an implicitly visible cached prefix.
                attention_mask = F.pad(attention_mask, (required_length - attention_mask.shape[-1], 0), value=1)

        # Derive per-row positions from the padding mask. Using `cache_position` for
        # every row is only correct when nothing is padded: with left padding the real
        # tokens of a short row sit at higher indices, so a shared position vector
        # shifts its RoPE phases and the row decodes to garbage.
        if position_ids is not None:
            # Callers may supply positions for the full prompt/capacity, or may
            # already have selected just the positions for the current block.
            if position_ids.shape[-1] != sequence_length:
                if new_len and position_ids.shape[-1] > int(cache_position[-1]):
                    position_ids = position_ids[:, cache_position]
                else:
                    position_ids = position_ids[:, -sequence_length:] if sequence_length else position_ids[:, :0]
        elif attention_mask is not None and attention_mask.dim() == 2:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            # The mask may include unused cache capacity after the current block.
            # Select its actual cache slots, not the final entries of that mask.
            position_ids = position_ids[:, cache_position]
        else:
            position_ids = cache_position.unsqueeze(0)

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        # Generation output flags and model-specific forward options must reach
        # forward. Labels are never generation inputs.
        model_inputs.update({key: value for key, value in kwargs.items() if key != "labels"})
        if use_cache is not False:
            # Generation only reads the final position. Requesting one row keeps
            # the 100k-way LM head from projecting the whole prompt.
            model_inputs.setdefault("logits_to_keep", 1)
        return model_inputs
