# SPDX-License-Identifier: Apache-2.0
# Adapted from ricursiveAI/Ripple-65B-Preview, revision
# 2f007aabfcedd0f8763d86d08f9aa0afbc344cea. See NOTICE.
"""Package-owned Ripple inference, preserving released parameter names and equations."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        value_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        num_kv_groups = q.shape[1] // k.shape[1]
        k = repeat_kv(k, num_kv_groups)
        v = repeat_kv(v, num_kv_groups)

        # The masks come from `masking_utils` for this layer type (sliding window
        # included). Without one, SDPA's causal flag covers prefill; cast to a
        # plain bool to avoid SymBool issues with torch.compile.
        is_causal = bool(attention_mask is None and q.shape[2] > 1)
        # Keep the SDPA result even when returning diagnostic weights. Rebuilding
        # the output from rounded BF16 probabilities changes logits and token IDs.
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_causal
        )
        attn_weights = None
        if kwargs.get("output_attentions", False):
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

        return attn_output, attn_weights


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


class RicursiveDecoderLayer(GradientCheckpointingLayer):
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
        value_residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        # `value_residual` is positional so reentrant gradient checkpointing
        # keeps the gradient path through the layer-0 embeddings.
        # Attention sub-block (PeriNorm: pre-norm -> attn -> post-norm -> residual)
        normed = self.input_layernorm(hidden_states)
        attn_output, _ = self.self_attn(
            normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            value_residual=value_residual,
            **kwargs,
        )
        hidden_states = hidden_states + self.post_attention_layernorm(attn_output)

        # FFN sub-block (PeriNorm: pre-norm -> ffn -> post-norm -> residual)
        normed = self.pre_feedforward_layernorm(hidden_states)
        ff_output = self.mlp(normed)
        return hidden_states + self.post_feedforward_layernorm(ff_output)


class RicursivePreTrainedModel(PreTrainedModel):
    config_class = RicursiveConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RicursiveDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _can_record_outputs = {"hidden_states": RicursiveDecoderLayer, "attentions": RicursiveAttention}


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

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
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

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            full_mask = create_causal_mask(**mask_kwargs)
            causal_mask_mapping = {"full_attention": full_mask, "sliding_attention": full_mask}
            # Without a window, local layers keep RoPE but attend causally over the whole prefix.
            if self.config.sliding_window is not None and "sliding_attention" in self.config.layer_types:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        value_residual = hidden_states if self.config.use_valres else None
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                value_residual,
                attention_mask=causal_mask_mapping[layer.self_attn.layer_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


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

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Optional[int] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
