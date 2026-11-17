# Adapted from NVIDIA Hymba, revision bc2d2b43bc73d62ad8bfc9e6a3dcc26ab0795e77.
# NVIDIA Open Model License Agreement, June 2024. See LICENSE.pdf and NOTICE.
# Attention/RMSNorm/MLP origins: Hugging Face Transformers (Apache-2.0).
"""Checkpoint-compatible Hymba with portable Torch attention and selective SSM.

The reference path intentionally has no flash-attn, Triton, causal-conv1d or
mamba-ssm dependency. It preserves fused Q/K/V/SSM projection rows, independent
SSM states per layer, producer-owned shared KV, and memory-aware local masks.
"""

import threading
from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast

from .configuration_hymba import HymbaConfig


class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        x = hidden_states.float()
        return self.weight * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)).to(
            hidden_states.dtype
        )


class PerheadHymbaRMSNorm(HymbaRMSNorm):
    def __init__(self, hidden_size, num_heads, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter(torch.ones(1, num_heads, 1, hidden_size))


class LlamaRotaryEmbedding(nn.Module):
    """NVIDIA's fixed/dynamic NTK convention, without model-owned mutable state."""

    def __init__(self, config, dim, base=10000):
        super().__init__()
        self.config, self.dim, self.base = config, dim, base
        self.register_buffer("inv_freq", self._frequencies(base), persistent=False)

    def _frequencies(self, base, device=None):
        return 1.0 / (base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))

    @torch.no_grad()
    def forward(self, x, position_ids):
        base = self.base
        config = self.config
        if config.rope_type == "ntk":
            base *= (2 * config.max_position_embeddings / config.orig_max_position_embeddings - 1) ** (
                self.dim / (self.dim - 2)
            )
        elif config.rope_type == "dynamic_ntk":
            length = int(position_ids.max()) + 1
            if length > config.orig_max_position_embeddings:
                base *= (2 * length / config.orig_max_position_embeddings - 1) ** (self.dim / (self.dim - 2))
        inv_freq = self._frequencies(base, x.device)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = position_ids.float()[..., None] * inv_freq
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate(x, cos, sin):
    left, right = x.chunk(2, dim=-1)
    return x * cos[:, None] + torch.cat((-right, left), dim=-1) * sin[:, None]


class HybridMambaAttentionDynamicCache(Cache):
    """Request-owned hybrid state. A partially updated request must be reset.

    KV tensors exist only for producer layers; consumers use the producer's
    tensor directly within each forward. Logical length includes memory tokens
    and is independent of KV length, user-token count and cache storage policy.
    """

    is_compileable = False

    def __init__(self, config, batch_size, dtype=torch.float32, device=None, layer_type=None):
        super().__init__(layers=[])
        self._lock = threading.RLock()
        self._depth = 0
        self._invalid = False
        self._owner = None
        self.signature = self._signature(config)
        self.dtype = dtype
        self._batch_size = batch_size
        self.layer_type = list(config.layer_type)
        inner = int(config.hidden_size * config.mamba_expand)
        self.conv_states = [torch.zeros(batch_size, inner, config.mamba_d_conv, device=device, dtype=dtype)
                            for _ in range(config.num_hidden_layers)]
        self.ssm_states = [torch.zeros(batch_size, inner, config.mamba_d_state, device=device, dtype=dtype)
                           for _ in range(config.num_hidden_layers)]
        self.key_cache = [None] * config.num_hidden_layers
        self.value_cache = [None] * config.num_hidden_layers
        self._length = 0
        self.user_seq_length = 0
        self.padding_mask = None
        self.memory_mask = None
        self.has_previous_state = False

    @staticmethod
    def _signature(config):
        return tuple(repr(getattr(config, key)) for key in (
            "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "mamba_expand", "mamba_d_state", "mamba_d_conv", "kv_sources", "num_memory_tokens",
            "memory_tokens_interspersed_every", "rope_type", "rope_theta", "sliding_window", "global_attn_idx",
        ))

    @contextmanager
    def transaction(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A Hymba cache cannot be used concurrently by two requests")
        try:
            if self._invalid:
                raise RuntimeError("Hymba cache is incomplete after a failed forward; reset it before retrying")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._invalid = True
                raise
            finally:
                self._depth -= 1
        finally:
            self._lock.release()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.key_cache[layer_idx] is not None:
            key_states = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
            value_states = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)
        self.key_cache[layer_idx], self.value_cache[layer_idx] = key_states, value_states
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        return self._length

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self._length

    def get_max_cache_shape(self, layer_idx=0):
        return -1

    def get_mask_sizes(self, query_length, layer_idx=0):
        return self._length + query_length, 0

    @property
    def max_batch_size(self):
        return self._batch_size

    @property
    def batch_size(self):
        return self._batch_size

    def reorder_cache(self, beam_idx):
        with self.transaction():
            for states in (self.key_cache, self.value_cache, self.conv_states, self.ssm_states):
                for i, state in enumerate(states):
                    if state is not None:
                        states[i] = state.index_select(0, beam_idx.to(state.device))
            if self.padding_mask is not None:
                self.padding_mask = self.padding_mask.index_select(0, beam_idx.to(self.padding_mask.device))
                self.memory_mask = self.memory_mask.index_select(0, beam_idx.to(self.memory_mask.device))
            self._batch_size = len(beam_idx)

    batch_select_indices = reorder_cache

    def batch_repeat_interleave(self, repeats):
        device = self.conv_states[0].device
        self.reorder_cache(torch.arange(self.batch_size, device=device).repeat_interleave(repeats))

    def reset(self):
        with self._lock:
            if self._depth:
                raise RuntimeError("Cannot reset an active Hymba cache")
            for states in (self.conv_states, self.ssm_states):
                for state in states:
                    state.zero_()
            self.key_cache = [None] * len(self.key_cache)
            self.value_cache = [None] * len(self.value_cache)
            self._length = self.user_seq_length = 0
            self.padding_mask = self.memory_mask = None
            self.has_previous_state = self._invalid = False
            self._owner = None

    def crop(self, max_length):
        raise NotImplementedError("Hymba recurrent state cannot be cropped; replay into a fresh cache")

    def to_legacy_cache(self):
        raise NotImplementedError("Hymba recurrent state has no legacy KV-only cache representation")


class HymbaAttention(nn.Module):
    def __init__(self, config, layer_idx, reuse_kv=False):
        super().__init__()
        self.config, self.layer_idx, self.reuse_kv = config, layer_idx, reuse_kv
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.kq_head_dim = config.hidden_size // self.num_heads
        self.v_head_dim = int(config.mamba_expand * config.hidden_size) // self.num_heads
        self.attention_dropout = config.attention_dropout
        self.q_norm = self.k_norm = None
        if config.kq_norm == "rms":
            self.q_norm = HymbaRMSNorm(self.kq_head_dim)
            self.k_norm = HymbaRMSNorm(self.kq_head_dim)
        elif config.kq_norm == "perhead-rms":
            self.q_norm = PerheadHymbaRMSNorm(self.kq_head_dim, self.num_heads)
            self.k_norm = PerheadHymbaRMSNorm(self.kq_head_dim, self.num_key_value_heads)
        if config.rope:
            self.rotary_emb = LlamaRotaryEmbedding(config, self.kq_head_dim, config.rope_theta)

    def forward(self, query_states, key_states=None, value_states=None, *, position_ids,
                attention_mask, kv_last_layer=None, past_key_value=None, use_cache=False,
                use_swa=False, memory_mask=None, **kwargs):
        batch, length, _ = query_states.shape
        query = query_states.view(batch, length, self.num_heads, self.kq_head_dim).transpose(1, 2)
        position_ids = position_ids.to(query.device)
        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.config.rope:
            cos, sin = self.rotary_emb(query, position_ids)
            query = _rotate(query, cos, sin)
        if self.reuse_kv:
            if kv_last_layer is None:
                raise ValueError("Hymba KV consumer requires its producer's current key/value tensors")
            key, value = kv_last_layer
        else:
            key = key_states.view(batch, length, self.num_key_value_heads, self.kq_head_dim).transpose(1, 2)
            value = value_states.view(batch, length, self.num_key_value_heads, self.v_head_dim).transpose(1, 2)
            if self.k_norm is not None:
                key = self.k_norm(key)
            if self.config.rope:
                key = _rotate(key, cos, sin)
            if use_cache:
                key, value = past_key_value.update(key, value, self.layer_idx)
        key, value = key.to(query.device), value.to(query.device)
        shared = (key, value)
        repeats = self.num_heads // self.num_key_value_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        key_length = key.shape[-2]
        key_index = torch.arange(key_length, device=query.device)
        query_index = torch.arange(key_length - length, key_length, device=query.device)
        causal = query_index[:, None] >= key_index[None, :]
        if use_swa:
            # NVIDIA Flex uses an inclusive distance bound. Memory keys remain
            # visible outside the local window, subject to causality/padding.
            local = query_index[:, None] - key_index[None, :] <= self.config.sliding_window
            if memory_mask is not None:
                local = local[None] | memory_mask.to(query.device)[:, None, :]
            else:
                local = local[None]
            causal = causal[None] & local
        else:
            causal = causal[None]
        mask = causal[:, None] & attention_mask.to(query.device)[:, None, None, :].bool()
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                               dropout_p=self.attention_dropout if self.training else 0.0)
        return output.transpose(1, 2).reshape(batch, length, -1), shared


def selective_scan_reference(u, delta, A, B, C, D, z, initial_state=None):
    """Mamba selective_scan equations in FP32, including nonzero continuation.

    u/delta/z: B,L,I; A: I,N; B/C: B,L,N. The last state is returned
    separately, with no persistent dequantized or expanded weight storage.
    """
    state = u.new_zeros((u.shape[0], u.shape[2], A.shape[1]), dtype=torch.float32)
    if initial_state is not None:
        state = initial_state.float()
    values = []
    for t in range(u.shape[1]):
        dt = F.softplus(delta[:, t].float())
        state = (dt[..., None] * A).exp() * state + (
            dt[..., None] * B[:, t, None].float() * u[:, t, :, None].float()
        )
        y = (state * C[:, t, None].float()).sum(-1) + D.float() * u[:, t].float()
        values.append(y * F.silu(z[:, t].float()))
    return torch.stack(values, dim=1).to(u.dtype), state


class HymbaBlock(nn.Module):
    def __init__(self, config, layer_idx, reuse_kv=False):
        super().__init__()
        self.config, self.layer_idx, self.reuse_kv = config, layer_idx, reuse_kv
        self.hidden_size = config.hidden_size
        self.intermediate_size = int(config.hidden_size * config.mamba_expand)
        self.time_step_rank, self.ssm_state_size = config.mamba_dt_rank, config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.attn_hidden_size = config.hidden_size
        self.k_hidden_size = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
        self.v_hidden_size = config.num_key_value_heads * (self.intermediate_size // config.num_attention_heads)
        self.latent_dim = self.intermediate_size + self.attn_hidden_size
        if not reuse_kv:
            self.latent_dim += self.k_hidden_size + self.v_hidden_size
        self.self_attn = HymbaAttention(config, layer_idx, reuse_kv)
        self.pre_avg_layernorm1 = HymbaRMSNorm(self.intermediate_size, config.rms_norm_eps)
        self.pre_avg_layernorm2 = HymbaRMSNorm(self.intermediate_size, config.rms_norm_eps)
        self.in_proj = nn.Linear(config.hidden_size, self.latent_dim + self.intermediate_size,
                                 bias=config.mamba_proj_bias)
        self.out_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=config.mamba_proj_bias)
        self.conv1d = nn.Conv1d(self.intermediate_size, self.intermediate_size, config.mamba_d_conv,
                                groups=self.intermediate_size, bias=config.mamba_conv_bias)
        self.x_proj = nn.ModuleList([nn.Linear(self.intermediate_size, self.time_step_rank + 2 * self.ssm_state_size,
                                              bias=False)])
        self.dt_proj = nn.ModuleList([nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)])
        A = torch.arange(1, self.ssm_state_size + 1).float()[None].expand(self.intermediate_size, -1)
        self.A_log = nn.ParameterList([nn.Parameter(A.log().contiguous())])
        self.D = nn.ParameterList([nn.Parameter(torch.ones(self.intermediate_size))])
        self.dt_layernorm = self.B_layernorm = self.C_layernorm = None
        if config.mamba_inner_layernorms:
            self.dt_layernorm = HymbaRMSNorm(self.time_step_rank, config.rms_norm_eps)
            self.B_layernorm = HymbaRMSNorm(self.ssm_state_size, config.rms_norm_eps)
            self.C_layernorm = HymbaRMSNorm(self.ssm_state_size, config.rms_norm_eps)

    def forward(self, hidden_states, past_key_value=None, attention_mask=None, position_ids=None,
                kv_last_layer=None, use_cache=False, use_swa=False, memory_mask=None, **kwargs):
        projected = self.in_proj(hidden_states)
        current_mask = attention_mask[:, -projected.shape[1]:, None].to(projected)
        projected = projected * current_mask
        latent, gate = projected.split((self.latent_dim, self.intermediate_size), dim=-1)
        if self.reuse_kv:
            query, x = latent.split((self.attn_hidden_size, self.intermediate_size), dim=-1)
            key = value = None
        else:
            query, key, value, x = latent.split(
                (self.attn_hidden_size, self.k_hidden_size, self.v_hidden_size, self.intermediate_size), dim=-1
            )
        raw = x.transpose(1, 2)
        if use_cache:
            # k-1 historical inputs plus every current input; works for short
            # prefill, one-token decode and multi-token continuation alike.
            history = past_key_value.conv_states[self.layer_idx].to(raw.device)
            extended = torch.cat((history[..., 1:], raw), dim=-1)
            conv_state = torch.cat((history, raw), dim=-1)[..., -self.conv_kernel_size:]
        else:
            extended = F.pad(raw, (self.conv_kernel_size - 1, 0))
        x = ACT2FN[self.config.hidden_act](self.conv1d(extended)).transpose(1, 2) * current_mask
        attention, shared = self.self_attn(
            query, key, value, position_ids=position_ids, attention_mask=attention_mask,
            kv_last_layer=kv_last_layer, past_key_value=past_key_value, use_cache=use_cache,
            use_swa=use_swa, memory_mask=memory_mask,
        )
        dt, B, C = self.x_proj[0](x).split((self.time_step_rank, self.ssm_state_size, self.ssm_state_size), dim=-1)
        if self.dt_layernorm is not None:
            dt, B, C = self.dt_layernorm(dt), self.B_layernorm(B), self.C_layernorm(C)
        # Call the actual module; never temporarily delete dt_proj.bias (the
        # checkpoint did that, which races with concurrent forwards).
        delta = self.dt_proj[0](dt)
        ssm, state = selective_scan_reference(
            x, delta, -self.A_log[0].float().exp(), B, C, self.D[0], gate,
            past_key_value.ssm_states[self.layer_idx].to(x.device) if use_cache else None,
        )
        if use_cache:
            past_key_value.conv_states[self.layer_idx] = conv_state.detach()
            past_key_value.ssm_states[self.layer_idx] = state.to(x.dtype).detach()
        mixed = (self.pre_avg_layernorm1(attention) + self.pre_avg_layernorm2(ssm)) / 2
        return self.out_proj(mixed), shared, past_key_value


class HymbaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act_fn_name = config.mlp_hidden_act
        self.act_fn = ACT2FN[self.act_fn_name]
        if self.act_fn_name == "silu":
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        if self.act_fn_name == "silu":
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act_fn(self.up_proj(x)))


class HymbaSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok if self.num_experts > 1 else 1
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False) if self.num_experts > 1 else None
        self.experts = nn.ModuleList([HymbaMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states):
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        if self.router is None:
            return self.experts[0](hidden_states), flat.new_ones((flat.shape[0], 1))
        logits = self.router(flat)
        weights, selected = F.softmax(logits, dim=-1, dtype=torch.float32).topk(self.top_k, dim=-1)
        # NVIDIA does not renormalize top-k probabilities.
        weights = weights.to(flat.dtype)
        output = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            tokens, slots = torch.where(selected == i)
            output.index_add_(0, tokens, expert(flat[tokens]) * weights[tokens, slots, None])
        return output.view(shape), logits


class HymbaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, reuse_kv=False):
        super().__init__()
        self.config, self.layer_idx, self.reuse_kv = config, layer_idx, reuse_kv
        self.mamba = HymbaBlock(config, layer_idx, reuse_kv)
        self.input_layernorm = HymbaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.intermediate_size = config.intermediate_size
        if self.intermediate_size > 0:
            self.pre_moe_layernorm = HymbaRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.moe = HymbaSparseMoeBlock(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                use_cache=False, output_attentions=False, output_router_logits=False,
                kv_last_layer=None, use_swa=False, memory_mask=None, **kwargs):
        output, shared, cache = self.mamba(
            self.input_layernorm(hidden_states), past_key_value=past_key_value,
            attention_mask=attention_mask, position_ids=position_ids, use_cache=use_cache,
            kv_last_layer=kv_last_layer, use_swa=use_swa, memory_mask=memory_mask,
        )
        hidden_states = hidden_states + output
        router = None
        if self.intermediate_size > 0:
            output, router = self.moe(self.pre_moe_layernorm(hidden_states))
            hidden_states = hidden_states + output
        result = (hidden_states,)
        if output_attentions:
            raise NotImplementedError("Hymba SDPA does not return attention probabilities")
        if use_cache:
            result += (cache,)
        if output_router_logits:
            result += (router,)
        return result + (shared,)


class HymbaPreTrainedModel(PreTrainedModel):
    config_class = HymbaConfig
    base_model_prefix = "model"
    _no_split_modules = ["HymbaDecoderLayer"]
    _supports_sdpa = True
    _is_stateful = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, HymbaRMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, HymbaBlock):
            with torch.no_grad():
                module.A_log[0].copy_(torch.arange(1, module.ssm_state_size + 1,
                                                  device=module.A_log[0].device).float().log()[None])
                module.D[0].fill_(1)
        elif isinstance(module, HymbaModel) and hasattr(module, "memory_tokens"):
            nn.init.normal_(module.memory_tokens)


class HymbaModel(HymbaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([HymbaDecoderLayer(config, i, source != i)
                                     for i, source in enumerate(config.kv_sources)])
        self.final_layernorm = HymbaRMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.num_memory_tokens:
            self.memory_tokens = nn.Parameter(torch.randn(config.num_memory_tokens, config.hidden_size))
        self.post_init()

    def _expand_memory(self, embeds, mask, user_offset):
        """Insert memory at true segment boundaries, including chunked prefill.

        Return a gather map for real tokens; memory positions never become
        language-model targets. No trailing synthetic padding enters the SSM.
        """
        pieces, masks, memories, indices = [], [], [], []
        physical = 0
        every, count = self.config.memory_tokens_interspersed_every, self.config.num_memory_tokens
        for i in range(embeds.shape[1]):
            at_boundary = (user_offset + i) % every == 0 if every else user_offset + i == 0
            if count and at_boundary:
                pieces.append(self.memory_tokens[None].expand(embeds.shape[0], -1, -1))
                masks.append(mask.new_ones((embeds.shape[0], count)))
                # Preserve NVIDIA Flex's strict lower boundary for interspersed
                # memory registers (the first is still visible in the window).
                memories.extend([not every or j > 0 for j in range(count)])
                physical += count
            pieces.append(embeds[:, i:i + 1])
            masks.append(mask[:, i:i + 1])
            memories.append(False)
            indices.append(physical)
            physical += 1
        return (torch.cat(pieces, dim=1), torch.cat(masks, dim=1),
                torch.tensor(memories, device=embeds.device)[None].expand(embeds.shape[0], -1),
                torch.tensor(indices, device=embeds.device)[None].expand(embeds.shape[0], -1))

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                output_router_logits=None, return_dict=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits
        if output_attentions:
            raise NotImplementedError("Hymba SDPA does not return attention probabilities")
        if past_key_values is not None and not use_cache:
            raise ValueError("Hymba past_key_values requires use_cache=True")
        embeds = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if not embeds.shape[1]:
            raise ValueError("Hymba requires at least one input token")
        if use_cache and past_key_values is None:
            past_key_values = HybridMambaAttentionDynamicCache(self.config, embeds.shape[0], embeds.dtype, embeds.device)
        if past_key_values is not None and not isinstance(past_key_values, HybridMambaAttentionDynamicCache):
            raise TypeError("Hymba requires HybridMambaAttentionDynamicCache, not a KV-only cache")
        context = past_key_values.transaction() if use_cache else nullcontext()
        with context:
            return self._forward(embeds, attention_mask, position_ids, past_key_values, use_cache,
                                 output_hidden_states, output_router_logits, return_dict)

    def _forward(self, embeds, attention_mask, position_ids, cache, use_cache,
                 output_hidden_states, output_router_logits, return_dict):
        batch, user_length, _ = embeds.shape
        user_offset = cache.user_seq_length if use_cache else 0
        past_length = cache.get_seq_length() if use_cache else 0
        if use_cache:
            if cache.signature != cache._signature(self.config) or cache.batch_size != batch:
                raise ValueError("Hymba cache configuration or batch size mismatch")
            if cache._owner not in (None, id(self)):
                raise ValueError("Hymba cache belongs to a different model")
            cache._owner = id(self)
        if attention_mask is None:
            current_mask = torch.ones((batch, user_length), device=embeds.device, dtype=torch.bool)
        else:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != batch:
                raise ValueError("Hymba attention_mask must be a batch x user-sequence padding mask")
            if attention_mask.shape[1] not in (user_length, user_offset + user_length):
                raise ValueError("Hymba attention_mask must cover current or all user tokens")
            current_mask = attention_mask[:, -user_length:].to(embeds.device).bool()
        expanded, current_mask, memory_mask, gather = self._expand_memory(embeds, current_mask, user_offset)
        length = expanded.shape[1]
        if position_ids is None:
            positions = torch.arange(past_length, past_length + length, device=embeds.device)[None].expand(batch, -1)
        else:
            if position_ids.shape[-1] != length:
                raise ValueError("Explicit Hymba position_ids must include newly inserted memory-token positions")
            positions = position_ids.to(embeds.device).expand(batch, -1)
        if self.config.num_memory_tokens and not past_length:
            # Match NVIDIA's stable padding-before-memory ordering, while
            # retaining the inverse map to return the original token positions.
            order = current_mask.long().argsort(dim=-1, stable=True)
            inverse = order.argsort(dim=-1)
            expanded = expanded.gather(1, order[..., None].expand(-1, -1, expanded.shape[-1]))
            current_mask = current_mask.gather(1, order)
            positions = positions.gather(1, order)
            memory_mask = memory_mask.gather(1, order)
            gather = inverse.gather(1, gather)
        if use_cache and past_length:
            mask = torch.cat((cache.padding_mask.to(embeds.device), current_mask), dim=-1)
            memory_mask = torch.cat((cache.memory_mask.to(embeds.device), memory_mask), dim=-1)
        else:
            mask = current_mask
        hidden = expanded
        all_hidden = () if output_hidden_states else None
        all_router = () if output_router_logits else None
        shared_kv = {}
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden += (hidden,)
            source = self.config.kv_sources[i]
            result = layer(hidden, attention_mask=mask, position_ids=positions, past_key_value=cache,
                           use_cache=use_cache, output_router_logits=output_router_logits,
                           kv_last_layer=shared_kv.get(source), memory_mask=memory_mask,
                           use_swa=self.config.sliding_window is not None and i not in self.config.global_attn_idx)
            hidden = result[0]
            shared_kv[i] = result[-1]
            if output_router_logits and result[-2] is not None:
                all_router += (result[-2],)
        hidden = self.final_layernorm(hidden)
        if output_hidden_states:
            all_hidden += (hidden,)
        hidden = hidden.gather(1, gather.to(hidden.device)[..., None].expand(-1, -1, hidden.shape[-1]))
        if use_cache:
            cache._length += length
            cache.user_seq_length += user_length
            cache.padding_mask = mask.detach()
            cache.memory_mask = memory_mask.detach()
            cache.has_previous_state = True
        result = MoeModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache if use_cache else None,
                                       hidden_states=all_hidden, router_logits=all_router)
        return result if return_dict is not False else result.to_tuple()


class HymbaForCausalLM(HymbaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = HymbaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, output_router_logits=None, return_dict=None,
                calc_logits_for_entire_prompt=True, logits_to_keep=0, **kwargs):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            source = inputs_embeds if inputs_embeds is not None else input_ids
            if source is not None:
                past_key_values = HybridMambaAttentionDynamicCache(self.config, source.shape[0], self.dtype, source.device)
        context = past_key_values.transaction() if isinstance(past_key_values, HybridMambaAttentionDynamicCache) else nullcontext()
        with context:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                 past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                                 output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                 output_router_logits=output_router_logits, return_dict=True)
            hidden = outputs.last_hidden_state
            if labels is None:
                if not calc_logits_for_entire_prompt:
                    hidden = hidden[:, -1:]
                elif isinstance(logits_to_keep, int):
                    hidden = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
                else:
                    hidden = hidden[:, logits_to_keep]
            logits = self.lm_head(hidden).float()
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size),
                                       labels[:, 1:].to(logits.device).reshape(-1))
            aux_loss = None
            if outputs.router_logits:
                if self.config.num_experts == 1:
                    aux_loss = logits.new_zeros(())
                else:
                    # Same unnormalized top-k load-balancing objective as the
                    # original, with memory rows already present in router data.
                    probs = F.softmax(torch.cat(outputs.router_logits).float(), dim=-1)
                    indices = probs.topk(self.config.num_experts_per_tok, dim=-1).indices
                    expert_mask = F.one_hot(indices, self.config.num_experts).float()
                    aux_loss = (expert_mask.mean(0) * probs.mean(0)[None]).sum() * self.config.num_experts
                if loss is not None:
                    loss = loss + self.config.router_aux_loss_coef * aux_loss
            result = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                              past_key_values=outputs.past_key_values,
                                              hidden_states=outputs.hidden_states, router_logits=outputs.router_logits)
            return result if return_dict is not False else result.to_tuple()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      inputs_embeds=None, **kwargs):
        past_users = past_key_values.user_seq_length if past_key_values is not None else 0
        if past_users:
            input_ids = input_ids[:, past_users:] if input_ids.shape[1] > past_users else input_ids[:, -1:]
            inputs_embeds = None
        result = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None else {"input_ids": input_ids}
        result.update(past_key_values=past_key_values, attention_mask=attention_mask,
                      use_cache=kwargs.get("use_cache", True),
                      calc_logits_for_entire_prompt=self.config.calc_logits_for_entire_prompt)
        for name in ("output_hidden_states", "output_router_logits", "position_ids"):
            if name in kwargs:
                # GenerationMixin supplies user-token positions; the decoder
                # inserts its own physical memory positions. With no memory,
                # retain explicitly supplied user positions and slice the tail.
                if name == "position_ids":
                    if not self.config.num_memory_tokens and kwargs[name] is not None:
                        result[name] = kwargs[name][:, -input_ids.shape[1]:]
                else:
                    result[name] = kwargs[name]
        return result
