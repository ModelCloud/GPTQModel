# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Package-owned SenseNova-Loop: native Qwen3.5 layers, a fixed Loop plan and mHC routers.

Adapted from `modeling_sensenova_loop.py` of sensenova/SenseNova-Loop-9B at
fa381f7c466739475e172a7d3234e77351e4f613 (Apache-2.0, see NOTICE). Every
physical layer, the vision tower, embeddings, norm, rotary embedding and head
are the exact Transformers 5 Qwen3.5 classes. This file adds only:

* the execution plan: physical layers ``start..end-1`` run ``eval_count``
  times with the *same* weights, each visit in its own cache slot;
* the mHC residual-stream routers (FP32) and the decoder layer that uses them.

Upstream rewrote native instances with ``__class__`` assignment; here the
layers are constructed directly so Transformers 5 initialization, meta loading
and ``_keep_in_fp32_modules`` see the real module types. State-dict names are
unchanged.
"""

import copy
import math

import torch
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import CacheLayerMixin, DynamicCache, LinearAttentionCacheLayerMixin
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
)
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_sensenova_loop import SenseNovaLoopConfig, execution_plan


_KEEP_FP32 = ["mhc_attn", "mhc_ffn"]
_NO_SPLIT = ["Qwen3_5DecoderLayer", "SenseNovaLoopMHCDecoderLayer", "Qwen3_5VisionBlock"]
# The released index also lists `model.language_model.loop_gate.logits`, an
# adaptive-count gate that the fixed-count upstream model never reads.
_IGNORE_UNEXPECTED = [r"^mtp.*", r"(^|\.)language_model\.loop_gate\."]


def sinkhorn(logits, iterations):
    """Normalize routing logits across rows and columns in log space."""

    logits = logits.float()
    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()


class MHCRouter(nn.Module):
    """FP32 residual-stream router: ``pre`` read weights, ``post`` write weights and a stream mix."""

    _INITIAL_PRE = {2: (0.25, 0.75), 4: (0.0625, 0.1875, 0.25, 0.5)}

    def __init__(self, dim, settings):
        super().__init__()
        self.streams = settings["streams"]
        self.iterations = settings["iterations"]
        self.fixed_identity = settings["fixed_identity"]
        self.pre_scale = settings.get("pre_scale", 1.0)
        output_size = 2 * self.streams + (0 if self.fixed_identity else self.streams**2)
        self.weight = nn.Parameter(torch.empty(output_size, self.streams * dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.float32))

    def initial_bias(self):
        """Upstream initialization: logit pre-weights, zero post logits and a dominant mix diagonal."""

        streams = self.streams
        bias = torch.zeros(self.bias.shape, dtype=torch.float32)
        bias[:streams] = torch.logit(torch.tensor(self._INITIAL_PRE[streams], dtype=torch.float32))
        if not self.fixed_identity:
            diagonal = math.log((1 - (streams - 1) / 16) * 16)
            bias[2 * streams:] = (torch.eye(streams) * diagonal).flatten()
        return bias

    def _apply(self, fn, recurse=True):
        # Upstream contract: model-wide dtype casts never demote the routers.
        def keep_fp32(tensor):
            result = fn(tensor)
            if tensor.is_floating_point() and result.dtype != torch.float32:
                if tensor.device.type == "meta" and result.device.type != "meta":
                    return torch.empty_like(result, dtype=torch.float32)
                return tensor.to(device=result.device, dtype=torch.float32)
            return result

        return super()._apply(keep_fp32, recurse=recurse)

    def forward(self, residual):
        if self.weight.dtype != torch.float32 or self.bias.dtype != torch.float32:
            raise RuntimeError("mHC routers require FP32 parameters")
        with torch.autocast(device_type=residual.device.type, enabled=False):
            flat = residual.flatten(-2).float()
            normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
            logits = nn.functional.linear(normalized, self.weight, self.bias)
            streams = self.streams
            pre = logits[..., :streams].sigmoid()
            post = 2 * logits[..., streams:2 * streams].sigmoid()
            if self.fixed_identity:
                mix = torch.eye(streams, device=flat.device, dtype=torch.float32).expand(
                    *flat.shape[:-1], streams, streams
                )
            else:
                mix = sinkhorn(logits[..., 2 * streams:].unflatten(-1, (streams, streams)), self.iterations)
            branch = torch.matmul(pre.to(residual.dtype).unsqueeze(-2), residual).squeeze(-2)
            if self.pre_scale != 1.0:
                branch = branch * self.pre_scale
        return branch, post, mix

    @staticmethod
    def merge(residual, branch, post, mix):
        """Mix residual streams and add the branch output to each stream."""

        with torch.autocast(device_type=residual.device.type, enabled=False):
            mixed_residual = torch.matmul(mix.to(residual.dtype), residual)
            return mixed_residual + post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)


class SenseNovaLoopMHCDecoderLayer(Qwen3_5DecoderLayer):
    """Native Qwen3.5 layer whose residual is ``streams`` parallel copies routed by two mHC routers.

    Between the first and last mHC layer the hidden state carries the streams
    flattened as ``[..., streams * hidden]``.
    """

    def __init__(self, config, layer_idx, mhc):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.mhc_start = mhc["start"]
        self.mhc_end = mhc["end"]
        self.mhc_streams = mhc["streams"]
        self.mhc_attn = MHCRouter(config.hidden_size, mhc)
        self.mhc_ffn = MHCRouter(config.hidden_size, mhc)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                past_key_values=None, **kwargs):
        if self.layer_idx == self.mhc_start:
            residual = hidden_states.unsqueeze(-2).expand(
                *hidden_states.shape[:-1], self.mhc_streams, hidden_states.shape[-1]
            ).contiguous()
        else:
            residual = hidden_states.unflatten(-1, (self.mhc_streams, -1))

        branch, post, mix = self.mhc_attn(residual)
        branch = self.input_layernorm(branch)
        if self.block_type == "linear_attention":
            branch = self.linear_attn(
                hidden_states=branch, cache_params=past_key_values, attention_mask=attention_mask, **kwargs
            )
        else:
            branch, _ = self.self_attn(
                hidden_states=branch,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        residual = MHCRouter.merge(residual, branch, post, mix)

        branch, post, mix = self.mhc_ffn(residual)
        branch = self.mlp(self.post_attention_layernorm(branch))
        residual = MHCRouter.merge(residual, branch, post, mix)
        return residual.mean(-2) if self.layer_idx == self.mhc_end - 1 else residual.flatten(-2)


class _VisitCache:
    """Route a physical layer's cache calls to its execution-slot index.

    Shared modules and their ``layer_idx`` are never mutated; the mapping only
    exists for one layer call.
    """

    __slots__ = ("cache", "physical", "visit")

    def __init__(self, cache, physical, visit):
        self.cache = cache
        self.physical = physical
        self.visit = visit

    def _slot(self, layer_idx):
        if layer_idx != self.physical:
            raise IndexError("A physical layer may only access its own visit cache slot")
        return self.visit

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return self.cache.update(key_states, value_states, self._slot(layer_idx), *args, **kwargs)

    def get_conv_state(self, layer_idx, *args, **kwargs):
        return self.cache.get_conv_state(self._slot(layer_idx), *args, **kwargs)

    def update_conv_state(self, states, layer_idx, *args, **kwargs):
        return self.cache.update_conv_state(states, self._slot(layer_idx), *args, **kwargs)

    def get_recurrent_state(self, layer_idx, *args, **kwargs):
        return self.cache.get_recurrent_state(self._slot(layer_idx), *args, **kwargs)

    def update_recurrent_state(self, states, layer_idx, *args, **kwargs):
        return self.cache.update_recurrent_state(states, self._slot(layer_idx), *args, **kwargs)

    def has_previous_state(self, layer_idx, *args, **kwargs):
        return self.cache.has_previous_state(self._slot(layer_idx), *args, **kwargs)

    def is_recording_past(self, layer_idx):
        return self.cache.is_recording_past(self._slot(layer_idx))


def _cache_is_empty(cache):
    for layer in cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin) and any(layer.has_previous_state.values()):
            return False
        if isinstance(layer, CacheLayerMixin) and layer.get_seq_length() > 0:
            return False
    return True


def _init_sensenova_weights(model, module):
    Qwen3_5PreTrainedModel._init_weights(model, module)
    if isinstance(module, MHCRouter):
        init.zeros_(module.weight)
        init.copy_(module.bias, module.initial_bias())


class SenseNovaLoopTextModel(Qwen3_5TextModel):
    """Qwen3.5 text decoder that executes ``loop_plan`` with one cache slot per visit."""

    _keep_in_fp32_modules = _KEEP_FP32
    _keep_in_fp32_modules_strict = _KEEP_FP32
    _no_split_modules = _NO_SPLIT

    def __init__(self, config, loop_plan, mhc=None):
        Qwen3_5PreTrainedModel.__init__(self, config)
        mhc_range = range(mhc["start"], mhc["end"]) if mhc is not None else range(0)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [
                SenseNovaLoopMHCDecoderLayer(config, index, mhc) if index in mhc_range
                else Qwen3_5DecoderLayer(config, index)
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.loop_plan = tuple(loop_plan)
        # Cache layout in execution order; the physical layer config is untouched.
        cache_config = copy.deepcopy(config)
        cache_config.num_hidden_layers = len(self.loop_plan)
        cache_config.layer_types = [config.layer_types[index] for index in self.loop_plan]
        self._loop_cache_config = cache_config
        self.post_init()

    _init_weights = torch.no_grad()(_init_sensenova_weights)

    def new_cache(self):
        cache = DynamicCache(config=self._loop_cache_config)
        cache._sensenova_loop_plan = self.loop_plan
        return cache

    def _visit_cache(self, cache):
        if cache is None:
            return self.new_cache()
        if not isinstance(cache, DynamicCache):
            raise TypeError("SenseNova-Loop generation supports DynamicCache only")
        if getattr(cache, "_sensenova_loop_plan", None) == self.loop_plan:
            return cache
        # `generate` allocates an empty physical-depth cache; a populated cache
        # from another layout cannot be reinterpreted.
        if not _cache_is_empty(cache):
            raise ValueError("A populated cache from another execution plan cannot be reused")
        return self.new_cache()

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache or past_key_values is not None:
            past_key_values = self._visit_cache(past_key_values)

        # Text, temporal, height and width rows, as in native Qwen3.5.
        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + seen
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids, position_ids = position_ids[0], position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(masks := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            masks = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for visit, physical in enumerate(self.loop_plan):
            hidden_states = self.layers[physical](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=masks[self.config.layer_types[physical]],
                position_ids=text_position_ids,
                past_key_values=None if past_key_values is None else _VisitCache(past_key_values, physical, visit),
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class SenseNovaLoopModel(Qwen3_5Model):
    config_class = SenseNovaLoopConfig
    _keep_in_fp32_modules = _KEEP_FP32
    _keep_in_fp32_modules_strict = _KEEP_FP32
    _no_split_modules = _NO_SPLIT
    _keys_to_ignore_on_load_unexpected = _IGNORE_UNEXPECTED

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        plan = execution_plan(config)
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = SenseNovaLoopTextModel._from_config(config.text_config, loop_plan=plan, mhc=config.mhc)
        self.rope_deltas = None
        self.post_init()

    _init_weights = torch.no_grad()(_init_sensenova_weights)


class SenseNovaLoopForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    config_class = SenseNovaLoopConfig
    _keep_in_fp32_modules = _KEEP_FP32
    _keep_in_fp32_modules_strict = _KEEP_FP32
    _no_split_modules = _NO_SPLIT
    _keys_to_ignore_on_load_unexpected = _IGNORE_UNEXPECTED

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = SenseNovaLoopModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    _init_weights = torch.no_grad()(_init_sensenova_weights)


__all__ = [
    "MHCRouter",
    "SenseNovaLoopForConditionalGeneration",
    "SenseNovaLoopMHCDecoderLayer",
    "SenseNovaLoopModel",
    "SenseNovaLoopTextModel",
    "sinkhorn",
]
