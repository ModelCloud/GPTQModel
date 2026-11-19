# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/mobilellm/SPEC.md; see NOTICE.
"""MobileLLM decoder on the Transformers 5.23 ABI.

MobileLLM is a pre-norm Llama decoder. The standard pieces (RMSNorm, SwiGLU MLP, GQA attention, half-split
rotary application) come from the vendored Apache-2.0 Llama port. MobileLLM adds:

* embedding sharing: without ``lm_head`` the logits multiply with the input embedding matrix;
* immediate block-wise layer sharing: each physical layer runs twice in a row, each use with its own cache slot;
* ``pretraining_tp`` weight-sliced projections that ignore biases;
* a per-attention rotary module whose dynamic-NTK base is stateful.
"""

from collections.abc import Callable
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ..native.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_mobilellm import MobileLLMConfig


def _column_sliced(x: torch.Tensor, weight: torch.Tensor, parts: int) -> torch.Tensor:
    """Output-dim weight slices, concatenated; no bias."""
    return torch.cat([F.linear(x, chunk) for chunk in weight.chunk(parts, dim=0)], dim=-1)


def _row_sliced(x: torch.Tensor, weight: torch.Tensor, parts: int) -> torch.Tensor:
    """Input-dim slices of activations and weight, partial products summed; no bias."""
    pieces = zip(x.chunk(parts, dim=-1), weight.chunk(parts, dim=1))
    total = None
    for activation, chunk in pieces:
        product = F.linear(activation, chunk)
        total = product if total is None else total + product
    return total


def _uses_per_layer(config) -> int:
    return 2 if config.layer_sharing else 1


class MobileLLMRMSNorm(LlamaRMSNorm):
    pass


class MobileLLMRotaryEmbedding(nn.Module):
    """Rotary frequencies of one attention module.

    ``_base`` is the RoPE base currently in effect. Dynamic NTK replaces it (and ``inv_freq``) whenever a call's
    positions exceed ``max_position_embeddings`` and keeps it for later shorter calls.
    """

    def __init__(self, config: MobileLLMConfig):
        super().__init__()
        parameters = config.rope_parameters
        self.rope_type = parameters["rope_type"]
        self.factor = float(parameters.get("factor", 1.0))
        self.rope_theta = float(parameters["rope_theta"])
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self._base = self.rope_theta
        self.inv_freq = nn.Buffer(self.frequencies(self._base), persistent=False)

    def frequencies(self, base, device=None) -> torch.Tensor:
        exponents = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim
        return 1.0 / (base ** exponents)

    def default_frequencies(self, device=None) -> torch.Tensor:
        return self.frequencies(self.rope_theta, device=device)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        # Moves recompute fp32 frequencies on the new device from the base in effect.
        self.inv_freq = self.frequencies(self._base, device=self.inv_freq.device)
        return result

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            if self.rope_type == "dynamic":
                length = position_ids.max() + 1
                if length > self.max_position_embeddings:
                    scale = self.factor * length.float() / self.max_position_embeddings - self.factor + 1
                    base = self.rope_theta * scale ** (self.dim / (self.dim - 2))
                    self.inv_freq = self.frequencies(base.to(self.inv_freq.device), device=self.inv_freq.device)
                    self._base = float(base)
            positions = position_ids.float()
            if self.rope_type == "linear":
                positions = positions / self.factor
            angles = positions[..., None] * self.inv_freq.to(device=x.device, dtype=torch.float32)
            emb = torch.cat((angles, angles), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MobileLLMMLP(LlamaMLP):
    def forward(self, x):
        parts = self.config.pretraining_tp
        if parts <= 1:
            return super().forward(x)
        gate = _column_sliced(x, self.gate_proj.weight, parts)
        up = _column_sliced(x, self.up_proj.weight, parts)
        return _row_sliced(self.act_fn(gate) * up, self.down_proj.weight, parts)


class MobileLLMAttention(LlamaAttention):
    def __init__(self, config: MobileLLMConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.rotary_emb = MobileLLMRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_slot: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        parts = self.config.pretraining_tp
        sliced = parts > 1 and self.config._attn_implementation != "sdpa"

        def project(module):
            return _column_sliced(hidden_states, module.weight, parts) if sliced else module(hidden_states)

        query_states = project(self.q_proj).view(hidden_shape).transpose(1, 2)
        key_states = project(self.k_proj).view(hidden_shape).transpose(1, 2)
        value_states = project(self.v_proj).view(hidden_shape).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            slot = self.layer_idx if cache_slot is None else cache_slot
            key_states, value_states = past_key_values.update(key_states, value_states, slot)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if sliced:
            attn_output = _row_sliced(attn_output, self.o_proj.weight, parts)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MobileLLMDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MobileLLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MobileLLMAttention(config=config, layer_idx=layer_idx)
        self.mlp = MobileLLMMLP(config)
        self.input_layernorm = MobileLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MobileLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_slot: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_slot=cache_slot,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@contextmanager
def _restore_state_on_error(layers, cache):
    """Undo cache growth and rotary updates of a failed forward; tensors are kept by reference."""
    cache_layers = getattr(cache, "layers", None) if cache is not None else None
    cache_snapshot = None
    if isinstance(cache_layers, list):
        cache_snapshot = (cache_layers, list(cache_layers), [vars(layer).copy() for layer in cache_layers])
    rotary_snapshot = [
        (layer.self_attn.rotary_emb, layer.self_attn.rotary_emb._base, layer.self_attn.rotary_emb.inv_freq)
        for layer in layers
    ]
    try:
        yield
    except BaseException:
        if cache_snapshot is not None:
            layer_list, members, states = cache_snapshot
            if cache.layers is not layer_list:
                cache.layers = layer_list
            layer_list[:] = members
            for member, state in zip(members, states):
                attributes = vars(member)
                attributes.clear()
                attributes.update(state)
        for rotary, base, frequencies in rotary_snapshot:
            rotary._base = base
            rotary.inv_freq = frequencies
        raise


class MobileLLMPreTrainedModel(PreTrainedModel):
    config_class = MobileLLMConfig
    config: MobileLLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MobileLLMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_attention_backend = True
    _keep_in_fp32_modules_strict = ["inv_freq"]
    _can_record_outputs = {
        "hidden_states": MobileLLMDecoderLayer,
        "attentions": MobileLLMAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MobileLLMRotaryEmbedding):
            init.copy_(module.inv_freq, module.default_frequencies(device=module.inv_freq.device))
            module._base = module.rope_theta


class MobileLLMModel(MobileLLMPreTrainedModel):
    def __init__(self, config: MobileLLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MobileLLMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MobileLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def _validate_cache(self, past_key_values):
        if past_key_values is None:
            return
        if not isinstance(past_key_values, DynamicCache):
            raise ValueError(
                f"MobileLLM supports only DynamicCache past_key_values, got {type(past_key_values).__name__}"
            )
        uses = self.config.num_hidden_layers * _uses_per_layer(self.config)
        layers = past_key_values.layers
        grows_lazily = getattr(past_key_values, "layer_class_to_replicate", None) is not None
        if layers and not grows_lazily and len(layers) != uses:
            raise ValueError(
                f"MobileLLM needs one slot for each decoder use ({uses}), but the cache has {len(layers)} layers; "
                "pass an empty DynamicCache()"
            )

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        self._validate_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, length = inputs_embeds.shape[:2]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if attention_mask is not None and (
            attention_mask.dim() != 2 or tuple(attention_mask.shape) != (batch, past_length + length)
        ):
            raise ValueError(
                f"MobileLLM attention_mask must have shape {(batch, past_length + length)}, "
                f"got {tuple(attention_mask.shape)}"
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        with _restore_state_on_error(self.layers, past_key_values):
            if position_ids is None:
                position_ids = torch.arange(past_length, past_length + length, device=inputs_embeds.device)
                position_ids = position_ids.unsqueeze(0)

            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            hidden_states = inputs_embeds
            repeats = _uses_per_layer(self.config)
            for index, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                for repeat in range(repeats):
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_slot=index * repeats + repeat,
                        **kwargs,
                    )
            hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class MobileLLMForCausalLM(MobileLLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MobileLLMConfig):
        super().__init__(config)
        self.model = MobileLLMModel(config)
        self.vocab_size = config.vocab_size
        if config.share_embedding:
            # Logits use the input embedding matrix; there is no head module to tie.
            self._tied_weights_keys = {}
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # The model creates its own lazily growing cache with one slot per decoder use.
        return False

    def get_output_embeddings(self):
        return self.model.embed_tokens if self.config.share_embedding else self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if self.config.share_embedding:
            self.model.embed_tokens = new_embeddings
        else:
            self.lm_head = new_embeddings

    def _logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        parts = self.config.pretraining_tp
        if parts > 1:
            return _column_sliced(hidden_states, self.get_output_embeddings().weight, parts).float()
        if self.config.share_embedding:
            return F.linear(hidden_states, self.model.embed_tokens.weight).float()
        return self.lm_head(hidden_states).float()

    @can_return_tuple
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
        with _restore_state_on_error(self.model.layers, past_key_values):
            outputs: BaseModelOutputWithPast = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
            logits = self._logits(hidden_states)
            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["MobileLLMForCausalLM", "MobileLLMModel", "MobileLLMPreTrainedModel"]
