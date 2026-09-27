# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Scale-aware W4A activation stream across Llama decoder operators.

The existing Hugging Face parameters and module tree are retained. Only
inference forward methods for fully selected decoder layers are replaced.
"""

from __future__ import annotations

from types import MethodType

import torch

from .w4a_activation import W4AActivation, pack_activation
from .w4a_floatx import W4AFP8Linear


def _as_stream(x: torch.Tensor | W4AActivation, mode: str, recipe: str | None) -> W4AActivation:
    if isinstance(x, W4AActivation):
        if x.mode != mode or x.recipe != recipe:
            raise ValueError(
                f"W4A stream policy mismatch: expected mode={mode}, recipe={recipe}; "
                f"got mode={x.mode}, recipe={x.recipe}."
            )
        if x.rotation_applied:
            raise ValueError("A rotated W4A operand cannot cross a decoder-layer boundary.")
        return x
    return pack_activation(x, mode, recipe=recipe)


def _decode(value: torch.Tensor | W4AActivation,
            dtype: torch.dtype | None = None) -> torch.Tensor:
    return value.decode(dtype) if isinstance(value, W4AActivation) else value


def _add_stream(left: W4AActivation,
                right: torch.Tensor | W4AActivation) -> W4AActivation:
    if isinstance(right, W4AActivation):
        if (left.mode != right.mode or left.shape != right.shape
                or left.model_dtype != right.model_dtype):
            raise ValueError("W4A residual operands must have matching mode, shape, and model dtype.")
        if left.recipe != right.recipe:
            raise ValueError("W4A residual operands must use the same activation recipe.")
        if right.rotation_applied:
            raise ValueError("A rotated W4A operand may only be consumed by its Linear.")
    elif tuple(right.shape) != left.shape:
        raise ValueError("A dense W4A residual branch must match the encoded residual shape.")
    if left.rotation_applied:
        raise ValueError("A rotated W4A operand may only be consumed by its Linear.")
    summed = left.decode(torch.float32) + _decode(right, torch.float32).float()
    return pack_activation(summed, left.mode, model_dtype=left.model_dtype, recipe=left.recipe)


def _norm_forward(self, hidden_states: W4AActivation) -> W4AActivation:
    if not isinstance(hidden_states, W4AActivation):
        raise TypeError("W4A RMSNorm requires an encoded activation and its scales.")
    if hidden_states.rotation_applied:
        raise ValueError("RMSNorm cannot consume a projection-rotated W4A operand.")
    x = hidden_states.decode(torch.float32)
    variance = x.square().mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(variance + self.variance_epsilon)
    y = y * self.weight.float()
    return pack_activation(
        y, hidden_states.mode,
        model_dtype=hidden_states.model_dtype, recipe=hidden_states.recipe
    )


def _final_norm_forward(self, hidden_states: torch.Tensor | W4AActivation) -> torch.Tensor:
    """Consume the encoded final decoder output at the unquantized head edge."""
    x = hidden_states.decode() if isinstance(hidden_states, W4AActivation) else hidden_states
    y = x.float()
    variance = y.square().mean(dim=-1, keepdim=True)
    y = y * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * y.to(x.dtype)


def _attention_forward(self, hidden_states: W4AActivation, position_embeddings=None,
                       attention_mask=None, past_key_values=None, **kwargs):
    from transformers.models.llama.modeling_llama import (
        ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward,
    )

    if not isinstance(hidden_states, W4AActivation):
        raise TypeError("W4A attention requires an encoded activation and its scales.")
    if hidden_states.rotation_applied:
        raise ValueError("Attention cannot consume a projection-rotated W4A operand.")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = _decode(self.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
    key_states = _decode(self.k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
    value_states = _decode(self.v_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    encoded = pack_activation(
        attn_output, hidden_states.mode, model_dtype=hidden_states.model_dtype,
        recipe=hidden_states.recipe,
    )
    return self.o_proj(encoded), attn_weights


def _mlp_forward(self, x: W4AActivation) -> W4AActivation:
    if not isinstance(x, W4AActivation):
        raise TypeError("W4A MLP requires an encoded activation and its scales.")
    gate = _decode(self.gate_proj(x))
    up = _decode(self.up_proj(x))
    product = self.act_fn(gate) * up
    down = self.down_proj
    rotated = bool(down.online_full_had or down.online_partial_had)
    if rotated:
        product = down._apply_rotation_to_input(product)
    encoded = pack_activation(
        product, x.mode, model_dtype=x.model_dtype, recipe=x.recipe,
        rotation_applied=rotated,
    )
    return self.down_proj(encoded)


def _layer_forward(self, hidden_states: torch.Tensor | W4AActivation,
                   attention_mask=None, position_ids=None, past_key_values=None,
                   use_cache=False, position_embeddings=None, **kwargs) -> W4AActivation:
    if self._w4a_stream_require_input and not isinstance(hidden_states, W4AActivation):
        raise TypeError("A W4A decoder layer after another W4A layer must receive encoded activations and scales.")
    x = _as_stream(hidden_states, self._w4a_stream_mode, self._w4a_stream_recipe)
    residual = x
    x = self.input_layernorm(x)
    x, _ = self.self_attn(
        hidden_states=x, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, use_cache=use_cache,
        position_embeddings=position_embeddings, **kwargs,
    )
    x = _add_stream(residual, x)
    residual = x
    x = self.post_attention_layernorm(x)
    x = self.mlp(x)
    return _add_stream(residual, x)


def _decode_input(_module, args):
    if args and isinstance(args[0], W4AActivation):
        return (args[0].decode(), *args[1:])
    return None


def _bind_forward(module, function) -> None:
    bound = MethodType(function, module)
    # Accelerate's placement hook wraps `forward` and invokes `_old_forward`.
    # Replacing only `forward` would be bypassed by that wrapper.
    if hasattr(module, "_hf_hook") and hasattr(module, "_old_forward"):
        module._old_forward = bound
    else:
        module.forward = bound


def install_w4a_llama_stream(model: torch.nn.Module, mode: str,
                             recipe: str | None = None, version: int = 3) -> None:
    """Install an encoded stream on complete W4A Llama decoder layers.

    A partial decoder selection is rejected so its remaining projections
    cannot silently convert the stream back to ordinary BF16 tensors.
    """
    if mode not in {"w4afp8"}:
        raise ValueError(f"Unsupported W4A activation stream mode: {mode}.")
    if version not in {2, 3}:
        raise ValueError(f"Unsupported W4A activation stream version: {version}.")
    if mode == "w4afp8" and recipe is not None:
        raise ValueError("W4AFP8 does not use an scale recipe.")
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is None or not layers or layers[0].__class__.__name__ != "LlamaDecoderLayer":
        raise ValueError("The W4A activation stream currently requires a Llama decoder model.")
    if getattr(model, "_w4a_stream_mode", None) == mode:
        if (getattr(model, "_w4a_stream_recipe", None) != recipe
                or getattr(model, "_w4a_stream_version", None) != version):
            raise ValueError("A different W4A stream policy is already installed.")
        return
    if getattr(model, "_w4a_stream_mode", None) is not None:
        raise ValueError("A different W4A activation stream is already installed.")

    required = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    selected = []
    for layer in layers:
        modules = []
        for path in required:
            parent, child = path.split(".")
            modules.append(getattr(getattr(layer, parent), child))
        present = [isinstance(m, W4AFP8Linear) for m in modules]
        if any(present) and not all(present):
            raise ValueError("A W4A activation stream requires all seven projections in a selected Llama layer.")
        if all(present) and any(module.out_features % 128 for module in modules):
            raise ValueError("W4A activation transport requires every selected projection output width divisible by 128.")
        selected.append(all(present))
    if not any(selected):
        raise ValueError("No complete W4A decoder layer was found.")

    for index, (layer, enabled) in enumerate(zip(layers, selected)):
        if not enabled:
            continue
        for path in required:
            parent, child = path.split(".")
            linear = getattr(getattr(layer, parent), child)
            linear._require_activation_stream = True
            linear._w4a_activation_recipe = recipe
            # Version 2 encoded every Linear output and often decoded it at
            # the very next BF16-only operator. Version 3 emits model dtype
            # directly for those branches and packs once at the next actual
            # FP8 consumer or decoder-layer boundary.
            linear._w4a_output_encoded = version == 2
        layer._w4a_stream_mode = mode
        layer._w4a_stream_recipe = recipe
        layer._w4a_stream_version = version
        layer._w4a_stream_require_input = index > 0 and selected[index - 1]
        _bind_forward(layer, _layer_forward)
        _bind_forward(layer.self_attn, _attention_forward)
        _bind_forward(layer.mlp, _mlp_forward)
        _bind_forward(layer.input_layernorm, _norm_forward)
        _bind_forward(layer.post_attention_layernorm, _norm_forward)

    handles = []
    for layer, enabled in zip(layers, selected):
        if not enabled:
            handles.append(layer.register_forward_pre_hook(_decode_input))
    if selected[-1]:
        _bind_forward(decoder.norm, _final_norm_forward)
    else:
        handles.append(decoder.norm.register_forward_pre_hook(_decode_input))
    model._w4a_stream_handles = handles
    model._w4a_stream_mode = mode
    model._w4a_stream_recipe = recipe
    model._w4a_stream_version = version
