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
from .w4a_nvfp4 import W4ANVFP4Linear
from .w4a_boundary import (
    NVFP4BoundaryQuantizer, llama_nvfp4_boundaries, pack_boundary, validate_producer_scales,
)


def _headroom_scale(recipe: str | None, consumer=None, explicit=None):
    if recipe not in {"nvidia_headroom", "least_squares_headroom"}:
        return None
    scale = explicit
    if scale is None and consumer is not None:
        scale = getattr(consumer, "activation_global_scale", None)
    if scale is None:
        raise RuntimeError("An NVFP4 headroom boundary is missing its calibrated global scale.")
    return scale


def _as_stream(x: torch.Tensor | W4AActivation, mode: str, recipe: str | None, owner=None) -> W4AActivation:
    if isinstance(x, W4AActivation):
        if x.mode != mode or x.recipe != recipe:
            raise ValueError(
                f"W4A stream policy mismatch: expected mode={mode}, recipe={recipe}; "
                f"got mode={x.mode}, recipe={x.recipe}."
            )
        if x.rotation_applied:
            raise ValueError("A rotated W4A operand cannot cross a decoder-layer boundary.")
        return x
    return pack_boundary(owner, "input", x, mode, recipe=recipe, reference=x)


def _set_boundary_policy(owner, boundary: str, mode: str, recipe: str | None) -> None:
    """Record the activation transport policy for one producer boundary.

    A mixed-precision stream declares its policy per boundary rather than per
    model, so attention boundaries can carry FP8 while MLP boundaries carry
    NVFP4 over the same native GPTQ INT4 weights.
    """
    setattr(owner, f"_w4a_{boundary}_mode", mode)
    setattr(owner, f"_w4a_{boundary}_recipe", recipe)


def _boundary_policy(owner, boundary: str) -> tuple[str, str | None]:
    mode = getattr(owner, f"_w4a_{boundary}_mode", None)
    if mode is None:
        raise RuntimeError(f"W4A boundary {boundary!r} has no activation policy installed.")
    return mode, getattr(owner, f"_w4a_{boundary}_recipe", None)


def _boundary_group(owner, boundary: str) -> str:
    """Map a producer boundary to the projection group that consumes it.

    In a Llama decoder only the post-attention residual and the MLP product
    feed the MLP; every other boundary feeds attention.
    """
    return "mlp" if boundary in {"attention_residual", "product"} else "attention"


def _decode(value: torch.Tensor | W4AActivation,
            dtype: torch.dtype | None = None) -> torch.Tensor:
    return value.decode(dtype) if isinstance(value, W4AActivation) else value


def _add_stream(left: W4AActivation,
                right: torch.Tensor | W4AActivation, *, owner=None, boundary="output") -> W4AActivation:
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
    # The residual stream is not a GEMM operand. Add the exact compute-dtype
    # values and re-pack only the hardware operand, so the running hidden state
    # never accumulates 4-bit rounding across a decoder stack.
    summed = left.exact(torch.float32) + _decode(right, torch.float32).float()
    mode, recipe = _boundary_policy(owner, boundary)
    return pack_boundary(owner, boundary, summed, mode, model_dtype=left.model_dtype, recipe=recipe,
                         reference=summed)


def _norm_forward(self, hidden_states: W4AActivation) -> W4AActivation:
    if not isinstance(hidden_states, W4AActivation):
        raise TypeError("W4A RMSNorm requires an encoded activation and its scales.")
    if hidden_states.rotation_applied:
        raise ValueError("RMSNorm cannot consume a projection-rotated W4A operand.")
    mode, recipe = _boundary_policy(self, "norm")
    x = hidden_states.exact(torch.float32)
    variance = x.square().mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + self.variance_epsilon)
    # The token-rescale shortcut leaves the FP4 codes and hardware scales
    # untouched, so it is only valid when this norm reproduces the carrier it
    # consumed. Across a mode change the operand must be repacked.
    if (getattr(self, "_w4a_preserve_norm_codes", False) and mode == "w4a_nvfp4"
            and hidden_states.mode == mode and hidden_states.recipe == recipe):
        return hidden_states.rescale_tokens(inv_rms.reshape(-1))
    y = x * inv_rms
    y = y * self.weight.float()
    return pack_activation(
        y, mode,
        global_scale=_headroom_scale(recipe, explicit=getattr(self, "_w4a_output_global_scale", None)),
        model_dtype=hidden_states.model_dtype, recipe=recipe, reference=y,
    )


def _final_norm_forward(self, hidden_states: torch.Tensor | W4AActivation) -> torch.Tensor:
    """Consume the encoded final decoder output at the unquantized head edge."""
    x = hidden_states.exact() if isinstance(hidden_states, W4AActivation) else hidden_states
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
    mode, recipe = _boundary_policy(self, "output")
    encoded = pack_boundary(
        self, "output", attn_output, mode, model_dtype=hidden_states.model_dtype,
        recipe=recipe,
        global_scale=_headroom_scale(recipe, self.o_proj),
        reference=attn_output,
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
    mode, recipe = _boundary_policy(self, "product")
    encoded = pack_boundary(
        self, "product", product, mode, model_dtype=x.model_dtype, recipe=recipe,
        global_scale=_headroom_scale(recipe, down),
        rotation_applied=rotated,
        reference=product,
    )
    return self.down_proj(encoded)


def _layer_forward(self, hidden_states: torch.Tensor | W4AActivation,
                   attention_mask=None, position_ids=None, past_key_values=None,
                   use_cache=False, position_embeddings=None, **kwargs) -> W4AActivation:
    if self._w4a_stream_require_input and not isinstance(hidden_states, W4AActivation):
        raise TypeError("A W4A decoder layer after another W4A layer must receive encoded activations and scales.")
    mode, recipe = _boundary_policy(self, "input")
    x = _as_stream(hidden_states, mode, recipe, owner=self)
    residual = x
    x = self.input_layernorm(x)
    x, _ = self.self_attn(
        hidden_states=x, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, use_cache=use_cache,
        position_embeddings=position_embeddings, **kwargs,
    )
    x = _add_stream(residual, x, owner=self, boundary="attention_residual")
    residual = x
    x = self.post_attention_layernorm(x)
    x = self.mlp(x)
    return _add_stream(residual, x, owner=self, boundary="output")


def _decode_input(_module, args):
    if args and isinstance(args[0], W4AActivation):
        return (args[0].exact(), *args[1:])
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
                             recipe: str | None = None, version: int = 3,
                             global_scales: dict[str, float] | None = None,
                             attention_mode: str | None = None,
                             attention_recipe: str | None = None,
                             mlp_fp8_layers: tuple[int, ...] | None = None) -> None:
    """Install an encoded stream on complete W4A Llama decoder layers.

    ``mode``/``recipe`` describe the MLP operand. ``attention_mode`` defaults to
    ``mode``; a mixed-precision recipe passes ``attention_mode="w4afp8"`` so the
    attention projections carry FP8 while the MLP keeps NVFP4, which is the
    split NVIDIA's W4A4 model recipes use.

    ``mlp_fp8_layers`` names decoder layers whose MLP boundaries carry FP8
    instead of the wide-range operand. Per-layer precision selection is how
    ModelOpt keeps the few most sensitive blocks out of the 4-bit activation
    grid; every other layer keeps the stream default.

    A partial decoder selection is rejected so its remaining projections
    cannot silently convert the stream back to ordinary BF16 tensors.
    """
    from ...quantization.activation_floatx import normalize_nvfp4_recipe

    attention_mode = attention_mode or mode
    if attention_mode not in {"w4afp8", "w4a_nvfp4"}:
        raise ValueError(f"Unsupported W4A attention activation mode: {attention_mode}.")
    if attention_recipe is None and attention_mode == "w4a_nvfp4":
        attention_recipe = recipe
    if attention_mode == "w4a_nvfp4":
        attention_recipe = normalize_nvfp4_recipe(attention_recipe)
    else:
        attention_recipe = None
    if mode not in {"w4afp8", "w4a_nvfp4"}:
        raise ValueError(f"Unsupported W4A activation stream mode: {mode}.")
    if mode == "w4a_nvfp4":
        from ...quantization.activation_floatx import normalize_nvfp4_recipe

        recipe = normalize_nvfp4_recipe(recipe)
    if version not in {2, 3, 4}:
        raise ValueError(f"Unsupported W4A activation stream version: {version}.")
    if version == 4 and mode != "w4a_nvfp4":
        raise ValueError("Version 4 token-scale RMSNorm currently requires NVFP4.")
    if global_scales is not None and (version != 4 or mode != "w4a_nvfp4"):
        raise ValueError("Calibrated producer scales require version-4 NVFP4")
    if mode == "w4afp8" and recipe is not None:
        raise ValueError("W4AFP8 does not use an NVFP4 scale recipe.")
    if mode == "w4a_nvfp4" and recipe not in {
        "nvidia", "nvidia_headroom", "four_six", "least_squares", "least_squares_headroom", "least_squares_grid"
    }:
        raise ValueError(
            "W4A NVFP4 requires an explicit `nvidia`, `nvidia_headroom`, "
            "`four_six`, `least_squares`, `least_squares_headroom`, or `least_squares_grid` recipe."
        )
    mlp_fp8_layers = tuple(mlp_fp8_layers or ())
    if len(set(mlp_fp8_layers)) != len(mlp_fp8_layers):
        raise ValueError("W4A per-layer MLP overrides must not repeat a decoder layer index.")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0
           for index in mlp_fp8_layers):
        raise ValueError("W4A per-layer MLP overrides require non-negative decoder layer indices.")
    if mlp_fp8_layers and mode != "w4a_nvfp4":
        raise ValueError("A per-layer MLP override only refines an NVFP4 activation stream.")
    # Attention boundaries carry the narrow-range operand, the MLP boundaries
    # the wide-range one. Both may name the same policy for a uniform stream.
    fp8_mlp_layers = frozenset(mlp_fp8_layers)

    def mlp_policy(layer_index: int) -> tuple[str, str | None]:
        """Return the MLP transport policy for one decoder layer."""
        return ("w4afp8", None) if layer_index in fp8_mlp_layers else (mode, recipe)

    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is None or not layers or layers[0].__class__.__name__ != "LlamaDecoderLayer":
        raise ValueError("The W4A activation stream currently requires a Llama decoder model.")
    out_of_range = sorted(index for index in mlp_fp8_layers if index >= len(layers))
    if out_of_range:
        raise ValueError(f"W4A per-layer MLP overrides exceed the decoder depth: {out_of_range}")
    if getattr(model, "_w4a_stream_mode", None) == mode:
        if (getattr(model, "_w4a_stream_recipe", None) != recipe
                or getattr(model, "_w4a_stream_attention_mode", None) != attention_mode
                or getattr(model, "_w4a_stream_attention_recipe", None) != attention_recipe
                or getattr(model, "_w4a_stream_mlp_fp8_layers", None) != mlp_fp8_layers
                or getattr(model, "_w4a_stream_version", None) != version
                or getattr(model, "_w4a_stream_global_scales", None) != global_scales):
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
        present = [isinstance(m, W4ANVFP4Linear) if mode == "w4a_nvfp4"
                   else isinstance(m, W4AFP8Linear) and not isinstance(m, W4ANVFP4Linear)
                   for m in modules]
        if any(present) and not all(present):
            raise ValueError("A W4A activation stream requires all seven projections in a selected Llama layer.")
        if all(present) and any(module.out_features % 128 for module in modules):
            raise ValueError("W4A activation transport requires every selected projection output width divisible by 128.")
        if all(present) and version == 4:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                if not bool((norm.weight == 1).all()):
                    raise ValueError("Version 4 requires RMSNorm weights fused into the GPTQ projections.")
        selected.append(all(present))
    if not any(selected):
        raise ValueError("No complete W4A decoder layer was found.")

    boundaries = []
    if version == 4:
        boundaries = list(llama_nvfp4_boundaries(layers, selected))
        # Only NVFP4 producers own a calibrated global scale; FP8 boundaries
        # pack dynamically, so a mixed stream needs a subset of the keys. The
        # subset follows the per-layer MLP policy, so a layer promoted to FP8
        # stops requiring an NVFP4 producer scale.
        def boundary_is_nvfp4(boundary: str, key: str) -> bool:
            if _boundary_group(None, boundary) == "mlp":
                return mlp_policy(int(key.split(".")[2]))[0] == "w4a_nvfp4"
            return attention_mode == "w4a_nvfp4"

        nvfp4_boundaries = [(owner, boundary, key) for owner, boundary, key in boundaries
                            if boundary_is_nvfp4(boundary, key)]
        if global_scales is not None:
            unknown = set(global_scales) - {key for _, _, key in boundaries}
            if unknown:
                raise ValueError(f"Unknown NVFP4 producer scales: {sorted(unknown)}")
            missing = {key for _, _, key in nvfp4_boundaries} - set(global_scales)
            if missing:
                raise ValueError(f"Incomplete NVFP4 producer scales: missing={sorted(missing)}")
        for owner, boundary, key in nvfp4_boundaries:
            device = layers[int(key.split(".")[2])].self_attn.q_proj.qweight.device
            # The module remains movable; these copies are not saved as new
            # weight tensors. Metadata is the single serialization authority.
            owner.add_module(f"_w4a_{boundary}_quantizer", NVFP4BoundaryQuantizer(
                key, device, global_scales[key] if global_scales is not None else None,
            ))

    for index, (layer, enabled) in enumerate(zip(layers, selected)):
        if not enabled:
            continue
        layer_mlp_mode, layer_mlp_recipe = mlp_policy(index)
        for path in required:
            parent, child = path.split(".")
            linear = getattr(getattr(layer, parent), child)
            linear._require_activation_stream = True
            linear._w4a_activation_recipe = (
                attention_recipe if parent == "self_attn" else layer_mlp_recipe
            )
            # Version 2 encoded every Linear output and often decoded it at
            # the very next BF16-only operator. Version 3 emits model dtype
            # directly for those branches and packs once at the next actual
            # FP8/FP4 consumer or decoder-layer boundary.
            linear._w4a_output_encoded = version == 2
        # Declare the transport policy at each producer boundary. Attention
        # boundaries may name a different mode than the MLP boundaries, which
        # is how a mixed FP8-attention / NVFP4-MLP stream is expressed.
        _set_boundary_policy(layer, "input", attention_mode, attention_recipe)
        _set_boundary_policy(layer, "attention_residual", layer_mlp_mode, layer_mlp_recipe)
        _set_boundary_policy(layer, "output", attention_mode, attention_recipe)
        _set_boundary_policy(layer.self_attn, "output", attention_mode, attention_recipe)
        _set_boundary_policy(layer.mlp, "product", layer_mlp_mode, layer_mlp_recipe)
        _set_boundary_policy(layer.input_layernorm, "norm", attention_mode, attention_recipe)
        _set_boundary_policy(layer.post_attention_layernorm, "norm", layer_mlp_mode, layer_mlp_recipe)
        layer._w4a_stream_mode = mode
        layer._w4a_stream_recipe = recipe
        layer._w4a_stream_version = version
        layer._w4a_stream_require_input = index > 0 and selected[index - 1]
        if (layer_mlp_mode == "w4a_nvfp4"
                and layer_mlp_recipe in {"nvidia_headroom", "least_squares_headroom"}):
            qkv_scales = [
                layer.self_attn.q_proj.activation_global_scale,
                layer.self_attn.k_proj.activation_global_scale,
                layer.self_attn.v_proj.activation_global_scale,
            ]
            gate_up_scales = [
                layer.mlp.gate_proj.activation_global_scale,
                layer.mlp.up_proj.activation_global_scale,
            ]
            if not all(torch.equal(qkv_scales[0], value) for value in qkv_scales[1:]):
                raise ValueError("Shared Q/K/V NVFP4 carriers require identical calibrated scales.")
            if not torch.equal(gate_up_scales[0], gate_up_scales[1]):
                raise ValueError("Shared gate/up NVFP4 carriers require identical calibrated scales.")
            layer.input_layernorm._w4a_output_global_scale = (
                qkv_scales[0]
            )
            layer.post_attention_layernorm._w4a_output_global_scale = (
                gate_up_scales[0]
            )
        _bind_forward(layer, _layer_forward)
        _bind_forward(layer.self_attn, _attention_forward)
        _bind_forward(layer.mlp, _mlp_forward)
        layer.input_layernorm._w4a_preserve_norm_codes = version == 4 and attention_mode == "w4a_nvfp4"
        layer.post_attention_layernorm._w4a_preserve_norm_codes = (
            version == 4 and layer_mlp_mode == "w4a_nvfp4"
        )
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
    model._w4a_stream_attention_mode = attention_mode
    model._w4a_stream_attention_recipe = attention_recipe
    model._w4a_stream_mlp_fp8_layers = mlp_fp8_layers
    model._w4a_stream_version = version
    model._w4a_stream_global_scales = dict(global_scales) if global_scales is not None else None
