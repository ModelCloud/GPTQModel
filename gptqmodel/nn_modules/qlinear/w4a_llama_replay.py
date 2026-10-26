# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tensor-valued calibration replay for the encoded Llama W4A stream.

GPTQ capture must receive the activation after quantization, while layer
replay must propagate the same rounded operator boundaries used at runtime.
The replay uses independent Torch arithmetic and retains ordinary tensors so
the GPTQ Hessian collector and Transformers control flow remain compatible.
"""

from __future__ import annotations

import torch

from ...quantization.activation_floatx import (
    fp8_token_qdq,
    nvfp4_block_qdq,
    nvfp4_global_scale,
    nvfp4_uses_headroom,
)
from .w4a_boundary import layer_mlp_policy, norm_codes_fused
from .w4a_llama_stream import _bind_forward, _track_bound_forward


def round_w4a_activation(x: torch.Tensor, mode: str,
                         recipe: str | None = None,
                         global_scale: torch.Tensor | float | None = None) -> torch.Tensor:
    """Reference W4A QDQ used by capture and per-layer GPTQ replay."""
    if mode == "w4afp8":
        return fp8_token_qdq(x)
    if mode == "w4a_nvfp4":
        scale = global_scale
        if scale is None:
            scale = nvfp4_global_scale(
                x.detach().abs().amax(), grid_dtype=x.dtype, recipe=recipe or "least_squares"
            )
        return nvfp4_block_qdq(
            x, scale, recipe or "least_squares",
        )
    raise ValueError(f"Unknown W4A replay mode: {mode}")


# Kept private at call sites so model replay reads naturally.
_round = round_w4a_activation


def round_w4a_replay_operand(x: torch.Tensor, mode: str, recipe=None, global_scale=None,
                             *, rounder=None) -> torch.Tensor:
    """Round one replay operand with the precision of its deployed carrier.

    The deployed stream carries every decoded operand in FP32 and consumes the
    source scale grid, so replay must not round-trip through the model dtype
    before the GEMM. FP8 boundaries carry no calibrated scale and keep the
    ordinary per-token dynamic rounding, so a mixed stream can round attention
    and MLP operands under different policies.
    """
    rounder = _round if rounder is None else rounder
    # The deployed GEMM consumes codes and FP32 scales directly for every
    # carrier, so an FP16/BF16 round-trip here would teach a different function
    # than inference.
    if mode == "w4a_nvfp4" and global_scale is None:
        global_scale = nvfp4_global_scale(x.detach().abs().amax(), grid_dtype=x.dtype,
                                          recipe=recipe or "least_squares")
    return rounder(x.float(), mode, recipe, global_scale)


def _replay_linear_forward(self, x):
    if getattr(self, "_w4a_replay_disabled", False):
        return self._w4a_replay_original_forward(x)
    if not self._w4a_replay_custom_forward:
        result = torch.nn.functional.linear(x.float(), self.weight.float(),
                                             self.bias.float() if self.bias is not None else None)
    else:
        # QAD's native-INT4/scale wrappers implement their own FP32 group
        # arithmetic and remain differentiable through the decoded operand.
        result = self._w4a_replay_original_forward(x)
    return result.to(self._w4a_replay_model_dtype)


def _replay_exit(_module, args):
    """Match the decode at an unquantized layer or final-norm boundary."""
    if not args or not isinstance(args[0], torch.Tensor):
        # A runtime encoded carrier already owns its dtype contract.
        return None
    return (args[0].to(_module._w4a_replay_boundary_dtype), *args[1:])


def set_w4a_replay_enabled(module: torch.nn.Module, enabled: bool) -> None:
    """Switch the current layer between pristine teacher and quantized capture.

    GPTAQ/FOEM require native inputs from the unrounded reference pass. Setting
    this on the layer (and its descendants) also follows replicated modules and
    HookedLinear replacement; a process-global flag would not provide that.
    """
    for child in module.modules():
        child._w4a_replay_disabled = not enabled


def _replay_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Create the down-projection operand after Hadamard and round it once."""
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    product = self.act_fn(gate) * up
    down = self.down_proj
    online_full_had = bool(getattr(down, "online_full_had", False))
    online_partial_had = bool(getattr(down, "online_partial_had", False))
    if online_full_had or online_partial_had:
        from ...quantization.rotation.hadamard_utils import apply_online_hadamard

        product = apply_online_hadamard(
            product,
            online_full_had=online_full_had,
            online_partial_had=online_partial_had,
            had_K=getattr(down, "had_K", None),
            K=getattr(down, "K", 1),
            had_dim=getattr(down, "had_dim", -1),
        )
    if (not getattr(down, "_w4a_headroom_probe", False)
            and not getattr(self, "_w4a_replay_disabled", False)):
        product = round_w4a_replay_operand(
            product, self._w4a_replay_mode, self._w4a_replay_recipe,
            getattr(down, "_w4a_activation_global_scale", None),
        )
    return down(product)


def _replay_norm_operand(norm: torch.nn.Module, pristine: torch.Tensor,
                         rounded: torch.Tensor) -> torch.Tensor:
    """Fused-norm RMSNorm shortcut over the deployed encoded stream.

    The encoded stream keeps the residual value in compute precision and only
    rounds the GEMM operand. Its fused RMSNorm therefore rescales the rounded
    carrier by the inverse RMS of the pristine residual instead of
    re-quantizing a normed value. The replay must reproduce that exactly or its
    captured Hessians and QAD signal describe a function the checkpoint never
    executes.
    """
    variance = pristine.float().square().mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + norm.variance_epsilon)
    return (rounded.float() * inv_rms).to(pristine.dtype)


def _probe_active(consumers) -> bool:
    """Return whether a headroom probe is collecting the consumer operands."""

    return any(getattr(consumer, "_w4a_headroom_probe", False) for consumer in consumers)


def _consumer_global_scale(consumers):
    """Resolve the frozen NVFP4 global scale owned by a boundary's consumers.

    A headroom probe freezes its scale on the projections that consume the
    normalized operand. The norm producer must apply that same value during
    capture and runtime. Shared Q/K/V and gate/up carriers are required to
    carry identical scales at install time, so the first one is authoritative.
    """

    for consumer in consumers:
        scale = getattr(consumer, "_w4a_activation_global_scale", None)
        if scale is not None:
            return scale
    return None


def _replay_norm(norm: torch.nn.Module, pristine: torch.Tensor, rounded: torch.Tensor,
                 mode: str, recipe: str | None, *, preserve_codes: bool,
                 consumers=()) -> torch.Tensor:
    """Reproduce one deployed RMSNorm boundary for its carrier policy.

    An NVFP4 boundary with fused norm weights and a non-headroom recipe reuses
    the incoming codes and rescales only the token multiplier. Every other
    boundary packs a freshly normed, weighted value. Headroom scales are
    calibrated on that normed value, so even fused norms bypass rounding while
    probing and then pack with the frozen consumer scale during capture.
    """
    if preserve_codes:
        return _replay_norm_operand(norm, pristine, rounded)
    variance = pristine.float().square().mean(dim=-1, keepdim=True)
    y = pristine.float() * torch.rsqrt(variance + norm.variance_epsilon)
    weight = getattr(norm, "weight", None)
    if weight is not None:
        y = y * weight.float()
    if nvfp4_uses_headroom(recipe):
        if _probe_active(consumers):
            # The probe must see the true normed operand, not a dynamically
            # rounded stand-in, so the frozen scale reflects deployment.
            return y
        return _round(y, mode, recipe, _consumer_global_scale(consumers))
    return _round(y, mode, recipe)


def _replay_layer_forward(self, hidden_states: torch.Tensor,
                          attention_mask=None, position_ids=None, past_key_values=None,
                          use_cache=False, position_embeddings=None, **kwargs) -> torch.Tensor:
    mode = self._w4a_replay_mode
    recipe = self._w4a_replay_recipe
    disabled = getattr(self, "_w4a_replay_disabled", False)
    attention_mode = getattr(self, "_w4a_replay_attention_mode", mode)
    attention_recipe = getattr(self, "_w4a_replay_attention_recipe", recipe)
    mlp_mode = getattr(self, "_w4a_replay_mlp_mode", mode)
    mlp_recipe = getattr(self, "_w4a_replay_mlp_recipe", recipe)
    if disabled:
        # Pristine teacher pass: reproduce the unmodified decoder forward so
        # GPTAQ/FOEM see native values.
        x = hidden_states
        residual = x
        x = self.input_layernorm(x)
        x, _ = self.self_attn(
            hidden_states=x, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            position_embeddings=position_embeddings, **kwargs,
        )
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x

    # Rebuild each norm's operand from the pristine residual and its boundary
    # policy without cross-layer state that reverse-order replay could corrupt.
    # Attention and MLP boundaries may carry different policies, so each is
    # resolved independently.
    pristine = hidden_states.float()
    attention_consumers = (self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj)
    mlp_consumers = (self.mlp.gate_proj, self.mlp.up_proj)
    # Rebuild the incoming carrier only when the norm actually reuses its
    # codes. Its scale belongs to the residual producer, not the normalized
    # Q/K/V operand; headroom consumer scales are applied by _replay_norm.
    attention_preserve_codes = getattr(self, "_w4a_attention_preserve_norm_codes", False)
    rounded = pristine
    if attention_preserve_codes:
        rounded = round_w4a_replay_operand(
            hidden_states, attention_mode, attention_recipe,
            getattr(self, "_w4a_replay_input_global_scale", None),
        )
    x = _replay_norm(
        self.input_layernorm, pristine, rounded, attention_mode, attention_recipe,
        preserve_codes=attention_preserve_codes,
        consumers=attention_consumers,
    )
    x, _ = self.self_attn(
        hidden_states=x, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, use_cache=use_cache,
        position_embeddings=position_embeddings, **kwargs,
    )
    summed = pristine + x.float()
    mlp_preserve_codes = getattr(self, "_w4a_mlp_preserve_norm_codes", False)
    rounded = summed
    if mlp_preserve_codes:
        rounded = round_w4a_replay_operand(
            summed, mlp_mode, mlp_recipe,
            self._w4a_replay_global_scales.get("post_attention_residual"),
        )
    x = _replay_norm(
        self.post_attention_layernorm, summed, rounded, mlp_mode, mlp_recipe,
        preserve_codes=mlp_preserve_codes,
        consumers=mlp_consumers,
    )
    x = self.mlp(x)
    return summed + x.float()


def install_w4a_llama_replay(model: torch.nn.Module, qcfg) -> None:
    """Prepare only complete, selected Llama decoder layers for GPTQ replay."""
    mode = qcfg.activation_mode
    recipe = qcfg.activation_recipe
    if mode == "w4a_nvfp4":
        from ...quantization.activation_floatx import normalize_nvfp4_recipe

        recipe = normalize_nvfp4_recipe(recipe)
    global_scales = getattr(qcfg, "activation_global_scales", None)
    if global_scales is not None and mode != "w4a_nvfp4":
        raise ValueError("Calibrated producer scales require NVFP4")
    # A mixed stream declares its policy per boundary; resolve the same
    # attention and per-layer MLP policies the runtime installs.
    attention_mode = getattr(qcfg, "activation_attention_mode", None) or mode
    attention_recipe = getattr(qcfg, "activation_attention_recipe", None)
    if attention_mode == "w4a_nvfp4":
        from ...quantization.activation_floatx import normalize_nvfp4_recipe

        attention_recipe = normalize_nvfp4_recipe(attention_recipe if attention_recipe is not None else recipe)
    else:
        attention_recipe = None
    mlp_fp8_layers = tuple(getattr(qcfg, "activation_mlp_fp8_layers", None) or ())
    fused_norms = bool(getattr(qcfg, "rotation", None))
    if getattr(model, "_w4a_replay_mode", None) == mode:
        if (getattr(model, "_w4a_replay_recipe", None) != recipe
                or getattr(model, "_w4a_replay_global_scales", None) != global_scales
                or getattr(model, "_w4a_replay_attention_mode", None) != attention_mode
                or getattr(model, "_w4a_replay_attention_recipe", None) != attention_recipe
                or getattr(model, "_w4a_replay_mlp_fp8_layers", None) != mlp_fp8_layers
                or getattr(model, "_w4a_fused_norms", None) != fused_norms):
            raise ValueError("A different W4A replay policy is already installed.")
        return
    if getattr(model, "_w4a_replay_mode", None) is not None:
        raise ValueError("A different W4A calibration replay is already installed.")
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is None or not layers or layers[0].__class__.__name__ != "LlamaDecoderLayer":
        raise ValueError("The W4A activation stream currently requires a Llama decoder model.")
    required = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    selected = []
    for index, layer in enumerate(layers):
        paths = [f"model.layers.{index}.{name}" for name in required]
        chosen = [qcfg.dynamic_get(layer_name=path) is not False for path in paths]
        if any(chosen) and not all(chosen):
            raise ValueError("W4A replay requires all seven projections in a selected Llama layer.")
        if all(chosen):
            for path in required:
                parent, child = path.split(".")
                if getattr(getattr(layer, parent), child).out_features % 128:
                    raise ValueError("W4A replay requires every selected projection output width divisible by 128.")
        selected.append(all(chosen))
    if global_scales is not None:
        from .w4a_boundary import nvfp4_producer_specs, validate_producer_scales

        validate_producer_scales(
            nvfp4_producer_specs(layers, selected, attention_mode=attention_mode, mode=mode,
                                 recipe=recipe, mlp_fp8_layers=mlp_fp8_layers),
            global_scales,
        )
    model_dtype = decoder.embed_tokens.weight.dtype
    replay_handles = []
    for index, (layer, enabled) in enumerate(zip(layers, selected)):
        if not enabled:
            # A layer that follows a selected one consumes its predecessor's
            # compute-precision residual; runtime casts that to the model dtype
            # at the unselected-layer edge (`_decode_input`), so replay does too.
            if index and selected[index - 1]:
                layer._w4a_replay_boundary_dtype = model_dtype
                replay_handles.append(layer.register_forward_pre_hook(_replay_exit))
            continue
        layer_mlp_mode, layer_mlp_recipe = layer_mlp_policy(index, mode, recipe, mlp_fp8_layers)
        layer._w4a_replay_mode = mode
        layer._w4a_replay_recipe = recipe
        layer._w4a_replay_attention_mode = attention_mode
        layer._w4a_replay_attention_recipe = attention_recipe
        layer._w4a_replay_mlp_mode = layer_mlp_mode
        layer._w4a_replay_mlp_recipe = layer_mlp_recipe
        layer._w4a_attention_preserve_norm_codes = (
            fused_norms and attention_mode == "w4a_nvfp4"
            and not nvfp4_uses_headroom(attention_recipe)
            and norm_codes_fused(layer.input_layernorm)
        )
        layer._w4a_mlp_preserve_norm_codes = (
            fused_norms and layer_mlp_mode == "w4a_nvfp4"
            and not nvfp4_uses_headroom(layer_mlp_recipe)
            and norm_codes_fused(layer.post_attention_layernorm)
        )
        layer._w4a_replay_global_scales = {
            boundary: global_scales[f"model.layers.{index}.{boundary}"]
            for boundary in ("input", "post_attention_residual", "output")
            if global_scales is not None and f"model.layers.{index}.{boundary}" in global_scales
        }
        round_input = index == 0 or not selected[index - 1]
        layer._w4a_replay_round_input = round_input
        # An interior layer receives the previous layer's output carrier
        # unchanged, so its input rounding must reuse that layer's calibrated
        # output scale instead of the (absent) per-layer input scale.
        layer._w4a_replay_input_global_scale = (
            global_scales.get(f"model.layers.{index}.input") if round_input
            else global_scales.get(f"model.layers.{index - 1}.output")
        ) if global_scales is not None else None
        _bind_forward(layer, _replay_layer_forward)
        layer.mlp._w4a_replay_mode = layer_mlp_mode
        layer.mlp._w4a_replay_recipe = layer_mlp_recipe
        _bind_forward(layer.mlp, _replay_mlp_forward)
        for path in required:
            parent, child = path.split(".")
            linear = getattr(getattr(layer, parent), child)
            linear_mode, linear_recipe = (
                (attention_mode, attention_recipe) if parent == "self_attn"
                else (layer_mlp_mode, layer_mlp_recipe)
            )
            producer_key = f"model.layers.{index}.{path}.input"
            if (global_scales is not None and path in {"self_attn.o_proj", "mlp.down_proj"}
                    and producer_key in global_scales):
                # A boundary promoted to FP8 owns no calibrated NVFP4 scale.
                linear._w4a_activation_global_scale = global_scales[producer_key]
            if not isinstance(linear, torch.nn.Linear):
                raise TypeError(f"W4A replay expects a dense Linear at model.layers.{index}.{path}.")

            def before(_module, args, *, replay_mode=linear_mode, replay_recipe=linear_recipe):
                if getattr(_module, "_w4a_replay_disabled", False):
                    return None
                if getattr(_module, "_w4a_norm_preapplied", False):
                    return None
                if getattr(_module, "_w4a_headroom_probe", False):
                    return None
                # The MLP wrapper applies Hadamard first and creates the exact
                # down-projection operand. Runtime marks that carrier as
                # pre-rotated and consumes it directly; replay must do the
                # same instead of passing it through a second QDQ pre-hook.
                if getattr(_module, "_w4a_rotation_preapplied", False):
                    return None
                return (round_w4a_replay_operand(
                    args[0], replay_mode, replay_recipe,
                    getattr(_module, "_w4a_activation_global_scale", None),
                ), *args[1:])

            replay_handles.append(linear.register_forward_pre_hook(before))
            linear._w4a_replay_input_hook_active = True
            linear._w4a_norm_preapplied = path in {
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"
            }
            linear._w4a_stream_replay_mode = linear_mode
            linear._w4a_stream_replay_recipe = linear_recipe
            linear._w4a_replay_model_dtype = model_dtype
            linear._w4a_stream_replay_pre_hook = True
            # Replay carries a compute-precision operand, so the linear must
            # compute in FP32 and return the model dtype rather than relying on
            # a matching-dtype `F.linear`.
            linear._w4a_replay_original_forward = getattr(linear, "_old_forward", linear.forward)
            linear._w4a_replay_custom_forward = (
                getattr(linear._w4a_replay_original_forward, "__func__", None) is not torch.nn.Linear.forward)
            # Replay falls back to this forward, so a data-parallel replica must
            # run its own rather than the original module's bound method.
            _track_bound_forward(linear, "_w4a_replay_original_forward")
            _bind_forward(linear, _replay_linear_forward)
            # The MLP wrapper handles rotation (when enabled) and quantization.
            # Even an identity rotation must not trigger a second input QDQ.
            if path == "mlp.down_proj":
                linear._w4a_rotation_preapplied = True
    if selected[-1]:
        # Runtime's final norm consumes the compute-precision residual through
        # `exact()`, which returns the model dtype, so cast at the same edge.
        decoder.norm._w4a_replay_boundary_dtype = model_dtype
        replay_handles.append(decoder.norm.register_forward_pre_hook(_replay_exit))
    model._w4a_replay_handles = replay_handles
    model._w4a_replay_mode = mode
    model._w4a_replay_recipe = recipe
    model._w4a_replay_global_scales = dict(global_scales) if global_scales is not None else None
    model._w4a_replay_attention_mode = attention_mode
    model._w4a_replay_attention_recipe = attention_recipe
    model._w4a_replay_mlp_fp8_layers = mlp_fp8_layers
    model._w4a_fused_norms = fused_norms


_REPLAY_MODEL_ATTRS = (
    "_w4a_replay_handles", "_w4a_replay_mode", "_w4a_replay_recipe",
    "_w4a_replay_global_scales",
    "_w4a_replay_attention_mode", "_w4a_replay_attention_recipe",
    "_w4a_replay_mlp_fp8_layers", "_w4a_fused_norms",
)

_REPLAY_LAYER_ATTRS = (
    "_w4a_replay_mode", "_w4a_replay_recipe",
    "_w4a_replay_attention_mode", "_w4a_replay_attention_recipe",
    "_w4a_replay_mlp_mode", "_w4a_replay_mlp_recipe",
    "_w4a_replay_global_scales", "_w4a_replay_input_global_scale",
    "_w4a_replay_round_input", "_w4a_replay_boundary_dtype",
    "_w4a_attention_preserve_norm_codes", "_w4a_mlp_preserve_norm_codes",
)


def uninstall_w4a_llama_replay(model: torch.nn.Module) -> None:
    """Remove replay hooks and policy metadata before runtime stream installation.

    Quantization installs replay on the dense model and the looper later
    replaces those Linears with packed modules. The replay's dtype-exit hooks
    live on the decoder layers and final norm, which are not replaced, so they
    must be removed explicitly. Otherwise a runtime carrier reaches
    ``_replay_exit`` and fails on a missing ``.to``.
    """
    for handle in getattr(model, "_w4a_replay_handles", None) or ():
        handle.remove()
    for attr in _REPLAY_MODEL_ATTRS:
        if hasattr(model, attr):
            delattr(model, attr)
    decoder = getattr(model, "model", None)
    for layer in getattr(decoder, "layers", None) or ():
        for attr in _REPLAY_LAYER_ATTRS:
            if hasattr(layer, attr):
                delattr(layer, attr)
    norm = getattr(decoder, "norm", None)
    if norm is not None and hasattr(norm, "_w4a_replay_boundary_dtype"):
        delattr(norm, "_w4a_replay_boundary_dtype")
