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

from ...quantization.activation_floatx import fp8_token_qdq, nvfp4_block_qdq, nvfp4_global_scale
from .w4a_llama_stream import _bind_forward


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
                             *, version: int = 3, rounder=None) -> torch.Tensor:
    """Keep the decoded version-4 carrier in FP32, using the source scale grid."""
    rounder = _round if rounder is None else rounder
    if version == 4:
        if mode != "w4a_nvfp4":
            raise ValueError("Version-4 replay requires NVFP4")
        if global_scale is None:
            global_scale = nvfp4_global_scale(x.detach().abs().amax(), grid_dtype=x.dtype,
                                               recipe=recipe or "least_squares")
        x = x.float()
    return rounder(x, mode, recipe, global_scale)


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
    if args:
        return (args[0].to(_module._w4a_replay_boundary_dtype), *args[1:])
    return None


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
            version=self._w4a_replay_version,
        )
    return down(product)


def _replay_norm_operand(norm: torch.nn.Module, pristine: torch.Tensor,
                         rounded: torch.Tensor) -> torch.Tensor:
    """Version-4 RMSNorm shortcut over the deployed encoded stream.

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


def _replay_layer_forward(self, hidden_states: torch.Tensor,
                          attention_mask=None, position_ids=None, past_key_values=None,
                          use_cache=False, position_embeddings=None, **kwargs) -> torch.Tensor:
    mode = self._w4a_replay_mode
    recipe = self._w4a_replay_recipe
    disabled = getattr(self, "_w4a_replay_disabled", False)
    version = getattr(self, "_w4a_replay_version", 3)
    if version == 4 and not disabled:
        # Version-4 carries the residual stream in compute precision. The
        # rounded operand is rebuilt from the pristine value at each boundary,
        # reproducing the deployed stream's "rounded numerator, pristine
        # denominator" RMSNorm without holding cross-layer state that reverse
        # order recomputation would corrupt.
        pristine = hidden_states.float()
        # A stream entry packs the raw model-dtype input; an interior layer
        # rebuilds the previous layer's rounded output from the same producer
        # dtype so the dynamic grid selection still matches the runtime.
        rounded = round_w4a_replay_operand(
            hidden_states, mode, recipe, getattr(self, "_w4a_replay_input_global_scale", None),
            version=version)
        x = _replay_norm_operand(self.input_layernorm, pristine, rounded)
        x, _ = self.self_attn(
            hidden_states=x, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            position_embeddings=position_embeddings, **kwargs,
        )
        summed = pristine + x.float()
        rounded = round_w4a_replay_operand(
            summed, mode, recipe, self._w4a_replay_global_scales.get("post_attention_residual"),
            version=version)
        x = _replay_norm_operand(self.post_attention_layernorm, summed, rounded)
        x = self.mlp(x)
        return summed + x.float()

    def rounded(value, boundary):
        return value if disabled else round_w4a_replay_operand(
            value, mode, recipe, self._w4a_replay_global_scales.get(boundary),
            version=version)
    x = rounded(hidden_states, "input") if self._w4a_replay_round_input else hidden_states
    residual = x
    x = self.input_layernorm(x)
    x, _ = self.self_attn(
        hidden_states=x, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, use_cache=use_cache,
        position_embeddings=position_embeddings, **kwargs,
    )
    x = rounded(residual.float() + x.float(), "post_attention_residual").to(residual.dtype)
    residual = x
    x = self.post_attention_layernorm(x)
    x = self.mlp(x)
    return rounded(residual.float() + x.float(), "output").to(residual.dtype)


def install_w4a_llama_replay(model: torch.nn.Module, qcfg) -> None:
    """Prepare only complete, selected Llama decoder layers for GPTQ replay."""
    mode = qcfg.activation_mode
    recipe = qcfg.activation_recipe
    if mode == "w4a_nvfp4":
        from ...quantization.activation_floatx import normalize_nvfp4_recipe

        recipe = normalize_nvfp4_recipe(recipe)
    version = getattr(qcfg, "activation_version", 3)
    if version == 4 and mode != "w4a_nvfp4":
        raise ValueError("Version-4 replay requires NVFP4")
    global_scales = getattr(qcfg, "activation_global_scales", None)
    if global_scales is not None and (version != 4 or mode != "w4a_nvfp4"):
        raise ValueError("Calibrated producer scales require version-4 NVFP4")
    if getattr(model, "_w4a_replay_mode", None) == mode:
        if (getattr(model, "_w4a_replay_recipe", None) != recipe
                or getattr(model, "_w4a_replay_version", None) != version
                or getattr(model, "_w4a_replay_global_scales", None) != global_scales):
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
        from .w4a_boundary import llama_nvfp4_boundaries, validate_producer_scales

        validate_producer_scales(list(llama_nvfp4_boundaries(layers, selected)), global_scales)
    model_dtype = decoder.embed_tokens.weight.dtype
    for index, (layer, enabled) in enumerate(zip(layers, selected)):
        if not enabled:
            if version == 4 and index and selected[index - 1]:
                layer._w4a_replay_boundary_dtype = model_dtype
                layer.register_forward_pre_hook(_replay_exit)
            continue
        layer._w4a_replay_mode = mode
        layer._w4a_replay_recipe = recipe
        layer._w4a_replay_version = version
        layer._w4a_replay_global_scales = {
            boundary: global_scales[f"model.layers.{index}.{boundary}"]
            for boundary in ("input", "post_attention_residual", "output")
            if global_scales is not None and f"model.layers.{index}.{boundary}" in global_scales
        }
        round_input = index == 0 or not selected[index - 1]
        layer._w4a_replay_round_input = round_input
        # Version 4 receives the previous layer's output carrier unchanged, so
        # its input rounding must reuse that layer's calibrated output scale
        # instead of the (absent) per-layer input scale.
        layer._w4a_replay_input_global_scale = (
            global_scales.get(f"model.layers.{index}.input") if round_input
            else global_scales.get(f"model.layers.{index - 1}.output")
        ) if global_scales is not None else None
        _bind_forward(layer, _replay_layer_forward)
        layer.mlp._w4a_replay_mode = mode
        layer.mlp._w4a_replay_recipe = recipe
        layer.mlp._w4a_replay_version = version
        _bind_forward(layer.mlp, _replay_mlp_forward)
        for path in required:
            parent, child = path.split(".")
            linear = getattr(getattr(layer, parent), child)
            if global_scales is not None and path in {"self_attn.o_proj", "mlp.down_proj"}:
                linear._w4a_activation_global_scale = global_scales[f"model.layers.{index}.{path}.input"]
            if not isinstance(linear, torch.nn.Linear):
                raise TypeError(f"W4A replay expects a dense Linear at model.layers.{index}.{path}.")

            def before(_module, args, *, replay_mode=mode, replay_recipe=recipe, replay_version=version):
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
                    version=replay_version,
                ), *args[1:])

            def after(_module, _args, output, *, replay_mode=mode, replay_recipe=recipe):
                if getattr(_module, "_w4a_replay_disabled", False):
                    return output
                return _round(output, replay_mode, replay_recipe)

            linear.register_forward_pre_hook(before)
            linear._w4a_replay_input_hook_active = True
            if version == 2:
                linear.register_forward_hook(after)
            linear._w4a_norm_preapplied = version == 4 and path in {
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"
            }
            linear._w4a_stream_replay_mode = mode
            linear._w4a_stream_replay_recipe = recipe
            linear._w4a_stream_replay_version = version
            linear._w4a_replay_model_dtype = model_dtype
            linear._w4a_stream_replay_pre_hook = True
            if version == 4:
                linear._w4a_replay_original_forward = getattr(linear, "_old_forward", linear.forward)
                linear._w4a_replay_custom_forward = (
                    getattr(linear._w4a_replay_original_forward, "__func__", None) is not torch.nn.Linear.forward)
                _bind_forward(linear, _replay_linear_forward)
            # The MLP wrapper handles rotation (when enabled) and quantization.
            # Even an identity rotation must not trigger a second input QDQ.
            if path == "mlp.down_proj":
                linear._w4a_rotation_preapplied = True
    if version == 4 and selected[-1]:
        decoder.norm._w4a_replay_boundary_dtype = model_dtype
        decoder.norm.register_forward_pre_hook(_replay_exit)
    model._w4a_replay_mode = mode
    model._w4a_replay_recipe = recipe
    model._w4a_replay_version = version
    model._w4a_replay_global_scales = dict(global_scales) if global_scales is not None else None
