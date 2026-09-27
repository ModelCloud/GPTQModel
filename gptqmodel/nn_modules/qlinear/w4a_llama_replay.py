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

from ...quantization.activation_floatx import fp8_token_qdq
from .w4a_llama_stream import _bind_forward


def round_w4a_activation(x: torch.Tensor, mode: str,
                         recipe: str | None = None,
                         global_scale: torch.Tensor | float | None = None) -> torch.Tensor:
    """Reference W4A QDQ used by capture and per-layer GPTQ replay."""
    if mode == "w4afp8":
        return fp8_token_qdq(x)
    raise ValueError(f"Unknown W4A replay mode: {mode}")


# Kept private at call sites so model replay reads naturally.
_round = round_w4a_activation


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
    product = _round(product, self._w4a_replay_mode, self._w4a_replay_recipe)
    return down(product)


def _replay_layer_forward(self, hidden_states: torch.Tensor,
                          attention_mask=None, position_ids=None, past_key_values=None,
                          use_cache=False, position_embeddings=None, **kwargs) -> torch.Tensor:
    mode = self._w4a_replay_mode
    recipe = self._w4a_replay_recipe
    x = _round(hidden_states, mode, recipe) if self._w4a_replay_round_input else hidden_states
    residual = x
    x = self.input_layernorm(x)
    x, _ = self.self_attn(
        hidden_states=x, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, use_cache=use_cache,
        position_embeddings=position_embeddings, **kwargs,
    )
    x = _round(residual.float() + x.float(), mode, recipe).to(residual.dtype)
    residual = x
    x = self.post_attention_layernorm(x)
    x = self.mlp(x)
    return _round(residual.float() + x.float(), mode, recipe).to(residual.dtype)


def install_w4a_llama_replay(model: torch.nn.Module, qcfg) -> None:
    """Prepare only complete, selected Llama decoder layers for GPTQ replay."""
    mode = qcfg.activation_mode
    recipe = qcfg.activation_recipe
    version = getattr(qcfg, "activation_version", 3)
    if getattr(model, "_w4a_replay_mode", None) == mode:
        if (getattr(model, "_w4a_replay_recipe", None) != recipe
                or getattr(model, "_w4a_replay_version", None) != version):
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
    for index, (layer, enabled) in enumerate(zip(layers, selected)):
        if not enabled:
            continue
        layer._w4a_replay_mode = mode
        layer._w4a_replay_recipe = recipe
        layer._w4a_replay_round_input = index == 0 or not selected[index - 1]
        _bind_forward(layer, _replay_layer_forward)
        layer.mlp._w4a_replay_mode = mode
        layer.mlp._w4a_replay_recipe = recipe
        _bind_forward(layer.mlp, _replay_mlp_forward)
        for path in required:
            parent, child = path.split(".")
            linear = getattr(getattr(layer, parent), child)
            if not isinstance(linear, torch.nn.Linear):
                raise TypeError(f"W4A replay expects a dense Linear at model.layers.{index}.{path}.")

            def before(_module, args, *, replay_mode=mode, replay_recipe=recipe):
                # The MLP wrapper applies Hadamard first and creates the exact
                # down-projection operand. Runtime marks that carrier as
                # pre-rotated and consumes it directly; replay must do the
                # same instead of passing it through a second QDQ pre-hook.
                if getattr(_module, "_w4a_rotation_preapplied", False):
                    return None
                return (_round(
                    args[0], replay_mode, replay_recipe,
                ), *args[1:])

            def after(_module, _args, output, *, replay_mode=mode, replay_recipe=recipe):
                return _round(output, replay_mode, replay_recipe)

            linear.register_forward_pre_hook(before)
            if version == 2:
                linear.register_forward_hook(after)
            linear._w4a_stream_replay_mode = mode
            linear._w4a_stream_replay_recipe = recipe
            linear._w4a_stream_replay_version = version
            linear._w4a_stream_replay_pre_hook = True
            if path == "mlp.down_proj" and (
                getattr(linear, "online_full_had", False)
                or getattr(linear, "online_partial_had", False)
            ):
                linear._w4a_rotation_preapplied = True
    model._w4a_replay_mode = mode
    model._w4a_replay_recipe = recipe
    model._w4a_replay_version = version
