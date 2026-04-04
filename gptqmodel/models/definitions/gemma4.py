# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import MethodType

import torch

from ...utils.device import get_device
from ...utils.model import get_module_by_name_prefix, move_to, nested_move_to
from ..base import BaseQModel
from . import LlamaQModel


_GEMMA4_ALL_PER_LAYER_INPUTS = "__gptqmodel_gemma4_all_per_layer_inputs"


def _gemma4_module_tree():
    """Return the Gemma 4 decoder traversal with optional attention and per-layer input modules."""

    return [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "q_proj:0",
                "k_norm:!",
                "k_proj:0",
                "v_norm:!",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
            "per_layer_input_gate": ("per_layer_input_gate:0",),
            "post_per_layer_input_norm": ("post_per_layer_input_norm:!",),
            "per_layer_projection": ("per_layer_projection:1",),
        },
    ]


def _capture_gemma4_positional_inputs(model_def, args, kwargs, batch_device):
    """Preserve Gemma 4 per-layer adapter inputs that flow through decoder layers positionally."""

    layer_input = super(type(model_def), model_def).capture_first_layer_positional_inputs(args, kwargs, batch_device)
    per_layer_input = args[1] if len(args) > 1 else kwargs.get("per_layer_input")
    if per_layer_input is not None:
        layer_input.append(move_to(per_layer_input, device=batch_device))
    return layer_input


def _prepare_gemma4_replay_kwargs(model_def, layer, layer_input, additional_inputs, target_device):
    """Refresh Gemma 4 rotary kwargs per layer so replay follows sliding/full attention boundaries."""

    rotary_path = getattr(model_def, "rotary_embedding", None)
    if not rotary_path or not layer_input:
        return additional_inputs

    rotary, _ = get_module_by_name_prefix(model_def.model, [rotary_path])
    if rotary is None:
        return additional_inputs

    layer_type = getattr(getattr(layer, "self_attn", None), "layer_type", None)
    if layer_type is None:
        return additional_inputs

    hidden_states = layer_input[0]
    seq_len = hidden_states.shape[1] if hidden_states.dim() >= 2 else hidden_states.shape[0]
    batch_dim = hidden_states.shape[0] if hidden_states.dim() >= 2 else 1

    position_ids = additional_inputs.get("position_ids")
    if position_ids is None or position_ids.shape[-1] != seq_len:
        position_ids = torch.arange(seq_len, device=target_device, dtype=torch.long).unsqueeze(0).expand(batch_dim, -1)
        additional_inputs["position_ids"] = position_ids

    try:
        rotary_device = get_device(rotary)
    except Exception:
        rotary_device = position_ids.device

    rotary_position_ids = move_to(position_ids, device=rotary_device)
    rotary_input = torch.empty(1, device=rotary_device, dtype=hidden_states.dtype)
    additional_inputs["position_embeddings"] = nested_move_to(
        rotary(rotary_input, rotary_position_ids, layer_type),
        device=target_device,
    )

    if len(layer_input) == 1:
        all_per_layer_inputs = additional_inputs.pop(_GEMMA4_ALL_PER_LAYER_INPUTS, None)
        layer_index = getattr(getattr(layer, "self_attn", None), "layer_idx", None)
        if all_per_layer_inputs is not None and layer_index is not None:
            additional_inputs["per_layer_input"] = move_to(
                all_per_layer_inputs[:, :, layer_index, :],
                device=target_device,
            )
    else:
        additional_inputs.pop(_GEMMA4_ALL_PER_LAYER_INPUTS, None)

    return additional_inputs


def _resolve_gemma4_language_model(model_def):
    """Return the Gemma 4 text stack that owns per-layer input projection state."""

    if hasattr(model_def.model, "model") and hasattr(model_def.model.model, "language_model"):
        return model_def.model.model.language_model
    return model_def.model.model


def _patch_gemma4_per_layer_input_capture(model_def):
    """Capture projected per-layer inputs during calibration so later decoder replays can slice them by layer."""

    language_model = _resolve_gemma4_language_model(model_def)
    if getattr(language_model, "_gptqmodel_project_per_layer_inputs_patched", False):
        return

    original = language_model.project_per_layer_inputs

    def patched(self, inputs_embeds, per_layer_inputs=None):
        result = original(inputs_embeds, per_layer_inputs)
        setattr(self, "_gptqmodel_cached_all_per_layer_inputs", result)
        return result

    language_model._gptqmodel_original_project_per_layer_inputs = original
    language_model.project_per_layer_inputs = MethodType(patched, language_model)
    language_model._gptqmodel_project_per_layer_inputs_patched = True


def _restore_gemma4_per_layer_input_capture(model_def):
    """Restore Gemma 4 per-layer input helpers after calibration capture completes."""

    language_model = _resolve_gemma4_language_model(model_def)
    original = getattr(language_model, "_gptqmodel_original_project_per_layer_inputs", None)
    if original is not None:
        language_model.project_per_layer_inputs = original
        delattr(language_model, "_gptqmodel_original_project_per_layer_inputs")
    if hasattr(language_model, "_gptqmodel_project_per_layer_inputs_patched"):
        delattr(language_model, "_gptqmodel_project_per_layer_inputs_patched")
    if hasattr(language_model, "_gptqmodel_cached_all_per_layer_inputs"):
        delattr(language_model, "_gptqmodel_cached_all_per_layer_inputs")


class Gemma4TextQModel(LlamaQModel):
    """Quantization definition for text-only Gemma 4 checkpoints."""

    # Gemma 4 mixes optional KV projections and per-layer residual adapters across variants.
    layer_modules_strict = False
    # Gemma 4 input preparation uses per-layer embeddings, so batch quantization stays conservative.
    support_batch_quantize = False
    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"
    module_tree = _gemma4_module_tree()

    def capture_first_layer_positional_inputs(self, args, kwargs, batch_device):
        """Keep Gemma 4 per-layer adapter inputs when decoder layers are replayed in isolation."""

        return _capture_gemma4_positional_inputs(self, args, kwargs, batch_device)

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        """Persist Gemma 4 per-layer adapter tensors for later decoder replays."""

        layer_input_kwargs = super().capture_first_layer_input_kwargs(args, kwargs, batch_device, layer_input_kwargs)
        language_model = _resolve_gemma4_language_model(self)
        all_per_layer_inputs = getattr(language_model, "_gptqmodel_cached_all_per_layer_inputs", None)
        if all_per_layer_inputs is not None:
            layer_input_kwargs[_GEMMA4_ALL_PER_LAYER_INPUTS] = move_to(all_per_layer_inputs, device=batch_device)
        return layer_input_kwargs

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Refresh Gemma 4 layer kwargs during cached replay."""

        return _prepare_gemma4_replay_kwargs(self, layer, layer_input, additional_inputs, target_device)

    def pre_quantize_generate_hook_start(self):
        _patch_gemma4_per_layer_input_capture(self)

    def pre_quantize_generate_hook_end(self):
        _restore_gemma4_per_layer_input_capture(self)
        super().pre_quantize_generate_hook_end()


class Gemma4ForConditionalGenerationGPTQ(BaseQModel):
    """Quantization definition for composite Gemma 4 checkpoints."""

    # Gemma 4 composite checkpoints share the same decoder quirks as the text-only model.
    layer_modules_strict = False
    support_batch_quantize = False
    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "q_proj:0",
                "k_norm:!",
                "k_proj:0",
                "v_norm:!",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
            "per_layer_input_gate": ("per_layer_input_gate:0",),
            "post_per_layer_input_norm": ("post_per_layer_input_norm:!",),
            "per_layer_projection": ("per_layer_projection:1",),
        },
    ]

    def capture_first_layer_positional_inputs(self, args, kwargs, batch_device):
        """Keep Gemma 4 per-layer adapter inputs when decoder layers are replayed in isolation."""

        return _capture_gemma4_positional_inputs(self, args, kwargs, batch_device)

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        """Persist Gemma 4 per-layer adapter tensors for later decoder replays."""

        layer_input_kwargs = super().capture_first_layer_input_kwargs(args, kwargs, batch_device, layer_input_kwargs)
        language_model = _resolve_gemma4_language_model(self)
        all_per_layer_inputs = getattr(language_model, "_gptqmodel_cached_all_per_layer_inputs", None)
        if all_per_layer_inputs is not None:
            layer_input_kwargs[_GEMMA4_ALL_PER_LAYER_INPUTS] = move_to(all_per_layer_inputs, device=batch_device)
        return layer_input_kwargs

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Refresh Gemma 4 layer kwargs during cached replay."""

        return _prepare_gemma4_replay_kwargs(self, layer, layer_input, additional_inputs, target_device)

    def pre_quantize_generate_hook_start(self):
        _patch_gemma4_per_layer_input_capture(self)

    def pre_quantize_generate_hook_end(self):
        _restore_gemma4_per_layer_input_capture(self)
        super().pre_quantize_generate_hook_end()
