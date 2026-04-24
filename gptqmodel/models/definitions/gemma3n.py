# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import math
from types import MethodType

import torch

from ...utils.device import get_device
from ...utils.model import get_module_by_name_prefix, move_to, nested_move_to
from ..base import BaseQModel
from . import LlamaQModel


_GEMMA3N_ALL_PER_LAYER_INPUTS = "__gptqmodel_gemma3n_all_per_layer_inputs"


def _gemma3n_decoder_block():
    return {
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
    }


def _set_non_persistent_buffer(module, name, tensor):
    """Replace a non-persistent buffer while preserving its registration semantics."""

    if not isinstance(tensor, torch.Tensor):
        return

    if name not in getattr(module, "_buffers", {}) and hasattr(module, name):
        delattr(module, name)

    if name in getattr(module, "_buffers", {}):
        module._buffers[name] = tensor
        non_persistent = getattr(module, "_non_persistent_buffers_set", None)
        if isinstance(non_persistent, set):
            non_persistent.add(name)
        return

    module.register_buffer(name, tensor, persistent=False)


def _resolve_gemma3n_language_model_from_model(model):
    """Return the text stack regardless of whether the checkpoint is text-only or multimodal."""

    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "model"):
        return model.model
    return model


def _restore_gemma3n_root_scale_buffers(model):
    """Materialize tiny root-level Gemma 3n scale buffers that are not covered by base-module loading."""

    language_model = _resolve_gemma3n_language_model_from_model(model)
    hidden_size = getattr(language_model, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(language_model, "config", None), "hidden_size", None)
    if hidden_size is None:
        return

    scale_specs = {
        "per_layer_projection_scale": hidden_size**-0.5,
        "per_layer_input_scale": 1 / math.sqrt(2.0),
    }

    for name, value in scale_specs.items():
        current = getattr(language_model, name, None)
        if not isinstance(current, torch.Tensor) or current.device.type != "meta":
            continue

        restored = torch.tensor(value, dtype=current.dtype, device="cpu")
        _set_non_persistent_buffer(language_model, name, restored)


def _capture_gemma3n_positional_inputs(model_def, args, kwargs, batch_device):
    """Preserve Gemma 3n decoder positional inputs in their original order."""

    layer_input = super(type(model_def), model_def).capture_first_layer_positional_inputs(args, kwargs, batch_device)

    # Keyword-based invocations already persist extras through layer_input_kwargs.
    if kwargs.get("hidden_states") is not None:
        return layer_input

    if len(args) > 1 and args[1] is not None:
        layer_input.append(nested_move_to(args[1], device=batch_device)) # position_embeddings
    if len(args) > 2 and args[2] is not None:
        layer_input.append(move_to(args[2], device=batch_device)) # per_layer_input
    return layer_input


def _prepare_gemma3n_replay_kwargs(model_def, layer, layer_input, additional_inputs, target_device):
    """Refresh Gemma 3n layer kwargs so replay keeps rotary state and per-layer inputs aligned."""

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
    # Gemma 3n decoder replay may see AltUp-expanded hidden states with layout
    # [num_altup_inputs, batch, seq, hidden], so batch/sequence are not the first
    # two dimensions. Fallbacks keep plain [batch, seq, hidden] and degenerate
    # tensor shapes working for tests and direct layer calls.
    if hidden_states.dim() >= 4:
        batch_dim = hidden_states.shape[1]
        seq_len = hidden_states.shape[2]
    elif hidden_states.dim() >= 2:
        batch_dim = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
    else:
        batch_dim = 1
        seq_len = hidden_states.shape[0]

    position_ids = additional_inputs.get("position_ids")
    if position_ids is None or position_ids.shape[-1] != seq_len:
        position_ids = (
            torch.arange(seq_len, device=target_device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_dim, -1)
        )
        additional_inputs["position_ids"] = position_ids

    try:
        rotary_device = get_device(rotary)
    except Exception:
        rotary_device = position_ids.device

    # If replay only cached hidden_states, rebuild rotary embeddings from
    # position_ids. When position_embeddings were captured positionally, keep the
    # explicit layer input and avoid injecting a duplicate kwarg copy.
    if len(layer_input) <= 1:
        rotary_position_ids = move_to(position_ids, device=rotary_device)
        rotary_input = torch.empty(1, device=rotary_device, dtype=hidden_states.dtype)
        additional_inputs["position_embeddings"] = nested_move_to(
            rotary(rotary_input, rotary_position_ids, layer_type),
            device=target_device,
        )
    else:
        additional_inputs.pop("position_embeddings", None)

    # Gemma 3n can also pass per_layer_input positionally. Only synthesize the
    # per-layer slice from the cached full tensor when replay did not already
    # preserve that positional argument.
    if len(layer_input) <= 2:
        all_per_layer_inputs = additional_inputs.pop(_GEMMA3N_ALL_PER_LAYER_INPUTS, None)
        layer_index = getattr(getattr(layer, "self_attn", None), "layer_idx", None)
        if all_per_layer_inputs is not None and layer_index is not None:
            additional_inputs["per_layer_input"] = move_to(
                all_per_layer_inputs[:, :, layer_index, :],
                device=target_device,
            )
    else:
        additional_inputs.pop(_GEMMA3N_ALL_PER_LAYER_INPUTS, None)
        additional_inputs.pop("per_layer_input", None)

    return additional_inputs


def _resolve_gemma3n_language_model(model_def):
    """Return the Gemma 3n text stack that owns per-layer input projection state."""

    return _resolve_gemma3n_language_model_from_model(model_def.model)


def _patch_gemma3n_per_layer_input_capture(model_def):
    """Capture projected Gemma 3n per-layer inputs during calibration replay staging."""

    language_model = _resolve_gemma3n_language_model(model_def)
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


def _restore_gemma3n_per_layer_input_capture(model_def):
    """Restore Gemma 3n per-layer input capture hooks after calibration input caching."""

    language_model = _resolve_gemma3n_language_model(model_def)
    original = getattr(language_model, "_gptqmodel_original_project_per_layer_inputs", None)
    if original is not None:
        language_model.project_per_layer_inputs = original
        delattr(language_model, "_gptqmodel_original_project_per_layer_inputs")
    if hasattr(language_model, "_gptqmodel_project_per_layer_inputs_patched"):
        delattr(language_model, "_gptqmodel_project_per_layer_inputs_patched")
    if hasattr(language_model, "_gptqmodel_cached_all_per_layer_inputs"):
        delattr(language_model, "_gptqmodel_cached_all_per_layer_inputs")


class Gemma3nTextQModel(LlamaQModel):
    layer_modules_strict = False
    support_batch_quantize = False
    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"
    module_tree = [
        "model",
        "layers",
        "#",
        _gemma3n_decoder_block(),
    ]

    def after_model_load(self, model, load_quantized_model=False):
        model = super().after_model_load(model, load_quantized_model=load_quantized_model)
        _restore_gemma3n_root_scale_buffers(model)
        return model

    def capture_first_layer_positional_inputs(self, args, kwargs, batch_device):
        """Keep Gemma 3n decoder positional inputs available for first-layer replay."""

        return _capture_gemma3n_positional_inputs(self, args, kwargs, batch_device)

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        """Persist Gemma 3n projected per-layer inputs for later decoder replays."""

        layer_input_kwargs = super().capture_first_layer_input_kwargs(args, kwargs, batch_device, layer_input_kwargs)
        language_model = _resolve_gemma3n_language_model(self)
        all_per_layer_inputs = getattr(language_model, "_gptqmodel_cached_all_per_layer_inputs", None)
        if all_per_layer_inputs is not None:
            layer_input_kwargs[_GEMMA3N_ALL_PER_LAYER_INPUTS] = move_to(all_per_layer_inputs, device=batch_device)
        return layer_input_kwargs

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Refresh Gemma 3n layer kwargs during cached replay."""

        return _prepare_gemma3n_replay_kwargs(self, layer, layer_input, additional_inputs, target_device)

    def pre_quantize_generate_hook_start(self):
        _restore_gemma3n_root_scale_buffers(self.model)
        _patch_gemma3n_per_layer_input_capture(self)

    def pre_quantize_generate_hook_end(self):
        _restore_gemma3n_per_layer_input_capture(self)
        super().pre_quantize_generate_hook_end()


class Gemma3nForConditionalGenerationGPTQ(BaseQModel):
    layer_modules_strict = False
    support_batch_quantize = False
    require_load_processor = True
    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        _gemma3n_decoder_block(),
    ]

    def after_model_load(self, model, load_quantized_model=False):
        model = super().after_model_load(model, load_quantized_model=load_quantized_model)
        _restore_gemma3n_root_scale_buffers(model)
        return model

    def capture_first_layer_positional_inputs(self, args, kwargs, batch_device):
        """Keep Gemma 3n decoder positional inputs available for first-layer replay."""

        return _capture_gemma3n_positional_inputs(self, args, kwargs, batch_device)

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        """Persist Gemma 3n projected per-layer inputs for later decoder replays."""

        layer_input_kwargs = super().capture_first_layer_input_kwargs(args, kwargs, batch_device, layer_input_kwargs)
        language_model = _resolve_gemma3n_language_model(self)
        all_per_layer_inputs = getattr(language_model, "_gptqmodel_cached_all_per_layer_inputs", None)
        if all_per_layer_inputs is not None:
            layer_input_kwargs[_GEMMA3N_ALL_PER_LAYER_INPUTS] = move_to(all_per_layer_inputs, device=batch_device)
        return layer_input_kwargs

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Refresh Gemma 3n layer kwargs during cached replay."""

        return _prepare_gemma3n_replay_kwargs(self, layer, layer_input, additional_inputs, target_device)

    def pre_quantize_generate_hook_start(self):
        _restore_gemma3n_root_scale_buffers(self.model)
        _patch_gemma3n_per_layer_input_capture(self)

    def pre_quantize_generate_hook_end(self):
        _restore_gemma3n_per_layer_input_capture(self)
        super().pre_quantize_generate_hook_end()

__all__ = ["Gemma3nForConditionalGenerationGPTQ", "Gemma3nTextQModel"]
