# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...utils.device import get_device
from ...utils.model import get_module_by_name_prefix, move_to, nested_move_to
from ..base import BaseQModel


def _prepare_gemma4_unified_replay_kwargs(model_def, layer, layer_input, additional_inputs, target_device):
    """Refresh Gemma 4 unified rotary kwargs per layer during cached replay.

    Gemma 4 unified builds one rope (cos, sin) per attention ``layer_type`` and hands each
    decoder layer the single tuple for its own type, so replaying a layer in isolation needs
    that tuple regenerated for the layer's ``layer_type`` (sliding vs full). This mirrors the
    Gemma 4 rope replay but without any per-layer-input handling, which this variant lacks.
    """

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

    return additional_inputs


# Shared decoder layout for both the composite (``gemma4_unified``) and text-only
# (``gemma4_unified_text``) variants: the standard Gemma sandwich-norm decoder with
# per-projection q/k/v norms. Unlike the per-layer-input Gemma 4 variants there is no
# per-layer input gate/projection, so those module-tree entries are intentionally absent.
_GEMMA4_UNIFIED_DECODER = {
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
}


class Gemma4UnifiedForConditionalGenerationGPTQ(BaseQModel):
    """Quantization definition for Gemma 4 unified (multimodal) checkpoints.

    Gemma 4 unified reuses the composite decoder layout (per-projection q/k/v norms and the
    dual pre/post feed-forward norms) but, unlike the per-layer-input Gemma 4 variants, has no
    per-layer input gate/projection, so those module-tree entries and the per-layer-input
    capture hooks are intentionally absent. The sliding/full rope boundaries still need the
    per-layer replay refresh below.
    """

    layer_modules_strict = False
    support_batch_quantize = False
    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        _GEMMA4_UNIFIED_DECODER,
    ]

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Refresh Gemma 4 unified rope kwargs during cached layer replay."""

        return _prepare_gemma4_unified_replay_kwargs(self, layer, layer_input, additional_inputs, target_device)


class Gemma4UnifiedTextQModel(BaseQModel):
    """Quantization definition for text-only Gemma 4 unified checkpoints (``gemma4_unified_text``).

    ``Gemma4UnifiedTextModel`` is the standalone language stack of the unified family, so its
    decoder layers live at ``model.layers`` (no ``language_model`` nesting) with the norm and
    rotary embedding at ``model.norm`` / ``model.rotary_emb``. The quantizable layout is identical
    to the composite variant, mirroring how ``gemma4_text`` parallels ``gemma4``.
    """

    layer_modules_strict = False
    support_batch_quantize = False
    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    module_tree = [
        "model",
        "layers",
        "#",
        _GEMMA4_UNIFIED_DECODER,
    ]

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Refresh Gemma 4 unified rope kwargs during cached layer replay."""

        return _prepare_gemma4_unified_replay_kwargs(self, layer, layer_input, additional_inputs, target_device)
