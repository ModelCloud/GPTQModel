# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...utils.device import get_device
from ...utils.model import get_module_by_name_prefix, move_to, nested_move_to
from ..base import BaseQModel


def _prepare_olmo3_replay_kwargs(model_def, layer, layer_input, additional_inputs, target_device):
    """Refresh the RoPE tuple for each OLMo 3 attention type during layer replay."""

    rotary_path = getattr(model_def, "rotary_embedding", None)
    if not rotary_path or not layer_input:
        return additional_inputs

    rotary, _ = get_module_by_name_prefix(model_def.model, [rotary_path])
    if rotary is None:
        return additional_inputs

    attention_type = getattr(getattr(layer, "self_attn", None), "attention_type", None)
    if attention_type is None:
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
        rotary(rotary_input, rotary_position_ids, attention_type),
        device=target_device,
    )

    return additional_inputs


class Olmo3QModel(BaseQModel):
    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    # The output projection only shares an AWQ scale when its input shape matches V.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": (
                "q_proj:0:q",
                "q_norm:0:!",
                "k_proj:0:k",
                "k_norm:0:!",
                "v_proj:0:v",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        },
    ]

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        return _prepare_olmo3_replay_kwargs(self, layer, layer_input, additional_inputs, target_device)


__all__ = ["Olmo3QModel"]
