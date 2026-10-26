# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from ...utils.model import nested_move_to
from ..base import BaseQModel


class JambaQModel(BaseQModel):
    require_pkgs = ["mamba-ssm"]

    # Jamba interleaves Mamba and attention decoder layers inside one `layers`
    # list, so a decoder layer owns either `mamba` or `self_attn`, never both.
    layer_modules_strict = False

    pre_lm_head_norm_module = "model.final_layernorm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "mamba": ("in_proj:0", "out_proj:1"),
            "pre_ff_layernorm": ("pre_ff_layernorm:!",),
            "feed_forward": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    @classmethod
    def attention_layer_replay_mask(cls, config, hidden_states, attention_mask, position_ids):
        """Rebuild the full-attention causal mask `JambaModel.forward` passes to attention layers."""

        from transformers.masking_utils import create_causal_mask

        return create_causal_mask(
            config=config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        additional_inputs = super().prepare_layer_replay_kwargs(layer, layer_input, additional_inputs, target_device)

        # Calibration captures the first (Mamba) layer kwargs, whose 2D padding mask is
        # not a full-attention mask: JambaAttention would silently run bidirectional
        # attention instead of causal attention. Rebuild the mask that
        # JambaModel.forward passes to attention decoder layers.
        if (
            getattr(layer, "self_attn", None) is not None
            and layer_input
            and torch.is_tensor(layer_input[0])
        ):
            additional_inputs["attention_mask"] = nested_move_to(
                self.attention_layer_replay_mask(
                    config=self.model.config,
                    hidden_states=layer_input[0],
                    attention_mask=additional_inputs.get("attention_mask"),
                    position_ids=additional_inputs.get("position_ids"),
                ),
                device=target_device,
            )

        return additional_inputs
