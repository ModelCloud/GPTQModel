# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

from ...utils.attn_mask import normalize_seq_mask
from ...utils.model import move_to
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


_PADDING_MASK_KEY = "_solar_open2_padding_mask"
_CAPTURE_PADDING_MASK_ATTR = "_solar_open2_capture_padding_mask"


class SolarOpen2QModel(BaseQModel):
    """GPTQ definition for Solar Open 2's hybrid-attention MoE decoder."""

    layer_modules_strict = False
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    # Full-attention GQA projections do not all have matching output shapes.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # Solar Open 2 stores both full attention and Kimi-Delta linear
            # attention under `self_attn`. Missing paths are skipped per layer.
            #
            # Keep the small/accuracy-sensitive KDA decay, beta, output-gate,
            # and depthwise-convolution parameters dense. The large Q/K/V/O
            # projections are shared by both attention variants, while g_proj
            # is the full-attention output gate in the released checkpoint.
            "self_attn": (
                "q_proj:0",
                "q_norm:!",
                "k_proj:0",
                "k_norm:!",
                "v_proj:0",
                "g_proj:0",
                "o_norm:!",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                # Dense fallback for configurations with leading dense layers.
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!",),
                # Match SolarOpen2MoE.forward(): routed experts run before the
                # shared expert contribution is added.
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    def run_input_capture(self, example, use_cache: bool, data_device):
        """Expose the original 2D mask while the first-layer hook is active."""

        previous_mask = self.__dict__.get(_CAPTURE_PADDING_MASK_ATTR)
        attention_mask = example.get("attention_mask")
        setattr(
            self,
            _CAPTURE_PADDING_MASK_ATTR,
            attention_mask if torch.is_tensor(attention_mask) and attention_mask.ndim == 2 else None,
        )
        try:
            return super().run_input_capture(example, use_cache=use_cache, data_device=data_device)
        finally:
            setattr(self, _CAPTURE_PADDING_MASK_ATTR, previous_mask)

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        """Retain the 2D padding signal hidden inside layer 0's causal mask."""

        layer_input_kwargs = super().capture_first_layer_input_kwargs(
            args,
            kwargs,
            batch_device,
            layer_input_kwargs,
        )
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        seq_len = hidden_states.shape[1] if torch.is_tensor(hidden_states) and hidden_states.ndim >= 2 else None
        padding_mask = self.__dict__.get(_CAPTURE_PADDING_MASK_ATTR)
        captured_mask = kwargs.get("attention_mask")
        if not torch.is_tensor(padding_mask) and torch.is_tensor(captured_mask):
            padding_mask = normalize_seq_mask(captured_mask, seq_len=seq_len)
        if padding_mask is not None:
            layer_input_kwargs[_PADDING_MASK_KEY] = move_to(padding_mask, device=batch_device)
        return layer_input_kwargs

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Rebuild the mask expected by each full- or linear-attention layer."""

        additional_inputs = super().prepare_layer_replay_kwargs(
            layer,
            layer_input,
            additional_inputs,
            target_device,
        )
        if not layer_input or not torch.is_tensor(layer_input[0]):
            return additional_inputs

        self_attn = getattr(layer, "self_attn", None)
        layer_config = getattr(self_attn, "config", None)
        if layer_config is None:
            return additional_inputs

        padding_mask = additional_inputs.pop(_PADDING_MASK_KEY, None)
        if not torch.is_tensor(padding_mask):
            captured_mask = additional_inputs.get("attention_mask")
            padding_mask = (
                normalize_seq_mask(captured_mask, seq_len=layer_input[0].shape[1])
                if torch.is_tensor(captured_mask)
                else None
            )

        mask_kwargs = {
            "config": layer_config,
            "inputs_embeds": layer_input[0],
            "attention_mask": padding_mask,
            "past_key_values": additional_inputs.get("past_key_values"),
            "position_ids": additional_inputs.get("position_ids"),
        }
        if getattr(layer, "layer_type", None) == "linear_attention":
            attention_mask = create_recurrent_attention_mask(**mask_kwargs)
        else:
            attention_mask = create_causal_mask(
                **mask_kwargs,
                layer_idx=getattr(self_attn, "layer_idx", None),
            )
        additional_inputs["attention_mask"] = attention_mask
        return additional_inputs


__all__ = ["SolarOpen2QModel"]
