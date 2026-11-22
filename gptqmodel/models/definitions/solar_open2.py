# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

from ...utils.attn_mask import normalize_seq_mask
from ...utils.model import move_to
from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


_PADDING_MASK_KEY = "_solar_open2_padding_mask"
_CAPTURE_PADDING_MASK_ATTR = "_solar_open2_capture_padding_mask"


class SolarOpen2QModel(BaseQModel):
    """GPTQ definition for Solar Open 2's hybrid-attention MoE decoder."""

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.solar_open2.configuration_solar_open2.SolarOpen2Config",
        model_class="gptqmodel.models.ports.solar_open2.modeling_solar_open2.SolarOpen2ForCausalLM",
    )
    fusable = None

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
                "A_log:!",
                "dt_bias:!",
                "b_proj:!",
                "f_proj:!",
                "f_a_proj:!",
                "f_b_proj:!",
                "g_a_proj:!",
                "g_b_proj:!",
                "q_conv1d:!",
                "k_conv1d:!",
                "v_conv1d:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                # Dense fallback for configurations with leading dense layers.
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!", "e_score_correction_bias:!"),
                # Match SolarOpen2MoE.forward(): routed experts run before the
                # shared expert contribution is added.
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """Generic groups with two tree-position corrections.

        Both attention variants end their Q/K/V block with KDA-only ``:!``
        leaves, so on a full-attention layer the tree position falls back to
        ``input_layernorm`` as ``o_proj``'s source whenever the attention width
        equals the hidden size. ``o_proj`` reads the (gated) attention output,
        never the normed residual, so only that consumer is removed.

        The dense-prefix MLP shares its gate/up block with the absent shared
        expert, which hides ``mlp.up_proj`` as ``mlp.down_proj``'s source. The
        SwiGLU product is linear in ``up_proj``'s output, so that group is
        restored explicitly.
        """

        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        attention = getattr(module, "self_attn", None)
        o_proj, v_proj = getattr(attention, "o_proj", None), getattr(attention, "v_proj", None)
        kept = []
        for node in nodes:
            layers = node.get("layers") or []
            if o_proj is not None and node.get("prev_op") is not v_proj and any(layer is o_proj for layer in layers):
                layers = [layer for layer in layers if layer is not o_proj]
                if not layers:
                    continue
                node = dict(node, layers=layers)
                search = node.get("_awq_search_layers")
                if search is not None:
                    node["_awq_search_layers"] = [layer for layer in search if layer is not o_proj]
            kept.append(node)

        mlp = getattr(module, "mlp", None)
        up_proj, down_proj = getattr(mlp, "up_proj", None), getattr(mlp, "down_proj", None)
        feature = input_feat.get("mlp.down_proj") if isinstance(input_feat, dict) else None
        if (
            isinstance(up_proj, nn.Linear)
            and isinstance(down_proj, nn.Linear)
            and up_proj.out_features == down_proj.in_features
            and torch.is_tensor(feature)
            and feature.numel() > 0
            and not any(any(layer is down_proj for layer in node.get("layers") or []) for node in kept)
        ):
            kwargs = dict(module_kwargs or {})
            feature_kwargs = kwargs.pop("_awq_feature_kwargs", None)
            if isinstance(feature_kwargs, dict) and isinstance(feature_kwargs.get("mlp.down_proj"), dict):
                kwargs.update(feature_kwargs["mlp.down_proj"])
            kept.append({
                "prev_op": up_proj,
                "layers": [down_proj],
                "inp": feature,
                "kwargs": kwargs,
                "_input_feature_name": "mlp.down_proj",
            })
        return kept

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
