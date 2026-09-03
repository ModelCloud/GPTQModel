# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForImageTextToText

from ...utils.model import move_to
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Glm5NextQModel(BaseQModel):
    """GLM-5.3-Flash hybrid KDA/DSA multimodal MoE model."""

    loader = AutoModelForImageTextToText
    require_load_processor = True
    require_trust_remote_code = False
    layer_modules_strict = False

    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "model.language_model.norm"

    # The checkpoint carries an auxiliary MTP decoder after the 45 inference
    # layers. Transformers intentionally ignores it while loading; preserve it
    # verbatim when a quantized checkpoint is saved.
    out_of_model_tensors = {"prefixes": ["model.language_model.layers.45"]}

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # Quantize Q/K/V/O in KDA, the post-LoRA Q/KV and output projections in DSA,
    # and both dense-MLP and routed-expert projections.
    # KDA state/gate projections, DSA LoRA/indexer projections, routers, shared
    # experts, norms, and hyper-connections remain in the native dtype.
    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                # KDA (linear-attention) layers.
                "q_proj:0:q",
                "k_proj:0:k",
                "v_proj:0:v",
                # DSA layers. The low-rank input and sparse indexer stay dense.
                "q_a_proj:!",
                "kv_a_proj_with_mqa:!",
                "indexer.wq_b:!",
                "indexer.wk:!",
                "indexer.weights_proj:!",
                "q_b_proj:0:q",
                "kv_b_proj:0:k:v",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                # Dense fallback used by the first three decoder layers.
                "": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                "gate": ("gate:!", "e_score_correction_bias:!"),
                "experts:routed:expert_activation=experts._apply_gate": {
                    "#": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                },
                "shared_experts:shared": (
                    "gate_proj:!",
                    "up_proj:!",
                    "down_proj:!",
                ),
            },
        },
    ]

    def update_layer_replay_kwargs_from_output(self, layer, layer_output, layer_input_kwargs, target_device):
        """Pass full DSA selections to subsequent shared-indexer DSA layers."""

        if not isinstance(layer_output, tuple) or len(layer_output) < 2:
            return layer_input_kwargs

        topk_indices = layer_output[1]
        if topk_indices is not None:
            layer_input_kwargs["prev_topk_indices"] = move_to(topk_indices, device=target_device)
        return layer_input_kwargs


__all__ = ["Glm5NextQModel"]
