# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ...utils.model import get_module
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class KimiK25QModel(BaseQModel):
    # Kimi-K2.5 wraps a DeepSeek-V3 text backbone with a vision tower and
    # projector. Quantize the language model and keep the vision path in base.
    require_trust_remote_code = True

    require_load_processor = True

    pre_lm_head_norm_module = "language_model.model.norm"

    dynamic_expert_index = "n_routed_experts"

    layer_modules_strict = False

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "language_model",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "q_a_proj:0", "kv_a_proj_with_mqa:0", "q_b_proj:1", "kv_b_proj:1", "o_proj:2"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    @classmethod
    def get_base_modules(cls, model):
        prefix, core_model = cls._resolve_multimodal_layout(model)
        base_modules = []
        for name, _ in core_model.named_children():
            if name != "language_model":
                base_modules.append(f"{prefix}.{name}" if prefix else name)
        return base_modules

    @classmethod
    def _resolve_multimodal_layout(cls, model):
        for prefix in ("model", ""):
            core_model = get_module(model, prefix) if prefix else model
            if core_model is None:
                continue
            if hasattr(core_model, "language_model"):
                return prefix, core_model
        raise AttributeError("Unable to resolve Kimi-K2.5 core model with a `language_model` module.")


__all__ = ["KimiK25QModel"]
