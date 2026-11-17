# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ...utils.model import get_module
from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class KimiK25QModel(BaseQModel):
    # Kimi-K2.5 wraps a DeepSeek-V3 text backbone with a vision tower and
    # projector. Quantize the language model and keep the vision path in base.
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.kimi_k25.native.Kimi_K25Config",
        model_class="gptqmodel.models.ports.kimi_k25.native.Kimi_K25ForConditionalGeneration",
        processor_class="gptqmodel.models.ports.kimi_k25.native.Kimi_K25Processor",
        tokenizer_class="gptqmodel.models.ports.kimi_k25.tokenization_kimi.TikTokenTokenizer",
    )

    require_trust_remote_code = False

    require_load_processor = True

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    dynamic_expert_index = "n_routed_experts"

    layer_modules_strict = False

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_a_layernorm:!", "kv_a_layernorm:!",
                "q_proj:0:q", "q_a_proj:0:q:in=h", "kv_a_proj_with_mqa:0:k:v:in=h",
                "q_b_proj:1:q:in=q_a", "kv_b_proj:1:k:v:in=kv_a", "o_proj:2:o",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
            },
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # MLA latent norms and routed/shared consumers are not qualified for folding/search.
        return []

    @classmethod
    def get_base_modules(cls, model):
        base_modules = super().get_base_modules(model)
        prefix, core_model = cls._resolve_multimodal_layout(model)
        for name, _ in core_model.named_children():
            if name != "language_model":
                module_name = f"{prefix}.{name}" if prefix else name
                if module_name not in base_modules:
                    base_modules.append(module_name)
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
