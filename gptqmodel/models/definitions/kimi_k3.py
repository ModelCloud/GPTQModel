# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 routed-expert geometry for bounded EXL3 calibration."""

from typing import Optional

from ..base import BaseQModel
from ..local import LocalModelLoader


_ROUTED_MODULES = {
    "input_layernorm": ("input_layernorm:!",),
    "post_attention_layernorm": ("post_attention_layernorm:!",),
    "self_attention_res_norm": ("self_attention_res_norm:!",),
    "self_attention_res_proj": ("self_attention_res_proj:!",),
    "mlp_res_norm": ("mlp_res_norm:!",),
    "mlp_res_proj": ("mlp_res_proj:!",),
    "self_attn:!": (
        "q_proj:!",
        "k_proj:!",
        "v_proj:!",
        "b_proj:!",
        "q_a_proj:!",
        "q_a_layernorm:!",
        "q_b_proj:!",
        "kv_a_proj_with_mqa:!",
        "kv_a_layernorm:!",
        "kv_b_proj:!",
        "o_proj:!",
        "g_proj:!",
        "f_a_proj:!",
        "f_b_proj:!",
        "g_a_proj:!",
        "g_b_proj:!",
        "q_conv1d:!",
        "k_conv1d:!",
        "v_conv1d:!",
        "o_norm:!",
    ),
    "mlp": ("gate_proj:!", "up_proj:!", "down_proj:!"),
    "block_sparse_moe:moe": {
        "gate": ("gate:!",),
        "routed_expert_down_proj": ("routed_expert_down_proj:!",),
        "routed_expert_up_proj": ("routed_expert_up_proj:!",),
        "routed_expert_norm": ("routed_expert_norm:!",),
        "experts": {"#": ("w1:0:in=latent", "w3:0:in=latent", "w2:1")},
        "shared_experts": ("gate_proj:!", "up_proj:!", "down_proj:!"),
    },
}


class KimiLinearQModel(BaseQModel):
    """Describe native w1/w3/w2 leaves without silently dropping AttnRes state.

    The stock looper forwards one hidden-state tensor between layers. K3 needs
    both the running prefix and the residual bank, including at resume points.
    Quantization therefore uses utils.kimi_k3.KimiK3Source and the explicit
    prepare_moe/finish_moe boundary instead of BaseQModel.quantize().
    """

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.kimi_k3.configuration_kimi_k3.KimiConfigFactory",
        model_class="gptqmodel.models.ports.kimi_k3.modeling_kimi_linear.KimiModelFactory",
        tokenizer_class="gptqmodel.models.ports.kimi_k3.tokenization_kimi.KimiTokenizerFactory",
    )
    layer_modules_strict = False  # first decoder layer is dense
    dynamic_expert_index = "num_experts"
    pre_lm_head_norm_module = "model.norm"
    module_tree = ["model", "layers", "#", _ROUTED_MODULES]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        raise NotImplementedError(
            "Kimi K3 AWQ is unqualified: AttnRes prefix_sum and block_residual require the bounded runner."
        )

    def quantize(self, *args, **kwargs):
        raise NotImplementedError(
            "Kimi K3 quantization requires the bounded KimiK3Source runner with "
            "explicit prefix_sum and block_residual state at every layer. "
            "The ordinary GPTQModel looper cannot preserve AttnRes continuation."
        )

    def pre_quantize(self, module, module_path: Optional[str] = None):
        # Also fail closed when callers instantiate ModuleLooper directly.
        if getattr(module, "use_attn_residuals", False):
            raise NotImplementedError(
                "Kimi K3 AttnRes layers require explicit prefix_sum and "
                "block_residual continuation through the bounded runner."
            )
        return super().pre_quantize(module, module_path=module_path)


class KimiK3QModel(KimiLinearQModel):
    """Bundled multimodal checkpoint; only routed text experts are quantized."""

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.kimi_k3.configuration_kimi_k3.KimiConfigFactory",
        model_class="gptqmodel.models.ports.kimi_k3.modeling_kimi_linear.KimiModelFactory",
        processor_class="gptqmodel.models.ports.kimi_k3.kimi_k3_processor.KimiK3Processor",
        tokenizer_class="gptqmodel.models.ports.kimi_k3.tokenization_kimi.TikTokenTokenizer",
    )
    require_load_processor = True
    module_tree = ["language_model", "model", "layers", "#", _ROUTED_MODULES]
    pre_lm_head_norm_module = "language_model.model.norm"
    out_of_model_tensors = {"prefixes": ["vision_tower.", "mm_projector."]}


__all__ = ["KimiK3QModel", "KimiLinearQModel"]
