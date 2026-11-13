# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 routed-expert geometry for bounded EXL3 calibration."""
from ..base import BaseQModel


_ROUTED_MODULES = {
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
    require_trust_remote_code = True
    layer_modules_strict = False  # first decoder layer is dense
    dynamic_expert_index = "num_experts"
    pre_lm_head_norm_module = "model.norm"
    module_tree = ["model", "layers", "#", _ROUTED_MODULES]

    def quantize(self, *args, **kwargs):
        raise NotImplementedError(
            "Kimi K3 quantization requires the bounded KimiK3Source runner with "
            "explicit prefix_sum and block_residual state at every layer. "
            "The ordinary GPTQModel looper cannot preserve AttnRes continuation."
        )

    def pre_quantize(self, module):
        # Also fail closed when callers instantiate ModuleLooper directly.
        if getattr(module, "use_attn_residuals", False):
            raise NotImplementedError(
                "Kimi K3 AttnRes layers require explicit prefix_sum and "
                "block_residual continuation through the bounded runner."
            )
        return super().pre_quantize(module)


class KimiK3QModel(KimiLinearQModel):
    """Bundled multimodal checkpoint; only routed text experts are quantized."""
    module_tree = ["language_model", "model", "layers", "#", _ROUTED_MODULES]
    pre_lm_head_norm_module = "language_model.model.norm"
    out_of_model_tensors = {"prefixes": ["vision_tower.", "mm_projector."]}


__all__ = ["KimiK3QModel", "KimiLinearQModel"]
