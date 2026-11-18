# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

from ..base import BaseQModel
from ..local import LocalModelLoader
from ..looped_capture import LoopedLayerCaptureMixin


class NanbeigeQModel(LoopedLayerCaptureMixin, BaseQModel):
    """Physical Nanbeige layers are shared across logical loop/cache slots."""

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.nanbeige.configuration_nanbeige.NanbeigeConfig",
        model_class="gptqmodel.models.ports.nanbeige.modeling_nanbeige.NanbeigeForCausalLM",
    )
    require_trust_remote_code = False
    fusable = None
    pre_lm_head_norm_module = "model.norm"
    modules_with_direct_meta_tensors = ["model"]

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
        },
    ]

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector=detector)
        config = getattr(model, "config", None)
        if config is None:
            return tree
        if getattr(config, "qk_layernorm", False):
            tree[-1]["self_attn"] = tuple(dict.fromkeys((*tree[-1]["self_attn"], "q_layernorm:!", "k_layernorm:!")))
        if getattr(config, "enable_hyper_connection", False):
            for name in ("self_attn_hc", "mlp_hc"):
                tree[-1][name] = (f"{name}:!", "mapping_proj:!")
            if config.mhc_diff_for_loop:
                count = (config.num_hidden_layers + config.loop_middle_layers) // config.loop_middle_layers
                for name in ("self_attn_mhc_loop_hcs", "mlp_mhc_loop_hcs"):
                    tree[-1][name] = {str(index): (f"{index}:!", "mapping_proj:!") for index in range(count - 1)}
        return tree

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Looped replay and optional cross-depth/HC consumers have no qualified fold/search boundary.
        return []
