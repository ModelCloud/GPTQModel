# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class BrumbyQModel(BaseQModel):
    require_trust_remote_code = True
    require_pkgs_version = ["retention>=1.0.7"]

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:0:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "g_proj:0:!",
                "o_proj:1",
                "q_norm:0:!",
                "k_norm:0:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:1:!",),
            "mlp": (
                "gate_proj:0",
                "up_proj:0",
                "down_proj:1",
            ),
        },
    ]

    def after_model_load(self, model, load_quantized_model=False):
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "use_cache"):
            generation_config.use_cache = False
        return model
