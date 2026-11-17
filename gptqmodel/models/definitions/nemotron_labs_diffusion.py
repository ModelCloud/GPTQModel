# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ..base import BaseQModel
from ..local import LocalModelLoader


class NemotronLabsDiffusionQModel(BaseQModel):
    require_trust_remote_code = False
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        model_class="gptqmodel.models.ports.nemotron_labs_diffusion.NemotronLabsDiffusionModel",
    )

    # Bidirectional denoising and causal verification have not qualified fold/search replay.
    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        return []

    lm_head = "diffusion_head"
    pre_lm_head_norm_module = "encoder.norm"

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    # The tri-mode decoder is checkpoint-owned under encoder.layers.
    module_tree = [
        "encoder",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]
