# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class ExaOneQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.exaone.configuration_exaone.ExaoneConfig",
        model_class="gptqmodel.models.ports.exaone.modeling_exaone.ExaoneForCausalLM",
    )

    pre_lm_head_norm_module = "transformer.ln_f"

    # Exaone keeps one rotary module on the model (`ExaoneModel.rotary`), not
    # per attention layer. Declaring it lets AWQ rebuild `position_ids` /
    # `position_embeddings` for the sample being replayed instead of reusing
    # another sample's sequence length (EXAONE-3.0 AWQ crashed at 522 vs 430).
    rotary_embedding = "transformer.rotary"

    module_tree= [
        "transformer",
        "h",
        "#",
        {
            "ln_1": ("ln_1:!",),
            "attn": {
                "attention": ("q_proj:0", "k_proj:0", "v_proj:0", "out_proj:1"),
            },
            "ln_2": ("ln_2:!",),
            "mlp": ("c_fc_0:0", "c_fc_1:0", "c_proj:1"),
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Scale-search/replay is unqualified for EXAONE 3; quantize every linear
        # without folding rather than inferring producer relationships.
        return []
