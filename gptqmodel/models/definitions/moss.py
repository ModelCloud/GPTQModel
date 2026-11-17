# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class MossQModel(BaseQModel):
    loader = LocalModelLoader(
        "gptqmodel.models.ports.moss.configuration_moss.MossConfig",
        "gptqmodel.models.ports.moss.modeling_moss.MossForCausalLM",
        tokenizer_class="gptqmodel.models.ports.moss.tokenization_moss.MossTokenizer",
    )
    fusable = None

    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "ln_1": ("ln_1:!:nosrc",),
            "attn:!": ("qkv_proj:0", "out_proj:1:nosrc"),
            "mlp": ("fc_in:0:nosrc", "fc_out:1"),
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # The shared norm feeds attention AND GELU MLP. Packed Q/V/K and
        # softmax/rotary are not sequential affine boundaries. Plain AWQ only.
        return []
