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
            # One LayerNorm feeds attention and the MLP in parallel. The shared
            # `:in=` tag folds it once into both readers.
            "ln_1": ("ln_1:!",),
            "attn:!": ("qkv_proj:0:in=x", "out_proj:1:nosrc"),
            # GELU between fc_in and fc_out is not positively homogeneous.
            "mlp": ("fc_in:0:in=x:nosrc", "fc_out:1"),
        }
    ]
