# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader


class GptJQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )

    shared_input_verified_model_types = frozenset({"gptj"})

    fusable = FusableConfig("gptj").merge(LlamaFusableConfig(), profiles=("qkv",))

    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            # GPT-J runs attention and the MLP in parallel from ``ln_1``; the
            # attention output projection is added to the residual, and the MLP
            # output projection follows the GELU.
            "ln_1": ("ln_1:!:nosrc",),
            "attn": ("q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x", "out_proj:1:nosrc"),
            "mlp": ("fc_in:0:nosrc", "fc_out:1"),
        }
    ]
