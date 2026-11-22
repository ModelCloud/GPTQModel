# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class CodeGenQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.codegen.configuration_codegen.CodeGenConfig",
        model_class="gptqmodel.models.ports.codegen.modeling_codegen.CodeGenForCausalLM",
    )

    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            # CodeGen runs attention and the MLP in parallel from ``ln_1`` (there
            # is no post-attention norm). The tree used to omit the norm, so AWQ
            # could not build a group at all; the shared norm, the residual
            # crossing, attention QKV rearrangement, and the GELU are declared
            # non-sources below. AWQ uses plain quantization for both layouts.
            "ln_1": ("ln_1:!:nosrc",),
            "attn": ("qkv_proj:0:nosrc", "out_proj:1:nosrc"),
            "mlp": ("fc_in:0:nosrc", "fc_out:1"),
        }
    ]
