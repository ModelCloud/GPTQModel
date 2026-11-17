# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class InternLM2QModel(BaseQModel):

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.internlm2.configuration_internlm2.InternLM2Config",
        model_class="gptqmodel.models.ports.internlm2.modeling_internlm2.InternLM2ForCausalLM",
    )
    lm_head = "output"
    pre_lm_head_norm_module = "model.norm"

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Packed, KV-group-interleaved QKV has no qualified AWQ fold/search/replay contract.
        return []

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "attention_norm": ("attention_norm:!",),
            "attention": ("wqkv:0", "wo:1"),
            "ffn_norm": ("ffn_norm:!",),
            "feed_forward": ("w1:0", "w3:0", "w2:1"),
        }
    ]
