# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..local import LocalModelLoader
from . import LlamaQModel


class DeciLMQModel(LlamaQModel):
    fusable = None  # No qualified executor for this definition.

    require_trust_remote_code = False
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.decilm.configuration_decilm.DeciLMConfig",
        model_class="gptqmodel.models.ports.decilm.modeling_decilm.DeciLMForCausalLM",
    )
    layer_modules_strict = False  # Some model variants skip modules.

    # NAS can replace either subblock with a single bias-free projection.
    module_tree = ["model", "layers", "#", {
        "input_layernorm": ("input_layernorm:!:norm",),
        "self_attn": ("q_proj:0:q:in=x", "k_proj:0:k:in=x", "v_proj:0:v:in=x", "o_proj:1:o",
                      "linear_attn:0"),
        "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
        "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down", "linear_mlp:0"),
    }]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Heterogeneous/pruned/linear/windowed layouts have not qualified AWQ
        # source/search/replay relationships. Empty groups use plain AWQ.
        return []
