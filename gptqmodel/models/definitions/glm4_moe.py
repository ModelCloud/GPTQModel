# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .._const import EXPERT_INDEX_PLACEHOLDER
from ..base import BaseQModel


class GLM4MoEGPTQ(BaseQModel):
    # GLM-4.5-Air MoE Model Structure:
    # Layer 0: Standard MLP (no MoE experts) - handled by ["mlp.down_proj"], ["mlp.gate_proj"], ["mlp.up_proj"]
    # Layers 1-46: MoE with shared_experts and individual experts (128 experts total) - handled by MoE components
    # Layer 46: Additional special structure with expanded parameters (embed_tokens, shared_head, eh_proj, etc.)
    #   This is handled dynamically through layer_modules_strict = False
    #
    # allow dynamic expert index for layer_modules so we don't need to write out 128 layers here
    # config.n_routed_experts contains the actual expert count used for index
    dynamic_expert_index = "n_routed_experts"

    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    # Set to False since GLM-4.5-Air may have dynamic module structures
    layer_modules_strict = False

    layers_node = ["model.layers"]

    _layers_modules_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("k_proj:0", "v_proj:0", "q_proj:0", "o_proj:1"),
            "mlp": {
                "shared_experts": {
                    "up_proj": ("up_proj:0",),
                    "gate_proj": ("gate_proj:0",),
                    "down_proj": ("down_proj:1",),
                },
                "gate": ("gate:!",),
                "experts": {
                    "#": ("up_proj:0", "gate_proj:0", "down_proj:1"),
                },
                # Standard MLP components for layer 0
                "down_proj": ("down_proj",),
                "gate_proj": ("gate_proj",),
                "up_proj": ("up_proj",),
            },
        }
    ]
