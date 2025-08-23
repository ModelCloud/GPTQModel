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
from ..base import BaseGPTQModel


class GLM4MoEGPTQ(BaseGPTQModel):
    # allow dynamic expert index for layer_modules so we don't need to write out 128 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_experts"

    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = ["model.layers"]
    layer_type = "GLM4MoEDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],

        # MoE components for layers 1-46 (46 layers total with experts)
        ["mlp.shared_experts.up_proj", "mlp.shared_experts.gate_proj"],
        ["mlp.shared_experts.down_proj"],
        ["mlp.gate"],
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.up_proj", f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj"],
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.down_proj"],
        
        # Standard MLP for first layer (layer 0)
        ["mlp.down_proj"],
        ["mlp.gate_proj"],
        ["mlp.up_proj"],
    ]

    layers_modules_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "mlp": {
                "shared_experts": ("up_proj", "gate_proj", "down_proj"),
                "gate": ("gate",),
                "experts": {
                    "#": ("up_proj", "gate_proj", "down_proj"),
                },
                # Standard MLP components for first layer
                "down_proj": ("down_proj",),
                "gate_proj": ("gate_proj",),
                "up_proj": ("up_proj",),
            },
        }
    ]