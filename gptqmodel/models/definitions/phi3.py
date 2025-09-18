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
from ..base import BaseQModel


class Phi3QModel(BaseQModel):
    pre_lm_head_norm_module = "model.norm"

    layers_modules_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("qkv_proj:0", "o_proj:1"),
            "mlp": ("gate_up_proj:0", "down_proj:1"),
        }
    ]

class PhiMoEGPTQForCausalLM(BaseQModel):
    require_pkgs_version = ["transformers<=4.44.2"]

    layers_modules_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "block_sparse_moe": {
                "experts": {
                    "#": ("w1:0", "w2:1"),
                },
            },
        }
    ]

__all__ = ["Phi3QModel", "PhiMoEGPTQForCausalLM"]
