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


class PanguAlphaQModel(BaseQModel):
    # Non-repeating layers at the root level: same level as `layers_node`
    # Excluding `layers_node`.
    base_modules = ["transformer.wte", "transformer.wpe", "transformer.wqe", "transformer.ln_f"]
    pre_lm_head_norm_module = "transformer.ln_f"

    # Below describes all the repeating layers in this transformer model
    # `model.layers` is a node/module that hold all the repeating layers. The parent node for all n-layers.
    layers_node = ["transformer.h"]

    # Full tree of quantizable modules
    # `#` str will match any number: useful for layers and moe indexing.
    # List[str] for serial linked nodes. List str are linear depth linked modules presented in a linear fashion with no divergence.
    # Dict{str: List[str] | Dict | Tuple[str]} for diverging nodes where a node splits into multiple paths/nodes.
    # Tuple(str) for final targeted modules/nodes: there are only strings representing the final targeted modules
    _layers_modules_tree = [
        "transformer",
        "h",
        "#",
        {
            "attn": ("k_proj:0", "v_proj:0", "q_proj:0", "c_proj:1"),
            "mlp": ("c_fc:0", "c_proj:1"),
        }
    ]
