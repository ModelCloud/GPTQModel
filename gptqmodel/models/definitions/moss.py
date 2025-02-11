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

from ..base import BaseGPTQModel


class MOSSGPTQ(BaseGPTQModel):
    base_modules = ["transformer.wte", "transformer.ln_f"]
    pre_lm_head_norm_module = "transformer.ln_f"

    layers_node = "transformer.h"
    layer_type = "MossBlock"
    layer_modules = [
        ["attn.qkv_proj"],
        ["attn.out_proj"],
        ["mlp.fc_in"],
        ["mlp.fc_out"],
    ]
