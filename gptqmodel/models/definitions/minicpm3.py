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


class MiniCPM3GPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens",]
    pre_lm_head_norm_module = "model.norm"

    layers_node = "model.layers"
    layer_type = "MiniCPM3DecoderLayer"
    layer_modules = [
        ["self_attn.q_a_proj","self_attn.kv_a_proj_with_mqa"],
        ["self_attn.q_b_proj","self_attn.kv_b_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj","mlp.up_proj"],
        ["mlp.down_proj"],
    ]
