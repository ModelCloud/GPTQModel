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
from typing import List

import torch

from ...utils.model import get_module_by_name_prefix
from ..base import BaseGPTQModel


class OPTGPTQ(BaseGPTQModel):
    base_modules = [
        "model.decoder.embed_tokens",
        "model.decoder.embed_positions",
        "model.decoder.project_out",
        "model.decoder.project_in",
        "model.decoder.final_layer_norm",
    ]
    pre_lm_head_norm_module = "model.decoder.final_layer_norm"

    layers_node = "model.decoder.layers"
    layer_type = "OPTDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        ["fc1"],
        ["fc2"],
    ]

    def lm_head_pre_quantize_generate_hook(self, inputs: List[List[torch.tensor]]) -> List[List[torch.tensor]]:
        if self.config.do_layer_norm_before and not self.config._remove_final_layer_norm:
            inputs = super().lm_head_pre_quantize_generate_hook(inputs)

        project_out = get_module_by_name_prefix(self.model, "model.decoder.project_out")
        if project_out is not None:
            self.pre_quantize(project_out)

            for element in inputs:
                for i in range(len(element)):
                    element[i] = project_out(element[i])

            self.post_quantize(project_out)

        return inputs

