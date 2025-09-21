# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from typing import List

import torch

from ...utils.model import get_module_by_name_prefix
from ..base import BaseQModel


class OptQModel(BaseQModel):
    pre_lm_head_norm_module = "model.decoder.final_layer_norm"

    module_tree = [
        "model",
        "decoder",
        "layers",
        "#",
        {
            "final_layer_norm": ("final_layer_norm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "out_proj:1"),
            "fc1": ("fc1",),
            "fc2": ("fc2",),
        }
    ]

    def lm_head_pre_quantize_generate_hook(self, inputs: List[List[torch.tensor]]) -> List[List[torch.tensor]]:
        if self.config.do_layer_norm_before and not self.config._remove_final_layer_norm:
            inputs = super().lm_head_pre_quantize_generate_hook(inputs)

        project_out, _ = get_module_by_name_prefix(self.model, ["model.decoder.project_out"])
        if project_out is not None:
            project_out = self.pre_quantize(project_out)

            for element in inputs:
                for i in range(len(element)):
                    element[i] = project_out(element[i])

            self.post_quantize(project_out)

        return inputs

