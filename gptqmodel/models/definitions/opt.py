# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import copy
from typing import List

import torch

from ...utils.model import get_module_by_name_prefix
from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig


# OPT-125m/1.7B/... apply both norms *before* attention and the MLP.
_PRE_NORM_TREE = [
    "model",
    "decoder",
    "layers",
    "#",
    {
        "self_attn_layer_norm": ("self_attn_layer_norm:!",),
        "self_attn": ("q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x", "out_proj:1"),
        "final_layer_norm": ("final_layer_norm:!",),
        "fc1": ("fc1",),
        "fc2": ("fc2",),
    }
]

# OPT-350m applies the attention norm *after* the residual add (so it feeds the
# MLP, not attention) and the final norm to the block output. The normed value
# is also the MLP's residual, so folding a scale into it changes the block
# output: it cannot be an AWQ scale source.
_POST_NORM_TREE = [
    "model",
    "decoder",
    "layers",
    "#",
    {
        "self_attn": ("q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x", "out_proj:1"),
        "self_attn_layer_norm": ("self_attn_layer_norm:!:nosrc",),
        "fc1": ("fc1",),
        "fc2": ("fc2",),
        "final_layer_norm": ("final_layer_norm:!",),
    }
]


class OptQModel(BaseQModel):
    shared_input_verified_model_types = frozenset({"opt"})

    fusable = FusableConfig("opt").merge(LlamaFusableConfig(), profiles=("qkv",))

    pre_lm_head_norm_module = "model.decoder.final_layer_norm"

    module_tree = _PRE_NORM_TREE

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        """Select the tree that matches OPT's ``do_layer_norm_before`` layout.

        ``do_layer_norm_before=False`` checkpoints (OPT-350m) run the attention
        norm after the residual add, so it feeds ``fc1`` rather than attention,
        and the final norm is applied to the block output. Tree order must match
        that forward data flow or AWQ folds scales across the residual.
        """

        if getattr(getattr(model, "config", None), "do_layer_norm_before", True) is False:
            return copy.deepcopy(_POST_NORM_TREE)
        return super()._resolve_effective_module_tree(model, quantize_config, detector=detector)

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

