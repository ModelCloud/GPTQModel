# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from .nemotron_h import NemotronHQModel


class Nemotron3NanoQModel(NemotronHQModel):
    pre_lm_head_norm_module = "backbone.norm_f"
    require_pkgs_version = ["transformers>=4.55.0"]
    dynamic_expert_index = "n_routed_experts"
    awq_scale_optimize_shape_dependent_modules = ["mixer.o_proj"]

    module_tree = [
        "backbone",
        "layers",
        "#",
        {
            "norm": ("norm:!",),
            "mixer": {
                "": (
                    "q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1",
                    "in_proj:0", "out_proj:1",
                    "up_proj:0", "down_proj:1",
                ),
                "shared_experts": ("up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("up_proj:0", "down_proj:1"),
                },
            },
        }
    ]
