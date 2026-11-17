# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class HymbaQModel(BaseQModel):
    supports_desc_act = [False]
    require_trust_remote_code = True
    require_monkeypatch = True
    write_shared_kv_cache = True
    require_pkgs = ["tiktoken>=0.7.0",
                            "sentencepiece>=0.2.0",
                            "protobuf>=5.28.3",
                            "ninja>=1.11.1.1",
                            "einops>=0.8.0",
                            "mamba_ssm>=2.2.2",
                            "causal_conv1d>=1.4.0",
                            "attn_gym>=0.0.3.dev5"]

    pre_lm_head_norm_module = "model.final_layernorm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # The conv/SSM path between `in_proj` and `out_proj` is not
            # homogeneous, so `in_proj` must never become a fold source:
            # folding a scale into it would scale that path's output. Same
            # guard as Falcon-H1's mixer (#4674). The mixer's dense leaves are
            # declared `:!` so the fold source resolution cannot skip over
            # them and land on `in_proj`.
            "mamba": ("in_proj:0:nosrc", "out_proj:1", "conv1d:!", "x_proj.0:!", "dt_proj.0:!"),
            # Hymba's MoE reads `pre_moe_layernorm`; it has no
            # `post_attention_layernorm`. Declaring a norm the layer does not
            # own left the routed-expert gate/up block without a producer of
            # its own, so it was folded into `input_layernorm` and every
            # layer's experts were scaled by the wrong norm (#4891).
            "pre_moe_layernorm": ("pre_moe_layernorm:!",),
            "moe:moe": {
                "experts": {
                    "0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                }
            }
        }
    ]

    def monkey_patch(self):
        if hasattr(self.config, 'conv_dim'):
            new_conv_dim = {}
            try:
                for k, v in self.config.conv_dim.items():
                    if isinstance(k, str):
                        new_conv_dim[int(k)] = v
                self.config.conv_dim = new_conv_dim
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError("The key of HymbaConfig.conv_dim should be a string of numbers.")
