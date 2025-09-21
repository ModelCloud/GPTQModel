# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class HymbaQModel(BaseQModel):
    supports_desc_act = [False]
    require_trust_remote_code = True
    require_monkeypatch = True
    require_pkgs_version = ["tiktoken>=0.7.0",
                            "sentencepiece>=0.2.0",
                            "protobuf>=5.28.3",
                            "ninja>=1.11.1.1",
                            "einops>=0.8.0",
                            "mamba_ssm>=2.2.2",
                            "causal_conv1d>=1.4.0",
                            "attn_gym>=0.0.3.dev5"]

    pre_lm_head_norm_module = "model.final_layernorm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "mamba": ("in_proj:0", "out_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "moe": {
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
