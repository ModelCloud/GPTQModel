# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


from ..base import BaseQModel


class Phi4MMGPTQ(BaseQModel):
    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": {
                "qkv_proj": {
                    "base_layer": ("base_layer",),
                },
                "o_proj": {
                    "base_layer": ("base_layer",),
                }
            },
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "gate_up_proj": {
                    "base_layer": ("base_layer",),
                },
                "down_proj": {
                    "base_layer": ("base_layer",),
                }
            }
        }
    ]

    require_monkeypatch = True

    def monkey_patch(self):
        if not self.quantized:
            original_forward = self.model.forward

            # patch so input_mode is default to 0 (InputMode.LANGUAGE) if not passed
            # phi4mm default is None which causes quant error as it expects it to be always passed
            def patched_forward(self, **kwargs):
                if "input_mode" not in kwargs:
                    kwargs["input_mode"] = 0
                return original_forward(**kwargs)

            # bind forward to instance
            self.model.forward = patched_forward.__get__(self.model, type(self.model))

__all__ = ["Phi4MMGPTQ"]
