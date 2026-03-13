# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ..base import BaseQModel


class BaiChuanQModel(BaseQModel):
    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("W_pack:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    def after_model_load(self, model, load_quantized_model=False):
        model = super().after_model_load(model, load_quantized_model=load_quantized_model)

        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            return model

        for layer in layers:
            rotary = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
            if rotary is None:
                continue

            inv_freq = getattr(rotary, "inv_freq", None)
            if not isinstance(inv_freq, torch.Tensor) or inv_freq.device.type != "meta":
                continue

            dim = inv_freq.numel() * 2
            base = getattr(rotary, "base", 10000)
            max_seq_len = getattr(rotary, "max_seq_len_cached", None)
            if max_seq_len is None:
                max_seq_len = getattr(rotary, "max_position_embeddings", 2048)

            inv_freq_cpu = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            t = torch.arange(max_seq_len, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq_cpu)
            emb = torch.cat((freqs, freqs), dim=-1)

            rotary.inv_freq = inv_freq_cpu
            rotary.cos_cached = emb.cos()[None, None, :, :].to(torch.float32)
            rotary.sin_cached = emb.sin()[None, None, :, :].to(torch.float32)

        return model
