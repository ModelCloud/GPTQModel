# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ..base import BaseQModel
from ..local import LocalModelLoader


class BaiChuanQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.baichuan.configuration_baichuan.BaichuanConfig",
        model_class="gptqmodel.models.ports.baichuan.modeling_baichuan.BaichuanForCausalLM",
    )
    fusable = None  # Fused QKV; no qualified fusion executor.
    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("W_pack:0:q:k:v", "o_proj:1", "rotary_emb:!"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Plain quantization only. Fused QKV plus RoPE/ALiBi replay has not
        # passed the model-specific AWQ fold/search qualification gate.
        return []

    @staticmethod
    def _set_non_persistent_buffer(module, name, tensor):
        if not isinstance(tensor, torch.Tensor):
            return

        if name not in getattr(module, "_buffers", {}) and hasattr(module, name):
            delattr(module, name)

        if name in getattr(module, "_buffers", {}):
            module._buffers[name] = tensor
            non_persistent = getattr(module, "_non_persistent_buffers_set", None)
            if isinstance(non_persistent, set):
                non_persistent.add(name)
            return

        module.register_buffer(name, tensor, persistent=False)

    @staticmethod
    def _build_rotary_cache(inv_freq, max_seq_len):
        inv_freq = inv_freq.to(dtype=torch.float32)
        t = torch.arange(max_seq_len, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (
            emb.cos()[None, None, :, :].to(torch.float32),
            emb.sin()[None, None, :, :].to(torch.float32),
        )

    def after_model_load(self, model, load_quantized_model=False):
        model = super().after_model_load(model, load_quantized_model=load_quantized_model)

        from ..ports.baichuan.modeling_baichuan import BaichuanForCausalLM

        if isinstance(model, BaichuanForCausalLM):
            # Local RoPE is derived per call; Baichuan 1's inv_freq remains
            # checkpoint-owned and must not be demoted to a runtime buffer.
            return model

        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            return model

        for layer in layers:
            rotary = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
            if rotary is None:
                continue

            inv_freq = getattr(rotary, "inv_freq", None)
            max_seq_len = getattr(rotary, "max_seq_len_cached", None)
            if max_seq_len is None:
                max_seq_len = getattr(rotary, "max_position_embeddings", 2048)

            if not isinstance(inv_freq, torch.Tensor) or inv_freq.device.type == "meta":
                if not isinstance(inv_freq, torch.Tensor):
                    continue
                dim = inv_freq.numel() * 2
                base = getattr(rotary, "base", 10000)
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
                )
                cos_cached, sin_cached = self._build_rotary_cache(inv_freq, max_seq_len)
            else:
                inv_freq = inv_freq.to(dtype=torch.float32)
                cos_cached = getattr(rotary, "cos_cached", None)
                sin_cached = getattr(rotary, "sin_cached", None)
                if (
                    not isinstance(cos_cached, torch.Tensor)
                    or not isinstance(sin_cached, torch.Tensor)
                    or cos_cached.device.type == "meta"
                    or sin_cached.device.type == "meta"
                ):
                    cos_cached, sin_cached = self._build_rotary_cache(inv_freq, max_seq_len)
                else:
                    cos_cached = cos_cached.to(dtype=torch.float32)
                    sin_cached = sin_cached.to(dtype=torch.float32)

            rotary.max_seq_len_cached = max_seq_len
            self._set_non_persistent_buffer(rotary, "inv_freq", inv_freq)
            self._set_non_persistent_buffer(rotary, "cos_cached", cos_cached)
            self._set_non_persistent_buffer(rotary, "sin_cached", sin_cached)

        return model
