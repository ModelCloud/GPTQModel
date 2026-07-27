# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from functools import wraps

import torch

from . import LlamaQModel


class NanbeigeQModel(LlamaQModel):
    """Nanbeige's looped decoder reuses a Llama-shaped set of physical layers."""

    require_trust_remote_code = True

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    @staticmethod
    def _restore_rotary_inv_freq(model) -> None:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            return

        for layer in layers:
            attention = getattr(layer, "self_attn", None)
            rotary = getattr(attention, "rotary_emb", None)
            if rotary is None:
                continue

            dim = getattr(rotary, "dim", None)
            base = getattr(rotary, "base", None)
            if not isinstance(dim, int) or dim <= 0 or not isinstance(base, (int, float)) or base <= 0:
                continue

            current = getattr(rotary, "inv_freq", None)
            if isinstance(current, torch.Tensor) and current.device.type != "meta":
                device = current.device
            else:
                weight = getattr(getattr(attention, "q_proj", None), "weight", None)
                weight_device = getattr(weight, "device", None)
                device = weight_device if weight_device is not None and weight_device.type != "meta" else torch.device("cpu")

            inv_freq = 1.0 / (
                float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            rotary.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _patch_generation_inputs(model) -> None:
        model_cls = type(model)
        original = getattr(model_cls, "prepare_inputs_for_generation", None)
        if not callable(original) or getattr(original, "_gptqmodel_nanbeige_cache_position_patch", False):
            return

        @wraps(original)
        def prepare_inputs_for_generation_compat(self, *args, **kwargs):
            model_inputs = original(self, *args, **kwargs)
            if not isinstance(model_inputs, dict):
                return model_inputs

            input_ids = model_inputs.get("input_ids")
            inputs_embeds = model_inputs.get("inputs_embeds")
            if isinstance(input_ids, torch.Tensor):
                sequence_length = input_ids.shape[-1]
            elif isinstance(inputs_embeds, torch.Tensor):
                sequence_length = inputs_embeds.shape[-2]
            else:
                return model_inputs

            for name in ("position_ids", "cache_position"):
                value = model_inputs.get(name)
                if isinstance(value, torch.Tensor) and value.shape[-1] != sequence_length:
                    model_inputs[name] = value[..., -sequence_length:].contiguous()
            return model_inputs

        prepare_inputs_for_generation_compat._gptqmodel_nanbeige_cache_position_patch = True
        model_cls.prepare_inputs_for_generation = prepare_inputs_for_generation_compat

    def after_model_load(self, model, load_quantized_model=False):
        model = super().after_model_load(model, load_quantized_model=load_quantized_model)
        self._restore_rotary_inv_freq(model)
        self._patch_generation_inputs(model)
        return model

    def pre_quantize_generate_hook_start(self):
        self._restore_rotary_inv_freq(self.model)
