# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .nemotron_h import NemotronHQModel
from .._const import CPU
from ...utils.model import move_to
from ...utils.offload import offload_to_disk


class NemotronOmniQModel(NemotronHQModel):
    """GPTQ wrapper for Nemotron Omni models that embed a Nemotron-H language backbone."""

    pre_lm_head_norm_module = "language_model.backbone.norm_f"

    module_tree = [
        "language_model",
        "backbone",
        "layers",
        "#",
        {
            "norm": ("norm:!",),
            "mixer": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
                "in_proj:2",
                "out_proj:2",
                "gate_proj:3",
                "up_proj:3",
                "down_proj:4",
            ),
        }
    ]

    @classmethod
    def get_base_modules(cls, model):
        base_modules = super().get_base_modules(model)
        for name, _ in model.named_children():
            if name != "language_model":
                base_modules.append(name)
        return base_modules

    @staticmethod
    def _has_multimodal_inputs(kwargs) -> bool:
        return any(
            kwargs.get(name) is not None
            for name in ("pixel_values", "pixel_values_videos", "sound_clips", "sound_length")
        )

    def _materialize_top_level_module(self, attr_name: str):
        module = getattr(self.model, attr_name, None)
        if module is None:
            return
        if "_turtle_lock" not in self.__dict__ and "shell_module_materialize" not in self.__dict__:
            setattr(self.model, attr_name, move_to(module, device=self.quantize_config.device))
            return
        setattr(
            self.model,
            attr_name,
            self.shell_module_materialize(module, self.quantize_config.device),
        )

    def pre_quantize_generate_hook_start(self):
        for attr_name in ("vision_model", "mlp1", "sound_encoder", "sound_projection"):
            if hasattr(self.model, attr_name):
                self._materialize_top_level_module(attr_name)

        self.shell_module_materialize(self.model.language_model.backbone.norm_f, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            for attr_name in ("vision_model", "mlp1", "sound_encoder", "sound_projection"):
                module = getattr(self.model, attr_name, None)
                if module is not None:
                    offload_to_disk(
                        model=self.model,
                        module=module,
                        disk_path=self.quantize_config.offload_to_disk_path,
                    )

            offload_to_disk(
                model=self.model.language_model.backbone,
                module=self.model.language_model.backbone.norm_f,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        for attr_name in ("vision_model", "mlp1", "sound_encoder", "sound_projection"):
            module = getattr(self.model, attr_name, None)
            if module is not None:
                setattr(self.model, attr_name, move_to(module, device=CPU))

        self.model.language_model.backbone.norm_f = move_to(self.model.language_model.backbone.norm_f, device=CPU)

    def forward(self, *args, **kwargs):
        # Text-only quantization/eval paths should bypass the multimodal wrapper,
        # whose `forward()` requires image tensors even when no visual inputs exist.
        if self._has_multimodal_inputs(kwargs):
            return self.model(*args, **kwargs)
        return self.model.language_model(*args, **kwargs)

    def run_input_capture(self, example, use_cache: bool, data_device):
        if self._has_multimodal_inputs(example):
            return super().run_input_capture(example, use_cache=use_cache, data_device=data_device)
        return self.model.language_model(**example, use_cache=use_cache)

    def monkey_patch(self):
        if not self.load_quantized_model:
            return

        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_attn_2_available,
            is_mamba_2_ssm_available,
        )

        if is_mamba_2_ssm_available():
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
        else:
            mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, selective_state_update = None, None, None

        if is_causal_conv1d_available():
            from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        else:
            causal_conv1d_update, causal_conv1d_fn = None, None

        if is_flash_attn_2_available():
            pass

        is_fast_path_available = all(
            (
                selective_state_update,
                mamba_chunk_scan_combined,
                mamba_split_conv1d_scan_combined,
                causal_conv1d_fn,
                causal_conv1d_update,
            )
        )

        def forward(
                self,
                hidden_states,
                cache_params=None,
                cache_position=None,
                attention_mask=None,
        ):
            if is_fast_path_available and "cuda" in self.in_proj.qweight.device.type:
                return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
            dtype = hidden_states.dtype
            if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

            return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)

        for layer in self.model.language_model.backbone.layers:
            if layer.mixer.__class__.__name__ == "NemotronHMamba2Mixer":
                mixer_class = type(layer.mixer)
                mixer_class.forward = forward
