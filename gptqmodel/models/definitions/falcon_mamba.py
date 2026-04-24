# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ..base import BaseQModel


class FalconMambaQModel(BaseQModel):
    pre_lm_head_norm_module = "backbone.norm_f"
    require_monkeypatch = True

    module_tree = [
        "backbone",
        "layers",
        "#",
        {
            "norm": ("norm:!",),
            "mixer": ("in_proj:0", "x_proj:0", "out_proj:1"),
        },
    ]

    def _restore_falcon_mamba_rms_buffers(self, model) -> None:
        backbone = getattr(model, "backbone", None)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            return

        for layer in layers:
            mixer = getattr(layer, "mixer", None)
            if mixer is None or mixer.__class__.__name__ != "FalconMambaMixer":
                continue

            target_device = getattr(getattr(mixer, "D", None), "device", None)
            if target_device is None or target_device.type == "meta":
                target_device = torch.device("cpu")

            def _restore_buffer(name: str, size: int) -> None:
                buffer = getattr(mixer, name, None)
                if isinstance(buffer, torch.Tensor) and not getattr(buffer, "is_meta", False) and buffer.device.type != "meta":
                    return

                target_dtype = buffer.dtype if isinstance(buffer, torch.Tensor) else torch.float32
                # FalconMambaMixer registers these as fixed all-ones non-persistent buffers.
                # LazyTurtle's template fallback cannot rebuild them offline because
                # FalconMambaMixer.__init__() lazily loads hub kernels during construction.
                mixer.register_buffer(
                    name,
                    torch.ones(size, dtype=target_dtype, device=target_device),
                    persistent=False,
                )

            _restore_buffer("b_c_rms", getattr(mixer, "ssm_state_size"))
            _restore_buffer("dt_rms", getattr(mixer, "intermediate_size"))

    def after_model_load(self, model, load_quantized_model=False):
        model = super().after_model_load(model, load_quantized_model=load_quantized_model)
        self._restore_falcon_mamba_rms_buffers(model)
        return model

    def pre_quantize_generate_hook_start(self):
        self._restore_falcon_mamba_rms_buffers(self.model)

    def monkey_patch(self):
        from gptqmodel.nn_modules.qlinear import BaseQuantLinear
        from transformers.models.falcon_mamba.modeling_falcon_mamba import (
            causal_conv1d_fn,
            causal_conv1d_update,
            falcon_mamba_inner_fn,
            is_torchdynamo_compiling,
            selective_scan_fn,
            selective_state_update,
        )

        is_fast_path_available = all(
            (
                selective_state_update,
                selective_scan_fn,
                causal_conv1d_fn,
                causal_conv1d_update,
                falcon_mamba_inner_fn,
            )
        )

        def _has_dense_cuda_weight(module) -> bool:
            # FalconMambaMixer.cuda_kernels_forward() passes raw dense weights from
            # x_proj/dt_proj/out_proj into the fused kernel. Quantized replacements
            # expose qweight or metadata shims instead, so they must stay on the
            # eager slow path.
            if isinstance(module, BaseQuantLinear):
                return False

            weight = getattr(module, "weight", None)
            device = getattr(weight, "device", None)
            return device is not None and "cuda" in device.type

        def forward(
            self,
            hidden_states,
            cache_params=None,
            attention_mask=None,
            **kwargs,
        ):
            if (
                is_fast_path_available
                and _has_dense_cuda_weight(self.x_proj)
                and _has_dense_cuda_weight(self.dt_proj)
                and _has_dense_cuda_weight(self.out_proj)
                and not is_torchdynamo_compiling()
            ):
                return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)
            return self.slow_forward(hidden_states, cache_params, attention_mask)

        for layer in self.model.backbone.layers:
            if layer.mixer.__class__.__name__ == "FalconMambaMixer":
                mixer_class = type(layer.mixer)
                mixer_class.forward = forward
