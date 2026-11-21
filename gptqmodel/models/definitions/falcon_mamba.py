# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ..base import BaseQModel
from ..local import LocalModelLoader


class FalconMambaQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )

    pre_lm_head_norm_module = "backbone.norm_f"

    module_tree = [
        "backbone",
        "layers",
        "#",
        {
            "norm": ("norm:!",),
            # Convolution, activation and SSM transforms separate the linears.
            "mixer:!": ("in_proj:0:nosrc", "x_proj:1:nosrc", "out_proj:2:nosrc", "conv1d:3:!:nosrc", "dt_proj:3:!:nosrc"),
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

            # Transformers 5.22 replaced the legacy ``b_c_rms`` / ``dt_rms``
            # buffers with ``FalconMambaWeightlessRMSNorm`` submodules whose
            # ``weight`` buffers are non-persistent, so they never appear in the
            # checkpoint. Rebuild them after the lazy shell is materialized.
            for norm_name, size in (
                ("dt_layernorm", getattr(mixer, "intermediate_size")),
                ("b_layernorm", getattr(mixer, "ssm_state_size")),
                ("c_layernorm", getattr(mixer, "ssm_state_size")),
            ):
                norm = getattr(mixer, norm_name, None)
                if not isinstance(norm, torch.nn.Module):
                    continue
                weight = getattr(norm, "weight", None)
                if isinstance(weight, torch.Tensor) and not getattr(weight, "is_meta", False) and weight.device.type != "meta":
                    continue
                target_dtype = weight.dtype if isinstance(weight, torch.Tensor) else torch.float32
                norm.register_buffer(
                    "weight",
                    torch.ones(size, dtype=target_dtype, device=target_device),
                    persistent=False,
                )

    def after_model_load(self, model, load_quantized_model=False):
        model = super().after_model_load(model, load_quantized_model=load_quantized_model)
        self._restore_falcon_mamba_rms_buffers(model)
        return model

    def pre_quantize_generate_hook_start(self):
        self._restore_falcon_mamba_rms_buffers(self.model)
