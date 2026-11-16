# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...utils.logger import setup_logger
from ..base import BaseQModel


log = setup_logger()


class FalconMambaQModel(BaseQModel):
    pre_lm_head_norm_module = "backbone.norm_f"
    require_monkeypatch = True

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

    def monkey_patch(self):
        from transformers.models.falcon_mamba import modeling_falcon_mamba

        from gptqmodel.nn_modules.qlinear import BaseQuantLinear

        # Transformers 5.22 removed the fused ``falcon_mamba_inner_fn`` entry
        # point and the mixer's explicit ``slow_forward`` / ``cuda_kernels_forward``
        # methods. The native mixer now selects the eager path itself, so the
        # legacy replacement must not be installed.
        causal_conv1d_fn = getattr(modeling_falcon_mamba, "causal_conv1d_fn", None)
        causal_conv1d_update = getattr(modeling_falcon_mamba, "causal_conv1d_update", None)
        falcon_mamba_inner_fn = getattr(modeling_falcon_mamba, "falcon_mamba_inner_fn", None)
        is_torchdynamo_compiling = getattr(modeling_falcon_mamba, "is_torchdynamo_compiling", None)
        selective_scan_fn = getattr(modeling_falcon_mamba, "selective_scan_fn", None)
        selective_state_update = getattr(modeling_falcon_mamba, "selective_state_update", None)

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
                and not (is_torchdynamo_compiling is not None and is_torchdynamo_compiling())
            ):
                return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)
            return self.slow_forward(hidden_states, cache_params, attention_mask)

        backbone = getattr(self.model, "backbone", None)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            return

        for layer in layers:
            mixer = getattr(layer, "mixer", None)
            if mixer is None or mixer.__class__.__name__ != "FalconMambaMixer":
                continue
            mixer_class = type(mixer)
            if not all(hasattr(mixer_class, name) for name in ("slow_forward", "cuda_kernels_forward")):
                log.info.once(
                    "FalconMamba mixer does not expose the legacy slow/fast forward methods; "
                    "leaving the upstream forward in place because it selects kernels internally."
                )
                continue
            mixer_class.forward = forward
