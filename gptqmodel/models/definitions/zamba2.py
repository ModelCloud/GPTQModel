# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ...utils.model import move_to
from ..base import BaseQModel


class Zamba2QModel(BaseQModel):
    layer_modules_strict = False
    require_monkeypatch = True
    require_pkgs = ["mamba_ssm>=2.3.1", "causal_conv1d>=1.2.0"]

    pre_lm_head_norm_module = "lm_head"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "mamba": ("in_proj:0", "out_proj:1"),
            "linear": ("linear:0",),
            "mamba_decoder": {
                "input_layernorm": ("input_layernorm:!",),
                "mamba": ("in_proj:0", "out_proj:1"),
            },
        },
    ]

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        layer_input_kwargs = super().capture_first_layer_input_kwargs(args, kwargs, batch_device, layer_input_kwargs)

        if len(args) > 1 and args[1] is not None:
            layer_input_kwargs["original_hidden_states"] = move_to(args[1], device=batch_device)
        if len(args) > 3 and args[3] is not None:
            layer_input_kwargs["attention_mask"] = move_to(args[3], device=batch_device)
        if len(args) > 4 and args[4] is not None:
            layer_input_kwargs["causal_mask"] = move_to(args[4], device=batch_device)

        return layer_input_kwargs

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        additional_inputs = super().prepare_layer_replay_kwargs(layer, layer_input, additional_inputs, target_device)

        layer_idx = getattr(layer, "layer_idx", None)
        if layer_idx is None:
            layer_idx = getattr(getattr(layer, "mamba_decoder", None), "layer_idx", None)
        if layer_idx is not None:
            additional_inputs["layer_idx"] = layer_idx

        return additional_inputs

    def monkey_patch(self):
        from transformers.models.zamba2 import modeling_zamba2

        from gptqmodel.nn_modules.qlinear import BaseQuantLinear

        is_fast_path_available = all(
            (
                modeling_zamba2.selective_state_update,
                modeling_zamba2.mamba_chunk_scan_combined,
                modeling_zamba2.mamba_split_conv1d_scan_combined,
                modeling_zamba2.causal_conv1d_fn,
                modeling_zamba2.causal_conv1d_update,
            )
        )

        def _has_dense_cuda_weight(module) -> bool:
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
                and _has_dense_cuda_weight(self.in_proj)
                and _has_dense_cuda_weight(self.out_proj)
                and not modeling_zamba2.is_torchdynamo_compiling()
            ):
                return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)

            return self.torch_forward(hidden_states, cache_params, attention_mask)

        for layer in self.model.model.layers:
            candidates = []

            mixer = getattr(layer, "mamba", None)
            if mixer is not None:
                candidates.append(mixer)

            mamba_decoder = getattr(layer, "mamba_decoder", None)
            if mamba_decoder is not None:
                nested_mixer = getattr(mamba_decoder, "mamba", None)
                if nested_mixer is not None:
                    candidates.append(nested_mixer)

            for mixer in candidates:
                if mixer.__class__.__name__ == "Zamba2MambaMixer":
                    type(mixer).forward = forward
