# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
from ...utils.model import move_to
from ..base import BaseQModel
from ..local import LocalModelLoader


class ZambaQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )

    layer_modules_strict = False
    require_pkgs = ["mamba_ssm>=2.3.1", "causal_conv1d>=1.2.0"]

    pre_lm_head_norm_module = "model.final_layernorm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "mamba": ("in_proj:0", "out_proj:1", "conv1d:!"),
            "linear": ("linear:0",),
            "mamba_decoder": {
                "input_layernorm": ("input_layernorm:!",),
                "mamba": ("in_proj:0", "out_proj:1", "conv1d:!"),
            },
            # Hybrid layers reuse a shared transformer block. It is
            # checkpoint-owned but its nested projections are not part of the
            # declared quantization plan, so exclude the whole subtree
            # explicitly instead of leaving it undeclared.
            "shared_transf": ("shared_transf:!",),
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
