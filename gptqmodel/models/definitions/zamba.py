# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
from ...utils.logger import setup_logger
from ...utils.model import move_to
from ..base import BaseQModel
from ..local import LocalModelLoader


log = setup_logger()


class ZambaQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )

    layer_modules_strict = False
    require_monkeypatch = True
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

    def monkey_patch(self):
        from ..ports.native.zamba import modeling_zamba

        def forward(
            self,
            hidden_states,
            cache_params=None,
            attention_mask=None,
            **kwargs,
        ):
            is_fast_path_available = all(
                (
                    getattr(modeling_zamba, "selective_state_update", None),
                    getattr(modeling_zamba, "selective_scan_fn", None),
                    getattr(modeling_zamba, "causal_conv1d_fn", None),
                    getattr(modeling_zamba, "causal_conv1d_update", None),
                    getattr(modeling_zamba, "mamba_inner_fn", None),
                )
            )

            if is_fast_path_available and "cuda" in hidden_states.device.type:
                return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask=attention_mask)

            return self.slow_forward(hidden_states, cache_params, attention_mask=attention_mask)

        model_root = getattr(self.model, "model", None)
        layers = getattr(model_root, "layers", None)
        if layers is None:
            return

        for layer in layers:
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
                if mixer.__class__.__name__ == "ZambaMambaMixer":
                    mixer_class = type(mixer)
                    if not all(hasattr(mixer_class, name) for name in ("slow_forward", "cuda_kernels_forward")):
                        log.info.once(
                            "Zamba mixer does not expose the legacy slow/fast forward methods; "
                            "leaving the upstream forward in place because it selects kernels internally."
                        )
                        continue
                    mixer_class.forward = forward
