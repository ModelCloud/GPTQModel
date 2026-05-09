# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
from itertools import cycle

from ...utils.model import move_to
from ..base import BaseQModel


class ZambaQModel(BaseQModel):
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
        from transformers.models.zamba import modeling_zamba

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
                if mixer.__class__.__name__ == "ZambaMambaMixer":
                    type(mixer).forward = forward

    # def after_model_load(self, model, load_quantized_model):
    #     print("model", model.model._tied_weights_keys)
    #
    #     zamba_model = model.model
    #     tied_weights_keys = {}
    #     unique_hybrid_blocks = []
    #     for layer_id, layer_type in enumerate(zamba_model.layers_block_type):
    #         prefix_pattern = f"layers.{layer_id}.shared_transf"
    #
    #         # Zamba ties Hybrid module weights by repeating blocks after every
    #         # `num_mem_blocks`. So if `num_mem_blocks=2`, the blocks looks like
    #         # [1, 2, 1, 2, 1, 2] where all "ones" share the same set of weights.
    #         if (
    #                 not isinstance(unique_hybrid_blocks, list)
    #                 or len(unique_hybrid_blocks) >= zamba_model.config.num_mem_blocks
    #         ):
    #             if isinstance(unique_hybrid_blocks, list):
    #                 unique_hybrid_blocks = cycle(unique_hybrid_blocks)
    #             target_pattern = next(unique_hybrid_blocks)
    #             tied_weights_keys.update({prefix_pattern: target_pattern})
    #         else:
    #             # Store source patterns to which the subsequent modules will be tied
    #             unique_hybrid_blocks.append(prefix_pattern)
    #
    #     zamba_model._tied_weights_keys = tied_weights_keys
    #
    #     return model