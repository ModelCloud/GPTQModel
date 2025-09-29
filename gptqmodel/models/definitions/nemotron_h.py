# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class NemotronHQModel(BaseQModel):
    require_trust_remote_code = True
    require_monkeypatch = True
    layer_modules_strict = False
    require_pkgs_version = ["transformers<=4.48.3"]

    module_tree = [
        "backbone",
        "layers",
        "#",
        {
            "norm": ("norm:!",),
            "mixer": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1", "in_proj:2", "out_proj:2", "gate_proj:3", "up_proj:3", "down_proj:4"),
        }
    ]

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
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

            return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)

        for layer in self.model.backbone.layers:
            if layer.mixer.__class__.__name__ == "NemotronHMamba2Mixer":
                mixer_class = type(layer.mixer)
                mixer_class.forward = forward
