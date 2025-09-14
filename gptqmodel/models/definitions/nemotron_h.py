# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..base import BaseGPTQModel


class NemotronHGPTQ(BaseGPTQModel):
    require_monkeypatch = True
    layer_modules_strict = False
    require_pkgs_version = ["transformers<=4.48.3"]

    base_modules = ["backbone.embeddings"]

    layers_node = "backbone.layers"
    layer_type = "NemotronHBlock"

    # TODO: full deprecation by gptqmodel v4.3
    # legacy definition (deprecated): migrate to layers_modules_tree
    layer_modules = [
        ["mixer.k_proj", "mixer.v_proj", "mixer.q_proj"],
        ["mixer.o_proj"],

        ["mixer.in_proj", "mixer.out_proj"],

        ["mixer.gate_proj", "mixer.up_proj"],
        ["mixer.down_proj"],
    ]

    def monkey_patch(self):
        if not self.load_quantized_model:
            return

        from transformers.utils.import_utils import (is_causal_conv1d_available,
                                                     is_flash_attn_2_available, is_mamba_2_ssm_available)

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
