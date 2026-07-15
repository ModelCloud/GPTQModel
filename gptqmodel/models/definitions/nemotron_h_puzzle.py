# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...utils.device import get_device
from .nemotron_h import NemotronHQModel


class NemotronHPuzzleQModel(NemotronHQModel):
    """GPTQ definition for NVIDIA Nemotron-Labs-3 Puzzle hybrid MoE models."""
    require_pkgs = ["mamba-ssm"]
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm_f"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "norm": ("norm:!",),
            "mixer:moe": {
                # Puzzle layers are heterogeneous. A mixer is either Mamba,
                # attention, or MoE, so missing paths are intentionally skipped.
                "": (
                    "q_proj:0",
                    "k_proj:0",
                    "v_proj:0",
                    "in_proj:0",
                    "o_proj:1",
                    "out_proj:1",
                ),
                "fc1_latent_proj:2": ("fc1_latent_proj:0",),
                "experts": {
                    "#": ("up_proj:0", "down_proj:1"),
                },
                "fc2_latent_proj:3": ("fc2_latent_proj:0",),
                "shared_experts": ("up_proj:0", "down_proj:1"),
            },
        },
    ]

    @staticmethod
    def _prepare_attention_replay_mask(attention_mask, hidden_states):
        """Build the per-layer causal mask skipped by direct decoder-layer replay."""

        batch_size, sequence_length = hidden_states.shape[:2]
        dtype = hidden_states.dtype
        device = hidden_states.device

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
            if attention_mask.dim() == 4:
                return attention_mask.to(dtype=dtype)
            if attention_mask.dim() != 2:
                raise ValueError(
                    "Nemotron-H Puzzle attention replay expects a 2D padding mask or 4D causal mask, "
                    f"got shape={tuple(attention_mask.shape)}."
                )
            if attention_mask.shape[0] != batch_size:
                raise ValueError(
                    "Nemotron-H Puzzle attention replay mask batch size does not match hidden states: "
                    f"mask={attention_mask.shape[0]}, hidden_states={batch_size}."
                )
            if attention_mask.shape[-1] < sequence_length:
                raise ValueError(
                    "Nemotron-H Puzzle attention replay mask is shorter than hidden states: "
                    f"mask={attention_mask.shape[-1]}, hidden_states={sequence_length}."
                )
            if attention_mask.shape[-1] > sequence_length:
                # Causal language-model tokenizers use left padding; retain the
                # tokens aligned with the replayed hidden-state suffix.
                attention_mask = attention_mask[:, -sequence_length:]

        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, sequence_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            padding_mask = causal_mask.eq(0.0) & attention_mask[:, None, None, :].eq(0.0)
            causal_mask.masked_fill_(padding_mask, min_dtype)

        return causal_mask

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        additional_inputs = super().prepare_layer_replay_kwargs(
            layer=layer,
            layer_input=layer_input,
            additional_inputs=additional_inputs,
            target_device=target_device,
        )
        # NemotronHBlock controls caching through `past_key_values` and does not
        # expose the generic decoder-layer `use_cache` argument.
        additional_inputs.pop("use_cache", None)

        if getattr(layer, "block_type", None) == "attention":
            # Input capture stops at layer 0 (Mamba), so the replay cache holds
            # its 2D padding mask. The full model normally replaces that mask
            # with an attention-specific causal mask before calling this layer.
            mixer_config = getattr(getattr(layer, "mixer", None), "config", None)
            if mixer_config is not None:
                mixer_config._attn_implementation = "eager"
            additional_inputs["attention_mask"] = self._prepare_attention_replay_mask(
                attention_mask=additional_inputs.get("attention_mask"),
                hidden_states=layer_input[0],
            )
        return additional_inputs

    def monkey_patch(self):
        if not self.load_quantized_model:
            return

        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_mamba_2_ssm_available,
        )

        if is_mamba_2_ssm_available():
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update
            from mamba_ssm.ops.triton.ssd_combined import (
                mamba_chunk_scan_combined,
                mamba_split_conv1d_scan_combined,
            )
        else:
            mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, selective_state_update = None, None, None

        if is_causal_conv1d_available():
            from gptqmodel.hf_kernels.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        else:
            causal_conv1d_update, causal_conv1d_fn = None, None

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
            attention_mask=None,
        ):
            if is_fast_path_available and get_device(self.in_proj).type == "cuda":
                return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)

            return self.torch_forward(hidden_states, cache_params, attention_mask)

        for layer in self.model.model.layers:
            if layer.mixer.__class__.__name__ == "NemotronHMamba2Mixer":
                type(layer.mixer).forward = forward
