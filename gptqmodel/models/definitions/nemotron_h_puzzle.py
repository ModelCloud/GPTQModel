# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from ..local import LocalModelLoader
from .nemotron_h import NemotronHQModel


class NemotronHPuzzleQModel(NemotronHQModel):
    """Package-owned heterogeneous Nemotron-Labs-3 Puzzle decoder."""

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.nemotron_h_puzzle.NemotronHPuzzleConfig",
        model_class="gptqmodel.models.ports.nemotron_h_puzzle.NemotronHPuzzleForCausalLM",
    )
    require_trust_remote_code = False
    require_monkeypatch = False
    require_pkgs = []
    fusable = None
    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "model.norm_f"
    out_of_model_tensors = {"prefixes": ["mtp"]}

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "norm": ("norm:!",),
            "mixer:moe": {
                "": ("q_proj:0", "k_proj:0", "v_proj:0", "in_proj:0", "o_proj:1", "out_proj:1",
                     "A_log:!", "D:!", "dt_bias:!"),
                "conv1d": ("conv1d:!",),
                "norm": ("norm:!",),
                "gate": ("gate:!",),
                "fc1_latent_proj:2": ("fc1_latent_proj:0",),
                "experts": {"#": ("up_proj:0:nosrc", "down_proj:1")},
                "fc2_latent_proj:3": ("fc2_latent_proj:0",),
                "shared_experts": ("up_proj:0:nosrc", "down_proj:1"),
            },
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Gated state-space mixing and latent/shared routed branches are not fold-qualified.
        return []

    @classmethod
    def awq_input_feature_aggregation(cls, module_name):
        # The shared mixer prefix also names sequence-sensitive Mamba/attention blocks.
        return None

    @staticmethod
    def _prepare_attention_replay_mask(attention_mask, hidden_states):
        batch_size, sequence_length = hidden_states.shape[:2]
        dtype, device = hidden_states.dtype, hidden_states.device
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
            if attention_mask.dim() == 4:
                if not attention_mask.is_floating_point() or attention_mask.shape[-2:] != (sequence_length, sequence_length):
                    raise ValueError("Puzzle attention replay expects a matching additive 4D causal mask")
                return attention_mask.to(dtype=dtype)
            if attention_mask.dim() != 2:
                raise ValueError("Puzzle attention replay expects a 2D padding mask or 4D causal mask")
            if attention_mask.shape[0] != batch_size:
                raise ValueError("Puzzle replay mask batch size does not match hidden states")
            if attention_mask.shape[-1] < sequence_length:
                raise ValueError("Puzzle replay mask is shorter than hidden states")
            attention_mask = attention_mask[:, -sequence_length:]
        min_dtype = torch.finfo(dtype).min
        causal = torch.full((sequence_length, sequence_length), min_dtype, dtype=dtype, device=device).triu(1)
        causal = causal[None, None].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal = causal.clone().masked_fill(causal.eq(0) & attention_mask[:, None, None].eq(0), min_dtype)
        return causal

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        additional_inputs = super().prepare_layer_replay_kwargs(
            layer=layer, layer_input=layer_input, additional_inputs=additional_inputs, target_device=target_device,
        )
        additional_inputs.pop("use_cache", None)
        if getattr(layer, "block_type", None) == "attention":
            additional_inputs["attention_mask"] = self._prepare_attention_replay_mask(
                additional_inputs.get("attention_mask"), layer_input[0],
            )
        return additional_inputs
