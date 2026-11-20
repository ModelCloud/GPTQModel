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
                # A scale cannot move across the experts' squared ReLU (#5739).
                "experts": {"#": ("up_proj:0:nosrc", "down_proj:1")},
                "fc2_latent_proj:3": ("fc2_latent_proj:0",),
                "shared_experts": ("up_proj:0:nosrc", "down_proj:1"),
            },
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """Generic AWQ groups minus the folds that do not follow the mixer data flow.

        The flat ``mixer`` declaration lets the planner fall back to the block
        ``norm`` for every later projection. Only the projections that read
        ``norm`` directly may receive its scale:

        * ``o_proj`` reads the attention output and ``out_proj`` reads the
          gated-RMSNorm SSM output, never the block norm.
        * With ``moe_latent_size`` the routed experts read the
          ``fc1_latent_proj`` output (a different, narrower tensor).
        * ``fc2_latent_proj`` reads the router-weighted sum of every routed
          expert, so no single expert ``down_proj`` can absorb its scale.

        What remains is ``norm -> q/k/v``, ``norm -> in_proj`` and
        ``norm -> [fc1_latent_proj | routed up_proj] + shared up_proj`` with the
        sigmoid router compensated.
        """

        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        mixer = getattr(module, "mixer", None)
        if mixer is None:
            return nodes
        excluded = [getattr(mixer, name, None) for name in ("o_proj", "out_proj", "fc2_latent_proj")]
        if isinstance(getattr(mixer, "fc1_latent_proj", None), torch.nn.Linear):
            excluded.extend(expert.up_proj for expert in mixer.experts)
        excluded = {id(target) for target in excluded if target is not None}
        kept = []
        for node in nodes:
            layers = [layer for layer in node.get("layers") or [] if id(layer) not in excluded]
            if not layers:
                continue
            if len(layers) != len(node.get("layers") or []):
                node = dict(node, layers=layers)
                search = node.get("_awq_search_layers")
                if search is not None:
                    node["_awq_search_layers"] = [layer for layer in search if id(layer) not in excluded]
            kept.append(node)
        return kept

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
