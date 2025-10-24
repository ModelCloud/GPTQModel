# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Optional

import torch
from ...quantization import METHOD
from ...quantization.config import VRAMStrategy
from ...utils.logger import setup_logger
from ..base import BaseQModel

log = setup_logger()


class Qwen3NextGPTQ(BaseQModel):
    require_monkeypatch = False
    supported_vram_strategies = [VRAMStrategy.EXCLUSIVE, VRAMStrategy.BALANCED]
    """
    GPTQ config for Qwen3-Next (HF: Qwen3Next*), supporting:
      - Mixed token mixers per layer: 'full_attention' (self_attn.*) and 'linear_attention' (linear_attn.*)
      - Dense MLP (Qwen3NextMLP) and Sparse MoE (Qwen3NextSparseMoeBlock)
      - Dynamic expert indexing via config.num_experts
    """

    layer_modules_strict = False

    pre_lm_head_norm_module = "model.norm"

    dynamic_expert_index = "num_experts"

    # -----------------------------------------------------------------------------
    # Preferred modern hierarchical spec. The loader will gracefully skip any
    # subpaths that don't exist on a given layer (e.g., dense vs MoE, or mixer type).
    # -----------------------------------------------------------------------------
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # Token mixers
            #"self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "linear_attn": ("in_proj_qkvz", "in_proj_ba:!", "out_proj"),  # conv1d intentionally excluded
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            # MLP / MoE
            "mlp": {
                # MoE router + shared expert (Qwen3NextSparseMoeBlock)
                "gate": ("gate:!",),  # router gate linear
                "shared_expert_gate": ("shared_expert_gate:!",), # <-- single (1, N) logic projections should not be quantized
                "shared_expert": ("gate_proj", "up_proj", "down_proj"),

                # Experts list with dynamic index
                "experts": {
                    "#": ("gate_proj", "up_proj", "down_proj"),
                },
            },
        },
    ]

    module_tree_overrides = {
        METHOD.AWQ: [
            {
                "mlp": {
                    "gate": ("gate",),
                }
            }
        ]
    }

    def monkey_patch(self) -> None:
        strategy = getattr(self.quantize_config, "vram_strategy", None)
        if strategy is None:
            return
        if isinstance(strategy, str):
            try:
                strategy = VRAMStrategy(strategy.lower())
            except ValueError:
                return
        if strategy != VRAMStrategy.BALANCED:
            log.debug("Qwen3Next: VRAM strategy %s does not require monkey patch.", strategy)
            return

        import torch.nn.functional as F
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextDecoderLayer,
            Qwen3NextSparseMoeBlock,
        )

        def _balanced_sparse_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            base_device = hidden_states.device
            hidden_states = hidden_states.view(-1, hidden_dim)

            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(dtype=hidden_states.dtype, device=base_device)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=base_device,
            )

            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.num_experts
            ).permute(2, 1, 0)

            expert_hits = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)
            for expert_idx_tensor in expert_hits:
                expert_idx = int(expert_idx_tensor.item())
                expert_layer = self.experts[expert_idx]

                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                if top_x.numel() == 0:
                    continue

                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

                target_device = getattr(expert_layer, "target_device", None)
                if target_device is None:
                    try:
                        target_device = next(expert_layer.parameters()).device
                    except StopIteration:
                        buffers = list(expert_layer.buffers())
                        target_device = buffers[0].device if buffers else base_device
                if target_device is None:
                    target_device = base_device

                if current_state.device != target_device:
                    current_state = current_state.to(device=target_device)

                expert_output = expert_layer(current_state)
                if isinstance(expert_output, tuple):
                    expert_output = expert_output[0]

                if expert_output.device != base_device:
                    expert_output = expert_output.to(device=base_device)

                weights = routing_weights[top_x, idx].unsqueeze(-1).to(device=base_device)
                scaled_output = expert_output * weights

                final_hidden_states.index_add_(0, top_x.to(device=base_device), scaled_output.to(hidden_states.dtype))

            if hasattr(self, "shared_expert") and self.shared_expert is not None:
                shared_input = hidden_states
                shared_target = getattr(self.shared_expert, "target_device", None)
                if shared_target is None:
                    try:
                        shared_target = next(self.shared_expert.parameters()).device
                    except StopIteration:
                        buffers = list(self.shared_expert.buffers())
                        shared_target = buffers[0].device if buffers else base_device
                if shared_target is None:
                    shared_target = base_device

                if shared_target != base_device:
                    self.shared_expert = self.shared_expert.to(device=base_device)
                    if hasattr(self.shared_expert, "gate_proj"):
                        setattr(self.shared_expert.gate_proj, "target_device", base_device)
                    if hasattr(self.shared_expert, "up_proj"):
                        setattr(self.shared_expert.up_proj, "target_device", base_device)
                    if hasattr(self.shared_expert, "down_proj"):
                        setattr(self.shared_expert.down_proj, "target_device", base_device)
                    shared_target = base_device

                if shared_input.device != shared_target:
                    shared_input = shared_input.to(device=shared_target)

                shared_output = self.shared_expert(shared_input)
                if isinstance(shared_output, tuple):
                    shared_output = shared_output[0]
                if shared_output.device != base_device:
                    shared_output = shared_output.to(device=base_device)

                shared_gate = getattr(self, "shared_expert_gate", None)
                if shared_gate is not None:
                    if getattr(shared_gate, "weight", None) is not None:
                        gate_device = shared_gate.weight.device
                    else:
                        gate_device = base_device
                    if gate_device != base_device:
                        self.shared_expert_gate = shared_gate.to(device=base_device)
                        gate_device = base_device
                    gate_values = self.shared_expert_gate(hidden_states.to(device=base_device))
                    shared_scaled = torch.sigmoid(gate_values) * shared_output
                else:
                    shared_scaled = shared_output

                final_hidden_states = final_hidden_states + shared_scaled.to(hidden_states.dtype)

            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

        Qwen3NextSparseMoeBlock.forward = _balanced_sparse_moe_forward

        def _decoder_layer_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> torch.FloatTensor:
            residual = hidden_states
            residual_device = residual.device

            hidden_states = self.input_layernorm(hidden_states)

            if self.layer_type == "linear_attention":
                hidden_states = self.linear_attn(
                    hidden_states=hidden_states,
                    cache_params=past_key_values,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )
            elif self.layer_type == "full_attention":
                hidden_states, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            if hidden_states.device != residual_device:
                hidden_states = hidden_states.to(residual_device)
            hidden_states = residual + hidden_states

            residual = hidden_states
            residual_device = residual.device

            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states, _ = hidden_states
            if hidden_states.device != residual_device:
                hidden_states = hidden_states.to(residual_device)
            hidden_states = residual + hidden_states

            return hidden_states

        Qwen3NextDecoderLayer.forward = _decoder_layer_forward

        log.info("Qwen3Next: Applied balanced MoE monkey patch for decoder layer and experts.")
