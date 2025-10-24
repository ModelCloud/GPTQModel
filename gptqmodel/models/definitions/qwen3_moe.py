# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Optional

import torch

from ...quantization import METHOD
from ...utils.logger import setup_logger
from ..base import BaseQModel

log = setup_logger()


class Qwen3MoeQModel(BaseQModel):
    require_monkeypatch = True

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    # awq scaling optimizations requires some modules within same subset to strictly match the shape of previous module
    # the o_proj must match v_proj or else scaling optimizations are skipped (GQA vs MHA)
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0",  "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "gate": ("gate:!",), # <-- 0.5MB per layer. Not worth quantizing
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        }
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

    def monkey_patch(self):
        import torch
        import torch.nn.functional as F
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeDecoderLayer,
            Qwen3MoeSparseMoeBlock,
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

            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

        Qwen3MoeSparseMoeBlock.forward = _balanced_sparse_moe_forward
        log.debug("Qwen3Moe: Patched Qwen3MoeSparseMoeBlock.forward for device-aware experts.")

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

            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
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

        try:
            from transformers.models.qwen3_moe.modeling_qwen3_moe import Cache  # type: ignore
        except Exception:
            Cache = None  # type: ignore[assignment]

        Qwen3MoeDecoderLayer.forward = _decoder_layer_forward
        log.info("Qwen3Moe: Applied balanced MoE monkey patch for decoder layer and experts.")
        log.info("Qwen3Moe: Applied balanced MoE monkey patch for device-aware expert execution.")
