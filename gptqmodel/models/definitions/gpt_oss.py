# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import torch
import torch.nn.functional as F
from torch import nn

from ..base import BaseQModel


class GptOssExpertsNew(nn.Module):
    def __init__(self, config, ori_experts=None):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.alpha = 1.702
        self.limit = 7.0
        self.quantizing = False

        self.gate_up = nn.ModuleList([
            nn.Linear(self.hidden_size, 2 * self.expert_dim, dtype=config.dtype)
            for _ in range(self.num_experts)
        ])

        self.down = nn.ModuleList([
            nn.Linear(self.expert_dim, self.hidden_size, dtype=config.dtype)
            for _ in range(self.num_experts)
        ])

        if ori_experts is not None:
            self.quantizing = True
            for i in range(self.num_experts):
                tgt_gu_w = self.gate_up[i].weight   # [2E, H]
                tgt_gu_b = self.gate_up[i].bias     # [2E]
                tgt_d_w  = self.down[i].weight      # [H, E]
                tgt_d_b  = self.down[i].bias        # [H]

                gu_w_src = ori_experts.gate_up_proj[i].detach().t().contiguous()
                gu_b_src = ori_experts.gate_up_proj_bias[i].detach()
                d_w_src  = ori_experts.down_proj[i].detach().t().contiguous()
                d_b_src  = ori_experts.down_proj_bias[i].detach()

                with torch.inference_mode():
                    tgt_gu_w.copy_(gu_w_src)
                    tgt_gu_b.copy_(gu_b_src)
                    tgt_d_w.copy_(d_w_src)
                    tgt_d_b.copy_(d_b_src)

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        if self.quantizing:
            # For quantization, we need to trigger computation of all experts
            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
            num_experts = routing_weights.shape[1]

            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
            gate_up = torch.stack([proj(hidden_states[i]) for i, proj in enumerate(self.gate_up)])
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            next_states = torch.stack([proj((up[i] + 1) * glu[i]) for i, proj in enumerate(self.down)])
            next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
            next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            next_states = next_states.sum(dim=0)

            return next_states

        # For non-quantization forward pass, reduce forward pass time by only computing active experts
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1] if len(hidden_states.shape) > 2 else 1
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)

        active_experts = torch.unique(router_indices.flatten())
        final_output = torch.zeros_like(hidden_states)
        for expert_idx in active_experts:
            expert_mask = (router_indices == expert_idx).any(dim=-1)  # (num_tokens,)
            if not expert_mask.any():
                continue

            expert_tokens = hidden_states[expert_mask]  # (selected_tokens, hidden_size)

            gate_up_output = self.gate_up[expert_idx](expert_tokens)  # (selected_tokens, 2*expert_dim)
            gate, up = gate_up_output[..., ::2], gate_up_output[..., 1::2]

            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)

            expert_output = self.down[expert_idx]((up + 1) * glu)  # (selected_tokens, hidden_size)

            expert_weights = routing_weights[expert_mask, expert_idx].unsqueeze(-1)  # (selected_tokens, 1)

            final_output[expert_mask] += expert_output * expert_weights

        if seq_len > 1:
            final_output = final_output.view(batch_size, seq_len, self.hidden_size)
        else:
            final_output = final_output.view(batch_size, self.hidden_size)

        return final_output

class GptOssTopKRouterNew(nn.Module):
    def __init__(self, config, ori_router=None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

        if ori_router is not None:
            with torch.inference_mode():
                self.weight.copy_(ori_router.weight.detach())
                self.bias.copy_(ori_router.bias.detach())

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight.to(hidden_states.dtype), self.bias.to(hidden_states.dtype))  # (seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices

class GPTOSSGPTQ(BaseQModel):
    dynamic_expert_index = "num_local_experts"

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "experts": {
                    "gate_up": {"#": ("#")},
                    "down": {"#": ("#")},
                }
            }
        }
    ]

    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_modeling

            gpt_oss_modeling.GptOssExperts = GptOssExpertsNew
            gpt_oss_modeling.GptOssTopKRouter = GptOssTopKRouterNew

    def after_model_load(self, model, load_quantized_model=False):
        if load_quantized_model:
            return model

        import os
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_modeling
        from transformers.integrations.hub_kernels import use_kernel_forward_from_hub

        @use_kernel_forward_from_hub("MegaBlocksMoeMLP")
        class GptOssMLPNew(nn.Module):
            def __init__(self, config, ori_mlp=None):
                super().__init__()
                self.router = ori_mlp.router
                experts_new = GptOssExpertsNew(config, ori_mlp.experts)
                self.experts = experts_new

            def forward(self, hidden_states):
                router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
                routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
                return routed_out, router_scores

        model = model.to("cpu")
        def process_module(name, module, model, config):
            if isinstance(module, gpt_oss_modeling.GptOssMLP):
                new_module = GptOssMLPNew(config=config, ori_mlp=module)
                parent, child = name.rsplit(".", maxsplit=1)
                parent = model.get_submodule(parent)
                setattr(parent, child, new_module)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            process_fn = partial(process_module, model=model, config=model.config)
            list(executor.map(lambda x: process_fn(x[0], x[1]), model.named_modules()))

        return model
