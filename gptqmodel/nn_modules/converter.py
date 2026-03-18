# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


def _resolve_text_decoder_config(config):
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config

    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        resolved = get_text_config()
        if resolved is not None:
            return resolved

    return config


def _convert_qwen_sparse_moe_layer(
    module,
    *,
    config,
    sparse_moe_cls,
    expert_mlp_cls,
    has_shared_expert: bool,
):
    import torch
    import torch.nn as nn

    from ..utils.hf import no_init_weights

    class SequentialQwenExperts(nn.ModuleList):
        def __init__(self, config, original):
            self.num_experts = original.gate_up_proj.shape[0]
            intermediate_size = original.down_proj.shape[-1]

            with no_init_weights():
                super().__init__([
                    expert_mlp_cls(config, intermediate_size=intermediate_size)
                    for _ in range(self.num_experts)
                ])

            with torch.inference_mode():
                gate_up_batch = original.gate_up_proj.detach()
                down_batch = original.down_proj.detach()
                self.to(device=gate_up_batch.device, dtype=gate_up_batch.dtype)
                target_gate_shape = self[0].gate_proj.weight.shape
                target_down_shape = self[0].down_proj.weight.shape

                if gate_up_batch.shape[-2:] == (target_gate_shape[1], 2 * target_gate_shape[0]):
                    gate_batch = gate_up_batch[:, :, :intermediate_size].transpose(-2, -1).contiguous()
                    up_batch = gate_up_batch[:, :, intermediate_size:].transpose(-2, -1).contiguous()
                elif gate_up_batch.shape[-2:] == (2 * target_gate_shape[0], target_gate_shape[1]):
                    gate_batch = gate_up_batch[:, :intermediate_size, :].contiguous()
                    up_batch = gate_up_batch[:, intermediate_size:, :].contiguous()
                else:
                    raise ValueError(
                        f"Unsupported Qwen fused expert layout: gate_up_proj shape {tuple(gate_up_batch.shape)} "
                        f"cannot map to gate_proj shape {tuple(target_gate_shape)}."
                    )

                if down_batch.shape[-2:] == target_down_shape:
                    down_batch = down_batch.contiguous()
                elif down_batch.shape[-2:] == (target_down_shape[1], target_down_shape[0]):
                    down_batch = down_batch.transpose(-2, -1).contiguous()
                else:
                    raise ValueError(
                        f"Unsupported Qwen fused expert layout: down_proj shape {tuple(down_batch.shape)} "
                        f"cannot map to down_proj shape {tuple(target_down_shape)}."
                    )

                for i in range(self.num_experts):
                    self[i].gate_proj.weight.data.copy_(gate_batch[i])
                    self[i].up_proj.weight.data.copy_(up_batch[i])
                    self[i].down_proj.weight.data.copy_(down_batch[i])

    class SequentialQwenSparseMoeBlock(nn.Module):
        def __init__(self, config, original):
            super().__init__()
            self.hidden_dim = getattr(config, "hidden_size", original.gate.weight.shape[-1])
            self.gate = original.gate
            self.experts = SequentialQwenExperts(config, original.experts)

            if has_shared_expert:
                self.shared_expert = original.shared_expert
                self.shared_expert_gate = original.shared_expert_gate

        def forward(self, hidden_states: torch.Tensor):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)

            _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
            final_hidden_states = hidden_states_reshaped.new_zeros(hidden_states_reshaped.shape)

            for expert_idx, expert in enumerate(self.experts):
                token_indices, top_indices = torch.where(selected_experts == expert_idx)
                if token_indices.numel() == 0:
                    continue

                expert_input = hidden_states_reshaped[token_indices]
                expert_output = expert(expert_input)
                expert_weight = routing_weights[token_indices, top_indices].unsqueeze(-1).to(expert_output.dtype)
                final_hidden_states.index_add_(0, token_indices, expert_output * expert_weight)

            if has_shared_expert:
                shared_expert_output = self.shared_expert(hidden_states_reshaped)
                shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
                final_hidden_states = final_hidden_states + shared_expert_output

            return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    if hasattr(module, "mlp") and isinstance(module.mlp, sparse_moe_cls):
        module.mlp = SequentialQwenSparseMoeBlock(config=config, original=module.mlp)

    return module


def convert_gpt_oss_expert_converter(module, config):
    import torch.nn as nn
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_modeling
    from transformers.integrations.hub_kernels import use_kernel_forward_from_hub

    from ..models.definitions.gpt_oss import GptOssExpertsNew, GptOssTopKRouterNew

    @use_kernel_forward_from_hub("MegaBlocksMoeMLP")
    class GptOssMLPNew(nn.Module):
        def __init__(self, config, ori_mlp=None):
            super().__init__()
            self.router = GptOssTopKRouterNew(config, ori_mlp.router)
            experts_new = GptOssExpertsNew(config, ori_mlp.experts)
            self.experts = experts_new

        def forward(self, hidden_states):
            router_output = self.router(hidden_states)
            if isinstance(router_output, tuple) and len(router_output) == 3:
                _, router_scores, router_indices = router_output
            elif isinstance(router_output, tuple) and len(router_output) == 2:
                router_scores, router_indices = router_output
            else:
                raise ValueError(
                    f"Unexpected GPT-OSS router output during conversion: {type(router_output).__name__}"
                )
            routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
            return routed_out, router_scores

    # loop sub module to replace GptOssMLP with GptOssMLPNew
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, gpt_oss_modeling.GptOssMLP):
            new_module = GptOssMLPNew(config=config, ori_mlp=sub_module)
            setattr(module, name, new_module)

    return module

def convert_llama4_expert_converter(module, config):
    import torch
    from transformers.models.llama4.modeling_llama4 import Llama4TextMLP, Llama4TextMoe

    from ..utils.hf import no_init_weights

    # adapted/modified from https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/llama4.py
    class SequentialLlama4TextExperts(torch.nn.ModuleList):
        def __init__(self, config, original):
            self.num_experts = original.gate_up_proj.shape[0]
            with no_init_weights():
                super().__init__([Llama4TextMLP(config) for _ in range(self.num_experts)])
            intermediate_size = original.down_proj.shape[1]

            with torch.inference_mode():
                # Batch process all expert parameters to avoid loops
                gate_up_batch = torch.stack([original.gate_up_proj[i] for i in range(self.num_experts)])
                down_batch = torch.stack([original.down_proj[i] for i in range(self.num_experts)])

                # Batch split and transpose
                gate_batch = gate_up_batch[:, :, :intermediate_size].transpose(-2, -1).contiguous()
                up_batch = gate_up_batch[:, :, intermediate_size:].transpose(-2, -1).contiguous()
                down_batch = down_batch.transpose(-2, -1).contiguous()

                # Batch assignment
                for i in range(self.num_experts):
                    self[i].gate_proj.weight.data = gate_batch[i]
                    self[i].up_proj.weight.data = up_batch[i]
                    self[i].down_proj.weight.data = down_batch[i]

    class SequentialLlama4TextMoe(torch.nn.Module):
        def __init__(self, config, original):
            super().__init__()
            self.top_k = config.num_experts_per_tok
            self.hidden_dim = config.hidden_size
            self.num_experts = config.num_local_experts
            self.experts = SequentialLlama4TextExperts(config, original.experts)
            self.router = original.router
            self.shared_expert = original.shared_expert

        def forward(self, hidden_states: torch.Tensor):
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = self.router(hidden_states)
            if isinstance(router_logits, tuple):
                router_scores, router_logits = router_logits
                router_scores = router_scores.t()
            else:
                # transformers < 4.54.0 only returns router_logits
                router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)

                router_scores = (
                    torch.full_like(router_logits, float("-inf"))
                    .scatter_(1, router_indices, router_top_value)
                    .transpose(0, 1)
                )
                router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

            out = self.shared_expert(hidden_states)
            for i in range(self.num_experts):
                out += self.experts[i](hidden_states) * router_scores[i].reshape(-1, 1)

            return out, router_logits

    for name, sub_module in module.named_modules():
        if isinstance(sub_module, Llama4TextMoe):
            new_module = SequentialLlama4TextMoe(config=config.get_text_config(), original=sub_module)
            setattr(module, name, new_module)

    return module

MODULE_CONVERTER_MAP = {
    "llama4": convert_llama4_expert_converter,
    "gpt_oss": convert_gpt_oss_expert_converter,
    # qwen2_moe/qwen3_moe/qwen3_next/qwen3_omni_moe are handled by Defuser>=0.0.10 during model load.
}
