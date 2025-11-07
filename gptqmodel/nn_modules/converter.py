# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


def convert_gpt_oss_expert_converter(module, config):
    import torch.nn as nn
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_modeling
    from transformers.integrations.hub_kernels import use_kernel_forward_from_hub

    from ..models.definitions.gpt_oss import GptOssExpertsNew

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

    # loop sub module to replace GptOssMLP with GptOssMLPNew
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, gpt_oss_modeling.GptOssMLP):
            new_module = GptOssMLPNew(config=config, ori_mlp=sub_module)
            setattr(module, name, new_module)

    return module

def convert_llama4_expert_converter(module, config):
    import torch
    from transformers.modeling_utils import no_init_weights
    from transformers.models.llama4.modeling_llama4 import Llama4TextMLP, Llama4TextMoe

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
}
