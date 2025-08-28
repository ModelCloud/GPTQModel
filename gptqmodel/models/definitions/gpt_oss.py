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

from .._const import EXPERT_INDEX_PLACEHOLDER
from ..base import BaseGPTQModel
from logbar import LogBar

logger = LogBar.shared()


class GPTOSSGPTQ(BaseGPTQModel):
    dynamic_expert_index = "num_local_experts"

    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = ["model.layers"]
    layer_type = "GptOssDecoderLayer"
    layer_modules = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],

        # uses dynamic_expert_index
        [f"mlp.experts.gate_up_projs.{EXPERT_INDEX_PLACEHOLDER}"],
        [f"mlp.experts.down_projs.{EXPERT_INDEX_PLACEHOLDER}"],
    ]

    def after_model_load(self, model, load_quantized_model):
        import torch
        from torch import nn
        from torch.nn import functional as F
        from transformers.integrations.hub_kernels import use_kernel_forward_from_hub
        import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_modeling

        class GptOssExpertsNew(nn.Module):
            def __init__(self, config, ori_experts=None):
                super().__init__()
                self.intermediate_size = config.intermediate_size
                self.num_experts = config.num_local_experts
                self.hidden_size = config.hidden_size
                self.expert_dim = self.intermediate_size
                self.alpha = 1.702
                self.limit = 7.0

                self.gate_up_projs = nn.ModuleList([
                    nn.Linear(self.hidden_size, 2 * self.expert_dim, dtype=config.dtype)
                    for _ in range(self.num_experts)
                ])

                self.down_projs = nn.ModuleList([
                    nn.Linear(self.expert_dim, self.hidden_size, dtype=config.dtype)
                    for _ in range(self.num_experts)
                ])

                if ori_experts is not None:
                    for i in range(self.num_experts):
                        tgt_gu_w = self.gate_up_projs[i].weight   # [2E, H]
                        tgt_gu_b = self.gate_up_projs[i].bias     # [2E]
                        tgt_d_w  = self.down_projs[i].weight      # [H, E]
                        tgt_d_b  = self.down_projs[i].bias        # [H]

                        gu_w_src = ori_experts.gate_up_proj[i].detach().t().contiguous()
                        gu_b_src = ori_experts.gate_up_proj_bias[i].detach()
                        d_w_src  = ori_experts.down_proj[i].detach().t().contiguous()
                        d_b_src  = ori_experts.down_proj_bias[i].detach()

                        with torch.no_grad():
                            tgt_gu_w.copy_(gu_w_src)
                            tgt_gu_b.copy_(gu_b_src)
                            tgt_d_w.copy_(d_w_src)
                            tgt_d_b.copy_(d_b_src)

            def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
                batch_size = hidden_states.shape[0]
                hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
                num_experts = routing_weights.shape[1]

                hidden_states = hidden_states.repeat(num_experts, 1)
                hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
                gate_up = torch.stack([proj(hidden_states[i]) for i, proj in enumerate(self.gate_up_projs)])
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                next_states = torch.stack([proj((up[i] + 1) * glu[i]) for i, proj in enumerate(self.down_projs)])
                next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
                next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
                next_states = next_states.sum(dim=0)

                return next_states

        class GptOssTopKRouterNew(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.top_k = config.num_experts_per_tok
                self.num_experts = config.num_local_experts
                self.hidden_dim = config.hidden_size
                self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
                self.bias = nn.Parameter(torch.empty(self.num_experts))

            def forward(self, hidden_states):
                hidden_states = hidden_states.reshape(-1, self.hidden_dim)
                router_logits = F.linear(hidden_states, self.weight.to(hidden_states.dtype), self.bias.to(hidden_states.dtype))  # (seq_len, num_experts)
                router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
                router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
                router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
                return router_scores, router_indices


        @use_kernel_forward_from_hub("MegaBlocksMoeMLP")
        class GptOssMLPNew(gpt_oss_modeling.GptOssMLP):
            def __init__(self, config, ori_mlp):
                super().__init__(config)
                experts_new = GptOssExpertsNew(config, self.experts)
                del self.experts
                self.router = GptOssTopKRouterNew(config)
                self.experts = experts_new

        if load_quantized_model:
            gpt_oss_modeling.GptOssExperts = GptOssExpertsNew
            gpt_oss_modeling.GptOssTopKRouter = GptOssTopKRouterNew

        else:
            model = model.to("cpu")
            pb = logger.pb(list(model.named_modules()))
            for name, module in pb:
                if isinstance(module, gpt_oss_modeling.GptOssMLP):
                    new_module = GptOssMLPNew(config=model.config, ori_mlp=module)
                    parent, child = name.rsplit(".", maxsplit=1)
                    parent = model.get_submodule(parent)
                    setattr(parent, child, new_module)
            
        return model
