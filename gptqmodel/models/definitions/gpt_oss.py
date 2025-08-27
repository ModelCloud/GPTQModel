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

    def before_model_load(self):
        import torch
        from torch import nn

        class GptOssExperts(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.intermediate_size = config.intermediate_size
                self.num_experts = config.num_local_experts
                self.hidden_size = config.hidden_size
                self.expert_dim = self.intermediate_size
                self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
                self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))
                self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
                self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

                self.alpha = 1.702
                self.limit = 7.0

                self.register_load_state_dict_post_hook(self._after_load)
                
            @torch.no_grad()
            def _after_load(self, module, incompatible_keys):
                # skip if already built (hook may be triggered multiple times)
                if not hasattr(self, "gate_up") or len(getattr(self, "gate_up", [])) != self.num_experts:
                    # Create empty linear layers
                    self.gate_up = nn.ModuleList([
                        nn.Linear(self.hidden_size, 2 * self.expert_dim, bias=True)
                        for _ in range(self.num_experts)
                    ])
                    self.down = nn.ModuleList([
                        nn.Linear(self.expert_dim, self.hidden_size, bias=True)
                        for _ in range(self.num_experts)
                    ])

                # align and copy for each expert
                for i in range(self.num_experts):
                    # source tensors
                    gate_up_w = self.gate_up_proj[i]         # Expected: [2*expert_dim, hidden_size]
                    gate_up_b = self.gate_up_proj_bias[i]    # Expected: [2*expert_dim]
                    down_w    = self.down_proj[i]            # Expected: [hidden_size, expert_dim]
                    down_b    = self.down_proj_bias[i]       # Expected: [expert_dim]

                    # target parameters
                    tgt_gu_w = self.gate_up[i].weight  # Expected: [2*expert_dim, hidden_size]
                    tgt_gu_b = self.gate_up[i].bias    # Expected: [2*expert_dim]
                    tgt_d_w  = self.down[i].weight     # Expected: [hidden_size, expert_dim]
                    tgt_d_b  = self.down[i].bias       # Expected: [expert_dim]

                    # transpose and check shapes
                    gu_w_src_T = gate_up_w.t().contiguous()
                    d_w_src_T  = down_w.t().contiguous()

                    assert gu_w_src_T.shape == tgt_gu_w.shape, f"gate_up weight shape mismatch: src {gu_w_src_T.shape} vs tgt {tgt_gu_w.shape}"
                    assert gate_up_b.shape  == tgt_gu_b.shape, f"gate_up bias shape mismatch: src {gate_up_b.shape} vs tgt {tgt_gu_b.shape}"
                    assert d_w_src_T.shape  == tgt_d_w.shape,  f"down weight shape mismatch: src {d_w_src_T.shape} vs tgt {tgt_d_w.shape}"
                    assert down_b.shape     == tgt_d_b.shape,  f"down bias shape mismatch: src {down_b.shape} vs tgt {tgt_d_b.shape}"

                    # copy data to target parameters
                    with torch.no_grad():
                        self.gate_up[i].weight = nn.Parameter(gu_w_src_T)
                        self.gate_up[i].bias = nn.Parameter(gate_up_b)
                        self.down[i].weight = nn.Parameter(d_w_src_T)
                        self.down[i].bias = nn.Parameter(down_b)

            def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
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

        # monkey patch GptOssExperts
        import transformers.models.gpt_oss.modeling_gpt_oss as gptoss_modeling
        gptoss_modeling.GptOssExperts = GptOssExperts


