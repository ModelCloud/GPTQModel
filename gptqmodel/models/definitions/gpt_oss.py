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
    require_monkeypatch = True
    dynamic_expert_index = "num_local_experts"

    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = ["model.layers"]
    layer_type = "GptOssDecoderLayer"
    layer_modules = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],

        # uses dynamic_expert_index
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.up_proj", f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj"],
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.down_proj"],
    ]

    layers_modules_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "mlp": {
                "experts": {
                    "#": ("up_proj", "gate_proj", "down_proj"),
                },
            },
        }
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

                self.gate_up_projs = nn.ModuleList([
                    nn.Linear(self.hidden_size, 2 * self.expert_dim)
                    for _ in range(self.num_experts)
                ])

                self.down_projs = nn.ModuleList([
                    nn.Linear(self.expert_dim, self.hidden_size)
                    for _ in range(self.num_experts)
                ])

                self.alpha = 1.702
                self.limit = 7.0

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

        # monkey patch GptOssExperts
        import transformers.models.gpt_oss.modeling_gpt_oss as gptoss_modeling
        gptoss_modeling.GptOssExperts = GptOssExperts


