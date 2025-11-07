# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForImageTextToText

from ..base import BaseQModel


class Llama4QModel(BaseQModel):
    # some bug in the attention_mask of transformers.modeling_llama4,
    # so batch quantization for Llama4 is temporarily not supported.
    support_batch_quantize = False
    loader = AutoModelForImageTextToText

    pre_lm_head_norm_module = "language_model.model.norm"

    dynamic_expert_index = "num_local_experts"

    module_tree = [
        "language_model",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "feed_forward": {
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_expert": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        }
    ]

    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            import torch
            import torch.nn as nn
            import transformers.models.llama4.modeling_llama4 as llama4_modeling
            from transformers.integrations.hub_kernels import use_kernel_forward_from_hub

            @use_kernel_forward_from_hub("Llama4TextMoe")
            class SequentialLlama4TextMoe(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.top_k = config.num_experts_per_tok
                    self.hidden_dim = config.hidden_size
                    print(config)
                    self.num_experts = 16
                    self.experts = nn.ModuleList(
                        [llama4_modeling.Llama4TextMLP(config) for _ in range(self.num_experts)]
                    )
                    self.router = llama4_modeling.Llama4Router(config)
                    self.shared_expert = llama4_modeling.Llama4TextMLP(config)

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

            llama4_modeling.Llama4TextMoe = SequentialLlama4TextMoe
