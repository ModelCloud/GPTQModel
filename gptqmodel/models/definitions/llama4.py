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

from transformers import AutoModelForImageTextToText

from .._const import EXPERT_INDEX_PLACEHOLDER
from ..base import BaseGPTQModel


class Llama4GPTQ(BaseGPTQModel):
    # some bug in the attention_mask of transformers.modeling_llama4,
    # so batch quantization for Llama4 is temporarily not supported.
    support_batch_quantize = False
    loader = AutoModelForImageTextToText

    base_modules = ["language_model.model.embed_tokens", "language_model.model.norm"]
    pre_lm_head_norm_module = "language_model.model.norm"

    layers_node = "language_model.model.layers"
    layer_type = "Llama4TextDecoderLayer"

    dynamic_expert_index = "num_local_experts"

    # TODO: full deprecation by gptqmodel v4.3
    # legacy definition (deprecated): migrate to layers_modules_tree
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj"],

        [f"feed_forward.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj", f"feed_forward.experts.{EXPERT_INDEX_PLACEHOLDER}.up_proj"],
        [f"feed_forward.experts.{EXPERT_INDEX_PLACEHOLDER}.down_proj"],

        ["feed_forward.shared_expert.gate_proj", "feed_forward.shared_expert.up_proj", "feed_forward.shared_expert.down_proj"],
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


    def after_model_load(self, model, load_quantized_model=False):
        if load_quantized_model:
            return model

        import os
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

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

                with torch.no_grad():
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

        model = model.to("cpu")
        def process_module(name, module, model, config):
            if isinstance(module, Llama4TextMoe):
                new_module = SequentialLlama4TextMoe(config=config, original=module)
                parent, child = name.rsplit(".", maxsplit=1)
                print("replace moe" + name + child)
                parent = model.get_submodule(parent)
                setattr(parent, child, new_module)
        print("cpu count", os.cpu_count())
        with ThreadPoolExecutor(max_workers=8) as executor:
            process_fn = partial(process_module, model=model, config=model.config.get_text_config())
            list(executor.map(lambda x: process_fn(x[0], x[1]), model.named_modules()))

        return model
