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
