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

    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj"],

        ["feed_forward.shared_expert.gate_proj", "feed_forward.shared_expert.up_proj", "feed_forward.shared_expert.down_proj"],
    ]

    def after_model_load(self, model, load_quantized_model=False):
        if load_quantized_model:
            return model

        from transformers.models.llama4.modeling_llama4 import Llama4TextMLP

        class SequentialLlama4TextExperts(torch.nn.ModuleList):
            def __init__(self, config, original):
                self.num_experts = original.gate_up_proj.shape[0]
                with no_init_weights():
                    super().__init__([Llama4TextMLP(config) for _ in range(self.num_experts)])
                intermediate_size = original.down_proj.shape[1]

                for i in range(self.num_experts):
                    gate_up = original.gate_up_proj[i]
                    down = original.down_proj[i]
                    gate_proj = gate_up[:, :intermediate_size]
                    up_proj = gate_up[:, intermediate_size:]

                    self[i].gate_proj.weight.data = gate_proj.t().contiguous()
                    self[i].up_proj.weight.data = up_proj.t().contiguous()
                    self[i].down_proj.weight.data = down.t().contiguous()

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
                new_module = SequentialLlama4TextMoe(config=config, ori_mlp=module)
                parent, child = name.rsplit(".", maxsplit=1)
                parent = model.get_submodule(parent)
                setattr(parent, child, new_module)
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            process_fn = partial(process_module, model=model, config=model.config.get_text_config())
            list(executor.map(lambda x: process_fn(x[0], x[1]), model.named_modules()))

        return model
