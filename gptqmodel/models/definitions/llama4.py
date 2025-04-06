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

        # feed_forward.router contains feed_forward.experts.gate_up_proj and feed_forward.experts.down_proj
        ["feed_forward.router"],
        ["feed_forward.shared_expert.gate_proj", "feed_forward.shared_expert.up_proj", "feed_forward.shared_expert.down_proj"],
    ]
