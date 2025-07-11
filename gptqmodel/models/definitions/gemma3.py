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

from ..base import BaseGPTQModel
from . import LlamaGPTQ


class Gemma3GPTQ(LlamaGPTQ):
    layer_type = "Gemma3DecoderLayer"

class Gemma3ForConditionalGenerationGPTQ(BaseGPTQModel):
    support_batch_quantize = False
    base_modules = ["model.language_model.embed_tokens", "model.language_model.norm"]
    pre_lm_head_norm_module = "model.language_model.norm"

    layers_node = "model.language_model.layers"
    layer_type = "Gemma3DecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    lm_head_module = "model.lm_head"
