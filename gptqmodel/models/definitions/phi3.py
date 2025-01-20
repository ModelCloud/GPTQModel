# Copyright 2025 ModelCloud
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


class Phi3GPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens", "embed_dropout", "model.norm"]

    layers_node = "model.layers"
    layer_type = "Phi3DecoderLayer"
    layer_modules = [
        ["self_attn.qkv_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_up_proj"],
        ["mlp.down_proj"],
    ]

class PhiMoEGPTQForCausalLM(BaseGPTQModel):
    require_pkgs_version = ["transformers<=4.44.2"]

    layer_type = "PhiMoEDecoderLayer"
    layers_node = "model.layers"
    base_modules = ["model.embed_tokens", "model.norm"]

    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        [f"block_sparse_moe.experts.{EXPERT_INDEX_PLACEHOLDER}.w1"],
        [f"block_sparse_moe.experts.{EXPERT_INDEX_PLACEHOLDER}.w2"],
    ]

__all__ = ["Phi3GPTQ", "PhiMoEGPTQForCausalLM"]