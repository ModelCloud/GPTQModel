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
    layer_type = "PhiMoEDecoderLayer"
    layers_block_name = "model.layers"
    base_modules = ["model.embed_tokens", "model.norm"]
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        [
            "block_sparse_moe.experts.0.w1",
            "block_sparse_moe.experts.1.w1",
            "block_sparse_moe.experts.2.w1",
            "block_sparse_moe.experts.3.w1",
            "block_sparse_moe.experts.4.w1",
            "block_sparse_moe.experts.5.w1",
            "block_sparse_moe.experts.6.w1",
            "block_sparse_moe.experts.7.w1",
            "block_sparse_moe.experts.0.w3",
            "block_sparse_moe.experts.1.w3",
            "block_sparse_moe.experts.2.w3",
            "block_sparse_moe.experts.3.w3",
            "block_sparse_moe.experts.4.w3",
            "block_sparse_moe.experts.5.w3",
            "block_sparse_moe.experts.6.w3",
            "block_sparse_moe.experts.7.w3",
            "block_sparse_moe.experts.8.w1",
            "block_sparse_moe.experts.9.w1",
            "block_sparse_moe.experts.10.w1",
            "block_sparse_moe.experts.11.w1",
            "block_sparse_moe.experts.12.w1",
            "block_sparse_moe.experts.13.w1",
            "block_sparse_moe.experts.14.w1",
            "block_sparse_moe.experts.15.w1",
            "block_sparse_moe.experts.8.w3",
            "block_sparse_moe.experts.9.w3",
            "block_sparse_moe.experts.10.w3",
            "block_sparse_moe.experts.11.w3",
            "block_sparse_moe.experts.12.w3",
            "block_sparse_moe.experts.13.w3",
            "block_sparse_moe.experts.14.w3",
            "block_sparse_moe.experts.15.w3",
        ],
        [
            "block_sparse_moe.experts.0.w2",
            "block_sparse_moe.experts.1.w2",
            "block_sparse_moe.experts.2.w2",
            "block_sparse_moe.experts.3.w2",
            "block_sparse_moe.experts.4.w2",
            "block_sparse_moe.experts.5.w2",
            "block_sparse_moe.experts.6.w2",
            "block_sparse_moe.experts.7.w2",
            "block_sparse_moe.experts.8.w2",
            "block_sparse_moe.experts.9.w2",
            "block_sparse_moe.experts.10.w2",
            "block_sparse_moe.experts.11.w2",
            "block_sparse_moe.experts.12.w2",
            "block_sparse_moe.experts.13.w2",
            "block_sparse_moe.experts.14.w2",
            "block_sparse_moe.experts.15.w2"
        ],
    ]

__all__ = ["Phi3GPTQ", "PhiMoEGPTQForCausalLM"]