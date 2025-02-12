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


class MixtralGPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = "model.layers"
    layer_type = "MixtralDecoderLayer"
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
        ],
    ]
