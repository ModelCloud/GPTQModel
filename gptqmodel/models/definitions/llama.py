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

from ..base import BaseQModel


class LlamaGPTQ(BaseQModel):
    # Non-repeating layers at the root level: same level as `layers_node`
    # Excluding `layers_node`.
    base_modules = ["model.embed_tokens", "model.norm", "model.rotary_embed"]
    pre_lm_head_norm_module = "model.norm"

    # awq scaling optimizations requires some modules within same subset to strictly match the shape of previous module
    # here the o_proj must match v_proj or else scaling optimizations are skipped (GQA vs MHA)
    shape_must_match_previous = ["self_attn.o_proj"]

    # Each repeating layer in `model.layers` is of type `LlamaDecoderLayer`
    layer_type = "LlamaDecoderLayer"

    _layers_modules_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]
