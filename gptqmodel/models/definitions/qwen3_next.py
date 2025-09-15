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

from .._const import EXPERT_INDEX_PLACEHOLDER
from ..base import BaseGPTQModel


class Qwen3NextGPTQ(BaseGPTQModel):
    """
    GPTQ config for Qwen3-Next (HF: Qwen3Next*), supporting:
      - Mixed token mixers per layer: 'full_attention' (self_attn.*) and 'linear_attention' (linear_attn.*)
      - Dense MLP (Qwen3NextMLP) and Sparse MoE (Qwen3NextSparseMoeBlock)
      - Dynamic expert indexing via config.num_experts
    """

    layer_modules_strict = False

    # Embeddings & final norm (pre lm_head)
    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    dynamic_expert_index = "num_experts"

    # Decoder layers container and per-layer type
    layers_node = ["model.layers"]
    layer_type = "Qwen3NextDecoderLayer"

    layer_modules = [
            ['linear_attn.in_proj_qkvz', 'linear_attn.in_proj_ba'],
            ['linear_attn.out_proj'],

            ['mlp.gate'],
            ['mlp.shared_expert_gate'],
            ['mlp.shared_expert.gate_proj', 'mlp.shared_expert.up_proj', 'mlp.shared_expert.down_proj'],
            [f'mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.up_proj', f'mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj'],
            [f'mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.down_proj']
        ]

    # -----------------------------------------------------------------------------
    # Preferred modern hierarchical spec. The loader will gracefully skip any
    # subpaths that don't exist on a given layer (e.g., dense vs MoE, or mixer type).
    # -----------------------------------------------------------------------------
    layers_modules_tree = [
        "model",
        "layers",
        "#",
        {
            # Token mixers
            #"self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "linear_attn": ("in_proj_qkvz", "in_proj_ba", "out_proj"),  # conv1d intentionally excluded

            # MLP / MoE
            "mlp": {
                # MoE router + shared expert (Qwen3NextSparseMoeBlock)
                "gate": ("gate",),  # router gate linear
                "shared_expert_gate": ("shared_expert_gate",),
                "shared_expert": ("gate_proj", "up_proj", "down_proj"),

                # Experts list with dynamic index
                "experts": {
                    "#": ("up_proj", "gate_proj", "down_proj"),
                },
            },
        },
    ]
