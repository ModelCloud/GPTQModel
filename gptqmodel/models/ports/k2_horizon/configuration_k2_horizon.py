# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""K2Horizon model configuration"""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class K2HorizonConfig(PreTrainedConfig):
    r"""
    decoder_sparse_step (`int`, *optional*, defaults to 1):
        The frequency of the MoE layer.
    mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
        Indicate which layers use K2HorizonMLP rather than K2HorizonSparseMoeBlock
        The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
        If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.

    ```python
    >>> from gptqmodel.models.ports.k2_horizon import K2HorizonModel, K2HorizonConfig

    >>> # Initializing a K2Horizon style configuration
    >>> configuration = K2HorizonConfig()
    >>> model = K2HorizonModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "k2_horizon"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    attention_dropout: float | int = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 768
    num_experts_per_tok: int = 8
    num_experts: int = 128
    norm_topk_prob: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list[int] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    head_dim: int = 128
    query_key_norm: bool = True
    moe_gate_bias: bool = False
    layernorm_num_groups: int = 1
    num_shared_experts: int = 0
    router_score_func: str = "softmax"
    router_scaling_factor: float | None = 1.0
    rope_head_dim: int | None = None
    attention_gate_func: str | None = None
    mova_num_experts: int = 0
    mova_num_experts_per_tok: int = 0

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers
        if self.router_scaling_factor is None:
            self.router_scaling_factor = 1.0
        super().__post_init__(**kwargs)


__all__ = ["K2HorizonConfig"]
