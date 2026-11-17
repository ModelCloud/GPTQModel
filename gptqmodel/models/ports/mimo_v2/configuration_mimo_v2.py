# coding=utf-8
#
# Copyright 2026 Xiaomi Corporation.
# Copyright 2026 The HuggingFace Inc. team.
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

from copy import deepcopy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


_MIMOV2_ATTENTION_PROJECTION_LAYOUTS = {"split", "fused_qkv"}

_MIMOV2_SPLIT_TP_PLAN = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}

_MIMOV2_FUSED_QKV_TP_PLAN = {
    "layers.*.self_attn.qkv_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}

_MIMOV2_PP_PLAN = {
    "embed_tokens": (["input_ids"], ["inputs_embeds"]),
    "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
    "norm": (["hidden_states"], ["hidden_states"]),
}


def _to_plain_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "to_dict"):
        return deepcopy(value.to_dict())
    if hasattr(value, "__dict__"):
        return deepcopy(vars(value))
    raise TypeError(f"Unsupported config value type: {type(value)!r}")


class MiMoV2Config(PretrainedConfig):

    model_type = "mimo_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = _MIMOV2_SPLIT_TP_PLAN
    base_model_pp_plan = _MIMOV2_PP_PLAN

    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        layernorm_epsilon=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        attention_bias=False,
        attention_value_scale=None,
        head_dim=None,
        v_head_dim=None,
        swa_num_attention_heads=None,
        swa_num_key_value_heads=None,
        swa_head_dim=None,
        swa_v_head_dim=None,
        swa_rope_theta=None,
        sliding_window=None,
        sliding_window_size=None,
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=False,
        hybrid_block_size=None,
        hybrid_layer_pattern=None,
        partial_rotary_factor=1.0,
        n_routed_experts=None,
        moe_intermediate_size=None,
        num_experts_per_tok=None,
        routed_scaling_factor=None,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=None,
        topk_group=None,
        norm_topk_prob=True,
        moe_layer_freq=None,
        attention_projection_layout="split",
        vision_config=None,
        audio_config=None,
        processor_config=None,
        image_token_id=None,
        video_token_id=None,
        vision_start_token_id=None,
        vision_end_token_id=None,
        vision_model_type=None,
        **kwargs,
    ):
        rope_parameters = kwargs.pop("rope_parameters", None)
        if rope_scaling is None and rope_parameters is not None:
            rope_scaling = rope_parameters

        if attention_projection_layout is None:
            attention_projection_layout = "split"
        if attention_projection_layout not in _MIMOV2_ATTENTION_PROJECTION_LAYOUTS:
            raise ValueError(f"Unsupported MiMoV2 attention projection layout: {attention_projection_layout}")

        self.attention_projection_layout = attention_projection_layout
        self.base_model_tp_plan = (
            _MIMOV2_FUSED_QKV_TP_PLAN.copy()
            if attention_projection_layout == "fused_qkv"
            else _MIMOV2_SPLIT_TP_PLAN.copy()
        )
        self.base_model_pp_plan = _MIMOV2_PP_PLAN.copy()

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layernorm_epsilon = layernorm_epsilon
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_parameters = deepcopy(rope_scaling or {})
        self.rope_parameters.setdefault("rope_type", self.rope_parameters.pop("type", "default"))
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.attention_value_scale = attention_value_scale

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.v_head_dim = v_head_dim if v_head_dim is not None else self.head_dim
        self.swa_num_attention_heads = (
            swa_num_attention_heads if swa_num_attention_heads is not None else num_attention_heads
        )
        self.swa_num_key_value_heads = (
            swa_num_key_value_heads if swa_num_key_value_heads is not None else num_key_value_heads
        )
        if self.swa_num_attention_heads % self.swa_num_key_value_heads != 0:
            raise ValueError("swa_num_attention_heads must be divisible by swa_num_key_value_heads")
        self.swa_head_dim = swa_head_dim if swa_head_dim is not None else self.head_dim
        self.swa_v_head_dim = swa_v_head_dim if swa_v_head_dim is not None else self.swa_head_dim
        self.swa_rope_theta = swa_rope_theta if swa_rope_theta is not None else rope_theta

        if sliding_window is None:
            sliding_window = sliding_window_size
        self.sliding_window = sliding_window
        self.sliding_window_size = sliding_window_size if sliding_window_size is not None else sliding_window
        self.add_full_attention_sink_bias = add_full_attention_sink_bias
        self.add_swa_attention_sink_bias = add_swa_attention_sink_bias

        if hybrid_block_size is not None and hybrid_layer_pattern is None:
            hybrid_layer_pattern = [0 if ((i + 1) % hybrid_block_size == 0) else 1 for i in range(num_hidden_layers)]
        elif hybrid_layer_pattern is None:
            hybrid_layer_pattern = [0] * num_hidden_layers
        if len(hybrid_layer_pattern) != num_hidden_layers:
            raise ValueError("hybrid_layer_pattern length must match num_hidden_layers")
        self.hybrid_block_size = hybrid_block_size
        self.hybrid_layer_pattern = hybrid_layer_pattern

        self.partial_rotary_factor = partial_rotary_factor

        self.n_routed_experts = n_routed_experts
        self.moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        if isinstance(moe_layer_freq, int):
            moe_layer_freq = [moe_layer_freq > 0 and i % moe_layer_freq == 0 for i in range(num_hidden_layers)]
        elif moe_layer_freq is None:
            moe_layer_freq = [False] * num_hidden_layers
        if len(moe_layer_freq) != num_hidden_layers:
            raise ValueError("moe_layer_freq length must match num_hidden_layers")
        self.moe_layer_freq = moe_layer_freq

        self.vision_config = _to_plain_dict(vision_config)
        self.audio_config = _to_plain_dict(audio_config)
        self.processor_config = _to_plain_dict(processor_config)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.vision_model_type = vision_model_type
        self.audio_token_id = self.processor_config.get("audio_token_id", None) if self.processor_config else None
        self.audio_start_token_id = (
            self.processor_config.get("audio_start_token_id", None) if self.processor_config else None
        )
        self.audio_end_token_id = (
            self.processor_config.get("audio_end_token_id", None) if self.processor_config else None
        )

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

__all__ = ["MiMoV2Config"]
