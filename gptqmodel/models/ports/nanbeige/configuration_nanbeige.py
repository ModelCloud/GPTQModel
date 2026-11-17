# SPDX-FileCopyrightText: Nanbeige contributors
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Adapted from Nanbeige/Nanbeige4.2-3B at b82e54bd609793562a75cbf9337970a93369eab5.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class NanbeigeConfig(PretrainedConfig):
    model_type = "nanbeige"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        qk_layernorm=False,
        emb_neighbor_num=None,
        emb_split_num=None,
        ngram_vocab_size_ratio=None,
        ngram_mod_force_prime=False,
        ngram_embedding_hidden_size=None,
        ngram_fused_mode="average",
        emb_tp_num=None,
        ngram_compressed_tokenizer=False,
        skip_ngram_for_input=False,
        insert_ngram_layer_idx=None,
        ngram_insert_all_layers=False,
        ngram_layer_downproject_size=None,
        num_loops=1,
        loop_loss_weights=None,
        skip_loop_final_norm=False,
        enable_double_loop_split=False,
        loop_middle_layers=None,
        loop_share_kv=False,
        mhc_diff_for_loop=False,
        mhc_double_stream_position_for_loop=None,
        enable_hyper_connection=False,
        enable_mhc=False,
        enable_h_res_identity=False,
        mhc_identity_nohresparam=False,
        num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        mhc_init_gating_factor=0.01,
        enable_depth_attention=False,
        depth_attention_stride=None,
        depth_attention_recent_window=0,
        depth_attention_static_anchor_once=True,
        **kwargs,
    ):
        physical = kwargs.pop("num_physical_layers", None)
        if physical is not None:
            if (
                not isinstance(physical, int)
                or physical < 1
                or num_hidden_layers not in (physical, physical * num_loops)
            ):
                raise ValueError(
                    "num_physical_layers must identify the physical checkpoint depth, not an ambiguous expansion."
                )
            num_hidden_layers = physical
            self.num_physical_layers = physical
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        rope = kwargs.pop("rope_parameters", None)
        if rope is None:
            rope = dict(rope_scaling or {})
        else:
            rope = dict(rope)
        rope["rope_type"] = rope.pop("type", rope.get("rope_type", "default"))
        rope.setdefault("rope_theta", rope_theta)
        self.rope_theta = rope["rope_theta"]
        self.rope_parameters = rope
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.qk_layernorm = qk_layernorm
        self.emb_neighbor_num = emb_neighbor_num
        self.emb_split_num = emb_split_num
        self.ngram_vocab_size_ratio = ngram_vocab_size_ratio
        self.ngram_mod_force_prime = ngram_mod_force_prime
        self.ngram_embedding_hidden_size = ngram_embedding_hidden_size
        self.ngram_fused_mode = ngram_fused_mode
        self.emb_tp_num = emb_tp_num
        self.ngram_compressed_tokenizer = ngram_compressed_tokenizer
        self.skip_ngram_for_input = skip_ngram_for_input
        self.insert_ngram_layer_idx = insert_ngram_layer_idx if insert_ngram_layer_idx is not None else []
        self.ngram_insert_all_layers = ngram_insert_all_layers
        self.ngram_layer_downproject_size = ngram_layer_downproject_size
        self.num_loops = num_loops
        self.loop_loss_weights = loop_loss_weights if loop_loss_weights is not None else []
        self.skip_loop_final_norm = skip_loop_final_norm
        self.enable_double_loop_split = enable_double_loop_split
        self.loop_middle_layers = loop_middle_layers
        self.loop_share_kv = loop_share_kv
        self.mhc_diff_for_loop = mhc_diff_for_loop
        self.mhc_double_stream_position_for_loop = mhc_double_stream_position_for_loop
        self.enable_hyper_connection = enable_hyper_connection
        self.enable_mhc = enable_mhc
        self.enable_h_res_identity = enable_h_res_identity
        self.mhc_identity_nohresparam = mhc_identity_nohresparam
        self.num_residual_streams = num_residual_streams
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations
        self.mhc_init_gating_factor = mhc_init_gating_factor
        self.enable_depth_attention = enable_depth_attention
        self.depth_attention_stride = depth_attention_stride
        self.depth_attention_recent_window = depth_attention_recent_window
        self.depth_attention_static_anchor_once = depth_attention_static_anchor_once
        self._hyper_connection_validation()
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        rope_scaling_type = self.rope_parameters["rope_type"]
        if rope_scaling_type == "default":
            return
        rope_scaling_factor = self.rope_parameters.get("factor", None)
        if rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

    def _hyper_connection_validation(self):
        ngram_fields = (self.emb_neighbor_num, self.emb_split_num, self.ngram_vocab_size_ratio)
        if any(value is not None for value in ngram_fields):
            if any(value is None or value < 1 for value in ngram_fields) or self.emb_neighbor_num < 2:
                raise ValueError("N-gram embeddings require positive ratio/split and emb_neighbor_num >= 2 together.")
            count = self.emb_split_num * (self.emb_neighbor_num - 1)
            embedding_size = self.ngram_embedding_hidden_size or self.hidden_size
            if embedding_size % count:
                raise ValueError("N-gram embedding hidden size must be divisible by the number of embedders.")
        if self.num_hidden_layers < 1 or self.num_loops < 1:
            raise ValueError("Nanbeige requires positive physical layers and loops.")
        if self.head_dim < 2 or self.head_dim % 2 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Nanbeige requires an even head_dim and query heads divisible by KV heads.")
        if self.mhc_diff_for_loop and (not self.enable_double_loop_split):
            raise ValueError("mhc_diff_for_loop requires enable_double_loop_split=True.")
        if self.mhc_diff_for_loop and (not self.enable_mhc):
            raise ValueError("mhc_diff_for_loop requires enable_mhc=True.")
        if self.loop_share_kv and (not self.enable_double_loop_split):
            raise ValueError("loop_share_kv requires enable_double_loop_split=True.")
        if self.enable_depth_attention:
            if self.num_hidden_layers <= 0:
                raise ValueError("enable_depth_attention requires num_hidden_layers to be greater than 0.")
            if self.depth_attention_stride is None:
                self.depth_attention_stride = self.num_hidden_layers // 2
            if self.depth_attention_stride <= 0:
                raise ValueError("depth_attention_stride must be greater than 0.")
            if self.depth_attention_recent_window < 0:
                raise ValueError("depth_attention_recent_window must be >= 0.")
            if self.depth_attention_recent_window != 0:
                raise ValueError("anchor-only Depth-Attention requires depth_attention_recent_window == 0.")
            if self.enable_double_loop_split and (not self.depth_attention_static_anchor_once):
                raise ValueError(
                    "enable_depth_attention with double-loop split requires depth_attention_static_anchor_once=True."
                )
        if self.mhc_double_stream_position_for_loop is not None:
            if self.mhc_double_stream_position_for_loop not in ("mid", "edge"):
                raise ValueError("mhc_double_stream_position_for_loop must be one of: mid, edge.")
            if not self.enable_double_loop_split:
                raise ValueError("mhc_double_stream_position_for_loop requires enable_double_loop_split=True.")
        if (self.enable_mhc or self.enable_h_res_identity) and (not self.enable_hyper_connection):
            raise ValueError("enable_mhc/enable_h_res_identity require enable_hyper_connection=True.")
        if self.mhc_identity_nohresparam:
            if not self.enable_h_res_identity:
                raise ValueError("mhc_identity_nohresparam requires enable_h_res_identity=True.")
            if not self.enable_mhc:
                raise ValueError("mhc_identity_nohresparam requires enable_mhc=True.")
        if self.enable_hyper_connection and self.num_residual_streams < 2:
            raise ValueError("num_residual_streams must be >= 2 when enable_hyper_connection=True.")
        if self.enable_mhc and self.mhc_sinkhorn_iterations < 1:
            raise ValueError("mhc_sinkhorn_iterations must be >= 1 when enable_mhc=True.")
        if self.ngram_insert_all_layers and self.insert_ngram_layer_idx:
            raise ValueError("ngram_insert_all_layers cannot be used with insert_ngram_layer_idx.")
        if self.ngram_layer_downproject_size is not None and self.ngram_layer_downproject_size <= 0:
            raise ValueError("ngram_layer_downproject_size must be greater than 0 when set.")
        if self.ngram_embedding_hidden_size is not None and self.ngram_embedding_hidden_size <= 0:
            raise ValueError("ngram_embedding_hidden_size must be greater than 0 when set.")
        if self.ngram_fused_mode not in ("average", "concat"):
            raise ValueError("ngram_fused_mode must be one of: average, concat.")
        if (
            not self.enable_double_loop_split
            and self.loop_loss_weights is not None
            and (sum(self.loop_loss_weights) > 1.0)
        ):
            raise ValueError("sum(loop_loss_weights) must be <= 1.0.")
        if self.enable_double_loop_split and self.loop_middle_layers is None:
            if self.num_hidden_layers <= 0:
                raise ValueError("enable_double_loop_split requires num_hidden_layers to be greater than 0.")
            if self.num_hidden_layers % 2 != 0:
                raise ValueError(
                    "enable_double_loop_split requires num_hidden_layers to be divisible by 2 when loop_middle_layers is not set."
                )
            self.loop_middle_layers = self.num_hidden_layers // 2
        if self.loop_middle_layers is not None:
            if self.num_hidden_layers <= 0:
                raise ValueError("loop_middle_layers requires num_hidden_layers to be greater than 0.")
            if self.loop_middle_layers <= 0:
                raise ValueError("loop_middle_layers must be greater than 0.")
            if self.num_hidden_layers % self.loop_middle_layers != 0:
                raise ValueError("loop_middle_layers must be a factor of num_hidden_layers.")
