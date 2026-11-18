# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Field semantics follow DeepSeek-V4.1-Flash ``config.json`` and ``inference/model.py``
# (Copyright (c) 2023 DeepSeek, MIT License; see LICENSE in this directory).
"""DeepSeek V4.1 configuration on the Transformers 5.23 ABI.

The released ``config.json`` is composite: ``text_config`` describes the decoder,
its compressed-KV/indexer schedule, mHC streams, Engram (PLE) tables and the
dSpark draft stages; ``vision_config`` describes the ViT that this port does not
implement. The released text config still spells RoPE as ``rope_theta`` plus
``rope_scaling``; it is converted once here into ``rope_parameters`` with the two
rope families the reference uses:

* ``main``: ``rope_theta`` without YaRN, for layers whose ``compress_ratio`` is 0
  and for the dSpark stages;
* ``compress``: ``compress_rope_theta`` with YaRN, for every layer that attends
  over compressed KV (its queries, window keys, compressed keys and indexer).

The reference applies no YaRN magnitude correction, so ``attention_factor`` is 1.
"""

from __future__ import annotations

import copy
from typing import ClassVar

from transformers.configuration_utils import PreTrainedConfig


_RELEASED_COMPRESS_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]


class DeepseekV41TextConfig(PreTrainedConfig):
    model_type = "deepseek_v41_text"
    # ``rope_parameters`` is nested by rope family, not by layer type.
    _rope_type_labels = ("main", "compress")
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]
    attribute_map: ClassVar[dict[str, str]] = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 5120,
        moe_intermediate_size: int = 2304,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1280,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        hidden_act: str = "silu",
        swiglu_limit: float = 10.0,
        rms_norm_eps: float = 1e-20,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        compress_rope_theta: float = 160000.0,
        rope_parameters: dict | None = None,
        n_routed_experts: int = 384,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        scoring_func: str = "sqrtsoftplus",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        sliding_window: int = 128,
        compress_ratios: list[int] | None = None,
        kv_source_layer_ids: list[int] | None = None,
        index_source_layer_ids: list[int] | None = None,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = 512,
        candidate_source_layer_id: int = 20,
        candidate_topk_blocks: int = 2048,
        candidate_block_size: int = 8,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        engram_layer_ids: list[int] | None = None,
        engram_num_embeddings: list[int] | None = None,
        engram_max_ngram_size: int = 4,
        engram_vocab_size: int = 16000000,
        engram_n_heads: int = 8,
        engram_head_dim: int = 256,
        engram_pad_token_id: int = 2,
        engram_compressed_vocab_size: int = 99092,
        num_nextn_predict_layers: int = 3,
        dspark_block_size: int = 5,
        dspark_noise_token_id: int = 128799,
        dspark_target_layer_ids: list[int] | None = None,
        dspark_markov_rank: int = 256,
        dspark_n_routed_experts: int = 128,
        dspark_num_experts_per_tok: int = 3,
        bos_token_id: int | None = 0,
        eos_token_id: int | list[int] | None = 1,
        pad_token_id: int | None = 2,
        **kwargs,
    ):
        # Released configs spell YaRN as ``rope_scaling``; convert it once, here.
        rope_scaling = kwargs.pop("rope_scaling", None)
        if num_key_value_heads != 1:
            raise ValueError("DeepSeek V4.1 attention shares one latent KV head")
        if n_shared_experts != 1:
            raise ValueError("the DeepSeek V4.1 reference MoE has exactly one shared expert")
        if scoring_func not in ("softmax", "sigmoid", "sqrtsoftplus"):
            raise ValueError(f"unsupported DeepSeek V4.1 scoring function: {scoring_func}")
        if hidden_act != "silu":
            raise ValueError("the DeepSeek V4.1 reference experts use SwiGLU")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.sliding_window = sliding_window
        if compress_ratios is None:
            compress_ratios = list(_RELEASED_COMPRESS_RATIOS)
        self.compress_ratios = list(compress_ratios)
        self.kv_source_layer_ids = list([2, 8, 14, 20] if kv_source_layer_ids is None else kv_source_layer_ids)
        self.index_source_layer_ids = list(
            [2, 8, 14, 20, 24, 28, 32, 36] if index_source_layer_ids is None else index_source_layer_ids
        )
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.engram_layer_ids = list([1, 14] if engram_layer_ids is None else engram_layer_ids)
        self.engram_num_embeddings = list(
            [384006168, 384016682] if engram_num_embeddings is None else engram_num_embeddings
        )
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = list([37, 38, 39] if dspark_target_layer_ids is None else dspark_target_layer_ids)
        self.dspark_markov_rank = dspark_markov_rank
        self.dspark_n_routed_experts = dspark_n_routed_experts
        self.dspark_num_experts_per_tok = dspark_num_experts_per_tok

        if rope_parameters is None:
            compress = {"rope_type": "default", "rope_theta": compress_rope_theta}
            if rope_scaling:
                scaling = dict(rope_scaling)
                rope_type = scaling.pop("rope_type", scaling.pop("type", "default"))
                compress = {**scaling, "rope_type": rope_type, "rope_theta": compress_rope_theta}
            rope_parameters = {"main": {"rope_type": "default", "rope_theta": rope_theta}, "compress": compress}
        self.rope_parameters = copy.deepcopy(rope_parameters)
        for family in ("main", "compress"):
            params = self.rope_parameters.get(family)
            if not isinstance(params, dict) or params.get("rope_type") not in ("default", "yarn"):
                raise ValueError("DeepSeek V4.1 needs default/YaRN rope_parameters for 'main' and 'compress'")
            if params["rope_type"] == "yarn":
                params.setdefault("attention_factor", 1.0)
                if params["attention_factor"] != 1.0:
                    raise ValueError("the DeepSeek V4.1 reference applies no YaRN attention factor")

        self._validate_layout()
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate_layout(self):
        layers = self.num_hidden_layers
        if len(self.compress_ratios) < layers:
            raise ValueError("compress_ratios must cover every decoder layer")
        if self.head_dim <= self.qk_rope_head_dim or self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be an even tail of head_dim")
        if self.num_attention_heads % self.o_groups:
            raise ValueError("o_groups must divide num_attention_heads")
        for layer in self.kv_source_layer_ids:
            if not 0 <= layer < layers or not self.compress_ratios[layer]:
                raise ValueError("every KV source layer must compress its own KV")
        for layer in range(layers):
            ratio = self.compress_ratios[layer]
            if not ratio:
                continue
            sources = [source for source in self.kv_source_layer_ids if source <= layer]
            if not sources or self.compress_ratios[max(sources)] != ratio:
                raise ValueError(f"layer {layer} has no preceding KV source with compress ratio {ratio}")
            indexers = [source for source in self.index_source_layer_ids if source <= layer]
            if not indexers:
                raise ValueError(f"compressed layer {layer} has no preceding index source")
        for layer in self.index_source_layer_ids:
            if not 0 <= layer < layers or not self.compress_ratios[layer]:
                raise ValueError("every index source layer must attend over compressed KV")
        if len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError("engram_num_embeddings needs one row count per Engram layer")
        if any(not 0 <= layer < layers for layer in self.engram_layer_ids + self.dspark_target_layer_ids):
            raise ValueError("Engram/dSpark target layer outside the decoder")

    def compress_ratio(self, layer_idx: int) -> int:
        return int(self.compress_ratios[layer_idx])

    def dspark_config(self) -> DeepseekV41TextConfig:
        """Block config for the dSpark stages stored under ``mtp.*``.

        The stages are sliding-window-only decoder blocks with their own expert
        count; they own no compressor, indexer or Engram table.
        """

        config = copy.deepcopy(self)
        config.num_hidden_layers = config.num_nextn_predict_layers
        config.n_routed_experts = config.dspark_n_routed_experts
        config.num_experts_per_tok = config.dspark_num_experts_per_tok
        config.compress_ratios = [0] * config.num_hidden_layers
        config.kv_source_layer_ids = []
        config.index_source_layer_ids = []
        config.candidate_source_layer_id = -1
        config.engram_layer_ids = []
        config.engram_num_embeddings = []
        return config


class DeepseekV41VisionConfig(PreTrainedConfig):
    """Released vision fields, kept for round-tripping; the ViT is not ported."""

    model_type = "deepseek_v41_vision"

    def __init__(
        self,
        num_hidden_layers: int = 32,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        intermediate_size: int = 2816,
        patch_size: int = 14,
        rope_theta: float = 10000.0,
        downsample_ratio: int = 3,
        max_image_tokens: int = 1024,
        min_pixels: int = 295936,
        max_wh_ratio: float | None = None,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.downsample_ratio = downsample_ratio
        self.max_image_tokens = max_image_tokens
        self.min_pixels = min_pixels
        self.max_wh_ratio = max_wh_ratio
        super().__init__(**kwargs)


class DeepseekV41Config(PreTrainedConfig):
    model_type = "deepseek_v41"
    sub_configs: ClassVar[dict] = {"text_config": DeepseekV41TextConfig, "vision_config": DeepseekV41VisionConfig}
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        text_config: dict | DeepseekV41TextConfig | None = None,
        vision_config: dict | DeepseekV41VisionConfig | None = None,
        image_token_id: int = 129264,
        bos_token_id: int | None = 0,
        eos_token_id: int | list[int] | None = 1,
        pad_token_id: int | None = 2,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if text_config is None:
            text_config = DeepseekV41TextConfig()
        elif isinstance(text_config, dict):
            text_config = DeepseekV41TextConfig(**text_config)
        if vision_config is None:
            vision_config = DeepseekV41VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = DeepseekV41VisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_text_config(self, decoder=None, encoder=None):
        return self.text_config


__all__ = ["DeepseekV41Config", "DeepseekV41TextConfig", "DeepseekV41VisionConfig"]
