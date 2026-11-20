# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from transformers import PreTrainedConfig


class DeepSeekV41TextConfig(PreTrainedConfig):
    model_type = "deepseek_v41_text"
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_parameters: dict | None = None
    use_cache: bool = True
    candidate_source_layer_id: int | None = None
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 1

    def __post_init__(self, **kwargs):
        if self.candidate_source_layer_id == -1:
            self.candidate_source_layer_id = None
        super().__post_init__(**kwargs)


class DeepSeekV41VisionConfig(PreTrainedConfig):
    model_type = "deepseek_v41_vision"


class DeepSeekV41FlashConfig(PreTrainedConfig):
    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepSeekV41TextConfig, "vision_config": DeepSeekV41VisionConfig}
    text_config: DeepSeekV41TextConfig | dict | None = None
    vision_config: DeepSeekV41VisionConfig | dict | None = None
    image_token_id: int | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = DeepSeekV41TextConfig(**self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = DeepSeekV41VisionConfig(**self.vision_config)
        super().__post_init__(**kwargs)


def attention_schedule(config):
    """Resolve the source layers before constructing any checkpoint tensors."""
    depth = config.num_hidden_layers
    ratios = tuple(config.compress_ratios[:depth])
    kv_sources = set(config.kv_source_layer_ids)
    index_sources = set(config.index_source_layer_ids)
    candidate_source = config.candidate_source_layer_id
    if (
        len(ratios) != depth
        or any(ratio < 0 for ratio in ratios)
        or not kv_sources <= index_sources <= set(range(depth))
    ):
        raise ValueError("Invalid DeepSeek V4.1 attention schedule")
    if candidate_source is not None and (
        candidate_source not in kv_sources or config.candidate_block_size < 1 or config.candidate_topk_blocks < 1
    ):
        raise ValueError("Invalid DeepSeek V4.1 candidate index configuration")

    kv_source = index_source = None
    schedule = []
    for layer, ratio in enumerate(ratios):
        if not ratio:
            if layer in index_sources:
                raise ValueError(f"Sliding-window layer {layer} cannot be a V4.1 index source")
            schedule.append((ratio, None, None))
            continue
        if layer in kv_sources:
            kv_source = layer
        if layer in index_sources:
            index_source = layer
        if kv_source is None or index_source is None or index_source < kv_source or ratios[kv_source] != ratio:
            raise ValueError(f"No matching V4.1 KV and index source for layer {layer}")
        if candidate_source is not None and layer > candidate_source and ratios[candidate_source] != ratio:
            raise ValueError(f"V4.1 candidate and consumer ratios differ at layer {layer}")
        schedule.append((ratio, kv_source, index_source))
    return schedule
