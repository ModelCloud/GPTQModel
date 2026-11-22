# Vendored from huggingface/transformers v5.23.0 (Apache-2.0) by GPT-QModel.
# Do not edit by hand except to keep it on the 5.23 ABI; see docs/transformers5-model-ports.md.
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LFM2-VL model."""

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig, SubConfigSpec
from transformers.utils import auto_docstring
from ..auto import AutoConfig
from ..auto import SubConfigSpec  # package-local registry


@auto_docstring(checkpoint="LiquidAI/LFM2-VL-1.6B")
@strict
class Lfm2VlConfig(PreTrainedConfig):
    r"""
    projector_use_layernorm (`bool`, *optional*, defaults to `True`):
        Whether to use layernorm in the multimodal projector.
    downsample_factor (`int`, *optional*, defaults to 2):
        The downsample_factor factor of the vision backbone.
    """

    model_type = "lfm2_vl"
    sub_configs_defaults = {
        "vision_config": SubConfigSpec(config_class=AutoConfig, model_type="siglip2_vision_model"),
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="lfm2"),
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 396
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 2560
    projector_bias: bool = True
    projector_use_layernorm: bool = True
    downsample_factor: int = 2
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.tie_word_embeddings = kwargs.pop("tie_embedding", self.tie_word_embeddings)
        super().__post_init__(**kwargs)


__all__ = ["Lfm2VlConfig"]
