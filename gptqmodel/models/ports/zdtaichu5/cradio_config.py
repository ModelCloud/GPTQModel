# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026, ZDTaichu-5.0-9B Contributors.  All rights reserved.
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
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone inference configuration for the C-RADIO vision tower."""

from typing import Dict, List, Optional, Tuple, Union

from transformers import PretrainedConfig


class RADIOConfig(PretrainedConfig):
    model_type = "radio"

    def __init__(
        self,
        args: Optional[dict] = None,
        version: str = "c-radio_v4-h",
        patch_size: int = 16,
        max_resolution: int = 2048,
        preferred_resolution: Tuple[int, int] = (768, 768),
        adaptor_names: Union[str, List[str], None] = None,
        adaptor_configs: Optional[Dict] = None,
        vitdet_window_size: Optional[int] = None,
        feature_normalizer_config: Optional[dict] = None,
        inter_feature_normalizer_config: Optional[dict] = None,
        hidden_size: int = 1280,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        image_size: int = 224,
        mlp_ratio: float = 4.0,
        **kwargs,
    ):
        self.args = args or {}
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.version = version
        self.patch_size = patch_size
        self.max_resolution = max_resolution
        self.preferred_resolution = preferred_resolution
        self.adaptor_names = adaptor_names
        self.adaptor_configs = adaptor_configs
        self.vitdet_window_size = vitdet_window_size
        self.feature_normalizer_config = feature_normalizer_config
        self.inter_feature_normalizer_config = inter_feature_normalizer_config
        super().__init__(**kwargs)


__all__ = ["RADIOConfig"]
