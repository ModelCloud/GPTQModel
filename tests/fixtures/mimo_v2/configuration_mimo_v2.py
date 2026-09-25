# SPDX-License-Identifier: Apache-2.0
"""Configuration for generated-weight MiMo decoder fixtures."""

from transformers import PretrainedConfig


class MiMoV2Config(PretrainedConfig):
    model_type = "mimo_v2"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
