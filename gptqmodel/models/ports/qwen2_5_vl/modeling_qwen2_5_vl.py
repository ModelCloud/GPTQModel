# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Select exact native composite or standalone headless text models."""

from transformers import PreTrainedConfig, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel


class Qwen2_5_VLConfigLoader:
    """Read declarative metadata without checkpoint Python dispatch."""

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        payload, _ = PreTrainedConfig.get_config_dict(path, **kwargs)
        types = {"qwen2_5_vl": Qwen2_5_VLConfig, "qwen2_5_vl_text": Qwen2_5_VLTextConfig}
        model_type = payload.get("model_type")
        config_type = types.get(model_type) if isinstance(model_type, str) else None
        if config_type is None:
            raise ValueError(f"Unsupported Qwen2.5-VL model_type: {model_type!r}")
        return config_type.from_pretrained(path, **kwargs)


class Qwen2_5_VLModelLoader:
    """Preserve native classes and weight schemas for both registered aliases."""

    @staticmethod
    def _model_class(config):
        if type(config) is Qwen2_5_VLConfig:
            return Qwen2_5_VLForConditionalGeneration
        if type(config) is Qwen2_5_VLTextConfig:
            return Qwen2_5_VLTextModel
        raise ValueError(f"Unsupported Qwen2.5-VL config: {type(config).__name__}")

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls._model_class(config)._from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, path, config=None, **kwargs):
        if config is None:
            config = Qwen2_5_VLConfigLoader.from_pretrained(path, **kwargs)
        return cls._model_class(config).from_pretrained(path, config=config, **kwargs)
