# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Task-aware selection of exact native Transformers Qwen2-VL classes.

The text model is headless. No synthetic language-modeling head is created.
Checkpoint class names and auto_map are never used as import instructions.
"""

from transformers import (
    PreTrainedConfig,
    Qwen2Tokenizer,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    Qwen2VLTextConfig,
    Qwen2VLTextModel,
)


__all__ = ["Qwen2VLConfigLoader", "Qwen2VLModelLoader", "Qwen2VLProcessor", "Qwen2Tokenizer"]


class Qwen2VLConfigLoader:
    @classmethod
    def from_pretrained(cls, model_id_or_path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        payload, _ = PreTrainedConfig.get_config_dict(model_id_or_path, **kwargs)
        model_type = payload.get("model_type")
        config_class = {"qwen2_vl": Qwen2VLConfig, "qwen2_vl_text": Qwen2VLTextConfig}.get(model_type)
        if config_class is None:
            raise ValueError(f"Unsupported Qwen2-VL model_type: {model_type!r}")
        return config_class.from_pretrained(model_id_or_path, **kwargs)


class Qwen2VLModelLoader:
    @staticmethod
    def _model_class(config):
        if type(config) is Qwen2VLConfig:
            return Qwen2VLForConditionalGeneration
        if type(config) is Qwen2VLTextConfig:
            return Qwen2VLTextModel
        raise ValueError("Qwen2-VL requires an exact native composite or standalone text configuration")

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls._model_class(config)._from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, model_id_or_path, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = Qwen2VLConfigLoader.from_pretrained(model_id_or_path, **kwargs)
        return cls._model_class(config).from_pretrained(model_id_or_path, config=config, **kwargs)
