# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Package-local model registry for the vendored Transformers 5.23 families.

Vendored composite models resolve sub-configs and sub-models by ``model_type``.
Transformers does that through its global ``transformers.models.auto`` tables,
which would silently return Transformers' own classes. This registry resolves
only to GPT-QModel's vendored classes and fails loudly for anything else. It
never registers anything with Transformers.
"""

from importlib import import_module

from transformers.configuration_utils import PreTrainedConfig
from transformers.configuration_utils import SubConfigSpec as _FrameworkSubConfigSpec

from . import _registry


def _resolve(table, model_type, kind):
    entry = table.get(model_type)
    if entry is None:
        raise ValueError(
            f"GPT-QModel has no vendored {kind} for model_type={model_type!r}. "
            "Port the family into gptqmodel/models/ports/native (see the model-porting skill)."
        )
    module, name = entry
    return getattr(import_module(module), name)


class _LazyMapping(dict):
    """``model_type -> class`` mapping that imports vendored modules on access."""

    def __init__(self, table, kind):
        super().__init__()
        self._table = table
        self._kind = kind

    def __contains__(self, model_type):
        return model_type in self._table

    def __getitem__(self, model_type):
        return _resolve(self._table, model_type, self._kind)

    def get(self, model_type, default=None):
        return self[model_type] if model_type in self._table else default

    def keys(self):
        return self._table.keys()


CONFIG_MAPPING = _LazyMapping(_registry.CONFIG, "config")
MODEL_MAPPING = _LazyMapping(_registry.MODEL, "base model")
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyMapping(_registry.CAUSAL_LM, "causal LM")


class SubConfigSpec(_FrameworkSubConfigSpec):
    """Resolve dynamic sub-configs through the package-local registry."""

    def get_config_class(self, model_type=None):
        return CONFIG_MAPPING[model_type]


class AutoConfig:
    @staticmethod
    def for_model(model_type, *args, **kwargs):
        return CONFIG_MAPPING[model_type](*args, **kwargs)


def _model_from_config(mapping, config, **kwargs):
    if not isinstance(config, PreTrainedConfig):
        raise TypeError(f"expected a config instance, got {type(config).__name__}")
    model_class = mapping[config.model_type]
    return model_class._from_config(config, **kwargs)


class AutoModel:
    @staticmethod
    def from_config(config, **kwargs):
        return _model_from_config(MODEL_MAPPING, config, **kwargs)


class AutoModelForCausalLM:
    @staticmethod
    def from_config(config, **kwargs):
        return _model_from_config(MODEL_FOR_CAUSAL_LM_MAPPING, config, **kwargs)


# ---------------------------------------------------------------------------
# Definition loaders. A definition binds one of these through LocalModelLoader;
# each resolves the checkpoint's model_type to a vendored class only.
# ---------------------------------------------------------------------------

_CLASS_INDEX = {}
for _table in (
    _registry.CONFIG, _registry.MODEL, _registry.CAUSAL_LM, _registry.IMAGE_TEXT_TO_TEXT,
    _registry.MULTIMODAL_LM, _registry.PRETRAINING, _registry.TEXT_TO_WAVEFORM, _registry.PROCESSOR,
    _registry.TOKENIZER, _registry.IMAGE_PROCESSOR, _registry.VIDEO_PROCESSOR, _registry.FEATURE_EXTRACTOR,
):
    for _module, _name in _table.values():
        _CLASS_INDEX.setdefault(_name, (_module, _name))


def _by_name(name):
    entry = _CLASS_INDEX.get(name)
    return None if entry is None else getattr(import_module(entry[0]), entry[1])


def _checkpoint_model_type(model_id_or_path, **kwargs):
    selection = {key: kwargs[key] for key in ("revision", "subfolder", "token", "cache_dir", "local_files_only") if key in kwargs}
    payload, _ = PreTrainedConfig.get_config_dict(model_id_or_path, **selection)
    model_type = payload.get("model_type")
    if not model_type:
        raise ValueError(f"{model_id_or_path}: config.json has no model_type")
    return model_type


def _require_vendored_config(config):
    if not type(config).__module__.startswith("gptqmodel.models.ports.native."):
        raise TypeError(
            f"{type(config).__name__} is not a vendored GPT-QModel config; load it with NativeConfig"
        )


class NativeConfig:
    """Load the checkpoint config with the vendored config class for its model_type."""

    @classmethod
    def from_pretrained(cls, model_id_or_path, **kwargs):
        config_class = CONFIG_MAPPING[_checkpoint_model_type(model_id_or_path, **kwargs)]
        return config_class.from_pretrained(model_id_or_path, **kwargs)


class _NativeModel:
    _table = None
    _kind = None

    @classmethod
    def _model_class(cls, config):
        _require_vendored_config(config)
        return _resolve(cls._table, config.model_type, cls._kind)

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls._model_class(config)._from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, model_id_or_path, *, config, **kwargs):
        return cls._model_class(config).from_pretrained(model_id_or_path, config=config, **kwargs)


class NativeModel(_NativeModel):
    _table, _kind = _registry.MODEL, "base model"


class NativeForCausalLM(_NativeModel):
    _table, _kind = _registry.CAUSAL_LM, "causal LM"


class NativeForImageTextToText(_NativeModel):
    _table, _kind = _registry.IMAGE_TEXT_TO_TEXT, "image-text-to-text model"


class NativeForMultimodalLM(_NativeModel):
    _table, _kind = _registry.MULTIMODAL_LM, "multimodal LM"


class NativeForPreTraining(_NativeModel):
    _table, _kind = _registry.PRETRAINING, "pretraining model"


class NativeForTextToWaveform(_NativeModel):
    _table, _kind = _registry.TEXT_TO_WAVEFORM, "text-to-waveform model"


class NativeTokenizer:
    """Vendored or framework tokenizer class; checkpoint Python is never imported."""

    @classmethod
    def tokenizer_class(cls, model_id_or_path, **kwargs):
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        try:
            declared = PreTrainedTokenizerBase.get_tokenizer_config(model_id_or_path, **kwargs).get("tokenizer_class")
        except Exception:  # missing/unreadable tokenizer_config.json: use the model_type default
            declared = None
        if declared:
            found = _by_name(declared) or _by_name(declared.removesuffix("Fast"))
            if found is not None:
                return found
        return _resolve(_registry.TOKENIZER, _checkpoint_model_type(model_id_or_path, **kwargs), "tokenizer")

    @classmethod
    def from_pretrained(cls, model_id_or_path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        return cls.tokenizer_class(model_id_or_path, **kwargs).from_pretrained(model_id_or_path, **kwargs)


_SUB_PROCESSOR_TABLES = {
    "image_processor": (_registry.IMAGE_PROCESSOR, "image processor"),
    "video_processor": (_registry.VIDEO_PROCESSOR, "video processor"),
    "feature_extractor": (_registry.FEATURE_EXTRACTOR, "feature extractor"),
}


class NativeProcessor:
    """Assemble the vendored processor from vendored sub-processors."""

    @classmethod
    def from_pretrained(cls, model_id_or_path, *, config=None, tokenizer=None, **kwargs):
        kwargs.pop("trust_remote_code", None)
        model_type = getattr(config, "model_type", None) or _checkpoint_model_type(model_id_or_path, **kwargs)
        processor_class = _resolve(_registry.PROCESSOR, model_type, "processor")
        arguments = []
        for attribute in processor_class.get_attributes():
            if "tokenizer" in attribute:
                arguments.append(tokenizer if tokenizer is not None else NativeTokenizer.from_pretrained(model_id_or_path, **kwargs))
                continue
            declared = getattr(processor_class, f"{attribute}_class", None)
            sub_class = _by_name(declared) if isinstance(declared, str) else None
            if sub_class is None:
                table, kind = _SUB_PROCESSOR_TABLES[attribute]
                sub_class = _resolve(table, model_type, kind)
            arguments.append(sub_class.from_pretrained(model_id_or_path, **kwargs))
        processor_dict, extra = processor_class.get_processor_dict(model_id_or_path, **kwargs)
        return processor_class.from_args_and_dict(arguments, processor_dict, **extra)


__all__ = [
    "NativeConfig",
    "NativeForCausalLM",
    "NativeForImageTextToText",
    "NativeForMultimodalLM",
    "NativeForPreTraining",
    "NativeForTextToWaveform",
    "NativeModel",
    "NativeProcessor",
    "NativeTokenizer",
    "CONFIG_MAPPING",
    "MODEL_FOR_CAUSAL_LM_MAPPING",
    "MODEL_MAPPING",
    "AutoConfig",
    "AutoModel",
    "AutoModelForCausalLM",
    "SubConfigSpec",
]
