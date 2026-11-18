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


__all__ = [
    "CONFIG_MAPPING",
    "MODEL_FOR_CAUSAL_LM_MAPPING",
    "MODEL_MAPPING",
    "AutoConfig",
    "AutoModel",
    "AutoModelForCausalLM",
    "SubConfigSpec",
]
