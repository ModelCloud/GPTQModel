# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Definition-owned, lazy loaders for modeling code shipped with GPT-QModel."""

from dataclasses import dataclass
from importlib import import_module


def _local_class(path: str):
    module, _, name = path.rpartition(".")
    if not module.startswith("gptqmodel.") or not name.isidentifier():
        raise ValueError(f"Local model classes must belong to gptqmodel: {path!r}")
    return getattr(import_module(module), name)


@dataclass(frozen=True)
class LocalModelLoader:
    """Use as a definition's ``loader``; checkpoint ``auto_map`` is never read.

    Both paths are trusted package metadata, never values from config.json.
    Importing a definition does not import its optional modeling dependencies.
    """

    config_class: str
    model_class: str

    def __post_init__(self):
        for path in (self.config_class, self.model_class):
            module, _, name = path.rpartition(".")
            if not module.startswith("gptqmodel.") or not name.isidentifier():
                raise ValueError(f"Local model classes must belong to gptqmodel: {path!r}")

    def load_config(self, model_id_or_path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        return _local_class(self.config_class).from_pretrained(model_id_or_path, **kwargs)

    def from_config(self, config, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        return _local_class(self.model_class)._from_config(config, **kwargs)

    def from_pretrained(self, model_id_or_path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        if kwargs.get("config") is None:
            kwargs["config"] = self.load_config(model_id_or_path, **kwargs)
        return _local_class(self.model_class).from_pretrained(model_id_or_path, **kwargs)


def local_model_loader(model_type):
    """Resolve only definition-owned metadata, with no global HF registration."""
    from .auto import MODEL_MAP_CASEFOLD

    definition = MODEL_MAP_CASEFOLD.get(str(model_type or "").lower())
    loader = getattr(definition, "loader", None)
    return loader if isinstance(loader, LocalModelLoader) else None
