# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Definition-owned, lazy loaders for modeling code shipped with GPT-QModel."""

from dataclasses import dataclass
from importlib import import_module


_OPTIONAL_MODULE_EXTRAS = {
    "torchvision": "vision",
    "cv2": "video",
    "qwen_vl_utils": "video",
    "soundfile": "audio",
    "scipy": "audio",
    "librosa": "audio",
}


def _local_class(path: str):
    module, _, name = path.rpartition(".")
    if not module.startswith("gptqmodel.") or not name.isidentifier():
        raise ValueError(f"Local model classes must belong to gptqmodel: {path!r}")
    try:
        return getattr(import_module(module), name)
    except ModuleNotFoundError as exc:
        extra = _OPTIONAL_MODULE_EXTRAS.get(exc.name)
        if extra is None:
            raise
        raise ImportError(f"{name} requires {exc.name}. Install it with `pip install 'gptqmodel[{extra}]'`.") from exc


@dataclass(frozen=True)
class LocalModelLoader:
    """Use as a definition's ``loader``; checkpoint ``auto_map`` is never read.

    Class paths are trusted package metadata, never values from config.json.
    Importing a definition does not import its optional modeling dependencies.
    """

    config_class: str
    model_class: str
    processor_class: str | None = None
    tokenizer_class: str | None = None
    # Explicit opt-in for released config.json files that omit model_type.
    # These are trusted class names supplied by the package, never import paths
    # or a heuristic derived from a checkpoint directory/model identifier.
    config_architectures: tuple[str, ...] = ()
    # Immutable, package-declared normalization defaults, consumed by Tokenicer
    # after class selection. Explicit caller options take precedence.
    tokenicer_options: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        if not isinstance(self.tokenicer_options, tuple) or any(
            not isinstance(option, tuple) or len(option) != 2
            or not isinstance(option[0], str) or not option[0].isidentifier()
            or option[0] in {"config", "model_config", "trust_remote_code", "code_revision"}
            for option in self.tokenicer_options
        ) or len(dict(self.tokenicer_options)) != len(self.tokenicer_options):
            raise ValueError("Local tokenicer_options must be unique normalization option pairs")
        if not isinstance(self.config_architectures, tuple) or any(
            not isinstance(name, str) or not name.isidentifier() for name in self.config_architectures
        ):
            raise ValueError("Local config_architectures must be a tuple of class names")
        paths = (self.config_class, self.model_class)
        if self.processor_class is not None:
            paths += (self.processor_class,)
        if self.tokenizer_class is not None:
            paths += (self.tokenizer_class,)
        for path in paths:
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

    def load_processor(self, model_id_or_path, **kwargs):
        if self.processor_class is None:
            raise ValueError("This local model loader does not declare a processor class")
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        return _local_class(self.processor_class).from_pretrained(model_id_or_path, **kwargs)

    def load_tokenizer(self, model_id_or_path, **kwargs):
        from tokenicer import Tokenicer

        if self.tokenizer_class is None:
            raise ValueError("This local model loader does not declare a tokenizer class")
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        kwargs = Tokenicer.prepare_load_kwargs(model_id_or_path, **kwargs)
        return _local_class(self.tokenizer_class).from_pretrained(model_id_or_path, **kwargs)


def local_model_loader(model_type):
    """Resolve only definition-owned metadata, with no global HF registration."""
    from .auto import MODEL_MAP_CASEFOLD

    definition = MODEL_MAP_CASEFOLD.get(str(model_type or "").lower())
    loader = getattr(definition, "loader", None)
    return loader if isinstance(loader, LocalModelLoader) else None


def local_model_loader_for_config(payload):
    """Select trusted package metadata without executing checkpoint code.

    A present model_type is authoritative, including invalid explicit values.
    Legacy inference requires every architecture to name the same unique
    package-declared loader; partial matches and conflicting declarations fail
    before either trust handling or a Transformers dynamic-module lookup.
    """
    if "model_type" in payload:
        return local_model_loader(payload["model_type"])
    architectures = payload.get("architectures")
    if not isinstance(architectures, (list, tuple)) or not architectures:
        return None
    names = {name for name in architectures if isinstance(name, str)}

    from .auto import MODEL_MAP_CASEFOLD

    candidates = []
    for definition in MODEL_MAP_CASEFOLD.values():
        loader = getattr(definition, "loader", None)
        if not isinstance(loader, LocalModelLoader) or not names.intersection(loader.config_architectures):
            continue
        if loader not in candidates:
            candidates.append(loader)
    if not candidates:
        # Unknown checkpoint families retain the ordinary explicit trust+bypass
        # policy. Merely having architectures metadata does not grant local trust.
        return None
    if any(not isinstance(name, str) for name in architectures) or len(candidates) != 1 or not names.issubset(candidates[0].config_architectures):
        raise ValueError("Ambiguous or conflicting package model architectures in checkpoint config")
    return candidates[0]
