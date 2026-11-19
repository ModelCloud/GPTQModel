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
from transformers.modeling_utils import PreTrainedModel

from . import _registry


class NativePreTrainedModel(PreTrainedModel):
    """Base for vendored Transformers 5.23 models.

    Transformers treats any class outside ``transformers.*`` as custom code and
    then skips its checkpoint conversion mappings (key renames, expert fusion),
    native weight-init and missing-key handling. Vendored classes are the
    library's own 5.23 code, so they must behave exactly like it.
    """

    @classmethod
    def is_custom_code(cls) -> bool:
        return cls.is_remote_code()


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
    """Resolve a class name to a vendored class or a Transformers framework class.

    Model-family classes from ``transformers.models`` are never returned.
    """

    entry = _CLASS_INDEX.get(name)
    if entry is not None:
        return getattr(import_module(entry[0]), entry[1])
    import transformers

    value = getattr(transformers, name, None)
    module = getattr(value, "__module__", "")
    if isinstance(value, type) and module.startswith("transformers.") and not module.startswith("transformers.models."):
        return value
    return None


def _checkpoint_json(model_id_or_path, filename, **kwargs):
    """Read a checkpoint JSON file (tokenizer/processor config) without importing checkpoint code."""

    import json

    from transformers.utils import cached_file

    selection = {key: kwargs[key] for key in ("revision", "subfolder", "token", "cache_dir", "local_files_only") if key in kwargs}
    path = cached_file(
        model_id_or_path,
        filename,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_connection_errors=False,
        **selection,
    )
    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def causal_lm_class_name(model_type):
    """Name of the vendored (= Transformers 5.23) causal-LM class for ``model_type``, or None."""

    entry = _registry.CAUSAL_LM.get(model_type)
    return entry[1] if entry is not None else None


def get_tokenizer_config(model_id_or_path, **kwargs):
    """Return the checkpoint's ``tokenizer_config.json`` (``{}`` when absent) as plain JSON."""

    return _checkpoint_json(model_id_or_path, "tokenizer_config.json", **kwargs)


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

    @staticmethod
    def _task_config(model_class, config):
        """Build from ``config.text_config`` when the task class is declared on it.

        Mirrors Transformers' auto factory: a composite checkpoint whose task
        class (for example a text-only causal LM) takes the text sub-config is
        built from that sub-config, carrying the parent's quantization config.
        """

        text_config_class = config.sub_configs.get("text_config", None)
        if text_config_class is None or getattr(model_class, "config_class", None) != text_config_class:
            return config
        parent_config = config
        config = config.get_text_config()
        parent_quant = getattr(parent_config, "quantization_config", None)
        if parent_quant is not None:
            config.quantization_config = parent_quant
        return config

    @classmethod
    def _from_config(cls, config, **kwargs):
        model_class = cls._model_class(config)
        return model_class._from_config(cls._task_config(model_class, config), **kwargs)

    @classmethod
    def from_pretrained(cls, model_id_or_path, *, config, **kwargs):
        model_class = cls._model_class(config)
        return model_class.from_pretrained(model_id_or_path, config=cls._task_config(model_class, config), **kwargs)


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
        declared = _checkpoint_json(model_id_or_path, "tokenizer_config.json", **kwargs).get("tokenizer_class")
        if isinstance(declared, str) and declared:
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

_SELECTION_KEYS = ("revision", "subfolder", "token", "cache_dir", "local_files_only", "force_download", "proxies")


def _sub_processor_framework(attribute):
    """(framework base class, checkpoint dict reader, declared-type key) for a sub-processor."""

    if attribute == "image_processor":
        from transformers.image_processing_base import ImageProcessingMixin

        return ImageProcessingMixin, ImageProcessingMixin.get_image_processor_dict, "image_processor_type"
    if attribute == "video_processor":
        from transformers.video_processing_utils import BaseVideoProcessor

        return BaseVideoProcessor, BaseVideoProcessor.get_video_processor_dict, "video_processor_type"
    from transformers.feature_extraction_utils import FeatureExtractionMixin

    return FeatureExtractionMixin, FeatureExtractionMixin.get_feature_extractor_dict, "feature_extractor_type"


def _image_processor_by_name(name, backend):
    """Transformers 5.23 image-processor naming: ``X`` is the torchvision backend,
    ``XPil`` the PIL backend, and a legacy ``XFast`` means ``X``. The requested
    backend is tried first, then the other one (as ``_load_class_with_fallback``)."""

    base = name.removesuffix("Fast")
    base = base.removesuffix("Pil") if base.endswith("ImageProcessorPil") else base
    torchvision_class = _by_name(base)
    pil_class = _by_name(base + "Pil")
    if pil_class is None and torchvision_class is not None and torchvision_class.__module__.startswith("gptqmodel."):
        # Vendored families keep the PIL backend next to the torchvision one
        # (``image_processing_<family>`` / ``image_processing_pil_<family>``).
        module = torchvision_class.__module__
        package, _, leaf = module.rpartition(".")
        for candidate in (module, f"{package}.{leaf.replace('image_processing_', 'image_processing_pil_', 1)}"):
            try:
                pil_class = getattr(import_module(candidate), base + "Pil", None)
            except ModuleNotFoundError:
                pil_class = None
            if pil_class is not None:
                break
    ordered = (pil_class, torchvision_class) if backend == "pil" else (torchvision_class, pil_class)
    for found in ordered:
        if isinstance(found, type):
            return found
    return _by_name(name) if name.endswith("Fast") else None


def _declared_sub_processor_class(attribute, model_id_or_path, config, backend, **kwargs):
    """Class named by the checkpoint for one sub-processor, resolved package-locally.

    Mirrors Transformers 5.23 ``AutoImageProcessor`` / ``AutoVideoProcessor`` /
    ``AutoFeatureExtractor``: the declared ``*_type`` in the nested
    ``processor_config.json`` entry or the legacy per-processor file, then the
    legacy cross-kind inference, then the model config's ``*_type``. Names are
    resolved through ``_by_name`` only (vendored or framework classes), so a
    checkpoint can never select a ``transformers.models`` class; ``None`` when
    nothing resolvable is declared.
    """

    base_class, read_dict, type_key = _sub_processor_framework(attribute)
    selection = {key: kwargs[key] for key in _SELECTION_KEYS if key in kwargs}
    try:
        payload, _ = read_dict(model_id_or_path, **selection)
    except (OSError, ValueError):
        payload = {}
    names = [payload.get(type_key)]
    if attribute == "image_processor" and isinstance(payload.get("feature_extractor_type"), str):
        names.append(payload["feature_extractor_type"].replace("FeatureExtractor", "ImageProcessor"))
    if attribute == "video_processor" and isinstance(payload.get("image_processor_type"), str):
        names.append(payload["image_processor_type"].replace("ImageProcessor", "VideoProcessor"))
    names.append(getattr(config, type_key, None))
    for name in names:
        if not isinstance(name, str) or not name:
            continue
        found = _image_processor_by_name(name, backend) if attribute == "image_processor" else _by_name(name)
        if isinstance(found, type) and issubclass(found, base_class):
            return found
    return None


def _resolve_backend(backend, use_fast):
    if backend is None and use_fast is not None:
        backend = "torchvision" if use_fast else "pil"
    if backend is None:
        from transformers.utils import is_torchvision_available

        backend = "torchvision" if is_torchvision_available() else "pil"
    return backend


class NativeProcessor:
    """Assemble the vendored processor from vendored sub-processors."""

    @classmethod
    def sub_processor_class(cls, processor_class, attribute, model_id_or_path, *, model_type, config=None, backend=None, **kwargs):
        """Package-local class for one non-tokenizer sub-processor.

        Order: the checkpoint's declared type (resolved through ``_by_name``),
        then the vendored ``model_type`` table, then the processor class's own
        ``<attribute>_class`` name. Never a ``transformers.models`` class.
        """

        sub_class = _declared_sub_processor_class(attribute, model_id_or_path, config, _resolve_backend(backend, None), **kwargs)
        if sub_class is not None:
            return sub_class
        table, kind = _SUB_PROCESSOR_TABLES[attribute]
        if model_type in table:
            return _resolve(table, model_type, kind)
        declared = getattr(processor_class, f"{attribute}_class", None)
        sub_class = _by_name(declared) if isinstance(declared, str) else None
        if sub_class is not None:
            return sub_class
        return _resolve(table, model_type, kind)

    @classmethod
    def from_pretrained(cls, model_id_or_path, *, config=None, tokenizer=None, **kwargs):
        kwargs.pop("trust_remote_code", None)
        backend = _resolve_backend(kwargs.pop("backend", None), kwargs.pop("use_fast", None))
        model_type = getattr(config, "model_type", None) or _checkpoint_model_type(model_id_or_path, **kwargs)
        declared = None
        for filename in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json"):
            declared = _checkpoint_json(model_id_or_path, filename, **kwargs).get("processor_class")
            if isinstance(declared, str) and declared:
                break
        processor_class = _by_name(declared) if declared else None
        if processor_class is None:
            processor_class = _resolve(_registry.PROCESSOR, model_type, "processor")
        arguments = []
        for attribute in processor_class.get_attributes():
            if "tokenizer" in attribute:
                arguments.append(tokenizer if tokenizer is not None else NativeTokenizer.from_pretrained(model_id_or_path, **kwargs))
                continue
            sub_class = cls.sub_processor_class(
                processor_class, attribute, model_id_or_path, model_type=model_type, config=config, backend=backend, **kwargs
            )
            arguments.append(sub_class.from_pretrained(model_id_or_path, **kwargs))
        processor_dict, extra = processor_class.get_processor_dict(model_id_or_path, **kwargs)
        return processor_class.from_args_and_dict(arguments, processor_dict, **extra)


__all__ = [
    "NativePreTrainedModel",
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
    "causal_lm_class_name",
    "get_tokenizer_config",
]
