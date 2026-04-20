# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import inspect
import json
import os
import sys
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import torch
import transformers
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedConfig,
    PreTrainedModel,
)

from ..nn_modules.qlinear.gguf import (
    PRISM_Q1_0_G128_BLOCK_SIZE,
    PRISM_Q1_0_G128_NAME,
    PRISM_Q1_0_G128_TYPE_SIZE,
    PRISM_Q1_0_G128_VALUE,
    _dequantize_prism_q1_0_g128,
    _is_prism_q1_0_g128,
)
from ..utils import _MONKEY_PATCH_LOCK, internal_gguf


# Compatibility wrapper for no_init_weights across different transformers versions
# transformers >= 5.0.0: from transformers.initialization import no_init_weights
# transformers < 5.0.0: from transformers.modeling_utils import no_init_weights
try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

from ..utils.logger import setup_logger


__all__ = [
    "no_init_weights",
    "suspend_hf_weight_init",
    "get_hf_config_dtype",
    "normalize_torch_dtype_kwarg",
    "normalize_hf_config_compat",
    "prepare_remote_code_compat",
    "prepare_remote_model_init_compat",
    "has_native_transformers_causallm_support",
    "get_hf_gguf_load_kwargs",
    "normalize_model_id_or_path_for_hf_gguf",
    "resolve_trust_remote_code",
    "set_hf_config_dtype",
    "load_tokenizer",
]

log = setup_logger()
_TRUST_REMOTE_CODE_OVERRIDE_WARNED: set[tuple[str, str, str]] = set()
_MISSING = object()
INTERNAL_HF_GGUF_FILE_KWARG = "_gptqmodel_hf_gguf_file"
_DENSE_MODEL_FILE_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
_INTERNAL_GGUF_TORCH_LOADER_ENV = "GPTQMODEL_INTERNAL_GGUF_TORCH_LOADER"
_FALSEY_ENV_VALUES = {"", "0", "false", "off", "no"}


def get_hf_config_dtype(config: Any) -> Optional[torch.dtype]:
    dtype = getattr(config, "dtype", None)
    if dtype is None:
        dtype = getattr(config, "torch_dtype", None)

    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype

    # If provided as a string (e.g., "float16"), resolve it via torch namespace
    if isinstance(dtype, str):
        try:
            return getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype string: {dtype}")

    raise ValueError(f"dtype must be torch.dtype or str, but got {dtype} (type={type(dtype)})")


def set_hf_config_dtype(config: Any, dtype: torch.dtype) -> None:
    current_dtype = get_hf_config_dtype(config)
    if current_dtype == dtype:
        return

    try:
        setattr(config, "dtype", dtype)
    except Exception:
        if getattr(config, "torch_dtype", None) != dtype:
            setattr(config, "torch_dtype", dtype)


def normalize_torch_dtype_kwarg(
    kwargs: dict[str, Any],
    *,
    api_name: str,
    explicit_dtype: Any = _MISSING,
) -> Any:
    current_dtype = explicit_dtype
    if explicit_dtype is _MISSING:
        current_dtype = kwargs.pop("dtype", _MISSING)

    torch_dtype = kwargs.pop("torch_dtype", _MISSING)
    if torch_dtype is _MISSING:
        if explicit_dtype is _MISSING and current_dtype is not _MISSING:
            kwargs["dtype"] = current_dtype
        return current_dtype

    log.warn.once(f"{api_name}: `torch_dtype` is deprecated; use `dtype` instead.")

    if current_dtype is _MISSING or current_dtype is None or current_dtype == "auto":
        current_dtype = torch_dtype
    elif current_dtype != torch_dtype:
        raise ValueError(
            f"{api_name}: received both `dtype` and deprecated `torch_dtype` with different values. "
            "Please pass only `dtype`."
        )

    if explicit_dtype is _MISSING:
        kwargs["dtype"] = current_dtype
    return current_dtype


@contextmanager
def suspend_hf_weight_init():
    """Disable HF/torch parameter init temporarily and always restore globals."""

    def _skip_init(*args, **kwargs):
        return None

    original_kaiming_uniform = torch.nn.init.kaiming_uniform_
    original_uniform = torch.nn.init.uniform_
    original_normal = torch.nn.init.normal_

    modeling_utils = transformers.modeling_utils
    had_init_flag = hasattr(modeling_utils, "_init_weights")
    original_init_flag = getattr(modeling_utils, "_init_weights", None)

    torch.nn.init.kaiming_uniform_ = _skip_init
    torch.nn.init.uniform_ = _skip_init
    torch.nn.init.normal_ = _skip_init
    modeling_utils._init_weights = False

    try:
        with no_init_weights():
            yield
    finally:
        torch.nn.init.kaiming_uniform_ = original_kaiming_uniform
        torch.nn.init.uniform_ = original_uniform
        torch.nn.init.normal_ = original_normal
        if had_init_flag:
            modeling_utils._init_weights = original_init_flag
        elif hasattr(modeling_utils, "_init_weights"):
            delattr(modeling_utils, "_init_weights")


def _raise_public_gguf_file_arg_error(api_name: str) -> None:
    raise TypeError(
        f"{api_name} does not accept `gguf_file`. Pass the GGUF checkpoint as `model_id_or_path`, "
        "or pass a model directory / repo containing a single GGUF file."
    )


def get_hf_gguf_load_kwargs(kwargs: dict[str, Any]) -> dict[str, str]:
    gguf_file = kwargs.get(INTERNAL_HF_GGUF_FILE_KWARG)
    if gguf_file is None:
        return {}
    return {"gguf_file": gguf_file}


def _normalize_repo_file_paths(file_names) -> list[str]:
    return [str(file_name).replace("\\", "/") for file_name in file_names]


def _infer_single_gguf_file(file_names) -> Optional[str]:
    normalized_files = _normalize_repo_file_paths(file_names)
    gguf_files = sorted(file_name for file_name in normalized_files if file_name.lower().endswith(".gguf"))
    if len(gguf_files) != 1:
        return None

    dense_files = [
        file_name
        for file_name in normalized_files
        if file_name.lower().endswith(_DENSE_MODEL_FILE_EXTENSIONS)
    ]
    if dense_files:
        return None

    return gguf_files[0]


def _iter_local_repo_files(root_dir: str) -> list[str]:
    repo_files = []
    for current_root, _dirs, files in os.walk(root_dir):
        for file_name in files:
            full_path = os.path.join(current_root, file_name)
            repo_files.append(os.path.relpath(full_path, root_dir).replace(os.sep, "/"))
    return repo_files


@lru_cache(maxsize=None)
def _resolve_hf_gguf_artifact(model_id_or_path: str) -> Optional[tuple[str, str]]:
    if os.path.isfile(model_id_or_path) and model_id_or_path.lower().endswith(".gguf"):
        model_root = os.path.dirname(os.path.abspath(model_id_or_path)) or "."
        return model_root, os.path.basename(model_id_or_path)

    if os.path.isdir(model_id_or_path):
        inferred_gguf_file = _infer_single_gguf_file(_iter_local_repo_files(model_id_or_path))
        if inferred_gguf_file is not None:
            return os.path.normpath(model_id_or_path), inferred_gguf_file
        return None

    try:
        from .hub import list_repo_files
    except Exception:
        return None

    try:
        repo_files = list_repo_files(repo_id=model_id_or_path)
    except Exception:
        return None

    inferred_gguf_file = _infer_single_gguf_file(repo_files)
    if inferred_gguf_file is None:
        return None
    return model_id_or_path, inferred_gguf_file


def _transformers_has_native_prism_gguf_support() -> bool:
    try:
        import transformers.modeling_gguf_pytorch_utils as gguf_utils
    except Exception:
        return False

    return hasattr(gguf_utils, "_dequantize_prism_q1_0_g128")


def _internal_gguf_torch_loader_enabled() -> bool:
    raw = os.getenv(_INTERNAL_GGUF_TORCH_LOADER_ENV)
    if raw is not None:
        return str(raw).strip().lower() not in _FALSEY_ENV_VALUES
    return bool(os.getenv("GPTQMODEL_INTERNAL_GGUF_DEQUANT_DEVICE", "").strip())


def _load_gguf_checkpoint_torch_direct(
    *,
    gguf_utils,
    original_load_gguf_checkpoint,
    gguf_checkpoint_path,
    return_tensors: bool = False,
    model_to_load=None,
):
    if not return_tensors or model_to_load is None or not _internal_gguf_torch_loader_enabled():
        return original_load_gguf_checkpoint(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
        )

    parsed_parameters = original_load_gguf_checkpoint(
        gguf_checkpoint_path,
        return_tensors=False,
        model_to_load=model_to_load,
    )
    config = parsed_parameters.get("config", {})
    model_type = config.get("model_type")
    if model_type != internal_gguf.MODEL_ARCH_QWEN3:
        return original_load_gguf_checkpoint(
            gguf_checkpoint_path,
            return_tensors=True,
            model_to_load=model_to_load,
        )

    processor_cls = gguf_utils.TENSOR_PROCESSORS.get(model_type, gguf_utils.TensorProcessor)
    if processor_cls is not gguf_utils.TensorProcessor:
        return original_load_gguf_checkpoint(
            gguf_checkpoint_path,
            return_tensors=True,
            model_to_load=model_to_load,
        )

    processor = processor_cls(config=config)
    tensor_key_mapping = gguf_utils.get_gguf_hf_weights_map(model_to_load, processor)
    target_device = internal_gguf._resolve_torch_dequant_device()
    reader = internal_gguf.GGUFReader(gguf_checkpoint_path)
    parsed_parameters["tensors"] = {}

    for tensor in gguf_utils.tqdm(reader.tensors, desc="Converting GGUF tensors directly to torch..."):
        name = tensor.name
        weights = internal_gguf.dequantize_to_torch(
            tensor.data,
            tensor.tensor_type,
            device=target_device,
        )

        result = processor.process(
            weights=weights,
            name=name,
            tensor_key_mapping=tensor_key_mapping,
            parsed_parameters=parsed_parameters,
        )

        weights = result.weights
        name = result.name
        if name not in tensor_key_mapping:
            continue

        if not torch.is_tensor(weights):
            weights = torch.from_numpy(np.array(weights, copy=True, order="C"))
            if target_device is not None:
                weights = weights.to(device=target_device)

        parsed_parameters["tensors"][tensor_key_mapping[name]] = weights.contiguous()

    return parsed_parameters


def _patch_transformers_internal_gguf_torch_loader(gguf_utils) -> None:
    with _MONKEY_PATCH_LOCK:
        if getattr(gguf_utils, "_GPTQMODEL_INTERNAL_GGUF_TORCH_LOADER_PATCHED", False):
            return

        original_load_gguf_checkpoint = gguf_utils.load_gguf_checkpoint

        def _wrapped_load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False, model_to_load=None):
            try:
                return _load_gguf_checkpoint_torch_direct(
                    gguf_utils=gguf_utils,
                    original_load_gguf_checkpoint=original_load_gguf_checkpoint,
                    gguf_checkpoint_path=gguf_checkpoint_path,
                    return_tensors=return_tensors,
                    model_to_load=model_to_load,
                )
            except Exception as exc:
                log.debug(
                    "HF: internal torch GGUF loader fell back to the stock loader for `%s`: %s",
                    gguf_checkpoint_path,
                    exc,
                )
                return original_load_gguf_checkpoint(
                    gguf_checkpoint_path,
                    return_tensors=return_tensors,
                    model_to_load=model_to_load,
                )

        gguf_utils._gptqmodel_original_load_gguf_checkpoint = original_load_gguf_checkpoint
        gguf_utils.load_gguf_checkpoint = _wrapped_load_gguf_checkpoint
        gguf_utils._GPTQMODEL_INTERNAL_GGUF_TORCH_LOADER_PATCHED = True


def _patch_transformers_prism_gguf_compat(*, api_name: str) -> None:
    try:
        import transformers.modeling_gguf_pytorch_utils as gguf_utils
        from transformers.utils import import_utils as hf_import_utils
    except Exception:
        return

    with _MONKEY_PATCH_LOCK:
        internal_gguf.install_runtime()
        _patch_transformers_internal_gguf_torch_loader(gguf_utils)

        def _is_gguf_available(*_args, **_kwargs) -> bool:
            return True

        if not gguf_utils.is_gguf_available():
            gguf_utils.is_gguf_available = _is_gguf_available
            if hasattr(hf_import_utils, "is_gguf_available"):
                hf_import_utils.is_gguf_available = _is_gguf_available

        if _transformers_has_native_prism_gguf_support():
            return

        gguf_utils.PRISM_Q1_0_G128_NAME = PRISM_Q1_0_G128_NAME
        gguf_utils.PRISM_Q1_0_G128_VALUE = PRISM_Q1_0_G128_VALUE
        gguf_utils.PRISM_Q1_0_G128_BLOCK_SIZE = PRISM_Q1_0_G128_BLOCK_SIZE
        gguf_utils.PRISM_Q1_0_G128_TYPE_SIZE = PRISM_Q1_0_G128_TYPE_SIZE
        gguf_utils._is_prism_q1_0_g128 = _is_prism_q1_0_g128
        gguf_utils._dequantize_prism_q1_0_g128 = _dequantize_prism_q1_0_g128

        log.warning(
            "HF: installed transformers lacks native Prism GGUF support; GPT-QModel registered its internal "
            "GGUF runtime and local Q1_0_g128 compatibility patch for `%s`.",
            api_name,
        )


def normalize_model_id_or_path_for_hf_gguf(
    model_id_or_path: Optional[str],
    kwargs: dict[str, Any],
    *,
    api_name: str,
) -> Optional[str]:
    if INTERNAL_HF_GGUF_FILE_KWARG in kwargs:
        return model_id_or_path

    if kwargs.pop("gguf_file", None) is not None:
        _raise_public_gguf_file_arg_error(api_name)

    if model_id_or_path is None:
        return None

    resolved = _resolve_hf_gguf_artifact(str(model_id_or_path))
    if resolved is None:
        return model_id_or_path

    normalized_model_id_or_path, gguf_file = resolved
    _patch_transformers_prism_gguf_compat(api_name=api_name)
    kwargs[INTERNAL_HF_GGUF_FILE_KWARG] = gguf_file
    return normalized_model_id_or_path


@lru_cache(maxsize=None)
def _detect_native_transformers_causallm_support(model_id_or_path: str) -> tuple[bool, Optional[str], Optional[str]]:
    config_load_kwargs: dict[str, Any] = {}
    normalized_model_id_or_path = normalize_model_id_or_path_for_hf_gguf(
        model_id_or_path,
        config_load_kwargs,
        api_name="_detect_native_transformers_causallm_support",
    )
    try:
        config = AutoConfig.from_pretrained(
            normalized_model_id_or_path,
            trust_remote_code=False,
            **get_hf_gguf_load_kwargs(config_load_kwargs),
        )
    except Exception as exc:
        log.debug("HF: native transformers support check failed for `%s`: %s", normalized_model_id_or_path, exc)
        return False, None, None

    model_type = getattr(config, "model_type", None)
    try:
        model_cls = AutoModelForCausalLM._model_mapping[type(config)]
    except Exception as exc:
        log.debug(
            "HF: config `%s` for `%s` has no native AutoModelForCausalLM mapping: %s",
            type(config).__name__,
            normalized_model_id_or_path,
            exc,
        )
        return False, model_type, None

    return True, model_type, getattr(model_cls, "__name__", str(model_cls))


def resolve_trust_remote_code(model_id_or_path: Optional[str], *, trust_remote_code: bool) -> bool:
    if not trust_remote_code or not model_id_or_path:
        return trust_remote_code

    native_supported, model_type, model_cls_name = _detect_native_transformers_causallm_support(str(model_id_or_path))
    if not native_supported:
        return True

    warning_key = (str(model_id_or_path), model_type or "unknown", model_cls_name or "unknown")
    if warning_key not in _TRUST_REMOTE_CODE_OVERRIDE_WARNED:
        log.warning(
            "HF: overriding trust_remote_code=True to False for `%s` because model_type `%s` is integrated in installed transformers as `%s`.",
            model_id_or_path,
            model_type or "unknown",
            model_cls_name or "unknown",
        )
        _TRUST_REMOTE_CODE_OVERRIDE_WARNED.add(warning_key)

    return False


def has_native_transformers_causallm_support(model_id_or_path: Optional[str]) -> bool:
    if not model_id_or_path:
        return False

    native_supported, _, _ = _detect_native_transformers_causallm_support(str(model_id_or_path))
    return native_supported

def _resolve_input_embedding_weight_name(model: PreTrainedModel) -> Optional[str]:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        return None

    try:
        input_embeddings = get_input_embeddings()
    except Exception:
        return None

    if input_embeddings is None:
        return None

    weight = getattr(input_embeddings, "weight", None)
    if weight is None:
        return None

    for name, param in model.named_parameters(remove_duplicate=False):
        if param is weight:
            return name

    for name, module in model.named_modules(remove_duplicate=False):
        if module is input_embeddings:
            return f"{name}.weight" if name else "weight"

    return None


# Older remote model files sometimes store `_tied_weights_keys` as a plain list
# like `["lm_head.weight"]`. transformers 5.x now expects `{target: source}`,
# and otherwise later save/load helpers fail with `'list' object has no attribute 'keys'`.
def _resolve_legacy_tied_weights_mapping(model: PreTrainedModel, tied_mapping) -> dict[str, str]:
    if not isinstance(tied_mapping, (list, tuple, set)):
        return {}

    if not getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        return {}

    source_name = _resolve_input_embedding_weight_name(model)
    if source_name is None:
        return {}

    return {
        # Legacy list entries only name the tied target, so resolve them back
        # to the input embedding weight name expected by transformers 5.x.
        target_name: source_name
        for target_name in tied_mapping
        if isinstance(target_name, str) and target_name != source_name
    }


# Rewrite legacy list-based `_tied_weights_keys` in-place so transformers 5.x
# save/load code stops crashing on older trust_remote_code models that still use
# the pre-5.x list format.
def _normalize_legacy_tied_weights_keys(model: PreTrainedModel) -> None:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return

    try:
        modules_iter = named_modules(remove_duplicate=False)
    except TypeError:
        modules_iter = named_modules()

    for _name, submodule in modules_iter:
        tied_mapping = getattr(submodule, "_tied_weights_keys", None)
        if not isinstance(tied_mapping, (list, tuple, set)):
            continue

        if isinstance(submodule, PreTrainedModel):
            submodule._tied_weights_keys = _resolve_legacy_tied_weights_mapping(submodule, tied_mapping)
        else:
            submodule._tied_weights_keys = {}


# Bridge a few transformers 5.x API changes so older trust_remote_code model
# files still import and initialize without editing the cached remote source.
def _patch_transformers_remote_code_compat() -> None:
    try:
        from transformers.utils import import_utils
    except Exception:
        return

    try:
        from transformers import cache_utils
    except Exception:
        cache_utils = None

    try:
        from transformers.generation import utils as generation_utils
    except Exception:
        generation_utils = None

    try:
        from transformers import utils
    except Exception:
        utils = None

    try:
        import transformers.modeling_rope_utils as rope_utils
    except Exception:
        rope_utils = None

    import transformers.utils.generic as generic
    with _MONKEY_PATCH_LOCK:
        if not hasattr(import_utils, "is_torch_fx_available"):
            # transformers 5.x removed `import_utils.is_torch_fx_available`, but
            # older remote model files still import it during module import.
            def is_torch_fx_available() -> bool:
                return hasattr(torch, "fx")

            import_utils.is_torch_fx_available = is_torch_fx_available

        if utils is not None and not hasattr(utils, "is_flash_attn_greater_or_equal_2_10"):
            legacy_flash_attn_probe = getattr(utils, "is_flash_attn_greater_or_equal", None)

            if legacy_flash_attn_probe:
                # Older trust_remote_code model files import the removed
                # `is_flash_attn_greater_or_equal_2_10` helper from
                # `transformers.utils`; newer transformers only expose the generic
                # comparator.
                def is_flash_attn_greater_or_equal_2_10() -> bool:
                    return bool(legacy_flash_attn_probe("2.1.0"))

                utils.is_flash_attn_greater_or_equal_2_10 = is_flash_attn_greater_or_equal_2_10

        if rope_utils is not None and "default" not in getattr(rope_utils, "ROPE_INIT_FUNCTIONS", {}):
            # transformers 5.x removed the legacy `"default"` RoPE entrypoint,
            # but older trust_remote_code model files still resolve it directly.
            # Recreate the unscaled/base initializer instead of aliasing to
            # `"linear"` so configs do not need an artificial `factor=1.0`.
            def _compute_default_rope_parameters_compat(
                config: Optional["PreTrainedConfig"] = None,
                device: Optional["torch.device"] = None,
                seq_len: int | None = None,
                layer_type: str | None = None,
            ) -> tuple["torch.Tensor", float]:
                del seq_len
                if config is None:
                    raise ValueError("`config` is required to compute default RoPE parameters.")

                standardize_rope_params = getattr(config, "standardize_rope_params", None)
                if callable(standardize_rope_params):
                    standardize_rope_params()

                rope_parameters = getattr(config, "rope_parameters", None)
                if layer_type is not None and isinstance(rope_parameters, dict):
                    rope_parameters = rope_parameters.get(layer_type, rope_parameters)

                rope_theta = None
                partial_rotary_factor = 1.0
                if isinstance(rope_parameters, dict):
                    rope_theta = rope_parameters.get("rope_theta")
                    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", partial_rotary_factor)

                if rope_theta is None:
                    rope_theta = getattr(config, "rope_theta", None)
                    if rope_theta is None:
                        rope_theta = getattr(config, "default_theta", 10_000.0)

                head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                dim = int(head_dim * partial_rotary_factor)
                attention_factor = 1.0
                inv_freq = 1.0 / (
                    rope_theta
                    ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
                )
                return inv_freq, attention_factor

            rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters_compat

        if cache_utils is not None and not hasattr(cache_utils, "SlidingWindowCache") and hasattr(cache_utils, "StaticCache"):
            # transformers 5.x folds sliding-window behavior into StaticCache
            # layers, but older remote code still imports the legacy symbol.
            cache_utils.SlidingWindowCache = cache_utils.StaticCache

        if not hasattr(generic, "check_model_inputs"):
            # transformers 5.x removed `transformers.utils.generic.check_model_inputs`, but
            # older remote model files still import it during module import.
            def check_model_inputs(func=None, **kwargs):
                def wrapper(fn):
                    def inner(self, *args, **kwargs):
                        return fn(self, *args, **kwargs)

                    return inner

                return wrapper(func) if func else wrapper

            generic.check_model_inputs = check_model_inputs

        if cache_utils is not None and not hasattr(cache_utils, "HybridCache") and hasattr(cache_utils, "StaticCache"):
            # transformers 5.x also collapsed the legacy HybridCache entrypoint
            # into StaticCache, which already instantiates hybrid/sliding layers
            # based on the model config.
            cache_utils.HybridCache = cache_utils.StaticCache

        cache_base_cls = getattr(cache_utils, "Cache", None) if cache_utils is not None else None
        if cache_base_cls is not None and not hasattr(cache_base_cls, "get_max_length") and hasattr(cache_base_cls, "get_max_cache_shape"):
            # Older remote decoders expect `get_max_length()`, while newer
            # transformers renamed that API to `get_max_cache_shape()`.
            def get_max_length(self, layer_idx: int = 0) -> Optional[int]:
                max_length = self.get_max_cache_shape(layer_idx)
                return None if max_length is None or max_length < 0 else max_length

            cache_base_cls.get_max_length = get_max_length

        if cache_base_cls is not None and not hasattr(cache_base_cls, "get_usable_length") and hasattr(cache_base_cls, "get_seq_length"):
            # Recreate the pre-5.x cache eviction helper so remote attention code
            # can compute usable KV length against the newer cache interface.
            def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
                max_length = self.get_max_length(layer_idx=layer_idx)
                previous_seq_length = self.get_seq_length(layer_idx)
                if max_length is not None and previous_seq_length + new_seq_length > max_length:
                    return max_length - new_seq_length
                return previous_seq_length

            cache_base_cls.get_usable_length = get_usable_length

        dynamic_cache_cls = getattr(cache_utils, "DynamicCache", None) if cache_utils is not None else None
        if dynamic_cache_cls is not None and not hasattr(dynamic_cache_cls, "to_legacy_cache"):
            # Older remote generation code still serializes cache state as
            # `Tuple[(key, value), ...]`; rebuild that view from layer storage.
            def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
                legacy_cache = ()
                for layer in self.layers:
                    if not getattr(layer, "is_initialized", False):
                        continue
                    legacy_cache += ((layer.keys, layer.values),)
                return legacy_cache

            dynamic_cache_cls.to_legacy_cache = to_legacy_cache

        if dynamic_cache_cls is not None and not hasattr(dynamic_cache_cls, "from_legacy_cache"):
            # Accept legacy tuple caches by replaying them into the current
            # layer-based DynamicCache implementation.
            @classmethod
            def from_legacy_cache(cls, past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None):
                cache = cls()
                if past_key_values is not None:
                    for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                        cache.update(key_states, value_states, layer_idx)
                return cache

            dynamic_cache_cls.from_legacy_cache = from_legacy_cache

        if generation_utils is not None and not hasattr(generation_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
            # Older remote generation code registers custom cache builders through
            # this module-global dict during import.
            generation_utils.NEED_SETUP_CACHE_CLASSES_MAPPING = {}

        generation_mixin_cls = getattr(generation_utils, "GenerationMixin", None) if generation_utils is not None else None
        if generation_mixin_cls is not None and not getattr(generation_mixin_cls, "_gptqmodel_custom_cache_impl_patch", False):
            original_prepare_cache_for_generation = generation_mixin_cls._prepare_cache_for_generation

            # transformers 5.x removed the custom cache registry path used by some
            # trust_remote_code models, so recreate just enough of that setup here.
            def _prepare_cache_for_generation_compat(
                self,
                generation_config: GenerationConfig,
                model_kwargs: dict,
                generation_mode,
                batch_size: int,
                max_cache_length: int,
            ) -> None:
                cache_mapping = getattr(generation_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING", None)
                cache_implementation = getattr(generation_config, "cache_implementation", None)
                if not isinstance(cache_mapping, dict) or not isinstance(cache_implementation, str):
                    original_prepare_cache_for_generation(
                        self,
                        generation_config,
                        model_kwargs,
                        generation_mode,
                        batch_size,
                        max_cache_length,
                    )
                    return None

                custom_cache_cls = cache_mapping.get(cache_implementation)
                if custom_cache_cls is None:
                    original_prepare_cache_for_generation(
                        self,
                        generation_config,
                        model_kwargs,
                        generation_mode,
                        batch_size,
                        max_cache_length,
                    )
                    return None

                is_linear_attn_cache = "mamba" in self.__class__.__name__.lower()
                cache_name = "past_key_values" if not is_linear_attn_cache else "cache_params"

                user_defined_cache = model_kwargs.get(cache_name)
                if user_defined_cache is not None:
                    if generation_config.cache_implementation is not None:
                        raise ValueError(
                            f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` "
                            "(`Cache` object) is unsupported. Please use only one of the two."
                        )
                    if isinstance(user_defined_cache, tuple):
                        raise ValueError(
                            "Passing a tuple of `past_key_values` is not supported anymore. Please use a `Cache` instance."
                        )
                    return

                if generation_config.use_cache is False:
                    return

                cache_config = generation_config.cache_config
                if cache_config is None:
                    cache_kwargs = {}
                elif isinstance(cache_config, dict):
                    cache_kwargs = dict(cache_config)
                elif hasattr(cache_config, "to_dict"):
                    cache_kwargs = dict(cache_config.to_dict())
                else:
                    cache_kwargs = dict(cache_config)

                text_config = self.config.get_text_config(decoder=True) if hasattr(self.config, "get_text_config") else self.config
                full_batch_size = max(generation_config.num_beams, generation_config.num_return_sequences) * batch_size
                cache_kwargs.setdefault("config", text_config)
                cache_kwargs.setdefault("batch_size", full_batch_size)
                cache_kwargs.setdefault("max_batch_size", full_batch_size)
                cache_kwargs.setdefault("max_cache_len", max_cache_length)

                model_dtype = getattr(self, "dtype", None) or get_hf_config_dtype(self.config)
                if model_dtype is not None:
                    cache_kwargs.setdefault("dtype", model_dtype)

                model_device = getattr(self, "device", None)
                if model_device is not None:
                    cache_kwargs.setdefault("device", model_device)

                model_kwargs["past_key_values"] = custom_cache_cls(**cache_kwargs)

                encoder_decoder_cache_cls = getattr(cache_utils, "EncoderDecoderCache", None) if cache_utils is not None else None
                if (
                    getattr(self.config, "is_encoder_decoder", False)
                    and "past_key_values" in model_kwargs
                    and encoder_decoder_cache_cls is not None
                    and not isinstance(model_kwargs["past_key_values"], encoder_decoder_cache_cls)
                    and dynamic_cache_cls is not None
                ):
                    model_kwargs["past_key_values"] = encoder_decoder_cache_cls(
                        model_kwargs["past_key_values"],
                        dynamic_cache_cls(config=text_config),
                    )

                return None

            generation_mixin_cls._prepare_cache_for_generation = _prepare_cache_for_generation_compat
            generation_mixin_cls._gptqmodel_custom_cache_impl_patch = True

        if not getattr(PreTrainedModel, "_gptqmodel_legacy_tied_weights_patch", False) and hasattr(PreTrainedModel, "get_expanded_tied_weights_keys"):
            original_get_expanded_tied_weights_keys = PreTrainedModel.get_expanded_tied_weights_keys

            def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
                # transformers 5.x expects `_tied_weights_keys` to be a dict, while
                # older trust_remote_code models still declare `["lm_head.weight"]`.
                # Handle the legacy form here so HF tied-weight expansion still works.
                tied_mapping = getattr(self, "_tied_weights_keys", None)
                if not isinstance(tied_mapping, (list, tuple, set)):
                    return original_get_expanded_tied_weights_keys(self, all_submodels=all_submodels)

                if all_submodels:
                    expanded_tied_weights = {}
                    for prefix, submodule in self.named_modules(remove_duplicate=False):
                        if isinstance(submodule, PreTrainedModel):
                            submodel_tied_weights = submodule.get_expanded_tied_weights_keys(all_submodels=False)
                            if prefix != "":
                                submodel_tied_weights = {
                                    f"{prefix}.{k}": f"{prefix}.{v}" for k, v in submodel_tied_weights.items()
                                }
                            expanded_tied_weights.update(submodel_tied_weights)
                    return expanded_tied_weights

                if not getattr(getattr(self, "config", None), "tie_word_embeddings", False):
                    return {}

                resolved_mapping = _resolve_legacy_tied_weights_mapping(self, tied_mapping)
                self._tied_weights_keys = resolved_mapping
                return resolved_mapping

            PreTrainedModel.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys
            PreTrainedModel._gptqmodel_legacy_tied_weights_patch = True

        if not hasattr(PreTrainedModel, "is_parallelizable"):
            # Older remote-code model wrappers read this legacy base-class flag
            # during init, but newer transformers dropped the default attribute.
            PreTrainedModel.is_parallelizable = False

        if not getattr(PreTrainedModel, "_gptqmodel_missing_all_tied_weights_patch", False):
            original_getattr = PreTrainedModel.__getattr__

            def __getattr__(self, name: str):
                if name == "all_tied_weights_keys":
                    # Older remote-code models may skip `post_init()`, so lazily
                    # synthesize the tied-weight map the first time HF asks for it.
                    tied_keys = self.get_expanded_tied_weights_keys(all_submodels=True)
                    object.__setattr__(self, name, tied_keys)
                    return tied_keys

                return original_getattr(self, name)

            PreTrainedModel.__getattr__ = __getattr__
            PreTrainedModel._gptqmodel_missing_all_tied_weights_patch = True


def _normalize_chatglm_remote_code_config_compat(config: Any) -> None:
    if getattr(config, "model_type", None) != "chatglm":
        return

    if not hasattr(config, "seq_length") or hasattr(config, "max_length"):
        return

    # Older ChatGLM remote model code still reads `config.max_length`, while
    # newer transformers only preserves the serialized `seq_length` field.
    config.attribute_map = dict(getattr(config, "attribute_map", {}) or {})
    config.attribute_map["max_length"] = "seq_length"

    if not hasattr(config, "use_cache"):
        # transformers v5 removed `use_cache` from PretrainedConfig,
        # but ChatGLM remote code still expects it. Add it back for compatibility.
        config.use_cache = True


def _normalize_rope_parameters_config_compat(config: Any) -> None:
    rope_parameters = getattr(config, "rope_parameters", None)
    if (
        isinstance(rope_parameters, dict)
        and rope_parameters.get("rope_type") is not None
        and rope_parameters.get("rope_theta") is not None
    ):
        return

    convert_rope_params = getattr(config, "convert_rope_params_to_dict", None)
    if callable(convert_rope_params):
        try:
            convert_rope_params()
        except Exception as exc:
            log.debug("Config: RoPE conversion fallback for %s failed: %s", type(config).__name__, exc)
        else:
            rope_parameters = getattr(config, "rope_parameters", None)
            if (
                isinstance(rope_parameters, dict)
                and rope_parameters.get("rope_type") is not None
                and rope_parameters.get("rope_theta") is not None
            ):
                return

    legacy_rope_scaling = getattr(config, "rope_scaling", None)
    rope_parameters = dict(legacy_rope_scaling) if isinstance(legacy_rope_scaling, dict) else dict(rope_parameters or {})

    if not rope_parameters and getattr(config, "rope_theta", None) is None and getattr(config, "default_theta", None) is None:
        return

    rope_parameters.setdefault("rope_type", rope_parameters.get("type", "default"))
    if rope_parameters.get("rope_theta") is None:
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = getattr(config, "default_theta", 10_000.0)
        rope_parameters["rope_theta"] = rope_theta

    partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
    if partial_rotary_factor is not None:
        rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)

    if rope_parameters["rope_type"] in {"llama3", "yarn", "longrope"}:
        original_max_position_embeddings = getattr(config, "original_max_position_embeddings", None)
        if original_max_position_embeddings is None:
            original_max_position_embeddings = getattr(config, "max_position_embeddings", None)
        if original_max_position_embeddings is not None:
            rope_parameters.setdefault("original_max_position_embeddings", original_max_position_embeddings)

    config.rope_parameters = rope_parameters


# Restore config fields renamed by transformers 5.x before older trust_remote_code
# model files instantiate their architectures from the config object.
def _normalize_remote_code_config_compat(config: Any) -> None:
    _normalize_chatglm_remote_code_config_compat(config)
    model_type = getattr(config, "model_type", None)
    model_type_lower = model_type.lower() if isinstance(model_type, str) else None

    if model_type_lower == "dream" or model_type == "brumby":
        import transformers.modeling_rope_utils as rope_utils
        # dream remote models expect "default"
        if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
            rope_utils.ROPE_INIT_FUNCTIONS["default"] = rope_utils.ROPE_INIT_FUNCTIONS["linear"]

        # transformers 5.x expects rope_parameters["factor"] for linear RoPE
        if getattr(config, "rope_parameters", None):
            config.rope_parameters.setdefault("factor", 1.0)

    # BrumbyConfig remote config may not define pad_token_id.
    # Ensure the attribute exists to avoid AttributeError in transformers 5.x.
    if model_type == "brumby":
        rope_scaling = getattr(config, "rope_scaling", None)

        config.pad_token_id = getattr(config, "pad_token_id", None)

    # transformers 5.x normalizes RoPE config to `rope_type`, but older
    # MiniCPM remote code still reads `rope_scaling["type"]` or expects `None`.
    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        return

    if model_type != "instella" and rope_scaling.get("rope_type") == "default" and set(rope_scaling).issubset({"rope_type", "rope_theta"}):
        # transformers 5.x materializes default RoPE metadata into
        # `rope_scaling`, but older remote MiniCPM code treats any dict here
        # as a scaled-RoPE config and expects an explicit `factor`.
        config.rope_scaling = None
        return

    if "type" in rope_scaling:
        return

    rope_type = rope_scaling.get("rope_type")
    if rope_type is None:
        return

    rope_scaling = dict(rope_scaling)
    if model_type == "instella":
        if rope_type == "default":
            rope_type = "linear"
        rope_scaling["factor"] = 1.0
    rope_scaling["type"] = rope_type
    config.rope_scaling = rope_scaling


def deci_init_compat(config):
    if config.model_type == "deci":
        from transformers.models.auto import modeling_auto
        with _MONKEY_PATCH_LOCK:
            orig_register = modeling_auto.AutoModelForCausalLM.register

            def patched_register(cls, config_class, model_class, exist_ok=False):
                # DeciLMForCausalLM inherits from LlamaForCausalLM, but does not override
                # `config_class` (thus still pointing to LlamaConfig). However, the model's
                # config.json declares its AutoConfig as DeciLMConfig. This leads to a mismatch
                # during AutoModel registration (model_class.config_class != config_class),
                # causing a ValueError. We patch this inconsistency at runtime.
                if hasattr(model_class, "config_class"):
                    model_class.config_class = config_class
                return orig_register(config_class, model_class, exist_ok=exist_ok)

            modeling_auto.AutoModelForCausalLM.register = classmethod(patched_register)


def normalize_hf_config_compat(config: Any, *, trust_remote_code: bool = False) -> None:
    # Some transformers 5.x model classes now read `config.rope_parameters`
    # directly during `from_config()`, but older local configs may only carry
    # legacy RoPE fields or nothing but a default `rope_theta`.
    _normalize_rope_parameters_config_compat(config)

    if not trust_remote_code:
        return

    _patch_transformers_remote_code_compat()
    _normalize_remote_code_config_compat(config)
    # Some config classes synchronize `rope_scaling` and `rope_parameters`, so
    # remote-code normalization that clears legacy default `rope_scaling` can
    # also reset `rope_parameters` back to None. Re-apply the RoPE backfill
    # after remote-code field cleanup so from_config() sees stable metadata.
    _normalize_rope_parameters_config_compat(config)


def prepare_remote_code_compat(config: Any) -> None:
    # Remote-code loads need both the transformers API shims and any config
    # field migrations applied before instantiation happens.
    normalize_hf_config_compat(config, trust_remote_code=True)


def prepare_remote_model_init_compat(model_id_or_path: Optional[str], config: Any) -> None:
    if not model_id_or_path:
        return

    deci_init_compat(config)

    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModelForCausalLM")
    if not isinstance(class_ref, str):
        return

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        model_cls = get_class_from_dynamic_module(class_ref, str(model_id_or_path))
    except Exception as exc:
        log.debug("HF: remote model init compat pre-import failed for `%s`: %s", model_id_or_path, exc)
        return

    module_root = model_cls.__module__.rsplit(".", maxsplit=1)[0]
    speech_module = sys.modules.get(f"{module_root}.speech_conformer_encoder")
    remote_module = sys.modules.get(model_cls.__module__)
    ovis_config_module = sys.modules.get(f"{module_root}.configuration_ovis")
    outer_model_cls = model_cls if isinstance(model_cls, type) else None
    input_mode_enum = getattr(remote_module, "InputMode", None) if remote_module is not None else None

    with _MONKEY_PATCH_LOCK:
        if outer_model_cls is not None:
            try_patch_legacy_flash_attn_flag(outer_model_cls)

        if config.model_type == "minicpmv" or config.model_type == "minicpmo":
            vision_model_cls = getattr(
                remote_module,
                "SiglipVisionTransformer",
                None,
            )
            if vision_model_cls:
                try_patch_legacy_flash_attn_flag(vision_model_cls)

        if (
            outer_model_cls is not None
            and hasattr(outer_model_cls, "tie_weights")
            and not getattr(outer_model_cls, "_gptqmodel_tie_weights_kwargs_patch", False)
        ):
            try:
                tie_weights_sig = inspect.signature(outer_model_cls.tie_weights)
            except (TypeError, ValueError):
                tie_weights_sig = None

            if tie_weights_sig is not None:
                tie_weight_params = tie_weights_sig.parameters.values()
                accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in tie_weight_params)
                supports_missing_keys = "missing_keys" in tie_weights_sig.parameters
                supports_recompute_mapping = "recompute_mapping" in tie_weights_sig.parameters

                if not accepts_kwargs and (not supports_missing_keys or not supports_recompute_mapping):
                    original_tie_weights = outer_model_cls.tie_weights

                    # transformers 5.x passes `missing_keys=` and `recompute_mapping=`
                    # into tie_weights(); older remote-code models still declare
                    # `tie_weights(self)` and only need the original no-arg behavior.
                    def tie_weights_compat(self, *args, **kwargs):
                        return original_tie_weights(self)

                    outer_model_cls.tie_weights = tie_weights_compat
                    outer_model_cls._gptqmodel_tie_weights_kwargs_patch = True

        if getattr(config, "model_type", None) == "ovis" and ovis_config_module is not None:
            formatter_cls = getattr(ovis_config_module, "Llama3ConversationFormatter", None)
            if formatter_cls is not None and not getattr(formatter_cls, "_gptqmodel_tokenizer_backend_patch", False):
                support_tokenizer_types = list(getattr(formatter_cls, "support_tokenizer_types", None) or [])
                if "TokenizersBackend" not in support_tokenizer_types:
                    # Current transformers/tokenicer fast-tokenizer backend exposes
                    # `TokenizersBackend` instead of the older
                    # `PreTrainedTokenizerFast` class name expected by Ovis remote code.
                    support_tokenizer_types.append("TokenizersBackend")
                    formatter_cls.support_tokenizer_types = support_tokenizer_types
                formatter_cls._gptqmodel_tokenizer_backend_patch = True

        if getattr(config, "model_type", None) != "phi4mm":
            return

        if speech_module is not None and not getattr(speech_module, "_gptqmodel_scalar_tensor_meta_patch", False):
            speech_torch = getattr(speech_module, "torch", None)
            original_tensor = getattr(speech_torch, "tensor", None)
            if speech_torch is not None and original_tensor is not None:
                def _is_phi4mm_subsampling_scalar_init() -> bool:
                    for frame_info in inspect.stack(context=0):
                        if frame_info.filename.endswith("speech_conformer_encoder.py") and frame_info.lineno == 1426:
                            return True
                    return False

                # Phi-4 MM remote audio init creates scalar tensors only to derive Python
                # output sizes in NemoConvSubsampling; forcing just that scalar tensor onto
                # CPU keeps meta init safe without perturbing other meta-only buffers.
                def tensor_compat(data, *args, **kwargs):
                    current_device = getattr(torch.utils._device, "CURRENT_DEVICE", None)
                    if (
                        kwargs.get("device") is None
                        and current_device == torch.device("meta")
                        and isinstance(data, (int, float, bool))
                        and _is_phi4mm_subsampling_scalar_init()
                    ):
                        kwargs = dict(kwargs)
                        kwargs["device"] = "cpu"
                    return original_tensor(data, *args, **kwargs)

                speech_torch.tensor = tensor_compat

            positional_encoding_cls = getattr(speech_module, "AbsolutePositionalEncoding", None)
            if positional_encoding_cls is not None and not getattr(positional_encoding_cls, "_gptqmodel_meta_extend_patch", False):
                original_extend_pe = positional_encoding_cls.extend_pe

                def _is_phi4mm_positional_seed_call() -> bool:
                    for frame_info in inspect.stack(context=0):
                        if frame_info.filename.endswith("speech_conformer_encoder.py") and frame_info.lineno == 895:
                            return True
                    return False

                # The remote implementation seeds extend_pe() with a CPU scalar tensor.
                # Under meta init, promote that seed tensor back to meta before the
                # original method allocates its positional buffer.
                def extend_pe_compat(self, x):
                    if isinstance(x, torch.Tensor) and x.device.type != "meta" and _is_phi4mm_positional_seed_call():
                        x = x.to(device="meta")
                    return original_extend_pe(self, x)

                positional_encoding_cls.extend_pe = extend_pe_compat
                positional_encoding_cls._gptqmodel_meta_extend_patch = True

            speech_module._gptqmodel_scalar_tensor_meta_patch = True

        if (
            outer_model_cls is not None
            and hasattr(outer_model_cls, "forward")
            and not getattr(outer_model_cls, "_gptqmodel_input_mode_patch", False)
        ):
            original_forward = outer_model_cls.forward

            # Text-only callers like lm_eval do not pass `input_mode`; infer the
            # correct Phi-4 MM mode from the provided modality tensors instead.
            def forward_compat(self, *args, **kwargs):
                if kwargs.get("input_mode") is None:
                    kwargs = dict(kwargs)
                    has_vision = any(
                        kwargs.get(name) is not None
                        for name in ("input_image_embeds", "image_sizes", "image_attention_mask")
                    )
                    has_audio = any(
                        kwargs.get(name) is not None
                        for name in ("input_audio_embeds", "audio_embed_sizes", "audio_attention_mask")
                    )

                    if has_vision and has_audio:
                        kwargs["input_mode"] = input_mode_enum.VISION_SPEECH if input_mode_enum is not None else 3
                    elif has_vision:
                        kwargs["input_mode"] = input_mode_enum.VISION if input_mode_enum is not None else 1
                    elif has_audio:
                        kwargs["input_mode"] = input_mode_enum.SPEECH if input_mode_enum is not None else 2
                    else:
                        kwargs["input_mode"] = input_mode_enum.LANGUAGE if input_mode_enum is not None else 0
                return original_forward(self, *args, **kwargs)

            outer_model_cls.forward = forward_compat
            outer_model_cls._gptqmodel_input_mode_patch = True

        inner_model_cls = getattr(remote_module, "Phi4MMModel", None) if remote_module is not None else None
        if inner_model_cls is not None and not hasattr(inner_model_cls, "prepare_inputs_for_generation"):
            # PEFT expects the inner model it wraps to expose this hook, even
            # though Phi-4 MM only defines the full implementation on the outer
            # CausalLM class.
            def prepare_inputs_for_generation(self, input_ids=None, past_key_values=None, inputs_embeds=None, **kwargs):
                model_inputs = dict(kwargs)
                if inputs_embeds is not None and past_key_values is None:
                    model_inputs["inputs_embeds"] = inputs_embeds
                else:
                    model_inputs["input_ids"] = input_ids
                model_inputs["past_key_values"] = past_key_values
                return model_inputs

            inner_model_cls.prepare_inputs_for_generation = prepare_inputs_for_generation

        try:
            import importlib.util

            import peft.import_utils as peft_import_utils
            import peft.tuners.lora.awq as peft_awq
        except Exception:
            pass
        else:
            if not getattr(peft_awq, "_gptqmodel_awq_probe_patch", False):
                # PEFT later imports `awq.modules.linear`, so the availability
                # probe must require that concrete submodule instead of top-level
                # namespace packages that are missing the actual runtime.
                @lru_cache(maxsize=None)
                def is_auto_awq_available() -> bool:
                    try:
                        return importlib.util.find_spec("awq.modules.linear") is not None
                    except ModuleNotFoundError:
                        return False

                peft_import_utils.is_auto_awq_available = is_auto_awq_available
                peft_awq.is_auto_awq_available = is_auto_awq_available
                peft_awq._gptqmodel_awq_probe_patch = True


def try_patch_legacy_flash_attn_flag(model_cls):
    with _MONKEY_PATCH_LOCK:
        if model_cls is None or not isinstance(model_cls, type):
            return

        # Find the most specific class that explicitly declares the newer
        # `_supports_flash_attn_2` flag used by newer transformers releases.
        base_with_flag = None
        for cls in model_cls.__mro__:
            if "_supports_flash_attn_2" in cls.__dict__:
                base_with_flag = cls
                break

        if base_with_flag is None:
            return

        # Respect remote models that already define the legacy flag themselves.
        for cls in model_cls.__mro__:
            if cls is base_with_flag:
                break
            if "_supports_flash_attn" in cls.__dict__:
                return

        flash_attn_2_val = base_with_flag.__dict__["_supports_flash_attn_2"]
        setattr(base_with_flag, "_supports_flash_attn", bool(flash_attn_2_val))


def load_tokenizer(tokenizer_or_path, *, model_config: Any = None, **kwargs):
    from tokenicer import Tokenicer

    warnings.warn(
        "gptqmodel.utils.hf.load_tokenizer() is deprecated; use Tokenicer.load(..., model_config=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Tokenicer.load(tokenizer_or_path, model_config=model_config, **kwargs)



_patch_transformers_remote_code_compat()

def _sanitize_generation_config(cfg: GenerationConfig, *, drop_sampling_fields: bool = False) -> bool:
    changed = False
    if cfg is None:
        return changed

    if getattr(cfg, "do_sample", None) is not True:
        cfg.do_sample = True
        changed = True

    return changed


def _load_sanitized_generation_config(path: str) -> Optional[GenerationConfig]:
    try:
        config_dict, kwargs = GenerationConfig.get_config_dict(path)
    except Exception:
        return None

    cleaned = dict(config_dict)
    if cleaned.get("do_sample") is not True:
        cleaned["do_sample"] = True

    cfg = GenerationConfig.from_dict(cleaned, **kwargs)
    _sanitize_generation_config(cfg, drop_sampling_fields=False)
    return cfg


# TODO FIXME! Pre-quantized use AutoModelForCausalLM.from_pretrained() but post-quantized use AutoModelForCausalLM.from_config()
def autofix_hf_model_config(model: PreTrainedModel, path: str = None):
    if model.can_generate():
        # sync config first
        if path:
            log.info(f"Model: Loaded `generation_config`: {model.generation_config}")
            try:
                cfg = _load_sanitized_generation_config(path)
                if cfg is None:
                    cfg = GenerationConfig.from_pretrained(pretrained_model_name=path, do_sample=True)
                    _sanitize_generation_config(cfg, drop_sampling_fields=False)
                if cfg != model.generation_config:
                    # migrated pad_token_id to config
                    if hasattr(model.generation_config, "pad_token_id"):
                        cfg.pad_token_id = model.generation_config.pad_token_id

                    model.generation_config = cfg
                    log.info(
                        "Model: Auto-fixed `generation_config` mismatch between model and `generation_config.json`.")
                    log.info(f"Model: Updated `generation_config`: {model.generation_config}")
                else:
                    pass
                    # logger.info(f"Model: loaded `generation_config` matching `generation_config.json`.")
            except Exception:
                log.info("Model: `generation_config.json` not found. Skipped checking.")

        # print(f"Before autofix_hf_model_config: {model.generation_config}")
        autofix_hf_generation_config(model.generation_config)
        # print(f"After autofix_hf_model_config: {model.generation_config}")


def autofix_hf_generation_config(cfg: GenerationConfig):
    _sanitize_generation_config(cfg, drop_sampling_fields=False)
    # HF has recently started to perform very strict validation model save which results in warnings on load()
    # to become exceptions on save().
    if cfg.do_sample is False:
        errors = 0
        if hasattr(cfg, "temperature") and cfg.temperature is not None and cfg.temperature != 1.0:
            errors += 1
        if hasattr(cfg, "top_p") and cfg.top_p is not None and cfg.top_p != 1.0:
            errors += 1
        if hasattr(cfg, "min_p") and cfg.min_p is not None:
            errors += 1
        if hasattr(cfg, "typical_p") and cfg.typical_p is not None and cfg.typical_p != 1.0:
            errors += 1
        # contrastive search uses top_k
        if (hasattr(cfg, "top_k") and cfg.top_k is not None and cfg.top_k != 50) and (hasattr(cfg, "penalty_alpha") and cfg.penalty_alpha is None):
            errors += 1
        if hasattr(cfg, "epsilon_cutoff") and cfg.epsilon_cutoff is not None and cfg.epsilon_cutoff != 0.0:
            errors += 1
        if hasattr(cfg, "eta_cutoff") and cfg.eta_cutoff is not None and cfg.eta_cutoff != 0.0:
            errors += 1

        # fix wrong do_sample
        if errors > 0:
            cfg.do_sample = True
            log.info("Model: Auto-Fixed `generation_config` by setting `do_sample=True`.")


def sanitize_model_config(config):
    if config.model_type == "chatglm" and hasattr(config, "max_length"):
        # max_length can only be stored in generation_config.
        # see _normalize_chatglm_remote_code_config_compat()
        del config.attribute_map["max_length"]


def sanitize_generation_config_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return False

    changed = False

    if data.get("do_sample") is not True:
        data["do_sample"] = True
        changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

    return changed

# load hf model with empty tensors on meta device (zero tensor memory usage)
def build_shell_model(
    loader,
    config: Any,
    trust_remote_code: bool = True,
    **model_init_kwargs,
):
    """
    Instantiate the HF architecture with all parameters and buffers on 'meta' (no CPU RAM).
    Preserves the full module topology (Linear/MLP/Attention/etc.).

    Args:
        model_id_or_path: Hugging Face model ID or local path.
        trust_remote_code: Allow loading custom model classes.
    """
    init_kwargs = model_init_kwargs.copy()

    del init_kwargs["device_map"]
    del init_kwargs["_fast_init"]
    # All nn.Parameters and buffers are created

    normalize_hf_config_compat(config, trust_remote_code=trust_remote_code)

    # All nn.Parameters and buffers are created on 'meta' and initializers are skipped.
    pb = log.spinner(title="Model loading...", interval=0.1)
    try:
        with init_empty_weights(include_buffers=True):
            shell = loader.from_config(
                config,
                trust_remote_code=trust_remote_code,
                **init_kwargs
            )
    finally:
        pb.close()

    if trust_remote_code and isinstance(shell, PreTrainedModel):
        _normalize_legacy_tied_weights_keys(shell)

    return shell
