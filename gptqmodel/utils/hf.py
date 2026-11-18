# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import ntpath
import os
import re
import threading
import warnings
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache, wraps
from typing import Any, Optional

import torch
from accelerate import init_empty_weights
from transformers import (  # Auto*: unsupported-model config/support probes only
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.utils import cached_file

from ..utils import _MONKEY_PATCH_LOCK


_GGUF_LAZY_ATTRS = frozenset({
    "PRISM_Q1_0_G128_BLOCK_SIZE",
    "PRISM_Q1_0_G128_NAME",
    "PRISM_Q1_0_G128_TYPE_SIZE",
    "PRISM_Q1_0_G128_VALUE",
    "_dequantize_prism_q1_0_g128",
    "_is_prism_q1_0_g128",
})


def __getattr__(name: str):
    """Resolve GGUF/PRISM names lazily so this module never imports NumPy."""
    if name in _GGUF_LAZY_ATTRS:
        from ..nn_modules.qlinear import gguf as gguf_module

        value = getattr(gguf_module, name)
    elif name == "internal_gguf":
        from ..utils import internal_gguf as value
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


from transformers import PreTrainedConfig
from transformers.initialization import no_init_weights

from ..utils.logger import setup_logger
from ..utils.security import RemoteCodeSecurityError, require_remote_code_permission, safety_checks_bypassed


__all__ = [
    "no_init_weights",
    "suspend_hf_weight_init",
    "assert_loaded_model_state",
    "record_construction_dtypes",
    "get_hf_config_dtype",
    "normalize_torch_dtype_kwarg",
    "has_native_transformers_causallm_support",
    "get_config_selection_kwargs",
    "get_hf_gguf_load_kwargs",
    "normalize_checkpoint_subfolder",
    "normalize_model_id_or_path_for_hf_gguf",
    "ensure_qwen_drive_registered",
    "resolve_trust_remote_code",
    "set_hf_config_dtype",
    "load_hf_tokenizer",
    "load_model_config",
    "load_tokenizer",
]

log = setup_logger()
_TRUST_REMOTE_CODE_OVERRIDE_WARNED: set[tuple[str, str, str]] = set()
_MISSING = object()
INTERNAL_HF_GGUF_FILE_KWARG = "_gptqmodel_hf_gguf_file"
_DENSE_MODEL_FILE_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")

_QWEN_DRIVE_MODEL_TYPE = "qwen_drive"
_QWEN_DRIVE_INSTALL_HINT = (
    "Qwen-Drive checkpoints require the official `qwen_drive` package. "
    "Install it from `https://github.com/QwenLM/Qwen-Drive-1.0` "
    "(or make that repository's `src` directory importable) "
    "before loading this model."
)


def _local_config_model_type(model_id_or_path: Optional[str]) -> Optional[str]:
    """Read a local model type without invoking Transformers' AutoConfig."""

    if not model_id_or_path or not os.path.isdir(model_id_or_path):
        return None

    config_path = os.path.join(model_id_or_path, "config.json")
    if not os.path.isfile(config_path):
        return None

    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, TypeError, ValueError):
        return None

    model_type = config.get("model_type") if isinstance(config, dict) else None
    return str(model_type).lower() if model_type is not None else None


def ensure_qwen_drive_registered(model_id_or_path: Optional[str] = None) -> bool:
    """Register legacy Qwen-Drive only when no package-owned loader exists.

    Package ownership must bypass the optional official package in every
    caller, including native probes and writer shell reconstruction. The
    fallback retains actionable errors for legacy externally registered
    definitions without importing an unrelated planning stack.
    """

    from ..models.local import local_model_loader

    # Package ownership precedes the legacy optional registration hook in
    # config loading, native probing and writer reconstruction alike.
    if local_model_loader(_QWEN_DRIVE_MODEL_TYPE) is not None:
        return False

    local_model_type = _local_config_model_type(model_id_or_path)
    if local_model_type != _QWEN_DRIVE_MODEL_TYPE:
        return False

    try:
        import qwen_drive  # noqa: F401
    except ImportError as exc:
        raise ImportError(_QWEN_DRIVE_INSTALL_HINT) from exc

    return True


def get_hf_config_dtype(config: Any) -> Optional[torch.dtype]:
    dtype = getattr(config, "dtype", _MISSING)
    if dtype is _MISSING:
        dtype = getattr(config, "torch_dtype", None)
    elif dtype is None:
        # Transformers 5.14 exposes `torch_dtype` as a deprecated property that
        # warns even when only read. Preserve genuinely stored legacy metadata
        # without touching that alias on modern configs.
        dtype = getattr(config, "__dict__", {}).get("torch_dtype")

    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype

    # If provided as a string (e.g., "float16" or "torch.bfloat16"), resolve it
    # via the torch namespace. The loader accepts the same prefixed spelling.
    if isinstance(dtype, str):
        name = dtype.strip().removeprefix("torch.")
        try:
            resolved = getattr(torch, name)
        except AttributeError:
            raise ValueError(f"Invalid dtype string: {dtype}")
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Invalid dtype string: {dtype}")
        return resolved

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
    def _resolve_dtype(value: Any) -> Any:
        if value is _MISSING or value is None:
            return value
        if isinstance(value, str):
            if value.strip().lower() == "auto":
                return "auto"
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            name = value.strip().removeprefix("torch.")
            resolved = getattr(torch, name, None)
            if isinstance(resolved, torch.dtype):
                return resolved
            raise ValueError(f"Invalid dtype string: {value}")
        raise ValueError(
            f"dtype must be torch.dtype, str, None, or 'auto', got {value!r}"
        )

    current_dtype = explicit_dtype
    if explicit_dtype is _MISSING:
        current_dtype = kwargs.pop("dtype", _MISSING)

    torch_dtype = kwargs.pop("torch_dtype", _MISSING)
    current_dtype = _resolve_dtype(current_dtype)
    torch_dtype = _resolve_dtype(torch_dtype)
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


_HF_INIT_LOCK = threading.RLock()
_HF_INIT_DEPTH = 0
_HF_INIT_SUSPENSION = None


def _install_init_suspension():
    """Patch the process-wide init entry points and return the originals."""

    def _skip_init(*args, **kwargs):
        return None

    originals = {
        "kaiming_uniform": torch.nn.init.kaiming_uniform_,
        "uniform": torch.nn.init.uniform_,
        "normal": torch.nn.init.normal_,
    }

    torch.nn.init.kaiming_uniform_ = _skip_init
    torch.nn.init.uniform_ = _skip_init
    torch.nn.init.normal_ = _skip_init

    # transformers' own guard patches many more torch initializers and
    # ``PreTrainedModel.init_weights``. Keep it entered for the whole outermost
    # scope: re-entering it per caller is what makes overlapping entries lose
    # the originals.
    no_init = no_init_weights()
    no_init.__enter__()
    originals["no_init"] = no_init
    return originals


def _restore_init_suspension(originals):
    try:
        originals["no_init"].__exit__(None, None, None)
    finally:
        torch.nn.init.kaiming_uniform_ = originals["kaiming_uniform"]
        torch.nn.init.uniform_ = originals["uniform"]
        torch.nn.init.normal_ = originals["normal"]


@contextmanager
def suspend_hf_weight_init():
    """Disable HF/torch parameter init until the last overlapping entry exits.

    ``transformers``' ``no_init_weights`` patches process-wide initializers and
    ``PreTrainedModel.init_weights``, and this module also silences three torch
    initializers. Both are process-global, so overlapping entries are reference
    counted: the first entry installs the patches, later entries only increment
    the depth, and the last exit restores the originals. Without the count the
    first caller to leave restored the real initializers while another thread
    was still building its model, and the second exit then restored the
    ``_skip_init`` stubs, disabling weight init for the rest of the process
    (issue #4129).
    """

    global _HF_INIT_DEPTH, _HF_INIT_SUSPENSION

    with _HF_INIT_LOCK:
        if _HF_INIT_DEPTH == 0:
            _HF_INIT_SUSPENSION = _install_init_suspension()
        _HF_INIT_DEPTH += 1
    try:
        yield
    finally:
        with _HF_INIT_LOCK:
            _HF_INIT_DEPTH -= 1
            if _HF_INIT_DEPTH == 0:
                originals, _HF_INIT_SUSPENSION = _HF_INIT_SUSPENSION, None
                _restore_init_suspension(originals)


_KEEP_IN_FP32_ATTRS = ("_keep_in_fp32_modules", "_keep_in_fp32_modules_strict")


def _iter_keep_in_fp32_declarations(model, attrs=_KEEP_IN_FP32_ATTRS):
    """Yield the keep-in-fp32 patterns a model declares on itself or a child.

    ``PreTrainedModel.post_init`` already unions child declarations onto the
    top-level model, but walking submodules keeps the guard correct for models
    built without that step.

    ``attrs`` narrows the walk to one declaration class. ``_strict`` entries
    stay fp32 for every half dtype; the non-strict ones only for fp16.
    """

    seen = set()
    modules = getattr(model, "modules", None)
    for module in (modules() if callable(modules) else (model,)):
        for attr in attrs:
            declared = getattr(module, attr, None)
            if not declared:
                continue
            if isinstance(declared, str):
                declared = (declared,)
            for name in declared:
                if isinstance(name, str) and name not in seen:
                    seen.add(name)
                    yield name


def _compile_keep_in_fp32_matcher(model, attrs=_KEEP_IN_FP32_ATTRS):
    """Compile the model's fp32-pinned parameter patterns, transformers-style.

    transformers 5.22 matches every ``_keep_in_fp32_modules`` and
    ``_keep_in_fp32_modules_strict`` entry with ``re.search`` against the full
    parameter key, expanding ``*`` to ``.*``
    (``core_model_loading.build_glob_alternation``). In practice those entries
    are module substrings such as ``gate.weight`` or ``wo``; the regex/glob
    behaviour is preserved for wildcard declarations.
    """

    patterns = [
        re.compile(name.replace("*", ".*"))
        for name in _iter_keep_in_fp32_declarations(model, attrs=attrs)
    ]
    if not patterns:
        return None

    def matches(name: str) -> bool:
        return any(pattern.search(name) is not None for pattern in patterns)

    return matches


CONSTRUCTION_DTYPES_ATTR = "_gptqmodel_construction_parameter_dtypes"


def _record_construction_dtypes(model) -> None:
    """Remember the dtype each parameter had when the model was constructed.

    ``PreTrainedModel.post_init`` runs after the constructor and before
    ``from_pretrained`` loads the checkpoint. Recording the dtypes there lets
    ``assert_loaded_model_state`` distinguish a parameter the model's own code
    allocated in fp32 (Hymba's ``mamba.A_log``) from one a broken load left in
    the wrong dtype (issue #4888). Only dtypes are retained, never tensors.

    Parameters matched by ``_keep_in_fp32_modules`` at construction are
    excluded: the current declaration already governs them, so clearing the
    declaration must make them reportable again (#4170).
    """

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return
    try:
        keep_in_fp32 = _compile_keep_in_fp32_matcher(model)
        dtypes = {
            name: parameter.dtype
            for name, parameter in named_parameters()
            if parameter.is_floating_point()
            and not (keep_in_fp32 is not None and keep_in_fp32(name))
        }
    except Exception:  # pragma: no cover - defensive: exotic parameter proxies
        return
    if not dtypes or all(dtype is torch.float32 for dtype in dtypes.values()):
        # Constructed entirely in fp32: transformers' dtype context was not
        # active, so the record says nothing about intent and must not be used
        # to exempt anything. Keeping the strict check here is what stops a
        # broken cast from being masked.
        return
    object.__setattr__(model, CONSTRUCTION_DTYPES_ATTR, dtypes)


_CONSTRUCTION_DTYPES_LOCK = threading.RLock()
_CONSTRUCTION_DTYPES_DEPTH = 0
_CONSTRUCTION_DTYPES_ORIGINAL_POST_INIT = None


@contextmanager
def record_construction_dtypes():
    """Record construction-time parameter dtypes for models built in this scope.

    ``from_pretrained`` constructs and loads in one call, and
    ``PreTrainedModel.post_init`` is the only point between the two. This
    scope wraps ``post_init`` so ``_record_construction_dtypes`` runs there,
    letting ``assert_loaded_model_state`` accept a parameter the model's own
    constructor allocated in fp32 (#4888). It is a GPT-QModel load-integrity
    check, not a Transformers compatibility shim, so it is installed only for
    the duration of a GPT-QModel load. Overlapping entries are reference
    counted and the last exit restores the original ``post_init``.
    """

    global _CONSTRUCTION_DTYPES_DEPTH, _CONSTRUCTION_DTYPES_ORIGINAL_POST_INIT

    with _CONSTRUCTION_DTYPES_LOCK:
        if _CONSTRUCTION_DTYPES_DEPTH == 0:
            original_post_init = PreTrainedModel.post_init

            @wraps(original_post_init)
            def post_init(self):
                original_post_init(self)
                _record_construction_dtypes(self)

            _CONSTRUCTION_DTYPES_ORIGINAL_POST_INIT = original_post_init
            PreTrainedModel.post_init = post_init
        _CONSTRUCTION_DTYPES_DEPTH += 1
    try:
        yield
    finally:
        with _CONSTRUCTION_DTYPES_LOCK:
            _CONSTRUCTION_DTYPES_DEPTH -= 1
            if _CONSTRUCTION_DTYPES_DEPTH == 0:
                PreTrainedModel.post_init = _CONSTRUCTION_DTYPES_ORIGINAL_POST_INIT
                _CONSTRUCTION_DTYPES_ORIGINAL_POST_INIT = None


def assert_loaded_model_state(
    model,
    *,
    api_name: str,
    default_dtype_before: Optional[torch.dtype] = None,
    expected_dtype: Optional[torch.dtype] = None,
    require_uniform_dtype: bool = False,
    require_materialized: bool = True,
) -> None:
    """Fail loudly when a load left the process or the model in a mixed state.

    transformers' ``from_pretrained`` saves, sets and restores the process-wide
    default dtype and builds parameters through meta-device and no-init
    contexts. When two loads overlap, a model can come back with mixed
    fp16/fp32 parameters, a tied weight left on the meta device, or the process
    default dtype stuck at the load dtype (issue #4129). A silently mixed model
    is worse than a slow, serialized load, so the loaders call this after every
    build and raise instead of returning the corrupted model.

    ``require_materialized`` and the dtype checks are skipped for the
    ``offload_to_disk`` meta shell, which is intentionally meta until its
    ``LazyTurtle`` materializes each layer. Objects without
    ``named_parameters`` (the MLX runtime model) only get the default-dtype
    check.

    transformers deliberately keeps the modules listed in
    ``_keep_in_fp32_modules`` / ``_keep_in_fp32_modules_strict`` in float32
    inside a half-precision model, so parameters matched by those declarations
    are excluded from both dtype checks *while they are float32*: that is the
    dtype transformers' ``_get_dtype_plan`` pins them to. A declared module left
    in any other dtype is still reported. Pinned parameters are still checked
    for materialization, and undeclared mixed dtypes still fail (issue #4170).

    A model's own constructor may also allocate a parameter in fp32 from an
    explicit fp32 tensor (Hymba's ``mamba.A_log``), and transformers loads the
    half-precision checkpoint value into it without changing its dtype. The
    dtype each parameter had at construction is recorded inside
    ``post_init`` by ``record_construction_dtypes()``; a parameter that is *still* fp32 after the load is treated
    as intentional. A parameter the load changed away from its construction
    dtype is still reported, so a broken cast is never masked (issue #4888).
    """

    problems = []

    if default_dtype_before is not None:
        default_dtype_after = torch.get_default_dtype()
        if default_dtype_after != default_dtype_before:
            problems.append(
                "torch.get_default_dtype() changed from "
                f"{default_dtype_before} to {default_dtype_after}"
            )

    named_parameters = getattr(model, "named_parameters", None)
    if named_parameters is not None and (
        require_materialized or require_uniform_dtype or expected_dtype is not None
    ):
        keep_in_fp32 = _compile_keep_in_fp32_matcher(model)
        construction_dtypes = getattr(model, CONSTRUCTION_DTYPES_ATTR, None)
        dtypes = Counter()
        meta = []
        for name, parameter in named_parameters():
            if not parameter.is_floating_point():
                continue
            if parameter.is_meta:
                meta.append(name)
                continue
            if (
                parameter.dtype is torch.float32
                and keep_in_fp32 is not None
                and keep_in_fp32(name)
            ):
                continue
            if (
                parameter.dtype is torch.float32
                and construction_dtypes is not None
                and construction_dtypes.get(name) is torch.float32
            ):
                # The constructor allocated this parameter in fp32 and the load
                # did not change it (Hymba's `mamba.A_log`, #4888). A parameter
                # constructed in a half dtype and then left/cast to fp32 is
                # still reported below.
                continue
            dtypes[parameter.dtype] += 1

        if require_materialized and meta:
            problems.append(
                f"{len(meta)} floating parameter(s) stayed on the meta device, "
                f"e.g. {meta[:3]}"
            )
        if expected_dtype is not None:
            wrong = {str(dtype): count for dtype, count in dtypes.items() if dtype != expected_dtype}
            if wrong:
                problems.append(
                    f"floating parameters are not all {expected_dtype}: {wrong}"
                )
        elif require_uniform_dtype and len(dtypes) > 1:
            problems.append(
                "floating parameters have mixed dtypes: "
                + str({str(dtype): count for dtype, count in dtypes.items()})
            )

    if problems:
        raise RuntimeError(
            f"{api_name}: model load left a corrupted state: " + "; ".join(problems)
        )


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


# Options that decide *which* config file a load reads. Model-type detection and
# the checkpoint loader must agree on all of them: otherwise dispatch can
# inspect the repository root or the default revision while the loader targets a
# nested directory or another revision.
CONFIG_SELECTION_KWARGS = (
    "subfolder",
    "revision",
    "cache_dir",
    "local_files_only",
    # A private or gated repository needs the same credential for the config
    # read as the weights download. Leaving `token` out made model-type
    # detection fail on a repository that `snapshot_download` could open
    # (#5007).
    "token",
    "use_auth_token",
    "force_download",
    "resume_download",
    "proxies",
    "_commit_hash",
)


def get_config_selection_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the config-selection options the caller actually supplied.

    ``None`` is skipped so an unset option keeps the ``AutoConfig`` default
    instead of being forced to ``None``; explicit ``False``/``""`` values are
    preserved because they are meaningful (``subfolder=""`` means the root).
    ``subfolder`` is validated here so every public entry point rejects an
    escaping path before it reaches the filesystem or the Hub.
    """

    selected = {name: kwargs[name] for name in CONFIG_SELECTION_KWARGS if kwargs.get(name) is not None}
    if "subfolder" in selected:
        selected["subfolder"] = normalize_checkpoint_subfolder(selected["subfolder"])
    return selected


def normalize_checkpoint_subfolder(subfolder: Optional[str]) -> str:
    """Validate a repo-relative ``subfolder`` and return its canonical spelling.

    A ``subfolder`` names a directory *inside* the checkpoint or Hub repository,
    so it may only narrow the base path. Absolute paths, Windows drive letters,
    backslash separators, and ``.``/``..`` segments are rejected instead of
    being normalized away, because silently escaping the checkpoint would make
    the quantize config, the weights, the tokenizer, and the adapter manifest
    resolve against different roots.

    Returns ``""`` when the checkpoint root itself is requested.
    """

    if subfolder is None:
        return ""
    text = str(subfolder)
    if text in ("", "."):
        return ""
    if "\\" in text or os.path.isabs(text) or ntpath.isabs(text) or ntpath.splitdrive(text)[0]:
        raise ValueError(
            f"`subfolder` must be a relative path inside the checkpoint, got `{text}`."
        )
    parts = [part for part in text.split("/") if part]
    if not parts:
        return ""
    if any(part in (".", "..") for part in parts):
        raise ValueError(
            f"`subfolder` must not contain `.` or `..` segments, got `{text}`."
        )
    return "/".join(parts)


def resolve_checkpoint_subfolder(base: Any, subfolder: Optional[str]) -> str:
    """Resolve a repo-relative ``subfolder`` under a local checkpoint root.

    See :func:`normalize_checkpoint_subfolder` for the accepted syntax. A
    resolved path that leaves the checkpoint through a symlink is rejected.
    """

    base_path = os.fspath(base)
    canonical = normalize_checkpoint_subfolder(subfolder)
    if not canonical:
        return base_path

    joined = os.path.join(base_path, *canonical.split("/"))
    if os.path.isdir(base_path) and os.path.exists(joined):
        base_real = os.path.realpath(base_path)
        try:
            inside = os.path.commonpath([base_real, os.path.realpath(joined)]) == base_real
        except ValueError:
            inside = False
        if not inside:
            raise ValueError(
                f"`subfolder` must stay inside the checkpoint directory, got `{canonical}`."
            )
    return joined


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


def _internal_gguf_torch_loader_enabled(policy=None) -> bool:
    """Read the load policy, never the process environment.

    ``None`` resolves the policy this load installed; a direct caller may pass
    the explicit ``KernelConfig`` it wants the dense import to honour.
    """
    from . import internal_gguf

    return internal_gguf.torch_dequant_loader_enabled(policy)


def _load_gguf_checkpoint_torch_direct(
    *,
    gguf_utils,
    original_load_gguf_checkpoint,
    gguf_checkpoint_path,
    return_tensors: bool = False,
    model_to_load=None,
    torch_dtype=None,
    policy=None,
):
    from ..utils import internal_gguf

    if not return_tensors or model_to_load is None or not _internal_gguf_torch_loader_enabled(policy):
        return original_load_gguf_checkpoint(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
            torch_dtype=torch_dtype,
        )

    parsed_parameters = original_load_gguf_checkpoint(
        gguf_checkpoint_path,
        return_tensors=False,
        model_to_load=model_to_load,
        torch_dtype=torch_dtype,
    )
    config = parsed_parameters.get("config", {})
    model_type = config.get("model_type")
    if model_type != internal_gguf.MODEL_ARCH_QWEN3:
        return original_load_gguf_checkpoint(
            gguf_checkpoint_path,
            return_tensors=True,
            model_to_load=model_to_load,
            torch_dtype=torch_dtype,
        )

    processor_cls = gguf_utils.TENSOR_PROCESSORS.get(model_type, gguf_utils.TensorProcessor)
    if processor_cls is not gguf_utils.TensorProcessor:
        return original_load_gguf_checkpoint(
            gguf_checkpoint_path,
            return_tensors=True,
            model_to_load=model_to_load,
            torch_dtype=torch_dtype,
        )

    processor = processor_cls(config=config)
    tensor_key_mapping = gguf_utils.get_gguf_hf_weights_map(model_to_load, processor)
    target_device = internal_gguf._resolve_torch_dequant_device(policy)
    dequantize_kwargs = {"device": target_device}
    if torch_dtype is not None:
        dequantize_kwargs["dtype"] = torch_dtype
    reader = internal_gguf.GGUFReader(gguf_checkpoint_path)
    parsed_parameters["tensors"] = {}

    for tensor in gguf_utils.tqdm(reader.tensors, desc="Converting GGUF tensors directly to torch..."):
        name = tensor.name
        weights = internal_gguf.dequantize_to_torch(
            tensor.data,
            tensor.tensor_type,
            **dequantize_kwargs,
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
            # GGUF processors return nested sequences or array-likes. Normalize
            # through the array's own API so big-endian dtypes and negative
            # strides become a native C-contiguous buffer, then let Torch own
            # the copy. No NumPy import is required.
            dtype = getattr(weights, "dtype", None)
            newbyteorder = getattr(dtype, "newbyteorder", None)
            if newbyteorder is not None:
                weights = weights.astype(newbyteorder("="), order="C", copy=True)
            weights = torch.as_tensor(weights).clone().contiguous()
            if target_device is not None:
                weights = weights.to(device=target_device)

        parsed_parameters["tensors"][tensor_key_mapping[name]] = weights.contiguous()

    return parsed_parameters


def _patch_transformers_internal_gguf_torch_loader(gguf_utils) -> None:
    with _MONKEY_PATCH_LOCK:
        if getattr(gguf_utils, "_GPTQMODEL_GGUF_TORCH_LOADER_PATCHED", False):
            return

        original_load_gguf_checkpoint = gguf_utils.load_gguf_checkpoint

        def _wrapped_load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False, model_to_load=None, torch_dtype=None):
            try:
                return _load_gguf_checkpoint_torch_direct(
                    gguf_utils=gguf_utils,
                    original_load_gguf_checkpoint=original_load_gguf_checkpoint,
                    gguf_checkpoint_path=gguf_checkpoint_path,
                    return_tensors=return_tensors,
                    model_to_load=model_to_load,
                    torch_dtype=torch_dtype,
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
                    torch_dtype=torch_dtype,
                )

        gguf_utils._gptqmodel_original_load_gguf_checkpoint = original_load_gguf_checkpoint
        gguf_utils.load_gguf_checkpoint = _wrapped_load_gguf_checkpoint
        gguf_utils._GPTQMODEL_GGUF_TORCH_LOADER_PATCHED = True


def _patch_transformers_prism_gguf_compat(*, api_name: str) -> None:
    from ..nn_modules.qlinear.gguf import (
        PRISM_Q1_0_G128_BLOCK_SIZE,
        PRISM_Q1_0_G128_NAME,
        PRISM_Q1_0_G128_TYPE_SIZE,
        PRISM_Q1_0_G128_VALUE,
        _dequantize_prism_q1_0_g128,
        _is_prism_q1_0_g128,
    )
    from ..utils import internal_gguf

    try:
        import transformers.modeling_gguf_pytorch_utils as gguf_utils
        from transformers.utils import import_utils as hf_import_utils
    except Exception:
        log.debug("prism GGUF compat skipped: transformers GGUF internals unavailable", exc_info=True)
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


def _detect_native_transformers_causallm_support(model_id_or_path: str, **config_selection_kwargs) -> tuple[bool, Optional[str], Optional[str]]:
    # Qwen-Drive's top-level config is registered by the optional official
    # package, so it must be imported before Transformers first sees config.json.
    ensure_qwen_drive_registered(model_id_or_path)
    config_load_kwargs: dict[str, Any] = dict(config_selection_kwargs)
    gguf_file = config_load_kwargs.pop("gguf_file", None)
    if gguf_file is not None:
        config_load_kwargs[INTERNAL_HF_GGUF_FILE_KWARG] = gguf_file
    normalized_model_id_or_path = normalize_model_id_or_path_for_hf_gguf(
        model_id_or_path,
        config_load_kwargs,
        api_name="_detect_native_transformers_causallm_support",
    )
    try:
        config = AutoConfig.from_pretrained(
            normalized_model_id_or_path,
            trust_remote_code=False,
            **get_config_selection_kwargs(config_selection_kwargs),
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


def load_model_config(model_id_or_path, *, trust_remote_code: bool = False, **kwargs):
    """Read config data first, then select package-owned or native config code.

    Preserve revision/subfolder/auth kwargs for both the data probe and final
    load. A model port opts in through its definition's LocalModelLoader.
    """
    from ..models.local import local_model_loader_for_config

    payload, _ = PreTrainedConfig.get_config_dict(model_id_or_path, **kwargs)
    loader = local_model_loader_for_config(payload)
    if loader is not None:
        return loader.load_config(model_id_or_path, **kwargs)
    ensure_qwen_drive_registered(model_id_or_path)
    config_selection_kwargs = get_config_selection_kwargs(kwargs)
    # GGUF metadata is the selected config. The trust preflight must inspect
    # the same file, including when an unrelated config.json is also present.
    if kwargs.get("gguf_file") is not None:
        config_selection_kwargs["gguf_file"] = kwargs["gguf_file"]
    trust_remote_code = resolve_trust_remote_code(
        model_id_or_path, trust_remote_code=trust_remote_code, **config_selection_kwargs
    )
    return AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code, **kwargs)


def resolve_trust_remote_code(model_id_or_path: Optional[str], *, trust_remote_code: bool, **config_load_kwargs) -> bool:
    # Disabling Python requires no metadata, network access or registration.
    # The config loader owns malformed/missing asset diagnostics and retries.
    if not trust_remote_code:
        return False

    from ..models.local import local_model_loader_for_config

    if model_id_or_path:
        payload, _ = PreTrainedConfig.get_config_dict(model_id_or_path, **config_load_kwargs)
        if local_model_loader_for_config(payload) is not None:
            return False
    if not model_id_or_path:
        return require_remote_code_permission(trust_remote_code)

    ensure_qwen_drive_registered(model_id_or_path)
    native_supported, model_type, model_cls_name = _detect_native_transformers_causallm_support(str(model_id_or_path), **config_load_kwargs)
    if not native_supported:
        # Multimodal definitions use auto classes other than causal LM. Inspect
        # their declared loader before considering any checkpoint Python.
        from ..models.auto import MODEL_MAP_CASEFOLD

        definition = MODEL_MAP_CASEFOLD.get(str(payload.get("model_type", "")).lower())
        if has_native_transformers_model_support(model_id_or_path, getattr(definition, "loader", None), **config_load_kwargs):
            return False
        return require_remote_code_permission(trust_remote_code)

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


def has_native_transformers_model_support(model_id_or_path, loader, **config_load_kwargs) -> bool:
    """Probe the definition's declared native auto class, including multimodal models."""
    from ..models.local import LocalModelLoader

    if isinstance(loader, LocalModelLoader):
        return True
    try:
        config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=False, **config_load_kwargs)
        return type(config) in loader._model_mapping
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        return False


def has_native_transformers_causallm_support(model_id_or_path: Optional[str]) -> bool:
    if not model_id_or_path:
        return False

    native_supported, _, _ = _detect_native_transformers_causallm_support(str(model_id_or_path))
    return native_supported

def load_tokenizer(tokenizer_or_path, *, model_config: Any = None, **kwargs):
    from tokenicer import Tokenicer

    warnings.warn(
        "gptqmodel.utils.hf.load_tokenizer() is deprecated; use Tokenicer.load(..., model_config=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Tokenicer.load(tokenizer_or_path, model_config=model_config, **kwargs)


_REMOTE_ONLY_TOKENIZER_WARNED: set[tuple[str, str]] = set()


def _remote_only_tokenizer_message(model_type: str) -> str:
    return (
        f"`{model_type}` checkpoints ship a customized tokenizer that exists only as checkpoint Python "
        "(tokenizer_config.json auto_map). Loading it is unsafe and requires both trust_remote_code=True "
        "and GPTQ_BYPASS_SAFETY_CHECKS=1. The model itself always uses GPT-QModel's package-owned implementation."
    )


def _warn_remote_only_tokenizer(model_type: str, tokenizer_path) -> None:
    key = (model_type, os.fspath(tokenizer_path))
    if key not in _REMOTE_ONLY_TOKENIZER_WARNED:
        _REMOTE_ONLY_TOKENIZER_WARNED.add(key)
        log.warning(_remote_only_tokenizer_message(model_type))


_VENDORED_TOKENIZER_PREFIX = "gptqmodel.models.ports.native."


def _register_vendored_tokenizer(tokenizer) -> None:
    """Tell Tokenicer a vendored 5.23 tokenizer class is Transformers' own class.

    Tokenicer gates normalization (SentencePiece handling for
    ``LlamaTokenizer``, serialized ``tokenizer.json`` preference) on the
    Transformers class. A class in ``gptqmodel.models.ports.native`` is an exact
    copy of ``transformers.<same name>``, so declare that equivalence explicitly
    instead of handing Tokenicer a ``transformers.models`` class.
    """

    from tokenicer import register_transformers_equivalent

    tokenizer_class = type(tokenizer)
    if (getattr(tokenizer_class, "__module__", "") or "").startswith(_VENDORED_TOKENIZER_PREFIX):
        register_transformers_equivalent(tokenizer_class)


def load_hf_tokenizer(
        tokenizer_or_path,
        *,
        model_config: Any = None,
        trust_remote_code: bool = False,
        return_tokenicer: bool = False,
        **kwargs,
):
    from tokenicer import Tokenicer

    from ..models.local import local_model_loader

    model_type = str(getattr(model_config, "model_type", None) or "").lower()
    loader = local_model_loader(model_type)
    checkpoint_path = isinstance(tokenizer_or_path, (str, os.PathLike))
    # Some package-owned models were released with a tokenizer that exists only
    # as checkpoint Python. The model stays package-owned; the tokenizer alone
    # may use checkpoint code under the same two-flag consent as any other.
    remote_tokenizer = bool(
        loader is not None and checkpoint_path and model_type in loader.remote_tokenizer_model_types
    )
    if remote_tokenizer:
        _warn_remote_only_tokenizer(model_type, tokenizer_or_path)
    # Native tokenizers can load even if a caller supplied the trust flag.
    # Keep the request separate from permission so a remote-only tokenizer
    # can report the missing consent without enabling Python to detect it.
    requested_remote_code = bool(trust_remote_code and checkpoint_path and (loader is None or remote_tokenizer))
    trust_remote_code = requested_remote_code and safety_checks_bypassed()
    tokenicer_option_names = {"normalize_sentencepiece"}
    if loader is not None:
        for key, value in loader.tokenicer_options:
            kwargs.setdefault(key, value)
            tokenicer_option_names.add(key)
    if checkpoint_path:
        tokenizer_or_path = os.fspath(tokenizer_or_path)
        if loader is not None and loader.tokenizer_class is not None:
            # Class selection belongs to the definition, including when two
            # checkpoints declare the same tokenizer class name. Tokenicer
            # receives the owned instance and remains the normalization owner.
            tokenizer_kwargs = {
                key: value for key, value in kwargs.items()
                if key not in {"config", "strict", "pad_tokens", "prefer_serialized_tokenizer"}
                and key not in tokenicer_option_names
            }
            tokenizer_or_path = loader.load_tokenizer(tokenizer_or_path, **tokenizer_kwargs)
    _register_vendored_tokenizer(tokenizer_or_path)
    if isinstance(model_config, PreTrainedConfig):
        # Tokenicer's model_config controls normalization; AutoTokenizer uses
        # the separate `config` argument for class selection. Reuse the config
        # already selected by our policy so tokenizer initialization cannot
        # import checkpoint AutoConfig code behind a definition-owned port.
        kwargs["config"] = model_config
    try:
        tokenicer = Tokenicer.load(
            tokenizer_or_path,
            model_config=model_config,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except ValueError as exc:
        # Tokenicer and Transformers both include this option in their
        # remote-code-required errors and expose no dedicated exception type.
        if requested_remote_code and not trust_remote_code and "`trust_remote_code=True`" in str(exc):
            require_remote_code_permission(requested_remote_code)
        if remote_tokenizer and not trust_remote_code and "trust_remote_code" in str(exc):
            raise RemoteCodeSecurityError(_remote_only_tokenizer_message(model_type)) from exc
        raise
    # Internal model loads retain the normalized wrapper through construction.
    # Direct callers keep the native-tokenizer return contract by default.
    return tokenicer if return_tokenicer else tokenicer.tokenizer



def _sampling_fields_request_sampling(cfg: GenerationConfig) -> bool:
    """Whether sampling-only fields contradict an explicitly greedy config.

    Transformers refuses to save a config that sets these fields while
    ``do_sample`` is ``False``. The checkpoint's fields are authoritative, so
    this is used only to repair that specific contradiction.
    """
    if getattr(cfg, "temperature", None) is not None and cfg.temperature != 1.0:
        return True
    if getattr(cfg, "top_p", None) is not None and cfg.top_p != 1.0:
        return True
    if getattr(cfg, "min_p", None) is not None:
        return True
    if getattr(cfg, "typical_p", None) is not None and cfg.typical_p != 1.0:
        return True
    # contrastive search uses top_k
    if (
        getattr(cfg, "top_k", None) is not None
        and cfg.top_k != 50
        and getattr(cfg, "penalty_alpha", None) is None
    ):
        return True
    if getattr(cfg, "epsilon_cutoff", None) is not None and cfg.epsilon_cutoff != 0.0:
        return True
    if getattr(cfg, "eta_cutoff", None) is not None and cfg.eta_cutoff != 0.0:
        return True
    return False


def _sanitize_generation_config(cfg: GenerationConfig, *, drop_sampling_fields: bool = False) -> bool:
    changed = False
    if cfg is None:
        return changed

    # `do_sample` is owned by the checkpoint: never force sampling on. An
    # absent value means the transformers default (greedy), and an explicit
    # value must survive load/save unchanged. The only repair is an explicitly
    # greedy config whose own sampling fields contradict it, which transformers
    # rejects when saving.
    if getattr(cfg, "do_sample", None) is False and _sampling_fields_request_sampling(cfg):
        cfg.do_sample = True
        changed = True
        log.info("Model: Auto-Fixed `generation_config` by setting `do_sample=True`.")

    return changed


def _read_generation_config_payload(path: str) -> Optional[dict]:
    """Read the raw ``generation_config.json`` without normalizing its values."""
    try:
        config_path = cached_file(path, "generation_config.json")
    except Exception:
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, TypeError, ValueError):
        return None

    return data if isinstance(data, dict) else None


def _load_sanitized_generation_config(path: str) -> Optional[GenerationConfig]:
    declared = _read_generation_config_payload(path)
    try:
        if declared is None:
            # No readable payload, so there is no declared `do_sample` to
            # preserve. Let transformers resolve the defaults.
            cfg = GenerationConfig.from_pretrained(pretrained_model_name=path)
        else:
            # Load with `do_sample=True` so transformers does not warn that the
            # file's sampling fields are invalid, then restore the declared
            # value. The file remains authoritative.
            cfg = GenerationConfig.from_pretrained(pretrained_model_name=path, do_sample=True)
            cfg.do_sample = declared.get("do_sample") if "do_sample" in declared else None
    except Exception:
        log.debug("generation config %s could not be loaded; falling back to defaults", path, exc_info=True)
        return None

    _sanitize_generation_config(cfg, drop_sampling_fields=False)
    return cfg


# TODO FIXME! Pre-quantized use AutoModelForCausalLM.from_pretrained() but post-quantized use AutoModelForCausalLM.from_config()
def autofix_hf_model_config(model: PreTrainedModel, path: str = None):
    if model.can_generate():
        # Specialized generation implementations (for example DiffusionGemma)
        # attach a GenerationConfig subclass with structured, model-specific
        # fields. Reconstructing it as the generic GenerationConfig destroys
        # those fields (notably sampler_config), and AR-only sanitization is not
        # applicable to their generate() implementation.
        if type(model.generation_config) is not GenerationConfig:
            log.info(
                "Model: Preserving specialized `generation_config` type `%s`.",
                type(model.generation_config).__name__,
            )
            return

        # sync config first
        if path:
            log.info(f"Model: Loaded `generation_config`: {model.generation_config}")
            try:
                cfg = _load_sanitized_generation_config(path)
                if cfg is None:
                    cfg = GenerationConfig.from_pretrained(pretrained_model_name=path)
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


def _generation_config_payload_contradicts_greedy(data: dict) -> bool:
    """JSON-payload counterpart of ``_sampling_fields_request_sampling``."""
    if data.get("do_sample") is not False:
        return False

    temperature = data.get("temperature")
    if temperature is not None and temperature != 1.0:
        return True
    top_p = data.get("top_p")
    if top_p is not None and top_p != 1.0:
        return True
    if data.get("min_p") is not None:
        return True
    typical_p = data.get("typical_p")
    if typical_p is not None and typical_p != 1.0:
        return True
    top_k = data.get("top_k")
    if top_k is not None and top_k != 50 and data.get("penalty_alpha") is None:
        return True
    epsilon_cutoff = data.get("epsilon_cutoff")
    if epsilon_cutoff is not None and epsilon_cutoff != 0.0:
        return True
    eta_cutoff = data.get("eta_cutoff")
    if eta_cutoff is not None and eta_cutoff != 0.0:
        return True
    return False


def sanitize_generation_config_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return False
    except (OSError, TypeError, ValueError):
        return False

    if not isinstance(data, dict):
        return False

    # Preserve the checkpoint's declared `do_sample`; never add or force it.
    # Only an explicit greedy declaration contradicted by its own sampling
    # fields is repaired, because transformers rejects that payload on save.
    if not _generation_config_payload_contradicts_greedy(data):
        return False

    data["do_sample"] = True
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
    return True

# load hf model with empty tensors on meta device (zero tensor memory usage)
def build_shell_model(
    loader,
    config: Any,
    trust_remote_code: bool = False,
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
    from ..models.local import LocalModelLoader

    if isinstance(loader, LocalModelLoader):
        trust_remote_code = False
    else:
        require_remote_code_permission(trust_remote_code)

    del init_kwargs["device_map"]
    del init_kwargs["_fast_init"]

    # All nn.Parameters and buffers are created on 'meta' and initializers are skipped.
    pb = log.spinner(title="Model loading...", interval=0.1)
    try:
        with init_empty_weights(include_buffers=True):
            shell = loader.from_config(
                config,
                trust_remote_code=trust_remote_code,
                **init_kwargs,
            )
    finally:
        pb.close()

    return shell
