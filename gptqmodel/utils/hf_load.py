# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native GPTQModel loading bridge for Transformers / Optimum.

Transformers delegates GPTQ weight loading to Optimum, which replaces every
linear layer with one homogeneous kernel chosen from the *global* quantization
config. GPTQModel checkpoints can carry per-module settings through
``dynamic``, so a homogeneous replacement loses the per-module bits and group
sizes and cannot rebuild structural rewrites such as fused or expanded MoE
experts.

The two entry points here let Transformers/Optimum hand the module manifest to
GPTQModel before weights are assigned, then run GPTQModel's own post-init once
the checkpoint tensors are resident. They are a stable public API.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from safetensors import safe_open

from ..models._const import DEVICE
from ..nn_modules.qlinear import BaseQuantLinear
from ..quantization import FORMAT, METHOD, QuantizeConfig
from ..quantization.config import (
    _normalize_quant_bits,
    dynamic_get,
    resolve_model_rotation,
    resolve_quant_format,
)
from .backend import BACKEND, backend_for_activation, normalize_backend
from .hf import get_hf_config_dtype
from .logger import setup_logger
from .torch import defuser_conversion_scope


log = setup_logger()

__all__ = [
    "HFGPTQModelLoadContext",
    "hf_gptqmodel_post_init_for_load",
    "hf_gptqmodel_prepare_model_for_load",
    "hf_gptqmodel_repair_awq_load",
    "hf_gptqmodel_reselect_context",
    "hf_gptqmodel_reselect_offloaded_kernels",
    "hf_gptqmodel_awq_tolerated_mismatch",
    "is_native_gptqmodel_checkpoint",
]

# Files a checkpoint directory is scanned for when resolving provenance.
_QUANTIZE_CONFIG_FILENAMES = ("quantize_config.json", "quant_config.json", "config.json")


@dataclass(frozen=True)
class HFGPTQModelLoadContext:
    """State shared between the pre- and post-weight-load hooks."""

    quantize_config: QuantizeConfig
    # Only the native path installs modules, so the minimal context built for
    # third-party checkpoints (see `hf_gptqmodel_reselect_context`) has none.
    quant_linear: Optional[Type[BaseQuantLinear]] = None
    # Needed to rebuild modules once Transformers resolves `device_map="auto"`
    # from `max_memory`; see `hf_gptqmodel_reselect_offloaded_kernels` (#5194).
    backend: Optional[BACKEND] = None
    dtype: Optional[torch.dtype] = None
    lm_head_name: Optional[str] = None


def _checkpoint_files(checkpoint_files) -> List[str]:
    """Normalize the ``checkpoint_files`` argument into a list of paths."""

    if checkpoint_files is None:
        return []
    if isinstance(checkpoint_files, (str, os.PathLike)):
        checkpoint_files = [checkpoint_files]
    elif isinstance(checkpoint_files, dict):
        checkpoint_files = list(checkpoint_files.values())
    return [os.fspath(path) for path in checkpoint_files if isinstance(path, (str, os.PathLike))]


def find_checkpoint_dir(model: nn.Module, checkpoint_files=None) -> Optional[str]:
    """Resolve the directory that holds the serialized quantization manifest."""

    candidates: List[str] = []
    files = _checkpoint_files(checkpoint_files)
    if len(files) > 1:
        try:
            common = os.path.commonpath([os.path.abspath(path) for path in files])
        except ValueError:
            common = None
        if common is not None:
            candidates.append(common if os.path.isdir(common) else os.path.dirname(common))
    candidates.extend(path if os.path.isdir(path) else os.path.dirname(path) for path in files)

    config = getattr(model, "config", None)
    for attribute in ("_name_or_path", "name_or_path"):
        model_path = getattr(config, attribute, None)
        if isinstance(model_path, (str, os.PathLike)) and os.path.isdir(os.fspath(model_path)):
            candidates.append(os.fspath(model_path))

    for candidate in dict.fromkeys(candidates):
        if any(os.path.isfile(os.path.join(candidate, name)) for name in _QUANTIZE_CONFIG_FILENAMES):
            return candidate
    return None


def _quantizer_entries(manifest: Dict[str, Any]) -> List[str]:
    """Return the normalized ``meta.quantizer`` provenance list from a manifest."""

    quantize_config = manifest.get("quantization_config", manifest)
    if not isinstance(quantize_config, dict):
        return []
    meta = quantize_config.get("meta")
    quantizers = meta.get("quantizer") if isinstance(meta, dict) else None
    if isinstance(quantizers, str):
        quantizers = [quantizers]
    if not isinstance(quantizers, (list, tuple)):
        return []
    return [str(entry).lower().replace("-", "").strip() for entry in quantizers]


def is_native_gptqmodel_checkpoint(model: nn.Module, checkpoint_files=None) -> bool:
    """Return whether the checkpoint names GPTQModel itself as its quantizer.

    Optimum also records a ``gptqmodel:<version>`` entry because it delegates the
    quantization math to GPTQModel, so a checkpoint that names Optimum keeps the
    Optimum path and only GPTQModel-only provenance is treated as native.
    """

    checkpoint_dir = find_checkpoint_dir(model, checkpoint_files)
    if checkpoint_dir is None:
        return False

    for filename in _QUANTIZE_CONFIG_FILENAMES:
        manifest_path = os.path.join(checkpoint_dir, filename)
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        quantizers = _quantizer_entries(manifest)
        if any(entry.startswith("gptqmodel") for entry in quantizers):
            return not any(entry.startswith("optimum") for entry in quantizers)
    return False


def _is_native_gptqmodel_config(qcfg: QuantizeConfig) -> bool:
    """Return whether a loaded config needs GPTQModel's native load path."""

    meta = qcfg.meta if isinstance(qcfg.meta, dict) else {}
    quantizers = meta.get("quantizer") or []
    if isinstance(quantizers, str):
        quantizers = [quantizers]
    normalized = {str(entry).lower().replace("-", "") for entry in quantizers}
    if any(entry.startswith("optimum") for entry in normalized):
        return False
    if any(entry.startswith("gptqmodel") for entry in normalized):
        return True

    uri = str(meta.get("uri") or "").lower().replace("-", "")
    return (
        "gptqmodel" in uri
        or bool(qcfg.dynamic)
        or resolve_quant_format(qcfg.format, qcfg.method) == FORMAT.GPTQ_V2
    )


def _hf_quantization_config_dict(model: nn.Module) -> Optional[Dict[str, Any]]:
    """Return the HF ``quantization_config`` as a plain dict, if present."""

    raw = getattr(getattr(model, "config", None), "quantization_config", None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    to_dict = getattr(raw, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        payload = to_dict()
    except Exception:
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _hf_quantization_config_is_native_awq(model: nn.Module) -> bool:
    """Whether the live HF quantize config describes a native GPT-QModel AWQ checkpoint.

    Reads the same provenance the load bridge uses (``quantize_config.json``
    through Transformers' ``quantization_config``): an explicit
    ``gptqmodel`` quantizer entry, GPT-QModel's ``meta.uri``, or non-empty
    ``dynamic`` per-module overrides. ``dynamic`` is what makes Transformers'
    global-config module build disagree with the checkpoint tensors.
    """

    raw = _hf_quantization_config_dict(model)
    if not raw:
        return False
    # ``quantization_config`` round-trips through Transformers' typed config,
    # whose ``quant_method`` is an enum when the live object is inspected.
    method = raw.get("quant_method") or ""
    method = str(getattr(method, "value", method)).strip().lower()
    if method != "awq":
        return False

    meta = raw.get("meta")
    quantizers = meta.get("quantizer") if isinstance(meta, dict) else None
    if isinstance(quantizers, str):
        quantizers = [quantizers]
    if isinstance(quantizers, (list, tuple)):
        normalized = [str(entry).lower().replace("-", "") for entry in quantizers if entry]
        if any(entry.startswith("optimum") for entry in normalized):
            return False
        if any(entry.startswith("gptqmodel") for entry in normalized):
            return True
        if normalized:
            # An explicit third-party provenance wins: do not reinterpret its
            # `dynamic` field as GPT-QModel's per-module override syntax.
            return False

    uri = str((meta or {}).get("uri") or "").lower().replace("-", "") if isinstance(meta, dict) else ""
    return "gptqmodel" in uri or bool(raw.get("dynamic"))


# Buffers a packed AWQ kernel owns. Only these can legitimately disagree with a
# module built from the global config; a dense tensor whose shape differs is a
# genuine checkpoint error and must stay reported.
_AWQ_PACKED_BUFFERS = ("qweight", "qzeros", "scales", "g_idx", "bias")

_AWQ_QCFG_CACHE_ATTR = "_gptqmodel_awq_tolerated_qcfg"
_AWQ_QCFG_UNSET = object()


def _awq_load_quantize_config(model: nn.Module) -> Optional[QuantizeConfig]:
    """Resolve the checkpoint manifest the AWQ repair will rebuild from.

    Reads the same source as ``hf_gptqmodel_repair_awq_load`` so the tolerance
    decision and the repair agree. Cached per model because Transformers calls
    the mismatch hook once for every packed tensor whose shape differs.
    """

    cached = getattr(model, _AWQ_QCFG_CACHE_ATTR, _AWQ_QCFG_UNSET)
    if cached is not _AWQ_QCFG_UNSET:
        return cached

    context = getattr(model, "_gptqmodel_hf_load_context", None)
    qcfg = getattr(context, "quantize_config", None)
    if qcfg is None:
        checkpoint_path = find_checkpoint_dir(model)
        if checkpoint_path is not None:
            try:
                qcfg = QuantizeConfig.from_pretrained(checkpoint_path)
            except (KeyError, OSError, TypeError, ValueError):
                qcfg = None
    if qcfg is None:
        raw = _hf_quantization_config_dict(model)
        if raw:
            try:
                qcfg = QuantizeConfig.from_quant_config(dict(raw))
            except (KeyError, TypeError, ValueError):
                qcfg = None
    try:
        setattr(model, _AWQ_QCFG_CACHE_ATTR, qcfg)
    except (AttributeError, TypeError):
        pass
    return qcfg


def _awq_module_matches_serialized_config(
    module: BaseQuantLinear,
    qcfg: QuantizeConfig,
    name: str,
    format_code: FORMAT,
) -> bool:
    """Whether a built module already has its serialized per-module geometry."""

    resolved = _resolved_module_config(qcfg, name, format_code)
    if resolved is None:
        return True
    if int(module.bits) != int(resolved.bits):
        return False
    if _awq_effective_group_size(module, module.group_size) != _awq_effective_group_size(
        module, resolved.group_size
    ):
        return False
    if hasattr(module, "desc_act") and bool(module.desc_act) != bool(resolved.desc_act):
        return False
    if hasattr(module, "sym") and bool(module.sym) != bool(resolved.sym):
        return False
    return True


def _awq_effective_group_size(module: BaseQuantLinear, group_size) -> int:
    """Return the group size a kernel actually uses (``-1`` means per-channel)."""

    value = int(group_size)
    if value == -1:
        in_features = getattr(module, "in_features", None)
        if in_features is not None:
            return int(in_features)
    return value


def hf_gptqmodel_awq_tolerated_mismatch(model: nn.Module, param_name: Optional[str] = None) -> bool:
    """Whether Transformers must not reject a native AWQ checkpoint's shape mismatch.

    Transformers 5.23 changed ``set_param_for_module`` so a checkpoint tensor
    whose shape differs from the meta module is recorded as a mismatched key
    unless ``hf_quantizer.param_needs_quantization`` is true:

    ``hf_quantizer is None or not hf_quantizer.param_needs_quantization(...)``

    Before 5.23 any active quantizer skipped that check. ``AwqQuantizer``
    inherits the base ``param_needs_quantization`` (always ``False``) because
    a pre-quantized checkpoint has nothing to quantize, so on 5.23 a native
    ``dynamic`` checkpoint's per-module ``qzeros``/``scales`` are reported as
    MISMATCH and ``_finalize_model_loading`` raises before the post-weight
    repair hook can rebuild the modules (#5322).

    ``replace_with_awq_linear`` imports ``gptqmodel`` mid-load, so the modules
    were built from the *global* config and the checkpoint tensors are the
    authority: accept them, then ``hf_gptqmodel_repair_awq_load`` rebuilds each
    module from the serialized per-module config and moves them across.

    Only a module whose built geometry genuinely disagrees with its serialized
    per-module config may be accepted here: that is exactly the case the repair
    rebuilds. For a native checkpoint without such an override the built module
    already has the serialized geometry, so a differing tensor is a corrupted
    checkpoint and Transformers must keep reporting it (#5345). The hook is only
    consulted for GPT-QModel AWQ checkpoints, so third-party AWQ loads keep
    their existing mismatch behavior.
    """

    if not _hf_quantization_config_is_native_awq(model):
        return False
    if param_name is None:
        return True

    module_path, _, buffer_name = param_name.rpartition(".")
    if buffer_name not in _AWQ_PACKED_BUFFERS:
        return False
    try:
        module = model.get_submodule(module_path)
    except AttributeError:
        return False
    if not isinstance(module, BaseQuantLinear):
        return False

    qcfg = _awq_load_quantize_config(model)
    if qcfg is None:
        # Without the serialized manifest the mismatch cannot be attributed to
        # a per-module geometry difference, so keep Transformers' rejection.
        return False
    format_code = resolve_quant_format(qcfg.format, qcfg.method)
    return not _awq_module_matches_serialized_config(module, qcfg, module_path, format_code)


def hf_gptqmodel_reselect_context(
    model: nn.Module,
    checkpoint_files=None,
    backend: Optional[Union[str, BACKEND]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[HFGPTQModelLoadContext]:
    """Build the minimal context needed to re-select offload-safe kernels.

    ``hf_gptqmodel_prepare_model_for_load`` returns ``None`` for third-party
    AutoAWQ/AutoGPTQ checkpoints, so Transformers/Optimum replace the modules
    with their own path. Those modules are still built before ``max_memory`` is
    applied and therefore keep a CUDA-only Marlin kernel; the resolved map is
    only visible from ``validate_environment`` (#5262).

    This context contains exactly the state
    :func:`hf_gptqmodel_reselect_offloaded_kernels` reads. It is published under
    a separate model attribute so the fallback load keeps its own post-init
    path instead of switching to the native one.
    """

    raw_config = _hf_quantization_config_dict(model)
    qcfg: Optional[QuantizeConfig] = None
    if raw_config is not None:
        try:
            qcfg = QuantizeConfig.from_quant_config(raw_config)
        except (KeyError, TypeError, ValueError):
            qcfg = None
    if qcfg is None:
        checkpoint_path = find_checkpoint_dir(model, checkpoint_files)
        if checkpoint_path is None:
            return None
        try:
            qcfg = QuantizeConfig.from_pretrained(checkpoint_path)
        except (KeyError, OSError, TypeError, ValueError):
            return None

    # The reselection candidates come from GPTQModel's own kernel registry, so
    # only the two methods the HF bridges cover can be re-routed here.
    if qcfg.method not in (METHOD.AWQ, METHOD.GPTQ):
        return None

    try:
        resolved_backend = _resolve_backend_for_checkpoint(qcfg, backend)
    except (NotImplementedError, TypeError, ValueError):
        # An impossible caller choice still belongs to the caller's own load
        # path; do not turn it into a bridge failure.
        return None

    if not isinstance(dtype, torch.dtype):
        config_dtype = get_hf_config_dtype(model.config)
        dtype = config_dtype if isinstance(config_dtype, torch.dtype) else None

    from ..models.auto import MODEL_MAP_CASEFOLD
    from ..models.base import BaseQModel

    model_type = str(getattr(model.config, "model_type", "")).lower()
    model_definition = MODEL_MAP_CASEFOLD.get(model_type, BaseQModel)
    return HFGPTQModelLoadContext(
        quantize_config=qcfg,
        backend=resolved_backend,
        dtype=dtype,
        lm_head_name=getattr(model_definition, "lm_head", None),
    )


def _checkpoint_tensor_keys(checkpoint_path: str, checkpoint_files) -> set:
    """Collect the tensor names a checkpoint provides, preferring the index."""

    keys = set()
    files = _checkpoint_files(checkpoint_files)

    index_files = [
        path for path in files if path.endswith(".index.json") and os.path.isfile(path)
    ]
    index_files.extend(
        os.path.join(checkpoint_path, filename)
        for filename in sorted(os.listdir(checkpoint_path))
        if filename.endswith(".index.json")
    )
    for index_file in dict.fromkeys(index_files):
        try:
            with open(index_file, "r", encoding="utf-8") as handle:
                keys.update(json.load(handle).get("weight_map", {}))
        except (OSError, TypeError, ValueError):
            continue
    if keys:
        return keys

    safetensor_files = [
        path for path in files if path.endswith(".safetensors") and os.path.isfile(path)
    ]
    if not safetensor_files:
        safetensor_files = [
            os.path.join(checkpoint_path, filename)
            for filename in sorted(os.listdir(checkpoint_path))
            if filename.endswith(".safetensors")
        ]
    for safetensor_file in dict.fromkeys(safetensor_files):
        with safe_open(safetensor_file, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


# Backends the normal loader allows for rotated (QuaRot/SpinQuant) checkpoints.
# Kept in sync with `BaseQModel.from_quantized` in `gptqmodel/models/loader.py`.
_ROTATION_BACKENDS = (
    BACKEND.GPTQ_TORCH,
    BACKEND.GPTQ_TRITON,
    BACKEND.GPTQ_W4AFP8,
    BACKEND.GPTQ_W4A_NVFP4,
)


def _resolve_backend_for_checkpoint(qcfg: QuantizeConfig, backend: Optional[Union[str, BACKEND]]) -> BACKEND:
    """Normalize the caller backend the way the normal loader does.

    ``make_quant`` already normalizes the backend on its own, but two checkpoint
    policies have to be applied *before* kernel selection:

    - a saved W4A activation stream pins the matching W4AFP8/NVFP4 kernel, which
      AUTO never selects on its own because those kernels have no AUTO priority;
    - rotated checkpoints are only valid on rotation-aware backends.
    """

    export_quant_method = qcfg.export_quant_method()
    backend = normalize_backend(backend, quant_method=export_quant_method)

    if resolve_model_rotation(qcfg):
        format_code = resolve_quant_format(qcfg.format, qcfg.method)
        if format_code not in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
            raise NotImplementedError(
                f"`rotation` is only supported for GPTQ/GPTQ_V2 checkpoints, got `{format_code}`."
            )
        if backend in (None, BACKEND.AUTO):
            backend = BACKEND.GPTQ_TORCH
        if backend not in _ROTATION_BACKENDS:
            raise NotImplementedError(
                "`rotation` is only supported with `gptq_torch`, `gptq_triton`, "
                f"`gptq_w4afp8`, or `gptq_w4a_nvfp4` backend, got `{backend}`."
            )
        return backend

    backend = backend_for_activation(qcfg.activation_mode, backend)
    return BACKEND.AUTO if backend is None else backend


def hf_gptqmodel_prepare_model_for_load(
    model: nn.Module,
    checkpoint_files=None,
    device_map=None,
    backend: Optional[Union[str, BACKEND]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[HFGPTQModelLoadContext]:
    """Prepare a native GPTQModel checkpoint for Transformers/Optimum loading.

    Rebuilds structural rewrites (fused blocks, expanded MoE experts) and installs
    one kernel per serialized module while honoring the full ``dynamic`` config.
    Returns ``None`` for checkpoints that should keep the caller's own path.
    """

    checkpoint_path = find_checkpoint_dir(model, checkpoint_files)
    if checkpoint_path is None:
        return None

    try:
        qcfg = QuantizeConfig.from_pretrained(checkpoint_path)
    except (KeyError, OSError, TypeError, ValueError):
        return None
    if not _is_native_gptqmodel_config(qcfg):
        return None

    # Serialized packed tensors are authoritative; dynamic rules may also name
    # dense modules. NF12 stores its packed weight under `nf12_packed`.
    packed_suffix = (
        ".nf12_packed"
        if resolve_quant_format(qcfg.format, qcfg.method) == FORMAT.NF12
        else ".qweight"
    )
    tensor_keys = _checkpoint_tensor_keys(checkpoint_path, checkpoint_files)
    quantized_module_names = tuple(
        sorted(
            key.removesuffix(packed_suffix)
            for key in tensor_keys
            if key.endswith(packed_suffix)
        )
    )
    if not quantized_module_names:
        raise ValueError(
            "The GPTQModel checkpoint does not expose any "
            f"`{packed_suffix}` tensors in its checkpoint manifest."
        )

    import defuser

    from ..models.auto import MODEL_MAP_CASEFOLD
    from ..models.base import BaseQModel
    from .importer import auto_cpu_offload, hf_normalize_device_device_map
    from .model import (
        _module_has_meta_tensors,
        _prepare_variable_quant_buffers,
        find_modules,
        get_module_by_name_prefix,
        make_quant,
    )

    model_type = str(getattr(model.config, "model_type", "")).lower()
    model_definition = MODEL_MAP_CASEFOLD.get(model_type, BaseQModel)

    # The checkpoint may use a legacy layout (for example the Transformers 4.x
    # ``language_model.model.layers.*`` keys of a Gemma-3 VL checkpoint) while
    # the live model exposes runtime names. Convert the manifest the same way
    # the native loader does before matching it to ``find_modules`` (#4802).
    key_converter = getattr(model_definition, "checkpoint_key_for_load", None)
    if not callable(key_converter):
        from ..models.loader import _checkpoint_key_converter

        key_converter = _checkpoint_key_converter(model_definition, model, to_runtime=True)
    if callable(key_converter):
        quantized_module_names = tuple(
            sorted({key_converter(name) for name in quantized_module_names})
        )

    defuser.replace_fused_blocks(model_type)

    # Rebuild fused model structures before matching serialized module names.
    def convert_fused_modules():
        defuser.convert_model(model, cleanup_original=True)
        for module_path in getattr(model_definition, "defuser_module_paths", ()) or ():
            module, _ = get_module_by_name_prefix(model, module_path)
            if module is not None:
                defuser.convert_model(module, cleanup_original=True)

    # Expand MoE experts on meta to avoid materializing dense expert weights.
    # Defuser's cache flush synchronizes the current device when it gets no
    # device list; scope it to this model's tensors so the conversion does not
    # claim cuda:0 for a checkpoint placed on cuda:1 (#5105).
    if _module_has_meta_tensors(model):
        with torch.device("meta"), defuser_conversion_scope(model):
            convert_fused_modules()
    else:
        with defuser_conversion_scope(model):
            convert_fused_modules()

    modules = find_modules(model)
    missing_modules = sorted(set(quantized_module_names).difference(modules))
    if missing_modules:
        sample = ", ".join(missing_modules[:8])
        suffix = "" if len(missing_modules) <= 8 else f", ... ({len(missing_modules)} total)"
        raise ValueError(
            "GPTQModel could not match quantized checkpoint tensors to model modules after "
            f"structural conversion: {sample}{suffix}"
        )
    quantized_modules = {name: modules[name] for name in quantized_module_names}

    # Transformers calls this before it applies ``max_memory``, so an accelerate
    # keyword map has no resolved placement yet. Keep the accelerator family so
    # the selected kernel is valid on every visible device instead of rejecting
    # a single-card budget for GPUs it never touches (#5084).
    target_device: DEVICE = hf_normalize_device_device_map(None, device_map)
    qcfg.device = target_device
    if not isinstance(dtype, torch.dtype):
        config_dtype = get_hf_config_dtype(model.config)
        dtype = config_dtype if isinstance(config_dtype, torch.dtype) else None

    backend = _resolve_backend_for_checkpoint(qcfg, backend)

    # Apply full dynamic settings while selecting a supported kernel per module.
    quant_linear = make_quant(
        model,
        qcfg=qcfg,
        quant_result=quantized_modules,
        backend=backend,
        lm_head_name=model_definition.lm_head,
        device=target_device,
        dtype=dtype,
        # A map with CPU entries runs `post_init` on the CPU-resident layers, so
        # AUTO must not select a kernel that needs CUDA there (#5022). Only
        # AUTO consults this: an explicit backend can repack from an in-memory
        # offload weights map.
        cpu_offload=auto_cpu_offload(backend, device_map),
    )
    if packed_suffix == ".nf12_packed":
        # The escape row count depends on the encoded weights, so resize the
        # placeholder buffers before Transformers assigns checkpoint tensors.
        for checkpoint_file in _checkpoint_files(checkpoint_files) or [checkpoint_path]:
            _prepare_variable_quant_buffers(model, checkpoint_file)
    log.info(
        "Transformers/Optimum: prepared `%s` native GPTQModel modules from the checkpoint manifest.",
        len(quantized_module_names),
    )
    return HFGPTQModelLoadContext(
        quantize_config=qcfg,
        quant_linear=quant_linear,
        backend=backend,
        dtype=dtype,
        lm_head_name=model_definition.lm_head,
    )


def _resolved_placement(device_map: Dict[str, Any], module_name: str):
    """Return the placement a resolved `hf_device_map` gives `module_name`.

    Accelerate maps whole modules, so a quantized projection inherits the
    entry of its nearest named ancestor.
    """

    name = module_name
    while True:
        if name in device_map:
            return device_map[name]
        if not name:
            return None
        name = name.rpartition(".")[0]


def hf_gptqmodel_reselect_offloaded_kernels(
    model: nn.Module,
    context: Optional[HFGPTQModelLoadContext],
    device_map,
) -> bool:
    """Rebuild kernels that cannot initialize on a CPU/disk-offloaded layer.

    Transformers hands the bridge the *unresolved* `device_map` keyword
    (`"auto"`, `"balanced"`, ...) during `preprocess_model` and only computes the
    real placement from `max_memory` afterwards, so `auto_cpu_offload` cannot see
    the spill and AUTO keeps `MarlinLinear` / `AwqMarlinLinear`. Those kernels
    need resident CUDA tensors in `post_init`, so the load crashed once
    Accelerate placed a layer on CPU or disk (#5194).

    This runs after the placement is known but before any checkpoint tensor is
    assigned: the affected modules are still meta placeholders, so they are
    rebuilt in place with an offload-safe kernel and no weights are copied.
    Returns whether any module was rebuilt.
    """

    if not isinstance(device_map, dict) or not device_map:
        return False
    if context is None:
        return False
    # An explicit backend is the caller's choice, exactly as in
    # `auto_cpu_offload`: only AUTO may be re-routed here.
    if context.backend not in (None, BACKEND.AUTO, BACKEND.AUTO_TRAINABLE):
        return False

    from .importer import (
        hf_normalize_device_device_map,
        is_offload_placement,
        select_quant_linear,
    )
    from .kernel import snapshot_kernel_config
    from .model import create_quant_layer, find_modules

    modules = find_modules(model, layers=[BaseQuantLinear])
    offloaded = {
        name: module
        for name, module in modules.items()
        if is_offload_placement(_resolved_placement(device_map, name))
        and not getattr(type(module), "SUPPORTS_CPU_OFFLOAD", False)
    }
    if not offloaded:
        return False

    qcfg = context.quantize_config
    format_code = resolve_quant_format(qcfg.format, qcfg.method)
    backend = context.backend
    device = hf_normalize_device_device_map(None, device_map)
    init_kwargs = qcfg.quant_linear_init_kwargs()
    init_kwargs["kernel"] = snapshot_kernel_config(qcfg)

    candidates = select_quant_linear(
        bits=qcfg.runtime_bits,
        group_size=qcfg.group_size,
        desc_act=qcfg.desc_act,
        sym=qcfg.sym,
        backend=backend,
        format=format_code,
        quant_method=qcfg.export_quant_method(),
        pack=False,
        dynamic=qcfg.dynamic,
        pack_dtype=qcfg.pack_dtype,
        device=device,
        dtype=context.dtype,
        multi_select=True,
        cpu_offload=True,
    )
    if not isinstance(candidates, (list, tuple)):
        candidates = [candidates]

    names = sorted(offloaded)
    log.info(
        "Transformers/Optimum: rebuilding `%s` CPU/disk-offloaded module(s) with an "
        "offload-safe kernel: %s",
        len(names),
        ", ".join(names[:4]) + (", ..." if len(names) > 4 else ""),
    )
    create_quant_layer(
        linear_candidates=list(candidates),
        bits=qcfg.runtime_bits,
        desc_act=qcfg.desc_act,
        dynamic=qcfg.dynamic,
        group_size=qcfg.group_size,
        quant_result=dict.fromkeys(names, True),
        module=model,
        sym=qcfg.sym,
        device=device,
        lm_head_name=context.lm_head_name,
        pack_dtype=qcfg.pack_dtype,
        backend=backend,
        adapter=qcfg.adapter,
        format=format_code,
        init_kwargs=init_kwargs,
        dtype=context.dtype,
    )
    return True


def hf_gptqmodel_post_init_for_load(
    model: nn.Module,
    context: HFGPTQModelLoadContext,
    max_input_length: Optional[int] = None,
):
    """Finish a native GPTQModel load after Transformers assigned checkpoint tensors."""

    from ..models.loader import _set_paged_attention_safe_cuda_graphs, _setup_rotation_online_had
    from .model import gptqmodel_post_init, hf_convert_gptq_v1_to_v2_format

    qcfg = context.quantize_config
    # v1->v2 conversion and backend setup require materialized checkpoint tensors.
    model, _ = hf_convert_gptq_v1_to_v2_format(
        model,
        bits=qcfg.bits,
        qlinear_kernel=context.quant_linear,
        checkpoint_format=qcfg.format,
        meta=qcfg.meta,
    )
    model.quantize_config = qcfg

    hf_quantization_config = getattr(getattr(model, "config", None), "quantization_config", None)
    if hf_quantization_config is not None and not isinstance(hf_quantization_config, dict):
        # Transformers GPTQConfig omits fields GPTQModel needs to save and reload.
        hf_quantization_config.dynamic = qcfg.dynamic
        hf_quantization_config.lm_head = qcfg.lm_head
        hf_quantization_config.method = (
            qcfg.method.value if hasattr(qcfg.method, "value") else qcfg.method
        )
        hf_quantization_config.pack_dtype = str(qcfg.pack_dtype).removeprefix("torch.")

    model = gptqmodel_post_init(
        model,
        use_act_order=qcfg.desc_act,
        quantize_config=qcfg,
        max_input_length=max_input_length,
    )

    # Remember the on-disk layout so a caller that later uses
    # `save_pretrained()` emits checkpoint-format qzeros rather than the
    # kernel's converted runtime words.
    from ..nn_modules.qlinear import GPTQQuantLinear

    checkpoint_format = resolve_quant_format(qcfg.format, qcfg.method)
    for module in model.modules():
        if isinstance(module, GPTQQuantLinear):
            module._checkpoint_format_on_disk = checkpoint_format

    # Rotated checkpoints fuse an orthogonal Q and an exact Hadamard H into the
    # weights; the H has to be re-applied to activations at inference. Without
    # this the rotated weights consume untransformed activations and produce
    # silently wrong outputs, so mirror the normal loader's post-init step.
    _setup_rotation_online_had(model, resolve_model_rotation(qcfg))

    # Legacy remote-code families need the same two repairs the native loader
    # applies: transformers 5 leaves checkpoint-absent rotary buffers
    # uninitialized when the class overrides ``_init_weights`` (EXAONE-3.0,
    # DeciLM, ...), and changed the ``generate()`` cache contract those classes
    # implement. Without them a checkpoint that ``GPTQModel.load`` reads
    # correctly produces garbage RoPE frequencies through transformers (#5311).
    from .hf import apply_legacy_remote_code_fixups

    apply_legacy_remote_code_fixups(model)

    # Paged Flash Attention is not CUDA-graph capture safe; the normal loader
    # opts paged models out of graph capture so ``generate_batch`` /
    # ``init_continuous_batching`` work without a caller-side monkey patch.
    _set_paged_attention_safe_cuda_graphs(model)

    # The placement resolved from ``max_memory`` is known now. Keep the
    # single-compute-capability refusal for a map that really spans generations
    # while allowing a budget that names one card (#5084).
    from .importer import validate_device_map_capability_groups

    validate_device_map_capability_groups(getattr(model, "hf_device_map", None))
    return model


def _resolve_checkpoint_dir(model: nn.Module, checkpoint_files=None) -> Optional[str]:
    """Resolve the checkpoint directory, including a hub repo loaded from the cache.

    ``find_checkpoint_dir`` only accepts a local directory, which is what
    Transformers hands the prepare hook. The post-weight repair has no
    ``checkpoint_files``, so a checkpoint loaded from a repo id has to be
    resolved through the hub cache instead.
    """

    checkpoint_path = find_checkpoint_dir(model, checkpoint_files)
    if checkpoint_path is not None:
        return checkpoint_path

    config = getattr(model, "config", None)
    for attribute in ("_name_or_path", "name_or_path"):
        repo_id = getattr(config, attribute, None)
        if not isinstance(repo_id, str) or not repo_id or os.path.isdir(repo_id):
            continue
        from .hub import cached_file

        for filename in _QUANTIZE_CONFIG_FILENAMES:
            try:
                resolved = cached_file(repo_id, filename)
            except Exception:
                continue
            if resolved:
                return os.path.dirname(os.fspath(resolved))
    return None


def _awq_checkpoint_module_names(checkpoint_path: str, checkpoint_files) -> tuple:
    """Return the module paths the AWQ checkpoint serializes packed weights for."""

    tensor_keys = _checkpoint_tensor_keys(checkpoint_path, checkpoint_files)
    return tuple(
        sorted(
            key.removesuffix(".qweight")
            for key in tensor_keys
            if key.endswith(".qweight")
        )
    )


def _resolved_module_config(qcfg: QuantizeConfig, name: str, format_code: FORMAT):
    """Resolve one module's constructor config from the checkpoint manifest."""

    from ..quantization.module_config import resolve_module_constructor_config

    return resolve_module_constructor_config(
        bits=_normalize_quant_bits(qcfg.runtime_bits, format_value=format_code),
        group_size=qcfg.group_size,
        desc_act=qcfg.desc_act,
        sym=qcfg.sym,
        pack_dtype=qcfg.pack_dtype,
        format=format_code,
        init_kwargs=qcfg.quant_linear_init_kwargs(),
        overrides=dynamic_get(dynamic=qcfg.dynamic, module_name=name),
    )


def _awq_modules_ignore_serialized_config(model: nn.Module, qcfg: QuantizeConfig, names) -> bool:
    """Whether Transformers built these modules from the global config only.

    Transformers' AWQ loader replaces every ``nn.Linear`` with one kernel chosen
    from the *global* ``bits``/``group_size`` and never consults ``dynamic``.
    The per-module config is authoritative, so a mismatch in bits, group size,
    activation order, or zero-point symmetry means the module decodes the
    checkpoint's tensors with the wrong geometry.
    """

    from .model import find_modules

    format_code = resolve_quant_format(qcfg.format, qcfg.method)
    # Transformers already replaced the quantized linears, so only the packed
    # modules remain candidates (``find_modules`` defaults to dense types).
    modules = find_modules(model, layers=[BaseQuantLinear])
    for name in names:
        module = modules.get(name)
        if not isinstance(module, BaseQuantLinear):
            continue
        if not _awq_module_matches_serialized_config(module, qcfg, name, format_code):
            return True
    return False


def hf_gptqmodel_repair_awq_load(
    model: nn.Module,
    *,
    backend: Optional[Union[str, BACKEND]] = None,
    checkpoint_files=None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[HFGPTQModelLoadContext]:
    """Repair a native AWQ load that started before the bridge was installed.

    ``transformers.integrations.awq.replace_with_awq_linear`` imports
    ``gptqmodel`` from inside its own body, so the AWQ quantizer's prepare hook
    is installed too late to run for the load that triggered that import: every
    module is built from the *global* ``bits``/``group_size`` and ``dynamic``
    per-module overrides are ignored (#4795). On CPU the load then decodes the
    checkpoint's group-sized tensors with the wrong group size (silently wrong
    logits); on CUDA the Marlin pack raises a qzeros shape error.

    By the time the post-weight hook runs the checkpoint tensors are resident,
    so this rebuilds each checkpoint-backed module from its resolved per-module
    config and kernel class, moves the loaded tensors into it, and returns the
    context for the native post-init. Returns ``None`` when the checkpoint is
    not a GPT-QModel-native AWQ checkpoint or when the modules Transformers
    built already match the serialized config, so those loads keep their
    existing path.
    """

    checkpoint_path = _resolve_checkpoint_dir(model, checkpoint_files)
    if checkpoint_path is None:
        return None
    try:
        qcfg = QuantizeConfig.from_pretrained(checkpoint_path)
    except (KeyError, OSError, TypeError, ValueError):
        return None
    if qcfg.method != METHOD.AWQ or not _is_native_gptqmodel_config(qcfg):
        return None

    quantized_module_names = _awq_checkpoint_module_names(checkpoint_path, checkpoint_files)
    if not quantized_module_names:
        return None

    from ..models.auto import MODEL_MAP_CASEFOLD
    from ..models.base import BaseQModel
    from ..utils.importer import select_quant_linear
    from .kernel import snapshot_kernel_config
    from .model import create_quant_layer, find_modules

    modules = find_modules(model, layers=[BaseQuantLinear])
    loaded = {
        name: modules[name]
        for name in quantized_module_names
        if name in modules
    }
    dropped = sorted(set(quantized_module_names) - set(loaded))
    if dropped:
        sample = ", ".join(dropped[:4])
        suffix = "" if len(dropped) <= 4 else f", ... ({len(dropped)} total)"
        raise ValueError(
            "This GPT-QModel AWQ checkpoint serializes quantized tensors that "
            f"Transformers' built-in AWQ loader did not consume: {sample}{suffix}. "
            "Its quantizer imports `gptqmodel` from inside "
            "`replace_with_awq_linear`, so the native bridge cannot rebuild "
            "fused or per-expert modules for this load, and continuing would "
            "leave those modules randomly initialized. Import `gptqmodel` "
            "before calling `AutoModelForCausalLM.from_pretrained`, or load the "
            "checkpoint with `GPTQModel.load`."
        )

    if not _awq_modules_ignore_serialized_config(model, qcfg, quantized_module_names):
        return None

    format_code = resolve_quant_format(qcfg.format, qcfg.method)
    export_method = qcfg.export_quant_method()
    backend = _resolve_backend_for_checkpoint(qcfg, backend)
    if not isinstance(dtype, torch.dtype):
        # The kernel's activation dtype is the width the checkpoint tensors
        # were assigned at, not necessarily the config's requested dtype:
        # Transformers' AWQ quantizer casts bf16 to fp16 on CUDA/XPU.
        scales = getattr(next(iter(loaded.values())), "scales", None)
        if isinstance(scales, torch.Tensor) and scales.is_floating_point():
            dtype = scales.dtype
        else:
            config_dtype = get_hf_config_dtype(model.config)
            dtype = config_dtype if isinstance(config_dtype, torch.dtype) else None

    devices = sorted({module.qweight.device for module in loaded.values()}, key=str)
    device = devices[0] if len(devices) == 1 else tuple(devices)
    bits = qcfg.runtime_bits
    init_kwargs = qcfg.quant_linear_init_kwargs()
    init_kwargs["kernel"] = snapshot_kernel_config(qcfg)

    candidates = select_quant_linear(
        bits=bits,
        group_size=qcfg.group_size,
        desc_act=qcfg.desc_act,
        sym=qcfg.sym,
        backend=backend,
        format=format_code,
        quant_method=export_method,
        pack=False,
        dynamic=qcfg.dynamic,
        pack_dtype=qcfg.pack_dtype,
        device=device,
        dtype=dtype,
        multi_select=True,
    )
    if not isinstance(candidates, (list, tuple)):
        candidates = [candidates]

    model_type = str(getattr(model.config, "model_type", "")).lower()
    model_definition = MODEL_MAP_CASEFOLD.get(model_type, BaseQModel)
    quant_linear = create_quant_layer(
        linear_candidates=list(candidates),
        bits=bits,
        desc_act=qcfg.desc_act,
        dynamic=qcfg.dynamic,
        group_size=qcfg.group_size,
        quant_result=dict.fromkeys(quantized_module_names, True),
        module=model,
        sym=qcfg.sym,
        device=device,
        lm_head_name=model_definition.lm_head,
        pack_dtype=qcfg.pack_dtype,
        backend=backend,
        format=format_code,
        init_kwargs=init_kwargs,
        dtype=dtype,
        transfer_from=loaded,
    )
    # The rebuild runs after Transformers put the model in eval, so the fresh
    # modules must follow the model's mode or the CPU int4pack kernels latch
    # the dense-dequant training path on the first forward (#4907).
    model.train(model.training)
    log.info(
        "Transformers: rebuilt `%s` native GPTQModel AWQ modules after the "
        "bridge missed this load's import order (#4795).",
        len(loaded),
    )
    return HFGPTQModelLoadContext(quantize_config=qcfg, quant_linear=quant_linear)
