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
from ..quantization import FORMAT, QuantizeConfig
from ..quantization.config import resolve_quant_format
from .backend import BACKEND, backend_for_activation, normalize_backend
from .hf import get_hf_config_dtype
from .logger import setup_logger


log = setup_logger()

__all__ = [
    "HFGPTQModelLoadContext",
    "hf_gptqmodel_post_init_for_load",
    "hf_gptqmodel_prepare_model_for_load",
    "is_native_gptqmodel_checkpoint",
]

# Files a checkpoint directory is scanned for when resolving provenance.
_QUANTIZE_CONFIG_FILENAMES = ("quantize_config.json", "quant_config.json", "config.json")


@dataclass(frozen=True)
class HFGPTQModelLoadContext:
    """State shared between the pre- and post-weight-load hooks."""

    quantize_config: QuantizeConfig
    quant_linear: Type[BaseQuantLinear]


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

    if qcfg.rotation:
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

    # Serialized qweights are authoritative; dynamic rules may also name dense modules.
    tensor_keys = _checkpoint_tensor_keys(checkpoint_path, checkpoint_files)
    quantized_module_names = tuple(
        sorted(key.removesuffix(".qweight") for key in tensor_keys if key.endswith(".qweight"))
    )
    if not quantized_module_names:
        raise ValueError(
            "The GPTQModel checkpoint does not expose any `.qweight` tensors in its checkpoint manifest."
        )

    import defuser

    from ..models.auto import MODEL_MAP
    from ..models.base import BaseQModel
    from .importer import hf_normalize_device_device_map
    from .model import _module_has_meta_tensors, find_modules, get_module_by_name_prefix, make_quant

    model_type = str(getattr(model.config, "model_type", "")).lower()
    model_definition = MODEL_MAP.get(model_type, BaseQModel)
    defuser.replace_fused_blocks(model_type)

    # Rebuild fused model structures before matching serialized module names.
    def convert_fused_modules():
        defuser.convert_model(model, cleanup_original=True)
        for module_path in getattr(model_definition, "defuser_module_paths", ()) or ():
            module, _ = get_module_by_name_prefix(model, module_path)
            if module is not None:
                defuser.convert_model(module, cleanup_original=True)

    # Expand MoE experts on meta to avoid materializing dense expert weights.
    if _module_has_meta_tensors(model):
        with torch.device("meta"):
            convert_fused_modules()
    else:
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
    )
    log.info(
        "Transformers/Optimum: prepared `%s` native GPTQModel modules from the checkpoint manifest.",
        len(quantized_module_names),
    )
    return HFGPTQModelLoadContext(quantize_config=qcfg, quant_linear=quant_linear)


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

    # Rotated checkpoints fuse an orthogonal Q and an exact Hadamard H into the
    # weights; the H has to be re-applied to activations at inference. Without
    # this the rotated weights consume untransformed activations and produce
    # silently wrong outputs, so mirror the normal loader's post-init step.
    _setup_rotation_online_had(model, qcfg.rotation)

    # Paged Flash Attention is not CUDA-graph capture safe; the normal loader
    # opts paged models out of graph capture so ``generate_batch`` /
    # ``init_continuous_batching`` work without a caller-side monkey patch.
    _set_paged_attention_safe_cuda_graphs(model)
    return model
