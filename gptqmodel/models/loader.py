# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Marlin format: IST-DASLab contributors, MIT, https://github.com/IST-DASLab/marlin
# BitBLAS format: Microsoft Research contributors, Apache-2.0, https://github.com/microsoft/BitBLAS

from __future__ import annotations

import copy
import json
import os
import shutil
import threading
import time
import weakref
from functools import partial, wraps
from importlib.metadata import PackageNotFoundError, version
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import torch
import transformers

from ..utils.kernel import KernelConfig, configure_kernel
from ..utils.modelscope import ensure_modelscope_available
from ..utils.p2p import P2PConfig, configure_p2p
from ..utils.rocm import IS_ROCM, warn_unsupported_rocm
from ..utils.structure import LazyTurtle, print_module_tree


if ensure_modelscope_available():
    from modelscope import snapshot_download
else:
    from huggingface_hub import snapshot_download

import defuser
from packaging.version import InvalidVersion, Version
from transformers import AutoConfig, PretrainedConfig
from transformers.utils import is_flash_attn_2_available

from ..adapter.adapter import Adapter, normalize_adapter
from ..adapter.remote import configure_explicit_remote_adapter, resolve_checkpoint_adapter_path
from ..nn_modules.exllamav3 import ExllamaV3Linear
from ..nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from ..nn_modules.qlinear import BaseQuantLinear, GPTQQuantLinear
from ..nn_modules.qlinear.allspark import AllSparkLinear
from ..nn_modules.qlinear.allspark_awq import AllSparkAwqLinear
from ..nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from ..nn_modules.qlinear.fp8 import TorchFP8Linear
from ..nn_modules.qlinear.vptq_torch import VPTQLinear
from ..quantization import QuantizeConfig
from ..quantization.config import (
    FORMAT,
    METHOD,
    MIN_VERSION_WITH_V2,
    AutoModuleDecoderConfig,
    BaseQuantizeConfig,
    configure_dynamic_override_cache_for_model,
    dynamic_override_cache_stats,
    log_dynamic_override_cache_stats,
    resolve_model_rotation,
    resolve_quant_format,
)
from ..quantization.dtype import device_supports_dtype, device_supports_native_fp4, quark_floatx_formats
from ..utils.backend import BACKEND, PROFILE, backend_for_activation, normalize_backend, normalize_profile
from ..utils.backend_config import Backend, RuntimeConfig, resolve_backend_config, resolve_quantize_backend
from ..utils.exllamav3 import replace_exllamav3_placeholders
from ..utils.hf import (
    INTERNAL_HF_GGUF_FILE_KWARG,
    assert_loaded_model_state,
    get_config_selection_kwargs,
    get_hf_config_dtype,
    get_hf_gguf_load_kwargs,
    has_native_transformers_causallm_support,
    load_hf_tokenizer,
    normalize_hf_config_compat,
    normalize_model_id_or_path_for_hf_gguf,
    normalize_torch_dtype_kwarg,
    prepare_remote_model_init_compat,
    resolve_checkpoint_subfolder,
    resolve_trust_remote_code,
    set_hf_config_dtype,
    suspend_hf_weight_init,
)
from ..utils.importer import (
    _selector_devices,
    auto_select_device,
    expand_selector_device_family,
    get_kernel_for_backend,
    normalize_device_device_map,
    select_quant_linear,
    selector_device_family,
)
from ..utils.inspect import safe_kwargs_call
from ..utils.logger import setup_logger
from ..utils.model import (
    apply_no_placement_to_device_map,
    auto_dtype,
    convert_gptq_v1_to_v2_format,
    find_config_seq_len,
    find_modules,
    get_checkpoints,
    get_layers_with_prefixes,
    get_module_by_name_prefix,
    get_module_name,
    gptqmodel_post_init,
    is_embeddings_module_quantized,
    load_checkpoint_in_model_then_tie_weights,
    make_quant,
    no_placement_module_names,
    simple_dispatch_model,
    validate_checkpoint_qweights,
)
from ..utils.torch import xla_device_count
from ._const import DEVICE, EXPERT_INDEX_PLACEHOLDER, HAS_NPU, normalize_device


if TYPE_CHECKING:
    from ..utils import internal_gguf


def __getattr__(name: str):
    """Resolve GGUF-only names on first use so the core never imports NumPy."""
    if name == "internal_gguf":
        from ..utils import internal_gguf as value
    elif name == "GGUFTorchLinear":
        from ..nn_modules.qlinear.gguf import GGUFTorchLinear as value
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


log = setup_logger()


def _checkpoint_key_converter(cls, model, *, to_runtime: bool):
    """Return the key converter the quantized loader should apply.

    A definition may declare an explicit ``checkpoint_key_for_load`` /
    ``checkpoint_key_for_save``. Every other family falls back to its
    Transformers checkpoint conversion map, inferred from the live model, so a
    legacy-layout checkpoint (Transformers 4.x keys such as ``model.layers.*``
    and ``visual.*``) is remapped to the runtime names. The map is idempotent
    for checkpoints already written with runtime keys: their names do not match
    the checkpoint-side patterns and pass through unchanged.
    """

    explicit = getattr(
        cls,
        "checkpoint_key_for_load" if to_runtime else "checkpoint_key_for_save",
        None,
    )
    if callable(explicit):
        return explicit

    try:
        mapping = cls.resolve_hf_conversion_map_reversed(target_model=model)
    except Exception:
        log.debug("Loader: could not infer an HF checkpoint conversion map", exc_info=True)
        return None
    if mapping is None:
        return None

    def convert(name: str) -> str:
        return LazyTurtle.convert_hf_checkpoint_key(name, mapping, to_runtime=to_runtime)

    return convert


def _native_quantized_loader_enabled(policy: Optional[KernelConfig] = None) -> bool:
    """Resolve the GGUF parser selection from the load policy.

    ``policy`` is the ``KernelConfig`` the public call resolved; ``None`` uses the
    policy installed for the current load. Nothing here reads the environment:
    the same call options must select the same parser in every process.
    """
    from ..utils.kernel import resolve_kernel_config

    return bool(resolve_kernel_config(policy).gguf_native_quantized_loader)


def _ggml_quantization_type():
    """Import the GGUF enum only when a GGUF checkpoint is actually present."""
    from ..utils.internal_gguf import GGMLQuantizationType

    return GGMLQuantizationType


ATTN_IMPLEMENTATION = "attn_implementation"

_EXTERNAL_BACKEND_FORMATS = {
    BACKEND.VLLM: {
        FORMAT.GPTQ,
        FORMAT.GEMM,
    },
    BACKEND.SGLANG: {
        FORMAT.GPTQ,
        FORMAT.GPTQ_V2,
        FORMAT.GEMM,
        FORMAT.MARLIN,
    },
}


def _auto_select_mlx_backend(backend, device, config, qcfg, quant_method, format_code, adapter):
    """Select MLX when its Metal runtime can use the checkpoint directly."""
    if backend != BACKEND.AUTO or adapter is not None or qcfg.dynamic or resolve_model_rotation(qcfg):
        return backend
    if quant_method == METHOD.GGUF and getattr(getattr(qcfg, "runtime_bits", None), "name", None) not in {
            "q1_0", "q1_0_g128", "q2_0", "q4_0", "q8_0", "q4_k", "q4_k_s", "q4_k_m",
            "q5_k", "q5_k_s", "q5_k_m", "q6_k"}:
        return backend
    if quant_method == METHOD.BITSANDBYTES:
        # CPU bitsandbytes is faster for token decode on this host; explicit
        # backend=MLX remains available for larger batches.
        return backend
    if selector_device_family(device) != DEVICE.MPS:
        return backend
    if quant_method == METHOD.EXL3:
        # The existing EXL3 Torch cache is faster for token decode on M5 Ultra.
        return backend
    try:
        from mlx_lm.utils import _get_classes

        select_quant_linear(
            bits=qcfg.bits,
            group_size=qcfg.group_size,
            desc_act=qcfg.desc_act,
            sym=qcfg.sym,
            backend=BACKEND.MLX,
            format=format_code,
            quant_method=quant_method,
            device=DEVICE.MPS,
            pack_dtype=qcfg.pack_dtype,
        )
        _get_classes(config.to_dict())
    except (ImportError, ValueError, NotImplementedError):
        return backend
    log.info("Loader: selected MLX Metal for Apple Silicon inference")
    return BACKEND.MLX


def native_floatx_source_format(
    config: PretrainedConfig,
    *,
    model_local_path: Optional[str] = None,
) -> Optional[str]:
    """Return the native floatx weight format declared by a source checkpoint.

    NVIDIA ModelOpt checkpoints have used both ``config.json`` and the separate
    ``hf_quant_config.json`` sidecar for this metadata.  The latter is needed
    for older FP8 exports whose model config has no ``quantization_config``.
    """

    from ..utils.mimo import is_mimo_mixed_source

    if is_mimo_mixed_source(config):
        return "mimo_mixed"

    payloads = []
    quantization_config = getattr(config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        payloads.append(quantization_config)

    if model_local_path:
        sidecar_path = os.path.join(model_local_path, "hf_quant_config.json")
        try:
            with open(sidecar_path, "r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
            if isinstance(sidecar, dict):
                payloads.append(sidecar.get("quantization", sidecar))
        except (OSError, ValueError, TypeError) as exc:
            log.debug(
                "Loader: native floatx sidecar `%s` is optional; continuing without it: %s",
                sidecar_path,
                exc,
            )

    algorithms = []
    for payload in payloads:
        # Hugging Face's Qwen FP8 exports declare ``quant_method: fp8`` while
        # ModelOpt uses ``quant_algo``.  Both are native floatx source
        # checkpoints when a caller supplies a fresh GPTQ/AWQ config.
        for key in ("quant_algo", "quant_format", "format", "quant_method"):
            value = payload.get(key)
            if isinstance(value, str):
                algorithms.append(value.lower())

        groups = payload.get("config_groups")
        if isinstance(groups, dict):
            algorithms.extend(
                str(group.get("format", "")).lower()
                for group in groups.values()
                if isinstance(group, dict)
            )
        algorithms.extend(quark_floatx_formats(payload))

    if any("mxfp4" in algorithm for algorithm in algorithms) and any(
        "mxfp8" in algorithm for algorithm in algorithms
    ):
        return "mixed-mx"
    if any("mxfp4" in algorithm for algorithm in algorithms):
        return "mxfp4"
    if any("mxfp8" in algorithm for algorithm in algorithms):
        return "mxfp8"
    if any("nvfp8" in algorithm for algorithm in algorithms):
        return "nvfp8"
    if any("fp4" in algorithm for algorithm in algorithms):
        return "nvfp4"
    if any("fp8" in algorithm for algorithm in algorithms):
        return "fp8"
    return None


def configure_native_floatx_source_quantization(
    config: PretrainedConfig,
    quantize_config: BaseQuantizeConfig,
    *,
    device: Union[DEVICE, torch.device, str],
    model_local_path: Optional[str] = None,
    model_dtype: torch.dtype = torch.bfloat16,
) -> Optional[str]:
    """Configure floatx checkpoint sources for module-local W4A16 quantization.

    The shell model remains on ``meta`` while the looper asks the lazy
    checkpoint reader for one linear module at a time.  That module is decoded
    into FP16/BF16 only for the duration of its GPTQ/AWQ pass, rather than expanding
    every source weight to FP16/BF16 at model load.
    """

    source_format = native_floatx_source_format(config, model_local_path=model_local_path)
    if source_format is None or quantize_config.method not in (METHOD.GPTQ, METHOD.AWQ):
        return None

    preprocessors = list(getattr(quantize_config, "preprocessors", None) or [])
    decode_dtype = torch.bfloat16
    if source_format == "mimo_mixed":
        if quantize_config.method == METHOD.AWQ:
            raise ValueError("MiMo mixed-source AWQ quantization is not supported; use GPTQ.")
        if model_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "MiMo mixed-source GPTQ quantization requires dtype=torch.float16 or dtype=torch.bfloat16; "
                f"got {model_dtype}. FP32 is supported for direct/offline source decoding only."
            )
        decode_dtype = model_dtype
        for preprocessor in preprocessors:
            if isinstance(preprocessor, AutoModuleDecoderConfig) and preprocessor.target_dtype != model_dtype:
                raise ValueError(
                    "MiMo mixed-source AutoModuleDecoderConfig.target_dtype "
                    f"({preprocessor.target_dtype}) must match the model dtype ({model_dtype})."
                )

    decode_dtype_name = "BF16" if decode_dtype == torch.bfloat16 else "FP16"
    target_device = device.to_torch_device() if isinstance(device, DEVICE) else torch.device(device)
    if not device_supports_dtype(target_device, decode_dtype, require_validation=True):
        raise EnvironmentError(
            f"Floatx source quantization requires validated {decode_dtype_name} linear support on the "
            f"quantization device. Device `{target_device}` does not provide it."
        )

    # GPTQ's Hessian is part of the quantization mathematics, not a property
    # of the packed source weight.  Preserve the normal FP32 collection path
    # even when the source forward view is floatx.
    hessian = getattr(quantize_config, "hessian", None)
    if hessian is not None and getattr(hessian, "staging_dtype", torch.float32) != torch.float32:
        log.warning(
            "Loader: native %s source requires FP32 Hessian collection; overriding hessian.staging_dtype=%s.",
            source_format.upper(),
            getattr(hessian, "staging_dtype", None),
        )
        hessian.staging_dtype = torch.float32

    native_dtype_supported = False
    native_dtype_name = None
    if source_format == "fp8":
        # NVIDIA ModelOpt FP8 checkpoints use E4M3 for weights.  The exact
        # tensor dtype is checked again per module before the wrapper is built.
        native_dtype = getattr(torch, "float8_e4m3fn", None)
        native_dtype_name = "float8_e4m3fn"
        native_dtype_supported = bool(
            native_dtype is not None
            and device_supports_dtype(target_device, native_dtype, require_validation=True)
        )
    elif source_format == "nvfp4":
        native_dtype_name = "float4_e2m1fn_x2"
        native_dtype_supported = device_supports_native_fp4(
            target_device,
            require_validation=True,
        )
    else:
        # MX block scales need a format-specific GEMM kernel.  The auto
        # decoder can still feed quantization from dense module-local FP16/BF16.
        # MiMo's mixed packing and TP-local QKV scales require dense decoding.
        native_dtype_name = source_format

    # This is deliberately transient runtime state: it is not part of the
    # exported W4 config.  A per-module check still guards mixed/odd shards,
    # but the expensive capability probe happens before model construction.
    quantize_config._native_floatx_forward_plan = {
        "source_format": source_format,
        "device": str(target_device),
        "bf16_validated": decode_dtype == torch.bfloat16,
        "native_dtype": native_dtype_name,
        "native_validated": native_dtype_supported,
        "mode": "native" if native_dtype_supported else "decode",
        "hessian_accumulation_dtype": "float32",
    }

    if native_dtype_supported:
        log.info(
            "Loader: native %s Hessian forward is validated on %s; BF16 decode will be deferred until %s optimization.",
            source_format.upper(),
            target_device,
            quantize_config.method.value.upper(),
        )
    else:
        log.warning(
            "Loader: native %s forward is unavailable on %s; using decoded %s for Hessian collection. "
            "Quantization remains supported, but calibration may be slower.",
            source_format.upper(),
            target_device,
            decode_dtype_name,
        )

    if not any(isinstance(preprocessor, AutoModuleDecoderConfig) for preprocessor in preprocessors):
        preprocessors.append(AutoModuleDecoderConfig(target_dtype=decode_dtype))
        quantize_config.preprocessors = preprocessors

    # Direct HF loading coerces packed floatx weights into the shell dtype and
    # loses their scale metadata. LazyTurtle preserves the original storage and
    # is what makes module-local decoding possible.
    if not quantize_config.offload_to_disk:
        log.info(
            "Loader: enabling checkpoint-backed module decoding for native %s source weights.",
            source_format.upper(),
        )
        quantize_config.offload_to_disk = True

    log.info(
        "Loader: detected native %s source; automatically enabled %s auto_module_decoder for %s.",
        source_format.upper(),
        decode_dtype_name,
        quantize_config.method.value.upper(),
    )
    return source_format


def _validate_external_backend_format(backend: BACKEND, format_code: FORMAT) -> None:
    supported_formats = _EXTERNAL_BACKEND_FORMATS.get(backend)
    if supported_formats is None or format_code in supported_formats:
        return
    supported = ", ".join(f"FORMAT.{item.name}" for item in sorted(supported_formats, key=lambda item: item.name))
    raise ValueError(f"{backend} backend only supports {supported}: actual = {format_code}")


def _explicit_requested_devices(device) -> Optional[List[torch.device]]:
    """Return the concrete accelerators when the caller named specific devices.

    `device=("cuda:1", "cuda:2")` is a request for exactly those two GPUs.
    Before this helper both the layer-wise placement map and the quantization
    device pool ignored the sequence and expanded to *every* visible
    accelerator, so asking for two GPUs silently used all four. An abstract
    family selector or a bare `cuda` keeps the historical family-wide path.
    A sequence of one or more concrete ordinals requests exactly those devices. Integer
    ordinals use the active accelerator namespace. Mixing a concrete ordinal
    with an abstract selector is ambiguous and must not expand the request.
    """
    if not isinstance(device, (tuple, list)):
        return None

    targets = _selector_devices(device)
    concrete = [
        target for target in targets
        if isinstance(target, torch.device) and target.type != "cpu" and target.index is not None
    ]
    if not concrete:
        return None
    if len(concrete) != len(targets):
        raise ValueError(
            "A device sequence cannot mix concrete accelerator ordinals with abstract selectors; "
            "use only concrete ordinals or one accelerator family."
        )
    selector_device_family(targets)  # validate every selected family/capability
    return concrete


def _explicit_layerwise_devices(device) -> Optional[List[str]]:
    """String form of :func:`_explicit_requested_devices` for map rendering."""
    requested = _explicit_requested_devices(device)
    return [str(target) for target in requested] if requested is not None else None


def _layerwise_device_count(device, backend: Optional[BACKEND] = None) -> int:
    """Return the visible device count for layer-wise checkpoint placement.

    The XLA family needs an explicit opt-out for the native TTNN backend.
    ``xla_device_count()`` initializes the PJRT plugin, and with
    ``pjrt-plugin-tt`` that opens and claims every Tenstorrent chip. The native
    TTNN backend owns those chips through TT-Metal instead, and TT-Metal allows
    exactly one owner per chip, so probing PJRT here poisoned the very next
    ``ttnn.open_device`` in ``post_init`` with a TT_FATAL ownership assert.
    Reading the PCIe device count names the same chips without opening one.
    """
    explicit = _explicit_layerwise_devices(device)
    if explicit is not None:
        return len(explicit)

    family = selector_device_family(device) or DEVICE.CPU
    if family in (DEVICE.CUDA, DEVICE.ROCM):
        return torch.cuda.device_count()
    if family == DEVICE.XPU:
        return torch.xpu.device_count()
    if family == DEVICE.NPU:
        return torch.npu.device_count()
    if family == DEVICE.XLA:
        if _native_ttnn_backend(backend):
            from ..utils.ttnn_runtime import ttnn_device_count

            return ttnn_device_count()
        return xla_device_count()
    return 1


def _layerwise_device_strings(
    device, num_gpus: int, backend: Optional[BACKEND] = None
) -> list[str]:
    """Render layer-wise placement targets using the selected accelerator.

    Like :func:`_layerwise_device_count`, this must not initialize PJRT for the
    native TTNN backend: ``xla_device_count()`` opens and claims every chip, and
    the checkpoint would then be dispatched onto ``xla:N`` through a runtime
    that cannot coexist with the TT-Metal mesh ``post_init`` opens.
    """
    explicit = _explicit_layerwise_devices(device)
    if explicit is not None:
        return explicit[:num_gpus]

    family = selector_device_family(device) or DEVICE.CPU
    if family in (DEVICE.CUDA, DEVICE.ROCM):
        if torch.cuda.is_available():
            return [f"cuda:{index}" for index in range(num_gpus)]
        raise RuntimeError("CUDA is not available")
    if family == DEVICE.XPU:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return [f"xpu:{index}" for index in range(num_gpus)]
        raise RuntimeError("XPU is not available")
    if family == DEVICE.NPU:
        if HAS_NPU:
            return [f"npu:{index}" for index in range(num_gpus)]
        raise RuntimeError("NPU is not available")
    if family == DEVICE.XLA:
        if _native_ttnn_backend(backend):
            from ..utils.ttnn_runtime import ttnn_device_count

            if ttnn_device_count() >= num_gpus:
                return [f"xla:{index}" for index in range(num_gpus)]
            raise RuntimeError("no Tenstorrent devices are visible to TTNN")
        if xla_device_count() >= num_gpus:
            return [f"xla:{index}" for index in range(num_gpus)]
        raise RuntimeError("XLA is not available")
    return ["cpu"] * num_gpus


def _native_ttnn_backend(backend: BACKEND) -> bool:
    """Return whether ``backend`` owns its chips without the Torch/XLA runtime."""
    return backend in (BACKEND.GPTQ_TTNN,)


def _module_tensor_bytes(module: torch.nn.Module) -> int:
    """Return the resident bytes of a module's own parameters and buffers.

    Meta tensors keep their shape and element size, so the estimate works
    before the checkpoint is materialized. A tied weight is one tensor object
    and is counted once.
    """
    seen = set()
    total = 0
    for tensor in chain(module.parameters(), module.buffers()):
        if not isinstance(tensor, torch.Tensor):
            continue
        key = id(tensor)
        if key in seen:
            continue
        seen.add(key)
        total += tensor.numel() * tensor.element_size()
    return total


def _default_accelerator_family() -> DEVICE:
    """Return the accelerator an integer ``max_memory`` key refers to.

    Accelerate treats integer keys as accelerator ordinals. When the caller
    passed no ``device`` the family has to come from the runtime instead of
    defaulting to CPU, otherwise ``{1: "20GiB"}`` would name ``cpu:1`` and the
    budget would look accelerator-free (#4632).
    """

    if torch.cuda.is_available():
        return DEVICE.ROCM if IS_ROCM else DEVICE.CUDA
    for candidate in (DEVICE.XPU, DEVICE.NPU):
        runtime = getattr(torch, candidate.type, None)
        is_available = getattr(runtime, "is_available", None)
        if is_available is not None:
            try:
                if is_available():
                    return candidate
            except Exception:
                continue
    return DEVICE.CPU


def _normalize_max_memory(max_memory, device) -> List[tuple[str, int]]:
    """Normalize an Accelerate-style ``max_memory`` map into ordered budgets.

    Keys are integer accelerator ordinals, ``"cpu"``, or concrete device
    strings; values are byte counts or size strings such as ``"4GiB"``. The
    returned order is the caller's insertion order and is also the placement
    priority.
    """
    if not isinstance(max_memory, dict) or not max_memory:
        raise TypeError("`max_memory` must be a non-empty dict of device -> size.")

    from accelerate.utils import convert_file_size_to_int

    family = selector_device_family(device) or _default_accelerator_family()
    budgets: List[tuple[str, int]] = []
    seen = set()
    for key, value in max_memory.items():
        if isinstance(key, int):
            # An integer key is an accelerator ordinal. Without an explicit
            # `device` the family must come from the active accelerator, not
            # from CPU: `max_memory={1: "20GiB"}` names `cuda:1` on a CUDA host
            # (#4632).
            if family == DEVICE.CPU:
                raise ValueError(
                    f"`max_memory` names the accelerator ordinal `{key}` but no "
                    "accelerator is available; use the `\"cpu\"` key for a CPU-only "
                    "budget."
                )
            name = f"{family.type}:{key}"
        else:
            name = str(key).strip().lower()
            if name != "cpu":
                try:
                    name = str(torch.device(name))
                except (RuntimeError, ValueError) as exc:
                    raise ValueError(
                        f"`max_memory` has an unsupported device key `{key}`."
                    ) from exc
        if name in seen:
            raise ValueError(f"`max_memory` names `{name}` more than once.")
        seen.add(name)
        try:
            budget = convert_file_size_to_int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"`max_memory[{key!r}]` is not a valid size: {value!r}."
            ) from exc
        if budget <= 0:
            raise ValueError(f"`max_memory[{key!r}]` must be greater than zero.")
        budgets.append((name, budget))
    return budgets


def _device_map_uses_cpu(device_map) -> bool:
    """Whether a placement map contains a CPU-resident entry."""
    if not isinstance(device_map, dict):
        return False
    for value in device_map.values():
        if value is None or isinstance(value, int):
            continue
        try:
            if torch.device(value).type == "cpu":
                return True
        except (RuntimeError, ValueError):
            continue
    return False


def _assign_greedy_device_ids(
    sizes: Sequence[int],
    budgets: List[Optional[int]],
    labels: Optional[Sequence[str]] = None,
) -> List[int]:
    """Place each unit on the first device that still has room, in order.

    ``budgets`` is mutated so the caller can keep reserving room for the
    embedding and output head. A unit that does not fit anywhere is an error:
    silently exceeding a memory cap is what the caller asked us not to do.
    """
    cursor = 0
    assignments: List[int] = []
    for index, size in enumerate(sizes):
        while cursor < len(budgets) and budgets[cursor] is not None and budgets[cursor] < size:
            cursor += 1
        if cursor >= len(budgets):
            label = labels[index] if labels is not None else f"unit {index}"
            raise ValueError(
                f"`max_memory` cannot hold `{label}` ({size} bytes); increase a "
                "device budget or add a `cpu` entry."
            )
        assignments.append(cursor)
        if budgets[cursor] is not None:
            budgets[cursor] -= size
    return assignments


def _resolve_default_layerwise_device(device, backend: BACKEND):
    """Keep explicit device sequences intact instead of collapsing singletons."""
    requested = _explicit_requested_devices(device)
    if requested is not None:
        return tuple(requested)
    normalized = normalize_device_device_map(device, None)
    selected = auto_select_device(normalized, backend)
    if _native_ttnn_backend(backend):
        # ``expand_selector_device_family`` asks PJRT for an addressable device
        # count, which initializes the Tenstorrent PJRT plugin and claims every
        # chip. The native TTNN backend opens its own mesh and never consumes an
        # ``xla:N`` ordinal, so keep the abstract family unexpanded here.
        return selected
    # The default placement stays inside one compute-capability group so a host
    # that mixes GPU generations still loads with no placement arguments (#4705).
    resolved = expand_selector_device_family(selected, single_capability_group=True)
    # Keep the resolved group explicit even when it holds one card. A scalar
    # reads as an abstract family downstream: ``_layerwise_device_count`` then
    # falls back to ``torch.cuda.device_count()`` and the map builder places
    # layers on every visible GPU, including the generation the group excluded
    # (#4734). The multi-device case already returns a tuple, so a one-card
    # group must too, or the mixed-capability guard is bypassed instead of
    # honored.
    if isinstance(resolved, torch.device) and resolved.index is not None and resolved.type != "cpu":
        return (resolved,)
    return resolved


_ACCELERATE_DEVICE_MAP_STRINGS = frozenset({"auto", "balanced", "sequential"})


def _is_accelerate_device_map_string(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in _ACCELERATE_DEVICE_MAP_STRINGS or lowered.startswith("balanced_low_")


def _validate_device_and_device_map_agree(device, device_map) -> None:
    """Reject a `device` request that contradicts an explicit `device_map`."""

    requested_family = selector_device_family(normalize_device_device_map(device, None))
    mapped_family = selector_device_family(normalize_device_device_map(None, device_map))
    if requested_family != mapped_family:
        requested_name = requested_family.value if requested_family is not None else None
        mapped_name = mapped_family.value if mapped_family is not None else None
        raise ValueError(
            f"`device={device!r}` conflicts with `device_map={device_map!r}`: "
            f"`device` selects {requested_name} but `device_map` selects {mapped_name}. "
            "Pass one placement argument, or make them agree."
        )


def _normalize_requested_device_map(device, device_map):
    """Apply Hugging Face string shorthand and check device/device_map agreement.

    Hugging Face accepts `device_map="cpu"` and `device_map="cuda:0"` as
    shorthand for `{"": value}`. GPT-QModel used to reject those strings and
    then let an explicit `device_map` silently replace `device`.
    """

    if device_map is None:
        return None
    if isinstance(device_map, str):
        if _is_accelerate_device_map_string(device_map):
            normalized = device_map
        else:
            try:
                torch.device(device_map)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    "If passing a string for `device_map`, choose 'auto', 'balanced', "
                    "'balanced_low_0', 'sequential', or a concrete device such as 'cpu' "
                    "or 'cuda:0'."
                ) from exc
            if device_map.strip().lower() in {"disk", "meta"}:
                raise ValueError(
                    "A full-model `device_map` cannot be an offload target; use a "
                    "per-module map for 'disk' or 'meta' entries."
                )
            normalized = {"": device_map}
    else:
        normalized = dict(device_map)
    if device is not None:
        _validate_device_and_device_map_agree(device, normalized)
    return normalized


def _indexed_device_request(device) -> Optional[str]:
    """Return an indexed ``device`` request as a placement string, if any.

    A bare integer is an accelerator ordinal, matching the documented
    ``device=(0, 1)`` sequence and ``device_map={"": N}`` semantics. Resolve
    it to the active accelerator's concrete ``<family>:N`` target here so it
    becomes a full-model map instead of a family selector that the layer-wise
    resolver would spread across every visible GPU (#4384). A host without an
    active accelerator cannot interpret the ordinal, so the selector
    conversion raises rather than silently treating it as CPU.
    """

    if isinstance(device, int) and not isinstance(device, bool):
        if device < 0:
            raise ValueError(f"device={device} is not an accelerator ordinal; ordinals start at 0.")
        targets = _selector_devices(device)
        assert targets is not None
        target = targets[0]
        # Fail here with the visible range rather than later with the
        # runtime's "invalid device ordinal".
        runtime = getattr(torch, target.type, None)
        count = runtime.device_count() if hasattr(runtime, "device_count") else None
        if count is not None and device >= count:
            raise ValueError(
                f"device={device} is not a visible {target.type} ordinal; this process sees "
                f"{count} {target.type} device(s)."
            )
        return str(target)

    if isinstance(device, str) and ":" in device:
        return device
    if isinstance(device, torch.device) and device.index is not None:
        return str(device)
    return None


def _normalize_explicit_placement(device, device_map, max_memory):
    """Resolve ``device``/``device_map``/``max_memory`` into placement inputs.

    An indexed ``device`` such as ``"cuda:0"`` is shorthand for a full-model
    map, but only when the caller did not also supply a memory budget. With
    ``max_memory`` the indexed device selects the accelerator family for the
    budgeted layer-wise map; converting it to ``{"": device}`` first would
    silently drop the cap.
    """

    requested_device_map = _normalize_requested_device_map(device, device_map)
    explicit_device_map = requested_device_map if isinstance(requested_device_map, dict) else None
    if max_memory is not None and explicit_device_map is not None:
        raise ValueError(
            "Pass either an explicit `device_map` dict or `max_memory`, not both."
        )
    if requested_device_map is None and max_memory is None:
        explicit_device = _indexed_device_request(device)
        if explicit_device is not None:
            explicit_device_map = {"": explicit_device}
            requested_device_map = explicit_device_map
    return requested_device_map, explicit_device_map


def _max_memory_device_name_is_cpu(name: str) -> bool:
    """Whether a normalized ``max_memory`` key is the CPU budget."""

    if name == "cpu":
        return True
    try:
        return torch.device(name).type == "cpu"
    except (RuntimeError, ValueError):
        return False


def _resolve_max_memory_device(max_memory, device) -> Optional[Union[DEVICE, tuple]]:
    """Resolve the devices a ``max_memory`` budget may place weights on.

    ``max_memory`` is the authoritative list of devices that may hold weights
    (#4082), so kernel selection must see exactly the accelerator devices the
    budget names. Expanding the CUDA family to every visible GPU made the
    mixed-capability check reject budgets that never touched the other card: on
    a 4090 + 5090 host even ``max_memory={0: "20GiB"}`` failed with
    ``cuda:0=12.0, cuda:1=8.9`` (#4632).

    Returns ``DEVICE.CPU`` for a CPU-only budget, the tuple of concrete
    accelerator devices for an accelerator budget, and ``None`` without a
    budget. An explicit ``device`` that contradicts the budget is rejected
    rather than silently ignored.
    """

    if max_memory is None:
        return None
    budgets = _normalize_max_memory(max_memory, device)
    accelerators: List[torch.device] = []
    for name, _ in budgets:
        if _max_memory_device_name_is_cpu(name):
            continue
        try:
            target = torch.device(name)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"`max_memory` has an unsupported device key `{name}`."
            ) from exc
        if target not in accelerators:
            accelerators.append(target)

    requested_family = selector_device_family(device) if device is not None else None
    if not accelerators:
        if requested_family is not None and requested_family != DEVICE.CPU:
            raise ValueError(
                "`max_memory` has no accelerator entry, so it can only place layers "
                f"on CPU; that contradicts `device={device!r}`. Add an accelerator "
                "entry to `max_memory`, or omit `device`/pass `device='cpu'` for "
                "CPU-only placement."
            )
        return DEVICE.CPU

    if requested_family is not None and requested_family != DEVICE.CPU:
        budget_family = selector_device_family(tuple(accelerators))
        if requested_family != budget_family:
            raise ValueError(
                f"`device={device!r}` selects {requested_family.value} but `max_memory` "
                f"funds {budget_family.value} devices ({', '.join(str(d) for d in accelerators)}). "
                "Pass one placement argument, or make them agree."
            )
        explicit = _explicit_requested_devices(
            device if isinstance(device, (tuple, list)) else (device,)
        )
        if explicit is not None:
            missing = [target for target in explicit if target not in accelerators]
            if missing:
                raise ValueError(
                    f"`device={device!r}` names {', '.join(str(d) for d in missing)} but "
                    "`max_memory` does not fund it. Add that device to `max_memory`, or "
                    "drop it from `device`."
                )
    return tuple(accelerators)


def _external_runtime_device_kwargs(
    device,
    requested_device_map: Optional[Union[str, Dict[str, Union[str, int]]]],
) -> Dict[str, Union[str, int]]:
    device_family = selector_device_family(device) or DEVICE.CPU
    runtime_kwargs: Dict[str, Union[str, int]] = {"device": device_family.type}
    if not requested_device_map or device_family not in {DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU, DEVICE.NPU, DEVICE.XLA}:
        return runtime_kwargs

    targets = (
        {requested_device_map}
        if isinstance(requested_device_map, str)
        else set(requested_device_map.values())
    )
    if len(targets) != 1:
        return runtime_kwargs
    target = targets.pop()

    if isinstance(target, int):
        runtime_kwargs["base_gpu_id"] = target
        return runtime_kwargs
    if not isinstance(target, str) or target in {"cpu", "disk", "meta"}:
        return runtime_kwargs

    try:
        explicit_device = torch.device(target)
    except (RuntimeError, ValueError):
        return runtime_kwargs
    if explicit_device.index is not None:
        runtime_kwargs["base_gpu_id"] = explicit_device.index
    return runtime_kwargs


def _build_sglang_runtime_kwargs(
    *,
    kwargs: Dict[str, object],
    device,
    requested_device_map: Optional[Union[str, Dict[str, Union[str, int]]]],
    resolved_dtype,
) -> Dict[str, object]:
    """Build SGLang kwargs from the already-resolved quantized-load options.

    ``dtype`` is a from_quantized argument rather than a runtime passthrough
    option. Put the final value into the kwargs mapping exactly once so the
    adapter cannot retain its historical hard-coded float16 default or receive
    duplicate dtype keywords.
    """
    runtime_kwargs = dict(kwargs)
    # The loaded HF config is passed explicitly below and is not an SGLang
    # ServerArgs field if a caller happened to include it in passthrough kwargs.
    runtime_kwargs.pop("config", None)
    for key, value in _external_runtime_device_kwargs(device, requested_device_map).items():
        runtime_kwargs.setdefault(key, value)
    runtime_kwargs["dtype"] = resolved_dtype
    return runtime_kwargs


def _normalize_sglang_load_dtype(dtype):
    """Resolve an explicit SGLang dtype string before the generic auto path."""
    if not isinstance(dtype, str) or dtype == "auto":
        return dtype

    name = dtype.removeprefix("torch.")
    normalized = getattr(torch, name, None)
    if not isinstance(normalized, torch.dtype):
        raise ValueError(f"Invalid SGLang dtype string: {dtype}")
    return normalized


def _should_print_module_tree() -> bool:
    """Keep expensive module-tree dumps opt-in during model loading."""

    raw = os.environ.get("GPTQMODEL_PRINT_MODULE_TREE")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _call_before_model_load(cls, *, model_local_path: str, load_quantized_model: bool) -> None:
    """Run a definition's pre-load hook for every load path.

    `before_model_load` is a plain method on the definition class, not an
    instance method: no model instance exists yet, so the class is passed as
    the receiver. Definitions that patch remote modules guard their work, so
    running the hook for quantize and quantized reload in the same process is
    idempotent. Native (non-quantized) loads need the same hook because the
    mask-helper and config patches it installs are also required by
    `generate()`/`forward()` on an unquantized model.
    """

    cls.before_model_load(
        cls,
        model_local_path=model_local_path,
        load_quantized_model=load_quantized_model,
    )


def _maybe_print_module_tree(model) -> None:
    """Print the module tree only when explicitly requested for debugging."""

    if _should_print_module_tree():
        print_module_tree(model=model)


# `requirements.txt` pins `defuser>=0.0.31`. Older releases do not split the
# fused 3-D expert tensors of afmoe / flint / granitemoe_swa, so their routed
# experts stay fused and are silently left unquantized; the only symptom is a
# "0 layer matches" line in the module-coverage report. Defuser 0.0.31 also
# fixes the GLM-4.x replacement MoE blocks, which crashed on the Transformers
# 5.22 router tuple.
DEFUSER_MIN_VERSION = "0.0.31"
_defuser_version_warning_emitted = False


def defuser_version_too_old(installed: Optional[str] = None) -> Optional[str]:
    """Return the installed `defuser` version when it predates the supported floor."""

    if installed is None:
        try:
            installed = version("defuser")
        except PackageNotFoundError:
            return None
    try:
        outdated = Version(installed) < Version(DEFUSER_MIN_VERSION)
    except InvalidVersion:
        return installed
    return installed if outdated else None


def _check_defuser_version() -> None:
    """Report an unsupported `defuser` before its conversion silently no-ops."""

    global _defuser_version_warning_emitted
    if _defuser_version_warning_emitted:
        return
    _defuser_version_warning_emitted = True
    outdated = defuser_version_too_old()
    if outdated is None:
        return
    log.error(
        "Loader: defuser %s is installed but >= %s is required. Fused 3-D expert "
        "tensors (afmoe, flint, granitemoe_swa, ...) will not be split and the "
        "GLM-4.x defused MoE blocks cannot run on Transformers 5.22; upgrade defuser.",
        outdated,
        DEFUSER_MIN_VERSION,
    )


def _convert_model_with_defuser(cls, model, cleanup_original: bool) -> bool:
    _check_defuser_version()
    converted = defuser.convert_model(model, cleanup_original=cleanup_original)

    # Defuser gates `convert_model()` on its own model registry. Model types
    # that GPT-QModel supports through the generic Transformers experts
    # interface (native Kimi-Linear / MiMo-V2-Flash) have no registry entry,
    # so run Defuser's generic replacement pass explicitly instead of leaving
    # the fused 3-D expert parameters unquantized.
    if not converted and getattr(cls, "defuser_generic_experts", False):
        from defuser.modeling.update_module import update_module

        update_module(model, cleanup_original=cleanup_original)
        converted = True

    defuser_module_paths = getattr(cls, "defuser_module_paths", ())
    if defuser_module_paths:
        for module_path in defuser_module_paths:
            module, _ = get_module_by_name_prefix(model, module_path)
            if module is None:
                log.warn("Loader: defuser module path `%s` was not found.", module_path)
                continue
            converted = defuser.convert_model(module, cleanup_original=cleanup_original) or converted

    after_conversion = getattr(cls, "after_defuser_conversion", None)
    # Model-specific finalizers may also normalize metadata that Defuser does
    # not own. Run them after the conversion attempt even when the installed
    # Defuser has no matching registration and reports no structural change.
    if callable(after_conversion):
        after_conversion(model)

    return converted


# Fused expert projections that Defuser splits from one 3-D tensor into a
# per-expert ``nn.Linear``. These names match
# ``defuser.modeling.moe_experts_interface.KNOWN_PROJECTION_PATTERNS``.
_FUSED_EXPERT_PROJECTION_NAMES = ("gate_up_proj", "gate_proj", "up_proj", "down_proj")


def _meta_fused_expert_params(model) -> int:
    """Park fused 3-D expert projections on meta before Defuser splits them.

    Defuser's tensor defusion copies the whole fused projection to the host and
    materializes one dense bf16 ``nn.Linear`` per expert. On the quantized-load
    path those linears are replaced by quantized kernels and filled from the
    checkpoint right afterwards, so the copies only inflate host memory: an
    OLMoE-1B-7B 4-bit reload peaked at 12.17 GiB RssAnon for a 3.66 GiB
    checkpoint (issue #4400).

    Defuser skips tensors that already live on meta, so move the fused
    projections (and their fused biases) there first; the checkpoint loader
    then materializes the packed weights straight from disk. Only call this
    from the quantized-load path: quantization needs the fused values in host
    memory so Defuser can produce real per-expert linears to quantize.

    Returns the number of parameters/buffers moved.
    """

    moved = 0
    for module in model.modules():
        for projection in _FUSED_EXPERT_PROJECTION_NAMES:
            weight = module._parameters.get(projection)
            if weight is None:
                weight = module._buffers.get(projection)
            if not isinstance(weight, torch.Tensor) or weight.dim() != 3:
                continue
            # The fused bias is a 2-D (num_experts, out_features) sibling.
            for attr in (projection, f"{projection}_bias"):
                parameter = module._parameters.get(attr)
                buffer = module._buffers.get(attr)
                tensor = parameter if parameter is not None else buffer
                if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
                    continue
                empty = torch.empty_like(tensor, device="meta")
                if parameter is not None:
                    module._parameters[attr] = torch.nn.Parameter(
                        empty, requires_grad=parameter.requires_grad
                    )
                else:
                    module._buffers[attr] = empty
                moved += 1
    return moved


def _materialize_meta_expert_linears(model) -> int:
    """Restore real CPU storage for the linears Defuser built on meta.

    Defuser materializes meta expert projections as meta ``nn.Linear`` shells
    (see :func:`_meta_fused_expert_params`). ``to_empty`` allocates their
    uninitialized storage without touching pages, so the host RSS stays flat;
    ``make_quant`` then replaces them with kernels whose own buffers are
    created on a real device. Without this step those kernels would inherit a
    meta shell, leaving non-checkpoint buffers (``qzeros_format``, ``input_perm``,
    ...) on meta and breaking the later ``model.to(device)`` dispatch.

    Only fully-meta modules are touched, so the real CPU weights built by
    ``from_config`` are never discarded.

    Returns the number of modules materialized.
    """

    moved = 0
    for module in model.modules():
        tensors = [*module._parameters.values(), *module._buffers.values()]
        tensors = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
        if not tensors or not any(tensor.device.type == "meta" for tensor in tensors):
            continue
        if any(tensor.device.type != "meta" for tensor in tensors):
            continue
        module.to_empty(device="cpu")
        moved += 1
    return moved


def _supports_flash_attn_2(config: PretrainedConfig) -> bool:
    """Detect whether the resolved HF architecture exposes FA2 kernels."""

    if not getattr(config, "architectures", None):
        return False

    model_class = getattr(transformers, config.architectures[0], None)
    if model_class is None:
        return False

    if hasattr(model_class, "_supports_flash_attn_2"):
        return bool(getattr(model_class, "_supports_flash_attn_2"))
    if hasattr(model_class, "_supports_flash_attn"):
        return bool(getattr(model_class, "_supports_flash_attn"))
    return False


def _iter_nested_pretrained_configs(config: PretrainedConfig):
    """Yield config and all nested PretrainedConfig nodes once."""

    stack = [config]
    visited = set()

    while stack:
        cur = stack.pop()
        if not isinstance(cur, PretrainedConfig):
            continue

        node_id = id(cur)
        if node_id in visited:
            continue
        visited.add(node_id)
        yield cur

        for value in vars(cur).values():
            if isinstance(value, PretrainedConfig):
                stack.append(value)
            elif isinstance(value, dict):
                for sub in value.values():
                    if isinstance(sub, PretrainedConfig):
                        stack.append(sub)
            elif isinstance(value, (list, tuple, set)):
                for sub in value:
                    if isinstance(sub, PretrainedConfig):
                        stack.append(sub)


def _override_attn_implementation(config: PretrainedConfig, attn_implementation: str) -> None:
    """Apply attention implementation override to root and nested configs."""

    for sub_config in _iter_nested_pretrained_configs(config):
        try:
            sub_config._attn_implementation = attn_implementation
        except Exception:
            # Some remote configs may expose read-only wrappers; ignore safely.
            pass


def _set_paged_attention_safe_cuda_graphs(model) -> None:
    """Disable CUDA graph capture for paged attention by default.

    ``transformers`` continuous batching defaults ``use_cuda_graph=True`` for flash-attention
    paths, but ``flash_attn_with_kvcache`` is not currently CUDA-graph capture-safe. Setting
    the model's default ``continuous_batching_config`` keeps paged Flash Attention working for
    direct ``generate_batch`` / ``init_continuous_batching`` callers without requiring a monkey
    patch.
    """
    if not hasattr(model, "config") or not isinstance(model.config, PretrainedConfig):
        return

    attn = getattr(model.config, "_attn_implementation", None) or getattr(
        model.config, "attn_implementation", None
    )
    if not isinstance(attn, str) or "paged" not in [part.strip() for part in attn.split("|")]:
        return

    try:
        from transformers import ContinuousBatchingConfig, GenerationConfig
    except Exception:
        return

    gen_config = getattr(model, "generation_config", None)
    if gen_config is None:
        try:
            gen_config = GenerationConfig.from_model_config(model.config)
        except Exception:
            return
        model.generation_config = gen_config

    cb_config = getattr(gen_config, "continuous_batching_config", None)
    if not isinstance(cb_config, ContinuousBatchingConfig):
        gen_config.continuous_batching_config = ContinuousBatchingConfig(use_cuda_graph=(False, False))
    elif getattr(cb_config, "use_cuda_graph", None) is None:
        cb_config.use_cuda_graph = (False, False)


def _setup_rotation_online_had(model, rotation: Optional[str], module_plan=None, layer_prefixes=None) -> None:
    """Attach online Hadamard transform state to QuantLinear modules for rotation inference.

    Rotation (QuaRot/SpinQuant/Hadamard) fuses an orthogonal ``Q`` and an exact
    Hadamard ``H`` into the weights. The ``H`` must be re-applied to activations
    at inference before the quantized matmul for ``mlp.down_proj``. The per-head
    Hadamard fused into ``v_proj``/``o_proj`` cancels inside the attention path,
    so ``self_attn.o_proj`` does not require an online transform. Only the
    ``rotation`` string is persisted; ``had_K``/``K`` are recomputed from the
    model dimensions at load time.
    """
    if not rotation:
        return

    from ..quantization.rotation.hadamard_utils import get_hadK

    if rotation not in {"hadamard", "random"}:
        raise ValueError(f"Unsupported rotation mode: `{rotation}`")

    online_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, BaseQuantLinear):
            continue

        relative = None
        if module_plan is not None:
            for prefix in layer_prefixes or ():
                if name.startswith(f"{prefix}."):
                    relative = name[len(prefix) + 1:]
                    break
            if relative is not None:
                _, separator, remainder = relative.partition(".")
                relative = remainder if separator and _.isdigit() else relative
        is_down_projection = (
            relative is not None and "down" in module_plan.roles(relative)
            if module_plan is not None
            else name.endswith("mlp.down_proj")
        )
        if is_down_projection:
            intermediate_size = module.in_features
            had_K, K = get_hadK(intermediate_size)
            module.online_full_had = True
            module.K = K
            if had_K is not None:
                module.set_had_K(had_K)
            online_count += 1

    if online_count == 0:
        log.warn(
            f"Rotation `{rotation}` requested but no `mlp.down_proj` QuantLinear "
            "modules were found; online Hadamard transform was not applied."
        )


def _is_accelerated_attention_device(device: object) -> bool:
    """Return True when the selected device can run CUDA/ROCm flash attention."""

    if isinstance(device, (tuple, list)):
        return bool(device) and _is_accelerated_attention_device(device[0])

    if isinstance(device, torch.device):
        return device.type in {"cuda", "hip"}
    if isinstance(device, DEVICE):
        return device in {DEVICE.CUDA, DEVICE.ROCM}
    if isinstance(device, str):
        return device in {"cuda", "rocm", "hip"}
    return False


def _resolve_native_gguf_profile(
    *,
    native_gguf_qspec: Optional["internal_gguf.GGUFQuantizedCheckpointSpec"],
    profile: PROFILE,
) -> PROFILE:
    """Resolve user profile intent for native GGUF checkpoints."""
    from ..utils import internal_gguf

    if (
        native_gguf_qspec is not None
        and native_gguf_qspec.tensor_qtype == internal_gguf.GGMLQuantizationType.Q1_0
        and profile == PROFILE.AUTO
    ):
        log.info("Loader: Q1_0 PROFILE.AUTO resolved to PROFILE.FAST.")
        return PROFILE.FAST
    return profile


def _should_use_dense_native_gguf_path(
    *,
    native_gguf_qspec: Optional["internal_gguf.GGUFQuantizedCheckpointSpec"],
    profile: PROFILE,
) -> bool:
    """Fast Bonsai mode stays on the dense HF GGUF import path."""
    from ..utils import internal_gguf

    return (
        native_gguf_qspec is not None
        and native_gguf_qspec.tensor_qtype == internal_gguf.GGMLQuantizationType.Q1_0
        and profile == PROFILE.FAST
    )


def parse_version_string(version_str: str):
    try:
        return Version(version_str)
    except InvalidVersion:
        raise ValueError(f"Invalid version format: {version_str}")


def parse_requirement(req):
    req = req.strip()
    for op in [">=", "<=", ">", "<", "=="]:
        if op in req:
            pkg, version_required = req.split(op, 1)
            return pkg.strip(), op, version_required.strip()
    if req:
        return req, None, None
    raise ValueError(f"Unsupported version constraint in: {req}")


def compare_versions(installed_version, required_version, operator):
    installed = parse_version_string(installed_version)
    required = parse_version_string(required_version)
    if operator == ">":
        return installed > required
    elif operator == ">=":
        return installed >= required
    elif operator == "<":
        return installed < required
    elif operator == "<=":
        return installed <= required
    elif operator == "==":
        return installed == required
    else:
        raise ValueError(f"Unsupported operator: {operator}")


# transformers' ``from_pretrained`` / ``from_config`` mutate process-wide state:
# they save, set and restore ``torch.set_default_dtype``
# (``transformers/modeling_utils.py``) and build parameters through meta-device
# and no-init contexts. Two loads in different threads interleave those changes
# and hand back models with mixed fp16/fp32 parameters, a meta ``lm_head`` and
# untied embeddings (issue #4129). Model construction is serialized process-wide
# through this lock; ``quantize()`` on already-loaded models stays concurrent.
_MODEL_LOAD_LOCK = threading.RLock()


def model_load_lock():
    """Return the process-wide, re-entrant lock guarding model construction.

    Every site that builds a model through transformers -- ``from_pretrained``,
    ``from_quantized``, the save-side shell rebuild, the dequant path and the
    TT block-float path -- must enter this same lock. Look the lock up through
    this accessor (not a module-level import of ``_MODEL_LOAD_LOCK``) so the
    single lock object is shared and can be spied on in tests.
    """

    return _MODEL_LOAD_LOCK


def _serialized_model_load(method):
    """Serialize a model-construction entry point across threads."""

    @wraps(method)
    def wrapper(*args, **kwargs):
        with model_load_lock():
            return method(*args, **kwargs)

    return wrapper


def _is_meta_shell_build_error(exc: Exception) -> bool:
    # Some trust_remote_code model constructors call int()/item() on tensors
    # during __init__, which breaks when the shell is built on the meta device.
    message = str(exc)
    return "cannot be called on meta tensors" in message and ".item()" in message


def _is_broken_transformers_dynamic_module_error(exc: Exception) -> bool:
    if not isinstance(exc, FileNotFoundError):
        return False
    missing_path = str(getattr(exc, "filename", "") or exc)
    return "transformers_modules" in missing_path and missing_path.endswith(".py")


def _hf_loader_from_pretrained_with_dynamic_module_retry(loader, model_local_path: str, **kwargs):
    try:
        return loader.from_pretrained(model_local_path, **kwargs)
    except Exception as exc:
        if not _is_broken_transformers_dynamic_module_error(exc):
            raise

        missing_path = str(getattr(exc, "filename", "") or "")
        missing_name = os.path.basename(missing_path)
        source_path = os.path.join(model_local_path, missing_name)
        if missing_path and os.path.isfile(source_path):
            os.makedirs(os.path.dirname(missing_path), exist_ok=True)
            shutil.copy2(source_path, missing_path)
            log.warn(
                "Loader: repaired missing dynamic-module file by copying `%s` -> `%s`.",
                source_path,
                missing_path,
            )

        retry_kwargs = dict(kwargs)
        retry_kwargs["force_download"] = True
        log.warn(
            "Loader: detected broken transformers dynamic-module cache while loading `%s`; "
            "retrying once with force_download=True: %s",
            model_local_path,
            exc,
        )
        return loader.from_pretrained(model_local_path, **retry_kwargs)


def _load_tt_blockfloat_model(model_cls, *, model_local_path, dtype, trust_remote_code=False):
    """Build a native block-float model and wrap it as a ``BaseQModel``.

    The checkpoint stores decoder weights as packed mantissa codes plus one
    exponent per block of 16 values, on the grid the Tenstorrent hardware
    unpacks natively. There is no GPTQ packing to bind, so the decoder linears
    become ``TTBlockfloatLinear`` modules that keep that storage, and the
    embedding, norms and head come from the checkpoint's own payloads.
    """

    from .tt_blockfloat_loader import (
        build_tt_blockfloat_model,
        quantize_config_from_checkpoint,
    )

    # ``dtype`` is passed through as-is: ``None``/the normaliser's "not
    # provided" sentinel means "the checkpoint's declared precision", which is
    # what the GPTQ/AWQ routes get from ``from_pretrained``. Forcing float32
    # here doubled the resident model for no numerical gain.
    model, restored, info = build_tt_blockfloat_model(model_local_path, dtype=dtype)

    model_config = model.config.to_dict()
    seq_len = find_config_seq_len(
        model_config,
        ["max_position_embeddings", "seq_length", "n_positions", "multimodal_max_length"],
    )
    model.seqlen = seq_len if seq_len is not None else 4096

    qcfg = quantize_config_from_checkpoint(model_local_path)
    quantization_config = info.get("quantization_config") or {}
    widths = quantization_config.get("bits_per_layer")
    width_label = (
        "/".join(str(w) for w in sorted(set(widths.values()))) + "-bit"
        if isinstance(widths, dict) and widths
        else f"{info['bits']}-bit"
    )
    log.info(
        f"Loader: tt-blockfloat checkpoint: {len(restored)} quantized modules "
        f"({width_label} codes, block {quantization_config.get('block', 16)}, "
        f"{quantization_config.get('embed_bits', '?')}-bit embedding)"
    )
    return model_cls(
        model=model,
        quantized=True,
        quantize_config=qcfg,
        load_quantized_model=True,
        trust_remote_code=trust_remote_code,
        model_local_path=model_local_path,
    )


def _reject_native_awq_gemv_on_rocm(export_quant_method, format_code, backend=None) -> None:
    """Refuse the CUDA-only AWQ GEMV/GEMV_FAST/LLM_AWQ layouts on ROCm.

    Those three layouts are only readable by the native CUDA AWQ extension
    (``gemv_forward``/``gemmv2_forward``/``gemv_fast_forward_decode``), which
    hipcc cannot build: the sources include ``cuda_fp16.h`` and CUDA-only inline
    asm, and the hipified warp-sync reduction rejects wave32.  Without this check
    an AUTO load fails later with a bare "No valid quant linear", so point at the
    layout and kernels that do run on ROCm.
    """

    if not IS_ROCM:
        return
    if export_quant_method != METHOD.AWQ:
        return
    if format_code not in (FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.LLM_AWQ):
        return
    # The portable Triton GEMV reads the GEMV layout without the extension, so an
    # explicit selection of it is a supported ROCm placement.
    if backend == BACKEND.AWQ_GEMV_TRITON:
        return

    from ..nn_modules.qlinear.gemv_awq import AWQ_GEMV_ROCM_UNSUPPORTED

    raise NotImplementedError(AWQ_GEMV_ROCM_UNSUPPORTED)


# The GEMV/GEMV_FAST/LLM_AWQ layouts are read by kernels that declare
# ``SUPPORTS_DTYPES = [torch.float16]``. AUTO has not resolved a concrete kernel
# at load time, but every kernel AUTO can select for these layouts is fp16-only,
# so the format alone decides the dtype.
_AWQ_FP16_ONLY_FORMATS = (FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.LLM_AWQ)


def _coerce_quantized_awq_dtype(
    *, backend: BACKEND, qcfg: QuantizeConfig, dtype, explicit_dtype: bool = False
):
    """Return a dtype the selected AWQ kernel can actually execute.

    An unsupported explicit ``dtype=`` is a configuration conflict, so raise
    here instead of letting layer-0 finalize fail after calibration has already
    run. An auto-selected dtype is fixed to fp16 and logged, and the loader then
    materializes the whole model in that dtype so the forward cannot mix dtypes.
    """
    if qcfg.quant_method not in (METHOD.AWQ, METHOD.PARO):
        return dtype
    if not isinstance(dtype, torch.dtype):
        return dtype

    format_code = resolve_quant_format(qcfg.format, qcfg.method)
    if backend in (None, BACKEND.AUTO, BACKEND.AUTO_TRAINABLE):
        if format_code not in _AWQ_FP16_ONLY_FORMATS:
            return dtype
        supported_dtypes = [torch.float16]
        kernel_name = f"FORMAT.{format_code.name} AWQ GEMV"
    else:
        try:
            qlinear = get_kernel_for_backend(backend, qcfg.quant_method, format_code)
        except ValueError:
            return dtype
        supported_dtypes = getattr(qlinear, "SUPPORTS_DTYPES", None) or []
        kernel_name = qlinear.__name__

    if dtype in supported_dtypes or torch.float16 not in supported_dtypes:
        return dtype

    if explicit_dtype:
        raise ValueError(
            f"`dtype={dtype}` is not supported by {kernel_name}: it only supports "
            f"{supported_dtypes}. Pass `dtype=torch.float16`, or use a GEMM format "
            "(for example `FORMAT.GEMM`) that supports bfloat16."
        )

    log.info(
        f"Loader: Auto fix `dtype` from `{dtype}` to `torch.float16` for "
        f"`{kernel_name}`"
    )
    return torch.float16


def check_versions(model_class, requirements: List[str]):
    if requirements is None:
        return
    for req in requirements:
        pkg, operator, version_required = parse_requirement(req)
        try:
            installed_version = version(pkg)
            if operator is None:
                continue
            if not compare_versions(installed_version, version_required, operator):
                raise ValueError(f"{model_class} requires version {req}, but current {pkg} version is {installed_version} ")
        except PackageNotFoundError:
            if operator is None:
                raise ValueError(f"{model_class} requires package {pkg}, but {pkg} not installed.")
            raise ValueError(f"{model_class} requires version {req}, but {pkg} not installed.")


def set_dtype_compat(model_init_kwargs: dict, torch_dtype):
    """
    Set dtype argument in a version-compatible way for Transformers.
    See: https://github.com/huggingface/transformers/releases/tag/v4.56.0

    Args:
        model_init_kwargs (dict): kwargs used to initialize model
        torch_dtype: torch dtype (e.g. torch.float16)
    """
    if Version(transformers.__version__) >= Version("4.56.0"):
        model_init_kwargs["dtype"] = torch_dtype
    else:
        model_init_kwargs["torch_dtype"] = torch_dtype

def get_model_local_path(pretrained_model_id_or_path, **kwargs):
    is_local = os.path.isdir(pretrained_model_id_or_path)
    if is_local or os.path.isabs(pretrained_model_id_or_path):
        return os.path.normpath(pretrained_model_id_or_path)
    kwargs.pop(INTERNAL_HF_GGUF_FILE_KWARG, None)
    def _log_removed(removed: list[str]):
        log.debug("Loader: dropping unsupported snapshot_download kwargs: %s", ", ".join(removed))

    return safe_kwargs_call(
        snapshot_download,
        pretrained_model_id_or_path,
        kwargs=kwargs,
        on_removed=_log_removed,
    )


def _get_tokenizer_load_kwargs(model_init_kwargs: Dict) -> Dict:
    """Tokenizer options that must match the config/weight selection.

    A nested checkpoint keeps its tokenizer beside the config, so forwarding
    only ``gguf_file`` would load the repository-root tokenizer (or fail when
    the root has none) while the weights came from the requested subfolder.
    """

    kwargs = get_config_selection_kwargs(model_init_kwargs)
    kwargs.update(get_hf_gguf_load_kwargs(model_init_kwargs))
    return kwargs


def _resolve_local_gguf_checkpoint_path(model_local_path: str, hf_gguf_load_kwargs: Dict[str, str]) -> Optional[str]:
    gguf_file = hf_gguf_load_kwargs.get("gguf_file")
    if not gguf_file:
        return None

    checkpoint_path = os.path.join(str(model_local_path), gguf_file)
    if not os.path.isfile(checkpoint_path):
        return None
    return checkpoint_path


def _resolve_native_quantized_gguf_checkpoint(
    model_local_path: str,
    hf_gguf_load_kwargs: Dict[str, str],
    *,
    policy: Optional[KernelConfig] = None,
) -> tuple[Optional[str], Optional[internal_gguf.GGUFQuantizedCheckpointSpec]]:
    if not _native_quantized_loader_enabled(policy):
        return None, None

    gguf_checkpoint_path = _resolve_local_gguf_checkpoint_path(model_local_path, hf_gguf_load_kwargs)
    if gguf_checkpoint_path is None:
        return None, None

    from ..utils import internal_gguf

    try:
        spec = internal_gguf.inspect_quantized_checkpoint(gguf_checkpoint_path)
    except Exception as exc:
        log.debug("Loader: failed to inspect GGUF checkpoint `%s`: %s", gguf_checkpoint_path, exc)
        return None, None

    if spec is None:
        return None, None
    return gguf_checkpoint_path, spec


def _resolve_model_slot(model: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    module_name, _, attr_name = name.rpartition(".")
    module = model.get_submodule(module_name) if module_name else model
    return module, attr_name


def _lookup_model_slot_tensor(model: torch.nn.Module, name: str) -> torch.Tensor:
    module, attr_name = _resolve_model_slot(model, name)
    if attr_name in module._parameters:
        return module._parameters[attr_name]
    if attr_name in module._buffers:
        return module._buffers[attr_name]
    raise KeyError(f"Loader: model slot `{name}` does not exist.")


def _assign_model_slot_tensor(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    module, attr_name = _resolve_model_slot(model, name)
    tensor = tensor.contiguous()

    if attr_name in module._parameters:
        current = module._parameters[attr_name]
        if current is not None and (tensor.device != current.device or tensor.dtype != current.dtype):
            tensor = tensor.to(device=current.device, dtype=current.dtype)
        requires_grad = current.requires_grad if isinstance(current, torch.nn.Parameter) else False
        module._parameters[attr_name] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
        return

    if attr_name in module._buffers:
        current = module._buffers[attr_name]
        if current is not None and (tensor.device != current.device or tensor.dtype != current.dtype):
            tensor = tensor.to(device=current.device, dtype=current.dtype)
        module._buffers[attr_name] = tensor
        return

    raise KeyError(f"Loader: model slot `{name}` does not exist.")


def _build_gguf_tensor_key_mapping(model: torch.nn.Module, config: PretrainedConfig) -> dict[str, str]:
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    processor_cls = gguf_utils.TENSOR_PROCESSORS.get(config.model_type, gguf_utils.TensorProcessor)
    if processor_cls is not gguf_utils.TensorProcessor:
        raise NotImplementedError(
            f"Loader: native quantized GGUF loading only supports the default tensor processor. "
            f"Actual processor for `{config.model_type}`: `{processor_cls.__name__}`."
        )

    processor = processor_cls(config=config.to_dict())
    return gguf_utils.get_gguf_hf_weights_map(model, processor)


def _load_quantized_gguf_checkpoint_into_model(
    *,
    model: torch.nn.Module,
    gguf_checkpoint_path: str,
    tensor_key_mapping: dict[str, str],
) -> None:
    from ..nn_modules.qlinear.gguf import GGUFTorchLinear
    from ..utils import internal_gguf

    reader = internal_gguf.GGUFReader(gguf_checkpoint_path)
    loaded: set[str] = set()

    for tensor in reader.tensors:
        target_name = tensor_key_mapping.get(tensor.name)
        if target_name is None:
            continue

        module_name, _, attr_name = target_name.rpartition(".")
        target_module = model.get_submodule(module_name) if module_name else model
        resolved_target_name = target_name

        if isinstance(target_module, GGUFTorchLinear) and attr_name == "weight":
            resolved_target_name = f"{module_name}.qweight" if module_name else "qweight"
            # ``astype`` is the array's own copy-and-cast, so it keeps the old
            # ``np.array(..., dtype=uint8, copy=True, order="C")`` semantics
            # (by-value cast, C-order, native byte order) without importing
            # NumPy into this module. ``as_tensor`` rejects big-endian dtypes
            # and negative strides, both of which the raw memmap view can carry.
            packed = torch.as_tensor(
                tensor.data.astype("uint8", copy=True), dtype=torch.uint8
            )
            expected = _lookup_model_slot_tensor(model, resolved_target_name)
            if tuple(packed.shape) != tuple(expected.shape):
                raise RuntimeError(
                    f"Loader: GGUF qweight shape mismatch for `{resolved_target_name}`. "
                    f"Expected {tuple(expected.shape)}, got {tuple(packed.shape)}."
                )
            _assign_model_slot_tensor(model, resolved_target_name, packed)
            loaded.add(resolved_target_name)
            continue

        reference = _lookup_model_slot_tensor(model, resolved_target_name)
        weights = internal_gguf.dequantize_to_torch(
            tensor.data,
            tensor.tensor_type,
            device=reference.device,
            dtype=reference.dtype,
        )
        _assign_model_slot_tensor(model, resolved_target_name, weights)
        loaded.add(resolved_target_name)

    missing_qweights = []
    for module_name, module in model.named_modules():
        if not isinstance(module, GGUFTorchLinear):
            continue
        qweight_name = f"{module_name}.qweight" if module_name else "qweight"
        if qweight_name not in loaded:
            missing_qweights.append(qweight_name)
    if missing_qweights:
        raise RuntimeError(
            "Loader: GGUF checkpoint did not populate required quantized weights: "
            + ", ".join(sorted(missing_qweights))
        )

    model.tie_weights()


def _resolve_exl3_auto_backend(device, *, cuda_available: Optional[bool] = None) -> "BACKEND":
    """Pick the EXL3 kernel for ``backend=AUTO``.

    Only the CUDA family gets ``EXL3_EXLLAMA_V3``: its tensor-core GEMM dispatch
    table is populated exclusively for CUDA.  On ROCm every entry of that table
    is ``nullptr`` (``exl3_gemm.cu`` already rejects HIP with an explicit
    ``TORCH_CHECK``), so the AUTO branch used to load a backend that cannot run.
    ROCm -- CDNA and RDNA4 alike -- therefore resolves to ``EXL3_TORCH``, the
    portable kernel that is the supported ROCm path.
    """

    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    device_family = selector_device_family(device)
    if cuda_available and device_family in ExllamaV3Linear.SUPPORTS_DEVICES:
        return BACKEND.EXL3_EXLLAMA_V3
    return BACKEND.EXL3_TORCH


def _apply_requested_module_tree(
    definition_cls,
    model,
    module_tree,
    quantize_config,
    *,
    allow_auto: bool,
) -> None:
    """Attach an explicit or auto-detected module tree to a loaded model.

    ``None`` keeps the existing fail-closed behavior.  A list/tuple is an
    explicit declaration.  ``"auto"`` asks the definition's structural
    discovery to build one complete variant per concurrent decoder stack; it is
    only valid for an unquantized source because a quantized checkpoint no
    longer exposes the dense ``Linear`` leaves discovery looks for.
    """

    if module_tree is None:
        return

    if isinstance(module_tree, str):
        if module_tree != "auto":
            raise ValueError(
                f"module_tree={module_tree!r} is not supported; "
                "pass a tree declaration (list/tuple) or 'auto'."
            )
        if quantize_config is None:
            raise ValueError(
                "module_tree='auto' requires a quantize_config so the quantization "
                "method is known; omit module_tree when loading only for evaluation."
            )
        if not allow_auto:
            raise ValueError(
                "module_tree='auto' requires an unquantized source model loaded through "
                "from_pretrained(); quantized checkpoints need an explicit module tree."
            )
        detected = definition_cls._auto_detect_module_tree(
            model, getattr(quantize_config, "method", None)
        )
        if detected is None:
            model_type = getattr(getattr(model, "config", None), "model_type", "<unknown>")
            raise ValueError(
                f"module_tree='auto' could not resolve any quantizable decoder branch for "
                f"model_type {model_type!r}. Register an explicit model definition or pass "
                "an explicit module tree declaration."
            )
        model.gptqmodel_effective_module_tree = copy.deepcopy(detected)
        return

    if not isinstance(module_tree, (list, tuple)):
        raise TypeError(
            "module_tree must be None, 'auto', or a list/tuple tree declaration; "
            f"got {type(module_tree).__name__}."
        )

    model.gptqmodel_effective_module_tree = copy.deepcopy(module_tree)


def ModelLoader(cls):
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_id_or_path: str,
            quantize_config: BaseQuantizeConfig,
            backend: Backend = BACKEND.AUTO,
            profile: Union[str, int, PROFILE] = PROFILE.AUTO,
            trust_remote_code: bool = False,
            dtype: [str | torch.dtype] = "auto",
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            device: Optional[Union[str, int, torch.device, Sequence[Union[str, int, torch.device]]]] = None,
            fuse: bool = False,
            p2p: Optional[Union[P2PConfig, Dict]] = None,
            kernel: Optional[Union[KernelConfig, Dict]] = None,
            module_tree: Optional[Union[str, List[Any], tuple]] = None,
            runtime: Optional[RuntimeConfig] = None,
            **model_init_kwargs,
    ):
        backend, kernel = resolve_quantize_backend(backend, quantize_config, kernel, runtime)
        if p2p is not None and quantize_config is not None:
            # The loopers read this field and install it before enabling peers.
            quantize_config.p2p = P2PConfig(**p2p) if isinstance(p2p, dict) else p2p

        # quantization is unsafe with GIL=0 and torch.compile/graphs
        import torch._dynamo
        torch._dynamo.disable()
        default_dtype_before = torch.get_default_dtype()

        pretrained_model_id_or_path = normalize_model_id_or_path_for_hf_gguf(
            pretrained_model_id_or_path,
            model_init_kwargs,
            api_name=f"{cls.__name__}.from_pretrained",
        )

        dtype = normalize_torch_dtype_kwarg(
            model_init_kwargs,
            api_name=f"{cls.__name__}.from_pretrained",
            explicit_dtype=dtype,
        )
        requested_dtype = dtype if isinstance(dtype, torch.dtype) else None
        backend = normalize_backend(backend)
        profile = normalize_profile(profile)
        hf_gguf_load_kwargs = get_hf_gguf_load_kwargs(model_init_kwargs)
        model_init_kwargs_without_internal = dict(model_init_kwargs)
        model_init_kwargs_without_internal.pop(INTERNAL_HF_GGUF_FILE_KWARG, None)

        tokenizer_trust_remote_code = model_init_kwargs_without_internal.pop("tokenizer_trust_remote_code", trust_remote_code)
        model_local_path = get_model_local_path(pretrained_model_id_or_path, **model_init_kwargs_without_internal)
        trust_remote_code = resolve_trust_remote_code(model_local_path, trust_remote_code=trust_remote_code)

        model_init_kwargs_without_internal["trust_remote_code"] = trust_remote_code

        config = AutoConfig.from_pretrained(model_local_path, **model_init_kwargs_without_internal, **hf_gguf_load_kwargs)

        defuser.replace_fused_blocks(config.model_type)

        normalize_hf_config_compat(config, trust_remote_code=trust_remote_code)
        prepare_remote_model_init_compat(
            model_local_path, config, trust_remote_code=trust_remote_code
        )

        from ..utils.mimo import is_mimo_mixed_source

        mimo_source = is_mimo_mixed_source(config)

        atten_impl = model_init_kwargs.get("attn_implementation", None)

        if atten_impl is not None and atten_impl != "auto":
            log.info(f"Loader: overriding attn_implementation in config to `{atten_impl}`")
            _override_attn_implementation(config, atten_impl)

        resolved_device = normalize_device_device_map(device, device_map)
        resolved_device = auto_select_device(resolved_device, backend)

        if cls.require_dtype:
            dtype = cls.require_dtype
        elif dtype is None or dtype == "auto" or not isinstance(dtype, torch.dtype):
            dtype_device = selector_device_family(resolved_device)
            if mimo_source and isinstance(quantize_config, BaseQuantizeConfig) and quantize_config.device is not None:
                dtype_device = selector_device_family(normalize_device(quantize_config.device))
            dtype = auto_dtype(
                config=config,
                device=dtype_device,
                quant_inference=False,
            )

        # Resolve fp16-only AWQ kernels (the GEMV layouts) before the shell
        # model is materialized, so every module -- including the non-quantized
        # ones -- loads in one dtype and the failure cannot surface in layer-0
        # finalize after calibration.
        if isinstance(quantize_config, BaseQuantizeConfig):
            dtype = _coerce_quantized_awq_dtype(
                backend=backend,
                qcfg=quantize_config,
                dtype=dtype,
                explicit_dtype=requested_dtype is not None,
            )

        if isinstance(dtype, torch.dtype) and get_hf_config_dtype(config) != dtype:
            # Align config metadata with the dtype we will materialize weights in.
            set_hf_config_dtype(config, dtype)

        # Do this before tokenizer/model construction so an unsupported target
        # fails immediately rather than after a lengthy checkpoint load.
        if isinstance(quantize_config, BaseQuantizeConfig):
            requested_quant_device = (
                auto_select_device(None, None)
                if quantize_config.device is None
                else normalize_device(quantize_config.device)
            )
            configure_native_floatx_source_quantization(
                config,
                quantize_config,
                device=requested_quant_device,
                model_local_path=model_local_path,
                model_dtype=dtype,
            )

        tokenizer = load_hf_tokenizer(
            model_local_path,
            model_config=config,
            trust_remote_code=tokenizer_trust_remote_code,
            **_get_tokenizer_load_kwargs(model_init_kwargs),
        )

        gguf_checkpoint_path, native_gguf_qspec = _resolve_native_quantized_gguf_checkpoint(
            model_local_path,
            hf_gguf_load_kwargs,
            policy=kernel,
        )
        effective_profile = _resolve_native_gguf_profile(
            native_gguf_qspec=native_gguf_qspec,
            profile=profile,
        )

        if quantize_config is None:
            if native_gguf_qspec is not None:
                if _should_use_dense_native_gguf_path(
                    native_gguf_qspec=native_gguf_qspec,
                    profile=effective_profile,
                ):
                    if backend != BACKEND.AUTO:
                        log.info(
                            "Loader: PROFILE.%s uses dense GGUF import for `%s`; backend `%s` is ignored.",
                            effective_profile.name,
                            gguf_checkpoint_path,
                            backend.value,
                        )
                else:
                    redirect_kwargs = dict(model_init_kwargs)
                    redirect_kwargs.pop("tokenizer_trust_remote_code", None)
                    log.info(
                        "Loader: detected native quantized GGUF checkpoint `%s`; redirecting `%s` to from_quantized() with PROFILE.%s.",
                        gguf_checkpoint_path,
                        cls.__name__,
                        effective_profile.name,
                    )
                    return cls.from_quantized(
                        model_id_or_path=pretrained_model_id_or_path,
                        device_map=device_map,
                        device=device,
                        backend=backend,
                        kernel=kernel,
                        dtype=dtype,
                        trust_remote_code=trust_remote_code,
                        tokenizer_trust_remote_code=tokenizer_trust_remote_code,
                        module_tree=module_tree,
                        **redirect_kwargs,
                    )

            if hf_gguf_load_kwargs:
                # The dense GGUF import runs inside transformers, whose checkpoint
                # hook is process-wide. Install this call's policy first (we hold
                # the model-load lock) so the dequant device and loader choice
                # come from these arguments rather than the previous load.
                configure_kernel(kernel)
            hf_model_init_kwargs = dict(model_init_kwargs_without_internal)
            hf_model_init_kwargs["device_map"] = device_map if device_map else "auto"
            set_dtype_compat(hf_model_init_kwargs, dtype)
            hf_model_init_kwargs.update(hf_gguf_load_kwargs)
            if (
                native_gguf_qspec is not None
                and native_gguf_qspec.tensor_qtype == _ggml_quantization_type().Q1_0
                and atten_impl in {None, "auto"}
                and _is_accelerated_attention_device(resolved_device)
                and (config.model_type == "qwen3" or _supports_flash_attn_2(config))
                and is_flash_attn_2_available()
            ):
                hf_model_init_kwargs[ATTN_IMPLEMENTATION] = "flash_attention_2"
                log.info("Loader: Auto enabling flash_attention_2 for dense Bonsai PROFILE.%s.", effective_profile.name)
            # Load a non-quantized model, but do not perform quantization. For example, for evaluation.
            _call_before_model_load(
                cls,
                model_local_path=model_local_path,
                load_quantized_model=False,
            )
            model = _hf_loader_from_pretrained_with_dynamic_module_retry(
                cls.loader,
                model_local_path,
                config=config,
                **hf_model_init_kwargs,
            )
            model._model_init_kwargs = hf_model_init_kwargs
            _maybe_print_module_tree(model=model)
            _apply_requested_module_tree(
                cls, model, module_tree, quantize_config, allow_auto=False
            )

            turtle_model = None

            instance = cls(
                model,
                turtle_model=turtle_model,
                quantized=False,
                fusion_enabled=fuse,
                quantize_config=quantize_config,
                tokenizer=tokenizer,
                trust_remote_code=trust_remote_code,
                model_local_path=model_local_path,
            )
            _set_paged_attention_safe_cuda_graphs(instance.model)

            assert_loaded_model_state(
                model,
                api_name=f"{cls.__name__}.from_pretrained",
                default_dtype_before=default_dtype_before,
                expected_dtype=requested_dtype,
                require_uniform_dtype=True,
            )
            return instance

        load_start = time.perf_counter()

        # non-quantized models are always loaded into cpu
        cpu_device_map = {"": "cpu"}

        if quantize_config is None or not isinstance(quantize_config, BaseQuantizeConfig):
            raise AttributeError("`quantize_config` must be passed and be an instance of BaseQuantizeConfig.")

        quantize_config.calculate_bits_per_weight()

        if quantize_config.device is not None:
            if device is not None or device_map is not None:
                raise AttributeError("Passing device and device_map is not allowed when QuantizeConfig.device is set. Non-quantized model is always loaded as cpu. Please set QuantizeConfig.device for accelerator used in quantization or do not set for auto-selection.")

        if quantize_config.desc_act not in cls.supports_desc_act:
            raise ValueError(f"{cls} only supports desc_act={cls.supports_desc_act}, "
                             f"but quantize_config.desc_act is {quantize_config.desc_act}.")

        native_support = has_native_transformers_causallm_support(model_local_path)
        if cls.require_trust_remote_code and not trust_remote_code and not native_support:
            raise ValueError(
                f"{pretrained_model_id_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        check_versions(cls, cls.require_pkgs)

        with suspend_hf_weight_init():
            # normalize and auto select quantization device is not passed
            if quantize_config.device is None:
                requested_devices = _explicit_requested_devices(device)
                # `quantize_config.device` only carries an accelerator family
                # (ordinals are stripped by `normalize_device`), so an explicit
                # device list is expressed through the compute filter.
                quantize_config.device = auto_select_device(None, None)
                if requested_devices is not None and quantize_config.compute_device_filter is None:
                    wanted = {str(target) for target in requested_devices}
                    quantize_config.compute_device_filter = (
                        lambda devices, _wanted=wanted: [
                            target for target in devices if str(target) in _wanted
                        ]
                    )
            else:
                quantize_config.device = normalize_device(quantize_config.device)

            if dtype is None or dtype == "auto" or not isinstance(dtype, torch.dtype):
                # TODO FIX ME for `dynamic`, non-quantized modules should be in native type
                dtype = auto_dtype(config=config, device=quantize_config.device, quant_inference=False)

            # enforce some values despite user specified
            # non-quantized models are always loaded into cpu
            model_init_kwargs_without_internal["device_map"] = cpu_device_map
            set_dtype_compat(model_init_kwargs_without_internal, dtype)
            model_init_kwargs_without_internal["_fast_init"] = cls.require_fast_init
            #model_init_kwargs["low_cpu_mem_usage"] = True

            _call_before_model_load(
                cls,
                model_local_path=model_local_path,
                load_quantized_model=False,
            )
            from ..utils.hf import build_shell_model

            # XIELUActivation will use some weights when activation init, so can't use init_empty_weights
            if hasattr(config, "hidden_act") and config.hidden_act == "xielu":
                quantize_config.offload_to_disk = False

            # some models need convert moe-experts after model loaded, like GPTOSS and Llama4
            # so offload_to_disk is not supported for them.
            if not cls.support_offload_to_disk:
                quantize_config.offload_to_disk = False
                log.warn(f"{cls} doesn't support offload_to_disk, set quantize_config.offload_to_disk to False.")

            if quantize_config.offload_to_disk:
                shell_config = copy.deepcopy(config)
                try:
                    model = build_shell_model(cls.loader, config=shell_config, **model_init_kwargs_without_internal)
                except RuntimeError as exc:
                    if not _is_meta_shell_build_error(exc):
                        raise
                    if mimo_source:
                        raise RuntimeError(
                            "MiMo mixed sources require a checkpoint-backed shell; "
                            "direct HF loading cannot preserve their packed layout."
                        ) from exc

                    log.warn(
                        "Loader: meta-device shell build failed for `%s`; falling back to direct CPU load without turtle_model: %s",
                        model_local_path,
                        exc,
                    )
                    log.info("Loader: loading model directly to CPU (meta shell unsupported; turtle_model disabled)")
                    fallback_init_kwargs = model_init_kwargs_without_internal.copy()
                    fallback_init_kwargs.pop("device_map", None)
                    fallback_init_kwargs["low_cpu_mem_usage"] = False
                    model = _hf_loader_from_pretrained_with_dynamic_module_retry(
                        cls.loader,
                        model_local_path,
                        config=config,
                        **fallback_init_kwargs,
                        **hf_gguf_load_kwargs,
                    )
                    if getattr(model, "config", None) is config:
                        model.config = copy.deepcopy(config)
                    _convert_model_with_defuser(cls, model, cleanup_original=False)
                    model._model_init_kwargs = fallback_init_kwargs
                    _maybe_print_module_tree(model=model)
                    turtle_model = None
                else:
                    _convert_model_with_defuser(cls, model, cleanup_original=False)
                    shell_model_init_kwargs = dict(model_init_kwargs_without_internal)
                    shell_model_init_kwargs.update(hf_gguf_load_kwargs)
                    model._model_init_kwargs = shell_model_init_kwargs
                    _maybe_print_module_tree(model=model)
                    turtle_model = LazyTurtle.maybe_create(
                        model_local_path=model_local_path,
                        config=config if mimo_source else model.config,
                        model_init_kwargs=shell_model_init_kwargs,
                        module_tree=copy.deepcopy(getattr(cls, "module_tree", None)),
                        hf_conversion_map_reversed=copy.deepcopy(
                            cls.resolve_hf_conversion_map_reversed(target_model=model)
                        ),
                        target_model=model,
                        # A `.bin`/`.pt` source is converted to a full safetensors
                        # copy.  Put that copy under the configured offload root
                        # instead of the system TMPDIR; the turtle only falls back
                        # to `tempfile.gettempdir()` when no path is configured.
                        offload_path=getattr(quantize_config, "offload_to_disk_path", None),
                    )

                    if turtle_model is None:
                        raise RuntimeError(
                            f"Loader: can't open model path `{model_local_path}` for offload_to_disk."
                        )

                    log.info(
                        "Loader: using checkpoint-backed lazy turtle source for `%s`",
                        model_local_path,
                    )
                    # LazyTurtle is only created by this quantization offload path,
                    # so a shard shared by several layers forces repeated shard I/O
                    # worth surfacing once.
                    turtle_model.warn_if_checkpoint_shards_are_not_layerwise()
            else:
                log.info("Loader: loading model directly to CPU (not using meta device or turtle_model)")
                model = _hf_loader_from_pretrained_with_dynamic_module_retry(
                    cls.loader,
                    model_local_path,
                    config=config,
                    **model_init_kwargs_without_internal,
                    **hf_gguf_load_kwargs,
                )
                if getattr(model, "config", None) is config:
                    model.config = copy.deepcopy(config)
                _convert_model_with_defuser(cls, model, cleanup_original=False)
                direct_model_init_kwargs = dict(model_init_kwargs_without_internal)
                direct_model_init_kwargs.update(hf_gguf_load_kwargs)
                model._model_init_kwargs = direct_model_init_kwargs
                _maybe_print_module_tree(model=model)

                turtle_model = None

        _apply_requested_module_tree(
            cls, model, module_tree, quantize_config, allow_auto=True
        )

        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions", "multimodal_max_length"]
        config_seq_len = find_config_seq_len(model_config, seq_len_keys)
        if config_seq_len is not None:
            model.seqlen = config_seq_len
        else:
            log.warn("Model: can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        model.eval()
        turtle_model.eval() if turtle_model is not None else None

        instance = cls(
            model,
            turtle_model=turtle_model,
            quantized=False,
            fusion_enabled=fuse,
            quantize_config=quantize_config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            model_local_path=model_local_path,
        )
        _set_paged_attention_safe_cuda_graphs(instance.model)

        if quantize_config.dynamic:
            log_dynamic_override_cache_stats(
                instance._dynamic_cache_stats_before, dynamic=quantize_config.dynamic,
            )

        timer = getattr(instance, "quant_region_timer", None)
        if timer is not None:
            source_label = getattr(instance, "model_local_path", None) or str(pretrained_model_id_or_path)
            timer.record("model_load", time.perf_counter() - load_start, source=source_label)

        # ``offload_to_disk`` builds a meta shell that ``LazyTurtle`` fills in
        # later, so only the default-dtype invariant applies to it.
        materialized = turtle_model is None
        assert_loaded_model_state(
            model,
            api_name=f"{cls.__name__}.from_pretrained",
            default_dtype_before=default_dtype_before,
            expected_dtype=requested_dtype if materialized else None,
            require_uniform_dtype=materialized,
            require_materialized=materialized,
        )
        return instance

    cls.from_pretrained = classmethod(_serialized_model_load(from_pretrained.__func__))

    @classmethod
    def from_quantized(
            cls,
            model_id_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            device: Optional[Union[str, int, torch.device, Sequence[Union[str, int, torch.device]]]] = None,
            backend: Backend = BACKEND.AUTO,
            adapter: Optional[Adapter | Dict] = None,
            dtype: [str | torch.dtype] = "auto",
            trust_remote_code: bool = False,
            allow_remote_adapter: bool = False,
            adapter_revision: Optional[str] = None,
            adapter_sha256: Optional[str] = None,
            fuse: bool = False,
            p2p: Optional[Union[P2PConfig, Dict]] = None,
            kernel: Optional[Union[KernelConfig, Dict]] = None,
            module_tree: Optional[Union[str, List[Any], tuple]] = None,
            runtime: Optional[RuntimeConfig] = None,
            **kwargs,
    ):
        backend, kernel = resolve_backend_config(backend, kernel, runtime)
        warn_unsupported_rocm()

        # Tenstorrent loads can leave the GDDR hot from the previous model. Pace
        # this one before any device is claimed. The native TTNN backend also
        # gates its own mesh open; the two gates share one rate limit, so a load
        # never pays the cooldown twice.
        if normalize_backend(backend) in (BACKEND.GPTQ_TTNN, BACKEND.GPTQ_TT_XLA):
            from ..utils.tt_thermal import before_device_open

            before_device_open(owner=f"{cls.__name__}.from_quantized[{normalize_backend(backend).value}]")

        # Install the peer-to-peer policy before any multi-device load so the
        # pre-enablement below and the inference-time copies use it.
        configure_p2p(p2p)
        # Install the kernel policy before any quantized module is built.
        configure_kernel(kernel)

        # post-quant is safe with GIL=0 and torch.compile/graphs

        import torch._dynamo
        torch._dynamo.reset()
        default_dtype_before = torch.get_default_dtype()
        model_id_or_path = normalize_model_id_or_path_for_hf_gguf(
            model_id_or_path,
            kwargs,
            api_name=f"{cls.__name__}.from_quantized",
        )
        dtype = normalize_torch_dtype_kwarg(
            kwargs,
            api_name=f"{cls.__name__}.from_quantized",
            explicit_dtype=dtype,
        )
        requested_dtype = dtype if isinstance(dtype, torch.dtype) else None
        hf_gguf_load_kwargs = get_hf_gguf_load_kwargs(kwargs)
        kwargs_without_internal = dict(kwargs)
        kwargs_without_internal.pop(INTERNAL_HF_GGUF_FILE_KWARG, None)
        max_memory = kwargs_without_internal.pop("max_memory", None)
        tokenizer_trust_remote_code = kwargs_without_internal.pop("tokenizer_trust_remote_code", trust_remote_code)
        requested_device_map, explicit_device_map = _normalize_explicit_placement(
            device, device_map, max_memory
        )
        # Kept for the block-float route, which builds on CPU and must not
        # silently ignore a placement request the caller made.
        requested_device = device

        # A budget that names only CPU can only place weights on CPU, so the
        # kernel must be a CPU kernel. The placement builder already honours the
        # budget; without this the selected CUDA kernel crashes on the first
        # CPU-resident layer's forward.
        max_memory_device = _resolve_max_memory_device(max_memory, device)
        if max_memory_device is not None:
            device = max_memory_device
            requested_device_map = None

        # A CPU-resident layer runs `post_init` on CPU. AUTO must therefore not
        # select a CUDA-only kernel for a map that offloads some layers.
        cpu_offload = _device_map_uses_cpu(explicit_device_map)
        if max_memory is not None and any(
            str(key).strip().lower() == "cpu" for key in max_memory
        ):
            cpu_offload = True

        # Keep string inputs compatible while allowing canonical method-prefixed names.
        backend = normalize_backend(backend)
        if requested_device_map is None:
            # Preserve explicit sequences; normalization reduces a singleton
            # sequence to a scalar, which would broaden the layer-wise map.
            device = _resolve_default_layerwise_device(device, backend)
        else:
            # The explicit device map determines placement and kernel targets.
            device = auto_select_device(normalize_device_device_map(None, requested_device_map), backend)

        model_local_path = get_model_local_path(model_id_or_path, **kwargs_without_internal)
        trust_remote_code = resolve_trust_remote_code(model_local_path, trust_remote_code=trust_remote_code)
        native_support = has_native_transformers_causallm_support(model_local_path)

        """load quantized model from local disk"""
        if cls.require_trust_remote_code and not trust_remote_code and not native_support:
            raise ValueError(
                f"{model_id_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        check_versions(cls, cls.require_pkgs)

        # A native block-float checkpoint carries none of the GPTQ keys this
        # loader binds (qweight/qzeros/scales/g_idx): it stores packed mantissa
        # codes plus one exponent per block of 16 values. Route it before any
        # GPTQ parsing, otherwise every decoder layer reads as missing. The
        # guards above still apply -- this path builds the model through
        # transformers like any other, so the same version and
        # trust_remote_code rules hold.
        from .tt_blockfloat_loader import is_tt_blockfloat_checkpoint

        if is_tt_blockfloat_checkpoint(model_local_path):
            if requested_device_map is not None or (
                requested_device is not None
                and str(requested_device).split(":")[0].lower() != "cpu"
            ):
                # The block-float kernel is the portable correctness reference,
                # not the device path. Accepting a placement request here and
                # ignoring it would report a device the model is not on.
                raise NotImplementedError(
                    "tt-blockfloat checkpoints load on CPU; the block-float "
                    "kernel is a correctness reference and device placement is "
                    "the caller's dispatch step. Omit device/device_map, or pass "
                    "device='cpu'."
                )
            return _load_tt_blockfloat_model(
                cls,
                model_local_path=model_local_path,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs_without_internal.pop("cache_dir", None)
        force_download = kwargs_without_internal.pop("force_download", False)
        resume_download = kwargs_without_internal.pop("resume_download", False)
        proxies = kwargs_without_internal.pop("proxies", None)
        local_files_only = kwargs_without_internal.pop("local_files_only", False)
        use_auth_token = kwargs_without_internal.pop("use_auth_token", None)
        revision = kwargs_without_internal.pop("revision", None)
        subfolder = kwargs_without_internal.pop("subfolder", "")
        # Resolve the requested repo subfolder once. The quantize config, the
        # checkpoint shards, and a relative adapter manifest must all be read
        # from the same directory, and that directory must stay inside the
        # checkpoint the caller named.
        checkpoint_dir = resolve_checkpoint_subfolder(model_local_path, subfolder)
        commit_hash = kwargs_without_internal.pop("_commit_hash", None)
        attn_implementation = kwargs_without_internal.pop("attn_implementation", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
            "attn_implementation": attn_implementation,
        }

        # == step1: prepare configs and file names == #
        config: PretrainedConfig = AutoConfig.from_pretrained(
            model_local_path,
            trust_remote_code=trust_remote_code,
            **cached_file_kwargs,
            **hf_gguf_load_kwargs,
        )

        defuser.replace_fused_blocks(config.model_type)

        normalize_hf_config_compat(config, trust_remote_code=trust_remote_code)
        prepare_remote_model_init_compat(
            model_local_path, config, trust_remote_code=trust_remote_code
        )

        if cls.require_dtype:
            dtype = cls.require_dtype

        if backend == BACKEND.SGLANG:
            # SGLang accepts string dtype names, but the generic loader's
            # non-torch dtype path means "auto". Preserve explicit strings by
            # resolving them before auto_dtype consults the model config.
            dtype = _normalize_sglang_load_dtype(dtype)

        if dtype is None or dtype == "auto" or not isinstance(dtype, torch.dtype) :
            # TODO FIX ME for `dynamic`, non-quantized modules should be in native type
            dtype = auto_dtype(
                config=config,
                device=selector_device_family(device),
                quant_inference=True,
            )

        if isinstance(dtype, torch.dtype) and get_hf_config_dtype(config) != dtype:
            # Ensure flash attention kernels see an explicit dtype instead of relying on defaults.
            set_hf_config_dtype(config, dtype)

        gguf_checkpoint_path, native_gguf_qspec = _resolve_native_quantized_gguf_checkpoint(
            model_local_path,
            hf_gguf_load_kwargs,
            policy=kernel,
        )
        if native_gguf_qspec is not None:
            qcfg = QuantizeConfig(
                bits=native_gguf_qspec.bits_alias,
                method=METHOD.GGUF,
                lm_head=native_gguf_qspec.lm_head_quantized,
            )
        else:
            qcfg = QuantizeConfig.from_pretrained(model_local_path, **cached_file_kwargs, **kwargs_without_internal)

        # Runtime policy belongs to this checkpoint's modules, not whichever
        # model most recently installed a process policy. It is a resolved
        # policy, not a caller-supplied `kernel=`, so keep it out of qcfg.kernel.
        qcfg._resolved_kernel = kernel

        if adapter is None:
            if allow_remote_adapter or adapter_revision is not None or adapter_sha256 is not None:
                raise ValueError(
                    "Loader: remote adapter options require an explicit `adapter=` argument; "
                    "checkpoint metadata cannot authorize remote adapter downloads."
                )
            if qcfg.adapter is not None and qcfg.adapter.path not in (None, ""):
                qcfg.adapter.path = resolve_checkpoint_adapter_path(qcfg.adapter.path, checkpoint_dir)
        else:
            qcfg.adapter = normalize_adapter(adapter)
            configure_explicit_remote_adapter(
                qcfg.adapter,
                allow_remote=allow_remote_adapter,
                revision=adapter_revision,
                sha256=adapter_sha256,
            )

        export_quant_method = qcfg.export_quant_method()
        format_code = resolve_quant_format(qcfg.format, qcfg.method)
        # Remember the layout the tensors actually have on disk.  A re-save
        # after mutating `quantize_config.format` must convert qzeros instead
        # of copying the old words under a new label.
        qcfg._checkpoint_format_on_disk = format_code
        qcfg._checkpoint_layout_snapshot = {
            "format": format_code,
            "bits": qcfg.bits,
            "group_size": qcfg.group_size,
            "sym": qcfg.sym,
            "desc_act": qcfg.desc_act,
            "pack_dtype": qcfg.pack_dtype,
        }
        backend = normalize_backend(backend, quant_method=export_quant_method)
        backend = backend_for_activation(qcfg.activation_mode, backend)
        backend = _auto_select_mlx_backend(
            backend, device, config, qcfg, export_quant_method, format_code, adapter
        )

        if (
            native_gguf_qspec is not None
            and native_gguf_qspec.tensor_qtype == _ggml_quantization_type().Q1_0
            and backend not in {BACKEND.AUTO, BACKEND.GGUF_TORCH, BACKEND.GGUF_TRITON, BACKEND.MLX}
        ):
            raise ValueError(
                "Native Q1_0 GGUF checkpoints support AUTO, GGUF_TORCH, GGUF_TRITON, or MLX. "
                f"Actual backend: `{backend}`."
            )
        elif (
            native_gguf_qspec is not None
            and native_gguf_qspec.tensor_qtype == _ggml_quantization_type().Q2_0
        ):
            if backend == BACKEND.AUTO:
                backend = BACKEND.GGUF_TORCH
            elif backend not in (BACKEND.GGUF_TORCH, BACKEND.MLX):
                raise ValueError(
                    "Native Q2_0 GGUF checkpoints require GGUF_TORCH or MLX. "
                    f"Actual backend: `{backend}`."
                )

        if format_code == FORMAT.EXL3:
            if backend not in (BACKEND.AUTO, BACKEND.EXL3_EXLLAMA_V3, BACKEND.EXL3_TORCH, BACKEND.MLX):
                raise TypeError("FORMAT.EXL3 requires AUTO, EXL3_EXLLAMA_V3, EXL3_TORCH, or MLX backend.")
            if backend == BACKEND.AUTO:
                backend = _resolve_exl3_auto_backend(device)
            if backend == BACKEND.EXL3_EXLLAMA_V3:
                if not torch.cuda.is_available():
                    raise ValueError("EXL3_EXLLAMA_V3 loading requires NVIDIA CUDA; use EXL3_TORCH instead.")
                ExllamaV3Linear.validate_device(selector_device_family(device))
        elif format_code == FORMAT.BITSANDBYTES:
            if backend not in (BACKEND.AUTO, BACKEND.BITSANDBYTES, BACKEND.MLX):
                raise TypeError("FORMAT.BITSANDBYTES requires AUTO, BITSANDBYTES, or MLX backend.")
            if backend == BACKEND.AUTO:
                backend = BACKEND.BITSANDBYTES

        _reject_native_awq_gemv_on_rocm(export_quant_method, format_code, backend)

        dtype = _coerce_quantized_awq_dtype(
            backend=backend,
            qcfg=qcfg,
            dtype=dtype,
            explicit_dtype=requested_dtype is not None,
        )

        qcfg.calculate_bits_per_weight()

        tokenizer = load_hf_tokenizer(
            model_local_path,
            model_config=config,
            trust_remote_code=tokenizer_trust_remote_code,
            # The tokenizer must come from the same directory and revision as
            # the config and weights. A nested checkpoint keeps all of them in
            # the requested subfolder, so resolving only the weights there would
            # load the root tokenizer (or fail on a repo that has none).
            **_get_tokenizer_load_kwargs(
                {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "resume_download": resume_download,
                    "proxies": proxies,
                    "local_files_only": local_files_only,
                    "use_auth_token": use_auth_token,
                    "revision": revision,
                    "subfolder": subfolder,
                }
            ),
            **hf_gguf_load_kwargs,
        )

        if backend == BACKEND.VLLM or backend == BACKEND.SGLANG:
            runtime_generate = None
            _validate_external_backend_format(backend, format_code)

            if backend == BACKEND.VLLM:
                from ..utils.vllm import (
                    get_vllm_device,
                    get_vllm_model_config,
                    load_model_by_vllm,
                    vllm_generate,
                )

                model = load_model_by_vllm(
                    model=model_local_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs_without_internal,
                )

                model.config = get_vllm_model_config(model)
                runtime_device = get_vllm_device(model)
                if runtime_device is not None:
                    model.device = runtime_device
                runtime_generate = vllm_generate

            elif backend == BACKEND.SGLANG:
                from ..utils.sglang import load_model_by_sglang, sglang_generate

                sglang_kwargs = _build_sglang_runtime_kwargs(
                    kwargs=kwargs_without_internal,
                    device=device,
                    requested_device_map=requested_device_map,
                    resolved_dtype=dtype,
                )
                model, hf_config = load_model_by_sglang(
                    model=model_local_path,
                    trust_remote_code=trust_remote_code,
                    config=config,
                    **sglang_kwargs,
                )
                model.config = hf_config
                runtime_generate = sglang_generate
            instance = cls(
                model,
                quantized=True,
                fusion_enabled=fuse,
                quantize_config=qcfg,
                tokenizer=tokenizer,
                qlinear_kernel=None,
                load_quantized_model=True,
                trust_remote_code=trust_remote_code,
                model_local_path=model_local_path,
            )
            instance._runtime_generate = runtime_generate
            _set_paged_attention_safe_cuda_graphs(instance.model)
            return instance

        if format_code == FORMAT.MARLIN:
            # CUDA uses the shared Marlin backend and lets the checkpoint
            # method/format route to the GPTQ or AWQ sub-kernel. MLX transcodes
            # AWQ/GPTQ source packing into the native Metal layout during load.
            if backend == BACKEND.MLX:
                pass
            elif backend not in (BACKEND.MARLIN, BACKEND.AUTO):
                raise TypeError(
                    f"FORMAT.MARLIN requires BACKEND.AUTO or BACKEND.MARLIN: actual = `{backend}`."
                )
            else:
                backend = BACKEND.MARLIN

        # marlin_compatible = False if backend == BACKEND.IPEX else _validate_marlin_device_support()
        # check for marlin compat for cuda device only
        # if backend != BACKEND.MARLIN and device == DEVICE.CUDA:
        #     unsupported = _validate_marlin_compatibility(qcfg)
        #     if unsupported is None and marlin_compatible:
        #         logger.info(
        #             "Hint: Model is compatible with the Marlin kernel. Use the canonical Marlin BACKEND enum."
        #         )

        if format_code == FORMAT.BITBLAS:
            # CUDA requires BitBLAS; MLX transcodes the portable AWQ/GPTQ
            # checkpoint tensors without importing the CUDA runtime.
            expected_backend = BACKEND.AWQ_BITBLAS if qcfg.quant_method == METHOD.AWQ else BACKEND.GPTQ_BITBLAS
            if backend == BACKEND.MLX:
                pass
            elif backend != expected_backend and backend != BACKEND.AUTO:
                raise TypeError(
                    f"FORMAT.BITBLAS requires BACKEND.AUTO or BACKEND.{expected_backend.name}: actual = `{backend}`."
                )
            else:
                backend = expected_backend

        if backend in [BACKEND.GPTQ_BITBLAS, BACKEND.AWQ_BITBLAS]:
            from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            # ``is False`` (not ``not``) keeps the pre-lazy semantics: an
            # explicit ``None``/``0`` override means "unknown", not "absent".
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        model_local_path = str(model_local_path)
        if native_gguf_qspec is not None:
            is_sharded = False
            model_save_name = gguf_checkpoint_path
        else:
            if format_code == FORMAT.EXL3:
                possible_model_basenames = ["model"]
            else:
                possible_model_basenames = [
                    f"gptq_model-{qcfg.bits}bit-{qcfg.group_size}g",
                    "model",
                ]

            extensions = [".safetensors"]

            # Retrieve (and if necessary download) the quantized checkpoint(s).
            is_sharded, resolved_archive_file, true_model_basename = get_checkpoints(
                model_id_or_path=model_local_path,
                extensions=extensions,
                possible_model_basenames=possible_model_basenames,
                **cached_file_kwargs,
            )

            # bin files have security issues: disable loading by default
            if ".bin" in resolved_archive_file:
                raise ValueError(
                    "Loading of .bin files are not allowed due to safety. Please convert your model to safetensor or pytorch format."
                )

            model_save_name = resolved_archive_file  # In case a model is sharded, this would be `model.safetensors.index.json` which may later break.

        qcfg.runtime_format = format_code

        # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
        gguf_tensor_key_mapping = None
        with suspend_hf_weight_init():
            _call_before_model_load(
                cls,
                model_local_path=model_local_path,
                load_quantized_model=True,
            )

            if config.architectures:
                model_class = getattr(transformers, config.architectures[0], None)
                if model_class is not None:
                    # backward-compatible fallback for "_supports_flash_attn" field
                    if hasattr(model_class, "_supports_flash_attn_2"):
                        supports_flash_attn = getattr(model_class, "_supports_flash_attn_2")
                    elif hasattr(model_class, "_supports_flash_attn"):
                        supports_flash_attn = getattr(model_class, "_supports_flash_attn")
                else:
                    supports_flash_attn = None
            else:
                supports_flash_attn = None

            args = {}
            if supports_flash_attn and selector_device_family(device) in [DEVICE.CUDA, DEVICE.ROCM]:
                if attn_implementation is not None:
                    args[ATTN_IMPLEMENTATION] = attn_implementation
                elif is_flash_attn_2_available():
                    args = {ATTN_IMPLEMENTATION: "flash_attention_2"}
                    log.info("Loader: Auto enabling flash attention2")
            set_dtype_compat(args, dtype)

            try:
                model = cls.loader.from_config(
                    config, trust_remote_code=trust_remote_code, **args
                )
            except FileNotFoundError as exc:
                # trust_remote_code dynamic-module caches can be incomplete for
                # legacy Deci files; rebuild missing shim + refresh once.
                missing_path = str(exc)
                if (
                    trust_remote_code
                    and "transformers_modules" in missing_path
                    and "No such file or directory" in missing_path
                ):
                    missing_file = None
                    if "'" in missing_path:
                        parts = missing_path.split("'")
                        if len(parts) >= 2:
                            missing_file = parts[1]

                    if (
                        missing_file
                        and missing_file.endswith("__configuration_llama.py")
                        and not os.path.exists(missing_file)
                    ):
                        os.makedirs(os.path.dirname(missing_file), exist_ok=True)
                        with open(missing_file, "w", encoding="utf-8") as fp:
                            fp.write("from transformers.models.llama.configuration_llama import *\n")

                    auto_map = getattr(config, "auto_map", None) or {}
                    class_ref = auto_map.get("AutoModelForCausalLM")
                    if isinstance(class_ref, str):
                        from transformers.dynamic_module_utils import get_class_from_dynamic_module
                        get_class_from_dynamic_module(
                            class_ref,
                            str(getattr(config, "_name_or_path", "")),
                            force_download=True,
                        )

                    model = cls.loader.from_config(
                        config, trust_remote_code=trust_remote_code, **args
                    )
                else:
                    raise
            # This path replaces the fused experts with quantized kernels and
            # fills them from the checkpoint, so keep Defuser's per-expert
            # copies off the host (issue #4400).
            _meta_fused_expert_params(model)
            _convert_model_with_defuser(cls, model, cleanup_original=True)
            # Defuser built the split experts on meta; give them real (but
            # uninitialized) host storage so the quantized kernels start on a
            # materialized device.
            _materialize_meta_expert_linears(model)
            model.checkpoint_file_name = model_save_name
            if native_gguf_qspec is not None:
                gguf_tensor_key_mapping = _build_gguf_tensor_key_mapping(model, config)

            dynamic_cache_stats_before = dynamic_override_cache_stats(qcfg.dynamic) if qcfg.dynamic else None
            _apply_requested_module_tree(
                cls, model, module_tree, qcfg, allow_auto=False
            )
            effective_module_tree = cls._resolve_effective_module_tree(model, qcfg)
            if effective_module_tree is None:
                raise ValueError(
                    f"Unsupported model_type {config.model_type}: no explicit module_tree is registered."
                )
            model.gptqmodel_effective_module_tree = copy.deepcopy(effective_module_tree)

            with cls._module_tree_context(effective_module_tree):
                extract_layers_node = cls.extract_layers_node()
                if qcfg.dynamic:
                    planning_layer_modules = cls.full_layer_modules(
                        model_config=config,
                        is_awq_quantize=qcfg.method == METHOD.AWQ,
                        include_capture_only=True,
                    )
                    configure_dynamic_override_cache_for_model(
                        model, qcfg,
                        layer_modules=planning_layer_modules,
                        layer_prefixes=extract_layers_node,
                    )
                    model._gptqmodel_dynamic_cache_prepared_for = weakref.ref(qcfg)
                    model._gptqmodel_dynamic_cache_stats_before = dynamic_cache_stats_before
                # Get the first layer to determine layer type
                layers, _ = get_layers_with_prefixes(model, extract_layers_node)

                modules = find_modules(model)
                ignore_modules = [cls.lm_head] + cls.get_base_modules(model)
                input_embeddings = model.get_input_embeddings()
                output_embeddings = model.get_output_embeddings()
                input_embed_name = get_module_name(model, input_embeddings) if input_embeddings is not None else None
                output_embed_name = get_module_name(model, output_embeddings) if output_embeddings is not None else None
                input_embed_quantized, output_embed_quantized = is_embeddings_module_quantized(
                    model_dir=model_local_path,
                    input_embed_name=input_embed_name,
                    output_embed_name=output_embed_name,
                )

                simple_layer_modules = cls.simple_layer_modules(config, qcfg)
            external_inclusions = qcfg.external_module_inclusions()
            for name in list(modules.keys()):
                if input_embed_quantized and name == input_embed_name:
                    continue
                if output_embed_quantized and name == output_embed_name:
                    continue

                # allow loading of quantized lm_head
                if qcfg.lm_head and name == cls.lm_head:
                    continue

                layer_prefix = next(
                    (f"{prefix}." for prefix in extract_layers_node if name.startswith(f"{prefix}.")),
                    None,
                )
                relative = name[len(layer_prefix):] if layer_prefix is not None else None
                if relative is not None:
                    layer_index, separator, remainder = relative.partition(".")
                    relative = remainder if separator and layer_index.isdigit() else relative
                declared = {
                    item.split(":", 1)[0]
                    for sublist in simple_layer_modules
                    for item in sublist
                }
                relative_match = (
                    relative is not None
                    and any(
                        relative == leaf
                        or (relative.startswith(f"{leaf}.") and EXPERT_INDEX_PLACEHOLDER in leaf)
                        for leaf in declared
                    )
                )
                inclusion_match = (
                    external_inclusions is None
                    or relative is None
                    or any(
                        relative.endswith(entry) or name.endswith(entry)
                        for entry in external_inclusions
                    )
                )
                if layer_prefix is None or any(
                        name == ignore_module or name.startswith(f"{ignore_module}.")
                        for ignore_module in ignore_modules
                ) or not relative_match or not inclusion_match:
                    # log non-lm-head quantized modules only
                    if name is not cls.lm_head:
                        log.info(f"The layer {name} is not quantized.")
                    del modules[name]
                    continue

                if not cls.should_quantize_module(model, name, modules[name], qcfg):
                    log.info(f"The layer {name} is not quantized.")
                    del modules[name]

            if resolve_model_rotation(qcfg):
                if format_code not in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
                    raise NotImplementedError(
                        f"`rotation` is only supported for GPTQ/GPTQ_V2 checkpoints, got `{format_code}`."
                    )
                backend = normalize_backend(backend, quant_method=qcfg.method)
                if backend == BACKEND.AUTO:
                    backend = BACKEND.GPTQ_TORCH
                if backend not in (
                    BACKEND.GPTQ_TORCH,
                    BACKEND.GPTQ_TRITON,
                    BACKEND.GPTQ_W4AFP8,
                    BACKEND.GPTQ_W4A_NVFP4,
                ):
                    raise NotImplementedError(
                        "`rotation` is only supported with `gptq_torch`, `gptq_triton`, "
                        f"`gptq_w4afp8`, or `gptq_w4a_nvfp4` backend, got `{backend}`."
                    )

            if format_code == FORMAT.EXL3:
                if not isinstance(qcfg.tensor_storage, dict) or not qcfg.tensor_storage:
                    raise ValueError("EXL3 checkpoints require `quantization_config.tensor_storage` metadata.")

                exl3_module_cls = (ExllamaV3TorchLinear if backend in (BACKEND.EXL3_TORCH, BACKEND.MLX)
                                   else ExllamaV3Linear)
                replace_exllamav3_placeholders(
                    model=model,
                    module_names=list(qcfg.tensor_storage.keys()),
                    tensor_storage=qcfg.tensor_storage,
                    module_cls=exl3_module_cls,
                )
                preload_qlinear_kernel = exl3_module_cls
            else:
                preload_qlinear_kernel = make_quant(
                    model,
                    qcfg=qcfg,
                    quant_result=modules,
                    backend=backend,
                    lm_head_name=cls.lm_head,
                    device=device,
                    dtype=dtype,
                    is_sharded=is_sharded,
                    cpu_offload=cpu_offload,
                )

        validate_checkpoint_qweights(
            model,
            model_save_name,
            format_code,
            key_converter=_checkpoint_key_converter(cls, model, to_runtime=False),
        )

        if isinstance(requested_device_map, str) and requested_device_map not in [
                "auto",
                "balanced",
                "balanced_low_0",
                "sequential",
            ]:
                raise ValueError(
                    "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                    "'sequential'."
                )


        def build_layerwise_device_map(
                model,
                device,
                layers: List[torch.nn.Module],
                ignore_modules: List[torch.nn.Module],
                num_gpus: Optional[int] = None,
                max_memory=None,
        ) -> Dict[str, str]:
            """
            Build a deterministic alternating device_map for multi-GPU loading.
            Designed for quantized GPTQ models.

            Rules:
              • Input embedding(s) → GPU 0
              • Each repeating layer → alternate across GPUs (round-robin)
              • Output head (lm_head / embed_out):
                  – If weight-tied with input embedding → GPU 0
                  – Else → last GPU
              • Ignore modules (e.g., norms, projections) → co-located with last layer’s GPU
            """

            if num_gpus is None:
                num_gpus = _layerwise_device_count(device, backend)
            if num_gpus < 1:
                family = selector_device_family(device) or DEVICE.CPU
                raise RuntimeError(f"No devices detected for accelerator family `{family.value}`")

            if max_memory is not None:
                # A memory cap names the devices that may be used. It replaces
                # the visible-device round-robin so an unlisted GPU is never
                # touched, and a `cpu` entry becomes the offload target.
                max_memory_budgets = _normalize_max_memory(max_memory, device)
                device_strs = [name for name, _ in max_memory_budgets]
                device_budgets: Optional[List[Optional[int]]] = [
                    budget for _, budget in max_memory_budgets
                ]
                num_gpus = len(device_strs)
            else:
                device_budgets = None
                device_strs = _layerwise_device_strings(device, num_gpus, backend)

            device_ids = list(range(num_gpus))
            device_map: Dict[str, str] = {}
            mod2name = {m: n for n, m in model.named_modules()}

            def assign(mod, device_id):
                if mod is None:
                    return
                name = mod2name.get(mod)
                if name is not None:
                    device_map[name] = device_strs[device_id]

            def default_fallback_device_id() -> int:
                if device_budgets is None:
                    return device_ids[-1]
                # A parameterless module (rotary embeddings, for example)
                # cannot carry an Accelerate CPU-offload hook: its
                # `pre_forward` reads `next(module.parameters())`. Keep the
                # structural modules on an accelerator when one is budgeted.
                return next(
                    (index for index, name in enumerate(device_strs) if name != "cpu"),
                    device_ids[-1],
                )

            # -------------------------------------------------------------
            # 1–3. Assign input embeddings, layers, and ignored modules
            # -------------------------------------------------------------
            # Input embeddings → GPU 0
            try:
                in_emb = model.get_input_embeddings()
            except NotImplementedError:
                log.warning("Model does not implement get_input_embeddings. Skipping input embeddings assignment.")
                in_emb = None

            # Alternating layers
            layer_name2devid: Dict[str, int] = {}
            in_emb_device_id: Optional[int] = None

            if device_budgets is not None:
                units = ([in_emb] if in_emb is not None else []) + list(layers)
                sizes = [_module_tensor_bytes(unit) for unit in units]
                labels = [mod2name.get(unit, "<unnamed>") for unit in units]
                placements = _assign_greedy_device_ids(sizes, device_budgets, labels)
                for unit, dev_id in zip(units, placements):
                    assign(unit, dev_id)
                    if unit is in_emb:
                        in_emb_device_id = dev_id
                    lname = mod2name.get(unit)
                    if lname is not None:
                        layer_name2devid[lname] = dev_id
            else:
                assign(in_emb, device_ids[0])
                if in_emb is not None:
                    in_emb_device_id = device_ids[0]
                for i, layer in enumerate(layers):
                    gpu = device_ids[i % num_gpus]
                    assign(layer, gpu)
                    lname = mod2name.get(layer)
                    if lname is not None:
                        layer_name2devid[lname] = gpu

            # Ignored modules - skip input embeddings to avoid overriding GPU 0 assignment
            # Iterate over modules that should be ignored during default layer-wise mapping
            for mod in ignore_modules:
                # Preserve GPU-0 placement for the input embedding module if it exists
                if in_emb is not None and mod is in_emb:
                    continue  # Skip input embedding to preserve GPU 0 assignment
                # Retrieve the module’s fully-qualified name
                name = mod2name.get(mod)
                if name is None:
                    continue
                # Walk up the module hierarchy to find the closest ancestor that already has a device assignment
                owner = name
                dev_id = None
                while owner:
                    if owner in layer_name2devid:
                        dev_id = layer_name2devid[owner]
                        break
                    if "." not in owner:
                        break
                    owner = owner.rsplit(".", 1)[0]
                # If no ancestor is found, fall back to the default device
                if dev_id is None:
                    dev_id = default_fallback_device_id()
                # Assign the current module to the determined device
                assign(mod, dev_id)
            # -------------------------------------------------------------
            # 4. Handle lm_head / output projection explicitly
            # -------------------------------------------------------------
            # Look for lm_head or similar projection
            head = getattr(model, "lm_head", None)
            if head is None:
                for cand in ("embed_out", "output_projection", "output_head"):
                    if hasattr(model, cand):
                        head = getattr(model, cand)
                        break

            if head is not None:
                tie_cfg = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
                if in_emb is not None and tie_cfg:
                    # A tied head is the same tensor as the input embedding, so
                    # it follows that placement instead of consuming budget.
                    assign(head, in_emb_device_id if in_emb_device_id is not None else device_ids[0])
                elif device_budgets is not None:
                    head_size = _module_tensor_bytes(head)
                    placement = _assign_greedy_device_ids(
                        [head_size], device_budgets, [mod2name.get(head, "<head>")]
                    )
                    assign(head, placement[0])
                else:
                    assign(head, device_ids[-1])

            # -------------------------------------------------------------
            # 5. Safety check: ensure all params are covered
            # -------------------------------------------------------------
            missing = [
                n for n, _ in chain(model.named_parameters(), model.named_buffers())
                if not any(n == k or n.startswith(k + ".") for k in device_map)
            ]
            module_names = set(mod2name.values())
            if missing:
                # map any leftover params (rare) to the default device
                fallback_device = default_fallback_device_id()
                for param_name in missing:
                    owner = param_name
                    while owner and owner not in module_names:
                        if "." not in owner:
                            owner = ""
                        else:
                            owner = owner.rsplit(".", 1)[0]
                    if owner:
                        device_map.setdefault(owner, device_strs[fallback_device])
                    else:
                        log.info(f"Loader: unable to map param '{param_name}' to a module; skipping fallback assignment.")

            # -------------------------------------------------------------
            # 6. Prune parent assignments that would override child devices
            # -------------------------------------------------------------
            for name, device_str in list(device_map.items()):
                if not name:
                    continue
                child_devices = {
                    device_map[child_name]
                    for child_name in device_map
                    if child_name != name and child_name.startswith(f"{name}.")
                }
                if child_devices and (len(child_devices) > 1 or device_str not in child_devices):
                    log.info(f"Loader: dropping parent '{name}' from device_map to preserve child placements.")
                    device_map.pop(name, None)

            # Collect parameters/buffers that were not assigned to any device in the current device_map
            missing_after_prune = [
                n for n, _ in chain(model.named_parameters(), model.named_buffers())
                if not any(n == k or n.startswith(k + ".") for k in device_map)
            ]
            # If any tensors remain unmapped, assign them to the last GPU as a fallback
            if missing_after_prune:
                fallback_device = default_fallback_device_id()
                for param_name in missing_after_prune:
                    # Walk up the module tree until we find a module name that exists in module_names
                    owner = param_name
                    while owner and owner not in module_names:
                        if "." not in owner:
                            owner = ""
                        else:
                            owner = owner.rsplit(".", 1)[0]
                    # Map the closest owning module to the fallback device
                    if owner:
                        device_map.setdefault(owner, device_strs[fallback_device])
                    else:
                        log.info(f"Loader: unable to map param '{param_name}' to a module; skipping fallback assignment.")
            # optional logging for debug
            log.info(f"Loader: Built map across {num_gpus} GPU(s), "
                  f"{len(device_map)} entries. First 8: {list(device_map.items())[:8]}")

            return device_map

        log.info(f"Loader: device = {device}")
        if explicit_device_map is None:
            if _native_ttnn_backend(backend):
                # The native TTNN kernel owns the chips through TT-Metal and
                # uploads a layer's packed weights on demand in its forward, so
                # its parameters stay on the host. A layer-wise ``xla:N`` map
                # would instead dispatch the checkpoint through PJRT, which
                # opens and claims every chip and therefore collides with the
                # mesh ``post_init`` opens (TT-Metal allows one owner per chip).
                # The collision surfaces as a TT_FATAL reading a stale device,
                # so keep the whole model on CPU for this backend.
                device_map = {"": "cpu"}
                log.info("Loader: native TTNN backend keeps packed weights on CPU until forward upload.")
            else:
                layers, _ = get_layers_with_prefixes(model, extract_layers_node)
                num_gpus = _layerwise_device_count(device, backend)
                device_map = build_layerwise_device_map(
                    model, device, layers, ignore_modules, num_gpus,
                    max_memory=max_memory,
                )
        else:
            device_map = dict(explicit_device_map)
            log.info(f"Loader: honoring explicit device_map request: {device_map}")
        original_device_map = dict(device_map)
        # Checkpoint loading needs a non-overlapping map: parent and child entries
        # would otherwise make Accelerate read the same PLE tensor on both devices.
        device_map = apply_no_placement_to_device_map(model, device_map)
        if device_map != original_device_map:
            cpu_modules = sorted(no_placement_module_names(model))
            log.info(f"Loader: keeping no-placement modules on CPU: {cpu_modules}")
        # Runtime dispatch keeps the parent entry so layer inputs still move to
        # the right GPU, while the explicit CPU leaf blocks recursive PLE moves.
        dispatch_device_map = dict(original_device_map)
        dispatch_device_map.update(dict.fromkeys(no_placement_module_names(model), "cpu"))
        log.info(f"Loader: device_map = {device_map}")

        # Multi-device loading moves shards between GPUs. Pre-enable peer access
        # once so the first transfer is already direct (PyTorch would otherwise
        # enable it lazily on that first copy). Best effort: never blocks the
        # load. Device maps may hold strings, ints or torch.device values.
        from ..utils.p2p import accelerator_devices_from_device_map, enable_for_devices

        map_devices = accelerator_devices_from_device_map(dispatch_device_map)
        if len(map_devices) > 1:
            enable_for_devices(map_devices)

        load_checkpoint_in_model = native_gguf_qspec is None
        load_extra_kwargs = {}
        if "offload_folder" in kwargs:
            load_extra_kwargs["offload_folder"] = kwargs["offload_folder"]
        elif (
            getattr(qcfg, "offload_to_disk_path", None) is not None
            and isinstance(device_map, dict)
            and "disk" in device_map.values()
        ):
            load_extra_kwargs["offload_folder"] = qcfg.offload_to_disk_path

        if "offload_state_dict" in kwargs:
            load_extra_kwargs["offload_state_dict"] = kwargs["offload_state_dict"]
        if "offload_buffers" in kwargs:
            load_extra_kwargs["offload_buffers"] = kwargs["offload_buffers"]

        # compat: runtime convert checkpoint gptq(v1) to gptq_v2 format
        if format_code in [FORMAT.GPTQ, FORMAT.GEMM, FORMAT.PAROQUANT]:
            load_checkpoint_in_model_then_tie_weights(
                model,
                dtype=dtype,
                # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
                checkpoint=model_save_name,
                device_map=device_map,
                key_converter=_checkpoint_key_converter(cls, model, to_runtime=True),
                **load_extra_kwargs,
            )

            load_checkpoint_in_model = False

            if format_code == FORMAT.GPTQ:
                # validate sym=False v1 loading needs to be protected for models produced with new v2 format codebase
                if not qcfg.sym and not qcfg.is_quantized_by_gptaq() and not qcfg.is_quantized_by_foem():
                    raise ValueError(
                        f"Format: Loading of a sym=False model with format={FORMAT.GPTQ} is only supported if produced by gptqmodel version >= {MIN_VERSION_WITH_V2}"
                    )

                if any(
                    isinstance(m, BaseQuantLinear) and getattr(m, "REQUIRES_FORMAT_V2", False)
                    for m in model.modules()
                ):
                    model = convert_gptq_v1_to_v2_format(
                        model,
                        cfg=qcfg,
                        qlinear_kernel=preload_qlinear_kernel,
                    )

                    qcfg.runtime_format = FORMAT.GPTQ_V2

        if backend == BACKEND.MACHETE:
            if is_sharded:
                raise ValueError(
                    "Format: The loading of sharded checkpoints with Machete is currently not supported."
                )

        if backend == BACKEND.MARLIN and (
                preload_qlinear_kernel == ExllamaV2Linear or format_code == FORMAT.MARLIN):
            if is_sharded:
                raise ValueError(
                    "Format: The loading of sharded checkpoints with Marlin is currently not supported."
                )
            # Both Marlin sub-kernels support fp16 and bf16 compute on Ampere+.
            if dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("Marlin kernel requires dtype=torch.float16 or dtype=torch.bfloat16.")

        if backend == BACKEND.SWORDFISH:
            if is_sharded:
                raise ValueError(
                    "Format: The loading of sharded checkpoints with Swordfish is currently not supported."
                )
            if dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("Swordfish kernel requires dtype=torch.float16 or dtype=torch.bfloat16.")

        if backend in [BACKEND.GPTQ_BITBLAS, BACKEND.AWQ_BITBLAS]:
            from ..utils.bitblas import prepare_model_for_bitblas_load

            # Prepare model for bitblas load.
            # If is bitblas serialized load then load directly. Otherwise, convert to bitblas.
            model = prepare_model_for_bitblas_load(
                model=model,
                qcfg=qcfg,
                quant_linear_class=preload_qlinear_kernel,
                dtype=dtype,
                model_save_name=model_save_name,
                device_map=device_map,
                desc_act=qcfg.desc_act,
                sym=qcfg.sym,
                load_checkpoint_in_model=load_checkpoint_in_model,
            )

        # If we use marlin or bitblas to load the quantized model, the model is already a converted model,
        # and we no longer need to call load_checkpoint_in_model()
        if load_checkpoint_in_model and backend not in [
            BACKEND.MACHETE,
            BACKEND.GPTQ_BITBLAS,
            BACKEND.AWQ_BITBLAS,
        ]:
            keep_in_fp32_modules = []
            fp32_bf16_buffer_names = []
            packed_fp8_buffer_dtypes = {}
            for name, module in model.named_modules():
                prefix = f"{name}." if name else ""
                if isinstance(module, TorchFP8Linear):
                    packed_fp8_buffer_dtypes[f"{prefix}weight"] = module.fp8_dtype
                if getattr(module, "weight_scale_inv", None) is not None:
                    scale_name = f"{prefix}weight_scale_inv"
                    keep_in_fp32_modules.append(scale_name)
                    fp32_bf16_buffer_names.append(scale_name)
                if getattr(module, "bias", None) is not None and (
                    getattr(module, "weight_scale_inv", None) is not None
                    or hasattr(module, "gguf_tensor_qtype")
                    or isinstance(module, (VPTQLinear, AllSparkLinear, AllSparkAwqLinear))
                ):
                    keep_in_fp32_modules.append(f"{prefix}bias")
                    is_gguf_linear = False
                    if hasattr(module, "gguf_tensor_qtype"):
                        from ..nn_modules.qlinear.gguf import GGUFTorchLinear

                        is_gguf_linear = isinstance(module, GGUFTorchLinear)
                    if is_gguf_linear or isinstance(module, (TorchFP8Linear, VPTQLinear,
                                                             AllSparkLinear, AllSparkAwqLinear)):
                        fp32_bf16_buffer_names.append(f"{prefix}bias")
            load_checkpoint_in_model_then_tie_weights(
                model,
                dtype=dtype,
                keep_in_fp32_modules=keep_in_fp32_modules or None,
                fp32_bf16_buffer_names=fp32_bf16_buffer_names,
                packed_fp8_buffer_dtypes=packed_fp8_buffer_dtypes,
                # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
                checkpoint=model_save_name,
                device_map=device_map,
                key_converter=_checkpoint_key_converter(cls, model, to_runtime=True),
                **load_extra_kwargs,
            )

            if format_code == FORMAT.GPTQ_V2:
                # Native v2 checkpoints store logical zero points; the v1->v2
                # conversion above only runs for v1 checkpoints. Mark the
                # loaded modules so decoders and repack bookkeeping read the
                # words as v2 instead of misreading every zero point one high.
                for m in model.modules():
                    if isinstance(m, GPTQQuantLinear):
                        m.qzero_format(format=2)

        if native_gguf_qspec is not None:
            model = simple_dispatch_model(model, dispatch_device_map)
            _load_quantized_gguf_checkpoint_into_model(
                model=model,
                gguf_checkpoint_path=gguf_checkpoint_path,
                tensor_key_mapping=gguf_tensor_key_mapping,
            )
        else:
            # TODO: Why are we using this custom function and not dispatch_model?
            model = simple_dispatch_model(model, dispatch_device_map)

        if format_code == FORMAT.EXL3:
            qlinear_kernel = (ExllamaV3TorchLinear if backend in (BACKEND.EXL3_TORCH, BACKEND.MLX)
                              else ExllamaV3Linear)
        else:
            qlinear_kernel = select_quant_linear(
                bits=qcfg.runtime_bits,
                dynamic=qcfg.dynamic,
                group_size=qcfg.group_size,
                desc_act=qcfg.desc_act,
                sym=qcfg.sym,
                backend=backend,
                format=format_code,
                quant_method=export_quant_method,
                device=device,
                pack_dtype=qcfg.pack_dtype,
                is_sharded=is_sharded,
                cpu_offload=cpu_offload,
            )

        # == step4: set seqlen == #
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions", "multimodal_max_length"]
        config_seq_len = find_config_seq_len(model_config, seq_len_keys)
        if config_seq_len is not None:
            model.seqlen = config_seq_len
        else:
            log.warn("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        if format_code != FORMAT.EXL3:
            # Any post-initialization that require device information, for example buffers initialization on device.
            model = gptqmodel_post_init(model, use_act_order=qcfg.desc_act, quantize_config=qcfg)

        model.eval()

        if backend == BACKEND.MLX:
            import tempfile
            try:
                from mlx_lm import load
                from mlx_lm.utils import save_config, save_model

                from ..utils.mlx import convert_gptq_to_mlx_weights, mlx_generate
            except ModuleNotFoundError as exception:
                raise type(exception)(
                    "GPT-QModel load mlx model required dependencies are not installed.",
                    "Please install via `pip install gptqmodel[mlx] --no-build-isolation`.",
                )

            with tempfile.TemporaryDirectory() as temp_dir:
                mlx_weights, mlx_config = convert_gptq_to_mlx_weights(model_local_path, model, qcfg.to_dict(), cls.lm_head)
                if (mlx_config.pop("_gptqmodel_group16_runtime", False)
                        or mlx_config.pop("_gptqmodel_custom_mlx_runtime", False)):
                    model = mlx_weights
                else:
                    save_model(temp_dir, mlx_weights, donate_model=True)
                    save_config(mlx_config, config_path=temp_dir + "/config.json")
                    tokenizer.save_pretrained(temp_dir)

                    model, _ = load(temp_dir)

        # Record the layout the tensors had on disk so a later
        # `model.save_pretrained()` can emit checkpoint-layout qzeros instead of
        # the kernel's converted runtime words.
        if hasattr(model, "modules"):
            for module in model.modules():
                if isinstance(module, GPTQQuantLinear):
                    module._checkpoint_format_on_disk = format_code

        instance = cls(
            model,
            quantized=True,
            fusion_enabled=fuse,
            quantize_config=qcfg,
            tokenizer=tokenizer,
            qlinear_kernel=qlinear_kernel,
            load_quantized_model=True,
            trust_remote_code=trust_remote_code,
            model_local_path=model_local_path,
            effective_module_tree=effective_module_tree,
        )
        if backend == BACKEND.MLX:
            instance._runtime_generate = partial(mlx_generate, tokenizer=tokenizer)
        _setup_rotation_online_had(
            instance.model,
            resolve_model_rotation(qcfg),
            module_plan=instance.module_plan(),
            layer_prefixes=instance.extract_layers_node(),
        )
        _set_paged_attention_safe_cuda_graphs(instance.model)
        if qcfg.dynamic:
            log_dynamic_override_cache_stats(instance._dynamic_cache_stats_before, dynamic=qcfg.dynamic)
        # Quantized loads keep their own scale dtypes, so the strict dtype
        # contract only applies to dense loads. A CPU-offloaded map may hold
        # meta placeholders by design, so only the default-dtype invariant is
        # asserted there.
        assert_loaded_model_state(
            model,
            api_name=f"{cls.__name__}.from_quantized",
            default_dtype_before=default_dtype_before,
            require_materialized=not cpu_offload,
        )
        return instance

    cls.from_quantized = classmethod(_serialized_model_load(from_quantized.__func__))

    return cls
