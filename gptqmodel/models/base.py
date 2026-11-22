# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import copy
import difflib
import inspect
import json
import os
import threading
import time
import types
import weakref
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from itertools import count
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union

import torch
import torch.nn as nn
from tokenicer import Tokenicer
from transformers import (  # Auto*: unsupported-model fallback only (see BaseQModel.loader)
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from .. import DEVICE_THREAD_POOL
from ..adapter.adapter import Adapter
from ..nn_modules.exllamav3 import ExllamaV3Linear
from ..nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear._compile_support import supports_compiled_generation
from ..nn_modules.qlinear._fused_train_state import ensure_cpu_inference_layout
from ..nn_modules.qlinear.fp4 import TorchFP4Linear
from ..nn_modules.qlinear.fp8 import TorchFP8Linear
from ..nn_modules.qlinear.lookahead import configure_default_lookahead
from ..nn_modules.qlinear.torch import TorchLinear
from ..quantization.awq.quantize.scale import is_foldable_source
from ..quantization.awq.utils.module import get_op_name
from ..quantization.config import (
    FORMAT,
    META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS,
    METHOD,
    QUANTIZE_BLACK_LIST,
    AutoDtypeCast,
    AutoModuleDecoderConfig,
    BaseQuantizeConfig,
    CalibrationConfig,
    GcMode,
    NormConfig,
    QuantizeEmbed,
    QuantizeEmbedConfig,
    VramStrategy,
    configure_dynamic_override_cache_for_model,
    dynamic_get,
    dynamic_override_cache_stats,
    encoder_stages_on_host,
    log_dynamic_override_cache_stats,
    normalize_norm_types,
    resolve_model_rotation,
    resolve_quant_format,
    validate_rotation_format,
)
from ..quantization.dtype import (
    available_float8_dtypes,
    decode_e8m0_scale,
    dequantize_f4_e2m1,
    dequantize_fp8,
    device_supports_dtype,
    device_supports_native_fp4,
    is_fp4_packed_dtype,
    quark_floatx_formats,
)
from ..quantization.norm_tuning import norm_type_of
from ..quantization.rotation.rotation import (
    fuse_layer_norms,
    rotate_model,
    validate_rotation_dimensions,
)
from ..utils.attn_mask import normalize_seq_mask
from ..utils.backend import (
    BACKEND,
    awq_gemm_pack_backend,
    awq_gemv_fast_pack_backend,
    awq_gemv_pack_backend,
    backend_for_activation,
    normalize_backend,
)
from ..utils.backend_config import Backend, resolve_quantize_backend
from ..utils.calibration import (
    AUTO_BATCH_SIZE,
    _chat_calibration_row_tokens,
    _columnar_mapping_examples,
    _frame_columns,
    _hf_dataset_rows,
    _is_columnar_mapping,
    _message_template,
    _messages_char_count,
    auto_batch_size,
    prepare_calibration_dataset,
)
from ..utils.checkpoint_paths import resolve_checkpoint_path
from ..utils.device import get_device
from ..utils.device_telemetry import emit_device_telemetry, with_quantization_device_telemetry
from ..utils.hf import (
    autofix_hf_model_config,
    get_config_selection_kwargs,
    load_model_config,
    resolve_checkpoint_subfolder,
    resolve_trust_remote_code,
)
from ..utils.importer import select_quant_linear
from ..utils.kv_cache import KVCacheEstimate, estimate_kv_cache
from ..utils.logger import QuantizationRegionTimer, format_quantize_summary, setup_logger
from ..utils.looper_helpers import normalize_device_like, select_forward_devices
from ..utils.model import (
    MODALITY,
    _module_has_meta_tensors,
    find_modules,
    fused_unit_check_module_names,
    get_module,
    get_module_by_name_prefix,
    get_module_name,
    has_cpu_offload_hooks,
    move_module_to_cpu_pinned,
    move_to,
    warn_fused_unit_dynamic_conflicts,
)
from ..utils.model_dequant import infer_block_shape
from ..utils.p2p import safe_module_to
from ..utils.quant_metadata import UnsupportedMetadata, describe_unsupported_metadata, find_unsupported_metadata
from ..utils.resource_budget import (
    DeviceBudget,
    ResourceBudget,
    RuntimeAllocationUnavailable,
    device_memory_bytes,
)
from ..utils.rocm import IS_ROCM, quant_blas_backend
from ..utils.structure import (
    LazyTurtle,
    _get_parent_and_leaf_by_path,
    _get_qualified_name,
    _resolve_qualified_name,
    alias_direct_meta_from_turtle_for_submodule,
    alias_from_turtle_for_submodule,
    restore_init_owned_rotary_buffers,
)
from ..utils.tied_weights import (
    lm_head_is_tied,
    lm_head_quantization_is_tied,
    word_embeddings_tied,
)
from ..utils.torch import (
    TORCH_HAS_COMPILE,
    as_torch_device,
    dynamo_is_compiling,
    dynamo_traced_frames,
    model_execution_device_scope,
    release_compile_trace_garbage,
    release_pinned_host_cache,
    torch_compile,
    torch_empty_cache,
)
from ._const import (
    CPU,
    DEFAULT_MAX_SHARD_SIZE,
    DEVICE,
    EXPERT_INDEX_PLACEHOLDER,
    META,
    normalize_device,
)
from .fusable import FusableConfig, require_fusable_config
from .loader import ModelLoader, _setup_rotation_online_had, model_load_lock
from .local import LocalModelLoader, local_model_loader_for_config
from .module_plan import ModulePlan
from .quantization_run_plan import ProcessorStep, QuantizationRunPlan, processor_class
from .shared_input import SharedInputPlan, build_shared_input_plan
from .writer import ModelWriter


if TYPE_CHECKING:
    from ..looper.checkpoint_store import CheckpointConfig
    from ..measure.config import MeasureConfig
    from ..measure.report import MeasurementReport
    from .fusion import FusionPlanner

    try:
        from datasets import Dataset as HFDatasetType
        from datasets import IterableDataset as HFIterableDatasetType
    except Exception:  # pragma: no cover - optional dependency
        HFDatasetType = HFIterableDatasetType = object

    from ..looper.named_module import NamedModule


# Processor steps whose quantization GEMMs must run on rocBLAS to be
# reproducible across processes (issues #4075, #4092, #4123, #4460).
#
# GPTQ, QQQ, VPTQ, MACaQ and EoRA accumulate an input
# Hessian/Gram from the calibration activations; EXL3 reuses GPTQ's capture.
# EoRA runs after the primary, so an AWQ + EoRA plan would otherwise keep its
# adapter Gram on hipBLASLt and produce a different adapter per process (#4614).
# Paro accumulates no Hessian, but its rotation optimization runs
# fp16/bf16 GEMMs and hipBLASLt selects kernels per process, so two runs with
# the same seed produced different checkpoints (21.26 vs 21.44 PPL on ROCm).
# Pinning rocBLAS makes those GEMMs bitwise reproducible at the cost of the
# faster hipBLASLt kernels. AWQ and weight-only (RTN) keep hipBLASLt.
_DETERMINISTIC_BLAS_PROCESSOR_STEPS = frozenset({
    ProcessorStep.GPTQ,
    ProcessorStep.QQQ,
    ProcessorStep.VPTQ,
    ProcessorStep.MACAQ,
    ProcessorStep.PARO,
    ProcessorStep.EXL3,
    ProcessorStep.EORA,
})


def _needs_deterministic_hessian(run_plan: QuantizationRunPlan) -> bool:
    """Whether a run's quantization GEMMs must be reproducible on ROCm.

    The name predates Paro: the original set accumulated an input
    Hessian/Gram. Paro is included because hipBLASLt chooses GEMM kernels
    per process, which made checkpoints from identical seeded runs differ.
    EoRA is included for its own fp32 input Gram and `eigh`; without it an
    AWQ + EoRA plan produced a different adapter in every process (#4614).
    """

    return not _DETERMINISTIC_BLAS_PROCESSOR_STEPS.isdisjoint(run_plan.processors)


# Model types whose definition class declares `supports_rotation`. This is the
# set named by the rotation gate's error message and documented in
# docs/rotation-model-support.md. tests/test_rotation_opt_in.py fails if it
# drifts from the declarations on the definition classes.
ROTATION_SUPPORTED_MODEL_TYPES = frozenset({"llama", "qwen2"})


def rotation_supported(model_def) -> bool:
    """Whether this exact definition class opted into model-wide rotation.

    The lookup is deliberately non-inheriting: ``LlamaQModel`` is the base of
    Gemma2/Gemma3/Qwen3/Olmo2 and of every other Llama-shaped definition, so an
    inherited opt-in would re-admit exactly the layouts the rotation math
    cannot preserve. A subclass must declare its own ``supports_rotation``.

    The declaration is a frozenset of verified ``model_type`` values. One class
    can serve several model types with different norm semantics (``LlamaQModel``
    covers ``llama``, ``mistral``, ``gemma``, ...), so the opt-in names the
    model types it covers instead of claiming the whole class.
    """

    declared = type(model_def).__dict__.get("supports_rotation", False)
    if not isinstance(declared, frozenset):
        return False
    config = getattr(getattr(model_def, "model", None), "config", None)
    return getattr(config, "model_type", None) in declared


def validate_rotation_model_support(model_def) -> None:
    """Fail before any weight is touched when rotation is not opted in."""

    if rotation_supported(model_def):
        return

    config = getattr(getattr(model_def, "model", None), "config", None)
    model_type = getattr(config, "model_type", None)
    supported = ", ".join(sorted(ROTATION_SUPPORTED_MODEL_TYPES))
    raise ValueError(
        f"rotation is not supported for model {type(model_def).__name__} "
        f"(model_type={model_type!r}); supported model types: {supported}. "
        "No weights were modified."
    )


def _torch_cuda_device(value: object) -> Optional[torch.device]:
    """Resolve one pool entry to a concrete CUDA device, or None."""

    try:
        if isinstance(value, torch.device):
            device = value
        elif hasattr(value, "to_torch_device"):
            device = value.to_torch_device()
        else:
            device = torch.device(str(value))
    except (TypeError, ValueError, RuntimeError):
        return None
    if device.type != "cuda":
        return None
    if device.index is None:
        try:
            return torch.device("cuda", torch.cuda.current_device())
        except Exception:  # pragma: no cover - driver/host edge cases
            return None
    return device


def _quantize_vram_devices(quantize_config: object = None) -> List[torch.device]:
    """Concrete CUDA devices whose peak counters describe this quantization run.

    `quantize_config.device` names an accelerator family only (ordinals are
    stripped by `normalize_device`), so the concrete devices are resolved from
    the same pool the loopers use: every visible device of that family, narrowed
    by `compute_device_filter` when the caller set one. A CPU run resolves to an
    empty list, so it must not reset or read any CUDA peak counter.
    """

    if quantize_config is None or not torch.cuda.is_available():
        return []
    raw_device = getattr(quantize_config, "device", None)
    if raw_device is None:
        # Auto selection on a CUDA host resolves to CUDA; keep reporting it.
        family_type = "cuda"
    else:
        try:
            family_type = getattr(normalize_device(raw_device), "type", None)
        except Exception:
            return []
    if family_type != "cuda":
        return []
    try:
        device_count = int(torch.cuda.device_count())
    except Exception:  # pragma: no cover - driver/host edge cases
        return []

    candidates: List[object] = [torch.device("cuda", index) for index in range(device_count)]
    compute_device_filter = getattr(quantize_config, "compute_device_filter", None)
    if compute_device_filter is not None:
        try:
            filtered = compute_device_filter(candidates)
        except Exception:
            filtered = None
        # The loopers keep the full pool when the filter rejects everything.
        if filtered:
            candidates = list(filtered)

    devices: List[torch.device] = []
    for candidate in candidates:
        device = _torch_cuda_device(candidate)
        if device is not None and device not in devices:
            devices.append(device)
    return devices


def _reset_peak_vram_stats(devices: Sequence[torch.device]) -> None:
    """Reset peak counters only for the devices this run quantizes on."""

    for device in devices:
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:  # pragma: no cover - driver/host edge cases
            continue


def _peak_vram_gib(
    quantize_config: object = None,
    devices: Optional[Sequence[torch.device]] = None,
) -> Optional[float]:
    """Largest per-device peak allocation across the GPUs this run quantized on.

    `torch.cuda.max_memory_allocated()` without a device reads the calling
    thread's current device, which is not necessarily the card a quantization
    worker used. Every tracked device is queried explicitly and the maximum is
    reported, so the summary never hides the busiest card. Returns None when the
    run did not quantize on CUDA.
    """

    if devices is None:
        devices = _quantize_vram_devices(quantize_config)
    if not devices:
        return None
    peaks: List[float] = []
    for device in devices:
        try:
            peaks.append(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        except Exception:  # pragma: no cover - driver/host edge cases
            continue
    return max(peaks) if peaks else None


def _quantize_summary_counts(result: object) -> object:
    """Normalize a `quantize()` result to the layer -> modules mapping the summary counts.

    The staged-GSQ helper returns `{'gsq_training': [block, ...]}`: one record
    per decoder layer, whose `stages` are that layer's projections. The generic
    formatter would count that single top-level key as one layer, so map every
    block to its own layer entry instead.
    """

    if not isinstance(result, dict) or set(result) != {"gsq_training"}:
        return result
    blocks = result["gsq_training"]
    if not isinstance(blocks, (list, tuple)):
        return result
    counts: Dict[str, Tuple[Any, ...]] = {}
    for position, block in enumerate(blocks):
        if isinstance(block, dict):
            stages = block.get("stages")
            index = block.get("layer_index", position)
        else:
            stages = None
            index = position
        counts[f"layer.{index}"] = tuple(stages or ())
    return counts


def _log_quantize_summary(
    result: object,
    *,
    run_started: float,
    vram_devices: Sequence[torch.device],
) -> None:
    """Log the end-of-run summary for a run that finished its finalization."""

    log.info(
        format_quantize_summary(
            _quantize_summary_counts(result),
            elapsed_s=time.perf_counter() - run_started,
            peak_vram_gib=_peak_vram_gib(devices=vram_devices),
        )
    )




def _device_index(tensor: object) -> int:
    device = getattr(tensor, "device", None)
    return device.index if getattr(device, "type", None) == "cuda" and device.index is not None else -1


def _runtime_weight_tensors(inner: object):
    """Yield weight tensors a backend keeps outside the parameter registry.

    The hook is looked up on the class, not the instance: a bound-method
    lookup for every module in a large model costs more than the walk itself,
    and this runs on the resource-reporting path.
    """

    modules = getattr(inner, "modules", None)
    if not callable(modules):
        return
    try:
        walker = modules()
    except Exception:  # pragma: no cover - exotic module trees
        return
    for module in walker:
        hook = getattr(type(module), "iter_runtime_weight_tensors", None)
        if hook is None:
            continue
        try:
            yield from hook(module)
        except Exception:  # pragma: no cover - backend bookkeeping
            continue


def _runtime_device_allocations(inner: object) -> Tuple[Dict[Tuple[str, int], int], Tuple[str, ...]]:
    """(measured) Accelerator weight storage a backend keeps outside Torch.

    A native runtime can hold packed weights in its own allocator: TTNN keeps
    them as ``ttnn.Tensor`` device tensors, which neither ``parameters()`` nor
    ``iter_runtime_weight_tensors`` can see (#5810). Each module publishes them
    through ``iter_runtime_device_allocations``; the entry names the backend and
    that backend's own device ordinal, so TTNN chips never collide with CUDA
    devices or the host.

    A backend that owns storage but cannot measure it is returned as unmeasured,
    never as zero: an unreadable allocator is unknown memory, not free memory.
    """

    allocations: Dict[Tuple[str, int], int] = {}
    unmeasured = []
    modules = getattr(inner, "modules", None)
    if not callable(modules):
        return allocations, ()
    try:
        walker = modules()
    except Exception:  # pragma: no cover - exotic module trees
        return allocations, ()
    for module in walker:
        hook = getattr(type(module), "iter_runtime_device_allocations", None)
        if hook is None:
            continue
        try:
            entries = tuple(hook(module))
        except RuntimeAllocationUnavailable as exc:
            unmeasured.append(exc.backend)
            continue
        except Exception:  # pragma: no cover - backend bookkeeping
            unmeasured.append("unknown")
            continue
        for entry in entries:
            backend = getattr(entry, "backend", None)
            index = getattr(entry, "index", None)
            nbytes = getattr(entry, "weights_bytes", None)
            if (
                not isinstance(backend, str) or not backend
                or not isinstance(index, int) or index < 0
                or not isinstance(nbytes, int) or nbytes < 0
            ):
                # A malformed entry is coverage we cannot trust; say so rather
                # than silently dropping the storage it was meant to describe.
                unmeasured.append(backend if isinstance(backend, str) and backend else "unknown")
                continue
            key = (backend, index)
            allocations[key] = allocations.get(key, 0) + nbytes
    return allocations, tuple(dict.fromkeys(unmeasured))


def _parameter_devices(inner: object) -> Dict[int, int]:
    """(measured) Weight bytes per device index, meta tensors skipped.

    Parameters and buffers whose storage does not live on a CUDA device are
    reported under ``-1`` so a CPU or meta-only model still has a measured
    weight total.

    Quantized kernels can move their weight out of the parameter/buffer
    registry into a derived packed cache (QQQ's AVX512 operands, #4856). Each
    module is asked for those tensors too; they are counted by storage so an
    alias of a registered tensor or of another cache entry is counted once
    (#5810).
    """

    per_device: Dict[int, int] = {}
    try:
        tensors = list(inner.parameters()) + list(inner.buffers())
    except Exception:  # pragma: no cover - exotic module trees
        return per_device
    seen_storages = set()
    for tensor in tensors:
        if getattr(tensor, "is_meta", False):
            continue
        index = _device_index(tensor)
        per_device[index] = per_device.get(index, 0) + tensor.numel() * tensor.element_size()
        try:
            seen_storages.add((index, tensor.untyped_storage().data_ptr()))
        except Exception:  # pragma: no cover - non-torch stand-ins
            pass
    for tensor in _runtime_weight_tensors(inner):
        if getattr(tensor, "is_meta", False):
            continue
        index = _device_index(tensor)
        try:
            storage = tensor.untyped_storage()
            key = (index, storage.data_ptr())
            nbytes = storage.nbytes()
        except Exception:  # pragma: no cover - non-torch stand-ins
            continue
        # A view adds nothing of its own: the storage it aliases is already
        # resident under its source tensor, so count that storage once.
        if nbytes == 0 or key in seen_storages:
            continue
        seen_storages.add(key)
        per_device[index] = per_device.get(index, 0) + nbytes
    return per_device


def _device_budgets(per_device: Dict[int, int]) -> Tuple[DeviceBudget, ...]:
    """(measured) One DeviceBudget per device that holds weights."""

    budgets = []
    for index, weights in sorted(per_device.items()):
        free, total = (None, None) if index < 0 else device_memory_bytes(index)
        budgets.append(DeviceBudget(index=index, free_bytes=free, total_bytes=total, weights_bytes=weights))
    return tuple(budgets)


def _layer_device_indices(wrapper: object, inner: object,
                          layer_count: int) -> Optional[Tuple[int, ...]]:
    """(measured) Device index of each decoder layer, or None when unresolvable.

    The layer paths live on the quantization wrapper (`BaseQModel`), while the
    modules themselves live on the HF model it wraps - the split that made an
    earlier revision return None for every real model.
    """

    extract = getattr(wrapper, "extract_layers_node", None)
    if not callable(extract):
        return None
    try:
        paths = list(extract())
    except Exception:  # pragma: no cover - exotic definitions
        return None
    if not paths:
        return None

    # A definition may name the same decoder stack more than once: VL wrappers
    # expose both ``model.language_model.layers`` and ``language_model.layers``,
    # which resolve to one ModuleList. Accept every resolvable path only when
    # they name the same modules; distinct stacks (HRM's L/H modules) stay
    # unresolved instead of being guessed.
    stacks = []
    for path in paths:
        for candidate in (inner, getattr(inner, "model", None)):
            if candidate is None or not hasattr(candidate, "get_submodule"):
                continue
            try:
                stacks.append(candidate.get_submodule(path))
                break
            except Exception:
                continue
    if not stacks:
        return None
    layers = stacks[0]
    for other in stacks[1:]:
        try:
            same = len(other) == len(layers) and all(a is b for a, b in zip(layers, other))
        except TypeError:  # a path that names a module rather than a stack
            same = False
        if not same:
            return None

    indices = []
    for layer in layers:
        tensors = list(layer.parameters()) + list(layer.buffers())
        device = None
        for tensor in tensors:
            if getattr(tensor, "is_meta", False):
                continue
            device = getattr(tensor, "device", None)
            break
        index = device.index if getattr(device, "type", None) == "cuda" and device.index is not None else -1
        indices.append(index)
    if len(indices) != layer_count:
        return None
    return tuple(indices)


def _budget_for(inner: object, *, wrapper: object = None, reserve: float = 0.1,
                dtype: Optional[torch.dtype] = None) -> ResourceBudget:
    """Resource budget for a decoder module: measured per-device numbers + estimates."""

    per_device = _parameter_devices(inner)
    allocations, unmeasured_backends = _runtime_device_allocations(inner)
    estimate = _kv_cache_estimate_for(inner, dtype=dtype)
    layer_devices = None
    if estimate is not None:
        layer_devices = _layer_device_indices(wrapper if wrapper is not None else inner,
                                              inner, len(estimate.layers))
    devices = list(_device_budgets(per_device))
    for (backend, index), weights in sorted(allocations.items()):
        devices.append(
            DeviceBudget(index=index, weights_bytes=weights, backend=backend)
        )
    return ResourceBudget(
        devices=tuple(devices),
        kv_cache=estimate,
        layer_devices=layer_devices,
        reserve_fraction=reserve,
        unmeasured_backends=unmeasured_backends,
    )


def _kv_cache_estimate_for(inner: object, *, dtype: Optional[torch.dtype] = None) -> Optional[KVCacheEstimate]:
    """KV-cache estimate for a decoder module, dtype taken from its weights."""

    return estimate_kv_cache(
        getattr(inner, "config", None),
        dtype=dtype or _module_weight_dtype(inner),
    )


def _calibration_rows(calibration) -> List[Any]:
    """The rows the quantizer will see, normalized like the looper does.

    ``batch_size="auto"`` used to read the raw ``calibration.dataset``, so the
    estimate depended on the container: a ``DataFrame`` iterated its column
    labels, a columnar mapping counted its columns as rows, a ``[1, S]`` tensor
    row reported ``shape[-2] == 1`` and a bare tensor row raised from
    ``Tensor.values()`` (#5505). Normalizing through the helpers
    ``prepare_calibration_dataset`` already uses keeps the estimate on the rows
    that are actually calibrated.
    """

    dataset = calibration.dataset
    columns = _frame_columns(dataset)
    if columns is not None:
        return _columnar_mapping_examples(columns)
    if isinstance(dataset, Mapping):
        if _is_columnar_mapping(dataset):
            return _columnar_mapping_examples(dataset)
        return [dataset]
    hf_rows = _hf_dataset_rows(dataset)
    if hf_rows is not None:
        return hf_rows
    try:
        iterator = iter(dataset)
    except TypeError:
        return [dataset]
    if iterator is dataset:
        # Sizing must leave one-shot inputs for calibration's first read.
        return []
    return list(iterator)


def _sequence_length(value: Any) -> int:
    """Length of the last sequence dimension of a token-id container, else 0."""

    if torch.is_tensor(value):
        return int(value.shape[-1]) if value.dim() else 0
    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        head = value[0]
        if isinstance(head, (list, tuple)) or torch.is_tensor(head):
            return _sequence_length(value[-1])
        return len(value)
    return 0


def _count_calibration_tokens(qmodel, text: Any) -> int:
    """Tokenizer length of one text row, 0 when it cannot be measured."""

    tokenizer = getattr(qmodel, "tokenizer", None)
    if tokenizer is None or not isinstance(text, str):
        return 0
    try:
        return len(tokenizer(text)["input_ids"])
    except Exception:  # pragma: no cover - exotic tokenizers
        return 0


def _calibration_row_count(calibration) -> int:
    """Rows the calibration set will produce, or a conservative estimate."""

    try:
        return max(1, len(_calibration_rows(calibration)))
    except Exception:
        return 1


def _calibration_tokens_per_row(qmodel, calibration) -> int:
    """Longest calibration row in tokens: declared concat size or measured."""

    if calibration.concat_size:
        return int(calibration.concat_size)

    try:
        rows = _calibration_rows(calibration)
    except Exception:
        return 512

    # A `messages` row only yields tokens after the chat template runs, so the
    # estimator could not measure it and kept the 512 default (#5568). Render
    # the longest few through the same path preparation uses.
    chat_template = _message_template(getattr(qmodel, "tokenizer", None))[0] is not None

    longest = 0
    texts: List[str] = []
    chat_rows: List[Any] = []
    for row in rows:
        ids = row.get("input_ids") if isinstance(row, Mapping) else row
        length = _sequence_length(ids) if ids is not None else 0
        if length:
            longest = max(longest, length)
            continue
        messages = row.get("messages") if isinstance(row, Mapping) else None
        if messages is not None and chat_template:
            chat_rows.append(row)
            continue
        text = row if isinstance(row, str) else (
            row.get("text") if isinstance(row, Mapping) else None
        )
        if isinstance(text, str):
            texts.append(text)
    if chat_rows:
        chat_rows.sort(key=_messages_char_count, reverse=True)
        longest = max(
            longest,
            max((_chat_calibration_row_tokens(qmodel, row) for row in chat_rows[:4]), default=0),
        )
    if texts:
        # Tokenizing every row would repeat the whole calibration pass. The
        # longest few by character count bound the padded row far better than
        # the first four rows, which is all the old estimate ever looked at.
        texts.sort(key=len, reverse=True)
        longest = max(
            longest,
            max((_count_calibration_tokens(qmodel, text) for text in texts[:4]), default=0),
        )
    return longest or 512


def _materialized_calibration_config(calibration):
    """Return a calibration config whose one-shot inputs are materialised once.

    ``batch_size="auto"`` sizes the run by reading the calibration rows before
    ``prepare_calibration_dataset`` runs, and ``quantize()`` may size the native
    and EoRA passes separately. A generator (or ``iter(list)``/``map``) is
    exhausted by the first read, so the later preparation saw no rows and
    failed with "Quantize: calibration dataset is empty" (#5567). Materialise
    the inputs up front so every read and every processor sees the same rows;
    sized containers and ``datasets.IterableDataset`` are returned unchanged.
    """

    from dataclasses import replace

    from ..looper.loop_processor import _reusable_calibration_dataset

    dataset = _reusable_calibration_dataset(calibration.dataset)
    adapter_dataset = _reusable_calibration_dataset(calibration.adapter_dataset)
    if dataset is calibration.dataset and adapter_dataset is calibration.adapter_dataset:
        return calibration

    bound = replace(calibration, dataset=dataset, adapter_dataset=adapter_dataset)
    if calibration.prepare_dataset is not None:
        # ``replace()`` drops the non-init field, and a bound config must keep
        # the preparer that its owner installed.
        object.__setattr__(bound, "prepare_dataset", calibration.prepare_dataset)
    return bound


def _module_weight_dtype(module: object) -> Optional[torch.dtype]:
    """dtype of the first floating parameter of a module, or None."""

    try:
        for parameter in module.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
    except Exception:  # pragma: no cover - exotic module trees
        return None
    return None


def _cast_floating_inputs(example: Dict[str, Any], dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    """Cast floating rows to ``dtype``; integers keep their dtype.

    Only used for definitions that declare ``calibration_inputs = "tensor_kwargs"``,
    where the rows are the user's own tensors rather than token ids.
    """

    if dtype is None:
        return example
    for key, value in example.items():
        if isinstance(value, list):
            example[key] = [
                item.to(dtype) if torch.is_tensor(item) and item.is_floating_point() else item
                for item in value
            ]
        elif torch.is_tensor(value) and value.is_floating_point() and value.dtype != dtype:
            example[key] = value.to(dtype)
    return example


class _CompiledModelForward:
    """Call a compiled unbound ``forward`` without pinning the model (#4937).

    ``torch.compile(model.forward)`` stores the bound method inside the
    compiled wrapper, so installing that wrapper as an instance attribute
    closes a ``model -> forward -> model`` cycle: the model, and its VRAM,
    then survives until a full cyclic GC. Compiling the class function and
    resolving the instance through a weak reference keeps the compiled fast
    path cycle-free, like ``TorchLinear.optimize()`` does for
    ``dequantize_weight`` (#4871).
    """

    __slots__ = ("_compiled", "_unbound", "_module_ref", "_traced_frames")

    def __init__(self, compiled, unbound, module):
        self._compiled = compiled
        self._unbound = unbound
        self._module_ref = weakref.ref(module)
        # Frames dynamo had already traced when this proxy was installed. Only a
        # call that adds frames produces new trace garbage (#5020).
        self._traced_frames = dynamo_traced_frames()

    def __call__(self, *args, **kwargs):
        module = self._module_ref()
        if module is None:
            raise RuntimeError(
                "compiled model `forward` called after its model was released"
            )
        try:
            # Inductor's lazy pattern init uses the *current* device when a
            # graph-break frame has no device-bearing input, which claimed
            # cuda:0 for a model compiled on cuda:1 (#5101). Enter the model's
            # execution device for the compiled call; the caller's device is
            # restored on exit, including on error.
            with model_execution_device_scope(module, args, kwargs):
                return self._compiled(module, *args, **kwargs)
        finally:
            self._release_trace_garbage()

    def _release_trace_garbage(self) -> None:
        """Collect dynamo's trace cycles once per newly traced frame.

        The proxy itself holds no strong reference to the model, but dynamo
        leaves the trace's ``VariableTracker`` objects in reference cycles that
        do, and reference counting cannot reclaim them. Collecting here, while
        the caller still owns the model, leaves the model reachable only from
        that owner, so dropping it frees its VRAM immediately instead of
        waiting for a user-issued cyclic ``gc.collect()`` (#5020). A call that
        reused the compiled cache adds no frames and needs no collection.
        """

        if dynamo_is_compiling():
            # An outer ``torch.compile`` (transformers' static-cache generate)
            # is tracing this call. ``gc.collect`` is untraceable, so running it
            # here is a graph break per model call (#5224), and collecting while
            # dynamo still holds the trace's trackers is the wrong time. Leave
            # the count unchanged so the next eager call releases the garbage.
            return
        # Read the counter only after the trace check: dynamo traces this
        # method when an outer compile calls through the proxy, and a traced
        # read of ``counters["frames"]`` installs a guard on dynamo's global
        # frame counter. Every later compile bumps that counter, so the guard
        # failed on the next call and every static-cache ``generate()``
        # re-traced forever (#5266).
        traced = dynamo_traced_frames()
        if traced == self._traced_frames:
            return
        self._traced_frames = traced
        release_compile_trace_garbage()

    @property
    def _torchdynamo_orig_callable(self):
        """The eager bound ``forward`` this proxy wraps.

        Returned on demand, never stored: a stored bound method would recreate
        the cycle. ``optimize()`` recompiles ``type(model).forward`` instead,
        so this is only what an unwrapping caller sees.
        """

        module = self._module_ref()
        if module is None:
            return self._unbound
        return self._unbound.__get__(module, type(module))

    def __repr__(self):
        return f"<compiled model forward {self._compiled!r}>"


class _QuantizedCheckpointSource:
    """Expose an on-disk quantized checkpoint through the shard-map interface.

    Module-replacement saves use this lightweight source when a model was loaded
    through the regular quantized loader and therefore has no LazyTurtle.
    """

    def __init__(self, model_local_path: str, checkpoint_file_name: Optional[str] = None):
        from safetensors import safe_open

        # The loader has already selected the basename (including legacy GPTQ
        # names) and whether it is sharded. Do not rediscover a different file.
        if checkpoint_file_name is None:
            checkpoint_file_name = os.path.join(model_local_path, "model.safetensors.index.json")
            if not os.path.exists(checkpoint_file_name):
                checkpoint_file_name = os.path.join(model_local_path, "model.safetensors")
        checkpoint_file_name = os.path.abspath(checkpoint_file_name)
        self.model_local_path = os.path.dirname(checkpoint_file_name)
        self._index_filename = None
        if checkpoint_file_name.endswith(".safetensors.index.json"):
            with open(checkpoint_file_name, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map", {})
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Checkpoint index at `{checkpoint_file_name}` has an empty or invalid `weight_map`.")
            self._weight_map = {
                str(tensor_name): str(shard_name)
                for tensor_name, shard_name in weight_map.items()
            }
            # Validate shard headers before any tuning; this maps files but does
            # not load checkpoint tensors into CPU/GPU memory.
            for shard_name in set(self._weight_map.values()):
                # A `weight_map` entry is untrusted: an absolute or `..` shard
                # name would later be joined to the output directory. Validate
                # it before the header probe so the source path is contained
                # lexically; HF blob symlinks must keep resolving.
                shard_path = resolve_checkpoint_path(
                    self.model_local_path,
                    shard_name,
                    kind="checkpoint weight_map shard",
                    follow_symlinks=False,
                )
                with safe_open(shard_path, framework="pt", device="cpu"):
                    pass
            self._index_filename = os.path.basename(checkpoint_file_name)
            return

        if not checkpoint_file_name.endswith(".safetensors"):
            raise NotImplementedError("Module replacement save currently supports safetensors checkpoints only.")
        with safe_open(checkpoint_file_name, framework="pt", device="cpu") as handler:
            self._weight_map = dict.fromkeys(handler.keys(), os.path.basename(checkpoint_file_name))


class _ClassPropertyDescriptor:
    def __init__(self, fget, fset=None):
        self.fget = fget

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return self.fget.__get__(instance, owner)()


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)


def generate_node_for_awq_scaling(inp, prev_op, module_kwargs, nodes_size, subset, module2inspect):
    n = {
        "prev_op": prev_op,
        "layers": subset,
        "inp": inp,
    }
    if module_kwargs is not None:
        # Preserve per-node kwargs for every scaling group. In multi-batch AWQ
        # replays these can differ by feature bucket, so falling back to a
        # layer-global "latest batch" mask on later nodes can reintroduce
        # sequence-length mismatches during scale search.
        n["kwargs"] = module_kwargs

    if module2inspect is not None:
        n["module2inspect"] = module2inspect

    return n, None

def apply_module_tree_override(module_tree, override):
    """
    Recursively find the corresponding key of override in ``module_tree``.

    The model definition is shared by every instance, so overrides must never
    mutate it in place.  Always copy first; callers may then freeze the result
    for safe use during a quantization run.
    """
    module_tree = copy.deepcopy(module_tree)
    if isinstance(module_tree, dict) and isinstance(override, dict):
        for k, v in override.items():
            if k in module_tree and isinstance(module_tree[k], (dict, list)) and isinstance(v, (dict, list)):
                module_tree[k] = _apply_module_tree_override_in_place(module_tree[k], v)
            else:
                module_tree[k] = copy.deepcopy(v)
    elif isinstance(module_tree, list) and isinstance(override, list):
        for o in override:
            if isinstance(o, dict):
                for b in module_tree:
                    if isinstance(b, dict):
                        _apply_module_tree_override_in_place(b, o)
    return module_tree


def _apply_module_tree_override_in_place(module_tree, override):
    """Apply an override to an already-owned mutable tree copy."""
    if isinstance(module_tree, dict) and isinstance(override, dict):
        for key, value in override.items():
            if key in module_tree and isinstance(module_tree[key], (dict, list)) and isinstance(value, (dict, list)):
                _apply_module_tree_override_in_place(module_tree[key], value)
            else:
                module_tree[key] = copy.deepcopy(value)
    elif isinstance(module_tree, list) and isinstance(override, list):
        for item in override:
            if isinstance(item, dict):
                for branch in module_tree:
                    if isinstance(branch, dict):
                        _apply_module_tree_override_in_place(branch, item)
    return module_tree


NOT_QUANTIZE_FLAG = ":!"
CAPTURE_ONLY_FLAG = ":?"
# Declares that a leaf's output is not the direct input of the following scaling
# consumer, so it must never be used as an AWQ `prev_op`. Used by definitions
# whose forward graph has a residual add, a non-homogeneous activation, or a
# second consumer between the leaf and the quantized block (see
# docs/awq-scaling-groups.md).
AWQ_NO_SCALE_SOURCE_FLAG = "nosrc"
MOE_FLAG = ":moe"
NORM_FLAG = ":norm"
# Marks a normalization that consumes the block output (a "sandwich" post-norm).
# It still owns checkpoint weights and may be trained, but it never feeds the
# next quantized projection, so AWQ must not fold the consumer's input scale
# into it.
POST_NORM_FLAG = ":postnorm"
NON_QUANTIZE_FLAGS = (NOT_QUANTIZE_FLAG, CAPTURE_ONLY_FLAG)


log = setup_logger()


# Methods whose packing kernel may be accelerator-only. The quantize loop packs
# every layer and ``post_quantize`` leaves it on CPU, so an in-session forward
# must use a kernel that supports the model's final device (issue #4381).
_IN_SESSION_KERNEL_METHODS = frozenset({METHOD.AWQ, METHOD.QQQ, METHOD.GGUF})


def _kernel_supports_device(kernel, device) -> bool:
    """Whether a quant-linear class declares support for ``device``."""

    supported = getattr(kernel, "SUPPORTS_DEVICES", None)
    if not supported:
        return True
    family = normalize_device(device)
    return DEVICE.ALL in supported or family in supported


def select_in_session_quant_linear(qcfg, device):
    """Select a quant-linear kernel that can execute on the final device.

    ``quantize()`` selects its packing kernel from ``quantize_config.device``,
    which may be an accelerator while the finished layers live on CPU. Resolve
    the same method/format for the device the model actually occupies instead
    of reusing the calibration-time choice.
    """

    method = qcfg.export_quant_method()
    return select_quant_linear(
        bits=qcfg.runtime_bits,
        dynamic=qcfg.dynamic,
        group_size=qcfg.group_size,
        desc_act=qcfg.desc_act,
        sym=qcfg.sym,
        backend=BACKEND.AUTO,
        format=resolve_quant_format(qcfg.format, method),
        quant_method=method,
        device=normalize_device(device),
        pack=True,
        pack_dtype=qcfg.pack_dtype,
    )


def _switch_quant_linear_class(module: BaseQuantLinear, target_cls) -> None:
    """Reinterpret one packed module with another layout-compatible kernel.

    The portable Torch kernels read the same packed buffers as their
    accelerator-only counterparts (AWQ GEMM, QQQ, GGUF). Swapping the class
    keeps the packed state and hooks intact while changing only the compute
    path, which is exactly the kernel switch the in-session model needs.
    """

    module.__class__ = target_cls
    # An accelerate offload hook captures the previous class's bound ``forward``
    # in ``_old_forward`` and calls it from its wrapper, so a class swap alone
    # would keep executing the accelerator-only kernel (#4947). Rebind the
    # captured forward to the target class.
    if getattr(module, "_hf_hook", None) is not None and hasattr(module, "_old_forward"):
        module._old_forward = types.MethodType(target_cls.forward, module)
    # ``__class__`` assignment never runs the target kernel's ``__init__``, so a
    # kernel that keeps runtime state on the instance has to install it here
    # (#4940; this path previously broke on #4451 and #4856).
    init_runtime_state = getattr(target_cls, "_init_runtime_state", None)
    if callable(init_runtime_state):
        init_runtime_state(module)
    backends = getattr(target_cls, "SUPPORTS_BACKENDS", None) or ()
    if backends:
        module.backend = backends[0]
    if not hasattr(module, "compute_dtype"):
        scales = getattr(module, "scales", None)
        module.compute_dtype = (
            scales.dtype
            if isinstance(scales, torch.Tensor) and scales.dtype in (torch.float16, torch.bfloat16)
            else torch.float16
        )
    clear = getattr(module, "clear_weight_cache", None)
    if callable(clear):
        clear()
    post_init = getattr(module, "post_init", None)
    if callable(post_init):
        post_init()


def rehome_quantized_kernel(model, device=None) -> None:
    """Make the in-session quantized layers runnable on their final device.

    The quantize loop deliberately leaves packed layers on CPU. For AWQ, QQQ
    and GGUF the calibration-time kernel can be CUDA-only, so the first
    in-session forward would fail. Switch those modules to the portable Torch
    kernel for the final device, or raise the reload-the-checkpoint guidance
    when no layout-compatible kernel exists.

    ``device`` overrides the probe of the packed modules' runtime device. A
    cancelled/failed run keeps its finished layers disk-offloaded as meta
    shells, so the caller passes the device the rest of the model occupies
    (#4947).
    """

    qcfg = getattr(model, "quantize_config", None)
    if qcfg is None or qcfg.export_quant_method() not in _IN_SESSION_KERNEL_METHODS:
        return

    modules = [
        module
        for module in model.model.modules()
        if isinstance(module, BaseQuantLinear)
    ]
    if not modules:
        return

    if device is None:
        device = next(
            (
                runtime_device
                for module in modules
                if (runtime_device := module.runtime_device()) is not None
                and runtime_device.type != "meta"
            ),
            None,
        )
    if device is None:
        return

    current = getattr(model, "qlinear_kernel", None)
    unsupported = [module for module in modules if not _kernel_supports_device(type(module), device)]
    if not unsupported:
        if current is not None and not _kernel_supports_device(current, device):
            model.qlinear_kernel = select_in_session_quant_linear(qcfg, device)
        return

    selection_error = None
    try:
        target = select_in_session_quant_linear(qcfg, device)
    except Exception as exc:  # pragma: no cover - defensive
        target = None
        selection_error = exc

    if target is not None:
        for module in unsupported:
            # Only reinterpret kernels this run actually produced. A dynamic
            # per-module format may have selected a different class; leaving it
            # untouched lets the final check raise the actionable error below.
            if current is not None and not isinstance(module, current):
                continue
            _switch_quant_linear_class(module, target)

    remaining = [module for module in modules if not _kernel_supports_device(type(module), device)]
    if remaining:
        kernels = ", ".join(sorted({type(module).__name__ for module in remaining}))
        raise RuntimeError(
            "Quantize: quantized layers use kernels that cannot run on the final "
            f"device `{device}` ({kernels}). The in-session model cannot run "
            "inference; save the quantized model and reload the saved checkpoint "
            "on this device (or use the portable Torch backend) before using it."
        ) from selection_error

    model.qlinear_kernel = target


# Class methods are intentionally kept as the public model-definition API.  At
# runtime, however, an instance can have a module tree selected by a quantizer
# method override or by auto-detection.  This context lets those class methods
# retain class-level semantics when called on a class while using the selected
# instance tree when called through an instance.
_ACTIVE_MODULE_TREE = contextvars.ContextVar("gptqmodel_active_module_tree", default=None)


class _FrozenDict(dict):
    """A dict-compatible, deeply immutable mapping used by effective trees."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("effective_module_tree is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable

    def __reduce__(self):
        # A reducer argument cannot be ``self``: deepcopy/pickle would try to
        # serialize this same frozen mapping again before calling the reducer.
        return (_freeze_module_tree, (_thaw_module_tree(self),))


def _freeze_module_tree(value):
    # Convert every mutable container, not just the outer list, so no caller
    # can change dispatch by mutating a nested branch after initialization.
    if isinstance(value, dict):
        return _FrozenDict((key, _freeze_module_tree(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_module_tree(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_module_tree(item) for item in value)
    return value


def _thaw_module_tree(value):
    if isinstance(value, dict):
        return {key: _thaw_module_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_module_tree(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {_thaw_module_tree(item) for item in value}
    return value


def _module_tree_for_class(cls):
    active = _ACTIVE_MODULE_TREE.get()
    return active[1] if active is not None and active[0] is cls else cls.module_tree


def _module_owns_direct_meta_tensors(module: nn.Module) -> bool:
    """Whether ``module`` itself (not a child) owns a tensor still on ``meta``."""

    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        if getattr(tensor, "is_meta", False):
            return True
        device = getattr(tensor, "device", None)
        if device is not None and getattr(device, "type", None) == "meta":
            return True
    return False


# Replay metadata key: the per-layer-type `attention_mask` / `position_embeddings`
# captured from the first decoder layer's forward. `capture_first_layer_input_kwargs`
# stores it in the cached batch kwargs and `prepare_layer_replay_kwargs` consumes it,
# selecting the entry for the layer being replayed and dropping the key again.
_LAYER_TYPE_REPLAY_KWARGS_KEY = "_gptqmodel_layer_type_replay_kwargs"

# `transformers` names the windowed attention layer type this string; every other
# declared type (`full_attention`, `hybrid`, ...) uses the plain causal mask.
_LAYER_TYPE_SLIDING = "sliding_attention"

# Returned when a declared layer type has no single mask builder (for example
# the composite `hybrid` pattern). Replay then keeps the captured layer-0 mask
# instead of guessing one.
_LAYER_TYPE_MASK_UNSUPPORTED = object()


def _capture_accepts_position_ids(model: object) -> bool:
    """Whether ``model.forward`` can take ``position_ids``.

    The derived ids are only passed to a model that declares the parameter (or
    swallows ``**kwargs``); the keyword would otherwise raise ``TypeError`` at
    capture time.
    """

    try:
        parameters = inspect.signature(model.forward).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return "position_ids" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _model_computes_own_position_ids(model: object) -> bool:
    """Whether ``model`` builds its own multimodal (mRoPE) position ids.

    Qwen3-VL/Qwen3.5/GLM-4V/Hunyuan-VL-class models call ``get_rope_index`` or
    ``compute_3d_position_ids`` from ``forward`` only when ``position_ids`` is
    None. Handing them the mask-derived 2-D text ids replaces the t/h/w grid
    ids, so image tokens are calibrated at sequential positions (#4933). The
    capability lives on the inner multimodal model, not the outer conditional
    generation wrapper, so inspect the module tree.
    """

    if not isinstance(model, nn.Module):
        return False
    for module in model.modules():
        if callable(getattr(module, "compute_3d_position_ids", None)):
            return True
        if callable(getattr(module, "get_rope_index", None)):
            return True
        if callable(getattr(module, "compute_mrope_position_ids", None)):
            return True
    return False


def _with_capture_position_ids(
    model: object,
    example: Dict[str, Any],
    *,
    derive: Optional[bool] = None,
) -> Dict[str, Any]:
    """Derive ``position_ids`` from the attention mask for a padded batch.

    A left-padded calibration batch otherwise embeds every real token at
    ``pad_len + i`` on models with learned absolute position embeddings
    (GPT-2, GPT-Neo), so every captured statistic comes from hidden states the
    model never produces at inference. ``generate()`` derives the same ids, and
    they equal the default ``arange`` for an unpadded row, so this is a no-op
    there (issue #4861).

    ``derive=None`` auto-detects the model: an mRoPE vision-language model must
    keep ``position_ids`` unset so its own ``get_rope_index`` /
    ``compute_3d_position_ids`` build the image/video grid ids (#4933).
    """

    if "position_ids" in example:
        return example
    mask = example.get("attention_mask")
    if not torch.is_tensor(mask) or mask.ndim != 2 or mask.numel() == 0:
        return example
    if not _capture_accepts_position_ids(model):
        return example
    if derive is None:
        derive = not _model_computes_own_position_ids(model)
    if not derive:
        return example
    # Exactly what `generate()` does: running positions for real tokens, 1 for
    # pad slots (attention masks those out).
    position_ids = mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 1)
    derived = dict(example)
    derived["position_ids"] = position_ids
    return derived


def _already_prepared_calibration_dataset(calibration_dataset, **_kwargs):
    """Return calibration batches that were prepared before the model mutation.

    ``quantize()`` prepares a rotation run's calibration input before it fuses
    the layer norms and rotates the weights (#4935). The loop processors call
    ``CalibrationConfig.prepare_dataset`` in their constructor, so a rotation
    run binds this no-op preparer to hand them the exact batches already
    validated, instead of tokenizing the corpus a second time.
    """

    return calibration_dataset


class BaseQModel(nn.Module):
    # Class-method APIs remain available to model definitions.  Calls through
    # an instance are wrapped in ``__getattribute__`` below so they resolve
    # against that instance's immutable effective tree.
    _MODULE_TREE_RUNTIME_METHODS = frozenset({
        # MoE roots that exist only in an effective tree must still capture
        # their shared pre-router input for the gate/up AWQ group.
        "awq_input_feature_aggregation",
        "extract_layers_node",
        "_iter_module_tree_variants",
        "_expand_module_tree_prefixes",
        "_collect_moe_modules_from_tree",
        "get_moe_modules",
        "is_moe_module",
        "get_moe_module_name",
        "get_norm_modules",
        "get_post_norm_modules",
        "build_layer_modules",
        "simple_layer_modules",
        "shared_input_plan",
        "full_layer_modules",
        "get_base_modules",
        "build_moe_modules_if_need",
    })

    # name of lm_head
    lm_head: str = "lm_head"

    # Special rotary_emb path
    rotary_embedding: str | None = None

    # a tree node of all the roots that contain quantizable modules
    module_tree: List[str] = None
    # Override module_tree according to different QUANT_METHOD
    module_tree_overrides: dict[METHOD, List[str]] = None

    # No profile means this definition cannot opt into projection fusion.
    fusable: Optional[FusableConfig] = None

    # `model_type`s whose `:in=<tag>` shared-input metadata was verified against a real (tiny, CPU)
    # forward in tests/module_tree/test_shared_input_cpu_forward.py. Not inherited: every concrete
    # definition (and every extra model_type mapped onto it) must list itself or its `:in=` tags
    # are ignored and Hessian dedup stays off. See `shared_input_verified()`.
    shared_input_verified_model_types: frozenset[str] = frozenset()

    # Strict=True -> all layer_modules must exists in model
    # Some models (deepseek2-lite) dynamically create lora modules based on config.rank
    layer_modules_strict = True

    pre_lm_head_norm_module: Union[str, List[str], None] = None

    # awq scaling optimizations requires some modules within same subset to strictly match the shape of previous module
    # list modules where they must match the shape of previous module in execution to consider for scaling optimization
    awq_scale_optimize_shape_dependent_modules: List[str] = None

    # Preserve adapter-supplied position_embeddings during AWQ replay instead of rebuilding them from the root rotary.
    awq_preserve_explicit_position_embeddings = False

    # Some definitions require an explicit request to execute checkpoint Python.
    require_trust_remote_code = None

    # some models require extra python packages and/or specific version of pkgs such as transformer version(internalm require '<=4.42.2')
    require_pkgs: Optional[List[str]] = None

    # some models require a specific dtype, such as float16
    require_dtype: Optional[str|torch.dtype] = None
    require_fast_init: bool = True

    # some models require Processor? For example, Qwen2VLImageProcessor.
    require_load_processor = False

    # Some checkpoints ship no tokenizer at all (Decision Transformer offline-RL
    # models). A definition that sets this to False is loaded without one; the
    # text models keep the default so a missing tokenizer still fails loudly.
    require_load_tokenizer: bool = True

    # How prepared calibration rows reach the model forward:
    #   "token_ids"     - normalise every row to `input_ids`/`attention_mask`
    #                     and collate by token length (default, text models).
    #   "tensor_kwargs" - pass a row whose values are all tensors straight
    #                     through as keyword arguments (Decision Transformer's
    #                     `states`/`actions`/`returns_to_go`/`timesteps`).
    calibration_inputs: str = "token_ids"

    # TODO: use a better name and what if the value is not at the config root?
    # allow dynamic expert n-count layer extraction
    # so moe model defs do not need to write out 64 layers if expert size is 64 (Qwen2Moe)
    # usage: set to property in model.config that holds this int value: total number of experts
    dynamic_expert_index: Optional[str] = None

    # Optional per-module expert cardinalities for architectures that contain
    # more than one expert family in a decoder layer.  Keys are layer-relative
    # prefixes (for example ``mlp.experts``) and values are config attributes.
    # Entries not matching a key continue to use ``dynamic_expert_index``.
    dynamic_expert_indices: Optional[Dict[str, str]] = None

    # Policy exception: only a model with no MODEL_MAP definition reaches this
    # default; it runs Transformers' own (or checkpoint) code, never a port.
    loader = AutoModelForCausalLM

    # monkey patch api for trust_remote_code=True models that have broken transformer compat
    require_monkeypatch = False

    # Dense-pool strategy support list
    supported_dense_vram_strategies: List[VramStrategy] = [
        VramStrategy.EXCLUSIVE,
        VramStrategy.BALANCED,
    ]

    # MoE expert-pool strategy support list
    supported_moe_vram_strategies: List[VramStrategy] = [
        VramStrategy.EXCLUSIVE,
        VramStrategy.BALANCED,
    ]

    # some models have broken attention mask codes so we need to only use batch 1 with no masks
    support_batch_quantize = True

    # Whether this model should publish a layer's KV tuple into the shared
    # replay cache even when that same layer does not consume `kv_last_layer`.
    # Models like "hymba" need some layers to write KV for later layers even if
    # the current layer itself does not consume `kv_last_layer`.
    write_shared_kv_cache = False

    # allow models to define optional notes that output messages to users that want to use this model
    # list of supported keys: [ "notes" = print the notes value on model load ]
    info: Dict[str, str] = {}

    # Some models have optional layers that are not loaded or supported by HF so even when they exist in the original
    # model, they are not properly saved on save(). GLM 4.5/4.6 (air) with MTP layers is such example.
    # Provide either a safetensors filename (the file is copied through if present),
    # a prefix (all `prefix.` tensors), a suffix, or an exact tensor name. Selected
    # tensors are merged into the main state dict so they end up in model.safetensors.
    out_of_model_tensors: Optional[Dict[str, Union[str | List[str]]]] = None

    supports_desc_act = [True, False]

    modality: List[MODALITY] = [MODALITY.TEXT]

    quant_override_files: Dict[str, Union[str | Dict[str, Any]]] = {}

    # Module paths that own meta params/buffers directly, not under child modules.
    modules_with_direct_meta_tensors: List[str] = []
    direct_meta_modules: List[str] = []  # Backward-compatible alias.

    server = None

    support_offload_to_disk = True
    # Optional runtime->checkpoint overrides for LazyTurtle. Prefer reversed
    # `WeightRenaming` entries; legacy runtime->checkpoint dicts are still accepted.
    HF_CONVERSION_MAP_REVERSED: Optional[Any] = None

    ATTENTION_MASKS_DTYPE = torch.bool # default to bool

    ATTENTION_MASKS_REQUIRED_FOR_INPUT: bool = False

    INPUT_EMBEDDING_EXTRA_ARGS = None

    # `run_input_capture` derives 2-D position_ids from the attention mask for
    # a padded batch (#4861). Leave None to auto-detect: a model that computes
    # its own multimodal (mRoPE) ids must keep `position_ids` unset (#4933).
    # Set explicitly only for a model whose capability the auto-detection
    # cannot see.
    CAPTURE_DERIVE_POSITION_IDS: Optional[bool] = None

    # Some models (e.g. ovis2_6_moe) do not contain MoE layers directly.
    # The actual experts live inside submodules (e.g. Qwen3MoeModel.mlp.experts),
    # so `defuser_module_paths` is used to explicitly locate and defuse them.
    defuser_module_paths = None

    # Some encoder/decoder checkpoints tie toward the checkpoint-owned decoder.
    # Their encoder capture path must materialize without immediately re-tying.
    turtle_materialize_tie_weights = True

    # Multimodal wrappers can reuse the checkpoint rules of their text model.
    hf_conversion_model_type_alias: Optional[str] = None

    def __init__(
        self,
        model: PreTrainedModel,
        quantized: bool,
        quantize_config: Optional[BaseQuantizeConfig],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        qlinear_kernel: nn.Module = None,
        load_quantized_model: bool = False,
        trust_remote_code: bool = False,
        model_local_path: str = None,
        # Lazy turtle is the checkpoint-backed source used to materialize shell modules on demand.
        turtle_model: Optional[LazyTurtle] = None,
        effective_module_tree: Optional[Any] = None,
        fusion_enabled: bool = False,
        processor_load_kwargs: Optional[Dict[str, Any]] = None,
        config_load_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if type(fusion_enabled) is not bool:
            raise TypeError("Fusion enabled must be bool")
        if fusion_enabled:
            require_fusable_config(type(self), getattr(model.config, "model_type", None))
        self._auto_fusion = fusion_enabled
        self._fusion_plan = None

        # Resolve per-instance module structure without ever changing the
        # declaration owned by the model definition class.  ``module_tree`` is
        # retained on the instance as a compatibility alias; class access is
        # still the static declaration.
        effective_tree = (
            copy.deepcopy(effective_module_tree)
            if effective_module_tree is not None
            else type(self)._resolve_effective_module_tree(model, quantize_config, detector=self)
        )

        # If module_tree is still None after auto-detection, raise an error indicating unsupported model type
        if effective_tree is None:
            raise ValueError(
                f"Unsupported model_type {model.config.model_type}: no explicit module_tree is registered. "
                "Add a model definition instead of relying on name-based auto-detection."
            )

        # Keep the selected tree stable for the lifetime of this instance;
        # class definitions remain reusable by later instances.
        self.effective_module_tree = _freeze_module_tree(effective_tree)
        self.module_tree = self.effective_module_tree
        self._module_plan = ModulePlan.from_tree(self.effective_module_tree)


        # record configuration early so model lifecycle hooks can rely on them
        self.compiled = False  # set to True while compile() is triggered successfully
        self.quantized = quantized
        self.load_quantized_model = load_quantized_model
        self.qlinear_kernel = qlinear_kernel
        self.quantize_backend: Optional[BACKEND] = None
        self.trust_remote_code = trust_remote_code
        # Wrapper-owned files live beside the selected weights. The save-time
        # config lookup still needs the repository root and its selection kwargs.
        self._checkpoint_repo_path = model_local_path
        self._config_load_kwargs = copy.deepcopy(config_load_kwargs or {})
        subfolder = self._config_load_kwargs.get("subfolder")
        self.model_local_path = (
            resolve_checkpoint_subfolder(model_local_path, subfolder)
            if model_local_path is not None and subfolder
            else model_local_path
        )
        self.quantize_config = quantize_config
        self.quantization_run_plan: Optional[QuantizationRunPlan] = None
        self.quant_region_timer = QuantizationRegionTimer(logger=log)
        self._runtime_generate = None
        # Populated by `serve()`; `serve_shutdown()`/`serve_wait_until_ready()` read it.
        self.server = None

        self.processor: ProcessorMixin = None

        from ..utils.mimo import is_mimo_mixed_source

        # Saving replaces MiMo metadata; decoding still needs the source layout.
        if is_mimo_mixed_source(getattr(model, "config", None)):
            self._source_model_config = copy.deepcopy(model.config)
        dynamic_cache_prepared = (
            bool(getattr(quantize_config, "dynamic", None))
            and getattr(model, "_gptqmodel_dynamic_cache_prepared_for", lambda: None)() is quantize_config
        )
        self._dynamic_cache_stats_before = (
            getattr(model, "_gptqmodel_dynamic_cache_stats_before")
            if dynamic_cache_prepared else dynamic_override_cache_stats(getattr(quantize_config, "dynamic", None))
        )
        self.model = self.after_model_load(model, load_quantized_model=load_quantized_model)
        # Hooks Transformers' ``from_pretrained(device_map=...)`` attached are
        # stock Accelerate hooks that flush the device cache on every offloaded
        # forward. Rebind them on this model instance only.
        from ..utils.offload_hooks import adopt_align_device_hooks

        adopt_align_device_hooks(self.model)
        # transformers 5 materializes checkpoint-absent non-persistent buffers
        # with `torch.empty_like`, and only re-derives native rotary buffers in
        # the default `_init_weights`. Restore constructor-owned rotary buffers
        # here so every load path (native eval, quantization, quantized reload)
        # sees real RoPE frequencies instead of uninitialized memory.
        restored_rotary_buffers = restore_init_owned_rotary_buffers(self.model)
        if restored_rotary_buffers:
            log.debug(
                "Model: restored %d init-owned rotary buffer(s) for `%s`.",
                restored_rotary_buffers,
                type(self.model).__name__,
            )
        hooks = getattr(self, "moe_lifecycle_hooks", None)
        if hooks is not None:
            hooks = copy.copy(hooks)
            self.moe_lifecycle_hooks = hooks
            hooks.module_plan = self.module_plan()
        if (getattr(quantize_config, "dynamic", None)
                and hasattr(self.model, "named_modules")
                and (not dynamic_cache_prepared or self.model is not model
                     or type(self).after_model_load is not BaseQModel.after_model_load)):
            planning_layer_modules = self.full_layer_modules(
                model_config=self.model.config,
                is_awq_quantize=quantize_config.method == METHOD.AWQ,
                include_capture_only=True,
            )
            configure_dynamic_override_cache_for_model(
                self.model, quantize_config,
                layer_modules=planning_layer_modules,
                layer_prefixes=self.extract_layers_node(),
            )
        self.turtle_model = turtle_model
        # Captures forward-role auto-decoder choices for regression tests and debug logs.
        self.auto_module_decoder_events: List[Dict[str, Any]] = []
        self.auto_dtype_cast_events: List[Dict[str, Any]] = []

        if tokenizer is not None:
            if isinstance(tokenizer, Tokenicer):
                self.tokenizer = tokenizer
            elif isinstance(tokenizer, PreTrainedTokenizerBase):
                self.tokenizer = Tokenicer.load(
                    tokenizer,
                    trust_remote_code=trust_remote_code,
                    model_config=getattr(self.model, "config", None),
                )
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")
            self.model.tokenizer = self.tokenizer.tokenizer # helpful for CI tests
        else:
            self.tokenizer = tokenizer # TODO none?
            self.model.tokenizer = tokenizer # helpful for CI tests # TODO none?

        # auto-fix model config erors
        if isinstance(self.model, PreTrainedModel):
            autofix_hf_model_config(self.model, path=self.model_local_path)
        # Reject activation-quantized checkpoints at load time so the rest of
        # the floatx decoder stack can continue assuming dense activations.
        self._configure_modelopt_runtime()

        self._turtle_lock = threading.RLock()

        # compat: state to assist in checkpoint_format gptq(v1) to gptq_v2 conversion
        # stores all per-layer quant stats such as avg loss and processing time
        self.quant_log = []

        self._processor_load_kwargs = get_config_selection_kwargs(processor_load_kwargs or {})
        if self.require_load_processor:
            loader = type(self).loader
            if isinstance(loader, LocalModelLoader) and loader.processor_class is not None:
                self.processor = loader.load_processor(
                    model_local_path,
                    config=self.model.config,
                    tokenizer=getattr(self.tokenizer, "tokenizer", self.tokenizer),
                    **self._processor_load_kwargs,
                )
            else:
                self.processor = AutoProcessor.from_pretrained(
                    model_local_path, trust_remote_code=self.trust_remote_code,
                    **self._processor_load_kwargs,
                )

        # apply patching of broken trust_remote_code models here
        if self.require_monkeypatch:
            self.monkey_patch()

        # hack: circular import
        from ..adapter.adapter import Lora

        # check adapter load and print info so users knows lora(s) are applied
        if quantize_config and isinstance(self.quantize_config.adapter, Lora):
            loaded_loras = 0
            qmodules = find_modules(self.model, layers=[BaseQuantLinear])
            for name, m in qmodules.items():
                if all(hasattr(m.adapter, name) for name in Lora.parameter_keys()):
                    loaded_loras += 1

            log.info(f"Adapter: `{loaded_loras}` EoRA/Lora adapters loaded for `{len(qmodules)}` modules.")

        # print kernel info:
        log.info(f"Kernel: loaded -> `[{', '.join(cls.__name__ for cls in self.kernels())}]`")

        self._auto_configure_lookahead()
        if self.quantized and self._auto_fusion:
            self.fuse()

    @classmethod
    def _model_module_tree(cls, model, quantize_config, detector=None):
        """Return the declared tree variant that matches this model's config.

        Model definitions override this to pick a layout variant (for example
        Falcon's two-norm parallel decoder). ``module_tree`` stays the default
        declaration other code reads; the selected tree still goes through
        ``module_tree_overrides`` in ``_resolve_effective_module_tree``.
        """

        return cls.module_tree

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        """Resolve a model tree before a loader needs to plan against it."""
        quant_method = getattr(quantize_config, "method", None) if quantize_config else None
        effective_tree = copy.deepcopy(getattr(model, "gptqmodel_effective_module_tree", None))
        if effective_tree is None:
            effective_tree = copy.deepcopy(
                cls._model_module_tree(model, quantize_config, detector=detector)
            )
        if quant_method is not None:
            override = (getattr(cls, "module_tree_overrides", None) or {}).get(quant_method)
            if override is not None and effective_tree is not None:
                log.info(f"Module Tree: overridden by METHOD.{quant_method.upper()}")
                effective_tree = apply_module_tree_override(effective_tree, override)

        return copy.deepcopy(effective_tree)

    @classmethod
    @contextmanager
    def _module_tree_context(cls, module_tree):
        """Temporarily bind a loader's effective tree to class-level planners."""
        token = _ACTIVE_MODULE_TREE.set((cls, module_tree))
        try:
            yield
        finally:
            _ACTIVE_MODULE_TREE.reset(token)

    def __getattribute__(self, name):
        """Make tree-sensitive class methods instance-aware without changing API."""
        attr = super().__getattribute__(name)
        runtime_names = type(self).__dict__.get("_MODULE_TREE_RUNTIME_METHODS")
        if runtime_names is None:
            runtime_names = getattr(type(self), "_MODULE_TREE_RUNTIME_METHODS", ())
        if name not in runtime_names:
            return attr
        try:
            tree = object.__getattribute__(self, "effective_module_tree")
        except AttributeError:
            return attr
        if tree is None or getattr(attr, "__gptqmodel_tree_bound__", False):
            return attr

        def invoke_with_effective_tree(*args, **kwargs):
            # Class methods read this context while they run, including nested
            # calls, then restore the previous value for the caller.
            token = _ACTIVE_MODULE_TREE.set((type(self), tree))
            try:
                return attr(*args, **kwargs)
            finally:
                _ACTIVE_MODULE_TREE.reset(token)

        invoke_with_effective_tree.__gptqmodel_tree_bound__ = True
        return invoke_with_effective_tree

    def module_plan(self) -> ModulePlan:
        """Return the immutable module plan compiled for this model instance."""

        try:
            plan = object.__getattribute__(self, "_module_plan")
        except AttributeError:
            plan = None
        if not isinstance(plan, ModulePlan):
            raise RuntimeError(
                f"{type(self).__name__}: module plan is not compiled yet; "
                "construct the quantized model before inspecting it"
            )
        return plan

    def fuse(self, enabled: bool = True, *, overrides: Optional[Dict[str, bool]] = None) -> Optional[FusionPlanner]:
        """Apply optional QKV/gate-up fusion and return its FusionPlanner.

        Fusion is disabled by default. Enabling it rebinds checkpoint operands
        to shared packed storage views. ``enabled=False`` restores independent
        contiguous operands and ordinary execution, returning None. Inspect the
        returned plan with its print(), render(), and to_dict() methods.
        Definition and per-call overrides can disable a family (``qkv``/``gate_up``)
        or exact parent path.
        The definition must declare a matching ``FusableConfig``; otherwise
        enabling fusion raises. Disabling never requires a profile.
        """
        if type(enabled) is not bool:
            raise TypeError("Fusion enabled must be bool")
        if enabled and not self.quantized:
            raise RuntimeError("Fusion requires a post-quantized model")
        if enabled:
            require_fusable_config(type(self), getattr(self.model.config, "model_type", None))
        from .fusion import FusionPlanner

        previous = self._fusion_plan
        # Build before changing installed execution so invalid requests leave
        # the previous plan in place. Temporarily unwrap for hook inspection.
        if previous is not None:
            previous.remove()
        try:
            plan = FusionPlanner(self, overrides=overrides) if enabled else None
            if plan is not None:
                plan.apply()
        except Exception:
            if previous is not None:
                previous.apply()
            raise
        self._fusion_plan = plan
        self._auto_fusion = enabled
        return plan

    @classmethod
    def extract_layers_node(cls):
        """
        Given a module_tree structure, return the layers_node string.
        It returns layer paths from each complete module_tree variant.
        Example:
            ["model", "layers", "#", {...}] -> ["model.layers"]
            [
                ["model", "L_module", "layers", "#", {...}],
                ["model", "H_module", "layers", "#", {...}],
            ]
                -> ["model.L_module.layers", "model.H_module.layers"]
        """
        paths = []
        for tree in cls._iter_module_tree_variants():
            for path in cls._expand_module_tree_prefixes(tree):
                if path not in paths:
                    paths.append(path)
        return paths

    @classmethod
    def _iter_module_tree_variants(cls, module_tree=None) -> List[List[Any]]:
        """Normalize module_tree into one or more complete tree variants."""

        tree = _module_tree_for_class(cls) if module_tree is None else module_tree
        if not isinstance(tree, (list, tuple)):
            return []

        if tree and all(isinstance(item, (list, tuple)) for item in tree):
            return [list(item) for item in tree]

        return [list(tree)]

    @classmethod
    def _expand_module_tree_prefixes(cls, tree=None) -> List[str]:
        """Return the module_tree prefix before `#` as a concrete path."""

        tree = _module_tree_for_class(cls) if tree is None else tree
        if tree is None:
            return []

        path = []
        for node in tree:
            if node == "#":
                break
            if not isinstance(node, str):
                break

            path.append(node.split(":", 1)[0])

        return [".".join(path)] if path else []

    @classmethod
    def _parse_module_aliases(cls, module_spec: str) -> List[str]:
        """
        Parse a module specification into its ordered runtime/checkpoint aliases.

        The first alias is the runtime shell name. Any later aliases are
        alternate checkpoint names declared directly in the model definition.
        """
        parts = module_spec.split(":") if isinstance(module_spec, str) else []
        name = parts[0] if parts else module_spec
        if not isinstance(name, str):
            return [name]
        aliases = [alias for alias in name.split("|") if alias]
        return aliases or [name]

    @classmethod
    def _parse_module_flags(cls, module_spec: str) -> tuple[str, List[str]]:
        """
        Parse a module specification into module name and flags.
        Example: "gate:moe:!" -> ("gate", ["moe", "!"])
        """
        parts = module_spec.split(":") if isinstance(module_spec, str) else []
        aliases = cls._parse_module_aliases(module_spec) if isinstance(module_spec, str) else [module_spec]
        name = aliases[0] if aliases else module_spec
        flags = [p for p in parts[1:] if p]
        return name, flags

    @classmethod
    def has_moe_flag(cls, module_spec: str) -> bool:
        """
        Check if a module specification has the :moe flag.
        """
        if not isinstance(module_spec, str):
            return False
        _, flags = cls._parse_module_flags(module_spec)
        return MOE_FLAG.lstrip(":") in flags

    @classmethod
    def resolve_hf_conversion_map_reversed(cls, target_model: Optional[nn.Module] = None) -> Optional[Any]:
        configured_map = getattr(cls, "HF_CONVERSION_MAP_REVERSED", None)
        if configured_map is not None:
            return copy.deepcopy(configured_map)

        model_type_alias = getattr(cls, "hf_conversion_model_type_alias", None)
        if model_type_alias:
            inferred_map = LazyTurtle.infer_hf_conversion_map_reversed(
                target_model=target_model,
                model_type=model_type_alias,
            )
            if inferred_map is not None:
                return copy.deepcopy(inferred_map)

        inferred_map = LazyTurtle.infer_hf_conversion_map_reversed(target_model=target_model)
        return copy.deepcopy(inferred_map) if inferred_map is not None else None

    @classmethod
    def _collect_moe_modules_from_tree(cls, tree_node, parent_path="", parent_is_moe=False) -> Set[str]:
        """
        Recursively collect all module paths that have the :moe flag.
        Returns a set of full module paths (e.g., "mlp", "mlp.experts", "mlp.shared_experts").
        """
        moe_modules = set()

        if isinstance(tree_node, dict):
            for key, value in tree_node.items():
                # Skip the layer index placeholder
                if key == "#":
                    # Recursively process the value if it's a dict
                    if isinstance(value, dict):
                        moe_modules.update(cls._collect_moe_modules_from_tree(value, parent_path, parent_is_moe))
                    continue

                # Build full path
                module_name, _ = cls._parse_module_flags(key) if isinstance(key, str) else (key, [])
                if parent_path:
                    full_path = f"{parent_path}.{module_name}"
                else:
                    full_path = module_name

                # Check if this key has :moe flag
                is_moe = cls.has_moe_flag(key) if isinstance(key, str) else False
                if is_moe or parent_is_moe:
                    moe_modules.add(full_path)

                # Recursively process nested structures
                if isinstance(value, (dict, tuple, list)):
                    moe_modules.update(
                        cls._collect_moe_modules_from_tree(value, full_path, parent_is_moe or is_moe)
                    )

        elif isinstance(tree_node, (tuple, list)):
            for item in tree_node:
                if isinstance(item, str) and cls.has_moe_flag(item):
                    module_name, _ = cls._parse_module_flags(item)
                    if parent_path:
                        moe_modules.add(f"{parent_path}.{module_name}")
                    else:
                        moe_modules.add(module_name)
                elif isinstance(item, dict):
                    moe_modules.update(cls._collect_moe_modules_from_tree(item, parent_path, parent_is_moe))

        return moe_modules

    @classmethod
    def get_moe_modules(cls) -> Set[str]:
        """
        Get all MoE module paths from the model's module_tree.
        Returns a set of module paths that have the :moe flag.

        Example: {"mlp", "mlp.experts", "mlp.shared_experts", "mlp.gate"}
        """
        if _module_tree_for_class(cls) is None:
            return set()
        moe_modules = set()
        for tree in cls._iter_module_tree_variants():
            moe_modules.update(cls._collect_moe_modules_from_tree(tree))
        return moe_modules

    @classmethod
    def is_moe_module(cls, module_path: str) -> bool:
        """
        Check if a given module path is an MoE module based on :moe flags.

        Args:
            module_path: Full module path like "model.layers.0.mlp.experts.5.gate_proj"

        Returns:
            True if any parent in the path is marked with :moe flag
        """
        if not isinstance(module_path, str) or not module_path:
            return False

        layers = cls.extract_layers_node()
        relative = None
        for layer_path in layers:
            prefix = f"{layer_path}."
            if module_path.startswith(prefix):
                relative = module_path[len(prefix):]
                break
            container_name = layer_path.rsplit(".", 1)[-1]
            shorthand = f"{container_name}."
            if module_path.startswith(shorthand):
                layer_index, separator, remainder = module_path[len(shorthand):].partition(".")
                if separator and layer_index.isdigit():
                    relative = remainder
                    break
        if relative is None:
            return False

        first, separator, remainder = relative.partition(".")
        if separator and first.isdigit():
            relative = remainder
        return ModulePlan.from_tree(_module_tree_for_class(cls)).is_moe(relative)

    @classmethod
    def get_moe_module_name(cls) -> Optional[List[str]]:
        """
        Get the name of the MoE module from module_tree.

        Each layer can have only ONE MoE module marked with :moe flag.
        For example:
        - GLM-4: "mlp:moe" -> returns "mlp"
        - MiniMax-M2 / MiniMax-Text-01: "mlp:moe" -> returns "mlp"

        Returns:
            The names of MoE modules (without flags), or ``None`` when no
            module tree is available.
        """
        if _module_tree_for_class(cls) is None:
            return None

        found_names = []
        for tree in cls._iter_module_tree_variants():
            layer_structure = None
            found_hash = False
            for item in tree:
                if item == "#":
                    found_hash = True
                    continue
                if found_hash and isinstance(item, dict):
                    layer_structure = item
                    break

            if layer_structure is None:
                continue

            for key in layer_structure.keys():
                if cls.has_moe_flag(key):
                    module_name, _ = cls._parse_module_flags(key)
                    if module_name not in found_names:
                        found_names.append(module_name)

        return found_names

    @classmethod
    def _collect_norm_modules_from_tree(
        cls,
        tree_node,
        parent_path: str = "",
        flag: str = NORM_FLAG,
    ) -> List[str]:
        """Collect layer-relative paths of modules tagged with ``flag``."""
        flag_name = flag.lstrip(":")
        found = []
        if isinstance(tree_node, dict):
            for key, value in tree_node.items():
                if key == "#":
                    continue
                module_name, flags = cls._parse_module_flags(key)
                path = f"{parent_path}.{module_name}" if parent_path else module_name
                if flag_name in flags:
                    found.append(path)
                found.extend(cls._collect_norm_modules_from_tree(value, path, flag))
        elif isinstance(tree_node, (tuple, list)):
            for item in tree_node:
                if isinstance(item, dict):
                    found.extend(cls._collect_norm_modules_from_tree(item, parent_path, flag))
                    continue
                if not isinstance(item, str):
                    continue
                module_name, flags = cls._parse_module_flags(item)
                if flag_name not in flags:
                    continue
                # Same leaf rule as the layer walker: a child named like its parent is the parent.
                if parent_path == module_name or parent_path.endswith(f".{module_name}"):
                    found.append(parent_path)
                else:
                    found.append(f"{parent_path}.{module_name}" if parent_path else module_name)
        return found

    @classmethod
    def _collect_flagged_layer_modules(cls, flag: str) -> List[str]:
        """Collect the deduplicated layer-relative paths tagged with ``flag``."""
        names = []
        for tree in cls._iter_module_tree_variants():
            found_hash = False
            for item in tree:
                if item == "#":
                    found_hash = True
                elif found_hash and isinstance(item, dict):
                    for name in cls._collect_norm_modules_from_tree(item, flag=flag):
                        if name not in names:
                            names.append(name)
                    break
        return names

    @classmethod
    def get_norm_modules(cls) -> List[str]:
        """
        Layer-relative paths of normalization modules declared in the module_tree via `:norm`.

        The flag marks eligibility; `NormConfig.types` decides which families are trained.

        Example (Qwen3): ["input_layernorm", "self_attn.q_norm", "self_attn.k_norm", "post_attention_layernorm"]
        """
        return cls._collect_flagged_layer_modules(NORM_FLAG)

    @classmethod
    def get_post_norm_modules(cls) -> List[str]:
        """Layer-relative paths of sandwich post-norms declared via `:postnorm`.

        These norms consume a block's output (after the residual add), so their
        output never feeds the next quantized projection. AWQ must not fold a
        consumer's input scale into them.
        """
        return cls._collect_flagged_layer_modules(POST_NORM_FLAG)

    def awq_post_norm_modules(self, layer: nn.Module) -> Set[str]:
        """Layer-relative post-norms for ``layer``.

        `get_post_norm_modules` is static because most definitions declare one
        `:postnorm` set for every layer. A hybrid model whose decoder layers
        alternate between pre-norm and post-norm shapes cannot: OLMo Hybrid's
        full-attention layer post-norms the attention output and adds it to the
        residual, while its linear-attention layer feeds the same norm straight
        into the MLP. Such a definition overrides this to add the norm only on
        the shapes where it sits on the residual branch.
        """

        return set(self.get_post_norm_modules())

    @classmethod
    def has_norm_flag(cls, module_spec: str) -> bool:
        """Check if a module specification has the `:norm` flag."""
        if not isinstance(module_spec, str):
            return False
        _, flags = cls._parse_module_flags(module_spec)
        return NORM_FLAG.lstrip(":") in flags

    def _pre_lm_head_norm_candidates(self) -> List[str]:
        """Return the declared pre-lm-head norm paths as a flat candidate list.

        A definition may declare one path or an ordered list of alternatives for
        model layouts that differ across transformers releases. Callers must
        never wrap the field again: a list is already the candidate list.
        """
        declared = self.pre_lm_head_norm_module
        if declared is None:
            return []
        if isinstance(declared, str):
            candidates = [declared]
        elif isinstance(declared, (list, tuple)):
            candidates = list(declared)
        else:
            raise TypeError(
                f"{type(self).__name__}.pre_lm_head_norm_module must be a string or a "
                f"list/tuple of strings, got {type(declared).__name__}."
            )

        normalized = []
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate:
                raise TypeError(
                    f"{type(self).__name__}.pre_lm_head_norm_module contains "
                    f"{candidate!r}; expected non-empty module path strings."
                )
            if candidate not in normalized:
                normalized.append(candidate)
        return normalized

    def _resolved_pre_lm_head_norm_candidates(self) -> List[Tuple[nn.Module, str]]:
        """Resolve the declared pre-lm-head norm candidates against ``self.model``.

        Exact submodule paths win over prefix matches, and candidates keep their
        declaration order. A non-empty declaration that resolves to nothing is
        an error: silently skipping the pre-lm-head norm corrupts calibration.
        """
        candidates = self._pre_lm_head_norm_candidates()
        if not candidates:
            return []

        resolved = []
        seen = set()
        for name in candidates:
            try:
                module = self.model.get_submodule(name)
            except AttributeError:
                continue
            if id(module) not in seen:
                seen.add(id(module))
                resolved.append((module, name))
        if resolved:
            return resolved

        # Some definitions intentionally name a module prefix (for example
        # ``lm_head``); preserve that compatibility only when no exact path
        # matched, so a stale candidate cannot win over a valid one.
        for name in candidates:
            module, matched_name = get_module_by_name_prefix(self.model, name)
            if module is not None and id(module) not in seen:
                seen.add(id(module))
                resolved.append((module, matched_name))
        if resolved:
            return resolved

        raise ValueError(
            f"{type(self).__name__}: cannot resolve `pre_lm_head_norm_module` "
            f"candidate(s) {candidates!r} in {type(self.model).__name__}."
        )

    def norm_module_names(self, types=None, skip=()) -> List[str]:
        """Full paths of every tagged norm module matching `types`, plus the pre-lm_head norm.

        `types` accepts `NormType` members or their string values and defaults to
        `(NormType.RMSNORM,)`, matching `NormConfig`.
        """
        requested = normalize_norm_types(types)
        relative = self.get_norm_modules()
        names = []
        for layers_path in self.extract_layers_node():
            try:
                layers = self.model.get_submodule(layers_path)
            except AttributeError:
                continue
            for index, layer in enumerate(layers):
                for name in relative:
                    full_name = f"{layers_path}.{index}.{name}"
                    if any(full_name.endswith(suffix) for suffix in skip):
                        continue
                    try:
                        module = layer.get_submodule(name)
                    except AttributeError:
                        continue
                    family = norm_type_of(module)
                    if family is None:
                        raise ValueError(
                            f"{type(self).__name__} tags `{name}` as `:norm` but it is `{type(module).__name__}`, "
                            "which no normalization handler supports."
                        )
                    if family in requested:
                        names.append(full_name)
        for module, name in self._resolved_pre_lm_head_norm_candidates():
            if any(name.endswith(suffix) for suffix in skip):
                continue
            family = norm_type_of(module)
            if family in requested:
                names.append(name)
            elif family is None:
                log.warn.once(
                    f"{type(self).__name__}: `{name}` is not a supported "
                    "normalization module and is skipped."
                )
            break
        return names

    @classmethod
    def build_moe_modules_if_need(cls, model_config, layer_modules, is_awq_quantize: bool = False):
        # MoE models
        if model_config is not None and cls.dynamic_expert_index is not None:
            num_experts = cls.get_num_experts(model_config)
            warned_unresolved = False

            def expert_count(module_name: str) -> Optional[int]:
                count = cls.get_num_experts_for_module(model_config, module_name)
                return num_experts if count is None else count

            def _warn_unresolved() -> None:
                nonlocal warned_unresolved
                if warned_unresolved:
                    return
                warned_unresolved = True
                log.warning(
                    "MoE: `%s` on `%s` is not a scalar expert count; routed-expert "
                    "specs keep `%s` and are recorded as misses instead of being "
                    "expanded.",
                    cls.dynamic_expert_index,
                    type(model_config).__name__,
                    EXPERT_INDEX_PLACEHOLDER,
                )

            moe_simple = []
            capture_only_modules = None
            for names in layer_modules:
                moe_simple.append([])

                has_expert = any(EXPERT_INDEX_PLACEHOLDER in n for n in names)
                has_capture_only = all(CAPTURE_ONLY_FLAG in n for n in names)
                if has_capture_only:
                    capture_only_modules = list(names)
                    continue

                if not has_expert:
                    moe_simple[-1].extend(names)
                    continue

                if is_awq_quantize:
                    # AWQ expands expert placeholders into concrete expert paths while
                    # preserving the non-expert segments exactly where the model
                    # definition placed them. This keeps the expanded block aligned
                    # with forward execution order instead of forcing shared-expert
                    # modules to the tail of every mixed MoE block.
                    segments = []
                    current_segment = []
                    current_is_expert_segment = None
                    for n in names:
                        is_expert_entry = EXPERT_INDEX_PLACEHOLDER in n
                        if current_is_expert_segment is None:
                            current_is_expert_segment = is_expert_entry
                        if is_expert_entry != current_is_expert_segment:
                            segments.append((current_is_expert_segment, current_segment))
                            current_segment = []
                            current_is_expert_segment = is_expert_entry
                        current_segment.append(n)

                    if current_segment:
                        segments.append((current_is_expert_segment, current_segment))

                    # Example:
                    # ['shared_expert.gate_proj', 'shared_expert.up_proj', 'experts.#.gate_proj', 'experts.#.up_proj']
                    # becomes
                    # ['shared_expert.gate_proj', 'shared_expert.up_proj', 'experts.0.gate_proj', 'experts.0.up_proj', ...]
                    for is_expert_segment, segment_names in segments:
                        if not is_expert_segment:
                            moe_simple[-1].extend(segment_names)
                            continue
                        # Most definitions use one cardinality per block.  If
                        # a block mixes families, retain each family's own
                        # count instead of expanding every placeholder with
                        # the global MoE count.
                        segment_counts = {expert_count(n) for n in segment_names}
                        if segment_counts == {None}:
                            # No scalar cardinality: keep the placeholders so the
                            # consumer reports a miss rather than raising here.
                            _warn_unresolved()
                            moe_simple[-1].extend(segment_names)
                            continue
                        if len(segment_counts) == 1:
                            for index in range(segment_counts.pop()):
                                for n in segment_names:
                                    moe_simple[-1].append(n.replace(EXPERT_INDEX_PLACEHOLDER, str(index)))
                        else:
                            for n in segment_names:
                                count = expert_count(n)
                                if count is None:
                                    _warn_unresolved()
                                    moe_simple[-1].append(n)
                                    continue
                                for index in range(count):
                                    moe_simple[-1].append(n.replace(EXPERT_INDEX_PLACEHOLDER, str(index)))
                    # Currently, only need to add `capture_only_modules` to `['mlp.experts.#.gate_proj', 'mlp.experts.#.up_proj']`
                    # or ['mlp.shared_expert.gate_proj', 'mlp.shared_expert.up_proj', 'mlp.experts.#.gate_proj', 'mlp.experts.#.up_proj']
                    # or ['mlp.shared_experts.gate_proj', 'mlp.shared_experts.up_proj', 'mlp.experts.#.gate_proj', 'mlp.experts.#.up_proj']
                    if capture_only_modules and any(EXPERT_INDEX_PLACEHOLDER in n for n in names):
                        # Extend all elements in capture_only_modules
                        moe_simple[-1].extend(capture_only_modules)
                else:
                    # result like: ['mlp.experts.0.gate_proj', 'mlp.experts.1.gate_proj', 'mlp.experts.0.up_proj', 'mlp.experts.1.up_proj', ...]
                    # A block only enters this branch when at least one name
                    # carries the placeholder, but it may also hold shared-expert
                    # or dense-MLP entries that must be emitted once. Expanding
                    # those with the expert count duplicated them once per
                    # expert (3x spec inflation at 192 experts for `afmoe`).
                    for n in names:
                        if EXPERT_INDEX_PLACEHOLDER not in n:
                            moe_simple[-1].append(n)
                            continue
                        count = expert_count(n)
                        if count is None:
                            # Keep the placeholder when the config does not
                            # expose a scalar cardinality (for example a
                            # per-modality list).  `range()` on that value
                            # would otherwise raise a bare `TypeError`.
                            _warn_unresolved()
                            moe_simple[-1].append(n)
                            continue
                        for index in range(count):
                            moe_simple[-1].append(n.replace(EXPERT_INDEX_PLACEHOLDER, str(index)))

            return moe_simple

        return layer_modules

    @classmethod
    def _expert_count_config(cls, model_config, field_name: Optional[str] = None):
        """Return the config object that owns the routed-expert cardinality.

        Multimodal and remote-code configs nest the language config, and DBRX
        nests its MoE fields under ``ffn_config``. Precedence is preserved for
        existing definitions: ``text_config``, then ``thinker_config``, then
        ``ffn_config``, then the top-level config.

        When ``field_name`` is given, ``ffn_config`` is only selected if it
        actually declares that field. An unrelated ``ffn_config`` on some other
        architecture must not shadow a cardinality kept on the top-level config;
        without this check the added branch would newly raise for such a config.
        """
        if hasattr(model_config, "text_config"):
            return model_config.text_config
        if hasattr(model_config, "thinker_config"):
            return model_config.thinker_config.text_config
        ffn_config = getattr(model_config, "ffn_config", None)
        if ffn_config is not None and (
            field_name is None or getattr(ffn_config, field_name, None) is not None
        ):
            return ffn_config
        return model_config

    @classmethod
    def _coerce_expert_count(cls, value) -> Optional[int]:
        """Return ``value`` as a non-negative int, or None when it is not a scalar.

        Some remote-code configs publish a per-modality list (for example
        ``moe_num_experts = [64, 64]``) instead of one routed-expert
        cardinality.  Such a value cannot size a placeholder expansion, so it
        is reported as unresolved rather than being silently truncated to the
        first family or crashing downstream in ``range()``.
        """
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (list, tuple, dict, set, frozenset)):
            return None
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return None

    @classmethod
    def get_num_experts(cls, model_config) -> Optional[int]:
        """Return the model-wide fallback expert count.

        A missing field keeps raising ``AttributeError`` (unchanged contract: the
        definition and the config disagree). A field that exists but is not a
        scalar returns None, which callers must treat as "cannot expand" and
        record as a miss instead of assuming a count.
        """
        field_name = cls.dynamic_expert_index
        config = cls._expert_count_config(model_config, field_name)
        return cls._coerce_expert_count(getattr(config, field_name))

    @classmethod
    def get_num_experts_for_module(cls, model_config, module_name: str) -> Optional[int]:
        """Resolve a placeholder's expert count from its longest matching prefix.

        ``module_name`` is a layer-relative module-tree path.  A missing or
        malformed mapped config field deliberately falls back to the legacy
        ``dynamic_expert_index`` behavior so existing model definitions remain
        compatible.
        """
        if not isinstance(module_name, str) or EXPERT_INDEX_PLACEHOLDER not in module_name:
            return None

        expert_prefix = module_name.split(EXPERT_INDEX_PLACEHOLDER, 1)[0].rstrip(".")
        mappings = getattr(cls, "dynamic_expert_indices", None) or {}
        matched = None
        for prefix, field_name in mappings.items():
            if not isinstance(prefix, str) or not isinstance(field_name, str):
                continue
            prefix = prefix.strip(".")
            if expert_prefix == prefix or expert_prefix.startswith(f"{prefix}."):
                if matched is None or len(prefix) > len(matched[0]):
                    matched = (prefix, field_name)

        field_name = matched[1] if matched is not None else cls.dynamic_expert_index
        if not field_name:
            return None

        config = cls._expert_count_config(model_config, field_name)
        value = getattr(config, field_name, None)
        if value is None and matched is not None:
            # Mapped fields are optional on dense/checkpoint compatibility
            # configs; preserve the legacy fallback in that case.
            value = getattr(config, cls.dynamic_expert_index, None) if cls.dynamic_expert_index else None
        return cls._coerce_expert_count(value)

    @classmethod
    def filter_not_quantize_module(cls, layer_modules, quantize_config):
        def should_quantize(name: str) -> bool:
            # Check if the module name contains any NON_QUANTIZE_FLAGS that indicates it should NOT be quantized
            return not any(flag in name for flag in NON_QUANTIZE_FLAGS)

        filtered_layer_modules = []
        for block in layer_modules:
            filtered_block = [name for name in block if should_quantize(name)]
            filtered_layer_modules.append(filtered_block)
        layer_modules = filtered_layer_modules

        layer_modules = [block for block in layer_modules if block]  # Remove empty blocks

        if getattr(quantize_config, "dynamic", None):
            new_layer_modules = []
            for modules in layer_modules:
                filtered = [
                    m for m in modules
                    if dynamic_get(quantize_config.dynamic, module_name=m) is not False
                ]
                if filtered:
                    new_layer_modules.append(filtered)
            layer_modules = new_layer_modules

        return layer_modules

    # Inside each `LlamaDecoderLayer` layer are many internal modules
    # List them in the order executed in model forward() code
    # Many models have same execution order of: attention (q_k_v) projection, attention (output) projection, mlp (n) projections
    @classmethod
    def simple_layer_modules(cls, model_config, quantize_config, is_awq_quantize: bool = False, include_capture_only: bool = False):
        layer_modules = cls.build_layer_modules(
            _module_tree_for_class(cls), include_capture_only=include_capture_only
        )

        layer_modules = cls.build_moe_modules_if_need(model_config, layer_modules, is_awq_quantize)

        layer_modules = cls.filter_not_quantize_module(layer_modules, quantize_config)

        # print(f"simple_layer_modules layer_modules: {layer_modules}")
        return layer_modules

    @classmethod
    def shared_input_verified(cls, model_config=None) -> bool:
        """
        True when this definition's `:in=<tag>` metadata was verified by a real forward
        for `model_config.model_type`.

        Only the model types a class lists in its own `shared_input_verified_model_types`
        count (`module_tree` is inherited, the verification set is not), so a subclass or
        an extra `model_type` mapped onto a verified definition stays singleton-only until
        it is covered by `tests/module_tree/test_shared_input_cpu_forward.py`.
        """
        model_type = getattr(model_config, "model_type", None)
        if not isinstance(model_type, str):
            return False
        verified = cls.__dict__.get("shared_input_verified_model_types", ())
        return model_type in verified

    @classmethod
    def shared_input_plan(
        cls,
        model_config=None,
        quantize_config=None,
        is_awq_quantize: bool = False,
    ) -> SharedInputPlan:
        """
        Group the quantizable modules of one decoder layer by shared input tensor.

        Modules in the same group consume identical activations, so Hessian (X^T X)
        collection only needs to run for the group leader. Every module is a singleton
        unless sibling leaves opt in with the same `:in=<tag>` flag *and* the model type
        is listed in `shared_input_verified_model_types` (see `shared_input_verified`).
        """
        layer_modules = cls.simple_layer_modules(model_config, quantize_config, is_awq_quantize=is_awq_quantize)
        return build_shared_input_plan(
            _module_tree_for_class(cls),
            layer_modules,
            explicit_tags=cls.shared_input_verified(model_config),
        )

    @classmethod
    def full_layer_modules(cls, model_config=None, is_awq_quantize: bool = False, include_capture_only: bool = False):
        full = cls.build_layer_modules(
            _module_tree_for_class(cls), include_capture_only=include_capture_only
        )
        full = cls.build_moe_modules_if_need(model_config, full, is_awq_quantize)
        # print(f"full layer_modules: {full}")
        return full

    @classmethod
    def should_quantize_layer(
        cls,
        layer: nn.Module,
        layer_name: str,
        layer_index: int | None,
        quantize_config: BaseQuantizeConfig,
    ) -> bool:
        """Return whether a decoder layer should be quantized during calibration."""
        return True

    @classmethod
    def should_quantize_module(
        cls,
        model: nn.Module,
        module_name: str,
        module: nn.Module,
        quantize_config: BaseQuantizeConfig,
    ) -> bool:
        """
        Return whether a concrete module should be replaced with a quantized module.

        `module_tree` describes reusable suffixes inside decoder layers, but some
        model families use different layer subclasses under the same layer list.
        Model definitions can override this hook when a suffix is quantizable only
        for specific layer subclasses.
        """
        return True

    def prepare_dataset(
        self,
        calibration_dataset: Union[
            List[Dict[str, Union[List[int], torch.LongTensor]]],
            List[str],
            List[List[int]],
            "HFDatasetType",
            "HFIterableDatasetType",
        ],
        calibration_dataset_concat_size: Optional[int] = None,
        calibration_dataset_sort: Optional[str] = None,
        batch_size: int = 1,
        calibration_data_min_length: int = 10,
        calibration_concat_separator: Optional[str] = None,
    ):
        return prepare_calibration_dataset(
            self,
            calibration_dataset=calibration_dataset,
            calibration_dataset_concat_size=calibration_dataset_concat_size,
            calibration_dataset_sort=calibration_dataset_sort,
            batch_size=batch_size,
            calibration_data_min_length=calibration_data_min_length,
            calibration_concat_separator=calibration_concat_separator,
            logger=log,
        )

    def calibrate_activations(self, samples: List[torch.Tensor], *, progress=None) -> Dict[str, Any]:
        """Calibrate a loaded NVFP4 stream without a GPTQ weight solve.

        Samples are unpadded one-dimensional token-ID tensors from a corpus
        disjoint from downstream evaluation. The caller is responsible for
        checking dataset separation; our W4A tooling requires an audited data
        artifact. This updates both runtime scales and the configuration used
        by save(). The model must already be in evaluation mode and every
        decoder layer must use the encoded NVFP4 stream.
        """
        if not self.quantized or self.quantize_config is None:
            raise ValueError("Activation calibration requires an already quantized model")
        from ..quantization.activation_calibration import calibrate_nvfp4_producers

        return calibrate_nvfp4_producers(
            self.model, samples, quantize_config=self.quantize_config, progress=progress,
        )

    def _warn_fused_unit_dynamic_overrides(self) -> None:
        """Warn about `dynamic` rules the fused inference engines cannot honor.

        Inference engines (vLLM, SGLang) merge the routed experts of a layer
        into one module and `gate_proj`/`up_proj` into `gate_up_proj` (likewise
        `q_proj`/`k_proj`/`v_proj` into `qkv_proj`), and they resolve `dynamic`
        from the merged module name only. A rule that matches just part of a
        merged module therefore saves a checkpoint those engines cannot load
        (issue #1455). Warned once per model instance.
        """

        if getattr(self, "_fused_unit_dynamic_warned", False):
            return
        self._fused_unit_dynamic_warned = True

        if not getattr(self.quantize_config, "dynamic", None):
            return

        try:
            module_names = fused_unit_check_module_names(self.model, self.extract_layers_node())
        except Exception as exc:
            log.debug(f"Dynamic: skipped fused-module override check: {exc}")
            return

        warn_fused_unit_dynamic_conflicts(module_names, self.quantize_config)

    def budget(self, *, reserve: float = 0.1, dtype: Optional[torch.dtype] = None) -> ResourceBudget:
        """What this model costs on the current device.

        Measured numbers (weight storage, a device-memory snapshot) sit next to
        the estimated KV-cache footprint, and every accessor on the result says
        which kind it is. ``reserve`` is the share of free device memory held
        back from the KV budget.
        """

        return _budget_for(getattr(self, "model", None), wrapper=self, reserve=reserve, dtype=dtype)

    def kv_cache_estimate(self, *, dtype: Optional[torch.dtype] = None) -> Optional[KVCacheEstimate]:
        """The estimated KV-cache slice of :meth:`budget` (kept as a shortcut)."""

        return _budget_for(getattr(self, "model", None), wrapper=self, dtype=dtype).kv_cache

    def log_budget(self) -> None:
        """Log the resource budget once, marked measured vs estimated."""

        for line in _budget_for(getattr(self, "model", None), wrapper=self).describe():
            log.info(line)

    def _calibration_batch_size(self, calibration) -> int:
        """Resolve ``CalibrationConfig(batch_size="auto")`` for any entry point.

        ``quantize()`` used to be the only caller that resolved the sentinel, so
        ``requantize(norm=...)`` and ``adapter.generate()`` handed the raw
        ``"auto"`` string to ``prepare_dataset`` and crashed in
        ``range(0, len(dataset), "auto")`` with a ``TypeError`` that never named
        ``batch_size`` (#5504).
        """

        batch_size = calibration.batch_size
        if batch_size == AUTO_BATCH_SIZE:
            batch_size = self._auto_calibration_batch_size(calibration)
        return batch_size

    def _auto_calibration_batch_size(self, calibration) -> int:
        """Pick a calibration batch that fits, or 1 when the device is unmeasurable."""

        config = getattr(getattr(self, "model", None), "config", None)
        # Composite configs (VLM / omni / Qwen3.5) keep the decoder dims on
        # `text_config` and leave the top-level fields unset. Reading them as 0
        # shrank the per-row estimate by `hidden * layers` and silently
        # disabled the memory guard (#5568).
        text_config = config
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            try:
                text_config = get_text_config() or config
            except Exception:  # pragma: no cover - exotic configs
                text_config = config
        hidden_size = int(getattr(text_config, "hidden_size", 0) or 0)
        num_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
        if hidden_size <= 0 or num_layers <= 0:
            log.warn(
                "Calibrate: batch_size=auto -> 1 (the model config exposes no hidden_size/num_hidden_layers)"
            )
            return 1

        dtype_bytes = 2
        inner = getattr(self, "model", None)
        try:
            for parameter in inner.parameters():
                if parameter.is_floating_point():
                    dtype_bytes = parameter.element_size()
                    break
        except Exception:  # pragma: no cover - exotic module trees
            dtype_bytes = 2

        free_bytes = None
        device = getattr(getattr(self, "quantize_config", None), "device", None)
        device_type = getattr(device, "type", None) or str(device).split(":")[0].lower()
        if device_type == "cuda":
            try:
                index = getattr(device, "index", None)
                free_bytes, _total = (
                    torch.cuda.mem_get_info() if index is None else torch.cuda.mem_get_info(index)
                )
            except Exception:  # pragma: no cover - driver edge cases
                free_bytes = None

        rows = _calibration_row_count(calibration)
        tokens = _calibration_tokens_per_row(self, calibration)
        batch = auto_batch_size(
            rows=rows,
            tokens_per_row=tokens,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dtype_bytes=dtype_bytes,
            free_bytes=free_bytes,
        )
        where = "device unmeasurable" if free_bytes is None else f"free {free_bytes / 1024 ** 3:.1f} GiB"
        log.info(
            "Calibrate: batch_size=auto -> %d (rows=%d, tokens/row=%d, %s)",
            batch, rows, tokens, where,
        )
        return batch

    def measure(self, config: "MeasureConfig") -> "MeasurementReport":
        """Compare this model against a reference on the same token stream.

        This is the fast divergence check (PPL/KLD/TOP_K/RMSE), not an
        end-to-end quality score: it answers whether quantization changed the
        computation, which is what a kernel or packing change would silently do.
        """

        from ..measure.config import Metric, PPLConfig
        from ..measure.runner import build_token_rows, measure_models

        if config.dataset is None:
            raise ValueError("measure() needs `dataset` (texts or token-id rows)")

        reference = self._measurement_reference(config)
        ppl_config = config.config_for(Metric.PPL)
        window = ppl_config.window if isinstance(ppl_config, PPLConfig) else 512
        stride = ppl_config.stride if isinstance(ppl_config, PPLConfig) else 0
        rows = build_token_rows(
            getattr(self, "tokenizer", None),
            config.dataset,
            window=window,
            stride=stride,
            max_rows=config.max_batches,
        )
        return measure_models(
            self.model,
            reference,
            rows,
            config,
            pad_token_id=getattr(getattr(self, "tokenizer", None), "pad_token_id", None),
        )

    def _measurement_reference(self, config: "MeasureConfig"):
        """Resolve the reference model: a module, or a checkpoint path to load."""

        reference = config.reference
        if reference is None:
            raise ValueError(
                "measure() needs `reference` (a model or a checkpoint path): a quantized "
                "checkpoint no longer holds the fp16 weights it diverged from"
            )
        if not isinstance(reference, str):
            return reference

        from transformers import AutoModelForCausalLM

        target = config.device or getattr(self.model, "device", None) or "cpu"
        model = AutoModelForCausalLM.from_pretrained(reference, torch_dtype=torch.bfloat16)
        return model.to(target)

    @with_quantization_device_telemetry
    def quantize(
        self,
        calibration: Optional[CalibrationConfig] = None,
        *,
        # Replaces the model's tokenizer: it tokenizes the calibration text and
        # is the tokenizer `save()` writes and `generate()` uses.
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        backend: Optional[Backend] = BACKEND.AUTO,
        # eora adapter generation needs config Lora(rank=1, path='lora.safetensors');
        # its samples are `CalibrationConfig.adapter_dataset`.
        adapter: Adapter = None,
        embed_quant_config: Optional[Union[QuantizeEmbedConfig, QuantizeEmbed]] = None,
        embed_quant_mode: Optional[QuantizeEmbed] = None,
        checkpoint: Optional["CheckpointConfig"] = None,
        fuse: Optional[bool] = None,
        # Weight-only direct-pack methods skip the
        # logging-only weight-error MAE by default; requesting it reconstructs
        # every dense weight. It never changes the packed payload.
        measure_weight_error: bool = False,
    ) -> Dict[str, List[Dict[str, str]]]:
        """Quantize the model with the calibration data and policy in ``calibration``.

        ``calibration`` may be ``None`` only for a weight-only quantize config.
        ``measure_weight_error`` requests the direct-pack weight-error MAE; it
        reconstructs each dense weight and never changes the packed payload.
        """
        run_started = time.perf_counter()
        if calibration is None:
            calibration = CalibrationConfig()
        elif not isinstance(calibration, CalibrationConfig):
            raise TypeError(
                "quantize(calibration=...) must be a CalibrationConfig, got "
                f"{type(calibration).__name__}; pass `CalibrationConfig(dataset=...)`."
            )
        elif calibration.prepare_dataset is not None:
            raise ValueError(
                "quantize() binds the model's dataset preparer; pass an unbound CalibrationConfig."
            )
        if self.quantize_config.is_load_only():
            raise ValueError(
                "FP8 is load-only: GPT-QModel no longer quantizes or exports FP8 "
                "checkpoints. Load a production FP8 checkpoint (transformers "
                "`FineGrainedFP8`, DeepSeek/Qwen block scales, NVIDIA ModelOpt, "
                "MXFP8, or NVFP8), or quantize with GPTQ/AWQ/RTN."
            )
        for pattern, override in (getattr(self.quantize_config, "dynamic", None) or {}).items():
            if isinstance(override, Mapping) and "exclude_attention_sinks" in override:
                # The sink rows are a property of the calibration batch, not of
                # one module, so a per-module rule would silently do nothing.
                raise ValueError(
                    f"QuantizeConfig: `dynamic['{pattern}'].exclude_attention_sinks` is not a "
                    "per-module option; use `CalibrationConfig(exclude_attention_sinks=True)`."
                )
        vram_devices = _quantize_vram_devices(self.quantize_config)
        _reset_peak_vram_stats(vram_devices)

        embed_quant_config = self._normalize_embed_quant_config(
            embed_quant_config=embed_quant_config,
            embed_quant_mode=embed_quant_mode,
        )
        if embed_quant_config is not None and not self.quantized:
            embed_mode = embed_quant_config.embed_quant_mode
            if (
                embed_mode in (QuantizeEmbed.OUTPUT, QuantizeEmbed.BOTH)
                and lm_head_quantization_is_tied(self)
            ):
                # A fresh quantize of the output endpoint replaces the tied
                # `lm_head` with a packed module. The run used to finish every
                # layer and only then fail to re-tie the head, leaving
                # `quantized=True` on an unusable model (#5111). Refuse before
                # any calibration work, matching the `lm_head=True` contract.
                # Requantize of a loaded checkpoint stays allowed: its
                # endpoints are already independent packed modules.
                raise NotImplementedError(
                    "quantize() cannot quantize the output embedding of a freshly "
                    "loaded model with tied input/output embeddings: `lm_head` "
                    "shares its weight with `embed_tokens`. Untie the model weights "
                    "before quantization, or load a quantized checkpoint and use "
                    "`requantize(embed_quant_mode=...)`."
                )
            if embed_mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH):
                # The capture forward has to execute the input embedding before
                # the pseudo-layer is packed. A fresh run starts from a lazy
                # shell, and materializing the endpoint for capture produced a
                # packed FP16 endpoint in front of a dense BF16 decoder, so the
                # in-session model could not forward (#5111). This flow is
                # requantize-only; refuse before any calibration work.
                raise NotImplementedError(
                    "quantize() on an unquantized model cannot quantize the input "
                    "embedding (QuantizeEmbed.INPUT/BOTH). Load a quantized "
                    "checkpoint and use `requantize(embed_quant_mode=...)`."
                )

        if fuse is not None and type(fuse) is not bool:
            raise TypeError("Fusion enabled must be bool")
        fusion_requested = fuse if fuse is not None else getattr(self, "_auto_fusion", False)
        if fusion_requested:
            require_fusable_config(type(self), getattr(self.model.config, "model_type", None))
        if fuse is not None:
            self._auto_fusion = fuse
            if not fuse:
                self.fuse(False)

        if self.quantize_config is None or not isinstance(self.quantize_config, BaseQuantizeConfig):
            raise AttributeError("`quantize_config` must be not None")

        # A typed choice on the dense-model load survives into quantize(),
        # while an explicit quantize backend may replace that choice.
        backend, _ = resolve_quantize_backend(backend, self.quantize_config, inherit=True)

        dynamic_cache_stats_before = (
            dynamic_override_cache_stats(self.quantize_config.dynamic) if self.quantize_config.dynamic else None
        )

        if os.environ.get("GPTQMODEL_RESUME") == "1":
            raise ValueError(
                "GPTQMODEL_RESUME is retired; use checkpoint=CheckpointConfig(path=...) in a new directory"
            )
        if checkpoint is not None:
            from ..looper.checkpoint_store import CheckpointConfig
            from ..looper.gptq_checkpoint import validate_checkpoint_support

            if not isinstance(checkpoint, CheckpointConfig):
                raise TypeError("checkpoint must be a CheckpointConfig")
            validate_checkpoint_support(self, embed_quant_config=embed_quant_config, adapter=adapter)

        if self.quantized and embed_quant_config is None:
            raise EnvironmentError("quantize() is called a model that is already quantized")

        timer = getattr(self, "quant_region_timer", None)
        if timer is not None:
            timer.reset()

        if self.quantize_config.method in QUANTIZE_BLACK_LIST:
            raise ValueError(
                f"Unsupported quantization operation for quant method: {self.quantize_config.method}"
            )

        staged_gsq = getattr(self.quantize_config, "gsq_training", None)
        if staged_gsq is not None and staged_gsq.enabled:
            if self.quantize_config.method == METHOD.GPTQ:
                from ..looper.gsq_training_model import quantize_llama_gsq_public

                # Staged GSQ builds the same input Hessian as the regular GPTQ
                # path, so its calibration forwards need the run-level
                # deterministic pin too. This path returns before the shared
                # `quant_blas_backend` scope below, which left it on hipBLASLt
                # and made the checkpoint process-dependent (issue #4232).
                with quant_blas_backend(deterministic=True):
                    result = quantize_llama_gsq_public(
                        self, calibration=calibration, tokenizer=tokenizer, backend=backend,
                        unsupported={
                            "adapter": adapter,
                            "CalibrationConfig.adapter_dataset": calibration.adapter_dataset,
                            "embed_quant_config": embed_quant_config,
                            "checkpoint": checkpoint,
                            "CalibrationConfig.exclude_attention_sinks": (
                                True if calibration.exclude_attention_sinks else None
                            ),
                        },
                    )
                self.after_quantize()
                if self._auto_fusion:
                    self.fuse()
                if timer is not None:
                    timer.flush()
                _log_quantize_summary(
                    result,
                    run_started=run_started,
                    vram_devices=vram_devices,
                )
                return result
            if self.quantize_config.method != METHOD.AWQ:
                raise ValueError(f"Staged GSQ is unsupported for {self.quantize_config.method}")

        calibration = _materialized_calibration_config(calibration)
        batch_size = self._calibration_batch_size(calibration)
        if not self.support_batch_quantize:
            log.warn("Quantize: batch_size overridden by model class definition to `disabled`")
            batch_size = 1 # but actually disabled

        format_code = resolve_quant_format(self.quantize_config.format, self.quantize_config.method)

        if format_code == FORMAT.MARLIN:
            raise ValueError(
                "FORMAT.MARLIN is deprecated for quantization. Please switch to FORMAT.GPTQ. GPTQMOdel will auto-use Marlin kernel for accelerated inference for FORMAT.GPTQ."
            )

        export_quant_method = self.quantize_config.export_quant_method()

        if export_quant_method == METHOD.AWQ:
            if format_code in [FORMAT.GEMV_FAST, FORMAT.LLM_AWQ]:
                # AWQ GEMV_FAST / LLM_AWQ only supports pack_dtype is torch.int16
                log.info("Quantize Model: Auto fix `pack_dtype` to `torch.int16`")
                self.quantize_config.pack_dtype = torch.int16

        if self.support_batch_quantize is False:
            batch_size = 1
            log.warn("Batch quantization is not supported for this model. Setting batch_size to 1.")

        requested_backend = backend
        requested_backend = normalize_backend(requested_backend, quant_method=export_quant_method)
        requested_backend = backend_for_activation(self.quantize_config.activation_mode, requested_backend)

        preferred_backend = requested_backend
        if preferred_backend in (None, BACKEND.AUTO):
            quant_device = self.quantize_config.device
            if isinstance(quant_device, DEVICE):
                quant_device_type = quant_device.type
            elif isinstance(quant_device, torch.device):
                quant_device_type = quant_device.type
            else:
                quant_device_type = str(quant_device).split(":")[0].lower()

            if export_quant_method == METHOD.AWQ:
                if quant_device_type == "npu" and format_code != FORMAT.GEMM:
                    raise ValueError(
                        "NPU AWQ quantization requires FORMAT.GEMM so the AwqTorchLinear runtime can run on NPU; "
                        f"actual format is `{format_code}`."
                    )
                if format_code == FORMAT.GEMM:
                    # The native AWQ GEMM extension is CUDA-only, so weight-only
                    # exports and ROCm/XPU/NPU/CPU hosts pack with the portable
                    # torch kernel, which emits the identical AWQ GEMM layout.
                    preferred_backend = awq_gemm_pack_backend(
                        device_type=quant_device_type,
                        is_rocm=IS_ROCM,
                        weight_only=self.quantize_config.uses_weight_only_lifecycle(),
                    )
                elif format_code == FORMAT.BITBLAS:
                    preferred_backend = BACKEND.AWQ_BITBLAS
                elif format_code == FORMAT.GEMV:
                    # The native GEMV extension is CUDA-only. ROCm packs the
                    # identical GEMV layout with the portable Triton kernel
                    # instead of selecting a class its runtime cannot build; a
                    # CPU-only host has neither accelerator kernel and packs
                    # with the portable Torch GEMV reader (issue #4554).
                    preferred_backend = awq_gemv_pack_backend(
                        device_type=quant_device_type, is_rocm=IS_ROCM
                    )
                elif format_code in [FORMAT.GEMV_FAST, FORMAT.LLM_AWQ]:
                    # The native GEMV_FAST extension is CUDA-only. A host
                    # without it -- CPU only (#4736) or ROCm, where hipcc
                    # rejects the extension (#5109) -- packs the identical
                    # interleaved layout with the portable Torch reader and
                    # keeps it as the in-session kernel.
                    preferred_backend = awq_gemv_fast_pack_backend(
                        device_type=quant_device_type, is_rocm=IS_ROCM
                    )
                else:
                    raise ValueError(f"Unsupported FORMAT: `{self.quantize_config.format}` with `METHOD.AWQ`")
            elif self.quantize_config.method == METHOD.QQQ:
                # QQQ's Marlin-derived W4A8 kernel is CUDA-only in practice: its
                # extension is built from CUDA inline PTX that hipify cannot
                # translate, so it cannot be compiled by hipcc on gfx120*.  CPU,
                # NPU and ROCm hosts must use the portable torch implementation
                # instead of selecting a backend that fails validation.
                preferred_backend = (
                    BACKEND.QQQ_TORCH
                    if IS_ROCM or quant_device_type in ("cpu", "npu")
                    else BACKEND.QQQ
                )
            elif self.quantize_config.method == METHOD.VPTQ:
                preferred_backend = BACKEND.VPTQ_TORCH
            elif self.quantize_config.method == METHOD.NF12:
                preferred_backend = BACKEND.NF12
            elif self.quantize_config.method == METHOD.CPTC:
                preferred_backend = BACKEND.CPTC
            elif self.quantize_config.method == METHOD.MACAQ:
                from ..utils.macaq_triton import triton_mxfp4_available

                preferred_backend = (
                    BACKEND.MACAQ_TRITON
                    if quant_device_type in ("cuda", "rocm")
                    and triton_mxfp4_available(quant_device)
                    else BACKEND.MACAQ_TORCH
                )
            elif self.quantize_config.method == METHOD.PARO:
                preferred_backend = BACKEND.PARO_CUDA
            elif self.quantize_config.method == METHOD.EXL3:
                preferred_backend = BACKEND.EXL3_EXLLAMA_V3
            elif self.quantize_config.method == METHOD.GGUF:
                preferred_backend = BACKEND.AUTO
            elif self.quantize_config.method == METHOD.BITSANDBYTES:
                preferred_backend = BACKEND.BITSANDBYTES
            else:
                preferred_backend = BACKEND.GPTQ_TORCH

        # Record the effective pack backend so the weight-only finalize path can
        # tell an AUTO request (which may fall back per module) from an explicit
        # one (which must raise instead).
        self.quantize_backend = preferred_backend

        if self.quantize_config.method == METHOD.EXL3:
            if preferred_backend not in (BACKEND.AUTO, BACKEND.EXL3_EXLLAMA_V3):
                raise ValueError("EXL3 quantization only supports BACKEND.AUTO or BACKEND.EXL3_EXLLAMA_V3.")

            if not torch.cuda.is_available():
                raise ValueError("EXL3 quantization requires CUDA/HIP.")

            if IS_ROCM and (
                self.quantize_config.codebook == "q4nx"
                or any(rule.get("codebook") == "q4nx" for rule in (self.quantize_config.dynamic or {}).values())
            ):
                raise ValueError("EXL3 q4nx quantization requires NVIDIA CUDA.")

            quant_device = self.quantize_config.device
            if isinstance(quant_device, DEVICE):
                quant_device_type = quant_device.type
            elif isinstance(quant_device, torch.device):
                quant_device_type = quant_device.type
            else:
                quant_device_type = str(quant_device).split(":")[0].lower()

            if quant_device_type != "cuda":
                raise ValueError("EXL3 quantization requires a CUDA/HIP quantization device.")
        else:
            # Validate quant linear before quantization starts
            _ = select_quant_linear(
                bits=self.quantize_config.runtime_bits,
                dynamic=self.quantize_config.dynamic,
                group_size=self.quantize_config.group_size,
                desc_act=self.quantize_config.desc_act,
                sym=self.quantize_config.sym,
                backend=preferred_backend,
                format=format_code,
                quant_method=export_quant_method,
                device=normalize_device(self.quantize_config.device),
                pack=True,
                pack_dtype=self.quantize_config.pack_dtype,
            )

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                # TODO FIX ME...this is a bug
                self.tokenizer = Tokenicer.load(
                    tokenizer,
                    trust_remote_code=self.trust_remote_code,
                    model_config=getattr(self.model, "config", None),
                )
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        if format_code == FORMAT.BITBLAS:
            from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            # ``is False`` (not ``not``) keeps the pre-lazy semantics: an
            # explicit ``None``/``0`` override means "unknown", not "absent".
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        # overwrite quantize_config.adapter
        if adapter is not None:
            self.quantize_config.adapter = adapter

        from ..adapter.adapter import Lora

        if calibration.adapter_dataset is not None and not isinstance(
            self.quantize_config.adapter, Lora
        ):
            raise ValueError(
                "CalibrationConfig.adapter_dataset requires a `Lora` adapter "
                "(quantize(adapter=Lora(...)) or quantize_config.adapter)."
            )

        if self.quantize_config.method == METHOD.EXL3:
            self.qlinear_kernel = ExllamaV3Linear
        else:
            self.qlinear_kernel = select_quant_linear(
                    bits=self.quantize_config.runtime_bits,
                    group_size=self.quantize_config.group_size,
                    desc_act=self.quantize_config.desc_act,
                    sym=self.quantize_config.sym,
                    pack=True,
                    dynamic=self.quantize_config.dynamic,
                    device=normalize_device(self.quantize_config.device),
                    pack_dtype=self.quantize_config.pack_dtype,
                    multi_select=False,
                    backend=preferred_backend,
                    format=format_code,
                    quant_method=export_quant_method,
                )

        # Validate rotation before any weight is touched. The old order fused
        # norms and rotated the weights first and only then rejected the
        # backend, so a refused run left an unusable in-session model (#4666).
        rotation = resolve_model_rotation(self.quantize_config)
        if rotation:
            validate_rotation_format(self.quantize_config, format_code=format_code)
            validate_rotation_model_support(self)

            # Validate every width rotation will transform before
            # `fuse_layer_norms` mutates the model in place. A width without an
            # exact Hadamard factor otherwise fails inside `rotate_mlp_output`
            # only after embeddings, head, norms and attention have already
            # been rotated and fused.
            validate_rotation_dimensions(self.model.config, rotation)

            backend = normalize_backend(backend, quant_method=self.quantize_config.method)
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

        # Resolve the run plan before the mutation below: a config the chain
        # cannot run (an unsupported checkpoint state, for example) is user
        # input too and must not leave a rotated model behind.
        run_plan = QuantizationRunPlan.build(
            self.quantize_config,
            format_code=format_code,
            pack_backend=preferred_backend,
            loop_backend=backend,
            batch_size=batch_size,
            needs_lora=isinstance(self.quantize_config.adapter, Lora),
        )
        if checkpoint is not None and not run_plan.checkpoint_supported:
            raise NotImplementedError("checkpoint has no state adapter for this processor chain")
        self.quantization_run_plan = run_plan

        # The rotation below mutates the model in place: it unties the
        # embeddings, fuses the layer norms and rotates the weights. Every
        # calibration-data error must be raised before that, or a refused run
        # leaves the in-session model rotated without the online Hadamard a
        # rotated `down_proj` needs; a retry on the same object then rotates
        # (and fuses) it a second time and silently quantizes from that broken
        # state (#4935). Prepare the input once here and hand the exact batches
        # to the processors below. Weight-only configs (for example RTN) need
        # no calibration rows and are unaffected.
        prepared_calibration_dataset = None
        prepared_adapter_dataset = None
        if rotation and not run_plan.weight_only:
            prepared_calibration_dataset = self._prepare_calibration_before_rotation(
                calibration=calibration,
                batch_size=batch_size,
                field="dataset",
            )
            if calibration.adapter_dataset is not None:
                prepared_adapter_dataset = self._prepare_calibration_before_rotation(
                    calibration=calibration,
                    batch_size=batch_size,
                    field="adapter_dataset",
                )

        # rotate model
        if rotation:
            if self.model.config.tie_word_embeddings:
                log.info("Rotation requires word embeddings to be untied. Untying.")
                self.model.config.tie_word_embeddings = False
                lm_head, _ = get_module_by_name_prefix(self.model, self.lm_head)
                lm_head.weight = nn.Parameter(lm_head.weight.data.clone())

            module_name_args = {
                "layers_node": self.extract_layers_node(),
                "lm_head_name": self.lm_head
            }
            self.model = fuse_layer_norms(model=self.model,
                                            pre_lm_head_norm_module_name=self.pre_lm_head_norm_module,
                                            plan=self.module_plan(),
                                            **module_name_args)

            # MPS does not support float64.
            rotation_device = (
                as_torch_device(self.quantize_config.device)
                if self.quantize_config.device is not None
                else None
            )
            if rotation_device is not None and rotation_device.type == "mps":
                rotation_device = CPU
            self.model, _ = rotate_model(model=self.model, rotate_mode=rotation,
                                            device=rotation_device, plan=self.module_plan(),
                                            embedding_name=self.get_input_embeddings_name(),
                                            **module_name_args)

        # Warn before the (possibly long) quantization run: inference engines
        # (vLLM, SGLang) fuse these modules and resolve `dynamic` from the fused
        # module name only, so a checkpoint saved with a partial rule fails to
        # load there (issue #1455).
        self._warn_fused_unit_dynamic_overrides()

        # On ROCm the deterministic rocBLAS selection is scoped to methods
        # whose quantization GEMMs must be reproducible (GPTQ, QQQ, VPTQ,
        # MACaQ, EXL3, and Paro). For the Hessian
        # builders, the calibration forward and replay produce the activations
        # accumulated into the Hessian, so those GEMMs must be reproducible
        # too: pinning only the fp32 X^T X GEMM regressed Qwen2.5-0.5B to
        # 332/795 differing tensors across processes. Paro accumulates no
        # Hessian, but hipBLASLt selects its rotation-optimization GEMM kernels
        # per process, so seeded runs diverged (issue #4460); rocBLAS is the
        # documented speed-for-reproducibility trade-off. AWQ and RTN keep
        # hipBLASLt, which is 8-14x faster for their fp16/bf16 GEMMs on
        # gfx1201. Every run pins its backend for the whole run; a concurrent
        # run that needs the other one waits rather than flipping it mid-run
        # (issue #4123).
        blas_context = quant_blas_backend(
            deterministic=_needs_deterministic_hessian(run_plan)
            or (run_plan.method == METHOD.CPTC and self.quantize_config.requires_calibration_dataset())
        )
        with blas_context:
            if run_plan.weight_only:
                if calibration.exclude_attention_sinks and not self.quantize_config.requires_calibration_dataset():
                    log.warn(
                        "Quantize: `CalibrationConfig.exclude_attention_sinks` has no effect on "
                        "weight-only quantization, which uses no calibration rows."
                    )
                result = self._quantize_weight_only(
                    run_plan=run_plan,
                    checkpoint=checkpoint,
                    calibration=calibration,
                    backend=backend,
                    embed_quant_config=embed_quant_config,
                    measure_weight_error=measure_weight_error,
                )
            else:
                if calibration.dataset is None:
                    raise ValueError(
                        "CalibrationConfig.dataset is required unless a weight-only quantize config is configured."
                    )
                # Record the calibration row policy for provenance. The key is
                # absent for the default policy, so a default run never changes
                # the serialized config.
                if calibration.exclude_attention_sinks:
                    self.quantize_config.meta_set(
                        key=META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS, value=True
                    )
                elif isinstance(self.quantize_config.meta, dict):
                    self.quantize_config.meta.pop(
                        META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS, None
                    )
                result = self._quantize_with_calibration(
                    run_plan=run_plan,
                    calibration=calibration,
                    backend=backend,
                    embed_quant_config=embed_quant_config,
                    checkpoint=checkpoint,
                    prepared_dataset=prepared_calibration_dataset,
                    prepared_adapter_dataset=prepared_adapter_dataset,
                )

        # Some definitions need the complete layer stack before they can
        # restore cross-branch aliases (for example DiffusionGemma's decoder).
        self.after_quantize()

        # The loop leaves packed layers on CPU, so an accelerator-only packing
        # kernel (AWQ GEMM, QQQ, GGUF Triton) would fail the first in-session
        # forward. Match the runtime kernel to the device the model now lives
        # on, or raise the reload-the-checkpoint guidance (issue #4381).
        rehome_quantized_kernel(self)

        if self.quantize_config.activation_mode in {"w4afp8", "w4a_nvfp4"}:
            from ..nn_modules.qlinear.w4a_floatx import W4AFP8Linear
            from ..nn_modules.qlinear.w4a_llama_stream import install_w4a_llama_stream

            # The looper packs on CPU and waits for all finalizers before it
            # returns. Prepare the derived operand for immediate inference;
            # the ordinary loader performs the same step after checkpoint load.
            for submodule in self.model.modules():
                if isinstance(submodule, W4AFP8Linear) and submodule.qweight.device.type != "meta":
                    submodule.post_init()
            install_w4a_llama_stream(
                self.model, self.quantize_config.activation_mode,
                self.quantize_config.activation_recipe,
                global_scales=self.quantize_config.activation_global_scales,
                attention_mode=self.quantize_config.activation_attention_mode,
                attention_recipe=self.quantize_config.activation_attention_recipe,
                mlp_fp8_layers=self.quantize_config.activation_mlp_fp8_layers,
                fused_norms=bool(resolve_model_rotation(self.quantize_config)),
            )

        timer = getattr(self, "quant_region_timer", None)
        if timer is not None:
            timer.flush()

        _setup_rotation_online_had(
            self.model,
            resolve_model_rotation(self.quantize_config),
            module_plan=self.module_plan(),
            layer_prefixes=self.extract_layers_node(),
        )
        if self.quantize_config.dynamic:
            log_dynamic_override_cache_stats(dynamic_cache_stats_before, dynamic=self.quantize_config.dynamic)
        if self._auto_fusion:
            self.fuse()

        _log_quantize_summary(
            result,
            run_started=run_started,
            vram_devices=vram_devices,
        )
        # Layer offload stages every device-to-host copy through a bounded
        # pinned buffer. Those buffers are gone once the loop returns, but
        # PyTorch keeps the pinned blocks cached (non-swappable ``RssShmem``)
        # for the process lifetime.
        release_pinned_host_cache()
        self.log_budget()
        return result

    @staticmethod
    def _normalize_embed_quant_config(
        embed_quant_config: Optional[Union[QuantizeEmbedConfig, QuantizeEmbed]],
        embed_quant_mode: Optional[QuantizeEmbed],
    ) -> Optional[QuantizeEmbedConfig]:
        """Normalize the embedding quantization API and its compatibility argument."""
        if embed_quant_config is not None and embed_quant_mode is not None:
            raise ValueError("Pass only one of `embed_quant_config` or `embed_quant_mode`.")
        if isinstance(embed_quant_config, QuantizeEmbed):
            return QuantizeEmbedConfig(embed_quant_mode=embed_quant_config)
        if embed_quant_config is not None:
            if not isinstance(embed_quant_config, QuantizeEmbedConfig):
                raise TypeError("`embed_quant_config` must be a `QuantizeEmbedConfig` or `QuantizeEmbed` instance.")
            # Re-validate: the caller may have mutated the instance after
            # construction. A bad mode/`embed_only` must fail here, before the
            # layer replay starts, not silently quantize nothing (#5124).
            return embed_quant_config.validate()
        if embed_quant_mode is not None:
            if not isinstance(embed_quant_mode, QuantizeEmbed):
                raise TypeError("`embed_quant_mode` must be a `QuantizeEmbed` instance.")
            return QuantizeEmbedConfig(embed_quant_mode=embed_quant_mode)
        return None

    def requantize(
        self,
        calibration: CalibrationConfig,
        *,
        embed_quant_config: Optional[Union[QuantizeEmbedConfig, QuantizeEmbed]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        backend: Optional[BACKEND] = BACKEND.AUTO,
        adapter: Adapter = None,
        embed_quant_mode: Optional[QuantizeEmbed] = None,
        norm: Optional[NormConfig] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        if not isinstance(calibration, CalibrationConfig):
            raise TypeError(
                "requantize(calibration=...) must be a CalibrationConfig, got "
                f"{type(calibration).__name__}; pass `CalibrationConfig(dataset=...)`."
            )
        calibration = _materialized_calibration_config(calibration)
        if not self.quantized:
            raise EnvironmentError("requantize() must be called on a model that has already been quantized.")
        embed_quant_config = self._normalize_embed_quant_config(embed_quant_config, embed_quant_mode)
        if norm is not None and not isinstance(norm, NormConfig):
            raise TypeError("`norm` must be a `NormConfig` instance.")
        if embed_quant_config is None and norm is None:
            raise ValueError("`requantize()` requires `embed_quant_config`, `embed_quant_mode` or `norm`.")

        # Fail before embedding packing, dataset preparation or teacher loading.
        # Loaded GPTQ-v1 checkpoints are fine: their runtime zeros are already v2.
        if (
            norm is not None
            and embed_quant_config is not None
            and resolve_quant_format(
                self.quantize_config.format, self.quantize_config.method
            ) == FORMAT.GPTQ
        ):
            raise ValueError(
                "Norm tuning cannot be combined with GPTQ-v1 embedding requantization. "
                "Requantize embeddings separately, save the model, then reload with "
                "`backend=BACKEND.TORCH` and call `requantize(norm=...)`."
            )

        if self.load_quantized_model and self.turtle_model is None:
            # Resolve persistence before embedding packing or norm updates can
            # mutate the loaded model. Revalidate on each requantize() call.
            self._module_replacement_source = _QuantizedCheckpointSource(
                str(self.model_local_path), checkpoint_file_name=getattr(self, "checkpoint_file_name", None)
            )

        norm_names = None
        if norm is not None:
            # The whole norm pre-flight: every rejection has to land before the
            # embeddings are replaced, or the in-session model keeps a quantized
            # embedding that no later requantize() accepts (#5118).
            norm_names = self._validate_requantize_norm(
                calibration=calibration, norm=norm, tokenizer=tokenizer
            )

        result = {}
        if embed_quant_config is not None:
            result = self._requantize_embeddings(
                calibration=calibration,
                embed_quant_config=embed_quant_config,
                tokenizer=tokenizer,
                backend=backend,
                adapter=adapter,
            )
        if norm is not None:
            result = dict(result or {})
            result["norm"] = self._requantize_norm(
                calibration=calibration,
                norm=norm,
                tokenizer=tokenizer,
                names=norm_names,
            )
        return result

    def _validate_requantize_norm(
        self,
        *,
        calibration: CalibrationConfig,
        norm: NormConfig,
        tokenizer: Optional[PreTrainedTokenizerBase],
    ) -> List[str]:
        """Reject every invalid norm request before the model is mutated (#5118).

        ``requantize(embed_quant_mode=..., norm=...)`` packs the embeddings
        first. A rejected norm half (calibration flags, no matching norm
        modules, a bad tokenizer, a mistyped teacher id) used to arrive after
        that, leaving the input embedding replaced by a quantized module that
        no later ``requantize()`` accepts. Resolve the teacher here too: a
        path/id typo must fail before the embedding is touched.
        """

        from ..quantization.norm_tuning import validate_tuning_model

        validate_tuning_model(self.model)
        if calibration.dataset is None:
            raise ValueError("Norm tuning requires CalibrationConfig.dataset.")
        if calibration.exclude_attention_sinks or calibration.adapter_dataset is not None:
            # Norm tuning distills whole-sequence logits; it has no per-row
            # calibration statistics and no adapter pass.
            raise ValueError(
                "Norm tuning does not support CalibrationConfig.exclude_attention_sinks "
                "or CalibrationConfig.adapter_dataset."
            )

        names = self.norm_module_names(norm.types, norm.skip)
        if not names:
            raise ValueError(
                f"{type(self).__name__} declares no `:norm` modules matching {[t.value for t in norm.types]} in its "
                "module_tree; norm tuning is unavailable."
            )

        if tokenizer is not None and not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`."
            )

        teacher = norm.teacher
        if not isinstance(teacher, nn.Module):
            # Resolve the config only: the weights are loaded by
            # `_requantize_norm` after the embeddings, but a bad path/id has to
            # fail here.
            # Same resolution `_requantize_norm` uses: vendored config for a
            # supported model_type, AutoConfig only for unsupported ones.
            load_model_config(
                teacher,
                trust_remote_code=resolve_trust_remote_code(
                    teacher, trust_remote_code=getattr(self, "trust_remote_code", False)
                ),
            )
        return names

    def _requantize_norm(
        self,
        *,
        calibration: CalibrationConfig,
        norm: NormConfig,
        tokenizer: Optional[PreTrainedTokenizerBase],
        names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Tune tagged norm parameters against an unquantized teacher; packed tensors stay untouched."""
        from ..quantization.norm_tuning import tune_norm_params

        if names is None:
            names = self._validate_requantize_norm(
                calibration=calibration, norm=norm, tokenizer=tokenizer
            )

        if tokenizer is not None:
            self.tokenizer = Tokenicer.load(
                tokenizer,
                trust_remote_code=self.trust_remote_code,
                model_config=getattr(self.model, "config", None),
            )

        batches = self.prepare_dataset(
            calibration_dataset=calibration.dataset,
            calibration_dataset_concat_size=calibration.concat_size,
            calibration_dataset_sort=calibration.sort,
            batch_size=self._calibration_batch_size(calibration),
            calibration_data_min_length=calibration.min_length,
            calibration_concat_separator=calibration.concat_separator,
        )

        teacher = norm.teacher
        owns_teacher = not isinstance(teacher, nn.Module)
        if owns_teacher:
            device = get_device(self.get_input_embeddings())
            # The teacher is a GPT-QModel-owned transformers construction, so it
            # takes the same load lock as every other build: another thread
            # must not be inside its own ``post_init`` while this one
            # constructs, or the class-wide no-tie patch leaves the shell
            # untied (#5421). The construction window still holds the
            # global-state writer by itself (#5304), so the teacher's weight
            # I/O stays concurrent with optimized forwards.
            teacher_trust = resolve_trust_remote_code(teacher, trust_remote_code=self.trust_remote_code)
            teacher_config = load_model_config(teacher, trust_remote_code=teacher_trust)
            teacher_loader = local_model_loader_for_config(teacher_config.to_dict()) or AutoModelForCausalLM
            with model_load_lock():
                teacher = teacher_loader.from_pretrained(
                    teacher,
                    config=teacher_config,
                    dtype=self.model.dtype,
                    device_map={"": device},
                    trust_remote_code=teacher_trust,
                )
        try:
            stats = tune_norm_params(self.model, teacher, batches, names, norm)
        finally:
            if owns_teacher:
                del teacher
                torch_empty_cache()

        # Persist only the tuned norm tensors; every other checkpoint tensor is kept byte-identical.
        self._module_replacement_prefixes = set(getattr(self, "_module_replacement_prefixes", set())) | set(stats["modules"])
        if self.load_quantized_model:
            self._model_free_weight_only_replacements_only = True
        return stats

    @staticmethod
    def _embedding_endpoint_state(module) -> Tuple[Optional[torch.device], Optional[torch.dtype]]:
        """The device and floating dtype a replaced embedding endpoint occupied."""

        weight = getattr(module, "weight", None)
        dtype = None
        if isinstance(weight, torch.Tensor) and weight.is_floating_point():
            dtype = weight.dtype
        return get_device(module), dtype

    def _snapshot_module_placement(self) -> Dict[str, torch.device]:
        """Record the device of every leaf module that owns tensors."""

        placement: Dict[str, torch.device] = {}
        model = getattr(self, "model", None)
        if model is None:
            return placement
        for name, module in model.named_modules():
            if not name:
                continue
            owns_tensors = any(
                True for _ in module.parameters(recurse=False)
            ) or any(True for _ in module.buffers(recurse=False))
            if not owns_tensors:
                continue
            placement[name] = get_device(module)
        return placement

    def _rehome_requantized_endpoints(
        self,
        endpoint_state: Dict[str, Tuple],
        placement: Optional[Dict[str, torch.device]] = None,
    ) -> None:
        """Place requantized endpoints back on the device and dtype they replaced.

        The looper packs on CPU and a quantized embedding's scales keep the
        checkpoint dtype, so without this the in-session model cannot run a
        forward and a BF16 checkpoint reloads with FP16 embeddings (#5116).
        Layers the loop returned to CPU are also moved back to where they were.
        """

        if not endpoint_state and not placement:
            return
        live_model = getattr(self, "model", None)
        if live_model is None:
            return

        model_dtype = getattr(live_model, "dtype", None)
        if not isinstance(model_dtype, torch.dtype) or not model_dtype.is_floating_point:
            model_dtype = None

        # Restore the pre-run placement of the modules the loop offloaded. The
        # endpoints are handled below with their dtype, so skip them here.
        for name, device in (placement or {}).items():
            if name in endpoint_state:
                continue
            if device is None or device.type == "meta":
                continue
            module = get_module(live_model, key=name)
            if module is None:
                continue
            if get_device(module) != device:
                safe_module_to(module, device)

        for name, (device, dtype) in endpoint_state.items():
            module = get_module(live_model, key=name)
            if module is None:
                continue
            setter = getattr(module, "set_embedding_output_dtype", None)
            target_dtype = dtype if dtype is not None else model_dtype
            if callable(setter) and target_dtype is not None:
                setter(target_dtype)
            if device is None or device.type == "meta":
                continue
            if get_device(module) != device:
                safe_module_to(module, device)

    def _requantize_embeddings(
        self,
        *,
        calibration: CalibrationConfig,
        embed_quant_config: QuantizeEmbedConfig,
        tokenizer: Optional[PreTrainedTokenizerBase],
        backend: Optional[BACKEND],
        adapter: Adapter,
    ):
        # Quantized checkpoint loading may use only a device map, leaving the
        # quantization device unset. Requantization still needs a concrete
        # device for kernel selection and calibration replay.
        quantize_config = getattr(self, "quantize_config", None)
        if quantize_config is not None and quantize_config.device is None:
            input_endpoint = self.get_input_embeddings()
            endpoint = input_endpoint if input_endpoint is not None else self.get_output_embeddings()
            endpoint_device = get_device(endpoint) if endpoint is not None else get_device(self.model)
            if endpoint_device.type == "meta":
                endpoint_device = CPU
            quantize_config.device = normalize_device(endpoint_device)

        # A requantized endpoint is rebuilt on CPU during packing, and its
        # scales keep the checkpoint dtype (FP16 for GPTQ). Record what each
        # replaced endpoint occupied so it can be placed back afterwards; the
        # first in-session forward otherwise fails on a device mismatch, and a
        # BF16 model receives FP16 embeddings (#5116).
        mode = embed_quant_config.embed_quant_mode
        # The loop returns every processed layer to CPU (`post_quantize`), which
        # is right for a fresh quantize but leaves a requantized *loaded* model
        # unrunnable on the device it came from. Record the placement of the
        # whole model so it can be restored after the run.
        placement = self._snapshot_module_placement()
        endpoint_state = {}
        live_model = getattr(self, "model", None)
        if live_model is not None:
            if mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH):
                input_name = self.get_input_embeddings_name()
                if input_name:
                    input_module = get_module(live_model, key=input_name)
                    if input_module is not None:
                        endpoint_state[input_name] = self._embedding_endpoint_state(input_module)
            if mode in (QuantizeEmbed.OUTPUT, QuantizeEmbed.BOTH):
                output_name = self.get_output_embeddings_name() or getattr(self, "lm_head", None)
                if output_name:
                    output_module = get_module(live_model, key=output_name)
                    if output_module is not None:
                        endpoint_state[output_name] = self._embedding_endpoint_state(output_module)

        result = self.quantize(
            calibration=calibration,
            tokenizer=tokenizer,
            backend=backend,
            adapter=adapter,
            embed_quant_config=embed_quant_config,
        )

        self._rehome_requantized_endpoints(endpoint_state, placement)

        # A loaded quantized model may contain inference-repacked or fused
        # modules that cannot safely be serialized as a complete checkpoint.
        # Preserve the source checkpoint and let the module-replacement writer
        # replace only the shards owned by the requested endpoints.
        prefixes = set(getattr(self, "_module_replacement_prefixes", set()))
        if mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH):
            input_name = self.get_input_embeddings_name()
            if input_name:
                prefixes.add(input_name)
        if mode in (QuantizeEmbed.OUTPUT, QuantizeEmbed.BOTH):
            output_name = self.get_output_embeddings_name() or self.lm_head
            if output_name:
                prefixes.add(output_name)
        self._module_replacement_prefixes = prefixes

        if self.load_quantized_model and prefixes:
            self._model_free_weight_only_replacements_only = True

        return result

    def _prepare_calibration_before_rotation(
        self,
        *,
        calibration: CalibrationConfig,
        batch_size: int,
        field: str,
    ):
        """Validate and prepare one calibration input before the model rotates.

        Mirrors the checks ``LoopProcessor.__init__`` applies, but runs before
        ``fuse_layer_norms``/``rotate_model`` so a calibration-data error cannot
        leave the in-session model rotated (#4935). The returned batches are the
        exact ones the processors would have prepared, so ``quantize()`` reuses
        them instead of tokenizing the corpus twice.
        """

        from ..looper.loop_processor import (
            _calibration_dataset_length,
            _sized_calibration_dataset,
        )

        dataset = getattr(calibration, field)
        if dataset is None:
            if field == "dataset":
                raise ValueError(
                    "CalibrationConfig.dataset is required unless a weight-only quantize config is configured."
                )
            return None
        # Match the processor's own empty-input error instead of the lower-level
        # ``prepare_calibration_dataset`` message.
        if _calibration_dataset_length(dataset) == 0:
            raise ValueError("Calibration dataset must not be empty.")

        prepared = self.prepare_dataset(
            calibration_dataset=dataset,
            calibration_dataset_concat_size=calibration.concat_size,
            calibration_dataset_sort=calibration.sort,
            batch_size=batch_size,
            calibration_data_min_length=calibration.min_length,
            calibration_concat_separator=calibration.concat_separator,
        )
        prepared = _sized_calibration_dataset(prepared)
        if len(prepared) == 0:
            raise ValueError(
                "Calibration dataset is empty after CalibrationConfig.prepare_dataset removed "
                "all samples (for example, every input_ids was shorter than "
                "CalibrationConfig.min_length). Lower CalibrationConfig.min_length "
                "or provide longer calibration samples."
            )
        return prepared

    def _quantize_with_calibration(
        self,
        *,
        calibration: CalibrationConfig,
        backend: Optional[BACKEND],
        embed_quant_config: Optional[QuantizeEmbedConfig],
        checkpoint: Optional["CheckpointConfig"] = None,
        run_plan: Optional[QuantizationRunPlan] = None,
        prepared_dataset=None,
        prepared_adapter_dataset=None,
    ):
        from ..looper.loop_processor import (
            _reusable_calibration_dataset,
            prepare_calibration_dataset_once,
            release_failed_run_processors,
        )
        from ..looper.module_looper import ModuleLooper

        # A one-shot iterator must be materialised once here or a later
        # traversal sees an exhausted dataset (issue #4307); the batches
        # themselves are prepared once below and shared by every processor
        # (#4869). A rotation run already prepared (and validated) its input
        # before the model was mutated, so reuse those batches instead (#4935).
        if prepared_dataset is not None:
            dataset = prepared_dataset
        else:
            dataset = _reusable_calibration_dataset(calibration.dataset)
        if prepared_adapter_dataset is not None:
            adapter_dataset = prepared_adapter_dataset
        else:
            adapter_dataset = _reusable_calibration_dataset(calibration.adapter_dataset)

        if run_plan is None:
            from ..adapter.adapter import Lora

            run_plan = QuantizationRunPlan.build(
                self.quantize_config,
                format_code=resolve_quant_format(
                    getattr(self.quantize_config, "format", FORMAT.GPTQ), self.quantize_config.method
                ),
                pack_backend=backend,
                loop_backend=backend,
                batch_size=calibration.batch_size,
                needs_lora=isinstance(self.quantize_config.adapter, Lora),
            )
        if run_plan.weight_only:
            raise ValueError("calibrated quantization requires a calibrated run plan")

        # The run plan owns the effective batch size (a model can disable
        # batched calibration); the adapter samples are consumed only by EoRA.
        # Rotation runs hand the processors batches that were already prepared
        # before the mutation; anything else keeps the model's preparer.
        prepare_dataset = (
            _already_prepared_calibration_dataset
            if prepared_dataset is not None
            else self.prepare_dataset
        )
        calibration_config = calibration.bind(
            prepare_dataset,
            dataset=dataset,
            adapter_dataset=None,
            batch_size=run_plan.batch_size,
        )
        # Prepare the calibration batches once for the whole run: GPTAQ/FOEM
        # pair the native and quantized captures by batch index, so two
        # independent preparations of `sort="shuffle"` would pair unrelated
        # sequences (#4869). Every calibrated processor replays this list, and
        # a rotation run keeps the batches it already prepared (#4935).
        if prepared_dataset is None:
            prepared_dataset = prepare_calibration_dataset_once(calibration_config)
        if (
            prepared_adapter_dataset is None
            and adapter_dataset is not None
            and ProcessorStep.EORA in run_plan.processors
        ):
            prepared_adapter_dataset = prepare_calibration_dataset_once(
                calibration_config.bind(self.prepare_dataset, dataset=adapter_dataset)
            )
        calibration_config = calibration.bind(
            _already_prepared_calibration_dataset,
            dataset=prepared_dataset,
            adapter_dataset=None,
            batch_size=run_plan.batch_size,
        )
        args = {
            "tokenizer": self.tokenizer,
            "qcfg": self.quantize_config,
            "calibration": calibration_config,
            "calculate_w_wq_diff": ProcessorStep.EORA in run_plan.processors,
        }
        # EoRA consumes state stored on the NamedModule, so the layer inventory
        # must keep that wrapper instead of unwrapping it to the live module.
        self._has_eora = bool(args["calculate_w_wq_diff"])
        # GPTQ's EoRA path needs ``w - wq`` in FP32. Keep the packed NF12 source
        # alive until the processor consumes it so the difference fuses the
        # widening decode with the subtraction instead of materializing an FP32
        # source copy. Other quantizers compute their diff in the source dtype,
        # so they keep the existing release-on-restore behavior.
        self._needs_w_wq_diff = (
            self._has_eora
            and ProcessorStep.GPTQ in run_plan.processors
        )

        processors = []
        try:
            for step in run_plan.processors:
                factory = processor_class(step)
                if step is ProcessorStep.NATIVE:
                    native_args = dict(args)
                    native_args.pop("calculate_w_wq_diff")
                    processors.append(factory(**native_args))
                elif step is ProcessorStep.PREPROCESSOR:
                    processors.append(
                        factory(
                            tokenizer=self.tokenizer,
                            qcfg=self.quantize_config,
                            calibration=CalibrationConfig(),
                        )
                    )
                elif step is ProcessorStep.EORA:
                    # EoRA replays the same prepared batches (or its own prepared
                    # adapter samples). Its rows are not quantizer statistics, so
                    # the attention-sink row policy does not apply to them.
                    eora_calibration = calibration_config.bind(
                        _already_prepared_calibration_dataset,
                        dataset=(
                            prepared_adapter_dataset
                            if prepared_adapter_dataset is not None
                            else prepared_dataset
                        ),
                        exclude_attention_sinks=False,
                    )
                    processors.append(factory(
                        tokenizer=self.tokenizer,
                        qcfg=self.quantize_config,
                        calibration=eora_calibration,
                    ))
                elif step is ProcessorStep.EXL3:
                    exl3_args = {key: value for key, value in args.items() if key != "calculate_w_wq_diff"}
                    processors.append(factory(**exl3_args, lm_head_name=self.lm_head))
                elif step is ProcessorStep.QQQ:
                    processors.append(factory(**args, backend=run_plan.loop_backend))
                elif step is ProcessorStep.MACAQ:
                    processors.append(factory(**args, backend=run_plan.pack_backend))
                elif step in (ProcessorStep.AWQ, ProcessorStep.PARO):
                    processors.append(factory(
                        **args, gptq_model=self, model=self.model,
                    ))
                else:
                    processors.append(factory(**args))

            gc_context = (
                DEVICE_THREAD_POOL.no_auto_gc()
                if self.quantize_config.gc_mode == GcMode.ON_STAGE_END
                else nullcontext()
            )

            from ..looper.gptq_checkpoint import checkpoint_session

            checkpoint_context = checkpoint_session(checkpoint, self) if checkpoint is not None else nullcontext()
            with gc_context, checkpoint_context as extension:
                if self.quantize_config.activation_mode in {"w4afp8", "w4a_nvfp4"}:
                    from ..nn_modules.qlinear.w4a_llama_replay import install_w4a_llama_replay

                    install_w4a_llama_replay(self.model, self.quantize_config)
                module_looper = ModuleLooper(
                    self, processors=processors, embed_quant_config=embed_quant_config,
                    extensions=(extension,) if extension is not None else (),
                    disk_write_expected=(
                        bool(self.quantize_config.offload_to_disk) or checkpoint is not None
                    ),
                )
                use_cache = getattr(self.model.config, "use_cache", False)
                try:
                    return module_looper.loop(backend=run_plan.loop_backend, fallback=self.quantize_config.fallback)
                finally:
                    module_looper.layer_module_coverage.report(type(self).__name__)
                    # Keep the run lease until every worker has stopped accessing
                    # the attempt directory, including exceptional/stop exits.
                    if extension is not None:
                        try:
                            DEVICE_THREAD_POOL.wait()
                            module_looper.wait_dangling_threads()
                        finally:
                            self.model.config.use_cache = use_cache
        except BaseException:
            # A failure before or around ``ModuleLooper.loop()`` never reaches
            # ``finalize()``, so the processors already built must release
            # their per-run resources (Device-SMI sampler threads, workspace)
            # here or they outlive the attempt (#4862). The loop's own failure
            # branch releases again; that is idempotent.
            release_failed_run_processors(processors)
            raise

    def _quantize_weight_only(
        self,
        *,
        checkpoint=None,
        calibration: CalibrationConfig,
        backend: Optional[BACKEND],
        embed_quant_config: Optional[QuantizeEmbedConfig],
        run_plan: Optional[QuantizationRunPlan] = None,
        measure_weight_error: bool = False,
    ):
        from ..looper.loop_processor import release_failed_run_processors
        from ..looper.weight_only_looper import WeightOnlyLooper

        if run_plan is None:
            from ..adapter.adapter import Lora

            run_plan = QuantizationRunPlan.build(
                self.quantize_config,
                format_code=resolve_quant_format(self.quantize_config.format, self.quantize_config.method),
                pack_backend=backend,
                loop_backend=backend,
                batch_size=calibration.batch_size,
                needs_lora=isinstance(self.quantize_config.adapter, Lora),
            )
        if run_plan.processors != (ProcessorStep.WEIGHT_ONLY,):
            raise ValueError("weight-only quantization requires a weight-only run plan")

        sketches = None
        if self.quantize_config.method == METHOD.CPTC and self.quantize_config.requires_calibration_dataset():
            from ..quantization.cptc_hessian import collect_model_hessian_sketches

            batches = self._prepare_calibration_before_rotation(
                calibration=calibration, batch_size=run_plan.batch_size, field="dataset",
            )
            sketches = collect_model_hessian_sketches(
                self, batches, exclude_attention_sinks=calibration.exclude_attention_sinks,
            )
        elif calibration.dataset is not None:
            log.info("Weight-only quantization selected; ignoring provided calibration dataset.")

        processor = None
        try:
            processor = processor_class(ProcessorStep.WEIGHT_ONLY)(
                tokenizer=self.tokenizer,
                qcfg=self.quantize_config,
                measure_weight_error=measure_weight_error,
                cptc_hessian_sketches=sketches,
            )

            gc_context = (
                DEVICE_THREAD_POOL.no_auto_gc()
                if self.quantize_config.gc_mode == GcMode.ON_STAGE_END
                else nullcontext()
            )

            from ..looper.gptq_checkpoint import checkpoint_session

            with gc_context, (checkpoint_session(checkpoint, self) if checkpoint else nullcontext()) as extension:
                module_looper = WeightOnlyLooper(
                    model=self, processor=processor, embed_quant_config=embed_quant_config,
                    extensions=(extension,) if extension else (),
                )
                try:
                    return module_looper.loop(backend=run_plan.loop_backend)
                finally:
                    module_looper.layer_module_coverage.report(type(self).__name__)
                    if checkpoint:
                        DEVICE_THREAD_POOL.wait()
        except BaseException:
            # The weight-only processor also owns Device-SMI sampler threads;
            # a failure before ``loop()`` never reaches ``finalize()`` (#4862).
            release_failed_run_processors((processor,))
            raise

    def _eora_generate(
        self,
        # eora adapter generation needs config Lora(rank=1, path='lora.safetensors')
        adapter: Adapter,
        quantized_modules: Dict[str, TorchLinear],
        calibration: CalibrationConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        if not isinstance(calibration, CalibrationConfig):
            raise TypeError(
                f"EoRA generation requires a CalibrationConfig, got {type(calibration).__name__}."
            )
        if calibration.dataset is None:
            raise ValueError("EoRA generation requires CalibrationConfig.dataset.")
        if calibration.exclude_attention_sinks or calibration.adapter_dataset is not None:
            # Post-quantization generation has a single calibration pass and
            # accumulates no quantizer statistics.
            raise ValueError(
                "EoRA generation uses CalibrationConfig.dataset only; "
                "exclude_attention_sinks and adapter_dataset do not apply."
            )
        if self.quantized:
            raise EnvironmentError("eora_generate() is called a model that is already quantized")

        calibration = _materialized_calibration_config(calibration)

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                # TODO FIX ME...this is a bug
                self.tokenizer = Tokenicer.load(
                    tokenizer,
                    trust_remote_code=self.trust_remote_code,
                    model_config=getattr(self.model, "config", None),
                )
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        from ..adapter.adapter import Lora
        from ..looper.dequantize_processor import DequantizeProcessor
        from ..looper.eora_processor import EoraProcessor
        from ..looper.loop_processor import release_failed_run_processors
        from ..looper.module_looper import ModuleLooper
        from ..looper.module_preprocessor import ModulePreProcessor

        self.quantize_config.adapter = adapter

        assert isinstance(self.quantize_config.adapter, Lora)

        # EoRA generation consumes and rewrites dense native weights: it needs
        # `w - wq` and installs `wq + BA` back into each module. AUTO_NF12
        # replaces the native linear with a packed wrapper that has no
        # `weight` attribute, so the generator runs on the dense source
        # instead. This is a generation-time layout choice, not a change to the
        # saved adapter dtype or to the quantized checkpoint.
        if self.quantize_config.auto_dtype_cast == AutoDtypeCast.AUTO_NF12:
            log.warn(
                "EoRA generation disables AUTO_NF12 for this pass: the adapter "
                "generator rewrites dense native weights (`w - wq`, then "
                "`wq + BA`) and cannot do that through the packed NF12 wrapper. "
                "The saved adapter is unaffected."
            )
            self.quantize_config.auto_dtype_cast = AutoDtypeCast.OFF

        # init processor with EoRA processor
        calibration_config = calibration.bind(
            self.prepare_dataset,
            batch_size=self._calibration_batch_size(calibration),
        )
        processors = []
        try:
            if getattr(self.quantize_config, "preprocessors", None):
                processors.append(
                    ModulePreProcessor(
                        tokenizer=self.tokenizer,
                        qcfg=self.quantize_config,
                    ),
                )
            processors.extend(
                [
                    DequantizeProcessor(
                        quantized_modules=quantized_modules,
                        qcfg=self.quantize_config,
                    ),
                    EoraProcessor(
                        tokenizer=self.tokenizer,
                        qcfg=self.quantize_config,
                        calibration=calibration_config,
                    ),
                ]
            )

            # prepare processor worker (looper)
            module_looper = ModuleLooper(model=self, processors=processors)

            try:
                module_looper.loop()
            finally:
                # The other two looper entry points report coverage in a `finally`
                # too; EoRA generation could otherwise skip every routed expert
                # without a warning.
                module_looper.layer_module_coverage.report(type(self).__name__)
        except BaseException:
            # EoRA generation builds processors before its looper; a rejected
            # calibration there must release them too (#4862).
            release_failed_run_processors(processors)
            raise

        self.eora_save(save_dir=adapter.path, model_save_dir=self.model_local_path)
        return

    def to(self, device: Union[str, torch.device]):
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
            return self
        else:
            raise NotImplementedError(
                f"{self.model.__class__.__name__} does not support the to() method"
            )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def move_input_capture_example(
        self,
        example: Dict[str, Any],
        data_device: torch.device,
    ) -> Dict[str, Any]:
        for key, value in example.items():
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if not torch.is_tensor(item):
                        continue

                    if item.ndim == 1:
                        item = item.unsqueeze(0)

                    value[index] = move_to(item, device=data_device)
            elif torch.is_tensor(value):
                if value.ndim == 1:
                    value = value.unsqueeze(0)

                example[key] = move_to(value, device=data_device)

        if getattr(self, "calibration_inputs", None) == "tensor_kwargs":
            # These rows are the user's own tensors; a bf16 model fed fp32 rows
            # fails inside the first matmul, far from the mistake.
            example = _cast_floating_inputs(
                example, _module_weight_dtype(getattr(self, "model", None))
            )

        return self.finalize_input_capture_example(example)

    def finalize_input_capture_example(
        self,
        example: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.ATTENTION_MASKS_DTYPE is torch.long and "attention_mask" in example:
            example["attention_mask"] = example["attention_mask"].long()

        return example

    def run_input_capture(
        self,
        example: Dict[str, Any],
        use_cache: bool,
        data_device: torch.device,
    ):
        input_ids = example.get("input_ids")
        if torch.is_tensor(input_ids):
            input_ids = self._sanitize_input_ids_for_embeddings(input_ids)
            if input_ids is not example.get("input_ids"):
                example = dict(example)
                example["input_ids"] = input_ids

        if self.INPUT_EMBEDDING_EXTRA_ARGS:
            return self.model.generate(
                **example,
                **self.INPUT_EMBEDDING_EXTRA_ARGS,
            )

        return self.model(
            **_with_capture_position_ids(
                self.model,
                example,
                derive=getattr(self, "CAPTURE_DERIVE_POSITION_IDS", None),
            ),
            use_cache=use_cache,
        )

    def _sanitize_input_ids_for_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.int32, torch.int64):
            return input_ids

        if input_ids.numel() == 0:
            return input_ids

        embedder = getattr(self.model, "get_input_embeddings", None)
        if not callable(embedder):
            return input_ids

        # Some multimodal wrappers expose get_input_embeddings but intentionally
        # raise NotImplementedError at runtime; skip sanitization in that case.
        try:
            embedding = embedder()
        except NotImplementedError:
            return input_ids
        except Exception:
            return input_ids
        if embedding is None or not hasattr(embedding, "num_embeddings"):
            return input_ids

        vocab_size = int(getattr(embedding, "num_embeddings", 0) or 0)
        if vocab_size <= 0:
            return input_ids

        invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
        if not bool(invalid_mask.any()):
            return input_ids

        tokenizer = getattr(self, "tokenizer", None)
        replacement_token_id = None
        for attr_name in ("unk_token_id", "pad_token_id", "eos_token_id", "bos_token_id"):
            candidate = getattr(tokenizer, attr_name, None) if tokenizer is not None else None
            if isinstance(candidate, int) and 0 <= candidate < vocab_size:
                replacement_token_id = candidate
                break
        if replacement_token_id is None:
            replacement_token_id = 0

        sanitized = input_ids.clone()
        sanitized[invalid_mask] = replacement_token_id

        invalid_count = int(invalid_mask.sum().item())
        log.warning(
            "Input capture: replaced %s out-of-range token ids with %s (valid range: [0, %s)).",
            invalid_count,
            replacement_token_id,
            vocab_size,
        )
        return sanitized

    def _generate_with_runtime(self, runtime_generate, inputs=None, **kwargs):
        def _normalize_generate_attention_mask(input_ids, attention_mask):
            if not torch.is_tensor(attention_mask) or attention_mask.ndim <= 2:
                return attention_mask

            seq_len = None
            if torch.is_tensor(input_ids) and input_ids.ndim >= 2:
                seq_len = input_ids.shape[-1]

            return normalize_seq_mask(attention_mask, seq_len=seq_len)

        if isinstance(inputs, str) or (isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)):
            kwargs.setdefault("prompts", inputs)
        elif hasattr(inputs, "get") and not torch.is_tensor(inputs):
            merged_kwargs = dict(inputs)
            merged_kwargs.update(kwargs)
            kwargs = merged_kwargs
        elif inputs is not None:
            kwargs.setdefault("input_ids", inputs)

        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = _normalize_generate_attention_mask(
                kwargs.get("input_ids"),
                kwargs["attention_mask"],
            )

        return runtime_generate(self.model, **kwargs)

    def _kernels_support_compile(self) -> bool:
        """Whether a compiled generation path can beat this model's eager forward.

        Transformers compiles the forward pass of a static-cache ``generate()``
        (``CompileConfig``, ``mode="reduce-overhead"``). That only pays off
        when the selected kernels are traceable. A custom op without a fake
        implementation, or a host-side ``data_ptr()``/``id()``/``._version``
        guard inside ``forward``, breaks the graph on every quantized linear,
        so CUDA-graph capture never covers a layer and each decode step pays
        for the fragments and guard checks: the static cache then runs 2-4x
        slower than eager (#4669). Kernels declare the answer with
        ``BaseQuantLinear.SUPPORTS_COMPILE``, which defaults to False and
        requires both a ``fullgraph=True`` trace with no graph breaks and a
        measured win on the hardware it claims (#4739). A kernel that declares
        a compile-capable ``SUPPORTS_COMPILE_LEVEL`` (``>= INDUCTOR``) yet
        keeps the policy off has to record the measured loss in
        ``SUPPORTS_COMPILE_EAGER_REASON``; otherwise the two declarations drift
        apart and a trace-clean kernel stays 2-9x slower than it could be
        (#4984).

        A model with no quantized linear (dense) has nothing that breaks the
        graph, so it keeps whatever compile policy the caller chose.
        """

        # ``optimize()`` wraps the model in an ``OptimizedModule``; the kernels
        # live on the original module.
        model = getattr(self.model, "_orig_mod", self.model)
        if not isinstance(model, nn.Module):
            # Nothing to inspect (a test double or a foreign runtime): keep the
            # caller's policy instead of guessing.
            return True
        # The native EXL3 kernel is a plain `nn.Module`, not a `BaseQuantLinear`,
        # so it has to be named here or a native EXL3 model looks dense and
        # keeps the auto-compile its `_ensure_inner` RLock breaks on every
        # layer (#4917). It declares no `SUPPORTS_COMPILE`, so the lookup below
        # answers False for it.
        modules = find_modules(
            model,
            layers=[BaseQuantLinear, ExllamaV3Linear, ExllamaV3TorchLinear],
        )
        if not modules:
            return True
        return all(supports_compiled_generation(module) for module in modules.values())

    def _generation_compile_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Return the generation kwargs to pass, with the compile policy applied.

        Transformers enables auto-compile for a static-cache generation unless
        the generation config says otherwise. With graph-breaking kernels that
        default is a pessimization (#4669), so it is turned off for this call.
        Every explicit caller choice wins: ``disable_compile`` as a keyword or
        on a supplied ``generation_config``, and an explicit ``compile_config``
        (the caller opted into compiling). The policy never mutates the
        caller's ``generation_config``.
        """

        if kwargs.get("disable_compile") is not None:
            return kwargs
        generation_config = kwargs.get("generation_config")
        if generation_config is not None:
            if getattr(generation_config, "disable_compile", None) is not None:
                return kwargs
            if getattr(generation_config, "compile_config", None) is not None:
                return kwargs
        elif kwargs.get("compile_config") is not None:
            return kwargs
        if self._kernels_support_compile():
            return kwargs

        log.warn.once(
            "Generation: the selected quantized kernels break `torch.compile` graphs, so the "
            "static-cache auto-compile is disabled for this call; it would be slower than eager. "
            "Pass `disable_compile=False` or a `compile_config` to compile anyway (#4669)."
        )
        merged = dict(kwargs)
        if generation_config is not None:
            config = copy.deepcopy(generation_config)
            config.disable_compile = True
            merged["generation_config"] = config
        else:
            merged["disable_compile"] = True
        return merged

    def generate(self, inputs=None, **kwargs):
        # Torch/XLA represents a slice as an in-place view update during graph
        # construction. Inference tensors cannot receive a version counter, so
        # ``torch.inference_mode`` fails before model math on the TT PJRT path.
        # no_grad keeps autograd disabled without changing tensor semantics.
        #
        # The same constraint applies to an Accelerate CPU-offload hook: its
        # per-forward ``Module._apply`` materializes parameters and buffers, and
        # under ``torch.inference_mode`` those become inference tensors without
        # a version counter. Dynamo reads version counters when it traces the
        # forward that ``optimize()`` compiled, so the first compiled
        # ``generate()`` crashed with "Inference tensors do not track version
        # counter" (#5099). ``no_grad`` disables autograd without changing the
        # moved tensors' semantics, so the offloaded layer keeps its counter.
        # Models without an offload hook keep inference_mode.
        model_device = getattr(self.model, "device", torch.device("cpu"))
        if getattr(model_device, "type", None) == "xla" or has_cpu_offload_hooks(self.model):
            generate_context = torch.no_grad()
        else:
            generate_context = torch.inference_mode()
        with generate_context:
            # fix hf generate not applying correct pad token
            pad_token_id = kwargs.get("pad_token_id", None)
            if pad_token_id is None and self.tokenizer:
                kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            runtime_generate = getattr(self, "_runtime_generate", None)
            if runtime_generate is not None:
                return self._generate_with_runtime(runtime_generate, inputs=inputs, **kwargs)

            def _normalize_generate_attention_mask(input_ids, attention_mask):
                if not torch.is_tensor(attention_mask) or attention_mask.ndim <= 2:
                    return attention_mask

                seq_len = None
                if torch.is_tensor(input_ids) and input_ids.ndim >= 2:
                    seq_len = input_ids.shape[-1]

                return normalize_seq_mask(attention_mask, seq_len=seq_len)

            if isinstance(inputs, str) or (isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)):
                if self.tokenizer is None:
                    raise ValueError("You passed in an `input` to `generate()` of type `str` but model is missing `model.tokenizer`. Please set `model.tokenizer = my_tokenizer`.")
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, padding_side="left")
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = _normalize_generate_attention_mask(
                        inputs.get("input_ids"),
                        inputs["attention_mask"],
                    )
                inputs = inputs.to(self.model.device)
                return self.model.generate(**inputs, **self._generation_compile_kwargs(kwargs))

            if hasattr(inputs, "get") and not torch.is_tensor(inputs):
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = _normalize_generate_attention_mask(
                        inputs.get("input_ids"),
                        inputs["attention_mask"],
                    )
                # A caller may put generation options in the mapping, so the
                # policy has to see both sides before it injects a default.
                return self.model.generate(**self._generation_compile_kwargs({**inputs, **kwargs}))

            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = _normalize_generate_attention_mask(
                    kwargs.get("input_ids", inputs),
                    kwargs["attention_mask"],
                )

            return self.model.generate(inputs=inputs, **self._generation_compile_kwargs(kwargs))

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def save(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
            meta_quantizer: Optional[str] = None,
            eora_path: Optional[str] = None,
            split_by: Optional[str] = None,
            **kwargs,
    ):
        timer = getattr(self, "quant_region_timer", None)
        start_time = time.perf_counter() if timer else None

        try:
            if self.quantized:
                # The quantized branch has no `save_pretrained` passthrough, so
                # any extra keyword used to be discarded without a word — a
                # typo of a real parameter, or an HF option such as
                # `push_to_hub=True`, reported success and did nothing (#4686).
                # This is the save-side counterpart of #4052/#4057.
                if kwargs:
                    accepted = (
                        "save_dir",
                        "safetensors_metadata",
                        "max_shard_size",
                        "meta_quantizer",
                        "eora_path",
                        "split_by",
                    )
                    details = []
                    for key in sorted(kwargs):
                        matches = difflib.get_close_matches(key, accepted, n=1)
                        details.append(
                            f"`{key}`" + (f" (did you mean `{matches[0]}`?)" if matches else "")
                        )
                    raise TypeError(
                        "GPTQModel.save: unexpected keyword argument(s) for a quantized "
                        "model: "
                        + ", ".join(details)
                        + ". A quantized save accepts "
                        + ", ".join(f"`{name}`" for name in accepted)
                        + "."
                    )

                # Saving an already quantized model (for example one loaded from
                # an older checkpoint) is the last point where a `dynamic` rule
                # the fused engines cannot honor can be reported.
                self._warn_fused_unit_dynamic_overrides()

                # Safetensors is unable to save tied weights, so we untie them here. Reference: https://github.com/huggingface/safetensors/issues/202
                #untie_weights(self.model)

                if getattr(self, "_model_free_weight_only_replacements_only", False):
                    # `save_quantized_embeddings` writes a module-replacement
                    # checkpoint and has no adapter or split support, so it used
                    # to drop two of `save()`'s own named parameters (#4686).
                    unsupported = [
                        name
                        for name, value in (("eora_path", eora_path), ("split_by", split_by))
                        if value is not None
                    ]
                    if unsupported:
                        raise ValueError(
                            "GPTQModel.save: "
                            + ", ".join(f"`{name}`" for name in unsupported)
                            + " is not supported when saving a weight-only replacement "
                            "(module-replacement) checkpoint; only `save_dir`, "
                            "`safetensors_metadata` and `meta_quantizer` apply."
                        )
                    self.save_quantized_embeddings(
                        save_dir=save_dir,
                        safetensors_metadata=safetensors_metadata,
                        max_shard_size=max_shard_size,
                        meta_quantizer=meta_quantizer,
                    )
                else:
                    self.save_quantized(
                        save_dir=save_dir,
                        safetensors_metadata=safetensors_metadata,
                        max_shard_size=max_shard_size,
                        meta_quantizer=meta_quantizer,
                        eora_path=eora_path,
                        split_by=split_by)

                # overwrite quant_override_files
                for name, value in self.quant_override_files.items():
                    json_path = os.path.join(save_dir, name)
                    with open(json_path, "w", encoding="utf-8") as f:
                        if isinstance(value, str):
                            f.write(value)
                        else:
                            f.write(json.dumps(value))
            else:
                self.save_pretrained(save_dir=save_dir, **kwargs)
        finally:
            # The per-run offload bundles are *not* released here: the quantized
            # modules stay on meta with an accelerate hook that reads them, so an
            # in-session forward after `save()` still needs them. Teardown, a
            # failed run, or rebinding releases the directory (#4368).
            if timer is not None and start_time is not None:
                try:
                    target = os.path.abspath(save_dir)
                except (TypeError, ValueError, OSError):
                    target = str(save_dir)
                timer.record(
                    "model_save",
                    time.perf_counter() - start_time,
                    source=target,
                )
                timer.flush()

    def _active_auto_module_decoder_config(self) -> Optional[AutoModuleDecoderConfig]:
        """Return the active auto-decoder preprocessor config, if any."""

        # Read the instance dictionary directly: lightweight loader-validation
        # harnesses may not have a quantize_config and BaseQModel.__getattr__
        # otherwise delegates that lookup to the wrapped HF model.
        quantize_config = self.__dict__.get("quantize_config")
        preprocessors = getattr(quantize_config, "preprocessors", None) or []
        for preprocessor in reversed(preprocessors):
            if isinstance(preprocessor, AutoModuleDecoderConfig):
                return preprocessor
        return None

    def materialize_passthrough_modules_for_save(self) -> int:
        """Decode passthrough floatx modules in-place before saving when configured."""

        decoder_cfg = self._active_auto_module_decoder_config()
        if decoder_cfg is None or decoder_cfg.passthrough_save_policy != "decode":
            return 0

        decoded_count = 0
        for module_name, module in list(self.model.named_modules()):
            if isinstance(module, BaseQuantLinear) or not hasattr(module, "weight"):
                continue

            checkpoint_tensors = None
            if isinstance(self.turtle_model, LazyTurtle):
                checkpoint_tensors = self.turtle_model.checkpoint_tensors_for_submodule(
                    target_model=self.model,
                    target_submodule=module,
                    recurse=False,
                    module_path=module_name,
                )
            if not checkpoint_tensors:
                checkpoint_tensors = dict(module.state_dict(keep_vars=True))
            weight = checkpoint_tensors.get("weight")
            if not isinstance(weight, torch.Tensor):
                continue

            decoder_kind = self._decoder_weight_format(
                weight=weight,
                checkpoint_tensors=checkpoint_tensors,
                result_shape=tuple(getattr(module.weight, "shape", weight.shape)),
                module_name=module_name,
            )
            if decoder_kind is None:
                continue

            decoded_module = self._build_decoder_quant_source_module(
                module,
                checkpoint_tensors=checkpoint_tensors,
                target_dtype=decoder_cfg.target_dtype,
                module_name=module_name,
            )
            self._replace_live_submodule(
                module, decoded_module, module_path=module_name
            )
            decoded_count += 1

        return decoded_count

    def materialize_passthrough_modules_for_eval(
        self,
        device: torch.device,
        *,
        respect_forward_policy: bool = False,
    ) -> int:
        """Materialize passthrough floatx modules into live evaluation modules."""

        decoder_cfg = self._active_auto_module_decoder_config()
        if decoder_cfg is None:
            return 0

        target_device = torch.device(device)
        decoded_count = 0
        for module_name, module in list(self.model.named_modules()):
            if isinstance(module, BaseQuantLinear) or not hasattr(module, "weight"):
                continue

            checkpoint_tensors = None
            if isinstance(self.turtle_model, LazyTurtle):
                checkpoint_tensors = self.turtle_model.checkpoint_tensors_for_submodule(
                    target_model=self.model,
                    target_submodule=module,
                    recurse=False,
                    module_path=module_name,
                )
            if not checkpoint_tensors:
                checkpoint_tensors = dict(module.state_dict(keep_vars=True))
            weight = checkpoint_tensors.get("weight")
            if not isinstance(weight, torch.Tensor):
                continue

            decoder_kind = self._decoder_weight_format(
                weight=weight,
                checkpoint_tensors=checkpoint_tensors,
                result_shape=tuple(getattr(module.weight, "shape", weight.shape)),
                module_name=module_name,
            )
            if decoder_kind is None:
                continue

            forward_module = None
            if respect_forward_policy and decoder_cfg.passthrough_forward_policy != "decode":
                if decoder_kind == "fp8" and device_supports_dtype(target_device, weight.dtype, require_validation=False):
                    forward_module = self._build_fp8_forward_module(
                        target_submodule=module,
                        checkpoint_tensors=checkpoint_tensors,
                        device=target_device,
                        target_dtype=decoder_cfg.target_dtype,
                    )
                elif decoder_kind == "fp4" and device_supports_native_fp4(target_device, require_validation=False):
                    forward_module = self._build_fp4_forward_module(
                        target_submodule=module,
                        checkpoint_tensors=checkpoint_tensors,
                        device=target_device,
                        target_dtype=decoder_cfg.target_dtype,
                    )

            if forward_module is None:
                decoded_module = self._build_decoder_quant_source_module(
                    module,
                    checkpoint_tensors=checkpoint_tensors,
                    target_dtype=decoder_cfg.target_dtype,
                    module_name=module_name,
                )
                forward_module = self._build_decoder_forward_module(
                    quant_source=decoded_module,
                    device=target_device,
                )
            self._replace_live_submodule(
                module, forward_module, module_path=module_name
            )
            decoded_count += 1

        return decoded_count

    def decoded_passthrough_state_dict_entries_for_save(self) -> tuple[Dict[str, torch.Tensor], List[str]]:
        """Return dense state-dict entries that should replace native passthrough tensors on save."""

        decoder_cfg = self._active_auto_module_decoder_config()
        if decoder_cfg is None or decoder_cfg.passthrough_save_policy != "decode":
            return {}, []

        decoded_entries: Dict[str, torch.Tensor] = {}
        decoded_prefixes: List[str] = []
        for module_name, module in list(self.model.named_modules()):
            if not module_name or isinstance(module, BaseQuantLinear) or not hasattr(module, "weight"):
                continue

            checkpoint_tensors = None
            if isinstance(self.turtle_model, LazyTurtle):
                checkpoint_tensors = self.turtle_model.checkpoint_tensors_for_submodule(
                    target_model=self.model,
                    target_submodule=module,
                    recurse=False,
                    module_path=module_name,
                )
            if not checkpoint_tensors:
                checkpoint_tensors = dict(module.state_dict(keep_vars=True))
            weight = checkpoint_tensors.get("weight")
            if not isinstance(weight, torch.Tensor):
                continue

            decoder_kind = self._decoder_weight_format(
                weight=weight,
                checkpoint_tensors=checkpoint_tensors,
                result_shape=tuple(getattr(module.weight, "shape", weight.shape)),
                module_name=module_name,
            )
            if decoder_kind is None:
                continue

            decoded_module = self._build_decoder_quant_source_module(
                module,
                checkpoint_tensors=checkpoint_tensors,
                target_dtype=decoder_cfg.target_dtype,
                module_name=module_name,
            )
            decoded_prefixes.append(module_name)
            for key, tensor in decoded_module.state_dict().items():
                decoded_entries[f"{module_name}.{key}"] = tensor.detach().cpu()

        return decoded_entries, decoded_prefixes


    # returns all the loaded qlinear types, returns empty [] if non-found
    def kernels(self) -> List[Type[BaseQuantLinear]]:
        if not isinstance(self.model, nn.Module):
            return []
        loaded_kernels = set()
        modules = find_modules(self.model, layers=[BaseQuantLinear])
        for k, v in modules.items():
            loaded_kernels.add(v.__class__)

        return list(loaded_kernels)

    def _auto_configure_lookahead(self) -> None:
        if not isinstance(self.model, nn.Module):
            return

        quant_modules = [module for module in self.model.modules() if isinstance(module, TorchLinear)]
        if not quant_modules:
            return

        if not any(getattr(module, "_lookahead_enabled", False) for module in quant_modules):
            return

        configure_default_lookahead(self.model)

    def compile(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        log.warn("Deprecation: `model.compile()` is deprecated. Please use `model.optimize()` instead.")
        return self.optimize(backend=backend, mode=mode, fullgraph=fullgraph)

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        """Compile the quantized kernels and the model ``forward``.

        The ``forward`` is compiled in place so ``generate()`` enters the
        compiled path; wrapping ``self.model`` did not, because transformers'
        ``generate`` calls the original module's own ``forward`` (#4669). The
        first ``generate()`` after this call pays the compile cost, and a
        graph-breaking kernel runs slower compiled than eager.

        Compiled calls do not pin the model: dynamo's per-frame trace cycles are
        collected as soon as a call traces a new frame, so dropping the model
        frees its weights and VRAM immediately instead of waiting for a
        user-issued ``gc.collect()`` (#4937, #5020).
        """

        if not self.quantized:
            log.warn("model is not quantized, skip compiling...")
            return self

        if not TORCH_HAS_COMPILE:
            self.compiled = False
            log.warn("To use compile(), you need to have torch version >= 2.6.0, please "
                           "upgrade it by `pip install -U torch torchaudio torchvision`")
            return self

        # needed by eora
        # torch._dynamo.config.capture_scalar_outputs = True

        log.info(f"Compiling qlinear modules with backend: `{backend}`, mode: `{mode}`")
        modules = find_modules(
            self.model,
            layers=[BaseQuantLinear, ExllamaV3TorchLinear],
        )
        # The CPU int4pack kernels only build their runtime layout in an eager
        # forward, so compiling before the first forward traced the dense
        # fallback and kept dequantizing every weight on every token (#5515).
        # Install the layout for every eval-mode CPU module first; a kernel
        # that cannot use the fused layout is left to its own forward.
        model_dtype = getattr(self.model, "dtype", None)
        if not isinstance(model_dtype, torch.dtype):
            # ``nn.Module`` has no ``dtype``; a quantized model still carries
            # the model dtype on its unquantized parameters (norms, embeddings).
            model_dtype = next((param.dtype for param in self.model.parameters()), None)
        if isinstance(model_dtype, torch.dtype):
            for module in modules.values():
                if (
                    getattr(module, "linear_mode", "unsupported") is not None
                    and not callable(
                        getattr(module, "_prepare_cpu_inference_layout", None)
                    )
                ):
                    continue
                try:
                    ensure_cpu_inference_layout(module, model_dtype)
                except Exception:
                    log.debug(
                        "CPU int4pack warm-up skipped for `%s`", type(module).__name__, exc_info=True
                    )
        for name in modules.keys():
            modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        # supress errors until PyTorch fixed: https://github.com/pytorch/pytorch/issues/132635
        # torch._dynamo.config.suppress_errors = True
        log.info(f"Compiling model with backend: `{backend}`, mode: `{mode}`")

        # ``generate()`` resolves to the original module's ``generate``, which
        # calls the original eager ``forward``. Replacing ``self.model`` with a
        # ``torch.compile`` wrapper (an ``OptimizedModule``) therefore compiled
        # nothing that ``generate()`` ran: its ``__getattr__`` hands ``generate``
        # back to the unwrapped module (#4669). Compile the forward in place so
        # both a direct ``model(...)`` call and ``generate()`` enter the compiled
        # path, and keep the compile fallback for a deferred backend failure.
        model = getattr(self.model, "_orig_mod", self.model)
        # Compiling the bound method would store it (and therefore the model)
        # inside the compiled wrapper, and installing that wrapper on the
        # instance closes a model -> forward -> model cycle that keeps the
        # whole model and its VRAM alive until a cyclic GC (#4937). Compile the
        # unbound class function and reach the instance through a weak
        # reference instead. A repeated ``optimize()`` recompiles the same
        # unbound function, so wrappers do not stack.
        unbound_forward = type(model).forward
        compiled_forward = torch_compile(
            unbound_forward, fullgraph=fullgraph, backend=backend, mode=mode
        )
        if compiled_forward is unbound_forward:
            # Compilation unavailable or disabled: drop a proxy from an earlier
            # call so the class method is used again (no instance attribute,
            # no cycle).
            model.__dict__.pop("forward", None)
            self.compiled = False
        else:
            model.forward = _CompiledModelForward(
                compiled_forward, unbound_forward, model
            )
            # Inductor compiles on the first call, so "compiled" means the
            # instance attribute is the compiled proxy, which is what the
            # `compiled` flag has always documented.
            self.compiled = True
        self.model = model

        #trigger kernel compilation hooks
        # if self.compiled:
        #     modules = find_modules(self.model, layers=[BaseQuantLinear])
        #     for name in modules.keys():
        #         modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        # logger.info(f"Compiling qlinear modules with backend: `{backend}`, mode: `{mode}`")
        # modules = find_modules(self.model, layers=[BaseQuantLinear])
        # for name in modules.keys():
        #     modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        return self

    def serve(self,
               host: str = "127.0.0.1",
               port: int = 80,
               async_mode: bool = False,
               api_key: Optional[Union[str, List[str]]] = None,
               allowed_hosts: Optional[List[str]] = None):
        from ..utils.openai_server import OpenAiServer
        self.server = OpenAiServer(model=self, api_key=api_key, allowed_hosts=allowed_hosts)
        self.server.start(host=host, port=port, async_mode=async_mode)

    def serve_shutdown(self):
        if self.server is not None:
            self.server.shutdown()

    def serve_wait_until_ready(self, timeout: int = 30, check_interval: float = 0.1):
        if self.server is not None:
            self.server.wait_until_ready(timeout=timeout, check_interval=check_interval)

    def before_model_load(self, model_local_path: str, load_quantized_model: bool):
        pass

    def after_model_load(self, model, load_quantized_model):
        return model

    @classmethod
    def after_make_quant(cls, model):
        """Reattach runtime modules that the quantized shell did not replace.

        A quantized load replaces only the leaves declared by ``module_tree``.
        Definitions whose runtime model shares those leaves with another branch
        (for example a weight-tied encoder/decoder stack) restore the aliases
        here: the checkpoint load and its unmaterialized-parameter guard run
        before ``after_model_load`` (#5514).
        """

        return model

    def release_device(self) -> bool:
        """Release Tenstorrent devices held by this model's kernels.

        Call this between model loads in a benchmark loop instead of relying on
        garbage collection, which leaves the chips owned (and hot) until the
        interpreter drops the last reference.

        Closing the mesh alone does not make the cards idle: measured on a
        p300c, AICLK stayed at 1350 MHz and ~50 W per board for more than 50 s
        after ``close_device``. The mesh close path therefore also sends the
        firmware ``AICLK_GO_LONG_IDLE`` message, which is what actually returns
        the cards to 800 MHz and ~40 W, so the cooldown that follows is real
        idle time.

        Returns ``True`` when a Tenstorrent device was actually released, and
        ``False`` for every other backend so the call is safe to make
        unconditionally.  The mesh is process-wide and may serve several
        wrappers, so this releases only the leases held by this wrapper's own
        native kernels and closes the shared handle once none remain.
        """

        native_backends = {BACKEND.GPTQ_TTNN, BACKEND.GPTQ_TT_BF8}
        model = getattr(self, "model", None)
        modules = getattr(model, "modules", None)
        if modules is None:
            return False
        owned = [
            module
            for module in modules()
            if native_backends.intersection(
                getattr(module, "SUPPORTS_BACKENDS", ()) or ()
            )
        ]
        if not owned:
            # A CPU (or any non-native) wrapper must not tear down a mesh that
            # another model in this process is still using.
            return False
        released = False
        for module in owned:
            release = getattr(module, "release_device", None)
            if release is not None:
                released = bool(release()) or released
        return released

    def pre_quantize_generate_hook_start(self):
        pass

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            # This hook is now disabled as it's handled by the ModuleLooper after input capture.
            # offload_to_disk(model=self.model, module=self.get_base_modules(model=self.model), disk_path=self.quantize_config.offload_to_disk_path)
            pass

    def prepare_input_capture_layer(
        self,
        layer: nn.Module,
        *,
        module_path: Optional[str],
        device: torch.device,
    ) -> nn.Module:
        """Materialize the first traversed layer before its early-stop hook.

        Auxiliary model definitions may override this when constructing the
        first layer input does not execute the layer body.  The ordinary model
        path retains the historical eager materialization behavior.
        """

        return self.shell_module_materialize(
            target_submodule=layer,
            device=device,
            module_path=module_path,
        )

    def capture_first_layer_positional_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        batch_device: torch.device,
    ) -> List[torch.Tensor]:
        """Normalize first-layer positional inputs so cached forwards can replay decoder layers directly."""

        if kwargs.get("hidden_states") is not None:
            return [move_to(kwargs["hidden_states"], device=batch_device)]
        if args:
            return [move_to(args[0], device=batch_device)]
        return []

    def begin_input_capture_example(
        self,
        example: Dict[str, Any],
        batch_device: torch.device,
    ) -> None:
        """Expose original model inputs while the first-layer hook executes.

        The first-layer hook only sees the mask and rotary embeddings that the
        outer decoder block selected for *that* layer. Rebuilding the kwargs of
        the other declared layer types needs the unexpanded inputs (the 2-D
        padding mask and the position ids), so keep a reference for the duration
        of one example and release it in `end_input_capture_example`.
        """

        del batch_device
        self._layer_type_replay_source = example

    def end_input_capture_example(self) -> None:
        """Release any model-specific original-input capture state."""

        self._layer_type_replay_source = None
        self._layer_type_replay_rotary = None

    def capture_first_layer_input_kwargs(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        batch_device: torch.device,
        layer_input_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Allow model definitions to persist extra first-layer replay metadata during calibration capture."""

        return layer_input_kwargs

    @staticmethod
    def _layer_type_param_name(module: nn.Module) -> Optional[str]:
        """Name of the rotary-embedding argument that selects a layer type."""

        try:
            parameters = inspect.signature(module.forward).parameters
        except (TypeError, ValueError):
            return None
        for name in ("layer_type", "attention_type"):
            if name in parameters:
                return name
        return None

    def _resolve_layer_type_replay_rotary(self) -> Optional[nn.Module]:
        """Rotary module that builds a different `position_embeddings` per layer type."""

        state = object.__getattribute__(self, "__dict__")
        cached = state.get("_layer_type_replay_rotary")
        if cached is not None:
            return cached

        model = self._replay_root_model()
        if model is None:
            return None

        declared = getattr(self, "rotary_embedding", None)
        if declared:
            module, _ = get_module_by_name_prefix(model, [declared])
            if module is not None and self._layer_type_param_name(module) is not None:
                self._layer_type_replay_rotary = module
                return module

        for name, module in model.named_modules():
            if "rotary" not in name.lower() and "rotary" not in type(module).__name__.lower():
                continue
            if self._layer_type_param_name(module) is not None:
                self._layer_type_replay_rotary = module
                return module
        return None

    def _replay_root_model(self) -> Optional[nn.Module]:
        """The loaded model, or None for a definition instance built without `nn.Module.__init__`.

        `BaseQModel.__getattr__` resolves missing attributes through `self.model`,
        so probing attributes on a bare instance recurses. Read the module table
        directly instead.
        """

        modules = object.__getattribute__(self, "__dict__").get("_modules") or {}
        model = modules.get("model")
        return model if isinstance(model, nn.Module) else None

    def _layer_type_replay_config(self):
        """Config that declares `layer_types` (the text tower for multimodal wrappers)."""

        model = self._replay_root_model()
        if model is None:
            return None
        config = getattr(model, "config", None)
        if config is None:
            return None
        text_config = getattr(config, "text_config", config)
        return text_config if text_config is not None else config

    def _declared_layer_types(self) -> Optional[tuple]:
        """Distinct-typed `config.layer_types`, or None when one type covers the stack."""

        text_config = self._layer_type_replay_config()
        layer_types = getattr(text_config, "layer_types", None)
        if not isinstance(layer_types, (list, tuple)) or not layer_types:
            return None
        if len(set(layer_types)) < 2:
            return None
        return tuple(layer_types)

    @staticmethod
    def _layer_type_or_mask(layer_type: str, text_config):
        """Mask overlay `Gemma3TextModel.forward` applies for bidirectional attention."""

        if layer_type != _LAYER_TYPE_SLIDING:
            return lambda *args, **kwargs: torch.tensor(True, dtype=torch.bool)
        try:
            from transformers.masking_utils import sliding_window_bidirectional_overlay
        except ImportError:  # pragma: no cover - older transformers
            return None
        window = getattr(text_config, "sliding_window", None)
        if window is None:
            return None
        return sliding_window_bidirectional_overlay(window)

    def _build_layer_type_attention_mask(
        self,
        text_config,
        layer_type: str,
        hidden_states: torch.Tensor,
        raw_attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ):
        """Rebuild the mask `transformers` would hand to a layer of `layer_type`."""

        from transformers.masking_utils import LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING

        mask_kwargs = {
            "config": text_config,
            "inputs_embeds": hidden_states,
            "attention_mask": raw_attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        if getattr(text_config, "use_bidirectional_attention", False):
            or_mask_function = self._layer_type_or_mask(layer_type, text_config)
            if or_mask_function is None:
                return _LAYER_TYPE_MASK_UNSUPPORTED
            mask_kwargs["or_mask_function"] = or_mask_function
        # Use the same pattern-to-builder table `create_masks_for_generate` uses.
        # A bare `create_causal_mask` fallback silently gives chunked, recurrent,
        # and other non-sliding types the wrong attention pattern.
        builder = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING.get(layer_type)
        if builder is None or isinstance(builder, dict):
            return _LAYER_TYPE_MASK_UNSUPPORTED
        return builder(**mask_kwargs)

    def _build_layer_type_position_embeddings(
        self,
        rotary: nn.Module,
        layer_type: str,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        target_device: torch.device,
    ):
        """Rebuild the per-layer-type rotary tuple with the model's own rotary module."""

        rotary_device = get_device(rotary)
        probe = torch.empty(1, device=rotary_device, dtype=hidden_states.dtype)
        result = rotary(probe, move_to(position_ids, device=rotary_device), layer_type)
        if isinstance(result, (tuple, list)):
            return type(result)(move_to(value, device=target_device) for value in result)
        return move_to(result, device=target_device)

    def capture_layer_type_replay_kwargs(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        batch_device: torch.device,
        layer_input_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Persist the replay kwargs of every declared layer type, not only layer 0's.

        Called by the capture stage right after `capture_first_layer_input_kwargs`;
        a no-op (the key is simply absent) when the model declares one layer type.
        """

        layer_type_kwargs = self._build_layer_type_replay_kwargs(
            args=args,
            kwargs=kwargs,
            batch_device=batch_device,
        )
        if layer_type_kwargs is not None:
            layer_input_kwargs[_LAYER_TYPE_REPLAY_KWARGS_KEY] = layer_type_kwargs
        return layer_input_kwargs

    def _build_layer_type_replay_kwargs(
        self,
        *,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        batch_device: torch.device,
    ) -> Optional[Dict[str, Any]]:
        """Rebuild the mask and rotary tuple of every declared layer type.

        `transformers` 5 builds one mask per layer *type*
        (`causal_mask_mapping[config.layer_types[i]]`) and Gemma3/OLMo3 also pass a
        different `position_embeddings` per type. The capture hook fires on the
        first decoder layer only, so a single captured mask would be replayed to
        layers whose type expects a different one. Rebuild the missing types from
        the same inputs the model used, and let replay select by layer index.
        """

        layer_types = self._declared_layer_types()
        if layer_types is None:
            return None
        source = object.__getattribute__(self, "__dict__").get("_layer_type_replay_source")
        if not isinstance(source, dict):
            return None

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if not torch.is_tensor(hidden_states) or hidden_states.dim() != 3:
            return None

        text_config = self._layer_type_replay_config()
        num_layers = getattr(text_config, "num_hidden_layers", None)
        if isinstance(num_layers, int) and num_layers != len(layer_types):
            # A per-layer list is the only shape this reconstruction can index.
            return None

        raw_attention_mask = source.get("attention_mask")
        if raw_attention_mask is not None:
            if not torch.is_tensor(raw_attention_mask) or raw_attention_mask.dim() != 2:
                return None
            if raw_attention_mask.shape[-1] != hidden_states.shape[1]:
                return None
            raw_attention_mask = move_to(raw_attention_mask, device=hidden_states.device)

        position_ids = kwargs.get("position_ids")
        if not torch.is_tensor(position_ids):
            position_ids = torch.arange(
                hidden_states.shape[1], device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0)
        position_ids = move_to(position_ids, device=hidden_states.device)
        if position_ids.shape[-1] != hidden_states.shape[1]:
            return None

        base_type = layer_types[0]
        captured_position_embeddings = kwargs.get("position_embeddings")
        rotary = (
            self._resolve_layer_type_replay_rotary()
            if captured_position_embeddings is not None
            else None
        )

        by_type: Dict[str, Dict[str, Any]] = {}
        for layer_type in sorted(set(layer_types)):
            entry: Dict[str, Any] = {}
            layer_type_mask = self._build_layer_type_attention_mask(
                text_config=text_config,
                layer_type=layer_type,
                hidden_states=hidden_states,
                raw_attention_mask=raw_attention_mask,
                position_ids=position_ids,
            )
            if layer_type_mask is not _LAYER_TYPE_MASK_UNSUPPORTED:
                entry["attention_mask"] = layer_type_mask
            # The first declared type already owns the captured rotary tuple, but
            # its mask is still needed: `support_batch_quantize=False` replay
            # forces `attention_mask=None`, so only a rebuilt base-type mask can
            # restore the sliding/chunked window HF applies to those layers.
            if rotary is not None and layer_type != base_type:
                entry["position_embeddings"] = self._build_layer_type_position_embeddings(
                    rotary=rotary,
                    layer_type=layer_type,
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    target_device=batch_device,
                )
            if entry:
                by_type[layer_type] = entry

        return {
            "base_type": base_type,
            "layer_types": layer_types,
            "by_type": by_type,
        }

    @staticmethod
    def _resolve_layer_replay_type(layer: nn.Module, layer_types: Sequence[str]) -> Optional[str]:
        """Layer type declared by the decoder layer being replayed."""

        def _declared(value: Any) -> Optional[str]:
            # Only a value the model actually declared is usable: a stray
            # `layer_type` attribute outside the declared list cannot be
            # looked up in `by_type`.
            return value if isinstance(value, str) and value in layer_types else None

        attention = getattr(layer, "self_attn", None)
        declared = _declared(getattr(attention, "layer_type", None))
        if declared is None:
            declared = _declared(getattr(layer, "layer_type", None))
        if declared is None:
            # Hybrid definitions such as GraniteMoE-Hybrid name the slot
            # `block_type`; a recurrent mixer is a sibling (`mamba`, `conv`)
            # rather than `self_attn` and carries its own `layer_type`
            # (issue #4671).
            declared = _declared(getattr(layer, "block_type", None))
        if declared is None:
            for child in layer.children():
                declared = _declared(getattr(child, "layer_type", None))
                if declared is not None:
                    break
        if declared is not None:
            return declared

        index = getattr(attention, "layer_idx", None)
        if not isinstance(index, int):
            index = getattr(layer, "layer_idx", None)
        if not isinstance(index, int):
            # A mixer sibling is also a reliable per-layer index source when
            # the decoder layer itself does not store one.
            for child in layer.children():
                candidate = getattr(child, "layer_idx", None)
                if isinstance(candidate, int):
                    index = candidate
                    break
        if isinstance(index, int) and 0 <= index < len(layer_types):
            return layer_types[index]
        return None

    def quantization_layer_output_required(
        self,
        *,
        layer_index: int,
        layer_name: str,
        layer_count: int,
    ) -> bool:
        """Return whether a model-specific consumer needs this replay output.

        The default layer loop does not replay the final transformer layer
        because no ordinary successor consumes it. Auxiliary graphs can opt
        into a bounded callback without pretending to be additional decoder
        layers.
        """

        del layer_index, layer_name, layer_count
        return False

    def receive_quantization_layer_outputs(
        self,
        *,
        layer_index: int,
        layer_name: str,
        layer_outputs: List[List[torch.Tensor]],
        layer_input_kwargs: List[Dict[str, Any]],
        position_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor | None],
    ) -> None:
        """Receive an opted-in post-quantization replay boundary."""

        del (
            layer_index,
            layer_name,
            layer_outputs,
            layer_input_kwargs,
            position_ids,
            attention_masks,
        )

    def prepare_layer_replay_kwargs(
        self,
        layer: nn.Module,
        layer_input: List[torch.Tensor],
        additional_inputs: Dict[str, Any],
        target_device: torch.device,
    ) -> Dict[str, Any]:
        """Allow model definitions to refresh layer-specific kwargs before cached layer replay."""

        layer_type_kwargs = additional_inputs.pop(_LAYER_TYPE_REPLAY_KWARGS_KEY, None)
        if layer_type_kwargs:
            self._apply_layer_type_replay_kwargs(
                layer=layer,
                layer_type_kwargs=layer_type_kwargs,
                additional_inputs=additional_inputs,
                target_device=target_device,
            )
        return additional_inputs

    def awq_module_replay_inputs(
        self,
        module: nn.Module,
        x: torch.Tensor,
        module_kwargs: Dict[str, Any],
    ):
        """Allow a definition to supply scaffolding an AWQ block replay needs.

        AWQ replays the common parent of a multi-module scaling group to score
        the combined block. Some attention modules require arguments that the
        per-linear activation capture never records (TeleChat2's
        ``TelechatAttention.forward`` needs ``residual`` and ``attention_mask``),
        so the definition may add them here. Returning ``x`` unchanged and the
        same kwargs is the default for every other family.
        """

        return x, module_kwargs

    def awq_module_replay_chunk_kwargs(
        self,
        module: nn.Module,
        module_kwargs: Dict[str, Any],
        *,
        start: int,
        length: int,
        full_batch_size: int,
    ) -> Dict[str, Any]:
        """Override chunk kwargs whose batch layout is not a leading batch axis.

        ``module_kwargs`` is the full, prepared replay context, before generic
        batch slicing. Return only keys needing model-specific slicing.
        """

        return {}

    def _apply_layer_type_replay_kwargs(
        self,
        *,
        layer: nn.Module,
        layer_type_kwargs: Dict[str, Any],
        additional_inputs: Dict[str, Any],
        target_device: torch.device,
    ) -> None:
        """Swap in the mask/rotary captured for the layer type being replayed."""

        layer_types = layer_type_kwargs.get("layer_types") or ()
        by_type = layer_type_kwargs.get("by_type") or {}
        if not by_type:
            return
        layer_type = self._resolve_layer_replay_type(layer, layer_types)
        if layer_type is None:
            return
        entry = by_type.get(layer_type)
        if not entry:
            return
        is_base_type = layer_type == layer_type_kwargs.get("base_type")

        # A `None` layer-0 mask can mean "causal skip" rather than "masks are
        # disabled", so a later layer type that needs a materialized mask (for
        # example chunked attention after a full-attention layer 0) must still
        # receive it. Only skip the write when both sides are `None`.
        if "attention_mask" in entry:
            layer_type_mask = entry["attention_mask"]
            if is_base_type:
                # The base type's captured mask is discarded by
                # `support_batch_quantize=False`; apply the rebuilt mask whenever
                # one exists instead of leaving the layer in full causal mode.
                if layer_type_mask is not None:
                    additional_inputs["attention_mask"] = move_to(
                        layer_type_mask, device=target_device
                    )
            elif layer_type_mask is not None or additional_inputs.get("attention_mask") is not None:
                additional_inputs["attention_mask"] = (
                    None
                    if layer_type_mask is None
                    else move_to(layer_type_mask, device=target_device)
                )
        position_embeddings = entry.get("position_embeddings")
        if (
            not is_base_type
            and position_embeddings is not None
            and additional_inputs.get("position_embeddings") is not None
        ):
            additional_inputs["position_embeddings"] = self._move_position_embeddings(
                position_embeddings, target_device
            )

    @staticmethod
    def _move_position_embeddings(position_embeddings, target_device: torch.device):
        if isinstance(position_embeddings, (tuple, list)):
            return type(position_embeddings)(
                move_to(value, device=target_device) for value in position_embeddings
            )
        return move_to(position_embeddings, device=target_device)

    def prepare_layer_replay_keep_mask(
        self,
        layer: nn.Module,
        layer_input: List[torch.Tensor],
        additional_inputs: Dict[str, Any],
        keep_mask: torch.Tensor | None,
        target_device: torch.device,
    ) -> torch.Tensor | None:
        """Allow transformed replay layers to align processor hook masks."""

        del layer, layer_input, additional_inputs, target_device
        return keep_mask

    def update_layer_replay_kwargs_from_output(
        self,
        layer: nn.Module,
        layer_output: Any,
        layer_input_kwargs: Dict[str, Any],
        target_device: torch.device,
    ) -> Dict[str, Any]:
        """Allow model definitions to persist cross-layer replay state from one layer output."""

        return layer_input_kwargs

    def lm_head_pre_quantize_generate_hook(self, inputs: List[List[torch.tensor]]) -> List[List[torch.tensor]]:
        resolved = self._resolved_pre_lm_head_norm_candidates()
        if resolved:
            norm, _ = resolved[0]
            norm = self.pre_quantize(norm)

            for element in inputs:
                for i in range(len(element)):
                    element[i] = norm(element[i])

            self.post_quantize(norm)
        return inputs

    def _pre_quantize_device(self) -> torch.device:
        """Concrete device on which `pre_quantize` stages one layer.

        ``quantize_config.device`` carries only the accelerator family (the
        loader strips ordinals through ``normalize_device``), so resolving it
        directly yields the index-less family device, i.e. the *current*
        device (cuda:0). A caller who restricted the run through
        ``compute_device_filter`` (``device=("cuda:1",)``) then gets every
        layer staged on an excluded GPU, which OOMs when that GPU is full
        (issue #4233). Follow the same filtered pool the loopers use.
        """

        base_device = normalize_device_like(getattr(self.quantize_config, "device", None))
        if base_device is None:
            return CPU
        devices = select_forward_devices(base_device)
        compute_device_filter = getattr(self.quantize_config, "compute_device_filter", None)
        if compute_device_filter is not None:
            filtered = compute_device_filter(devices)
            # The loopers keep the full pool when the filter rejects everything.
            if filtered:
                devices = list(filtered)
        return as_torch_device(devices[0] if devices else base_device)

    def _pre_quantize_staging_device(self) -> torch.device:
        """Device `pre_quantize` stages one layer on.

        A CPU-backed encoder (GGUF, BitsAndBytes, NF12) reads its host source
        directly -- NF12 stages an eligible host tensor itself -- so moving the
        dense layer to the quantization device only to send it back is a wasted
        round trip of the whole tensor, and a lazy shell would hold VRAM for a
        layer that never encodes there (#5757). The requested device still
        selects the inference backend and owns the packed checkpoint handoff.
        """

        quantize_config = getattr(self, "quantize_config", None)
        if quantize_config is not None and encoder_stages_on_host(quantize_config):
            return CPU
        return self._pre_quantize_device()

    def pre_quantize(
        self,
        module: nn.Module,
        module_path: Optional[str] = None,
    ) -> nn.Module:
        if get_device(module) == META or _module_has_meta_tensors(module):
            device = self._pre_quantize_staging_device()
            if self._materialize_floatx_layer_shell(
                module, device, module_path=module_path
            ):
                return module
            return self.shell_module_materialize(
                target_submodule=module,
                device=device,
                module_path=module_path,
            )
        elif get_device(module) == CPU and self.quantize_config.device != CPU:
            return move_to(module, device=self._pre_quantize_staging_device())
        else:
            return module

    def _materialize_floatx_layer_shell(
        self,
        module: nn.Module,
        device: torch.device,
        module_path: Optional[str] = None,
    ) -> bool:
        """Materialize one lazy layer without copying packed floatx storage into dense shells.

        Layer staging happens before its quantizable children have ``NamedModule``
        plans.  A regular LazyTurtle materialization therefore attempts to copy
        an NVFP4 packed tensor into the dense ``nn.Linear.weight`` parameter.
        Decode only those leaf shells long enough to construct the layer; the
        later module-local forward plan still reloads the original checkpoint
        tensors and selects native FP8/NVFP4 execution for Hessian collection.
        """

        decoder_cfg = self._active_auto_module_decoder_config()
        if decoder_cfg is None or not isinstance(self.turtle_model, LazyTurtle):
            return False

        device = as_torch_device(device)

        decoded_any = False
        # Resolve the layer path once, then derive every child path from its
        # ``named_modules()`` walk.  Looking each child up in the whole model
        # made floatx shell materialization grow with layer depth.
        layer_path = _resolve_qualified_name(self.model, module, module_path)
        module_paths: Dict[int, str] = {}
        for relative_name, candidate in module.named_modules():
            full_name = (
                f"{layer_path}.{relative_name}" if relative_name else layer_path
            )
            module_paths.setdefault(id(candidate), full_name)
        # Traverse leaves first, because replacement invalidates the old
        # submodule's qualified path in the live model tree.
        for submodule in reversed(list(module.modules())):
            if submodule is module or not hasattr(submodule, "weight"):
                continue
            module_name = module_paths.get(id(submodule))
            if module_name is None:
                module_name = _get_qualified_name(self.model, submodule)
            checkpoint_tensors = self.turtle_model.checkpoint_tensors_for_submodule(
                target_model=self.model,
                target_submodule=submodule,
                recurse=False,
                module_path=module_name,
            )
            weight = checkpoint_tensors.get("weight")
            if not isinstance(weight, torch.Tensor):
                continue
            if self._decoder_weight_format(
                weight=weight,
                checkpoint_tensors=checkpoint_tensors,
                module_name=module_name,
            ) is None:
                continue
            decoded = self._build_decoder_quant_source_module(
                submodule,
                checkpoint_tensors=checkpoint_tensors,
                target_dtype=decoder_cfg.target_dtype,
                module_name=module_name,
            )
            self._replace_live_submodule(
                submodule, decoded, module_path=module_name
            )
            module_paths[id(decoded)] = module_name
            decoded_any = True

        if not decoded_any:
            return False

        # Materialize normal direct tensors (norm weights, etc.) after every
        # packed leaf has been replaced by a shape-compatible dense shell.
        for submodule in module.modules():
            self.shell_direct_meta_materialize(
                target_submodule=submodule,
                device=device,
                module_path=module_paths.get(id(submodule)),
            )
        move_to(module, device=device)
        return True

    def forward_device_for_module(self, module: nn.Module, planned_device: torch.device) -> torch.device:
        """Apply model-declared placement exclusions to subset replay planning."""

        turtle_model = self.turtle_model
        if not isinstance(turtle_model, LazyTurtle):
            return planned_device

        # LazyTurtle matches exclusions by dotted parameter path, not module type.
        module_paths = getattr(self, "_forward_module_paths_by_id", None)
        if module_paths is None or id(module) not in module_paths:
            module_paths = {id(candidate): name for name, candidate in self.model.named_modules() if name}
            self._forward_module_paths_by_id = module_paths
        module_path = module_paths.get(id(module))
        if module_path is None:
            return planned_device
        # Check only tensors owned by this leaf; descendants receive their own plan entry.
        for rel_name, _ in module.named_parameters(recurse=False):
            if turtle_model.is_no_placement_tensor(module_path, rel_name):
                return torch.device(CPU)
        return planned_device

    def has_forward_device_overrides(self) -> bool:
        """Return whether replay must preserve model-declared tensor placement."""

        turtle_model = self.turtle_model
        return isinstance(turtle_model, LazyTurtle) and bool(
            getattr(turtle_model, "_no_placement_params", ())
        )

    def post_quantize(self, module: nn.Module) -> nn.Module:
        #return self.offload_to_disk(module=module)
        if self.quantize_config.activation_mode in {"w4afp8", "w4a_nvfp4"}:
            # GB10 can stall in the pageable CUDA device-to-host copy used by
            # Module.to('cpu') during post-quant layer offload. Keep its staged
            # copies small; the default path uses a larger pinned chunk.
            return move_module_to_cpu_pinned(module, chunk_bytes=1024 * 1024)
        return move_module_to_cpu_pinned(module)

    def after_quantize(self) -> None:
        """Run model-specific finalization after the complete quantization loop."""

        return None

    def _replace_live_submodule(
        self,
        current_submodule: nn.Module,
        replacement: nn.Module,
        module_path: Optional[str] = None,
    ) -> nn.Module:
        """Replace one live model submodule in place and return the replacement.

        ``module_path`` lets callers that already hold the dotted name skip the
        full-tree scan in ``_get_qualified_name``.
        """

        module_path = _resolve_qualified_name(
            self.model, current_submodule, module_path
        )
        parent, leaf = _get_parent_and_leaf_by_path(self.model, module_path)
        setattr(parent, leaf, replacement)
        return replacement

    def _maybe_auto_dtype_cast_forward_module(
        self,
        module: nn.Module,
        *,
        device: torch.device,
        named_module: "NamedModule",
    ) -> nn.Module:
        """Keep a calibration source packed until its quantizer needs it."""

        qcfg = getattr(self, "quantize_config", None)
        if qcfg is None or qcfg.auto_dtype_cast != AutoDtypeCast.AUTO_NF12:
            return module
        if qcfg.method == METHOD.NF12:
            return module
        native_plan = getattr(qcfg, "_native_floatx_forward_plan", None)
        if isinstance(native_plan, dict) and native_plan.get("mode") == "native":
            return module
        # EoRA captures statistics after quantization; its forward must consume
        # the quantized layer instead of a freshly packed dense source.
        if named_module is not None and named_module.state.get("auto_dtype_cast_forward_disabled"):
            return module
        # A post-quantization forward (EoRA adapter capture) must consume the
        # quantized layer that already lives in the model tree. Re-packing the
        # restored dense source here would replace that layer for the forward.
        full_name = getattr(named_module, "full_name", None)
        # The lm_head is its own quantization stage rather than a decoder
        # projection. It is quantized from a dense CPU source that must stay
        # the module in the tree, so packing it as an NF12 forward wrapper both
        # saves nothing (the dense source is required anyway) and detaches the
        # head from the tree that `post_quantize` moves back to CPU.
        if full_name is not None and full_name == getattr(self, "lm_head", None):
            return module
        if full_name:
            try:
                live = self.model.get_submodule(full_name)
            except AttributeError:
                live = None
            if getattr(live, "QUANT_TYPE", None) is not None:
                return module
        from ..utils.auto_dtype_cast import cast_source_module, eligible_source_module

        spec = eligible_source_module(module, device)
        if spec is None:
            return module
        replacement = cast_source_module(module, spec)
        named_module.state["auto_dtype_cast_source_spec"] = spec
        named_module.state["auto_dtype_cast_was_applied"] = True
        replacement = self._replace_live_submodule(
            module, replacement, module_path=named_module.full_name
        )
        source_bias = getattr(module, "bias", None)
        source_bytes = module.weight.numel() * module.weight.element_size()
        if isinstance(source_bias, torch.Tensor):
            source_bytes += source_bias.numel() * source_bias.element_size()
        event = {
            "module": named_module.full_name,
            "device": str(device),
            "kind": spec.kind,
            "source_dtype": str(spec.dtype).split(".")[-1],
            "source_bytes": source_bytes,
            "packed_bytes": sum(t.numel() * t.element_size() for t in replacement.buffers()),
        }
        events = getattr(self, "auto_dtype_cast_events", None)
        if events is not None:
            events.append(event)
        emit_device_telemetry(
            "auto_dtype_cast_nf12",
            **event,
        )
        return replacement

    def _build_decoder_quant_source_module(
        self,
        target_submodule: nn.Module,
        *,
        checkpoint_tensors: Optional[Dict[str, torch.Tensor]] = None,
        target_dtype: torch.dtype,
        module_name: Optional[str] = None,
    ) -> nn.Module:
        """Build a dense CPU source module from checkpoint tensors for quantization."""

        missing = [
            name for name, parameter in target_submodule.named_parameters()
            if parameter.is_meta
            and not isinstance((checkpoint_tensors or {}).get(name), torch.Tensor)
        ]
        if missing:
            raise RuntimeError(
                f"Cannot reload quantization source: missing checkpoint tensors {missing}"
            )
        quant_source = copy.deepcopy(target_submodule)
        if _module_has_meta_tensors(quant_source):
            quant_source = quant_source.to_empty(device=CPU)
        else:
            quant_source = quant_source.to(device=CPU)
        weight = None if checkpoint_tensors is None else checkpoint_tensors.get("weight")
        if isinstance(weight, torch.Tensor) and hasattr(quant_source, "weight"):
            result_shape = tuple(getattr(quant_source.weight, "shape", weight.shape))
            direct_scale = checkpoint_tensors.get("weight_scale")
            if not isinstance(direct_scale, torch.Tensor):
                direct_scale = checkpoint_tensors.get("scale")
            is_mx_block_scale = isinstance(direct_scale, torch.Tensor) and direct_scale.dtype in {
                torch.uint8, getattr(torch, "float8_e8m0fnu", None)
            }
            decoder_kind = self._decoder_weight_format(
                weight=weight,
                checkpoint_tensors=checkpoint_tensors,
                module_name=module_name,
                result_shape=result_shape,
            )
            if decoder_kind == "mimo_mixed":
                from ..utils.mimo import decode_mimo_weight

                prefix = module_name + "."
                decoded_weight = decode_mimo_weight(
                    self._decoder_source_config(),
                    prefix + "weight",
                    weight,
                    lambda key: checkpoint_tensors.get(key.removeprefix(prefix)),
                    target_dtype=target_dtype,
                )
                if decoded_weight.shape != quant_source.weight.shape:
                    raise ValueError(f"MiMo decoded shape does not match module {module_name}")
            else:
                result_shape = tuple(getattr(quant_source.weight, "shape", weight.shape))
                if decoder_kind == "fp4":
                    scale = self._decoder_fp4_effective_scale(
                        checkpoint_tensors=checkpoint_tensors,
                        result_shape=result_shape,
                    )
                elif decoder_kind == "fp8" and is_mx_block_scale:
                    scale = self._decoder_mx_block_scale(
                        scale_tensor=direct_scale,
                        result_shape=result_shape,
                        format_name="FP8",
                    )
                else:
                    scale = self._decoder_scale_tensor(
                        scale_tensor=direct_scale,
                        result_shape=result_shape,
                    )
                if decoder_kind == "fp8" and isinstance(scale, torch.Tensor):
                    scale_2 = checkpoint_tensors.get("weight_scale_2")
                    if isinstance(scale_2, torch.Tensor):
                        if is_mx_block_scale:
                            scale_2 = self._decoder_mx_secondary_scale(scale_2, result_shape)
                        scale = scale.to(torch.float32) * scale_2.to(torch.float32)
                scale_inv = None
                if not isinstance(scale, torch.Tensor):
                    scale_inv = self._decoder_scale_tensor(
                        scale_tensor=checkpoint_tensors.get("weight_scale_inv"),
                        result_shape=result_shape,
                    )
                if decoder_kind == "fp8":
                    decoded_weight = dequantize_fp8(
                        weight,
                        scale=scale if isinstance(scale, torch.Tensor) else None,
                        scale_inv=scale_inv if isinstance(scale_inv, torch.Tensor) else None,
                        axis=None,
                        target_dtype=torch.float32,
                        # Resolve the checkpoint convention once so this decoder
                        # and ``_build_fp8_forward_module`` agree. A direct
                        # ``weight_scale`` is already normalized to a multiplier.
                        scale_semantics=(
                            self._decoder_fp8_scale_semantics()
                            if isinstance(scale_inv, torch.Tensor)
                            else "multiplier"
                        ),
                    )
                elif decoder_kind == "fp4":
                    decoded_weight = dequantize_f4_e2m1(
                        weight.view(torch.uint8) if weight.dtype is torch.int8 else weight,
                        scale=scale if isinstance(scale, torch.Tensor) else None,
                        scale_inv=scale_inv if isinstance(scale_inv, torch.Tensor) else None,
                        axis=None,
                        target_dtype=torch.float32,
                    )
                else:
                    decoded_weight = weight.to(dtype=target_dtype)

            existing_weight = getattr(quant_source, "weight")
            quant_source.weight = nn.Parameter(
                decoded_weight.to(device=CPU, dtype=target_dtype),
                requires_grad=getattr(existing_weight, "requires_grad", False),
            )

        bias = None if checkpoint_tensors is None else checkpoint_tensors.get("bias")
        if isinstance(bias, torch.Tensor) and getattr(quant_source, "bias", None) is not None:
            existing_bias = quant_source.bias
            quant_source.bias = nn.Parameter(
                bias.to(device=CPU, dtype=target_dtype),
                requires_grad=getattr(existing_bias, "requires_grad", False),
            )

        quant_source = quant_source.to(dtype=target_dtype)
        quant_source.eval()
        setattr(quant_source, "target_device", torch.device(CPU))
        return quant_source

    def _decoder_block_size(self) -> Optional[tuple[int, int]]:
        """Read the checkpoint's floatx block size metadata when present."""

        quant_config = getattr(getattr(self.model, "config", None), "quantization_config", None)
        if isinstance(quant_config, dict):
            block_size = quant_config.get("weight_block_size")
        else:
            block_size = getattr(quant_config, "weight_block_size", None)
        if isinstance(block_size, (list, tuple)) and len(block_size) == 2:
            return int(block_size[0]), int(block_size[1])
        return None

    def _decoder_fp8_scale_semantics(self) -> str:
        """Resolve how a checkpoint's ``weight_scale_inv`` is applied, once.

        HF/DeepSeek/Qwen FP8 checkpoints store ``amax / 448`` under the
        historical ``weight_scale_inv`` name and *multiply* it. A checkpoint
        that stores a true inverse scale declares
        ``quantization_config.weight_scale_semantics = "inverse"``.

        Both the native forward wrapper and the dense quant-source decoder read
        this single resolved value so they can never disagree about the same
        checkpoint tensor.
        """

        cached = self.__dict__.get("_resolved_fp8_scale_semantics")
        if cached is not None:
            return cached

        config = self._decoder_source_config()
        quant_config = getattr(config, "quantization_config", None)
        if isinstance(quant_config, dict):
            declared = quant_config.get("weight_scale_semantics")
        else:
            declared = getattr(quant_config, "weight_scale_semantics", None)

        semantics = "multiplier" if declared is None else str(declared).strip().lower()
        if semantics not in {"multiplier", "inverse"}:
            raise ValueError(
                "FP8 `weight_scale_semantics` must be 'multiplier' or 'inverse', "
                f"got {declared!r}."
            )
        self.__dict__["_resolved_fp8_scale_semantics"] = semantics
        return semantics

    def _decoder_declared_floatx_format(self) -> str:
        """Read flat and compressed-tensors group format labels."""

        quant_config = getattr(getattr(self.model, "config", None), "quantization_config", None)
        if not isinstance(quant_config, dict):
            return str(getattr(quant_config, "format", "") or getattr(quant_config, "quant_algo", "")).lower()
        labels = [str(quant_config.get(key) or "").lower() for key in ("format", "quant_algo", "variant")]
        groups = quant_config.get("config_groups")
        if isinstance(groups, dict):
            labels.extend(
                str(group.get("format") or "").lower()
                for group in groups.values()
                if isinstance(group, dict)
            )
        labels.extend(quark_floatx_formats(quant_config))
        return " ".join(labels)

    def _decoder_quant_method_name(self) -> str:
        """Return the checkpoint quantizer family declared in model config."""

        quant_config = getattr(getattr(self.model, "config", None), "quantization_config", None)
        if isinstance(quant_config, dict):
            value = quant_config.get("quant_method")
        else:
            value = getattr(quant_config, "quant_method", None)
        return str(value or "").strip().lower()

    def _uses_modelopt_runtime(self) -> bool:
        """Return ``True`` when the checkpoint declares ModelOpt runtime semantics."""

        return self._decoder_quant_method_name() == "modelopt"

    def _modelopt_unsupported_metadata(self) -> Optional[UnsupportedMetadata]:
        """Describe unsupported quantization metadata in a ModelOpt checkpoint.

        The config payload is classified by the shared helper so a KV-cache
        scheme is reported as a KV-cache scheme, not as activation
        quantization. Checkpoints that pre-quantize activations also ship the
        corresponding scales, which the weight map exposes as side tensors.
        """

        quant_config = getattr(getattr(self.model, "config", None), "quantization_config", None)
        found = find_unsupported_metadata(quant_config)
        if found is not None:
            return found

        if isinstance(self.turtle_model, LazyTurtle):
            keys = self.turtle_model._weight_map.keys()
        else:
            keys = self.model.state_dict().keys()
        if any(str(name).endswith((".input_scale", ".k_scale", ".v_scale")) for name in keys):
            return UnsupportedMetadata(field="checkpoint_scales", kind="checkpoint_scales")
        return None

    def _configure_modelopt_runtime(self) -> None:
        """Reject unsupported ModelOpt activation quantization at load time."""

        if not self._uses_modelopt_runtime():
            return

        # A ModelOpt FP8/NVFP4 checkpoint can be a *source* for a fresh
        # weight-only quantization.  In that path the loader has already
        # installed the decoder and LazyTurtle supplies dense BF16 weights one
        # module at a time, so ModelOpt's original activation metadata is never
        # used for runtime execution.
        if self._active_auto_module_decoder_config() is not None:
            return

        unsupported = self._modelopt_unsupported_metadata()
        if unsupported is not None:
            message = describe_unsupported_metadata(unsupported)
            log.error(message)
            raise ValueError(message)

    def _decoder_scale_tensor(
        self,
        *,
        scale_tensor: Optional[torch.Tensor],
        result_shape: tuple[int, ...],
    ) -> Optional[torch.Tensor]:
        """Expand padded floatx block grids to the dense weight shape when needed."""

        if not isinstance(scale_tensor, torch.Tensor):
            return None
        if scale_tensor.ndim != 2 or len(result_shape) != 2:
            return scale_tensor

        rows, cols = result_shape
        blocks_r, blocks_c = scale_tensor.shape
        if rows % blocks_r == 0 and cols % blocks_c == 0:
            return scale_tensor

        block_size = self._decoder_block_size()
        if block_size is None:
            return scale_tensor

        block_rows, block_cols = block_size
        if blocks_r * block_rows < rows or blocks_c * block_cols < cols:
            return scale_tensor

        expanded = scale_tensor.repeat_interleave(block_rows, dim=0)
        expanded = expanded.repeat_interleave(block_cols, dim=1)
        return expanded[:rows, :cols].contiguous()

    @staticmethod
    def _decoder_mx_block_scale(
        *,
        scale_tensor: torch.Tensor,
        result_shape: tuple[int, ...],
        format_name: str,
    ) -> torch.Tensor:
        """Clip padded E8M0 block grids before decoding or expanding their values."""

        if len(result_shape) != 2 or result_shape[1] % 32 or scale_tensor.ndim != 2:
            raise ValueError(f"Unsupported {format_name} E8M0 scale layout for weight {result_shape}")
        rows, cols = result_shape
        blocks = cols // 32
        if scale_tensor.shape[0] < rows or scale_tensor.shape[1] < blocks:
            raise ValueError(
                f"{format_name} E8M0 scale shape {tuple(scale_tensor.shape)} is smaller than {(rows, blocks)}"
            )
        return decode_e8m0_scale(scale_tensor[:rows, :blocks])

    @staticmethod
    def _decoder_mx_secondary_scale(scale_tensor: torch.Tensor, result_shape: tuple[int, ...]) -> torch.Tensor:
        """Trim padded block factors while preserving scalar or row broadcasting."""

        if scale_tensor.ndim != 2:
            return scale_tensor
        rows, cols = result_shape
        return scale_tensor[:min(scale_tensor.shape[0], rows), :min(scale_tensor.shape[1], cols // 32)]

    def _decoder_fp4_effective_scale(
        self,
        *,
        checkpoint_tensors: Dict[str, torch.Tensor],
        result_shape: tuple[int, ...],
    ) -> Optional[torch.Tensor]:
        """Resolve NVFP4 weight scales, including ModelOpt's secondary global scale."""

        direct_scale = checkpoint_tensors.get("weight_scale")
        if not isinstance(direct_scale, torch.Tensor):
            direct_scale = checkpoint_tensors.get("scale")
        if not isinstance(direct_scale, torch.Tensor):
            return None
        is_mx_block_scale = direct_scale.dtype == getattr(torch, "float8_e8m0fnu", None) or (
            direct_scale.dtype == torch.uint8 and "mxfp4" in self._decoder_declared_floatx_format()
        )
        if is_mx_block_scale:
            scale = self._decoder_mx_block_scale(
                scale_tensor=direct_scale,
                result_shape=result_shape,
                format_name="MXFP4",
            )
        else:
            scale = self._decoder_scale_tensor(
                scale_tensor=direct_scale,
                result_shape=result_shape,
            )
        scale_2 = checkpoint_tensors.get("weight_scale_2")
        if isinstance(scale_2, torch.Tensor):
            if is_mx_block_scale:
                scale_2 = self._decoder_mx_secondary_scale(scale_2, result_shape)
            scale = scale.to(torch.float32) * scale_2.to(torch.float32)
        return scale

    def _decoder_source_config(self) -> Any:
        """Keep checkpoint encoding metadata after the output config changes."""

        turtle_config = getattr(getattr(self, "turtle_model", None), "config", None)
        if turtle_config is not None:
            return turtle_config
        return self.__dict__.get("_source_model_config", getattr(self.model, "config", None))

    def _decoder_weight_format(
        self,
        *,
        weight: torch.Tensor,
        checkpoint_tensors: Dict[str, torch.Tensor],
        result_shape: Optional[tuple[int, ...]] = None,
        module_name: Optional[str] = None,
    ) -> Optional[str]:
        """Infer which floatx decoder matches one checkpoint weight tensor."""

        from ..utils.mimo import is_mimo_encoded_weight

        if module_name and is_mimo_encoded_weight(
            self._decoder_source_config(), module_name + ".weight", weight
        ):
            return "mimo_mixed"

        if weight.dtype in available_float8_dtypes():
            return "fp8"
        if is_fp4_packed_dtype(weight.dtype):
            return "fp4"
        direct_scale = checkpoint_tensors.get("weight_scale")
        if not isinstance(direct_scale, torch.Tensor):
            direct_scale = checkpoint_tensors.get("scale")
        if weight.dtype not in {torch.uint8, torch.int8} or not isinstance(direct_scale, torch.Tensor):
            return None
        if (
            result_shape is not None
            and len(result_shape) == weight.ndim
            and tuple(result_shape[:-1]) == tuple(weight.shape[:-1])
            and result_shape[-1] == weight.shape[-1] * 2
        ):
            return "fp4"
        if isinstance(checkpoint_tensors.get("weight_scale_2"), torch.Tensor):
            return "fp4"

        quant_config = getattr(getattr(self.model, "config", None), "quantization_config", None)
        if isinstance(quant_config, dict):
            format_name = quant_config.get("format") or quant_config.get("quant_method")
        else:
            format_name = getattr(quant_config, "format", None) or getattr(quant_config, "quant_method", None)
        if str(format_name or "").strip().lower() in {"nvfp4", "fp4", "mxfp4"} or "mxfp4" in self._decoder_declared_floatx_format():
            return "fp4"
        return None

    def _build_decoder_forward_module(
        self,
        *,
        quant_source: nn.Module,
        device: torch.device,
    ) -> nn.Module:
        """Clone the decoded quant source into a live forward module on ``device``."""

        forward_module = copy.deepcopy(quant_source)
        forward_module = forward_module.to(device=device)
        forward_module.eval()
        setattr(forward_module, "target_device", torch.device(device))
        return forward_module

    def _infer_fp8_forward_layout(
        self,
        *,
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
    ) -> tuple[str, Optional[tuple[int, int]]]:
        """Infer the FP8 scale layout needed to rebuild a TorchFP8Linear wrapper."""

        if scale_inv.numel() == 1:
            return "tensor", None
        if scale_inv.ndim == 1 and scale_inv.shape[0] == weight.shape[0]:
            return "row", None
        return "block", infer_block_shape(tuple(weight.shape), scale_inv)

    def _infer_fp4_forward_block_size(
        self,
        *,
        target_submodule: nn.Module,
        scale: torch.Tensor,
    ) -> int:
        """Infer the NVFP4 block size used along the input-feature axis."""

        block_size = self._decoder_block_size()
        if block_size is not None and target_submodule.in_features % block_size[1] == 0:
            return int(block_size[1])

        if scale.ndim >= 1 and scale.shape[-1] > 0 and target_submodule.in_features % scale.shape[-1] == 0:
            return int(target_submodule.in_features // scale.shape[-1])

        raise ValueError(
            f"Cannot infer FP4 block size for in_features={target_submodule.in_features} "
            f"and scale shape={tuple(scale.shape)}."
        )

    def _fp8_compact_block_size(
        self,
        *,
        weight: torch.Tensor,
        scale: Optional[torch.Tensor],
    ) -> Optional[tuple[int, int]]:
        """Return the checkpoint block size when ``scale`` is its compact grid.

        Some FP8 checkpoints use ceil blocking, so the scale grid is
        ``ceil(N / br) x ceil(K / bc)``. That compact grid is exactly the
        per-block layout ``TorchFP8Linear`` stores, so the loader must keep it
        instead of expanding it to ``N x K`` (an FP32 tensor four times the
        size of the FP8 weight) for the module's lifetime (#5593).
        """

        if not isinstance(scale, torch.Tensor) or scale.ndim != 2 or weight.ndim != 2:
            return None
        block_size = self._decoder_block_size()
        if block_size is None:
            return None
        block_rows, block_cols = block_size
        rows, cols = weight.shape
        compact = (
            (rows + block_rows - 1) // block_rows,
            (cols + block_cols - 1) // block_cols,
        )
        return block_size if tuple(scale.shape) == compact else None

    def _build_fp8_forward_module(
        self,
        *,
        target_submodule: nn.Module,
        checkpoint_tensors: Dict[str, torch.Tensor],
        device: torch.device,
        target_dtype: torch.dtype,
    ) -> Optional[nn.Module]:
        """Rebuild one linear submodule as a TorchFP8Linear forward wrapper."""

        if not isinstance(target_submodule, nn.Linear):
            return None
        if any(name in self._decoder_declared_floatx_format() for name in ("mxfp8", "nvfp8")):
            # The plain FP8 wrapper does not execute MX block scales natively.
            return None

        weight = checkpoint_tensors.get("weight")
        if not isinstance(weight, torch.Tensor):
            return None

        raw_scale_inv = checkpoint_tensors.get("weight_scale_inv")
        # A compact ceil-blocked grid is already the wrapper's per-block layout;
        # expanding it to N x K would keep an FP32 scale four times the size of
        # the FP8 weight for the module's lifetime (#5593).
        compact_block_size = self._fp8_compact_block_size(weight=weight, scale=raw_scale_inv)
        if isinstance(raw_scale_inv, torch.Tensor):
            scale_semantics = self._decoder_fp8_scale_semantics()
            if compact_block_size is not None:
                scale_inv = raw_scale_inv
            else:
                scale_inv = self._decoder_scale_tensor(
                    scale_tensor=raw_scale_inv,
                    result_shape=tuple(weight.shape),
                )
        else:
            # ModelOpt-style FP8 checkpoints store direct scales instead of inverse scales;
            # normalize them here so TorchFP8Linear can use one consistent metadata form.
            direct_scale = checkpoint_tensors.get("weight_scale")
            if not isinstance(direct_scale, torch.Tensor):
                direct_scale = checkpoint_tensors.get("scale")
            compact_block_size = self._fp8_compact_block_size(weight=weight, scale=direct_scale)
            if compact_block_size is not None:
                scale = direct_scale
            else:
                scale = self._decoder_scale_tensor(
                    scale_tensor=direct_scale,
                    result_shape=tuple(weight.shape),
                )
            if not isinstance(scale, torch.Tensor):
                return None
            scale = scale.to(torch.float32)
            tiny = torch.finfo(torch.float32).tiny
            scale_inv = torch.where(
                scale != 0,
                torch.reciprocal(scale),
                torch.full_like(scale, 1.0 / tiny),
            )
            # The synthesized tensor is an inverse scale regardless of any
            # checkpoint-declared semantics, so the wrapper must divide.
            scale_semantics = "inverse"

        format_name = str(weight.dtype).split(".")[-1]
        try:
            # Infer the wrapper layout from the normalized inverse-scale tensor so native
            # FP8 execution works for either checkpoint convention. A compact
            # ceil-blocked grid keeps the checkpoint block size, which its shape
            # alone cannot express.
            if compact_block_size is not None:
                weight_scale_method, weight_block_size = "block", compact_block_size
            else:
                weight_scale_method, weight_block_size = self._infer_fp8_forward_layout(
                    weight=weight,
                    scale_inv=scale_inv,
                )
            forward_module = TorchFP8Linear(
                bits=8,
                group_size=-1,
                desc_act=False,
                sym=True,
                in_features=target_submodule.in_features,
                out_features=target_submodule.out_features,
                bias=target_submodule.bias is not None,
                pack_dtype=torch.int32,
                format=format_name,
                weight_scale_method=weight_scale_method,
                weight_block_size=weight_block_size,
                weight_scale_semantics=scale_semantics,
                register_buffers=False,
            ).to(device=device)
        except Exception:
            # Some checkpoints use padded or otherwise non-TorchFP8Linear layouts and must
            # fall back to the decoded dense path even on native-FP8-capable GPUs.
            return None
        forward_module.register_buffer("weight", weight.to(device=device))
        forward_module.register_buffer(
            "weight_scale_inv",
            scale_inv.to(device=device, dtype=torch.float32),
        )

        bias = checkpoint_tensors.get("bias")
        if isinstance(bias, torch.Tensor):
            forward_module.register_buffer(
                "bias",
                bias.to(device=device, dtype=target_dtype),
            )
        else:
            forward_module.bias = None

        forward_module.eval()
        setattr(forward_module, "target_device", torch.device(device))
        return forward_module

    def _build_fp4_forward_module(
        self,
        *,
        target_submodule: nn.Module,
        checkpoint_tensors: Dict[str, torch.Tensor],
        device: torch.device,
        target_dtype: torch.dtype,
    ) -> Optional[nn.Module]:
        """Rebuild one linear submodule as a native NVFP4 forward wrapper."""

        if not isinstance(target_submodule, nn.Linear):
            return None
        if "mxfp4" in self._decoder_declared_floatx_format():
            # torchao's NVFP4 wrapper expects 16-wide E4M3 scales, not MX E8M0.
            return None

        weight = checkpoint_tensors.get("weight")
        scale = self._decoder_fp4_effective_scale(
            checkpoint_tensors=checkpoint_tensors,
            result_shape=(target_submodule.out_features, target_submodule.in_features),
        )
        if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
            return None

        try:
            block_size = self._infer_fp4_forward_block_size(
                target_submodule=target_submodule,
                scale=scale,
            )
            forward_module = TorchFP4Linear(
                in_features=target_submodule.in_features,
                out_features=target_submodule.out_features,
                weight=weight.to(device=device),
                weight_scale=scale.to(device=device),
                weight_block_size=block_size,
                orig_dtype=target_dtype,
                # Keep the checkpoint bias at its stored width. Narrowing it to the
                # activation dtype here would overflow a value such as an FP32 -65536
                # into FP16 -inf before the epilogue can apply it, defeating the
                # wide-bias handling in TorchFP4Linear.
                bias=checkpoint_tensors.get("bias").to(device=device)
                if isinstance(checkpoint_tensors.get("bias"), torch.Tensor)
                else None,
            )
        except Exception:
            return None

        forward_module.eval()
        setattr(forward_module, "target_device", torch.device(device))
        return forward_module

    def _record_auto_module_decoder_event(
        self,
        *,
        named_module: "NamedModule",
        device: torch.device,
        forward_mode: str,
        source_dtype: torch.dtype,
        target_dtype: torch.dtype,
        reason: str,
        deferred_quant_source_decode: bool,
    ) -> None:
        """Store one auto-decoder decision so tests can assert the chosen path."""

        if named_module.state.get("_auto_module_decoder_event_recorded"):
            return

        event = {
            "module": named_module.full_name,
            "device": str(device),
            "forward_mode": forward_mode,
            "source_dtype": str(source_dtype).split(".")[-1],
            "target_dtype": str(target_dtype).split(".")[-1],
            "reason": reason,
            "deferred_quant_source_decode": deferred_quant_source_decode,
            "hessian_accumulation_dtype": (
                (named_module.state.get("auto_module_decoder") or {}).get("runtime_plan") or {}
            ).get("hessian_accumulation_dtype", "float32"),
        }
        self.auto_module_decoder_events.append(event)
        emit_device_telemetry("floatx_forward_plan", **event)
        if forward_mode == "native":
            emit_device_telemetry("floatx_native_forward_used", **event)
        else:
            emit_device_telemetry("floatx_bf16_fallback_forward_used", **event)
        named_module.state["_auto_module_decoder_event_recorded"] = True

    def _prepare_auto_decoder_forward_module(
        self,
        *,
        target_submodule: nn.Module,
        device: torch.device,
        named_module: "NamedModule",
    ) -> nn.Module:
        """Build the first-pass floatx view and defer dense BF16 until quantization."""

        decoder_plan = named_module.state.get("auto_module_decoder")
        if not isinstance(decoder_plan, dict):
            # Passthrough modules participate in the layer forward but are not
            # visited by ModulePreProcessor. They still need the run's active
            # checkpoint decoder when a lazy packed source backs their shell.
            decoder_config = self._active_auto_module_decoder_config()
            if decoder_config is not None:
                decoder_plan = {
                    "code": decoder_config.code,
                    "source_dtype": decoder_config.source_dtype,
                    "target_dtype": decoder_config.target_dtype,
                }
                named_module.state["auto_module_decoder"] = decoder_plan
        turtle_model = self.turtle_model
        if not isinstance(decoder_plan, dict) or turtle_model is None:
            return target_submodule

        # A subset may request the same module again before its quantization
        # hand-off.  Keep the already materialized native/decode forward view.
        if named_module.state.get("auto_module_decoder_forward_mode") in {"native", "decode"}:
            return target_submodule

        checkpoint_tensors = turtle_model.checkpoint_tensors_for_submodule(
            target_model=self.model,
            target_submodule=target_submodule,
            recurse=False,
            module_path=named_module.full_name,
        )
        weight = checkpoint_tensors.get("weight")
        if not isinstance(weight, torch.Tensor):
            return target_submodule

        decoder_kind = self._decoder_weight_format(
            weight=weight,
            checkpoint_tensors=checkpoint_tensors,
            result_shape=tuple(getattr(target_submodule.weight, "shape", weight.shape)),
            module_name=named_module.full_name,
        )
        if decoder_kind is None:
            return target_submodule

        target_dtype = decoder_plan.get("target_dtype", target_submodule.weight.dtype)
        forward_policy = str(decoder_plan.get("passthrough_forward_policy", "native")).strip().lower()
        preserve_subclass_forward = type(target_submodule) is not nn.Linear
        defer_quant_source_decode = (
            decoder_plan.get("quant_source_decode_policy") == "defer_until_quantization"
        )
        # Keep only lightweight shell/metadata references here.  In particular,
        # do not expand a full BF16 copy before the Hessian collection forward.
        named_module.state.setdefault("decoder_quant_source_template", target_submodule)
        named_module.state.setdefault("decoder_checkpoint_tensors", checkpoint_tensors)

        forward_mode = "decode"
        reason = "native_execution_disabled"
        replacement = target_submodule
        if forward_policy != "decode" and not preserve_subclass_forward and decoder_kind == "fp8" and device_supports_dtype(device, weight.dtype, require_validation=True):
            fp8_module = self._build_fp8_forward_module(
                target_submodule=target_submodule,
                checkpoint_tensors=checkpoint_tensors,
                device=device,
                target_dtype=target_dtype,
            )
            if fp8_module is not None:
                replacement = self._replace_live_submodule(
                    target_submodule,
                    fp8_module,
                    module_path=named_module.full_name,
                )
                forward_mode = "native"
                reason = "validated_native_fp8"
            else:
                reason = "native_fp8_wrapper_unavailable"
        elif forward_policy != "decode" and not preserve_subclass_forward and decoder_kind == "fp4" and device_supports_native_fp4(device, require_validation=True):
            fp4_module = self._build_fp4_forward_module(
                target_submodule=target_submodule,
                checkpoint_tensors=checkpoint_tensors,
                device=device,
                target_dtype=target_dtype,
            )
            if fp4_module is not None:
                replacement = self._replace_live_submodule(
                    target_submodule,
                    fp4_module,
                    module_path=named_module.full_name,
                )
                forward_mode = "native"
                reason = "validated_native_nvfp4"
            else:
                reason = "native_nvfp4_wrapper_unavailable"
        elif forward_policy != "decode":
            reason = f"native_{decoder_kind}_unsupported_on_{torch.device(device)}"
        if forward_mode == "decode":
            quant_source = named_module.state.get("quant_source_module")
            if not isinstance(quant_source, nn.Module):
                quant_source = self._build_decoder_quant_source_module(
                    target_submodule,
                    checkpoint_tensors=checkpoint_tensors,
                    target_dtype=target_dtype,
                    module_name=named_module.full_name,
                )
                named_module.state["quant_source_module"] = quant_source
            decoded_forward = self._build_decoder_forward_module(
                quant_source=quant_source,
                device=device,
            )
            replacement = self._replace_live_submodule(
                target_submodule,
                decoded_forward,
                module_path=named_module.full_name,
            )

        if forward_mode == "native":
            log.info(
                "Floatx forward plan: %s on %s uses native %s for Hessian collection; BF16 decode is deferred until quantization.",
                named_module.full_name,
                device,
                decoder_kind.upper(),
            )
        else:
            log.warning(
                "Floatx forward fallback: %s on %s uses decoded BF16 (%s); Hessian collection may be slower.",
                named_module.full_name,
                device,
                reason,
            )

        named_module.state["auto_module_decoder_forward_mode"] = forward_mode
        self._record_auto_module_decoder_event(
            named_module=named_module,
            device=torch.device(device),
            forward_mode=forward_mode,
            source_dtype=weight.dtype,
            target_dtype=target_dtype,
            reason=reason,
            deferred_quant_source_decode=defer_quant_source_decode and forward_mode == "native",
        )
        return replacement

    def move_embed(self, device: str):
        for embed_module_name in self.get_base_modules(self.model):
            embed_module, _ = get_module_by_name_prefix(self.model, embed_module_name)
            if embed_module is not None:
                self.shell_module_materialize(
                    target_submodule=embed_module,
                    device=device,
                )

    def awq_skip_modules_for_scaling(self) -> bool:
        pass

    @classmethod
    def awq_input_feature_aggregation(cls, module_name: str) -> Optional[Dict[str, Any]]:
        """Declare bounded token-row aggregation for every AWQ module.

        Variable-length calibration captures cannot be concatenated on the
        batch axis. Before #4063 only MoE modules opted into token-row packing;
        every dense module fell back to the latest (shortest, because
        calibration is sorted by length) capture, so AWQ scale search saw less
        than one percent of the calibration tokens. Token-row packing is the
        default for every module and keeps a bounded, deterministic sample of
        all captures. MoE roots additionally capture their shared pre-router
        input for the gate/up scaling group.
        """

        if not isinstance(module_name, str):
            return None

        plan = ModulePlan.from_tree(_module_tree_for_class(cls))
        for moe_root in (spec.template for spec in plan.moe_roots()) or cls.get_moe_module_name() or []:
            if module_name == moe_root:
                return {
                    "mode": "token_rows",
                    "capture_root": True,
                }
            if module_name.startswith(f"{moe_root}."):
                return {
                    "mode": "token_rows",
                }

        return {
            "mode": "token_rows",
        }

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        nodes = []
        last_module = None  # most recent norm obj (from a '!...' block)
        last_module_name = None
        last_module_root = None  # self_attn.* has root == self_attn, mlp.* has root == mlp
        if isinstance(module_kwargs, dict):
            per_feature_kwargs = module_kwargs.get("_awq_feature_kwargs", {})
            base_module_kwargs = {
                key: value
                for key, value in module_kwargs.items()
                if key != "_awq_feature_kwargs"
            }
        else:
            per_feature_kwargs = {}
            base_module_kwargs = module_kwargs

        if self.model.config is not None and self.dynamic_expert_index is not None:
            self.get_num_experts(self.model.config)

        def strip_non_quantize_flags(module_name):
            for flag in NON_QUANTIZE_FLAGS:
                if flag in module_name:
                    module_name = module_name.replace(flag, "")
            return module_name

        # A sandwich post-norm (for example LFM3's `post_operator_norm`) is
        # applied to the block output, after the residual add. It can appear in
        # the tree immediately before the next consumer and share that
        # consumer's width, so position and width cannot identify a fold
        # source. The definition marks it explicitly instead.
        post_norm_modules = set(self.awq_post_norm_modules(module))

        def _select_feature_name(names):
            """Return the first quantized child that has captured activations."""
            for raw in names:
                stripped = strip_non_quantize_flags(raw)
                if stripped in input_feat:
                    return stripped
            return strip_non_quantize_flags(names[0]) if names else None

        def _feature_numel(name):
            """Number of captured activation elements for ``name`` (0 if absent).

            A declared MoE root is seeded as a capture task even on a layer that
            never runs an expert (ERNIE-4.5-MoE's dense prefix), so the key
            exists with an empty tensor. Treat "key present" and "captured
            something" as different questions (#4903).
            """
            feature = input_feat.get(name) if name else None
            return int(feature.numel()) if feature is not None else 0

        def _try_update_last_module(candidate_name: str) -> bool:
            nonlocal last_module, last_module_name, last_module_root

            resolved_module, _ = get_module_by_name_prefix(module, candidate_name)
            if resolved_module is None:
                log.debug(
                    "awq_get_modules_for_scaling: last-module candidate `%s` missing; retaining previous `%s`",
                    candidate_name,
                    last_module_name,
                )
                return False

            last_module = resolved_module
            last_module_name = candidate_name
            if "." in candidate_name:
                last_module_root = candidate_name.split(".", 1)[0]
            return True

        def _is_gain_norm(candidate_module, candidate_name: str, input_width: int | None) -> bool:
            """Whether ``candidate_module`` can absorb the consumer's input scale."""
            if candidate_name in post_norm_modules:
                return False
            if norm_type_of(candidate_module) is None:
                return False
            weight = getattr(candidate_module, "weight", None)
            if not isinstance(weight, nn.Parameter) or weight.ndim != 1:
                return False
            return input_width is None or weight.numel() == input_width

        last_gain_norm = None
        # The last norm-like module, even when it cannot absorb a fold (a
        # weightless or gated norm). A MoE gate/up block reads this module's
        # output; resolving to it lets the foldability filter drop the group
        # instead of folding into an unrelated preceding linear.
        last_norm_candidate = None

        def _select_prev_op(
            input_width: int | None,
            allow_block_norm: bool = False,
            shared_source: nn.Module | None = None,
            refuse_last_module_fallback: bool = False,
        ):
            """Return the normalization that immediately feeds the consumer block.

            Consecutive non-quantized blocks can contain several modules: the
            post-attention/post-MLP norms, MoE routers, and capture-only roots.
            Only the last gain norm whose width matches the consumer's input
            actually feeds it; earlier entries in the run belong to sibling
            branches (for example the router) or to a preceding residual.
            ``last_module`` remains the fallback when no norm matches.

            ``allow_block_norm`` admits the last gain norm seen in the layer when
            the run was already consumed by a sibling MoE block. A MoE gate/up
            group always reads the MoE block input, so its fold source is that
            norm even when the intervening routed-expert blocks cleared the run
            (DeepSeek orders routed experts before ``shared_experts``).

            ``shared_source`` is the source of an earlier block whose consumers
            declare the same ``:in=<tag>`` input. Parallel decoder blocks (for
            example Cohere/Cohere2 attention and MLP) read one norm without an
            intervening norm block, so the consumer's input tag is the only
            declaration that both branches share the same source.

            ``refuse_last_module_fallback`` is set when the block immediately
            before this one declared only modules that this layer does not own
            **and** that absent declaration is a data-flow predecessor of this
            block (for example the MLA `q_b_proj`/`kv_b_proj` pair before
            `attention.dense` on a Ling 2.0 layer). `last_module` then still
            points at the run before that absent block, which is not this
            block's producer: falling back to it folded `attention.dense` into
            `input_layernorm` (#4677).

            An absent *sibling branch* never clears the source: `mamba` vs
            `self_attn`, `conv` vs `self_attn`, or `self_attn` vs `attention`
            are alternative mixers of one layer, so the norm before them still
            feeds this block (#4759). The caller also clears the flag when this
            block declares the same `:in=<tag>` shared input as the absent
            block: the two are siblings reading one tensor (#4732).
            """
            for candidate_name, candidate_module in reversed(pending_nonquant):
                if _is_gain_norm(candidate_module, candidate_name, input_width):
                    return candidate_module
            if last_module_name in post_norm_modules:
                # Folding into a post-norm corrupts the block output. With no
                # data-flow source in this run, skip the scale instead.
                return None
            if allow_block_norm and last_gain_norm is not None:
                # A MoE gate/up block always reads the block input norm. Return
                # it even when it is not foldable or its width differs; the
                # foldability/width filters drop the group rather than folding a
                # scale into an unrelated preceding linear (for example the last
                # routed expert's down projection). The most recent gain norm wins
                # over a shared `:in=` source because a sequential MoE block reads
                # its post-attention norm, not the attention input norm.
                return last_gain_norm
            if allow_block_norm and last_norm_candidate is not None:
                return last_norm_candidate
            if shared_source is not None and _norm_width(shared_source) == input_width:
                return shared_source
            if refuse_last_module_fallback:
                # Nothing in this run produces the consumer's input and the
                # preceding declaration resolved to no module on this layer.
                # Leave the group unscaled instead of folding a stale source.
                return None
            # A fallback norm whose width differs from the consumer's input
            # cannot absorb this scale: `scale_ln_fcs` would multiply a
            # differently sized weight (or raise). DeepSeek MLA is the concrete
            # case - `o_proj` reads the attention output, not the input norm, so
            # the last-module fallback would fold a wrong-width scale into
            # `input_layernorm`. Refuse the fold and leave that group unscaled.
            if last_module is not None and _norm_width(last_module) is not None:
                return last_module if _norm_width(last_module) == input_width else None
            return last_module

        def _norm_width(candidate_module) -> int | None:
            """Return the gain width of a norm-like module, else ``None``."""

            weight = getattr(candidate_module, "weight", None)
            if isinstance(weight, nn.Parameter) and weight.ndim == 1:
                return int(weight.numel())
            return None

        def _norm_scale_consumers(prev_op_module, input_width: int | None, extra_names=(), block=()):
            """Non-quantized modules in this run that read ``prev_op``'s output.

            A MoE router (``mlp.gate``, ``router``) and Qwen2's
            ``shared_expert_gate`` consume the same normed hidden state as the
            quantized expert gate/up projections but are not part of the fold
            group, so ``apply_scale`` has to multiply their input columns by the
            same scale. Routers are not ``nn.Linear``; they own a 2-D ``weight``
            of shape ``[experts, hidden]`` and apply it as a linear map, so both
            shapes are accepted here and ``scale_input_columns`` performs the
            multiplication.

            Non-quantized siblings declared in the same block (for example
            Laguna's ``self_attn.g_proj:!``) read the same normed input too, so
            they are collected from ``block`` as well.
            """

            width = _norm_width(prev_op_module)
            if width is None or input_width is None or width != input_width:
                return []
            consumers = []
            for candidate_name, candidate_module in pending_nonquant:
                if candidate_module is prev_op_module or candidate_name in post_norm_modules:
                    continue
                if isinstance(candidate_module, nn.Linear):
                    if candidate_module.in_features == width:
                        consumers.append(candidate_name)
                    continue
                weight = getattr(candidate_module, "weight", None)
                if isinstance(weight, nn.Parameter) and weight.ndim == 2 and weight.shape[1] == width:
                    consumers.append(candidate_name)
            for raw_name in block:
                if all(flag not in raw_name for flag in NON_QUANTIZE_FLAGS):
                    continue
                candidate_name = strip_non_quantize_flags(raw_name)
                if candidate_name in consumers:
                    continue
                candidate_module, _ = get_module_by_name_prefix(module, candidate_name)
                if candidate_module is None or candidate_module is prev_op_module:
                    continue
                if isinstance(candidate_module, nn.Linear):
                    if candidate_module.in_features == width:
                        consumers.append(candidate_name)
                    continue
                weight = getattr(candidate_module, "weight", None)
                if isinstance(weight, nn.Parameter) and weight.ndim == 2 and weight.shape[1] == width:
                    consumers.append(candidate_name)
            for extra_name in extra_names:
                if extra_name not in consumers:
                    consumers.append(extra_name)
            return consumers

        def _block_moe_root(block):
            """Return the declared MoE root template that owns ``block``."""

            for spec in plan.moe_roots():
                template = spec.template
                if any(
                    strip_non_quantize_flags(name) == template
                    or strip_non_quantize_flags(name).startswith(f"{template}.")
                    for name in block
                ):
                    return template
            return None

        def _moe_block_consumers(block, width, excluded):
            """Direct children of the MoE block that read its input.

            Some definitions do not declare the router (DeepSeek V2/V3 trees list
            only the projections), so the router cannot be found in the
            non-quantized run. A router is always a direct child of the MoE block
            with a 2-D ``weight`` of the block-input width; taking it from the
            module hierarchy keeps the rule structural instead of name-based.
            """

            moe_root = _block_moe_root(block)
            if moe_root is None:
                return []
            root_module, _ = get_module_by_name_prefix(module, moe_root)
            if root_module is None:
                return []
            consumers = []
            for child_name, child in root_module.named_children():
                if any(child is excluded_module for excluded_module in excluded):
                    continue
                child_path = f"{moe_root}.{child_name}"
                spec = plan.find(child_path)
                if spec is not None and spec.quantize:
                    continue
                weight = getattr(child, "weight", None)
                if isinstance(weight, nn.Parameter) and weight.ndim == 2 and weight.shape[1] == width:
                    consumers.append(child_path)
            return consumers

        def _module_kwargs_for_feature(feature_name: str | None):
            kwargs_for_feature = dict(base_module_kwargs)
            if feature_name and isinstance(per_feature_kwargs, dict):
                feature_specific_kwargs = per_feature_kwargs.get(feature_name)
                if isinstance(feature_specific_kwargs, dict):
                    kwargs_for_feature.update(feature_specific_kwargs)
            return kwargs_for_feature

        full_layer_modules = self.full_layer_modules(
            self.model.config,
            is_awq_quantize=True,
            include_capture_only=True,
        )
        # Non-quantized blocks seen since the last quantized block. The run is
        # resolved at the consumer, where the input width is known.
        pending_nonquant = []
        # Source norm recorded for each declared ``:in=<tag>`` input. Parallel
        # decoder blocks consume one norm through two quantized branches; the
        # second branch reuses the first branch's source instead of folding into
        # the previous branch's output projection.
        shared_input_sources: Dict[str, nn.Module] = {}
        prev_not_quantized = False
        # True when the previous block declared only modules this layer does not
        # own (`layer_modules_strict=False` alternative layouts).
        prev_block_absent = False
        # Input tags declared by that absent block. A block that resolved to no
        # module still declares which tensor it would have read, so a following
        # block with the same tag is its sibling rather than its consumer.
        prev_absent_block_tags: set[str] = set()
        # Branch (plan parent) of each block in that run of absent declarations.
        prev_absent_block_parents: set[str] = set()
        # Branch of every quantized block emitted since the last declared
        # non-quantized block. A run of absent declarations that follows a
        # present block of the same branch is a data-flow predecessor of the
        # consumer: it sits downstream of a projection this layer owns.
        present_block_parents: set[str] = set()
        for block in full_layer_modules:
            block_absent = False
            not_quantized = all(any(flag in name for flag in NON_QUANTIZE_FLAGS) for name in block)
            if not_quantized:
                # An empty block carries no module to remember; it also must not
                # break the surrounding run of non-quantized blocks.
                if not block:
                    continue
                # Remember the first block of the run as the fallback `last_module`.
                # Every block in the run is kept as a candidate prev_op; the
                # consumer selects the gain norm that actually feeds it.
                candidate_name = strip_non_quantize_flags(block[-1])
                resolved_module, _ = get_module_by_name_prefix(module, candidate_name)
                if resolved_module is not None:
                    pending_nonquant.append((candidate_name, resolved_module))
                    if _is_gain_norm(resolved_module, candidate_name, None):
                        last_gain_norm = resolved_module
                    if candidate_name not in post_norm_modules and norm_type_of(resolved_module) is not None:
                        last_norm_candidate = resolved_module
                if not prev_not_quantized:
                    _try_update_last_module(candidate_name)
                prev_not_quantized = True
                # A declared norm block is a real producer for the next run.
                prev_block_absent = False
                prev_absent_block_tags = set()
                prev_absent_block_parents = set()
                present_block_parents = set()
                continue

            prev_not_quantized = False
            try:
                plan = self.module_plan()
            except RuntimeError:
                plan = ModulePlan.from_tree(_module_tree_for_class(type(self)))
            block_roles = {
                name: plan.roles(strip_non_quantize_flags(name))
                for name in block
            }
            is_moe_block = any(
                plan.is_expert(strip_non_quantize_flags(name))
                for name in block
            )
            is_moe_family = any(
                plan.is_moe(strip_non_quantize_flags(name))
                for name in block
            )
            has_gate_up_roles = any("gate" in roles or "up" in roles for roles in block_roles.values())
            is_moe_down_block = is_moe_block and any("down" in roles for roles in block_roles.values())
            is_moe_gate_up_block = is_moe_block and has_gate_up_roles
            block_input_tags = set()
            block_parents = set()
            for name in block:
                if all(flag not in name for flag in NON_QUANTIZE_FLAGS):
                    spec = plan.find(self._parse_module_flags(name)[0])
                    if spec is not None:
                        block_input_tags.update(flag[3:] for flag in spec.flags if flag.startswith("in="))
                        block_parents.add(spec.parent)
            shared_tag = next(iter(block_input_tags)) if len(block_input_tags) == 1 else None
            # Qwen3.5-style hybrid trees declare `self_attn` and `linear_attn`
            # against one `:in=` tag, and a linear-attention layer owns no
            # `self_attn`. The absent `self_attn` block still declares that tag,
            # so a following block with the same tag reads the same tensor the
            # absent block read: the two are siblings, and the norm before them
            # is still the fold source. Refusing the fallback here dropped the
            # whole DeltaNet input group (#4732).
            sibling_of_absent_block = (
                shared_tag is not None and shared_tag in prev_absent_block_tags
            )
            # An absent declaration may only take the fold source away when it
            # is a data-flow predecessor of this block, never when it is a
            # sibling branch the layer does not own. Sibling branches are
            # declared under a different plan parent (`mamba` vs `self_attn`,
            # `conv` vs `self_attn`, `self_attn` vs `attention`). The remaining
            # case is the same branch (`attention.q_b/kv_b` before
            # `attention.dense`), which only interrupts the flow when a
            # quantized block of that branch was already emitted (#4677).
            absent_in_same_branch = bool(block_parents & prev_absent_block_parents)
            branch_already_quantized = bool(block_parents & present_block_parents)
            absent_is_predecessor = absent_in_same_branch and branch_already_quantized
            skip_block = False
            if is_moe_down_block and last_module is not None and last_module_name is not None:
                pending_nonquant.clear()
                # mlp.experts.0.down_proj
                target_suffix = last_module_name.split(".")[-1]
                for name in block:
                    prev_op_name = ".".join(name.split(".")[:-1] + [target_suffix])
                    prev_op, _ = get_module_by_name_prefix(module, prev_op_name)
                    if prev_op is None or name not in input_feat:
                        log.debug("awq_get_modules_for_scaling: skipping expert `%s` due to missing prev_op or features", name)
                        continue

                    m, _ = get_module_by_name_prefix(module, name)
                    if m is None:
                        log.debug("awq_get_modules_for_scaling: skipping missing expert module `%s`", name)
                        continue
                    subset = [m]
                    feature_name = name
                    n, root = generate_node_for_awq_scaling(inp=input_feat[name], prev_op=prev_op,
                                                            module_kwargs=_module_kwargs_for_feature(feature_name), nodes_size=len(nodes),
                                                            subset=subset, module2inspect=None)
                    n["_input_feature_name"] = feature_name
                    if root is not None and last_module_root != root:
                        last_module_root = root

                    nodes.append(n)
            else:
                # Normal execution subset
                subset = []  # preserve execution order while collecting quantizable modules
                subset_names = []
                skip = False
                for name in block:
                    if all(flag not in name for flag in NON_QUANTIZE_FLAGS):
                        m, _ = get_module_by_name_prefix(module, name)
                        if m is None:
                            log.debug("awq_get_modules_for_scaling: skipping missing module `%s`", name)
                            # Definitions with alternative layouts (DeepSeek MLA
                            # ships `q_proj` only when `q_lora_rank` is unset)
                            # declare every variant. `layer_modules_strict=False`
                            # means the missing one is simply not executed, so it
                            # must not drop the rest of the block.
                            if self.layer_modules_strict:
                                skip = True
                                break
                            continue
                        # If the Model uses GQA (Grouped Query Attention), attention out will be skipped.
                        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                        if (self.awq_scale_optimize_shape_dependent_modules is not None
                                and name in self.awq_scale_optimize_shape_dependent_modules
                                and isinstance(last_module, nn.Linear)
                                and last_module.weight.shape != m.weight.shape):
                            # log.debug(f'"{name}" attention out skipped.')
                            skip = True

                        subset.append(m)
                        subset_names.append(name)

                block_absent = not subset and not skip
                if subset:
                    # This block really owns modules, so a later absent
                    # declaration in the same branch is downstream of it.
                    present_block_parents = present_block_parents | block_parents
                if skip or not subset:
                    pending_nonquant.clear()
                    skip_block = True

                if not skip_block:
                    # Match the activation bucket to the first quantized child in this block
                    feature_name = _select_feature_name(block) or strip_non_quantize_flags(block[0])
                    root_split = feature_name.split(".")
                    root = None
                    module2inspect = None
                    if len(root_split) >= 2:
                        root = root_split[0]
                        if root != last_module_root:
                            last_module_root = root
                            module2inspect, _ = get_module_by_name_prefix(module, root)

                    # A MoE gate/up block always reads the MoE block input, which
                    # is captured once per calibration batch. Sibling blocks that
                    # share the root (a shared expert, or a dense MLP declared next
                    # to `mlp:moe`) move `last_module_root` to the root before the
                    # experts block, so the `root != last_module_root` gate above
                    # leaves `module2inspect` unset and anchors the group on one of
                    # its own members. #4675's partial split then never runs
                    # (#4760). Prefer the root capture whenever it exists.
                    moe_gate_up_root = (
                        _block_moe_root(block) if has_gate_up_roles and is_moe_family else None
                    )
                    # A seeded-but-uncaptured root is not a usable search input:
                    # the dense prefix of a hybrid MoE (`mlp:moe` with a dense
                    # `mlp` block) seeds `mlp` but never runs the root hook, and
                    # preferring that empty tensor dropped the dense layer's
                    # norm -> gate/up group (#4903). Fall through to the block's
                    # own captured feature in that case; a routed-expert block
                    # keeps the fail-closed path below.
                    if moe_gate_up_root is not None and _feature_numel(moe_gate_up_root) > 0:
                        feature_name = moe_gate_up_root
                    # process ['mlp.experts.#.gate_proj', 'mlp.experts.#.gup_proj']
                    elif is_moe_gate_up_block and module2inspect is not None:
                        if last_module_root not in input_feat:
                            log.debug(
                                "awq_get_modules_for_scaling: missing input feature for `%s` while processing experts block (layer block size=%s)",
                                last_module_root,
                                len(block),
                            )
                        feature_name = last_module_root if last_module_root in input_feat else _select_feature_name(block)
                    else:
                        feature_name = _select_feature_name(block)
                    inp = input_feat.get(feature_name)

                    if inp is None:
                        pending_nonquant.clear()
                        log.debug("awq_get_modules_for_scaling: skipping block %s due to missing input features", block)
                        skip_block = True
                    elif inp.numel() == 0:
                        # An empty feature means the declared MoE-root capture
                        # never ran. A zero-width "source" also makes every
                        # consumer look width-foreign below, which used to drop
                        # the routed-expert gate/up group without a trace
                        # (#4697). Warn before skipping it, and warn for a
                        # non-expert gate/up block too: the dense prefix of a
                        # hybrid MoE used to lose its own group the same silent
                        # way (#4903).
                        if is_moe_gate_up_block:
                            log.warning(
                                "awq_get_modules_for_scaling: declared MoE root `%s` captured no "
                                "activations; skipping the routed-expert gate/up scaling group "
                                "and leaving those projections on plain quantization",
                                last_module_root,
                            )
                        elif has_gate_up_roles:
                            log.warning(
                                "awq_get_modules_for_scaling: feature `%s` captured no activations; "
                                "skipping the gate/up scaling group for `%s` and leaving those "
                                "projections on plain quantization",
                                feature_name,
                                block[0] if block else "<empty block>",
                            )
                        pending_nonquant.clear()
                        skip_block = True

                input_width = None
                scale_consumers = []
                if not skip_block:
                    input_width = inp.shape[-1] if inp.dim() > 0 else None
                    allow_block_norm = is_moe_family and has_gate_up_roles
                    shared_source = shared_input_sources.get(shared_tag) if shared_tag else None
                    prev_op = _select_prev_op(
                        input_width,
                        allow_block_norm=allow_block_norm,
                        shared_source=shared_source,
                        refuse_last_module_fallback=(
                            prev_block_absent
                            and absent_is_predecessor
                            and not sibling_of_absent_block
                        ),
                    )
                    moe_consumers = (
                        _moe_block_consumers(block, input_width, subset)
                        if is_moe_family and input_width is not None
                        else []
                    )
                    scale_consumers = _norm_scale_consumers(
                        prev_op,
                        input_width,
                        extra_names=moe_consumers,
                        block=block,
                    )
                    if (
                        prev_op is not None
                        and shared_tag
                        and _is_gain_norm(prev_op, "", input_width)
                    ):
                        shared_input_sources[shared_tag] = prev_op
                    pending_nonquant.clear()
                    if prev_op is None:
                        log.debug("awq_get_modules_for_scaling: skipping block %s due to missing previous module", block)
                        skip_block = True
                    elif not is_foldable_source(prev_op):
                        # A Conv1d (Falcon-H1's `mamba.conv1d`), a weightless
                        # norm or a gated norm has no plain gain to divide, so
                        # `apply_scale` would discard the group after paying for
                        # the search. Do not build it at all.
                        log.debug(
                            "awq_get_modules_for_scaling: skipping block %s; `%s` is not a foldable scale source",
                            block,
                            get_op_name(module, prev_op),
                        )
                        pending_nonquant.clear()
                        skip_block = True

                # One group shares one scale vector, so it can only contain
                # consumers of the same tensor. A layer whose input width differs
                # from the fold source reads another tensor (DeepSeek MLA:
                # `q_b_proj` reads `q_a`, `kv_b_proj` reads `kv_a`), and
                # multiplying it by this norm's scale would either raise in
                # `scale_ln_fcs` or silently corrupt the layer. Keep those layers
                # in their own quantized-but-unscaled group.
                if not skip_block and input_width is not None:
                    fold_subset, fold_names, foreign_layers = [], [], []
                    for layer_module, layer_name in zip(subset, subset_names):
                        weight = getattr(layer_module, "weight", None)
                        if isinstance(weight, torch.Tensor) and weight.ndim == 2 and weight.shape[1] != input_width:
                            foreign_layers.append(layer_name)
                            continue
                        fold_subset.append(layer_module)
                        fold_names.append(layer_name)
                    if foreign_layers:
                        log.debug(
                            "awq_get_modules_for_scaling: leaving %s unscaled; their input width "
                            "differs from the %s-wide fold source",
                            foreign_layers,
                            input_width,
                        )
                    if not fold_subset:
                        log.debug(
                            "awq_get_modules_for_scaling: skipping block %s; no consumer matches the fold source width",
                            block,
                        )
                        skip_block = True
                    subset, subset_names = fold_subset, fold_names

                # `_search_best_scale` scores a candidate scale by re-running
                # `module2inspect`. A multi-module group needs a block forward;
                # scoring a single member would miss the rest of the group.
                if not skip_block and module2inspect is None and len(subset) > 1:
                    common_parents = {strip_non_quantize_flags(name).rsplit(".", 1)[0] for name in subset_names}
                    inspect_name = common_parents.pop() if len(common_parents) == 1 else root
                    if inspect_name:
                        module2inspect, _ = get_module_by_name_prefix(module, inspect_name)

                if not skip_block:
                    n, root = generate_node_for_awq_scaling(inp=inp, prev_op=prev_op,
                                                            module_kwargs=_module_kwargs_for_feature(feature_name), nodes_size=len(nodes),
                                                            subset=subset, module2inspect=module2inspect)
                    n["_input_feature_name"] = feature_name
                    if scale_consumers:
                        n["_scale_consumers"] = scale_consumers

                    nodes.append(n)

            prev_block_absent = block_absent
            # Keep the tags of a whole run of absent blocks: Qwen3.5 splits
            # `self_attn` into the `q/k/v` block and the `o_proj` block, and only
            # the first declares `:in=`. Unioning them keeps the sibling tag
            # visible to the block that follows the run (#4732).
            prev_absent_block_tags = (
                (prev_absent_block_tags | set(block_input_tags)) if block_absent else set()
            )
            # Same union for the declared branch of the run, so the block that
            # follows a whole run of absent declarations still knows whether
            # they were its siblings or its predecessors.
            prev_absent_block_parents = (
                (prev_absent_block_parents | block_parents) if block_absent else set()
            )
            # Update tracker to the LAST item of this block
            if is_moe_gate_up_block:
                # Mixed MoE blocks can legitimately place shared-expert projections
                # before or after routed experts depending on real forward order.
                # For AWQ scaling, we still want the last routed expert gate/up proj
                # as the effective boundary for the expert segment in this block.
                gate_up_proj_indices = [
                    i for i, name in enumerate(block)
                    if plan.is_expert(strip_non_quantize_flags(name))
                    and ({"gate", "up"} & set(block_roles[name]))
                ]

                # Use the last one if any exist
                assert len(gate_up_proj_indices) > 0, "No expert gate_proj/up_proj found in block."
                last_up_proj_index = gate_up_proj_indices[-1]

                candidate_name = strip_non_quantize_flags(block[last_up_proj_index])
                assert {"gate", "up"} & set(block_roles[block[last_up_proj_index]])
            else:
                candidate_name = strip_non_quantize_flags(block[-1])
            _try_update_last_module(candidate_name)

        def format_nodes(nodes):
            out = []
            for n in nodes:
                entry = {}
                for k, v in n.items():
                    if isinstance(v, torch.Tensor):
                        entry[k] = f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
                    elif isinstance(v, dict):
                        entry[k] = [
                            f"Key: {kk}, Value: Tensor(shape={tuple(x.shape)}, dtype={x.dtype}); " if isinstance(x,
                                                                                                                 torch.Tensor) else type(
                                x).__name__
                            for kk, x in v.items()
                        ]
                    else:
                        entry[k] = v
                out.append(entry)
            return out

        # print("DEBUG AWQ NODES:", format_nodes(nodes))
        nodes = self._merge_shared_source_nodes(module, nodes)
        return self._filter_awq_scaling_sources(module, nodes)

    def _merge_shared_source_nodes(self, layer_module, nodes):
        """Fold every group that shares one gain norm into a single scale group.

        Routed experts and ``shared_experts`` read the same post-attention norm,
        and parallel decoder blocks read one input norm from two quantized
        branches. Folding those as separate groups divides that norm once per
        group, so each fold leaves the other group's consumers unscaled. One
        norm is divided exactly once, so it can have exactly one scale group.
        """

        merged = []
        index_by_source: Dict[int, int] = {}
        for node in nodes:
            prev_op = node.get("prev_op")
            gain = getattr(prev_op, "weight", None)
            if not isinstance(prev_op, nn.Module):
                merged.append(node)
                continue
            # A gated or weightless norm is still a single source, even though it
            # has no 1-D gain to divide; merging keeps one group per norm so a
            # later foldability check drops the whole group instead of half.
            has_gain = isinstance(gain, torch.Tensor) and gain.ndim == 1
            if not has_gain and norm_type_of(prev_op) is None:
                merged.append(node)
                continue
            key = id(prev_op)
            existing = index_by_source.get(key)
            if existing is None:
                index_by_source[key] = len(merged)
                merged.append(node)
                continue
            base = merged[existing]
            base_layers = list(base.get("layers") or [])
            for layer in node.get("layers") or []:
                if all(layer is not known for known in base_layers):
                    base_layers.append(layer)
            base["layers"] = base_layers
            consumers = list(base.get("_scale_consumers") or [])
            for consumer in node.get("_scale_consumers") or []:
                if consumer not in consumers:
                    consumers.append(consumer)
            if consumers:
                base["_scale_consumers"] = consumers
            # The merged group needs one forward that executes every member, so
            # re-anchor the inspection module on the deepest common ancestor of
            # all consumers (routed + shared experts resolve to the MoE root).
            try:
                names = [get_op_name(layer_module, layer) for layer in base_layers]
            except ValueError:
                names = []
            inspect_name = ""
            if names:
                inspect_name = names[0].rsplit(".", 1)[0] if "." in names[0] else ""
                for name in names[1:]:
                    parent = name.rsplit(".", 1)[0] if "." in name else ""
                    while inspect_name and not (parent == inspect_name or parent.startswith(f"{inspect_name}.")):
                        inspect_name = inspect_name.rsplit(".", 1)[0] if "." in inspect_name else ""
            # With no shared ancestor the branches live under different roots
            # (attention + MoE); keep the inspection module the first group
            # already resolved instead of widening it to the whole layer.
            if inspect_name:
                inspect_module, _ = get_module_by_name_prefix(layer_module, inspect_name)
                if inspect_module is not None:
                    base["module2inspect"] = inspect_module
            # Parallel branches of one norm (Falcon-H1's attention and Mamba
            # mixer) have no shared ancestor below the decoder layer, and the
            # captured replay input is the post-norm activation, so replaying
            # the layer itself would apply that norm twice. Score the members
            # the current boundary covers and fold the same scale into the rest:
            # every member reads this one tensor, so the fold is correct for
            # all of them even though only some can be scored (#4741).
            inspect_module = base.get("module2inspect")
            if inspect_module is not None and len(base_layers) > 1:
                inside = {id(child) for child in inspect_module.modules()}
                covered = [layer for layer in base_layers if id(layer) in inside]
                if covered and len(covered) < len(base_layers):
                    base["_awq_search_layers"] = covered
                    base["_awq_allow_outside_fold"] = True
        return merged

    def _filter_awq_scaling_sources(self, layer_module, nodes):
        """Drop scaling groups whose ``prev_op`` cannot be folded without changing the layer output.

        ``awq_get_modules_for_scaling`` selects ``prev_op`` from tree order, which
        cannot see a residual add, a parallel branch, or a non-homogeneous
        activation between the candidate and the consumer. A definition marks
        those leaves with the ``nosrc`` tree flag; folding a scale into them
        changes the decoder layer output, so the group is skipped here and the
        consumer falls back to plain quantization (the processor falls back to
        ``_quantize_layer_fallback`` when no group survives).

        The flag is a declaration of the definition's forward data flow, so it
        is validated by the ``apply_scale`` output-preservation invariant in
        ``tests/test_awq_scaling_groups.py`` for every family that uses it.
        """

        if not nodes:
            return nodes
        try:
            plan = self.module_plan()
        except RuntimeError:
            effective_tree = getattr(self, "effective_module_tree", None)
            plan = ModulePlan.from_tree(
                effective_tree if effective_tree is not None else _module_tree_for_class(type(self))
            )

        kept = []
        for node in nodes:
            prev_op = node.get("prev_op")
            prev_name = None
            if prev_op is not None:
                try:
                    prev_name = get_op_name(layer_module, prev_op)
                except Exception:
                    prev_name = None
            spec = plan.find(prev_name) if prev_name else None
            if spec is not None and AWQ_NO_SCALE_SOURCE_FLAG in spec.flags:
                log.debug(
                    "awq_get_modules_for_scaling: skipping scaling group with declared non-source prev_op `%s`",
                    prev_name,
                )
                continue
            # An excluded projection is not a compensated producer: dividing its
            # output rows changes a module that stays in the source dtype and
            # nothing multiplies it back, so the group cannot be
            # output-preserving. Only a gain norm (1-D gain, divided through
            # `scale_ln_fcs`) or a projection the plan quantizes may absorb a
            # fold. #4651: Jamba's excluded `mamba.dt_proj` became the source
            # for `mamba.x_proj` and was rescaled by up to 70%.
            source_weight = getattr(prev_op, "weight", None)
            if (
                spec is not None
                and not spec.quantize
                and isinstance(source_weight, torch.Tensor)
                and source_weight.ndim == 2
            ):
                log.debug(
                    "awq_get_modules_for_scaling: skipping scaling group with non-quantized prev_op `%s`",
                    prev_name,
                )
                continue
            # A gain norm can only absorb the consumer's input scale when its
            # width matches that input. A stale tracker entry (for example a
            # norm followed by a projection the definition does not list) would
            # otherwise reach apply_scale and fail on the shape mismatch.
            gain = getattr(prev_op, "weight", None)
            consumers = node.get("layers") or []
            if (
                isinstance(gain, torch.Tensor)
                and gain.ndim == 1
                and consumers
                and isinstance(consumers[0], nn.Linear)
                and gain.numel() != consumers[0].weight.shape[1]
            ):
                log.debug(
                    "awq_get_modules_for_scaling: skipping scaling group with width-mismatched prev_op `%s`",
                    prev_name,
                )
                continue
            kept.append(node)
        return kept

    # Materialize the target shell module from the lazy turtle source on the requested device.
    def shell_module_materialize(
            self,
            target_submodule: torch.nn.Module,
            device: torch.device,
            non_blocking: bool = False,
            role: str = "default",
            named_module: Optional["NamedModule"] = None,
            module_path: Optional[str] = None,
    ) -> torch.nn.Module:
        device = as_torch_device(device)
        if module_path is None and named_module is not None:
            module_path = named_module.full_name
        with self._turtle_lock:
            if role == "forward" and named_module is not None and named_module.state.get("auto_dtype_cast_source_spec") is not None:
                return target_submodule
            if role == "quant_source" and named_module is not None:
                source_spec = named_module.state.get("auto_dtype_cast_source_spec")
                quant_source = named_module.state.get("quant_source_module")
                if not isinstance(quant_source, nn.Module):
                    if source_spec is not None:
                        from ..utils.auto_dtype_cast import restore_source_module

                        quant_source = restore_source_module(target_submodule, source_spec, device=device)
                if not isinstance(quant_source, nn.Module):
                    decoder_plan = named_module.state.get("auto_module_decoder") or {}
                    target_dtype = decoder_plan.get(
                        "target_dtype",
                        getattr(getattr(target_submodule, "weight", None), "dtype", torch.float16),
                    )
                    checkpoint_tensors = named_module.state.get("decoder_checkpoint_tensors")
                    quant_source_template = named_module.state.get("decoder_quant_source_template")
                    if not isinstance(quant_source_template, nn.Module):
                        quant_source_template = target_submodule
                    if not isinstance(checkpoint_tensors, dict) and isinstance(self.turtle_model, LazyTurtle):
                        checkpoint_tensors = self.turtle_model.checkpoint_tensors_for_submodule(
                            target_model=self.model,
                            target_submodule=quant_source_template,
                            recurse=False,
                            module_path=module_path,
                        )
                    quant_source = self._build_decoder_quant_source_module(
                        quant_source_template,
                        checkpoint_tensors=checkpoint_tensors,
                        target_dtype=target_dtype,
                        module_name=named_module.full_name,
                    )
                    named_module.state["quant_source_module"] = quant_source
                    decoder_plan = named_module.state.get("auto_module_decoder") or {}
                    emit_device_telemetry(
                        "floatx_deferred_quant_source_decode",
                        module=getattr(named_module, "full_name", named_module.name),
                        device=device,
                        target_dtype=str(target_dtype).split(".")[-1],
                        source_format=(decoder_plan.get("runtime_plan") or {}).get("source_format", "auto"),
                    )

                module = self._replace_live_submodule(
                    target_submodule,
                    quant_source,
                    module_path=module_path,
                )
                if source_spec is not None:
                    from ..utils.auto_dtype_cast import release_packed_source_storage

                    if getattr(self, "_needs_w_wq_diff", False):
                        # EoRA builds ``w - wq`` from this packed source; the
                        # processor releases it once the FP32 diff exists.
                        named_module.state["auto_dtype_cast_packed_source"] = target_submodule
                    else:
                        release_packed_source_storage(target_submodule)
                named_module.state.pop("auto_dtype_cast_source_spec", None)
                if get_device(module) != device:
                    safe_module_to(module, device)
                return module

            turtle_model = self.turtle_model
            if role == "forward" and named_module is not None and isinstance(turtle_model, LazyTurtle):
                checkpoint_tensors = turtle_model.checkpoint_tensors_for_submodule(
                    target_model=self.model,
                    target_submodule=target_submodule,
                    recurse=False,
                    module_path=module_path,
                )
                weight = checkpoint_tensors.get("weight")
                if isinstance(weight, torch.Tensor):
                    decoder_kind = self._decoder_weight_format(
                        weight=weight,
                        checkpoint_tensors=checkpoint_tensors,
                        result_shape=tuple(getattr(target_submodule.weight, "shape", weight.shape)),
                        module_name=named_module.full_name,
                    )
                    if decoder_kind is not None:
                        # Packed floatx checkpoints can require decoder-specific
                        # materialization before any dense shell weight exists.
                        return self._prepare_auto_decoder_forward_module(
                            target_submodule=target_submodule,
                            device=torch.device(device),
                            named_module=named_module,
                        )

            if turtle_model is None:
                if get_device(target_submodule) != device:
                    safe_module_to(target_submodule, device)
                module = target_submodule
            else:
                module = alias_from_turtle_for_submodule(
                    target_model=self.model,
                    turtle_model=turtle_model,
                    target_submodule=target_submodule,
                    device=device,
                    module_path=module_path,
                    tie_weights=self.turtle_materialize_tie_weights,
                )

            if role == "forward" and named_module is not None:
                module = self._prepare_auto_decoder_forward_module(
                    target_submodule=module,
                    device=torch.device(device),
                    named_module=named_module,
                )
                module = self._maybe_auto_dtype_cast_forward_module(
                    module, device=torch.device(device), named_module=named_module,
                )
        return module

    def shell_direct_meta_materialize(
            self,
            target_submodule: torch.nn.Module,
            device: Optional[torch.device] = None,
            module_path: Optional[str] = None,
    ):
        if device is not None:
            device = as_torch_device(device)
        with self._turtle_lock:
            if self.turtle_model is None:
                return None
            return alias_direct_meta_from_turtle_for_submodule(
                target_model=self.model,
                turtle_model=self.turtle_model,
                target_submodule=target_submodule,
                device=device,
                module_path=module_path,
            )

    ## overrides nn.module.train()
    def train(self, mode: bool = True):
        """Switch training mode without leaving a partially-updated model.

        ``nn.Module.train`` recurses into every child, and each inference-only
        quant kernel raises from its own ``train``. By then the modules visited
        earlier are already in training mode, so a retry starts from mixed
        state. Reject the switch before touching any child.
        """
        if mode:
            model = getattr(self, "_modules", {}).get("model")
            if model is not None:
                unsupported = [
                    name
                    for name, module in model.named_modules()
                    if isinstance(module, BaseQuantLinear)
                    and not module.SUPPORTS_TRAINING
                    and not module.training
                ]
                if unsupported:
                    raise NotImplementedError(
                        f"{self.__class__.__name__}: {len(unsupported)} inference-only "
                        f"quant kernel(s) cannot switch to training mode (first: "
                        f"`{unsupported[0]}`). Reload the model with "
                        '`backend="auto_trainable"` or a trainable backend for this '
                        "quantization method before fine-tuning."
                    )
        return super().train(mode)

    @classmethod
    def _build_layer_modules_for_tree(cls, tree, include_capture_only: bool = False):
        """
        tree format:
          [<model_name>, <submodule>, "#", { parent_module: ( "child[:!][:grp]", ... ), ... }]
        Rules:
          - ':!' means participates in inference but is NOT quantized; keep this marker in output.
          - ':?' marks capture-only nodes; activations are recorded but the module is not quantized.
          - ':<digit>' means grouping; children with the same group id are emitted in the same block.
          - ':in=<tag>' declares which sibling children consume the same input tensor (see shared_input.py);
            it does not affect the emitted blocks.
          - Flags can be combined, e.g. 'module_name:!:2' or 'module_name:1:in=q'.
          - Supports nested dict structures for MoE models with experts.
          - Special key "#" in nested dicts means direct children under parent (no additional nesting).
          - EXPERT_INDEX_PLACEHOLDER in keys will be handled by simple_layer_modules for MoE expansion.
        Output:
          _layer_modules = [ [items...], [items...], ... ]
        """
        mapping = None
        for item in tree:
            if isinstance(item, dict):
                mapping = item
                break
        if mapping is None:
            raise ValueError("Mapping configuration not found in the tree.")

        out_blocks = []
        alias_groups: Dict[tuple[str | None, int], List[tuple[str, bool, bool]]] = {}
        alias_meta: Dict[tuple[str | None, int], Dict[str, int]] = {}
        alias_seq = count()
        group_seq = count()

        def _parse_token(token: str) -> tuple[str, List[str]]:
            return cls._parse_module_flags(token)

        def _group_from_flags(flags: List[str]) -> int:
            for flag in flags:
                if flag.isdigit():
                    return int(flag)
            return 0

        def _has_numeric_flag(flags: List[str]) -> bool:
            return any(flag.isdigit() for flag in flags)

        def _get_scope(parent_name: str) -> str | None:
            if not parent_name:
                return None
            return parent_name.split(".", 1)[0]

        def process_entries(parent_token: str, entries, parent_group_offset: int = 0, scope_key: str | None = None):
            """Process entries recursively to handle nested dict structures for MoE"""
            groups: defaultdict[int, List[tuple]] = defaultdict(list)

            parent_name, parent_flags = _parse_token(parent_token)
            parent_rel_group = _group_from_flags(parent_flags)
            parent_group = parent_group_offset + parent_rel_group
            parent_has_bang = "!" in parent_flags
            parent_capture_only = "?" in parent_flags
            parent_has_numeric = _has_numeric_flag(parent_flags)

            scope = scope_key if scope_key is not None else _get_scope(parent_name)
            parent_alias_scope = scope if parent_has_numeric else parent_name

            def _make_entry(full_path: str, has_bang: bool, capture_only: bool, *, alias_base: int, alias_rel: int, alias_scope: str | None) -> tuple:
                return (full_path, has_bang, capture_only, alias_scope, (alias_base, alias_rel))

            child_group_offset = parent_group_offset
            add_parent = parent_has_bang or (parent_capture_only and include_capture_only)
            if add_parent:
                alias_base = parent_rel_group if parent_has_numeric else parent_group
                parent_entry_scope = f"{parent_alias_scope}.__parent__" if parent_alias_scope is not None else None
                groups[parent_group].append(
                    _make_entry(
                        parent_name,
                        parent_has_bang,
                        parent_capture_only,
                        alias_base=alias_base,
                        alias_rel=0,
                        alias_scope=parent_entry_scope,
                    )
                )
                child_group_offset = max(child_group_offset, parent_group + 1)

            # Handle tuple/list of strings (traditional format)
            if isinstance(entries, (tuple, list)):
                for ent in entries:
                    child_name, child_flags = _parse_token(ent)

                    has_bang = "!" in child_flags
                    capture_only = "?" in child_flags
                    # first numeric tag is the group id; default 0
                    child_rel_group = _group_from_flags(child_flags)
                    grp = child_group_offset + child_rel_group
                    # Apply parent group offset to avoid conflicts between different nesting levels
                    # Store the full path including parent for later use
                    if parent_name.endswith(f".{child_name}") or parent_name == child_name:
                        full_path = parent_name
                    elif parent_name:
                        full_path = f"{parent_name}.{child_name}"
                    else:
                        full_path = child_name

                    if capture_only and not include_capture_only:
                        continue
                    alias_scope = scope if parent_has_numeric else parent_name
                    alias_base = parent_rel_group if parent_has_numeric else grp
                    alias_rel = child_rel_group if parent_has_numeric else 0
                    groups[grp].append(
                        _make_entry(
                            full_path,
                            has_bang,
                            capture_only,
                            alias_base=alias_base,
                            alias_rel=alias_rel,
                            alias_scope=alias_scope,
                        )
                    )

            elif isinstance(entries, dict):
                # Calculate max group number used at current level to avoid conflicts
                max_current_group = 0
                for sub_parent, sub_entries in entries.items():
                    if isinstance(sub_entries, (tuple, list)):
                        for ent in sub_entries:
                            _, ent_flags = _parse_token(ent)
                            max_current_group = max(max_current_group, _group_from_flags(ent_flags))

                # Process nested entries with appropriate group offset
                current_offset = child_group_offset
                for sub_parent, sub_entries in entries.items():
                    if sub_parent == "#":
                        # Special case: "#" means expert index placeholder
                        # Create a template path that will be expanded later by simple_layer_modules
                        template_parent = (
                            f"{parent_name}.{EXPERT_INDEX_PLACEHOLDER}"
                            if parent_name else EXPERT_INDEX_PLACEHOLDER
                        )
                        template_parent_token = (
                            f"{template_parent}:{parent_rel_group}"
                            if parent_has_numeric
                            else template_parent
                        )
                        # Use a higher offset for expert modules to avoid conflicts with parent level
                        expert_offset = current_offset + max_current_group + 100  # Large offset to avoid conflicts

                        # Handle special case where sub_entries is ("#",) or "#" - this means use the parent path directly
                        if (isinstance(sub_entries, (tuple, list)) and len(sub_entries) == 1 and sub_entries[0] == "#") or sub_entries == "#":
                            # For ("#",) or "#" format, use the template_parent directly with default group 0
                            alias_scope = scope if parent_has_numeric else template_parent
                            alias_base = parent_rel_group if parent_has_numeric else expert_offset
                            groups[expert_offset].append(
                                _make_entry(
                                    template_parent,
                                    False,
                                    False,
                                    alias_base=alias_base,
                                    alias_rel=0,
                                    alias_scope=alias_scope,
                                )
                            )
                        else:
                            sub_groups = process_entries(template_parent_token, sub_entries, expert_offset, scope)
                            for grp, items in sub_groups.items():
                                groups[grp].extend(items)
                    else:
                        # Nested structure: process recursively with full path
                        # Special case: empty string key means use parent path directly
                        if sub_parent == "":
                            full_sub_parent = parent_name
                        else:
                            full_sub_parent = (
                                f"{parent_name}.{sub_parent}"
                                if parent_name else sub_parent
                            )
                        sub_groups = process_entries(full_sub_parent, sub_entries, current_offset, scope)
                        for grp, items in sub_groups.items():
                            groups[grp].extend(items)
                        # Update offset for next sibling to avoid conflicts
                        if sub_groups:
                            current_offset = max(sub_groups.keys()) + 1

            return groups

        def _register_alias(order_idx: int, item: tuple[str, bool, bool, str | None, tuple[int, int]]):
            full_path, has_bang, capture_only, scope, alias_parts = item
            if capture_only and not include_capture_only:
                return
            alias_scope = scope
            alias_base, alias_rel = alias_parts
            alias_index = alias_base + alias_rel
            key = (alias_scope, alias_index)
            meta = alias_meta.get(key)
            if meta is None:
                alias_meta[key] = {"order": order_idx, "seq": next(alias_seq)}
                alias_groups[key] = [(full_path, has_bang, capture_only)]
            else:
                meta["order"] = min(meta["order"], order_idx)
                alias_groups[key].append((full_path, has_bang, capture_only))

        for parent, entries in mapping.items():
            groups = process_entries(parent, entries)

            for g in sorted(groups):
                order_idx = next(group_seq)
                items = groups[g]
                for item in items:
                    if len(item) == 3:
                        full_path, has_bang, capture_only = item
                        scope = full_path
                        alias_parts = (g, 0)
                        _register_alias(order_idx, (full_path, has_bang, capture_only, scope, alias_parts))
                    else:
                        _register_alias(order_idx, item)

        for key in sorted(alias_groups.keys(), key=lambda k: (alias_meta[k]["order"], alias_meta[k]["seq"])):
            block = []
            for full_path, has_bang, capture_only in alias_groups[key]:
                name = full_path
                if has_bang:
                    name += NOT_QUANTIZE_FLAG
                if capture_only and include_capture_only:
                    name += CAPTURE_ONLY_FLAG
                block.append(name)
            out_blocks.append(block)

        return out_blocks

    @classmethod
    def build_layer_modules(cls, tree, include_capture_only: bool = False):
        blocks = []
        seen = set()
        for tree_variant in cls._iter_module_tree_variants(tree):
            for block in cls._build_layer_modules_for_tree(tree_variant, include_capture_only=include_capture_only):
                key = tuple(block)
                # Shared blocks across tree variants should keep first-seen order.
                if key in seen:
                    continue
                seen.add(key)
                blocks.append(block)
        return blocks

    @classmethod
    def _module_tree_prefix_paths(cls) -> List[tuple[str, ...]]:
        """Return every module-tree prefix path before the layer container."""

        all_prefix_paths: List[tuple[str, ...]] = []
        for tree in cls._iter_module_tree_variants(_module_tree_for_class(cls)):
            try:
                sharp_idx = tree.index("#")
            except ValueError:
                raise ValueError("module_tree must contain '#' to separate hierarchy")

            assert sharp_idx > 0, "failed to get_base_modules"
            for path in cls._expand_module_tree_prefixes(tree):
                prefix = tuple(path.split("."))
                if prefix not in all_prefix_paths:
                    all_prefix_paths.append(prefix)
        return all_prefix_paths

    @classmethod
    def get_modules_with_direct_meta_tensors(cls, model: nn.Module):
        """
        Return module paths whose direct params/buffers are materialized outside
        normal layer/base-module traversal.
        """
        out = []
        seen = set()
        module_names = cls.modules_with_direct_meta_tensors or cls.direct_meta_modules or []
        for module_name in module_names:
            module = get_module(model, module_name)
            if isinstance(module, nn.Module) and module_name not in seen:
                seen.add(module_name)
                out.append(module_name)

        # A decoder root (or another module-tree prefix node) can own a bare
        # parameter/buffer that is not a child module, e.g. Hymba's
        # `model.memory_tokens`. Nothing in the base-module or layer traversal
        # materializes it, so a lazy shell leaves it on `meta` and the capture
        # forward fails. Detect those owners generically instead of requiring
        # every affected definition to declare them (#4888).
        for parts in cls._module_tree_prefix_paths():
            for depth in range(1, len(parts) + 1):
                module_name = ".".join(parts[:depth])
                if module_name in seen:
                    continue
                module = get_module(model, module_name)
                if not isinstance(module, nn.Module):
                    continue
                if not _module_owns_direct_meta_tensors(module):
                    continue
                seen.add(module_name)
                out.append(module_name)
        return out

    @classmethod
    def get_direct_meta_modules(cls, model: nn.Module):
        return cls.get_modules_with_direct_meta_tensors(model)

    @classmethod
    def get_base_modules(cls, model: nn.Module):
        """
        Return list of base modules directly under the root path but not the layer container.
        """
        all_prefix_paths = cls._module_tree_prefix_paths()
        if not all_prefix_paths:
            return []

        exclude_by_parent: Dict[tuple[str, ...], set[str]] = {}
        for parts in all_prefix_paths:
            for i in range(len(parts) - 1):
                parent_path = parts[: i + 1]
                # Each tree reserves the next path segment for layer traversal.
                exclude_by_parent.setdefault(parent_path, set()).add(parts[i + 1])

        out = []
        seen = set()
        for parts in all_prefix_paths:
            for i in range(len(parts) - 1):
                path = parts[: i + 1]
                base = model
                exclude = exclude_by_parent.get(path, set())

                for node in path:
                    base = getattr(base, node, None)
                    if base is None:
                        break
                if base is None:
                    continue

                for name, _ in base.named_children():
                    # print("name", base, name, exclude)
                    if name in exclude:
                        continue
                    full_name = ".".join((*path, name))
                    if full_name in seen:
                        continue
                    seen.add(full_name)
                    out.append(full_name)
        return out

    def generate_layers_modules_tree_simple(self, node):
        """
        Recursively walk a nested list/dict structure and:
          1. Drop dict entries where *all* values are ':!' or ':?' flagged.
          2. Remove ':!' / ':?' and ':<digit>' markers from strings.
        """

        # If it's a list, recurse into each element
        if isinstance(node, list):
            return [self.generate_layers_modules_tree_simple(x) for x in node]

        # If it's a dict, process each key -> value
        if isinstance(node, dict):
            new_dict = {}
            for k, v in node.items():
                clean_key = self._parse_module_flags(k)[0] if isinstance(k, str) else k
                # Expand tuple-of-strings blocks (special handling)
                if isinstance(v, (tuple, list)) and all(isinstance(x, str) for x in v):
                    # Rule 1: check if ALL entries are :!
                    if all(any(p in {"!", "?"} for p in x.split(":")[1:]) for x in v):
                        continue  # skip this parent entirely

                    # Rule 2: strip :! and :digit markers
                    cleaned = tuple(self._parse_module_flags(x)[0] for x in v)
                    new_dict[clean_key] = cleaned
                else:
                    # Recurse deeper
                    new_dict[clean_key] = self.generate_layers_modules_tree_simple(v)
            return new_dict

        # If it's a plain string (unlikely here), strip markers
        if isinstance(node, str):
            return self._parse_module_flags(node)[0]

        # For other types, return as-is
        return node

    def tied_word_embedding(self) -> bool:
        # The load-time snapshot resolves tying from the decoder config and the
        # checkpoint, so a composite wrapper's default flag cannot claim a tie
        # the checkpoint does not have (#4799).
        resolved = lm_head_is_tied(self.model, self.lm_head)
        if resolved is not None:
            return resolved
        return word_embeddings_tied(getattr(self.model, "config", None))

    # Direct children that carry the token embedding or the language head. The
    # structural `nn.Module` check is the real guard; these names only break
    # ties when a decoder parent owns more than one candidate.
    _INPUT_EMBEDDING_CHILD_NAMES = (
        "embed_tokens",
        "wte",
        "word_embeddings",
        "tok_embeddings",
        "embed_in",
    )
    _OUTPUT_HEAD_CHILD_NAMES = ("lm_head", "embed_out", "output")

    def _declared_decoder_ancestor_paths(self) -> List[str]:
        """Paths from each declared decoder up to the wrapper root, nearest first.

        ``extract_layers_node()`` names the declared ``layers`` container, and
        a decoder keeps its token embedding (and sometimes its language head)
        as a sibling of that container. Walking the parents covers wrappers
        that hoist the head above the decoder, such as the Qwen Omni
        ``thinker``.
        """
        paths = self.extract_layers_node()
        if isinstance(paths, str):
            paths = [paths]
        ancestors: List[str] = []
        for path in paths or []:
            if not isinstance(path, str) or not path:
                continue
            parts = path.split(".")
            for end in range(len(parts) - 1, -1, -1):
                candidate = ".".join(parts[:end])
                if candidate not in ancestors:
                    ancestors.append(candidate)
        return ancestors

    def _declared_embedding_endpoint(self, *, output: bool) -> Optional[nn.Module]:
        """Resolve an embedding endpoint the HF wrapper refuses to expose.

        Composite wrappers can leave ``get_input_embeddings`` /
        ``get_output_embeddings`` unimplemented while the module tree still
        declares the decoder that owns them (#4816). Resolution is structural:
        only an ``nn.Embedding`` (input) or ``nn.Linear`` (output) that is a
        direct child of a declared decoder ancestor is accepted.
        """
        names = self._OUTPUT_HEAD_CHILD_NAMES if output else self._INPUT_EMBEDDING_CHILD_NAMES
        kind = nn.Linear if output else nn.Embedding
        modules = []
        for path in self._declared_decoder_ancestor_paths():
            module = self.model if not path else get_module_by_name_prefix(self.model, path)[0]
            if module is not None and not any(module is seen for seen in modules):
                modules.append(module)

        for module in modules:
            for name in names:
                child = getattr(module, name, None)
                if isinstance(child, kind):
                    return child

        if output:
            return None
        # A decoder whose embedding is spelled differently is still unambiguous
        # when its parent owns exactly one `nn.Embedding`.
        for module in modules:
            candidates = [child for child in module.children() if isinstance(child, nn.Embedding)]
            if len(candidates) == 1:
                return candidates[0]
        return None

    def get_input_embeddings(self) -> Optional[nn.Module]:
        getter = getattr(self.model, "get_input_embeddings", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except NotImplementedError:
            # Some composite wrappers never implement the getter; resolve the
            # declared endpoint instead of aborting the quantization run.
            resolved = self._declared_embedding_endpoint(output=False)
            if resolved is None:
                raise
            return resolved

    def get_input_embeddings_name(self) -> Optional[str]:
        module = self.get_input_embeddings()
        return get_module_name(self.model, module) if module is not None else None

    def get_output_embeddings(self) -> Optional[nn.Module]:
        getter = getattr(self.model, "get_output_embeddings", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except NotImplementedError:
            resolved = self._declared_embedding_endpoint(output=True)
            if resolved is None:
                raise
            return resolved

    def get_output_embeddings_name(self) -> Optional[str]:
        module = self.get_output_embeddings()
        return get_module_name(self.model, module) if module is not None else None

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception as exc:  # torch Modules raise AttributeError here
            model = self.__dict__.get("model")
            if model is None:
                # ``self._modules`` must be read from ``__dict__``: during
                # ``__init__`` (and for instances built with ``object.__new__``
                # in tests) it does not exist yet, and ``hasattr(self,
                # "_modules")`` re-enters this hook forever instead of
                # returning False.
                modules = self.__dict__.get("_modules")
                model = modules.get("model") if modules is not None else None
            if model is not None and item != "model":
                return getattr(model, item)
            raise exc

    @classmethod
    def _auto_detect_module_tree(cls, model: PreTrainedModel, quant_method: METHOD):
        """Build a module tree from a live model that has no registered definition.

        A model can expose more than one decoder stack as concurrent siblings
        (for example ``model.layers`` for text and ``vision_tower.layers`` for
        vision).  The declaration format already represents that shape as one
        complete tree variant per stack, so discovery walks the whole module
        tree and returns every stack it can resolve, not just the first match.

        Discovery is deliberately structural: a candidate must be a sequence of
        modules whose first element owns ``Linear``/``Conv`` leaves and either
        carries a known decoder-stack name or looks like a transformer block.
        The returned tree uses the full dotted path for every branch, which
        keeps layer names, ``dynamic`` patterns and per-module overrides
        unambiguous across branches.
        """

        log.warn("Model not yet support, attempting Module Tree AutoCompat...")

        if quant_method not in {
            METHOD.GPTQ,
            METHOD.GGUF,
            METHOD.FP8,
            METHOD.BITSANDBYTES,
            METHOD.EXL3,
            METHOD.PARO,
            METHOD.VPTQ,
            METHOD.MACAQ,
            METHOD.NF12,
            METHOD.CPTC,
        }:
            log.warn(
                f"Module Tree AutoCompat: Failed, quant_method={quant_method}, "
                "only support GPTQ/GGUF/FP8/BITSANDBYTES/EXL3/PARO/VPTQ/MACAQ/NF12/CPTC"
            )
            return None

        # A stack keeps its historical priority when a model exposes more than
        # one candidate path.  Paths outside this list follow discovery order.
        legacy_candidates = [
            "model.layers",
            "language_model.layers",
            "model.decoder.layers",
            "transformer.h",
            "transformer.blocks",
            "layers",
            "blocks",
            "model.blocks",
        ]
        legacy_order = {path: index for index, path in enumerate(legacy_candidates)}
        stack_names = frozenset({"layers", "h", "blocks", "block"})
        transformer_hints = (
            "attn",
            "attention",
            "self_attn",
            "mlp",
            "ffn",
            "feed",
            "dense",
            "mixer",
        )
        leaf_types = [nn.Linear, nn.Conv1d, nn.Conv2d]

        def _is_sequence(module) -> bool:
            return (
                isinstance(module, (nn.ModuleList, list, tuple))
                and len(module) > 0
                and isinstance(module[0], nn.Module)
            )

        def _looks_like_transformer_layer(layer: nn.Module) -> bool:
            for child_name, _ in layer.named_children():
                lowered = child_name.lower()
                if any(hint in lowered for hint in transformer_hints):
                    return True
            return False

        def _linear_names(module: nn.Module):
            return list(find_modules(module, layers=leaf_types).keys())

        def _named_children(module):
            if isinstance(module, nn.Module):
                return module.named_children()
            # Diagnostic callers may pass a lightweight stand-in object that
            # exposes torch modules as attributes instead of subclassing
            # nn.Module.  Keep that shape usable without changing real models.
            return (
                (name, value)
                for name, value in vars(module).items()
                if isinstance(value, nn.Module)
            )

        branches: List[tuple[str, nn.Module]] = []
        seen_ids = set()

        def _walk(module: nn.Module, path: str) -> None:
            for name, child in _named_children(module):
                child_path = f"{path}.{name}" if path else name
                if _is_sequence(child):
                    layer0 = child[0]
                    known_stack = name in stack_names
                    transformer_like = _looks_like_transformer_layer(layer0)
                    if known_stack or transformer_like:
                        if not _linear_names(layer0):
                            log.warn(
                                f"Module Tree AutoCompat: Skipping '{child_path}': "
                                f"{type(layer0).__name__} has no Linear/Conv leaves."
                            )
                        elif id(child) not in seen_ids:
                            seen_ids.add(id(child))
                            branches.append((child_path, child))
                            log.warn(
                                f"Module Tree AutoCompat: Matched branch '{child_path}', "
                                f"type={type(child).__name__}"
                            )
                        # A decoder stack owns its blocks.  Never descend into it
                        # and mistake a nested ModuleList (for example experts)
                        # for another concurrent decoder branch.
                        continue
                _walk(child, child_path)

        _walk(model, "")

        if not branches:
            log.warn("Module Tree AutoCompat: No decoder layer container found, return None")
            return None

        # Stable sort: historical candidates first in their declared order, then
        # the remaining stacks in module-registration order.
        branches.sort(key=lambda item: legacy_order.get(item[0], len(legacy_order)))

        possible_parent = ["attn", "attention", "self_attn", "mlp", "ffn", "feed", "dense"]
        variants = []
        for branch_path, container in branches:
            layer0 = container[0]
            all_linear = _linear_names(layer0)
            if not all_linear:
                raise ValueError(
                    f"Module Tree AutoCompat cannot resolve branch '{branch_path}': "
                    f"{type(layer0).__name__} exposes no Linear/Conv leaves. "
                    "Register an explicit model definition or pass an explicit module tree."
                )

            mapping = {}
            for name, _ in layer0.named_children():
                lowered = name.lower()
                if not any(hint in lowered for hint in possible_parent):
                    continue
                leaves = tuple(
                    path.split(".")[-1]
                    for path in all_linear
                    if path.startswith(f"{name}.")
                )
                if leaves:
                    mapping[name] = leaves

            if not mapping:
                # Preserve the historical fallback, but keep the full relative
                # path so nested leaves remain addressable.
                mapping[""] = tuple(all_linear)
                log.warn(
                    f"Module Tree AutoCompat: '{branch_path}' has no recognized "
                    f"parent, using all Linear/Conv leaves: {mapping['']}"
                )

            variant = branch_path.split(".") + ["#", mapping]
            variants.append(variant)
            log.warn(f"Module Tree AutoCompat: Resolved branch '{branch_path}' -> {variant}")

        if len(variants) == 1:
            tree = variants[0]
            log.warn(f"Module Tree AutoCompat: Final module_tree: {tree}")
            return tree

        log.warn(
            f"Module Tree AutoCompat: Final module_tree has {len(variants)} concurrent branches: "
            f"{variants}"
        )
        return variants

__all__ = ["BaseQModel"]

BaseQModel = ModelLoader(ModelWriter(BaseQModel))
