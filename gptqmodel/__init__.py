# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import atexit
import importlib.util
import os
import sys
import threading


# isort: off
# Must be the first check in the package: it refuses an affected PyTorch before
# the allocator is initialized, a device is opened, or a checkpoint is read.
# ``gptqmodel._torch_safety`` is standalone on purpose; importing
# ``gptqmodel.utils`` here would initialize CUDA first.
# `UnsafeTorchVersionError` is re-exported so callers can catch the gate's error
# as `gptqmodel.UnsafeTorchVersionError`.
from ._torch_safety import UnsafeTorchVersionError, ensure_safe_torch  # noqa: E402, F401

ensure_safe_torch()

# Must run before any submodule import: importing `.utils` initialises the CUDA
# caching allocator, and Torch only reads the allocator configuration once. A
# late assignment is silently ignored, which leaves large quantization runs
# fragmenting a device until a single multi-GiB request can no longer be served.
from ._alloc_conf import apply_default_alloc_conf  # noqa: E402

_installed_alloc_conf = apply_default_alloc_conf()

# Still before any other submodule import: rocBLAS forwards its GEMMs to
# hipBLASLt on ROCm 10.1/gfx1201, and hipBLASLt picks a different kernel per
# process for the same shape. That makes two identical quantization runs in
# separate processes save different checkpoints. ``ROCBLAS_USE_HIPBLASLT=0``
# keeps the GEMMs in rocBLAS and is deterministic; the user's explicit value,
# if any, wins. rocBLAS reads it on first use, so it has to be set before the
# first BLAS call.
from ._rocm_env import apply_default_rocm_blas_env  # noqa: E402

_installed_rocm_blas_env = apply_default_rocm_blas_env()

# Still before any other Torch use: HIP reports no last-level cache, so on
# RDNA3/RDNA4 Torch exposes only L2 (8 MiB on an R9700) and never the 64 MiB
# Infinity Cache / MALL behind it. Anything that sizes a cache-flush buffer
# from L2 then measures a warm cache. This back-ports the Python half of
# https://github.com/pytorch/pytorch/pull/203965 and has to run before a device
# property is read or an Inductor benchmarker caches its size hint.
from .rocm_llc import apply as _apply_rocm_llc  # noqa: E402

_rocm_llc_status = _apply_rocm_llc()

from ._banner import get_startup_banner  # noqa: E402
from .utils import _MONKEY_PATCH_LOCK  # noqa: E402
from .utils.logger import setup_logger  # noqa: E402
from .utils.nogil_patcher import TritonPatch, patch_safetensors_save_file  # noqa: E402
from .utils.rocm import warn_unsupported_rocm  # noqa: E402

# isort: on


log = setup_logger()


# ``#4364`` replaced every path-selecting environment variable with a typed
# policy field. The old variables are intentionally not read anymore, but
# silently ignoring one changes numerics or thread counts without telling the
# caller. Warn once per retired variable and name its replacement.
_RETIRED_KERNEL_ENV_VARS = {
    "APHRODITE_SWORDFISH_DENSE_M": "KernelConfig.swordfish_dense_m",
    "APHRODITE_SWORDFISH_DETERMINISTIC": "KernelConfig.swordfish_deterministic",
    "GPTQMODEL_AWQ_DISABLE_FUSED_SPLITK_REDUCE": "KernelConfig.awq_fused_splitk_reduce",
    "GPTQMODEL_AWQ_TRITON_SCHEDULE": "KernelConfig.awq_triton_schedule",
    "GPTQMODEL_BITSANDBYTES_CPU_INT8_DENSE": "KernelConfig.bitsandbytes_cpu_int8_dense",
    "GPTQMODEL_BITSANDBYTES_CPU_PACKED": "KernelConfig.bitsandbytes_cpu_packed",
    "GPTQMODEL_BLOCKFLOAT_NO_CUBLAS": "KernelConfig.blockfloat_cublas",
    "GPTQMODEL_BNB_TRITON": "KernelConfig.bitsandbytes_triton",
    "GPTQMODEL_CPU_BF16X3": "KernelConfig.cpu_bf16x3_matmul",
    "GPTQMODEL_CPU_WORKERS": "RuntimeConfig.cpu_workers",
    "GPTQMODEL_CUDA_WORKERS": "RuntimeConfig.cuda_workers",
    "GPTQMODEL_DISABLE_BITBLAS": "KernelConfig.disable_bitblas",
    "GPTQMODEL_DISABLE_GPTQ_CUDA": "KernelConfig.gptq_fused_block_update (set it to False)",
    "GPTQMODEL_EXL3_CAPTURE_BATCH_CHECKPOINT_INTERVAL": "KernelConfig.exl3_capture_batch_checkpoint_interval",
    "GPTQMODEL_EXL3_CAPTURE_BATCH_SPOOL": "KernelConfig.exl3_capture_batch_spool",
    "GPTQMODEL_EXL3_CAPTURE_FRONTIER": "KernelConfig.exl3_capture_frontier",
    "GPTQMODEL_EXL3_CPU_BF16X3": "KernelConfig.cpu_bf16x3_matmul",
    "GPTQMODEL_EXL3_CUDA_ALLOCATION_LIMIT_BYTES": "KernelConfig.exl3_cuda_allocation_limit_bytes",
    "GPTQMODEL_EXL3_HOST_RSS_LIMIT_BYTES": "KernelConfig.exl3_host_rss_limit_bytes",
    "GPTQMODEL_EXL3_MEMORY_TELEMETRY_INTERVAL_BATCHES": "KernelConfig.exl3_memory_telemetry_interval_batches",
    "GPTQMODEL_EXL3_TILE_BATCH": "KernelConfig.exl3_tile_batch",
    "GPTQMODEL_FLOATX_CPU_DISABLE_AVX2": "KernelConfig.floatx_cpu_simd_level",
    "GPTQMODEL_FLOATX_CPU_DISABLE_AVX512": "KernelConfig.floatx_cpu_simd_level",
    "GPTQMODEL_FLOATX_CPU_FORCE_NATIVE_FP8": "KernelConfig.floatx_cpu_native_fp8",
    "GPTQMODEL_FLOATX_CPU_THREADS": "KernelConfig.floatx_cpu_threads",
    "GPTQMODEL_FP32_ACCUM": "KernelConfig.fp32_accum",
    "GPTQMODEL_GGUF_CPP_NATIVE_MATMUL": "KernelConfig.gguf_cpp_native_matmul",
    "GPTQMODEL_GGUF_CPP_THREADS": "KernelConfig.gguf_cpp_threads",
    "GPTQMODEL_GGUF_FUSED_AUTOTUNE_ITERS": "removed; GGUF fused dispatch is static",
    "GPTQMODEL_GGUF_FUSED_AUTOTUNE_MARGIN": "removed; GGUF fused dispatch is static",
    "GPTQMODEL_GGUF_FUSED_AUTOTUNE_WARMUP": "removed; GGUF fused dispatch is static",
    "GPTQMODEL_GGUF_FUSED_CHUNK_BLOCKS": "KernelConfig.gguf_fused_chunk_blocks",
    "GPTQMODEL_GGUF_FUSED_CHUNK_BYTES": "KernelConfig.gguf_fused_chunk_bytes",
    "GPTQMODEL_GGUF_FUSED_CPU_MAX_ROWS": "KernelConfig.gguf_fused_cpu_max_rows",
    "GPTQMODEL_GGUF_FUSED_CPU_MIN_MATRIX_ELEMENTS": "KernelConfig.gguf_fused_cpu_min_matrix_elements",
    "GPTQMODEL_GGUF_FUSED_CUDA_MAX_ROWS": "KernelConfig.gguf_fused_cuda_max_rows",
    "GPTQMODEL_GGUF_FUSED_CUDA_MIN_MATRIX_ELEMENTS": "KernelConfig.gguf_fused_cuda_min_matrix_elements",
    "GPTQMODEL_GSQ_DISABLE_CUDA_RELAXATION": "KernelConfig.gsq_disable_cuda_relaxation",
    "GPTQMODEL_GSQ_DISABLE_HIP_RELAXATION": "KernelConfig.gsq_disable_hip_relaxation",
    "GPTQMODEL_INTERNAL_GGUF_DEQUANT_DEVICE": "KernelConfig.gguf_dequant_device",
    "GPTQMODEL_INTERNAL_GGUF_DEQUANT_MAX_BYTES": "KernelConfig.gguf_dequant_chunk_bytes",
    "GPTQMODEL_INTERNAL_GGUF_QUANTIZED_LOADER": "KernelConfig.gguf_native_quantized_loader",
    "GPTQMODEL_INTERNAL_GGUF_TORCH_LOADER": "KernelConfig.gguf_torch_dequant_loader",
    "GPTQMODEL_MARLIN_USE_FP32": "KernelConfig.marlin_fp32",
    "GPTQMODEL_MAX_PACKERS": "RuntimeConfig.max_packers",
    "GPTQMODEL_NO_ROCM_LLC": "KernelConfig.rocm_llc",
    "GPTQMODEL_NPU_CHOLESKY_BLOCK_SIZE": "KernelConfig.npu_cholesky_block_size",
    "GPTQMODEL_PARO_AFFINE_STE_TRITON": "KernelConfig.paro_affine_ste_triton",
    "GPTQMODEL_PARO_AFFINE_STE_TRITON_BACKWARD": "KernelConfig.paro_affine_ste_triton_backward",
    "GPTQMODEL_PARO_AUTOTUNE_CACHE": "KernelConfig.paro_autotune_cache",
    "GPTQMODEL_PARO_AUTO_CACHE_BF16_ROTATION_DTYPE": "KernelConfig.paro_auto_cache_bf16_rotation_dtype",
    "GPTQMODEL_PARO_AUTO_CACHE_BF16_RUNTIME_DTYPE": "KernelConfig.paro_auto_cache_bf16_runtime_dtype",
    "GPTQMODEL_PARO_CACHE_ROTATION_DTYPE": "KernelConfig.paro_cache_rotation_dtype",
    "GPTQMODEL_PARO_CACHE_RUNTIME_DTYPE": "KernelConfig.paro_cache_runtime_dtype",
    "GPTQMODEL_PARO_CPU_AVX512": "KernelConfig.paro_cpu_avx512",
    "GPTQMODEL_PARO_OPT_FUSED_ROTATION": "QuantizeConfig.opt_fused_rotation",
    "GPTQMODEL_PARO_ROTATE_AUTOTUNE": "KernelConfig.paro_rotate_autotune",
    "GPTQMODEL_PARO_ROTATE_CTA_M": "KernelConfig.paro_rotate_cta_m",
    "GPTQMODEL_PARO_ROTATE_ROW_PAD": "KernelConfig.paro_rotate_row_pad",
    "GPTQMODEL_PARO_SYM_STE_TRITON": "KernelConfig.paro_sym_ste_triton",
    "GPTQMODEL_PARO_TRITON_AUTOTUNE": "KernelConfig.paro_triton_autotune",
    "GPTQMODEL_PARO_TRITON_AUTOTUNE_ITERS": "KernelConfig.paro_triton_autotune_iters",
    "GPTQMODEL_PARO_TRITON_AUTOTUNE_MARGIN": "KernelConfig.paro_triton_autotune_margin",
    "GPTQMODEL_PARO_TRITON_AUTOTUNE_WARMUP": "KernelConfig.paro_triton_autotune_warmup",
    "GPTQMODEL_PARO_TRITON_DECODE_MAX_ROWS": "KernelConfig.paro_triton_decode_max_rows",
    "GPTQMODEL_RESUME": "CheckpointConfig(path=...)",
    "GPTQMODEL_ROCM_QUANT_BLAS": "KernelConfig.rocm_quant_blas",
    "GPTQ_TORCH_CACHE_WEIGHTS": "KernelConfig.torch_cache_weights",
    "GPTQ_TORCH_LOOKAHEAD": "KernelConfig.torch_lookahead",
    "GPTQ_TORCH_ROCM_GEMM": "KernelConfig.torch_rocm_gemm",
    "GPTQ_TORCH_ROCM_GEMV": "KernelConfig.torch_rocm_gemv",
    "GPTQ_TORCH_STREAMING": "KernelConfig.torch_streaming",
    "GPTQ_TORCH_STREAM_TILE": "KernelConfig.torch_stream_tile",
    "GPTQ_TORCH_TRITON_DEQUANT": "KernelConfig.torch_triton_dequant",
}
_WARNED_RETIRED_KERNEL_ENV_VARS: set[str] = set()


def warn_retired_kernel_env_vars() -> list[str]:
    """Warn once for every retired kernel/thread env var that is still set."""

    warned = []
    for name, replacement in _RETIRED_KERNEL_ENV_VARS.items():
        if name not in os.environ or name in _WARNED_RETIRED_KERNEL_ENV_VARS:
            continue
        _WARNED_RETIRED_KERNEL_ENV_VARS.add(name)
        log.warn(
            f"ENV: {name} is retired and has no effect; use {replacement} instead."
        )
        warned.append(name)
    return warned


warn_retired_kernel_env_vars()

if any(result.startswith("applied") for result in _rocm_llc_status.values()):
    log.debug("rocm_llc: %s", _rocm_llc_status)

if _installed_alloc_conf is not None:
    log.info(
        "ENV: Auto setting PYTORCH_ALLOC_CONF='%s' for memory saving.",
        _installed_alloc_conf,
    )

if _installed_rocm_blas_env is not None:
    log.info(
        "ENV: Auto setting ROCBLAS_USE_HIPBLASLT='%s' for reproducible quantization.",
        _installed_rocm_blas_env,
    )

# Fail loudly (once) on ROCm releases older than the supported floor.
warn_unsupported_rocm()

patch_safetensors_save_file()

# TODO: waiting for official fix from triton
#  monkeypatching triton threading issues is fragile
try:
    TritonPatch.apply()
except Exception:
    log.warning("TritonPatch.apply() failed; triton autotuner threading fix is NOT active", exc_info=True)


def _patch_transformers_gptq_device_map_compat():
    """Preserve concrete single-device GPTQ maps for Optimum's later packing step."""
    try:
        from functools import wraps

        from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
    except ImportError:
        # transformers is a hard dependency, so a missing symbol means the
        # upstream API moved and this compatibility shim is now inert.
        log.warning(
            "transformers GPTQ device-map compat is inactive: GptqHfQuantizer API changed",
            exc_info=True,
        )
        return
    except Exception:
        log.warning("transformers GPTQ device-map compat probe failed", exc_info=True)
        return

    with _MONKEY_PATCH_LOCK:
        original_process = GptqHfQuantizer._process_model_before_weight_loading
        if getattr(original_process, "_gptqmodel_device_map_compat", False):
            return

        @wraps(original_process)
        def _process_model_before_weight_loading_with_device_map(self, model, **kwargs):
            """Backfill `hf_device_map` when GPTQ uses a single concrete device."""
            device_map = kwargs.get("device_map")
            if (
                isinstance(device_map, dict)
                and device_map
                and len(set(device_map.values())) == 1
                and not hasattr(model, "hf_device_map")
            ):
                model.hf_device_map = dict(device_map)
            return original_process(self, model, **kwargs)

        _process_model_before_weight_loading_with_device_map._gptqmodel_device_map_compat = True
        GptqHfQuantizer._process_model_before_weight_loading = _process_model_before_weight_loading_with_device_map


def _patch_transformers_gptqmodel_gguf_quantizer_compat():
    """Make transformers refuse GPT-QModel packed GGUF checkpoints loudly.

    `transformers` reserves `quant_method: "gguf"` for its own
    `GgufHfQuantizer`, which expects a `.gguf` file. GPT-QModel publishes
    `gptqmodel_gguf`; registering a refusing quantizer for that name turns an
    otherwise silent random-weight load into an actionable error.
    """
    try:
        from transformers.quantizers import auto as hf_quant_auto
    except ImportError:
        log.debug("GPT-QModel GGUF quantizer guard inactive: transformers quantizer API changed", exc_info=True)
        return

    from ._quant_methods import GGUF_EXPORT_QUANT_METHOD
    from .utils.hf_gguf_quantizer import (
        GPTQModelGGUFHfQuantizer,
        GPTQModelGGUFQuantizationConfig,
    )

    with _MONKEY_PATCH_LOCK:
        hf_quant_auto.AUTO_QUANTIZATION_CONFIG_MAPPING.setdefault(
            GGUF_EXPORT_QUANT_METHOD, GPTQModelGGUFQuantizationConfig
        )
        hf_quant_auto.AUTO_QUANTIZER_MAPPING.setdefault(
            GGUF_EXPORT_QUANT_METHOD, GPTQModelGGUFHfQuantizer
        )


def _patch_transformers_refusing_quantizers():
    """Make transformers refuse every GPT-QModel-only packed checkpoint format.

    Without a registered quantizer, transformers 5.x only warns about the
    unknown `quant_method` and returns a plain model with randomly initialized
    packed weights. The names live in one table so parallel method fixes do not
    race on this registration.
    """
    try:
        from transformers.quantizers import auto as hf_quant_auto
    except ImportError:
        log.debug("GPT-QModel refusing quantizers inactive: transformers quantizer API changed", exc_info=True)
        return

    from ._quant_methods import REFUSED_HF_QUANT_METHODS
    from .utils.hf_refusing_quantizer import (
        GPTQModelRefusingHfQuantizer,
        GPTQModelRefusingQuantizationConfig,
        refusal_message,
    )

    with _MONKEY_PATCH_LOCK:
        for method in REFUSED_HF_QUANT_METHODS:
            hf_quant_auto.AUTO_QUANTIZATION_CONFIG_MAPPING.setdefault(
                method, GPTQModelRefusingQuantizationConfig
            )
            hf_quant_auto.AUTO_QUANTIZER_MAPPING.setdefault(
                method, GPTQModelRefusingHfQuantizer
            )

        # transformers special-cases `quant_method == "bitsandbytes"` in
        # `AutoHfQuantizer.from_config`: it demands an instance of its own
        # `BitsAndBytesConfig` and only then appends the `_4bit`/`_8bit`
        # suffix. That branch runs before the mapping lookup, so the refusing
        # quantizer registered above is never constructed and GPT-QModel's
        # layout raises a generic TypeError instead of the actionable error
        # (#4664). Refuse only GPT-QModel's layout: transformers' own
        # `BitsAndBytesConfig` carries the same plain `quant_method`, so
        # refusing every occurrence broke all genuine HF bnb loads (QLoRA, any
        # Hub bnb checkpoint) in a process that imported GPT-QModel (#4731).
        original_from_config = hf_quant_auto.AutoHfQuantizer.__dict__["from_config"]
        if not getattr(original_from_config, "_gptqmodel_refusal_guard", False):

            @classmethod
            def _refusing_from_config(cls, quantization_config, **kwargs):
                if _is_gptqmodel_bitsandbytes_layout(quantization_config):
                    raise ValueError(refusal_message("bitsandbytes"))
                return original_from_config.__func__(cls, quantization_config, **kwargs)

            _refusing_from_config._gptqmodel_refusal_guard = True
            hf_quant_auto.AutoHfQuantizer.from_config = _refusing_from_config


def _is_gptqmodel_bitsandbytes_layout(quantization_config) -> bool:
    """Whether a ``bitsandbytes`` config is GPT-QModel's layout, not HF's own.

    transformers' ``BitsAndBytesConfig`` and GPT-QModel's checkpoint layout
    share ``quant_method == "bitsandbytes"``. Only the GPT-QModel layout must
    be refused: HF's config is loadable and must keep working (#4731).

    A dict is GPT-QModel's layout only when it carries neither ``load_in_4bit``
    nor ``load_in_8bit``; ``AutoQuantizationConfig.from_dict`` maps a dict with
    either flag to a real ``BitsAndBytesConfig`` through the
    ``bitsandbytes_4bit``/``bitsandbytes_8bit`` names. An already-built object
    is GPT-QModel's layout when it is not a real ``BitsAndBytesConfig``.
    """

    if isinstance(quantization_config, dict):
        if quantization_config.get("quant_method") != "bitsandbytes":
            return False
        return not (
            quantization_config.get("load_in_4bit", False)
            or quantization_config.get("load_in_8bit", False)
        )
    if getattr(quantization_config, "quant_method", None) != "bitsandbytes":
        return False
    from transformers import BitsAndBytesConfig

    return not isinstance(quantization_config, BitsAndBytesConfig)


def _patch_transformers_save_pretrained_runtime_layout_guard():
    """Refuse an HF checkpoint write while a kernel holds a runtime layout.

    Marlin, AWQ-Marlin, ExLlamaV2 and AWQ-ExLlamaV2 permute or shuffle their
    packed tensors in place during ``post_init()`` and have no inverse.
    ``PreTrainedModel.save_pretrained()`` collects ``state_dict()`` directly,
    so without this guard it publishes that private layout under an ordinary
    GPTQ/AWQ config; the resulting directory then reloads as garbage (or with a
    shape error) instead of failing at save.

    The kernels refuse from ``_save_to_state_dict`` only inside
    ``checkpoint_serialization()``, because PEFT calls ``state_dict()`` merely
    to read tensors and must keep working. Wrapping the Transformers writer puts
    every checkpoint write on the refusing side while leaving those read-only
    callers alone. ``GPTQModel.save()``/``save_quantized()`` build a separate
    export shell, so they never serialize a runtime layout and are unaffected.
    """

    try:
        from transformers import PreTrainedModel
    except ImportError:
        log.debug("GPT-QModel save_pretrained guard inactive: transformers API changed", exc_info=True)
        return

    from functools import wraps

    from .utils.checkpoint_state import checkpoint_serialization

    with _MONKEY_PATCH_LOCK:
        original = PreTrainedModel.save_pretrained
        if getattr(original, "_gptqmodel_runtime_layout_guard", False):
            return

        @wraps(original)
        def save_pretrained_with_runtime_layout_guard(self, *args, **kwargs):
            with checkpoint_serialization():
                return original(self, *args, **kwargs)

        save_pretrained_with_runtime_layout_guard._gptqmodel_runtime_layout_guard = True
        PreTrainedModel.save_pretrained = save_pretrained_with_runtime_layout_guard


def _patch_openvino_gptqmodel_compat():
    """Extend OpenVINO's GPTQ patcher to understand GPTQModel new kernels."""
    try:
        from openvino.frontend.pytorch import gptq as ov_gptq
    except ImportError:
        log.debug("OpenVINO GPTQ compat skipped: openvino not installed", exc_info=True)
        return
    except Exception:
        log.warning("OpenVINO GPTQ compat probe failed", exc_info=True)
        return

    with _MONKEY_PATCH_LOCK:
        if getattr(ov_gptq, "_gptqmodel_torch_quant_compat", False):
            return

        class MatchAll(list):
            def __contains__(self, item):
                return True

        ov_gptq.supported_quant_types = MatchAll()
        ov_gptq._gptqmodel_torch_quant_compat = True


def _patch_transformers_causal_conv1d_hub_kernel_compat():
    """Prioritize local kernel ports before falling back to Transformers hub kernels."""
    try:
        import importlib
        from pathlib import Path

        import transformers.integrations as hf_integrations
        from transformers.integrations import hub_kernels
    except ImportError:
        log.warning(
            "causal-conv1d hub-kernel compat is inactive: transformers integrations API changed",
            exc_info=True,
        )
        return
    except Exception:
        log.warning("causal-conv1d hub-kernel compat probe failed", exc_info=True)
        return

    with _MONKEY_PATCH_LOCK:
        if not hasattr(hf_integrations, "lazy_load_kernel"):
            return

        if getattr(hub_kernels, "_gptqmodel_local_causal_conv1d_kernel", False):
            hf_integrations.lazy_load_kernel = hub_kernels.lazy_load_kernel
            return

        original_lazy_load_kernel = hub_kernels.lazy_load_kernel
        project_root = Path(__file__).resolve().parent.parent

        def _resolve_local_kernel_module(kernel_name):
            module_name = kernel_name.replace("-", "_")
            namespaced_module = f"gptqmodel.hf_kernels.{module_name}"
            if (project_root / "gptqmodel" / "hf_kernels" / module_name / "__init__.py").exists() or (
                project_root / "gptqmodel" / "hf_kernels" / f"{module_name}.py"
            ).exists():
                return importlib.import_module(namespaced_module)
            package_path = project_root / module_name / "__init__.py"
            module_path = project_root / f"{module_name}.py"
            if not package_path.exists() and not module_path.exists():
                return None
            return importlib.import_module(module_name)

        def _lazy_load_kernel_with_local_causal_conv1d(kernel_name, mapping=hub_kernels._KERNEL_MODULE_MAPPING):
            local_kernel = _resolve_local_kernel_module(kernel_name)
            if local_kernel is not None:
                mapping[kernel_name] = local_kernel
                return local_kernel
            return original_lazy_load_kernel(kernel_name, mapping)

        hub_kernels.lazy_load_kernel = _lazy_load_kernel_with_local_causal_conv1d
        local_causal_conv1d = _resolve_local_kernel_module("causal-conv1d")
        if local_causal_conv1d is not None:
            hub_kernels._KERNEL_MODULE_MAPPING["causal-conv1d"] = local_causal_conv1d
            # Keep legacy imports working after moving the local port under gptqmodel.hf_kernels.
            sys.modules.setdefault("causal_conv1d", local_causal_conv1d)
        if hasattr(hf_integrations, "lazy_load_kernel"):
            hf_integrations.lazy_load_kernel = _lazy_load_kernel_with_local_causal_conv1d
        hub_kernels._gptqmodel_local_causal_conv1d_kernel = True


def _hf_quantizer_backend(quantizer):
    """Return the caller's backend from an HF quantizer, wherever it lives.

    Optimum's ``GPTQQuantizer`` stores it on the quantizer itself
    (``.backend``). transformers' ``AwqQuantizer`` is a plain ``HfQuantizer``
    with no such attribute: the choice lives on
    ``quantization_config.backend``. Reading only ``.backend`` therefore
    dropped every explicit AWQ backend and resolved AUTO (#4729). Prefer the
    quantizer's own value, then the config's, so both bridges forward the
    caller's choice and an impossible one (e.g. marlin on CPU) still raises.
    """

    backend = getattr(quantizer, "backend", None)
    if backend is not None:
        return backend
    return getattr(getattr(quantizer, "quantization_config", None), "backend", None)


# Optimum's GPTQ bridge install state. ``optimum.gptq`` imports ``gptqmodel``
# while it is still executing, so the package-init probe can observe a
# partially initialized module; that is a *deferred* failure, never "Optimum is
# not installed". These flags keep the retry cheap and idempotent (#4749).
_OPTIMUM_GPTQ_BRIDGE_INSTALLED = False
_OPTIMUM_GPTQ_BRIDGE_UNAVAILABLE = False

# Transformers' AWQ quantizer install state. Its first ``gptqmodel`` import
# happens inside ``replace_with_awq_linear`` (after its prepare hook would have
# run), so the bridge must be retried from later entry points (#4795).
_AWQ_BRIDGE_INSTALLED = False
_AWQ_BRIDGE_UNAVAILABLE = False


def _patch_optimum_gptqmodel_load_bridge():
    """Route Optimum's GPTQ loading through GPTQModel's native bridge.

    Transformers delegates pre-quantized GPTQ loading to Optimum, which
    replaces every ``nn.Linear`` with one kernel chosen from the *global*
    quantization config. GPTQModel checkpoints can carry per-module ``dynamic``
    settings and structural rewrites (expanded MoE experts, fused blocks), so
    Optimum's homogeneous replacement silently drops every per-expert tensor.
    Wrapping ``GPTQQuantizer.convert_model`` / ``post_init_model`` lets
    ``hf_gptqmodel_prepare_model_for_load`` rebuild the checkpoint's real module
    manifest first; checkpoints Optimum owns keep Optimum's own path because the
    bridge returns ``None`` for them.
    """

    global _OPTIMUM_GPTQ_BRIDGE_INSTALLED

    try:
        from functools import wraps

        from optimum.gptq import GPTQQuantizer
    except ImportError:
        # Either Optimum is not installed (optional dependency of Transformers'
        # GPTQ loader) or ``optimum.gptq`` is importing ``gptqmodel`` right now
        # and has not defined ``GPTQQuantizer`` yet. The caller decides which;
        # ``_ensure_optimum_gptqmodel_load_bridge`` retries the deferred case.
        return False
    except Exception:
        log.warning("Optimum GPTQ load-bridge compat probe failed", exc_info=True)
        return False

    with _MONKEY_PATCH_LOCK:
        original_convert_model = GPTQQuantizer.convert_model
        if getattr(original_convert_model, "_gptqmodel_load_bridge", False):
            _OPTIMUM_GPTQ_BRIDGE_INSTALLED = True
            return True
        original_post_init_model = GPTQQuantizer.post_init_model

        @wraps(original_convert_model)
        def _convert_model_with_gptqmodel_bridge(self, model, **kwargs):
            from .utils.hf_load import hf_gptqmodel_prepare_model_for_load

            context = hf_gptqmodel_prepare_model_for_load(
                model,
                checkpoint_files=kwargs.get("checkpoint_files"),
                device_map=kwargs.get("device_map"),
                # Forward the caller's `quantization_config["backend"]`. Without
                # it `_resolve_backend_for_checkpoint` always sees `None` and
                # resolves AUTO, so a documented PEFT/trainable-kernel choice (or
                # a deliberately incompatible one, e.g. marlin on CPU) was
                # silently ignored for native checkpoints (#4680). Optimum keeps
                # it on the quantizer; fall back to the config so a bridge that
                # loses `.backend` still forwards it (#4729).
                backend=_hf_quantizer_backend(self),
                dtype=kwargs.get("dtype"),
            )
            if context is None:
                return original_convert_model(self, model, **kwargs)
            model._gptqmodel_hf_load_context = context
            return model

        @wraps(original_post_init_model)
        def _post_init_model_with_gptqmodel_bridge(self, model):
            context = getattr(model, "_gptqmodel_hf_load_context", None)
            if context is None:
                return original_post_init_model(self, model)
            from .utils.hf_load import hf_gptqmodel_post_init_for_load

            return hf_gptqmodel_post_init_for_load(model, context)

        _convert_model_with_gptqmodel_bridge._gptqmodel_load_bridge = True
        _post_init_model_with_gptqmodel_bridge._gptqmodel_load_bridge = True
        GPTQQuantizer.convert_model = _convert_model_with_gptqmodel_bridge
        GPTQQuantizer.post_init_model = _post_init_model_with_gptqmodel_bridge
        _OPTIMUM_GPTQ_BRIDGE_INSTALLED = True
        return True


def _ensure_optimum_gptqmodel_load_bridge():
    """Install the Optimum GPTQ bridge at the first moment it can succeed.

    ``optimum.gptq`` imports ``gptqmodel`` mid-import, so the package-init
    attempt legitimately fails with a partially initialized module and must not
    be cached as "Optimum not installed" (#4749). Everything Transformers or
    Optimum calls after that import completes retries through here: the
    ``GptqHfQuantizer`` hooks and the ``hf_select_quant_linear_v2`` /
    ``hf_gptqmodel_post_init`` entry points Optimum resolves from this package.

    The retry is cheap and idempotent: one module-level flag after success, one
    after a genuine absence, and no import attempt while ``optimum.gptq`` is
    visibly still initializing.
    """

    global _OPTIMUM_GPTQ_BRIDGE_UNAVAILABLE

    if _OPTIMUM_GPTQ_BRIDGE_INSTALLED:
        return True
    if _OPTIMUM_GPTQ_BRIDGE_UNAVAILABLE:
        return False

    optimum_gptq = sys.modules.get("optimum.gptq")
    if optimum_gptq is None:
        try:
            spec = importlib.util.find_spec("optimum.gptq")
        except (AttributeError, ImportError, ValueError):
            spec = None
        if spec is None:
            # Optimum is genuinely absent; do not retry the failed import.
            _OPTIMUM_GPTQ_BRIDGE_UNAVAILABLE = True
            return False
    elif not hasattr(optimum_gptq, "GPTQQuantizer"):
        # ``optimum.gptq`` is the import that is currently importing us. The
        # class does not exist yet, so importing it again cannot succeed.
        return False

    if _patch_optimum_gptqmodel_load_bridge():
        return True
    _OPTIMUM_GPTQ_BRIDGE_UNAVAILABLE = True
    return False


def _reselect_kernels_for_resolved_device_map(quantizer, device_map) -> None:
    """Re-pick kernels once Transformers resolves a keyword ``device_map``.

    ``preprocess_model`` runs before ``_get_device_map`` turns ``"auto"`` plus
    ``max_memory`` into a placement, so the bridge cannot see a CPU/disk spill
    and AUTO keeps a CUDA-only Marlin kernel. ``_get_device_map`` calls
    ``validate_environment`` again with the *resolved* dict, which is the first
    point both the model and its final placement are known (#5194).
    """

    if not isinstance(device_map, dict):
        return
    model = getattr(quantizer, "_gptqmodel_model", None)
    context = getattr(model, "_gptqmodel_hf_load_context", None)
    if model is None or context is None:
        return
    from .utils.hf_load import hf_gptqmodel_reselect_offloaded_kernels

    hf_gptqmodel_reselect_offloaded_kernels(model, context, device_map)


def _patch_transformers_gptq_quantizer_bridge():
    """Retry the Optimum GPTQ bridge from Transformers' GPTQ quantizer.

    Transformers is fully imported before ``GptqHfQuantizer`` touches Optimum,
    so its pre/post weight-loading hooks always run after ``optimum.gptq`` has
    finished importing. That gives the deferred bridge install a slot before
    Optimum's ``convert_model`` replaces any layer, independent of whether the
    user imported ``gptqmodel`` first (#4749).
    """

    try:
        from functools import wraps

        from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
    except ImportError:
        # Older Transformers vendors its own GPTQ loader without Optimum.
        return
    except Exception:
        log.warning("transformers GPTQ load-bridge compat probe failed", exc_info=True)
        return

    with _MONKEY_PATCH_LOCK:
        original_before = GptqHfQuantizer._process_model_before_weight_loading
        if getattr(original_before, "_gptqmodel_load_bridge", False):
            return
        original_after = GptqHfQuantizer._process_model_after_weight_loading

        original_validate = getattr(GptqHfQuantizer, "validate_environment", None)

        @wraps(original_before)
        def _before_with_gptqmodel_bridge(self, model, **kwargs):
            _ensure_optimum_gptqmodel_load_bridge()
            result = original_before(self, model, **kwargs)
            # Keep the model reachable from `validate_environment`, which
            # Transformers calls with the resolved placement but without it.
            self._gptqmodel_model = model
            return result

        @wraps(original_after)
        def _after_with_gptqmodel_bridge(self, model, **kwargs):
            _ensure_optimum_gptqmodel_load_bridge()
            return original_after(self, model, **kwargs)

        _before_with_gptqmodel_bridge._gptqmodel_load_bridge = True
        _after_with_gptqmodel_bridge._gptqmodel_load_bridge = True
        GptqHfQuantizer._process_model_before_weight_loading = _before_with_gptqmodel_bridge
        GptqHfQuantizer._process_model_after_weight_loading = _after_with_gptqmodel_bridge
        if original_validate is not None and not getattr(
            original_validate, "_gptqmodel_load_bridge", False
        ):

            @wraps(original_validate)
            def _validate_with_gptqmodel_bridge(self, *args, **kwargs):
                result = original_validate(self, *args, **kwargs)
                _reselect_kernels_for_resolved_device_map(
                    self, kwargs.get("device_map")
                )
                return result

            _validate_with_gptqmodel_bridge._gptqmodel_load_bridge = True
            GptqHfQuantizer.validate_environment = _validate_with_gptqmodel_bridge


def _patch_transformers_awq_gptqmodel_load_bridge():
    """Route Transformers' own AWQ loader through GPTQModel's native bridge.

    Optimum's GPTQ path is patched above, but Transformers loads
    ``quant_method: "awq"`` through ``transformers.quantizers.quantizer_awq``
    with no Optimum involvement. That path calls
    ``integrations/awq.py::replace_with_awq_linear``, which only swaps
    ``nn.Linear`` modules. A GPT-QModel AWQ checkpoint can store per-expert
    ``experts.<n>.*.qweight`` tensors for fused 3-D expert parameters, so the
    experts stay dense and every packed expert tensor is reported UNEXPECTED:
    the model loads without error and the routed experts are randomly
    initialized. The same path builds every module from the global
    ``bits``/``group_size``, so ``dynamic`` per-module overrides are ignored
    (#4795).

    Wrapping the AWQ quantizer's pre/post weight-loading hooks lets
    ``hf_gptqmodel_prepare_model_for_load`` rebuild the checkpoint's real module
    manifest first. Checkpoints Transformers' AWQ path owns keep that path
    because the bridge returns ``None`` for them. The post hook additionally
    repairs a load that started before this patch could be installed: that is
    exactly the plain-``AutoModelForCausalLM`` flow, where
    ``replace_with_awq_linear`` imports ``gptqmodel`` from inside itself.

    Returns ``True`` when the hooks are installed.
    """

    global _AWQ_BRIDGE_INSTALLED

    try:
        from functools import wraps

        from transformers.quantizers.quantizer_awq import AwqQuantizer
    except ImportError:
        # Transformers is a hard dependency; a missing symbol means the upstream
        # AWQ API moved and this compatibility shim is now inert.
        log.warning(
            "transformers AWQ load-bridge is inactive: AwqQuantizer API changed",
            exc_info=True,
        )
        return False
    except Exception:
        log.warning("transformers AWQ load-bridge compat probe failed", exc_info=True)
        return False

    with _MONKEY_PATCH_LOCK:
        original_prepare = AwqQuantizer._process_model_before_weight_loading
        if getattr(original_prepare, "_gptqmodel_load_bridge", False):
            _AWQ_BRIDGE_INSTALLED = True
            return True
        original_post_init = AwqQuantizer._process_model_after_weight_loading
        original_validate = getattr(AwqQuantizer, "validate_environment", None)

        @wraps(original_prepare)
        def _prepare_with_gptqmodel_bridge(self, model, **kwargs):
            from .utils.hf_load import hf_gptqmodel_prepare_model_for_load

            context = hf_gptqmodel_prepare_model_for_load(
                model,
                checkpoint_files=kwargs.get("checkpoint_files"),
                device_map=kwargs.get("device_map"),
                # Same forwarding as the GPTQ bridge: an explicit
                # `quantization_config["backend"]` must not be replaced by AUTO
                # (#4680). transformers' `AwqQuantizer` has no `.backend`; the
                # choice is on its `quantization_config`, so reading only the
                # quantizer dropped every AWQ backend (#4729).
                backend=_hf_quantizer_backend(self),
                dtype=kwargs.get("dtype"),
            )
            if context is None:
                return original_prepare(self, model, **kwargs)
            model._gptqmodel_hf_load_context = context
            # `validate_environment` gets the resolved placement but no model.
            self._gptqmodel_model = model
            return model

        @wraps(original_post_init)
        def _post_init_with_gptqmodel_bridge(self, model, **kwargs):
            from .utils.hf_load import (
                hf_gptqmodel_post_init_for_load,
                hf_gptqmodel_repair_awq_load,
            )

            _ensure_transformers_awq_gptqmodel_load_bridge()
            context = getattr(model, "_gptqmodel_hf_load_context", None)
            if context is None:
                # No prepare step ran for this load: ``gptqmodel`` was first
                # imported from inside transformers' ``replace_with_awq_linear``
                # while it was already building modules (#4795). The checkpoint
                # tensors are resident now, so rebuild the modules that ignored
                # the serialized per-module config before the native post-init.
                context = hf_gptqmodel_repair_awq_load(
                    model,
                    backend=_hf_quantizer_backend(self),
                    checkpoint_files=kwargs.get("checkpoint_files"),
                    dtype=kwargs.get("dtype"),
                )
            if context is None:
                return original_post_init(self, model, **kwargs)
            return hf_gptqmodel_post_init_for_load(model, context)

        _prepare_with_gptqmodel_bridge._gptqmodel_load_bridge = True
        _post_init_with_gptqmodel_bridge._gptqmodel_load_bridge = True
        AwqQuantizer._process_model_before_weight_loading = _prepare_with_gptqmodel_bridge
        AwqQuantizer._process_model_after_weight_loading = _post_init_with_gptqmodel_bridge
        if original_validate is not None and not getattr(
            original_validate, "_gptqmodel_load_bridge", False
        ):

            @wraps(original_validate)
            def _validate_with_gptqmodel_bridge(self, *args, **kwargs):
                result = original_validate(self, *args, **kwargs)
                _reselect_kernels_for_resolved_device_map(
                    self, kwargs.get("device_map")
                )
                return result

            _validate_with_gptqmodel_bridge._gptqmodel_load_bridge = True
            AwqQuantizer.validate_environment = _validate_with_gptqmodel_bridge
        _AWQ_BRIDGE_INSTALLED = True
        return True


def _ensure_transformers_awq_gptqmodel_load_bridge():
    """Install the AWQ bridge at the first moment it can succeed.

    Unlike Optimum's GPTQ quantizer, transformers' ``AwqQuantizer`` never
    imports ``gptqmodel`` before it starts loading: the first import happens
    inside ``replace_with_awq_linear``, i.e. after the prepare hook would have
    run. This installer is retried from every later entry point (the quantizer's
    post-weight hook, ``hf_select_quant_linear_v2``, and the package init) so
    every *subsequent* load in the process gets the native prepare step, while
    the post hook repairs the load that triggered the import (#4795).

    Cheap and idempotent: one module-level flag after success, one after the
    AWQ quantizer module is genuinely absent (an older Transformers), and no
    latch while that module is visibly still initializing.
    """

    global _AWQ_BRIDGE_UNAVAILABLE

    if _AWQ_BRIDGE_INSTALLED:
        return True
    if _AWQ_BRIDGE_UNAVAILABLE:
        return False

    quantizer_awq = sys.modules.get("transformers.quantizers.quantizer_awq")
    if quantizer_awq is not None and not hasattr(quantizer_awq, "AwqQuantizer"):
        # The module that is importing us has not defined its quantizer yet;
        # importing it again cannot succeed.
        return False

    if _patch_transformers_awq_gptqmodel_load_bridge():
        return True
    _AWQ_BRIDGE_UNAVAILABLE = True
    return False


from .utils import has_gil_disabled
from .utils.env import env_flag
from .utils.kernel import get_kernel_config
from .utils.modelscope import ensure_modelscope_available


DEBUG_ON = env_flag("DEBUG")

from .utils.linalg_warmup import run_torch_linalg_warmup
from .utils.threadx import DeviceThreadPool, WarmUpCtx, WarmupTask


_DEVICE_THREAD_POOL = None
_DEVICE_THREAD_POOL_LOCK = threading.Lock()


def _default_cpu_workers() -> int:
    """Return the CPU worker count selected by the kernel policy."""

    configured = get_kernel_config().cpu_workers
    if configured is not None:
        return configured

    # Cap CPU workers to avoid OpenMP team explosion under free-threading.
    # On GIL builds keep the historical default; only free-threading needs the aggressive cap.
    if has_gil_disabled():
        return min(8, max(2, (os.cpu_count() or 1) // 16))
    return min(12, max(1, ((os.cpu_count() or 1) + 1) // 2))


def _default_cuda_workers() -> int:
    """Return the per-CUDA worker count selected by the kernel policy."""

    configured = get_kernel_config().cuda_workers
    if configured is not None:
        return configured
    return 4


def _device_thread_pool_settings(cpu_workers=None) -> tuple:
    """Return the pool parameters the current kernel policy selects.

    The tuple is ``(cpu_workers, cuda_workers_per_device, intraop_threads)``.
    Worker threads are fixed at construction, so any change to one of these
    requires a rebuild.
    """

    return (
        _default_cpu_workers() if cpu_workers is None else int(cpu_workers),
        _default_cuda_workers(),
        get_kernel_config().pool_intraop_threads,
    )


def _build_device_thread_pool(warmup_devices=None, cpu_workers=None, settings=None):
    if settings is None:
        settings = _device_thread_pool_settings(cpu_workers)
    cpu_workers, cuda_workers, intraop_threads = settings
    pool = DeviceThreadPool(
        inference_mode=True,
        warmup_devices=warmup_devices,
        intraop_threads_per_worker=intraop_threads,
        warmups={
            "cuda": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
            "xpu": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
            "mps": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
            "cpu": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
        },
        workers={
            "cuda:per": cuda_workers,
            "xpu:per": 1,
            "npu:per": 1,
            "mps": 8,
            "cpu": cpu_workers,
            "model_loader:cpu": 2,
        },
        empty_cache_every_n=512,
    )
    # Remember what the pool was built for so a later load can tell whether
    # its policy asks for a different pool.
    pool._gptqmodel_settings = settings
    return pool


def _device_thread_pool_settings_of(pool) -> tuple | None:
    return getattr(pool, "_gptqmodel_settings", None)


def get_device_thread_pool(warmup_devices=None):
    global _DEVICE_THREAD_POOL
    if _DEVICE_THREAD_POOL is None:
        with _DEVICE_THREAD_POOL_LOCK:
            if _DEVICE_THREAD_POOL is None:
                pool = _build_device_thread_pool(warmup_devices=warmup_devices)
                # Register before publishing the pool so every caller that
                # receives it has a matching exit cleanup. Capture this pool:
                # a later replacement must not leave its workers unjoined.
                atexit.register(_shutdown_device_thread_pool, pool)
                _DEVICE_THREAD_POOL = pool
    return _DEVICE_THREAD_POOL


def resize_device_thread_pool(warmup_devices=None, cpu_workers=None, *, create: bool = True):
    """Match the singleton pool to the current kernel policy.

    Worker threads cannot be resized in place, so a later load whose
    ``RuntimeConfig`` asks for a different ``cpu_workers``, ``cuda_workers`` or
    ``pool_intraop_threads`` replaces the pool at the load boundary. Before
    this, the per-process singleton kept the first load's settings and
    silently ignored later requests (issues #4396, #4448).

    ``create=False`` only retunes an existing pool; callers that must not build
    one eagerly (post-quantized loads) leave the lazy build to the first use,
    which reads the already-installed policy.
    """

    global _DEVICE_THREAD_POOL
    target = _device_thread_pool_settings(cpu_workers)
    pool = _DEVICE_THREAD_POOL
    if pool is not None and _device_thread_pool_settings_of(pool) == target:
        return pool
    if pool is None and not create:
        return None

    with _DEVICE_THREAD_POOL_LOCK:
        pool = _DEVICE_THREAD_POOL
        if pool is not None and _device_thread_pool_settings_of(pool) == target:
            return pool
        if pool is not None:
            log.info(
                "ENV: rebuilding device thread pool (cpu_workers, cuda_workers, pool_intraop_threads) "
                "from %s to %s for the new kernel policy.",
                _device_thread_pool_settings_of(pool),
                target,
            )
            # Detach before shutting down so no concurrent caller can submit to
            # a pool whose workers are stopping.
            _DEVICE_THREAD_POOL = None
            pool.shutdown(wait=True)
        pool = _build_device_thread_pool(warmup_devices=warmup_devices, settings=target)
        atexit.register(_shutdown_device_thread_pool, pool)
        _DEVICE_THREAD_POOL = pool
        return pool


def _shutdown_device_thread_pool(pool: DeviceThreadPool) -> None:
    """Join pool workers at exit so finalization cannot abort."""
    try:
        pool.shutdown(wait=True)
    except Exception:
        # Exit cleanup must never turn a successful run into a failed one.
        pass


class _LazyDeviceThreadPoolProxy:
    def __init__(self):
        object.__setattr__(self, "_overrides", {})

    def __getattribute__(self, name):
        if name in {
            "_overrides",
            "__class__",
            "__dict__",
            "__getattribute__",
            "__setattr__",
            "__delattr__",
            "__repr__",
            "__dir__",
            "_get_pool",
        }:
            return object.__getattribute__(self, name)

        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]

        return getattr(self._get_pool(), name)

    def __setattr__(self, name, value):
        if name == "_overrides":
            object.__setattr__(self, name, value)
            return
        self._overrides[name] = value

    def __delattr__(self, name):
        overrides = self._overrides
        if name in overrides:
            del overrides[name]
            return
        delattr(self._get_pool(), name)

    def __repr__(self):
        pool = _DEVICE_THREAD_POOL
        if pool is None:
            return "<LazyDeviceThreadPoolProxy uninitialized>"
        return repr(pool)

    def __dir__(self):
        attrs = set(self._overrides.keys())
        pool = _DEVICE_THREAD_POOL
        if pool is not None:
            attrs.update(dir(pool))
        return sorted(attrs)

    @staticmethod
    def _get_pool():
        return get_device_thread_pool()


DEVICE_THREAD_POOL = _LazyDeviceThreadPoolProxy()

_patch_transformers_gptq_device_map_compat()
_patch_transformers_gptqmodel_gguf_quantizer_compat()
_patch_transformers_refusing_quantizers()
_patch_transformers_save_pretrained_runtime_layout_guard()
_patch_openvino_gptqmodel_compat()
_patch_transformers_causal_conv1d_hub_kernel_compat()


def exllama_set_max_input_length(model, max_input_length: int):
    """Resize exllama scratch buffers through the legacy package-root API."""
    from .utils.model import hf_gptqmodel_post_init

    quantize_config = getattr(model, "quantize_config", None)
    use_act_order = bool(getattr(quantize_config, "desc_act", False))
    return hf_gptqmodel_post_init(
        model,
        use_act_order=use_act_order,
        quantize_config=quantize_config,
        max_input_length=max_input_length,
    )


import torch

from . import extension
from .backend import (  # noqa: F401
    AwqGemmConfig,
    AwqTorchConfig,
    AwqTritonConfig,
    BackendConfig,
    BitsAndBytesConfig,
    GGUFCPPConfig,
    GGUFCUDAConfig,
    GGUFTorchConfig,
    MarlinConfig,
    ParoCUDAConfig,
    ParoTritonConfig,
    RuntimeConfig,
    SwordfishConfig,
    TorchConfig,
    TTBlockFloatConfig,
)
from .looper.checkpoint import CheckpointStopped
from .looper.checkpoint_store import CheckpointConfig
from .models import (
    FusableConfig,
    FusableProfile,
    GPTQModel,
    LlamaFusableConfig,
    ModulePlan,
    ModuleSpec,
    get_best_device,
)
from .models.auto import ASCII_LOGO, TRANSFORMERS_VERSION
from .quantization import (
    AutoDtypeCast,
    AWQConfig,
    BaseQuantizeConfig,
    CalibrationConfig,
    FOEMConfig,
    GGUFConfig,
    GPTAQConfig,
    GPTQConfig,
    MSEConfig,
    NormConfig,
    NormType,
    QQQConfig,
    QuantizeConfig,
    QuantizeEmbed,
    QuantizeEmbedConfig,
    RTNConfig,
    TelemetryConfig,
    WeightOnlyConfig,
)
from .reshard import ShardConfig, ShardLayout, reshard
from .utils import BACKEND, PROFILE
from .utils.p2p import P2PConfig
from .version import __version__


log.info(
    "\n%s",
    get_startup_banner(
        ASCII_LOGO,
        gptqmodel_version=__version__,
        transformers_version=TRANSFORMERS_VERSION,
        torch_version=torch.__version__,
    ),
)

if ensure_modelscope_available():
    try:
        from modelscope.utils.hf_util.patcher import patch_hub

        patch_hub()
    except Exception as exc:
        raise ModuleNotFoundError(
            "env `GPTQMODEL_USE_MODELSCOPE` used but modelscope pkg is not found: please install with `pip install modelscope`."
        ) from exc


# Installed last: `optimum.gptq` imports `gptqmodel` (BACKEND, QuantizeConfig)
# while it loads, so this eager attempt can only succeed once the package has
# finished defining its public names. When it runs *during* that import the
# probe sees a partially initialized `optimum.gptq`; `_ensure_...` then retries
# from Transformers' GPTQ hooks and from the entry points Optimum calls, so the
# bridge no longer depends on the user importing `gptqmodel` first (#4749).
_patch_optimum_gptqmodel_load_bridge()
_patch_transformers_gptq_quantizer_bridge()
_ensure_transformers_awq_gptqmodel_load_bridge()
