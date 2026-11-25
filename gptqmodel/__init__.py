# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import atexit
import os
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

from ._banner import get_startup_banner  # noqa: E402
from .utils.logger import setup_logger  # noqa: E402
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
    # The direct-to-Torch dense GGUF import redirected Transformers' loader and
    # was removed with every other Transformers patch; it has no replacement.
    "GPTQMODEL_INTERNAL_GGUF_TORCH_LOADER": None,
    "GPTQMODEL_MARLIN_USE_FP32": "KernelConfig.marlin_fp32",
    "GPTQMODEL_MAX_PACKERS": "RuntimeConfig.max_packers",
    # The torch last-level-cache back-port patched torch and Inductor globally;
    # GPT-QModel never patches third-party libraries, so it has no replacement.
    "GPTQMODEL_NO_ROCM_LLC": None,
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
        if replacement is None:
            log.warn(f"ENV: {name} is retired and has no effect; the feature was removed.")
        else:
            log.warn(
                f"ENV: {name} is retired and has no effect; use {replacement} instead."
            )
        warned.append(name)
    return warned


warn_retired_kernel_env_vars()

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


def _patch_transformers_model_construction_global_state():
    """Hold the model-global-state writer for each construction window.

    transformers opens a process-global ``torch.set_default_dtype`` window for
    each construction (#5304) and, in the same init-context group, disables
    ``PreTrainedModel.tie_weights`` class-wide and swaps registered classes
    (#5421). Only the window was exclusive, so a foreign construction in
    another thread could untie a GPT-QModel shell. Every construction now holds
    the writer for the window; a compiled forward still does not wait for a
    dense load's weight I/O, dtype casts and device placement.
    """

    from .utils.torch import install_model_global_state_dtype_guard

    install_model_global_state_dtype_guard()


from .utils import has_gil_disabled
from .utils.env import env_flag
from .utils.kernel import get_kernel_config


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

# The port installs one Transformers guard: the model-global-state writer for
# the construction window (#5304, #5421). It now covers the dtype window, the
# init-context group and the class-wide patches, so the port's older
# dtype-only entry point and the model-compatibility patches main still keeps
# for its Transformers 4 tree are not installed here.
_patch_transformers_model_construction_global_state()


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
    EXL3CUDAConfig,
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
    CPTCConfig,
    FOEMConfig,
    G2PTQConfig,
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
from .utils.leapquant import LeapQuantConfig
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
