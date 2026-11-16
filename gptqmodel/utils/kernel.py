# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Explicit selection of kernel, numerical-path and thread-count behaviour.

GPT-QModel never reads an environment variable to decide which kernel runs,
which numerical path is taken, or how many threads are used.  Those choices
are selected with a typed ``BackendConfig`` passed to ``backend=`` and shared
``RuntimeConfig`` settings. :class:`KernelConfig` is the legacy adapter,
accepted through ``kernel=`` and installed with :func:`configure_kernel`.

Quantized modules own a policy snapshot. Shared quantizer and host utilities
use the process-wide policy installed before a run. Calling
:func:`configure_kernel` replaces that fallback; ``None`` restores defaults.
It does not retune already constructed modules.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional


_BOOL_FIELDS = (
    "gptq_fused_block_update",
    "floatx_cpu_native_fp8",
    "torch_streaming",
    "torch_rocm_gemv",
    "torch_rocm_gemm",
    "torch_cache_weights",
    "torch_lookahead",
    "marlin_fp32",
    "fp32_accum",
    "awq_fused_splitk_reduce",
    "swordfish_deterministic",
    "blockfloat_cublas",
    "cpu_bf16x3_matmul",
    "paroquant_cache_runtime_dtype",
    "paroquant_auto_cache_bf16_runtime_dtype",
    "paroquant_cache_rotation_dtype",
    "paroquant_auto_cache_bf16_rotation_dtype",
    "paroquant_cpu_avx512",
    "paroquant_triton_autotune",
    "paroquant_sym_ste_triton",
    "paroquant_affine_ste_triton",
    "paroquant_affine_ste_triton_backward",
    "paroquant_rotate_autotune",
    "paroquant_autotune_cache",
    "bitsandbytes_cpu_packed",
    "bitsandbytes_cpu_int8_dense",
    "bitsandbytes_triton",
    "disable_bitblas",
    "gsq_disable_cuda_relaxation",
    "gsq_disable_hip_relaxation",
    "rocm_llc",
)

_OPTIONAL_BOOL_FIELDS = ("torch_triton_dequant", "gguf_cpp_native_matmul")

_MIN_INT_FIELDS = {
    "npu_cholesky_block_size": 1,
    "floatx_cpu_simd_level": 0,
    "swordfish_dense_m": 0,
    "torch_stream_tile": 1,
    "gguf_fused_cuda_max_rows": 0,
    "gguf_fused_cuda_min_matrix_elements": 0,
    "gguf_fused_cpu_max_rows": 0,
    "gguf_fused_cpu_min_matrix_elements": 0,
    "gguf_fused_chunk_blocks": 0,
    "gguf_fused_chunk_bytes": 1,
    "gguf_fused_autotune_warmup": 0,
    "gguf_fused_autotune_iters": 1,
    "paroquant_triton_autotune_warmup": 0,
    "paroquant_triton_autotune_iters": 1,
    "paroquant_triton_decode_max_rows": 1,
    "exl3_tile_batch": 1,
    "exl3_host_rss_limit_bytes": 0,
    "exl3_cuda_allocation_limit_bytes": 0,
    "exl3_memory_telemetry_interval_batches": 0,
    "exl3_capture_batch_checkpoint_interval": 1,
}

_OPTIONAL_MIN_INT_FIELDS = {
    "floatx_cpu_threads": 1,
    "gguf_cpp_threads": 1,
    "cpu_workers": 1,
    "cuda_workers": 1,
    "max_packers": 1,
}

_MIN_FLOAT_FIELDS = {
    "gguf_fused_autotune_margin": 0.0,
    "paroquant_triton_autotune_margin": 0.0,
}

_OPTIONAL_CHOICE_FIELDS = {
    "paroquant_rotate_cta_m": frozenset({4, 8, 16}),
    "paroquant_rotate_row_pad": frozenset({0, 2}),
}

_AWQ_SCHEDULE_CHOICES = frozenset({"auto", "legacy"})


@dataclass
class KernelConfig:
    """Kernel, numerical-path and thread-count selection.

    Every field replaces one former ``GPTQMODEL_*`` / ``GPTQ_TORCH_*``
    environment variable.  Defaults are the values those variables defaulted
    to, so omitting the config preserves the previous default behaviour.

    Quantization-time fields (GPTQ block update, NPU Cholesky, GSQ relaxation,
    packer/worker counts, FloatX decode) are read while a run executes.
    Inference-time fields (Marlin, Torch, GGUF, ParoQuant, bitsandbytes, Triton
    accumulation, BitBLAS) are read when a quantized module is built or runs.
    """

    # GPTQ block-update kernel. Replaces ``GPTQMODEL_DISABLE_GPTQ_CUDA``
    # (which was the inverse). ``False`` forces the eager per-column path.
    gptq_fused_block_update: bool = True

    # Blocked Cholesky on Ascend NPU. Replaces
    # ``GPTQMODEL_NPU_CHOLESKY_BLOCK_SIZE``.
    npu_cholesky_block_size: int = 64

    # FloatX CPU decode. Replace ``GPTQMODEL_FLOATX_CPU_THREADS`` and
    # ``GPTQMODEL_FLOATX_CPU_FORCE_NATIVE_FP8``. ``None`` keeps the
    # size-derived default thread count.
    floatx_cpu_threads: Optional[int] = None
    floatx_cpu_native_fp8: bool = False
    # SIMD level for the FloatX CPU decode: 0 = scalar, 1 = AVX2/F16C,
    # 2 = AVX512. Replaces ``GPTQMODEL_FLOATX_CPU_DISABLE_AVX2`` and
    # ``GPTQMODEL_FLOATX_CPU_DISABLE_AVX512``. The default always probes.
    floatx_cpu_simd_level: int = 2

    # Torch GPTQ kernel. Replace ``GPTQ_TORCH_*``.
    torch_streaming: bool = False
    torch_stream_tile: int = 512
    torch_triton_dequant: Optional[bool] = None
    torch_rocm_gemv: bool = True
    torch_rocm_gemm: bool = True
    torch_cache_weights: bool = False
    torch_lookahead: bool = False

    # Marlin reduction precision. Replaces ``GPTQMODEL_MARLIN_USE_FP32``.
    marlin_fp32: bool = True

    # Triton GEMM accumulation. Replaces ``GPTQMODEL_FP32_ACCUM``.
    fp32_accum: bool = True

    # oneDNN BF16x3 FP32 matmul on CPU. Replaces ``GPTQMODEL_CPU_BF16X3`` and
    # ``GPTQMODEL_EXL3_CPU_BF16X3``; both used to gate the same capability, so
    # one field now covers the EXL3 and generic CPU matmul callers.
    cpu_bf16x3_matmul: bool = True

    # Tenstorrent block-float CUDA linear. Replaces
    # ``GPTQMODEL_BLOCKFLOAT_NO_CUBLAS`` (which was the inverse). ``False``
    # forces the fused dequant+MMA fallback instead of the cuBLAS prefill.
    blockfloat_cublas: bool = True

    # Swordfish Blackwell GEMM dispatch. Replaces
    # ``APHRODITE_SWORDFISH_DETERMINISTIC`` and
    # ``APHRODITE_SWORDFISH_DENSE_M``. ``swordfish_dense_m`` is ``0`` for the
    # measured per-SM heuristic; a positive value pins the dense-tier crossover.
    swordfish_deterministic: bool = False
    swordfish_dense_m: int = 0

    # AWQ CUDA split-K reduce. Replaces
    # ``GPTQMODEL_AWQ_DISABLE_FUSED_SPLITK_REDUCE`` (which was the inverse).
    # ``False`` reduces through the eager ``sum(0)`` instead of the fused
    # reduction kernel.
    awq_fused_splitk_reduce: bool = True

    # AWQ Triton schedule. Replaces ``GPTQMODEL_AWQ_TRITON_SCHEDULE``.
    awq_triton_schedule: str = "auto"

    # GGUF fused forward and native C++ matmul. Replace
    # ``GPTQMODEL_GGUF_FUSED_*``, ``GPTQMODEL_GGUF_CPP_NATIVE_MATMUL`` and
    # ``GPTQMODEL_GGUF_CPP_THREADS``.
    gguf_fused_cuda_max_rows: int = 32
    gguf_fused_cuda_min_matrix_elements: int = 8_388_608
    gguf_fused_cpu_max_rows: int = 64
    gguf_fused_cpu_min_matrix_elements: int = 0
    gguf_fused_chunk_blocks: int = 0
    gguf_fused_chunk_bytes: int = 32 * 1024 * 1024
    gguf_fused_autotune_warmup: int = 1
    gguf_fused_autotune_iters: int = 2
    gguf_fused_autotune_margin: float = 0.05
    gguf_cpp_native_matmul: Optional[bool] = None
    gguf_cpp_threads: Optional[int] = None

    # ParoQuant inference/optimization. Replace ``GPTQMODEL_PAROQUANT_*``.
    paroquant_cache_runtime_dtype: bool = False
    paroquant_auto_cache_bf16_runtime_dtype: bool = True
    paroquant_cache_rotation_dtype: bool = False
    paroquant_auto_cache_bf16_rotation_dtype: bool = True
    paroquant_cpu_avx512: bool = True
    paroquant_triton_autotune: bool = True
    paroquant_triton_autotune_warmup: int = 2
    paroquant_triton_autotune_iters: int = 4
    paroquant_triton_autotune_margin: float = 0.05
    paroquant_triton_decode_max_rows: int = 8
    paroquant_sym_ste_triton: bool = True
    paroquant_affine_ste_triton: bool = True
    paroquant_affine_ste_triton_backward: bool = False
    paroquant_rotate_autotune: bool = True
    paroquant_rotate_cta_m: Optional[int] = None
    paroquant_rotate_row_pad: Optional[int] = None
    paroquant_autotune_cache: bool = True

    # bitsandbytes / BitBLAS. Replace ``GPTQMODEL_BITSANDBYTES_*``,
    # ``GPTQMODEL_BNB_TRITON`` and ``GPTQMODEL_DISABLE_BITBLAS``.
    bitsandbytes_cpu_packed: bool = False
    bitsandbytes_cpu_int8_dense: bool = True
    bitsandbytes_triton: bool = False
    disable_bitblas: bool = False

    # GSQ fused relaxation. Replace ``GPTQMODEL_GSQ_DISABLE_*_RELAXATION``.
    gsq_disable_cuda_relaxation: bool = False
    gsq_disable_hip_relaxation: bool = False

    # EXL3 scratch, telemetry and capture. Replace ``GPTQMODEL_EXL3_*``.
    exl3_tile_batch: int = 256
    exl3_host_rss_limit_bytes: int = 0
    exl3_cuda_allocation_limit_bytes: int = 0
    exl3_memory_telemetry_interval_batches: int = 0
    exl3_capture_batch_spool: Optional[str] = None
    exl3_capture_batch_checkpoint_interval: int = 1
    exl3_capture_frontier: Optional[str] = None

    # ROCm quantization BLAS and LLC back-port. Replace
    # ``GPTQMODEL_ROCM_QUANT_BLAS`` and ``GPTQMODEL_NO_ROCM_LLC``.
    rocm_quant_blas: Optional[str] = None
    rocm_llc: bool = True

    # Thread counts. Replace ``GPTQMODEL_CPU_WORKERS``,
    # ``GPTQMODEL_CUDA_WORKERS`` and ``GPTQMODEL_MAX_PACKERS``. ``None`` keeps
    # the platform-derived default.
    cpu_workers: Optional[int] = None
    cuda_workers: Optional[int] = None
    max_packers: Optional[int] = None

    def __post_init__(self) -> None:
        for name in _BOOL_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(
                    f"KernelConfig.{name} must be a bool; got {type(value).__name__}"
                )
        for name in _OPTIONAL_BOOL_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(
                    f"KernelConfig.{name} must be a bool or None; got {type(value).__name__}"
                )

        for name, minimum in _MIN_INT_FIELDS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"KernelConfig.{name} must be an integer; got {type(value).__name__}"
                )
            if value < minimum:
                raise ValueError(f"KernelConfig.{name} must be >= {minimum}; got {value}")

        for name, minimum in _OPTIONAL_MIN_INT_FIELDS.items():
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"KernelConfig.{name} must be an integer or None; got {type(value).__name__}"
                )
            if value < minimum:
                raise ValueError(f"KernelConfig.{name} must be >= {minimum}; got {value}")

        for name, minimum in _MIN_FLOAT_FIELDS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"KernelConfig.{name} must be a number; got {type(value).__name__}"
                )
            value = float(value)
            if value < minimum:
                raise ValueError(f"KernelConfig.{name} should be >= {minimum}; got {value}")
            setattr(self, name, value)

        for name, choices in _OPTIONAL_CHOICE_FIELDS.items():
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"KernelConfig.{name} should be an integer or None; got {type(value).__name__}"
                )
            if value not in choices:
                raise ValueError(
                    f"KernelConfig.{name} should be one of {sorted(choices)}; got {value}"
                )

        schedule = str(self.awq_triton_schedule).strip().lower()
        if schedule not in _AWQ_SCHEDULE_CHOICES:
            raise ValueError(
                "KernelConfig.awq_triton_schedule should be 'auto' or 'legacy'; "
                f"got {self.awq_triton_schedule!r}"
            )
        self.awq_triton_schedule = schedule

        for name in ("exl3_capture_batch_spool", "exl3_capture_frontier"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError(
                    f"KernelConfig.{name} should be a string or None; got {type(value).__name__}"
                )
            setattr(self, name, value.strip() or None)

        # An explicit empty string is a documented opt-out spelling, so it must
        # stay distinct from ``None`` (which keeps the reproducible default).
        if self.rocm_quant_blas is not None:
            if not isinstance(self.rocm_quant_blas, str):
                raise TypeError(
                    "KernelConfig.rocm_quant_blas should be a string or None; got "
                    f"{type(self.rocm_quant_blas).__name__}"
                )
            self.rocm_quant_blas = self.rocm_quant_blas.strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_kernel_config(config: KernelConfig | Mapping[str, Any] | None) -> KernelConfig:
    if config is None:
        return KernelConfig()
    if isinstance(config, KernelConfig):
        return config
    if isinstance(config, Mapping):
        return KernelConfig(**dict(config))
    raise TypeError(
        "kernel config should be a KernelConfig, a mapping of its fields, or None; "
        f"got {type(config).__name__}"
    )


_CONFIG_LOCK = threading.RLock()
_ACTIVE_CONFIG: KernelConfig = KernelConfig()
# True only while a ``rocm_llc=False`` policy is what removed the back-port, so
# re-enabling restores it without re-applying a patch the import already made.
_ROCM_LLC_SUPPRESSED = False


def configure_kernel(config: KernelConfig | Mapping[str, Any] | None) -> KernelConfig:
    """Install the process-wide kernel policy.

    Quantization installs ``BaseQuantizeConfig.kernel``; inference installs the
    ``kernel=`` argument of ``GPTQModel.load``.  Calling it again replaces the
    policy, and ``None`` restores the defaults.
    """

    global _ACTIVE_CONFIG
    resolved = _coerce_kernel_config(config)
    with _CONFIG_LOCK:
        _ACTIVE_CONFIG = resolved
    _sync_rocm_llc(resolved.rocm_llc)
    return resolved


def _sync_rocm_llc(enabled: bool) -> None:
    """Apply or undo the ROCm LLC back-port to match the active policy.

    The back-port is applied at import, so an enabled policy leaves it alone.
    Only a policy that turns it *off* undoes it, and turning it back on restores
    the patch that this module removed rather than re-running a patch the import
    already made (which would reset a caller's own state).
    """

    global _ROCM_LLC_SUPPRESSED
    try:
        from .. import rocm_llc
    except Exception:  # pragma: no cover - torch/import failures are non-fatal
        return
    try:
        if enabled:
            if _ROCM_LLC_SUPPRESSED:
                rocm_llc.apply()
                _ROCM_LLC_SUPPRESSED = False
        else:
            rocm_llc.unapply()
            _ROCM_LLC_SUPPRESSED = True
    except Exception:  # pragma: no cover - the back-port never blocks a run
        pass


def get_kernel_config() -> KernelConfig:
    """Return the active kernel policy."""

    return _ACTIVE_CONFIG


def resolve_kernel_config(owner: Any = None) -> KernelConfig:
    """Return the policy that applies to ``owner`` (a quantize config).

    An explicit ``owner.kernel`` wins; otherwise the process-wide policy
    installed by a looper or ``GPTQModel.load`` applies.
    """

    if isinstance(owner, KernelConfig):
        return owner
    from .backend_config import BackendConfig, RuntimeConfig

    if isinstance(owner, (BackendConfig, RuntimeConfig)):
        return owner.to_kernel_config()
    kernel = getattr(owner, "kernel", None) if owner is not None else None
    if isinstance(kernel, KernelConfig):
        return kernel
    runtime = getattr(owner, "runtime", None) if owner is not None else None
    if runtime is not None:
        return runtime.to_kernel_config()
    return get_kernel_config()


def snapshot_kernel_config(owner: Any = None) -> KernelConfig:
    """Own a policy copy so later loads and caller mutation cannot retune it."""
    return replace(resolve_kernel_config(owner))


def reset_kernel_config() -> KernelConfig:
    """Restore the default kernel policy (used by tests)."""

    return configure_kernel(None)


__all__ = [
    "KernelConfig",
    "configure_kernel",
    "get_kernel_config",
    "resolve_kernel_config",
    "reset_kernel_config",
]
