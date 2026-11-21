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

import torch


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
    "paro_cache_runtime_dtype",
    "paro_auto_cache_bf16_runtime_dtype",
    "paro_cache_rotation_dtype",
    "paro_auto_cache_bf16_rotation_dtype",
    "paro_cpu_avx512",
    "paro_triton_autotune",
    "paro_sym_ste_triton",
    "paro_affine_ste_triton",
    "paro_affine_ste_triton_backward",
    "paro_rotate_autotune",
    "paro_autotune_cache",
    "bitsandbytes_cpu_packed",
    "bitsandbytes_cpu_int8_dense",
    "bitsandbytes_triton",
    "disable_bitblas",
    "exl3_fp16_gemv",
    "exl3_fused_gemm",
    "gguf_native_quantized_loader",
    "gsq_disable_cuda_relaxation",
    "gsq_disable_hip_relaxation",
)

_OPTIONAL_BOOL_FIELDS = ("torch_triton_dequant", "gguf_cpp_native_matmul")

# A GGUF dequant target is a Torch execution device, not a hardware family:
# ROCm is addressed through Torch's ``cuda`` spelling, as everywhere else.
_GGUF_DEQUANT_DEVICE_TYPES = frozenset({"cpu", "cuda"})

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
    "gguf_dequant_chunk_bytes": 1,
    "paro_triton_autotune_warmup": 0,
    "paro_triton_autotune_iters": 1,
    "paro_triton_decode_max_rows": 1,
    "exl3_int8_gemv": 0,
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
    "exl3_tile_batch": 1,
    "max_packers": 1,
    "pool_intraop_threads": 0,
}

_MIN_FLOAT_FIELDS = {
    "paro_triton_autotune_margin": 0.0,
}

_OPTIONAL_CHOICE_FIELDS = {
    "paro_rotate_cta_m": frozenset({4, 8, 16}),
    "paro_rotate_row_pad": frozenset({0, 2}),
}

_AWQ_SCHEDULE_CHOICES = frozenset({"auto", "legacy"})

# ``rocblas`` pins rocBLAS; ``hipblaslt``/``cublaslt`` opt into torch's
# hipBLASLt default; ``""``/``0``/``off``/``default`` skip the pin. Mirrors
# ``utils.rocm.QUANT_BLAS_BACKEND_DISABLED | _QUANT_BLAS_BACKEND_CHOICES``.
_ROCM_QUANT_BLAS_CHOICES = frozenset({"rocblas", "hipblaslt", "cublaslt", "", "0", "off", "default"})


@dataclass
class KernelConfig:
    """Kernel, numerical-path and thread-count selection.

    Every field replaces one former ``GPTQMODEL_*`` / ``GPTQ_TORCH_*``
    environment variable.  Defaults are the values those variables defaulted
    to, so omitting the config preserves the previous default behaviour.

    Quantization-time fields (GPTQ block update, NPU Cholesky, GSQ relaxation,
    packer/worker counts, FloatX decode) are read while a run executes.
    Inference-time fields (Marlin, Torch, GGUF, Paro, bitsandbytes, Triton
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
    gguf_cpp_native_matmul: Optional[bool] = None
    gguf_cpp_threads: Optional[int] = None

    # GGUF checkpoint load strategy. These replace the retired
    # ``GPTQMODEL_INTERNAL_GGUF_*`` process switches: the same checkpoint must be
    # parsed by the same loader, on the same device, for the same public call
    # options.
    #
    # ``gguf_native_quantized_loader`` parses a checkpoint whose tensors declare
    # one quantized layout with the native GGUF reader. A dense import goes
    # through Transformers' own GGUF loader, which GPT-QModel never redirects.
    # ``gguf_dequant_device`` is the Torch device GPT-QModel's own GGUF
    # dequantization targets;
    # ``gguf_dequant_chunk_bytes`` bounds one NumPy-sign-only conversion chunk.
    gguf_native_quantized_loader: bool = True
    gguf_dequant_device: Optional[str] = None
    gguf_dequant_chunk_bytes: int = 256 * 1024 * 1024

    # Paro inference/optimization. Replace ``GPTQMODEL_PARO_*``.
    paro_cache_runtime_dtype: bool = False
    paro_auto_cache_bf16_runtime_dtype: bool = True
    paro_cache_rotation_dtype: bool = False
    paro_auto_cache_bf16_rotation_dtype: bool = True
    paro_cpu_avx512: bool = True
    paro_triton_autotune: bool = True
    paro_triton_autotune_warmup: int = 2
    paro_triton_autotune_iters: int = 4
    paro_triton_autotune_margin: float = 0.05
    paro_triton_decode_max_rows: int = 8
    paro_sym_ste_triton: bool = True
    paro_affine_ste_triton: bool = True
    paro_affine_ste_triton_backward: bool = False
    paro_rotate_autotune: bool = True
    paro_rotate_cta_m: Optional[int] = None
    paro_rotate_row_pad: Optional[int] = None
    paro_autotune_cache: bool = True

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
    exl3_tile_batch: Optional[int] = None
    # New CUDA variants change summation order. Existing integer checkpoints
    # use their original GEMM by default. INT8 uses 0 (off), 1 (residual),
    # or 2 (plain).
    exl3_fp16_gemv: bool = False
    exl3_fused_gemm: bool = False
    exl3_int8_gemv: int = 0
    exl3_host_rss_limit_bytes: int = 0
    exl3_cuda_allocation_limit_bytes: int = 0
    exl3_memory_telemetry_interval_batches: int = 0
    exl3_capture_batch_spool: Optional[str] = None
    exl3_capture_batch_checkpoint_interval: int = 1
    exl3_capture_frontier: Optional[str] = None

    # ROCm quantization BLAS. Replaces ``GPTQMODEL_ROCM_QUANT_BLAS``.
    rocm_quant_blas: Optional[str] = None

    # Thread counts. Replace ``GPTQMODEL_CPU_WORKERS``,
    # ``GPTQMODEL_CUDA_WORKERS`` and ``GPTQMODEL_MAX_PACKERS``. ``None`` keeps
    # the platform-derived default.
    cpu_workers: Optional[int] = None
    cuda_workers: Optional[int] = None
    max_packers: Optional[int] = None
    # Torch intra-op pool size for each device-pool worker. ``None`` splits the
    # process pool across the pool's workers; ``0`` keeps the inherited pool.
    pool_intraop_threads: Optional[int] = None

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

        device = self.gguf_dequant_device
        if device is not None:
            if not isinstance(device, (str, torch.device)):
                raise TypeError(
                    "KernelConfig.gguf_dequant_device must be a device string or None; "
                    f"got {type(device).__name__}"
                )
            text = str(device).strip()
            if not text:
                raise ValueError(
                    "KernelConfig.gguf_dequant_device must be a device string or None; got an empty string"
                )
            try:
                target = torch.device(text)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(
                    "KernelConfig.gguf_dequant_device must be a Torch device or None; "
                    f"got {device!r}: {exc}"
                ) from exc
            if target.type not in _GGUF_DEQUANT_DEVICE_TYPES:
                raise ValueError(
                    "KernelConfig.gguf_dequant_device must be 'cpu' or 'cuda[:<index>]' "
                    f"(ROCm is exposed through the cuda spelling); got {device!r}"
                )
            self.gguf_dequant_device = str(target)

        for name, minimum in _MIN_INT_FIELDS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"KernelConfig.{name} must be an integer; got {type(value).__name__}"
                )
            if value < minimum:
                raise ValueError(f"KernelConfig.{name} must be >= {minimum}; got {value}")

        if self.exl3_int8_gemv > 2:
            raise ValueError("KernelConfig.exl3_int8_gemv must be 0 (off), 1 (residual), or 2 (plain)")

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
        # Any other unknown spelling is rejected: a typo such as "hipblas_lt"
        # must not silently pin rocBLAS.
        if self.rocm_quant_blas is not None:
            if not isinstance(self.rocm_quant_blas, str):
                raise TypeError(
                    "KernelConfig.rocm_quant_blas should be a string or None; got "
                    f"{type(self.rocm_quant_blas).__name__}"
                )
            blas = self.rocm_quant_blas.strip().lower()
            if blas not in _ROCM_QUANT_BLAS_CHOICES:
                raise ValueError(
                    "KernelConfig.rocm_quant_blas should be None or one of "
                    f"{sorted(_ROCM_QUANT_BLAS_CHOICES)}; got {self.rocm_quant_blas!r}"
                )
            self.rocm_quant_blas = blas

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
    return resolved


def get_kernel_config() -> KernelConfig:
    """Return the active kernel policy."""

    return _ACTIVE_CONFIG


def resolve_kernel_config(owner: Any = None, *, use_process_default: bool = True) -> KernelConfig:
    """Return the policy that applies to ``owner`` (a quantize config).

    The policy an API boundary resolved for the config (typed backend plus its
    runtime or legacy kernel, kept in ``owner._resolved_kernel``) wins; then an
    explicit ``owner.kernel``, then ``owner.runtime``; otherwise the
    process-wide policy installed by a looper or ``GPTQModel.load`` applies.
    Run boundaries pass ``use_process_default=False`` so an unset run resets
    to fresh defaults rather than inheriting the previous run's process
    fallback.
    """

    if isinstance(owner, KernelConfig):
        return owner
    from .backend_config import BackendConfig, RuntimeConfig

    if isinstance(owner, (BackendConfig, RuntimeConfig)):
        return owner.to_kernel_config()
    resolved = getattr(owner, "_resolved_kernel", None) if owner is not None else None
    if isinstance(resolved, KernelConfig):
        return resolved
    kernel = getattr(owner, "kernel", None) if owner is not None else None
    if isinstance(kernel, KernelConfig):
        return kernel
    runtime = getattr(owner, "runtime", None) if owner is not None else None
    if runtime is not None:
        return runtime.to_kernel_config()
    return get_kernel_config() if use_process_default else KernelConfig()


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
