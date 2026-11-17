# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Typed backend selection. Import the public API from :mod:`gptqmodel.backend`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import ClassVar, Optional, Union

from .backend import BACKEND, normalize_backend
from .kernel import KernelConfig, _coerce_kernel_config


class _KernelSettings:
    _kernel_fields: ClassVar[dict[str, str]] = {}

    def to_kernel_config(self) -> KernelConfig:
        """Build a fresh, validated adapter for the existing kernel consumers."""
        return KernelConfig(**self.kernel_options())

    def kernel_options(self) -> dict:
        return {
            self._kernel_fields.get(field.name, field.name): getattr(self, field.name)
            for field in fields(self)
        }

    def __post_init__(self):
        self.to_kernel_config()


@dataclass(frozen=True)
class BackendConfig(_KernelSettings, ABC):
    """Base for immutable backend-specific settings with a fixed identity.

    Subclasses declare ``backend`` as a class constant and only expose options
    consumed by that backend. The base itself cannot be instantiated.
    """

    @property
    @abstractmethod
    def backend(self) -> BACKEND:
        raise NotImplementedError


@dataclass(frozen=True)
class MarlinConfig(BackendConfig):
    backend: ClassVar[BACKEND] = BACKEND.MARLIN
    _kernel_fields: ClassVar[dict[str, str]] = {"fp32": "marlin_fp32"}
    fp32: bool = True


@dataclass(frozen=True)
class TorchConfig(BackendConfig):
    backend: ClassVar[BACKEND] = BACKEND.GPTQ_TORCH
    _kernel_fields: ClassVar[dict[str, str]] = {
        name: f"torch_{name}"
        for name in ("streaming", "stream_tile", "triton_dequant", "rocm_gemv", "rocm_gemm")
    }
    streaming: bool = False
    stream_tile: int = 512
    triton_dequant: Optional[bool] = None
    rocm_gemv: bool = True
    rocm_gemm: bool = True


@dataclass(frozen=True)
class AwqTorchConfig(BackendConfig):
    backend: ClassVar[BACKEND] = BACKEND.AWQ_TORCH


@dataclass(frozen=True)
class AwqGemmConfig(BackendConfig):
    backend: ClassVar[BACKEND] = BACKEND.AWQ_GEMM
    _kernel_fields: ClassVar[dict[str, str]] = {"fused_splitk_reduce": "awq_fused_splitk_reduce"}
    fp32_accum: bool = True
    fused_splitk_reduce: bool = True


@dataclass(frozen=True)
class AwqTritonConfig(BackendConfig):
    backend: ClassVar[BACKEND] = BACKEND.AWQ_GEMM_TRITON
    _kernel_fields: ClassVar[dict[str, str]] = {"schedule": "awq_triton_schedule"}
    fp32_accum: bool = True
    schedule: str = "auto"


@dataclass(frozen=True)
class SwordfishConfig(BackendConfig):
    backend: ClassVar[BACKEND] = BACKEND.SWORDFISH
    _kernel_fields: ClassVar[dict[str, str]] = {
        "deterministic": "swordfish_deterministic", "dense_m": "swordfish_dense_m",
    }
    deterministic: bool = False
    dense_m: int = 0


@dataclass(frozen=True)
class GGUFTorchConfig(BackendConfig):
    backend: ClassVar[BACKEND] = BACKEND.GGUF_TORCH
    _kernel_fields: ClassVar[dict[str, str]] = {
        name: f"gguf_{name}" for name in (
            "fused_cuda_max_rows", "fused_cuda_min_matrix_elements", "fused_cpu_max_rows",
            "fused_cpu_min_matrix_elements", "fused_chunk_blocks", "fused_chunk_bytes",
        )
    }
    fused_cuda_max_rows: int = 32
    fused_cuda_min_matrix_elements: int = 8_388_608
    fused_cpu_max_rows: int = 64
    fused_cpu_min_matrix_elements: int = 0
    fused_chunk_blocks: int = 0
    fused_chunk_bytes: int = 32 * 1024 * 1024


@dataclass(frozen=True)
class GGUFCPPConfig(GGUFTorchConfig):
    backend: ClassVar[BACKEND] = BACKEND.GGUF_CPP_CPU
    _kernel_fields: ClassVar[dict[str, str]] = {
        **GGUFTorchConfig._kernel_fields,
        "native_matmul": "gguf_cpp_native_matmul", "threads": "gguf_cpp_threads",
    }
    native_matmul: Optional[bool] = None
    threads: Optional[int] = None


@dataclass(frozen=True)
class GGUFCUDAConfig(BackendConfig):
    """ggml-cuda runs every GGUF matmul; the Torch fused-forward gates do not apply."""

    backend: ClassVar[BACKEND] = BACKEND.GGUF_CPP_CUDA


@dataclass(frozen=True)
class ParoQuantConfig(BackendConfig):
    backend: ClassVar[BACKEND] = BACKEND.PAROQUANT_CUDA
    _kernel_fields: ClassVar[dict[str, str]] = {
        "fused_splitk_reduce": "awq_fused_splitk_reduce",
        **{
        name: f"paroquant_{name}" for name in (
            "cache_runtime_dtype", "auto_cache_bf16_runtime_dtype", "cache_rotation_dtype",
            "auto_cache_bf16_rotation_dtype", "cpu_avx512",
        )
        },
    }
    fp32_accum: bool = True
    fused_splitk_reduce: bool = True
    cache_runtime_dtype: bool = False
    auto_cache_bf16_runtime_dtype: bool = True
    cache_rotation_dtype: bool = False
    auto_cache_bf16_rotation_dtype: bool = True
    cpu_avx512: bool = True


@dataclass(frozen=True)
class ParoQuantTritonConfig(ParoQuantConfig):
    backend: ClassVar[BACKEND] = BACKEND.PAROQUANT_TRITON
    _kernel_fields: ClassVar[dict[str, str]] = {
        **ParoQuantConfig._kernel_fields,
        **{name: f"paroquant_triton_{name}" for name in (
            "autotune", "autotune_warmup", "autotune_iters", "autotune_margin", "decode_max_rows",
        )},
    }
    autotune: bool = True
    autotune_warmup: int = 2
    autotune_iters: int = 4
    autotune_margin: float = 0.05
    decode_max_rows: int = 8


@dataclass(frozen=True)
class BitsAndBytesConfig(BackendConfig):
    backend: ClassVar[BACKEND] = BACKEND.BITSANDBYTES
    _kernel_fields: ClassVar[dict[str, str]] = {
        "cpu_packed": "bitsandbytes_cpu_packed", "cpu_int8_dense": "bitsandbytes_cpu_int8_dense",
        "triton": "bitsandbytes_triton",
    }
    cpu_packed: bool = False
    cpu_int8_dense: bool = True
    triton: bool = False


@dataclass(frozen=True)
class TTBlockFloatConfig(BackendConfig):
    backend: ClassVar[BACKEND] = BACKEND.TT_BLOCKFLOAT
    _kernel_fields: ClassVar[dict[str, str]] = {"cublas": "blockfloat_cublas"}
    cublas: bool = True


@dataclass(frozen=True)
class RuntimeConfig(_KernelSettings):
    """Shared quantizer, worker and host controls; does not select a backend."""

    cpu_workers: Optional[int] = None
    cuda_workers: Optional[int] = None
    max_packers: Optional[int] = None
    pool_intraop_threads: Optional[int] = None
    gptq_fused_block_update: bool = True
    npu_cholesky_block_size: int = 64
    floatx_cpu_threads: Optional[int] = None
    floatx_cpu_native_fp8: bool = False
    floatx_cpu_simd_level: int = 2
    cpu_bf16x3_matmul: bool = True
    disable_bitblas: bool = False
    gsq_disable_cuda_relaxation: bool = False
    gsq_disable_hip_relaxation: bool = False
    rocm_quant_blas: Optional[str] = None
    rocm_llc: bool = True
    paroquant_sym_ste_triton: bool = True
    paroquant_affine_ste_triton: bool = True
    paroquant_affine_ste_triton_backward: bool = False
    paroquant_rotate_autotune: bool = True
    paroquant_rotate_cta_m: Optional[int] = None
    paroquant_rotate_row_pad: Optional[int] = None
    paroquant_autotune_cache: bool = True
    exl3_tile_batch: int = 256
    exl3_host_rss_limit_bytes: int = 0
    exl3_cuda_allocation_limit_bytes: int = 0
    exl3_memory_telemetry_interval_batches: int = 0
    exl3_capture_batch_spool: Optional[str] = None
    exl3_capture_batch_checkpoint_interval: int = 1
    exl3_capture_frontier: Optional[str] = None


Backend = Union[str, BACKEND, BackendConfig]


def resolve_backend_config(
    backend: Backend | None,
    kernel=None,
    runtime: RuntimeConfig | None = None,
) -> tuple[BACKEND | None, KernelConfig | None]:
    """Resolve once at the API boundary, retaining settings separately.

    The explicit legacy policy and a typed selection are mutually exclusive.
    Internal forwarding uses the returned enum/policy pair, so redirects do
    not reinterpret or discard a typed selection.
    """
    if runtime is not None and not isinstance(runtime, RuntimeConfig):
        raise TypeError("runtime must be a RuntimeConfig or None")
    if isinstance(backend, BackendConfig):
        if kernel is not None:
            raise ValueError("backend=BackendConfig(...) and legacy kernel= cannot be supplied together")
        options = runtime.kernel_options() if runtime is not None else {}
        options.update(backend.kernel_options())
        return normalize_backend(backend.backend), KernelConfig(**options)
    resolved = normalize_backend(backend)
    if runtime is None:
        if kernel is None:
            return resolved, None
        # The legacy kernel object is caller-owned and mutable. Resolution
        # returns an independent value snapshot, so an in-place edit after the
        # API boundary cannot retune execution (#4615).
        return resolved, replace(_coerce_kernel_config(kernel))
    if kernel is not None:
        raise ValueError("runtime= and legacy kernel= cannot be supplied together")
    return resolved, runtime.to_kernel_config()


def resolve_quantize_backend(backend, qcfg, kernel=None, runtime=None, *, inherit=False):
    """Resolve a quantization config's backend and policy at an API boundary.

    ``qcfg.kernel`` and ``qcfg.runtime`` hold only caller-supplied settings:
    those given on the config itself and the loader's ``kernel=`` /
    ``runtime=`` arguments, which replace the field of the same kind. The
    policy derived from them and a typed backend is kept in
    ``qcfg._resolved_kernel``, never written back to ``qcfg.kernel``, so a
    later call cannot mistake it for a caller-supplied legacy policy.

    A legacy kernel conflicts with a typed backend or a runtime wherever each
    was supplied (argument or config field). ``inherit=True`` lets
    ``quantize(backend=AUTO)`` keep the dense load's typed or explicit
    enum/string selection; every other call replaces it.
    """
    if qcfg is None:
        return resolve_backend_config(backend, kernel, runtime)
    if inherit and not isinstance(backend, BackendConfig) and normalize_backend(backend) == BACKEND.AUTO:
        # Typed choices carry their options in `_backend_config`; an explicit
        # enum/string load choice is recorded in `_load_backend`. Both must
        # survive into quantize(backend=AUTO).
        inherited = getattr(qcfg, "_backend_config", None)
        if inherited is None:
            inherited = getattr(qcfg, "_load_backend", None)
        backend = inherited or backend
    if kernel is not None:
        kernel = _coerce_kernel_config(kernel)
    if runtime is not None and not isinstance(runtime, RuntimeConfig):
        raise TypeError("runtime must be a RuntimeConfig or None")
    effective_kernel = kernel if kernel is not None else qcfg.kernel
    effective_runtime = runtime if runtime is not None else qcfg.runtime
    if effective_kernel is not None:
        kernel_source = "kernel=" if kernel is not None else "QuantizeConfig.kernel"
        if isinstance(backend, BackendConfig):
            raise ValueError(
                f"backend=BackendConfig(...) and legacy {kernel_source} cannot be supplied together; "
                "move shared settings to RuntimeConfig"
            )
        if effective_runtime is not None:
            runtime_source = "runtime=" if runtime is not None else "QuantizeConfig.runtime"
            raise ValueError(
                f"{runtime_source} and legacy {kernel_source} cannot be supplied together; "
                "use one policy source"
            )
    selected, policy = resolve_backend_config(backend, effective_kernel, effective_runtime)
    # Record caller settings only once they validated together.
    if kernel is not None:
        qcfg.kernel = kernel
    if runtime is not None:
        qcfg.runtime = runtime
    if isinstance(backend, BackendConfig):
        qcfg._backend_config = backend
        qcfg._load_backend = None
    else:
        qcfg._backend_config = None
        if not inherit:
            # Record the loader's explicit enum/string choice. A quantize()
            # argument is a one-shot override, so it does not replace it.
            resolved_backend = normalize_backend(backend)
            qcfg._load_backend = (
                None if resolved_backend in (None, BACKEND.AUTO) else resolved_backend
            )
    qcfg._resolved_kernel = policy
    return selected, policy


__all__ = [
    "BackendConfig", "MarlinConfig", "TorchConfig", "AwqTorchConfig", "AwqGemmConfig",
    "AwqTritonConfig", "SwordfishConfig", "GGUFTorchConfig", "GGUFCPPConfig", "GGUFCUDAConfig",
    "ParoQuantConfig", "ParoQuantTritonConfig", "BitsAndBytesConfig", "TTBlockFloatConfig",
    "RuntimeConfig",
]
