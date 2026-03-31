# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path

import torch

from ..quantization.awq.utils.module import try_import


log = logging.getLogger(__name__)

_AWQ_PYBIND_MODULE = "gptqmodel_awq_kernels"
_AWQ_OPS_NAME = "gptqmodel_awq_ops"
_AWQ_OPS_NAMESPACE = "gptqmodel_awq"


def _awq_sources() -> list[str]:
    root = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "awq"
    return [
        str(root / "torch_bind.cpp"),
        str(root / "quantization" / "gemm_cuda_gen.cu"),
        str(root / "quantization" / "gemv_cuda.cu"),
    ]


def _awq_include_paths() -> list[str]:
    root = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "awq"
    return [str(root)]


def _awq_extension_build_root() -> Path:
    override = os.getenv("GPTQMODEL_AWQ_BUILD_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "gptqmodel" / "torch_extensions" / "awq"


def _awq_extension_binary_paths(build_root: Path) -> list[Path]:
    return [
        build_root / f"{_AWQ_OPS_NAME}.so",
        build_root / f"{_AWQ_OPS_NAME}.pyd",
        build_root / f"{_AWQ_OPS_NAME}.dylib",
    ]


def _awq_force_rebuild_enabled() -> bool:
    return os.getenv("GPTQMODEL_AWQ_FORCE_REBUILD", "0") == "1"


def clear_awq_extension_cache() -> None:
    _load_awq_pybind_module.cache_clear()
    _load_awq_extension.cache_clear()
    build_root = _awq_extension_build_root()
    if build_root.exists():
        shutil.rmtree(build_root, ignore_errors=True)


@lru_cache(maxsize=1)
def _load_awq_pybind_module():
    if _awq_force_rebuild_enabled():
        return None, "AWQ pybind module disabled while GPTQMODEL_AWQ_FORCE_REBUILD=1."
    return try_import(_AWQ_PYBIND_MODULE)


def _awq_ops_available() -> bool:
    namespace = getattr(torch.ops, _AWQ_OPS_NAMESPACE, None)
    return namespace is not None and hasattr(namespace, "gemm_forward")


def _try_load_prebuilt_awq_extension(build_root: Path) -> bool:
    for library_path in _awq_extension_binary_paths(build_root):
        if not library_path.is_file():
            continue
        try:
            torch.ops.load_library(str(library_path))
            if _awq_ops_available():
                return True
        except Exception as exc:  # pragma: no cover - host/runtime dependent
            log.debug("AWQ: failed to load cached custom-op library %s: %s", library_path, exc)
    return False


def _awq_cxx11_abi_flag() -> int:
    return int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1))


@lru_cache(maxsize=1)
def _load_awq_extension() -> bool:
    if not torch.cuda.is_available():
        return False

    if _awq_ops_available():
        return True

    build_root = _awq_extension_build_root()
    build_root.mkdir(parents=True, exist_ok=True)

    if not _awq_force_rebuild_enabled() and _try_load_prebuilt_awq_extension(build_root):
        return True

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - runtime dependent
        log.debug("AWQ: torch cpp_extension unavailable: %s", exc)
        return False

    abi = _awq_cxx11_abi_flag()
    extra_cflags = [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
    ]
    extra_cuda_cflags = [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--threads",
        os.getenv("NVCC_THREADS", "2"),
        "--optimize=3",
        "-Xptxas",
        "-v,-O3,-dlcm=ca",
        "-lineinfo",
        "-Xfatbin",
        "-compress-all",
        "-diag-suppress=179,39,177",
    ]

    try:
        load(
            name=_AWQ_OPS_NAME,
            sources=_awq_sources(),
            build_directory=str(build_root),
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=_awq_include_paths(),
            is_python_module=False,
            verbose=False,
        )
        return _awq_ops_available()
    except Exception as exc:  # pragma: no cover - build depends on runtime
        log.debug("AWQ: custom-op build failed: %s", exc)
        return False


def awq_runtime_available() -> bool:
    module, _ = _load_awq_pybind_module()
    return module is not None or _load_awq_extension()


def awq_runtime_error() -> str:
    module, message = _load_awq_pybind_module()
    if module is not None:
        return ""
    if _load_awq_extension():
        return ""
    return message or "CUDA AWQ runtime unavailable."


def prewarm_awq_extension() -> bool:
    return awq_runtime_available()


def _awq_runtime_namespace():
    module, _ = _load_awq_pybind_module()
    if module is not None:
        return module
    if _load_awq_extension():
        return getattr(torch.ops, _AWQ_OPS_NAMESPACE)
    raise RuntimeError(awq_runtime_error())


def awq_gemm_forward(input, qweight, scales, qzeros, split_k_iters, fp32_accum: bool):
    runtime = _awq_runtime_namespace()
    if runtime is getattr(torch.ops, _AWQ_OPS_NAMESPACE, None):
        return runtime.gemm_forward(input, qweight, scales, qzeros, split_k_iters, fp32_accum)

    try:
        return runtime.gemm_forward_cuda(input, qweight, scales, qzeros, split_k_iters, fp32_accum)
    except TypeError:
        if fp32_accum:
            fp32_reduce_kernel = getattr(runtime, "gemm_forward_cuda_fp32_reduce", None)
            if fp32_reduce_kernel is not None:
                return fp32_reduce_kernel(input, qweight, scales, qzeros, split_k_iters)
        return runtime.gemm_forward_cuda(input, qweight, scales, qzeros, split_k_iters)


def awq_dequantize_weights(qweight, scales, qzeros, split_k_iters, thx, thy, dbg):
    runtime = _awq_runtime_namespace()
    if runtime is getattr(torch.ops, _AWQ_OPS_NAMESPACE, None):
        return runtime.dequantize_weights(qweight, scales, qzeros, split_k_iters, thx, thy, dbg)
    return runtime.dequantize_weights_cuda(qweight, scales, qzeros, split_k_iters, thx, thy, dbg)


def awq_gemmv2_forward(input, qweight, scales, qzeros, group_size, split_k_iters):
    runtime = _awq_runtime_namespace()
    if runtime is getattr(torch.ops, _AWQ_OPS_NAMESPACE, None):
        return runtime.gemmv2_forward(input, qweight, scales, qzeros, group_size, split_k_iters)
    return runtime.gemmv2_forward_cuda(input, qweight, scales, qzeros, group_size, split_k_iters)


def awq_gemv_forward(input, qweight, scales, qzeros, group_size):
    runtime = _awq_runtime_namespace()
    if runtime is getattr(torch.ops, _AWQ_OPS_NAMESPACE, None):
        return runtime.gemv_forward(input, qweight, scales, qzeros, group_size)
    return runtime.gemv_forward_cuda(input, qweight, scales, qzeros, group_size)


__all__ = [
    "awq_dequantize_weights",
    "awq_gemm_forward",
    "awq_gemmv2_forward",
    "awq_gemv_forward",
    "awq_runtime_available",
    "awq_runtime_error",
    "clear_awq_extension_cache",
    "prewarm_awq_extension",
]
