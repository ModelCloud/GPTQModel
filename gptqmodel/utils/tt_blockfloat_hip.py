# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""HIP emulation of Tenstorrent ``bfloat8_b`` / ``bfloat4_b`` execution.

Tenstorrent stores these formats as 16-value blocks that share one exponent and
expands them inside the Tensix unpack stage, so a weight never exists as a dense
matrix in DRAM.  The kernels in ``gptqmodel_ext/tt_blockfloat_hip`` reproduce
that model on ROCm GPUs: they consume the checkpoint's packed ``codes`` and
int8 ``exponents`` unchanged, unpack nibbles/bytes in registers, and fuse the
dequantisation into the GEMM.  Expanded weights live only in shared memory for
one K tile.

The dequantised values are exact float32 grid points, so ``linear`` matches
``recompose_blockfloat`` followed by an fp32 GEMM.  Decode (``m == 1``) uses a
bandwidth-bound GEMV that streams each packed weight row once; prefill uses a
32x64x32 register-tiled GEMM.

This module is the ROCm counterpart of the Tenstorrent kernels in
``gptqmodel.utils.tt_int4_kernel`` and ``gptqmodel.utils.ttnn_runtime``.  It is
intended for testing and emulating the dtypes on non-Tenstorrent hardware, not
as a replacement for the device kernels.

Nothing is shipped precompiled: the extension is built from the sources in
``gptqmodel_ext/tt_blockfloat_hip`` for the visible gfx target.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch

from .cpp import (
    TorchOpsJitExtension,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


_OPS_NAME = "gptqmodel_tt_blockfloat_hip_ops"
_NAMESPACE = "tt_blockfloat_hip"

_SOURCE_FILES = (
    "tt_blockfloat_hip_torch.cpp",
    "tt_blockfloat_hip.hip",
    "tt_blockfloat_hip.h",
)

_DTYPE_CODES = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


def _root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "tt_blockfloat_hip"


def _staging_root() -> Path:
    override = os.environ.get("GPTQMODEL_TT_BLOCKFLOAT_HIP_BUILD_ROOT")
    base = Path(override).expanduser() if override else default_torch_ops_build_root("tt_blockfloat_hip")
    return base / "_src"


def _sources() -> list[str]:
    """Copy the sources into the writable cache and return the staged paths.

    torch's ROCm loader may write hipified siblings next to a source file, which
    is not writable for an installed package.  Staging keeps the shipped tree
    untouched; invalidation still comes from the content hashes the loader
    records.
    """

    stage = _staging_root()
    stage.mkdir(parents=True, exist_ok=True)
    originals = _root()
    for name in _SOURCE_FILES:
        source = originals / name
        if not source.exists():
            raise FileNotFoundError(f"tt_blockfloat_hip source is missing: {source}")
        (stage / name).write_bytes(source.read_bytes())
    return [str(stage / name) for name in _SOURCE_FILES if name.endswith((".cpp", ".hip"))]


def _rocm_root() -> Optional[Path]:
    """Locate a ROCm root with the development layout hipcc needs."""

    from torch.utils import cpp_extension

    for candidate in (
        os.environ.get("ROCM_PATH"),
        os.environ.get("ROCM_HOME"),
        getattr(cpp_extension, "ROCM_HOME", None),
        "/opt/rocm",
    ):
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "lib" / "libamdhip64.so").exists():
            return root
    return None


def _cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
    )
    flags += [
        # torch's HIP build hides the bf16 conversion operators; the kernel
        # uses ordinary casts.
        "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
    rocm_root = _rocm_root()
    if rocm_root is not None:
        flags.append(f"--rocm-path={rocm_root}")
    return flags


def _ldflags() -> list[str]:
    rocm_root = _rocm_root()
    return [f"-L{rocm_root / 'lib'}"] if rocm_root is not None else []


_EXTENSION = TorchOpsJitExtension(
    name=_OPS_NAME,
    namespace=_NAMESPACE,
    required_ops=("dequant", "quantize", "embedding", "decompose", "linear", "selftest"),
    sources=_sources,
    build_root_env="GPTQMODEL_TT_BLOCKFLOAT_HIP_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("tt_blockfloat_hip"),
    display_name="Tenstorrent bf8/bf4 (HIP)",
    extra_cuda_cflags=_cuda_cflags,
    extra_ldflags=_ldflags,
    force_rebuild_env="GPTQMODEL_TT_BLOCKFLOAT_HIP_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)

_SELFTESTED = False


def _device_supported(device: Optional[torch.device]) -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    return True


def tt_blockfloat_hip_supported(device: Optional[torch.device] = None) -> bool:
    """Whether this host/device can run the block-float HIP kernels.

    The kernels are portable HIP (no RDNA4-only builtins), so any ROCm GPU is
    accepted.  The check is cheap and does not build the extension.
    """

    return _device_supported(device)


def prewarm_tt_blockfloat_hip_extension() -> bool:
    """Build and self-test the extension once per process."""

    global _SELFTESTED
    if not tt_blockfloat_hip_supported():
        return False
    if not _EXTENSION.load():
        return False
    if not _SELFTESTED:
        probe = torch.zeros(1, device="cuda")
        if not torch.ops.tt_blockfloat_hip.selftest(probe):
            raise RuntimeError(
                "tt_blockfloat_hip: kernel did not execute (sentinel missing) - "
                "device code compiled empty or the wrong gfx target was used"
            )
        _SELFTESTED = True
    return True


def available(device: Optional[torch.device] = None) -> bool:
    """Whether the extension can run on ``device`` (builds it on first use)."""

    return tt_blockfloat_hip_supported(device) and _EXTENSION.load()


#: Code widths the HIP kernels implement.  ``bfloat2_b`` is a grid the host
#: quantiser can express, but it has no kernel, so it takes the torch path.
HIP_BITS = (4, 8)


def supports_bits(bits: int) -> bool:
    """Whether the HIP kernels implement this block-float width."""

    return int(bits) in HIP_BITS


def last_error() -> str:
    return _EXTENSION.last_error_message()


def _dtype_code(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_CODES[dtype]
    except KeyError as exc:  # pragma: no cover - caller error
        raise TypeError(
            f"tt_blockfloat_hip: unsupported dtype {dtype}; expected float32, float16 or bfloat16"
        ) from exc


def _require_loaded() -> None:
    if not _EXTENSION.load():
        raise RuntimeError(_EXTENSION.last_error_message() or "tt_blockfloat_hip extension is unavailable")


def dequantize(
    codes: torch.Tensor,
    exponents: torch.Tensor,
    bits: int,
    shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Expand packed codes into a dense ``shape`` tensor.

    The result is the exact grid the checkpoint describes; this is the explicit
    export path, not something the fused inference kernels use.
    """

    _require_loaded()
    if len(shape) != 2:
        raise ValueError(f"tt_blockfloat_hip: shape must be 2-D, got {tuple(shape)}")
    rows, cols = int(shape[0]), int(shape[1])
    return torch.ops.tt_blockfloat_hip.dequant(codes, exponents, int(bits), rows, cols, _dtype_code(dtype))


def quantize(weight: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise a float32 ``[rows, cols]`` matrix onto the block-float grid."""

    _require_loaded()
    return torch.ops.tt_blockfloat_hip.quantize(weight.to(torch.float32), int(bits))


def decompose(weight: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split an already-on-grid matrix into packed codes and exponents."""

    _require_loaded()
    return torch.ops.tt_blockfloat_hip.decompose(weight.to(torch.float32), int(bits))


def linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    exponents: torch.Tensor,
    bits: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``x @ dequant(codes).T + bias`` without materialising the weight."""

    _require_loaded()
    return torch.ops.tt_blockfloat_hip.linear(x, codes, exponents, int(bits), bias)


def embedding(
    input_ids: torch.Tensor,
    codes: torch.Tensor,
    exponents: torch.Tensor,
    bits: int,
    rows: int,
    cols: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Gather block-float rows by id, dequantising only the selected rows."""

    _require_loaded()
    return torch.ops.tt_blockfloat_hip.embedding(
        input_ids,
        codes,
        exponents,
        int(bits),
        int(rows),
        int(cols),
        _dtype_code(dtype),
    )


__all__ = [
    "available",
    "decompose",
    "dequantize",
    "embedding",
    "last_error",
    "linear",
    "prewarm_tt_blockfloat_hip_extension",
    "quantize",
    "supports_bits",
    "tt_blockfloat_hip_supported",
]
