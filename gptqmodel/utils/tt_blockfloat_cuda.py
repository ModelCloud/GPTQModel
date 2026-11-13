# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the Tenstorrent block-float CUDA emulation kernels.

``ttnn`` stores ``bf4_b`` / ``bf8_b`` weights as blocks of 16 values that share
one exponent.  Running such a checkpoint on a Tenstorrent part needs no
conversion, but validating one -- or benchmarking a model before the hardware
lands -- does.  The kernels in ``gptqmodel_ext/tt_blockfloat_cuda`` reproduce
that storage exactly, so a checkpoint behaves here the way it will on Tensix.
The bit layout is defined once, in ``hw/dtypes.md`` section 8.

This is the CUDA sibling of ``gptqmodel/utils/tt_blockfloat_hip.py``.  Both are
loaded lazily and expose the same surface; only the device differs.  Nothing is
imported or compiled until a block-float CUDA module actually runs a forward
pass, so a CPU-only or ROCm install never pays for it.

The kernels need ``mma.sync.m16n8k16`` with bf16 inputs, i.e. sm_80 or newer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .cpp import (
    TorchOpsJitExtension,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)
from .logger import setup_logger


log = setup_logger()

_OPS_NAME = "gptqmodel_blockfloat_cuda"
_NAMESPACE = "gptqmodel_blockfloat_cuda"

#: Values per shared exponent.  Fixed by the storage format; see ``hw/dtypes.md``
#: section 8.1.  Restated here rather than imported from ``tt_blockfloat`` so the
#: loader stays importable without pulling in the module tree.
BLOCKFLOAT_BLOCK = 16

#: ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` is sm_80+.
MIN_CAPABILITY = 80

_REQUIRED_OPS = (
    "blockfloat_linear",
    "blockfloat_dequant",
    "blockfloat_quantize",
    "blockfloat_decompose",
    "blockfloat_embedding",
    "blockfloat_selftest",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "tt_blockfloat_cuda"


def _sources() -> list[str]:
    root = _root()
    return [
        str(root / "tt_blockfloat_cuda_torch.cpp"),
        str(root / "tt_blockfloat_cuda.cu"),
    ]


def _cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
    )


# cuBLAS is resolved with dlopen inside the kernel source, so there is no
# link-time ``-lcublas`` here: pip CUDA wheels ship ``libcublas.so.13`` with no
# ``libcublas.so`` symlink and ``-lcublas`` would fail to link.
_EXTENSION = TorchOpsJitExtension(
    name=_OPS_NAME,
    namespace=_NAMESPACE,
    required_ops=_REQUIRED_OPS,
    sources=_sources,
    build_root_env="GPTQMODEL_BLOCKFLOAT_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("tt_blockfloat_cuda"),
    display_name="Tenstorrent block-float (CUDA)",
    extra_cuda_cflags=_cuda_cflags,
    force_rebuild_env="GPTQMODEL_BLOCKFLOAT_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)


def supported(device: torch.device | str | None = None) -> bool:
    """Whether the block-float CUDA kernels can run on ``device``."""

    if not torch.cuda.is_available():
        return False
    # ROCm reports a gfx-derived compute capability, which would pass the check
    # below and send the loader at a CUDA-only kernel.  The HIP path is separate.
    if torch.version.hip is not None:
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    capability = torch.cuda.get_device_capability(device)
    return capability[0] * 10 + capability[1] >= MIN_CAPABILITY


def prewarm() -> bool:
    """Compile/load the extension ahead of the first forward pass."""

    return _EXTENSION.load()


_READY: Optional[bool] = None


def ready() -> bool:
    """Whether the kernels are usable *and* already built, cached after the first probe.

    A missing compiler or an unsupported device must degrade to the dense
    reference path rather than fail a forward pass, so callers use this instead
    of letting :func:`_ops` raise.
    """

    global _READY
    if _READY is None:
        _READY = supported() and prewarm()
        if not _READY:
            log.warning(
                "Tenstorrent block-float CUDA kernels are unavailable, falling back to the dense reference path: %s",
                _EXTENSION.last_error_message() or "no sm_80+ CUDA device",
            )
    return _READY


def _ops():
    if not _EXTENSION.load():
        raise RuntimeError(
            f"the Tenstorrent block-float CUDA extension is unavailable: {_EXTENSION.last_error_message()}"
        )
    return torch.ops.gptqmodel_blockfloat_cuda


def _exponent_rows(exponents: torch.Tensor, k: int) -> int:
    per_row = k // BLOCKFLOAT_BLOCK
    if per_row <= 0 or exponents.numel() % per_row:
        raise ValueError(
            f"exponents with {exponents.numel()} values do not describe whole rows "
            f"of {k} values at {BLOCKFLOAT_BLOCK} values per exponent"
        )
    return exponents.numel() // per_row


def linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    exponents: torch.Tensor,
    bits: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``x @ dequant(codes, exponents).T + bias`` without a dense weight.

    ``x`` is ``[m, k]`` in float16, bfloat16 or float32 and the result has the
    same dtype and shape ``[m, n]``, with ``n`` taken from ``exponents``.
    """

    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")
    k = x.size(1)
    if k % BLOCKFLOAT_BLOCK:
        raise ValueError(f"K={k} must be a multiple of the block size {BLOCKFLOAT_BLOCK}")
    n = _exponent_rows(exponents, k)
    out = torch.empty((x.size(0), n), dtype=x.dtype, device=x.device)
    _ops().blockfloat_linear(x, codes, exponents, int(bits), bias, out)
    return out


def dequant(
    codes: torch.Tensor,
    exponents: torch.Tensor,
    bits: int,
    rows: int,
    k: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Expand packed codes plus exponents into a dense ``[rows, k]`` tensor."""

    out = torch.empty((rows, k), dtype=dtype, device=codes.device)
    _ops().blockfloat_dequant(codes, exponents, int(bits), int(rows), int(k), out)
    return out


def quantize(x: torch.Tensor, bits: int):
    """Quantise a float32 ``[rows, k]`` matrix; returns ``(codes, exponents)``."""

    if x.dim() != 2 or x.dtype != torch.float32:
        raise ValueError("x must be a 2D float32 tensor")
    rows, k = x.shape
    codes = torch.empty(
        (rows * k + 1) // 2 if bits == 4 else rows * k,
        dtype=torch.uint8,
        device=x.device,
    )
    exponents = torch.empty((rows, k // BLOCKFLOAT_BLOCK), dtype=torch.int8, device=x.device)
    _ops().blockfloat_quantize(x, int(bits), codes, exponents)
    return codes, exponents


def decompose(x: torch.Tensor, bits: int):
    """Split an on-grid float32 matrix into ``(codes, exponents, failed)``."""

    if x.dim() != 2 or x.dtype != torch.float32:
        raise ValueError("x must be a 2D float32 tensor")
    rows, k = x.shape
    codes = torch.empty(
        (rows * k + 1) // 2 if bits == 4 else rows * k,
        dtype=torch.uint8,
        device=x.device,
    )
    exponents = torch.empty((rows, k // BLOCKFLOAT_BLOCK), dtype=torch.int8, device=x.device)
    failed = torch.zeros(1, dtype=torch.int32, device=x.device)
    _ops().blockfloat_decompose(x, int(bits), codes, exponents, failed)
    return codes, exponents, failed


def embedding(
    ids: torch.Tensor,
    codes: torch.Tensor,
    exponents: torch.Tensor,
    bits: int,
    rows: int,
    k: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Gather the rows named by ``ids`` and dequantise only those."""

    out = torch.empty((ids.numel(), k), dtype=dtype, device=ids.device)
    _ops().blockfloat_embedding(ids, codes, exponents, int(bits), int(rows), int(k), out)
    return out


def selftest() -> int:
    """Run one kernel and return the sentinel it wrote, proving device code ran."""

    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    _ops().blockfloat_selftest(out)
    return int(out.item())
