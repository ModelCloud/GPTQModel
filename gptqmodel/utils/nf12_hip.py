# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the NF12 HIP/RDNA4 codec and inference operators.

NF12 (NyoomFloat12) is Together AI's lossless 12-bit BF16 weight format
(Liberman, Fang, Zhang, Dao, Fu, 2026). The reference implementation and CUDA
kernels are AlpinDale's at https://github.com/dphnAI/nf12.

The vendored CUDA kernels in ``gptqmodel_ext/nf12`` use ``mma.sync`` m16n8k16,
``wgmma``, ``tcgen05`` and TMA, none of which exist on RDNA4. This module
builds the HIP port in ``gptqmodel_ext/nf12_hip``, which decodes the identical
NF12 byte stream to exact BF16 and hands it to rocBLAS through
``torch.matmul`` (SPEC.md section 6.5, the decode-then-GEMM route). It also
gathers embedding rows directly from packed storage.

Validated on ROCm SDK 10.1.0 / HIP 7.16.26385 (R9700, gfx1201); see the
vendored HIP README for the accompanying PyTorch build.

Nothing is shipped precompiled: the extension is always built from the sources
in ``gptqmodel_ext/nf12_hip`` for the visible gfx target.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .cpp import TorchOpsJitExtension, default_jit_cuda_cflags, default_torch_ops_build_root
from .env import env_flag


_OPS_NAME = "gptqmodel_nf12_hip_ops"
_NAMESPACE = "nf12_hip"

# RDNA4 (gfx12xx). gfx1250 shares the wave32 ISA subset used here.
SUPPORTED_ARCHS = ("gfx1200", "gfx1201", "gfx1250")


def _root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "nf12_hip"


_SOURCE_FILES = (
    "nf12_torch.cpp",
    "nf12_decode.hip",
    "nf12_embedding.hip",
    "nf12_rowmajor_embedding.hip",
    "nf12_encode.hip",
    "nf12_wmma.hip",
    "nf12_moe.hip",
    "nf12_pack_bf16t.hip",
    "nf12_format_hip.cuh",
    "nf12_validation.cuh",
)

# Portable format helpers shared with the CUDA kernels. The HIP build stages
# them beside the kernels so the include stays backend-local.
_SHARED_SOURCES = (
    ("../nf12/csrc/common/rowmajor_format.cuh", "rowmajor_format.cuh"),
    ("../nf12/csrc/common/rowmajor_pack.cuh", "rowmajor_pack.cuh"),
)


def _staging_root() -> Path:
    """Writable directory that receives a copy of the kernel sources."""

    override = os.environ.get("GPTQMODEL_NF12_HIP_BUILD_ROOT")
    base = Path(override).expanduser() if override else default_torch_ops_build_root("nf12_hip")
    return base / "_src"


def _sources() -> list[str]:
    """Stage the sources into the cache and return the staged paths.

    torch's ROCm loader hipifies out-of-tree sources by writing
    ``<name>_hip.{hip,cpp}`` *next to the original*, i.e. straight into
    ``gptqmodel_ext/nf12_hip``. That works in an editable checkout but fails
    outright for an installed package whose site-packages is read-only, and it
    mutates the shipped tree. Staging inside the writable cache keeps every
    generated file there, so the source tree is never written to.

    The staged directory is deliberately a sibling of the fingerprinted build
    directories (not their child): the cache fingerprint hashes the resolved
    source paths, so a path derived from that fingerprint would be circular.
    Every call refreshes the copies, and invalidation still comes from the
    content hashes the loader records.
    """

    stage = _staging_root()
    stage.mkdir(parents=True, exist_ok=True)
    originals = _root()
    for name in _SOURCE_FILES:
        source = originals / name
        if not source.exists():
            raise FileNotFoundError(f"NF12 HIP source is missing: {source}")
        (stage / name).write_bytes(source.read_bytes())
    for relative, staged_name in _SHARED_SOURCES:
        source = originals / relative
        if not source.exists():
            raise FileNotFoundError(f"NF12 HIP shared source is missing: {source}")
        (stage / staged_name).write_bytes(source.read_bytes())
    # Derived from _SOURCE_FILES so a new kernel cannot be staged but silently
    # left out of the build. Headers are staged above but are not sources.
    return [str(stage / name) for name in _SOURCE_FILES if name.endswith((".cpp", ".hip"))]


def _rocm_root() -> Path | None:
    """Locate a ROCm root with the development layout hipcc needs.

    Mirrors ``allspark_hip._rocm_root``: the pip SDK splits the runtime
    (``_rocm_sdk_core``) from the devel tree, and only the latter has the
    unversioned ``libamdhip64.so`` symlink and the gfx12 device bitcode.
    """
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
    if env_flag("GPTQMODEL_NF12_HIP_FORCE_DOT_FALLBACK"):
        # gfx1250 has no `dot12-insts`, so the kernel takes the portable FMA
        # path there. Force that path on gfx1200/gfx1201 to validate it against
        # the same oracle without gfx1250 hardware.
        flags.append("-DNF12_HIP_FORCE_NO_BF16_DOT2=1")
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
    required_ops=(
        "encode",
        "decode",
        "embedding",
        "rowmajor_pack",
        "rowmajor_embedding",
        "linear",
        "selftest",
        "arch_info",
    ),
    sources=_sources,
    build_root_env="GPTQMODEL_NF12_HIP_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("nf12_hip"),
    display_name="NF12 codec (HIP/RDNA4)",
    extra_cuda_cflags=_cuda_cflags,
    extra_ldflags=_ldflags,
    force_rebuild_env="GPTQMODEL_NF12_HIP_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)

_SELFTESTED = False
_EXTENSION_READY = False


class _OperatorCache:
    """Resolved dispatcher handles for the hot-path operator calls."""

    __slots__ = ("linear", "embedding", "rowmajor_pack", "rowmajor_embedding")

    def __init__(self) -> None:
        self.linear = None
        self.embedding = None
        self.rowmajor_pack = None
        self.rowmajor_embedding = None


_OPERATORS = _OperatorCache()


def _op(name: str):
    """Return a cached dispatcher handle, building the extension on first use."""

    global _EXTENSION_READY
    if not _EXTENSION_READY:
        if not _EXTENSION.load():
            raise RuntimeError(_EXTENSION.last_error_message() or "nf12_hip extension is unavailable")
        _EXTENSION_READY = True
    op = getattr(_OPERATORS, name)
    if op is None:
        # The dispatcher handle is immutable, so publishing it from concurrent
        # free-threaded callers is safe.
        op = getattr(torch.ops.nf12_hip, name)
        setattr(_OPERATORS, name, op)
    return op


def nf12_hip_supported(device: torch.device | None = None) -> bool:
    """Return whether this host is an RDNA4 device the HIP kernel targets."""

    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:  # noqa: BLE001 - device probe
        return False
    arch = (getattr(props, "gcnArchName", "") or "").split(":", 1)[0]
    return arch in SUPPORTED_ARCHS


def prewarm_nf12_hip_extension() -> bool:
    """Build and self-test the extension once per process."""

    global _EXTENSION_READY, _SELFTESTED
    if not nf12_hip_supported():
        return False
    if not _EXTENSION.load():
        return False
    if not _SELFTESTED:
        ref = torch.zeros(1, dtype=torch.float32, device="cuda")
        if not torch.ops.nf12_hip.selftest(ref):
            raise RuntimeError(
                "nf12_hip: kernel did not execute (sentinel missing) - "
                "device code compiled empty or the wrong gfx target was used"
            )
        _SELFTESTED = True
    _EXTENSION_READY = True
    if _OPERATORS.embedding is None:
        _OPERATORS.embedding = torch.ops.nf12_hip.embedding
        _OPERATORS.rowmajor_pack = torch.ops.nf12_hip.rowmajor_pack
        _OPERATORS.rowmajor_embedding = torch.ops.nf12_hip.rowmajor_embedding
    return True


def available(device: torch.device | None = None) -> bool:
    """Return whether the HIP decode kernel can run on this device."""

    global _EXTENSION_READY
    if not nf12_hip_supported(device):
        return False
    if _EXTENSION_READY:
        return True
    if not _EXTENSION.load():
        return False
    _EXTENSION_READY = True
    return True


def last_error() -> str:
    return _EXTENSION.last_error_message()


def encode(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode a BF16 ``[N, K]`` matrix on RDNA4.

    Byte-identical to :func:`gptqmodel.utils.nf12_codec.encode`: the same
    ``exp_base``, the same packed bytes and the same escape order. Runs the
    whole analyze/count/scan/write pipeline on the GPU instead of the
    pure-Torch CPU codec, which dominated quantization time before this port.
    """

    if not _EXTENSION.load():
        raise RuntimeError(_EXTENSION.last_error_message() or "nf12_hip extension is unavailable")
    packed, escapes, exp_base = torch.ops.nf12_hip.encode(weight)
    return packed, escapes, int(exp_base)


def decode(packed: torch.Tensor, escapes: torch.Tensor, exp_base: int) -> torch.Tensor:
    """Decode NF12 bytes to the exact BF16 ``[N, K]`` matrix on RDNA4."""

    if not _EXTENSION.load():
        raise RuntimeError(_EXTENSION.last_error_message() or "nf12_hip extension is unavailable")
    return torch.ops.nf12_hip.decode(packed, escapes, int(exp_base))


def decode_fp32(packed: torch.Tensor, escapes: torch.Tensor, exp_base: int) -> torch.Tensor:
    """Decode NF12 bytes straight to FP32 on RDNA4, with no BF16 intermediate.

    BF16 -> FP32 is an exact bit shift, so the result is bit-identical to
    ``decode(...).float()``.
    """

    if not _EXTENSION.load():
        raise RuntimeError(_EXTENSION.last_error_message() or "nf12_hip extension is unavailable")
    native = getattr(torch.ops.nf12_hip, "decode_fp32", None)
    if callable(native):
        return native(packed, escapes, int(exp_base))
    return torch.ops.nf12_hip.decode(packed, escapes, int(exp_base)).float()


def sub_fp32(packed: torch.Tensor, escapes: torch.Tensor, exp_base: int,
             wq: torch.Tensor) -> torch.Tensor:
    """``decode_fp32(...) - wq`` in one pass on RDNA4."""

    if not _EXTENSION.load():
        raise RuntimeError(_EXTENSION.last_error_message() or "nf12_hip extension is unavailable")
    native = getattr(torch.ops.nf12_hip, "sub_fp32", None)
    if callable(native):
        return native(packed, escapes, int(exp_base), wq.contiguous())
    return decode_fp32(packed, escapes, int(exp_base)) - wq


def linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    escapes: torch.Tensor,
    exp_base: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode-then-GEMM on RDNA4 (SPEC.md 6.5)."""

    return _op("linear")(x, packed, escapes, int(exp_base), bias)


def embedding(
    ids: torch.Tensor,
    packed: torch.Tensor,
    escapes: torch.Tensor,
    exp_base: int,
) -> torch.Tensor:
    """Gather exact BF16 rows from an NF12 table on RDNA4.

    ``ids`` may be int32 or int64, with any shape. Invalid IDs produce zero
    rows, matching the vendored CUDA embedding operator. Only requested rows
    are decoded, and no dense weight is retained between calls.
    """

    return _op("embedding")(ids, packed, escapes, int(exp_base))


def rowmajor_pack(
    packed: torch.Tensor,
    escapes: torch.Tensor,
    exp_base: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Convert canonical NF12 to the embedding-only row-major layout."""

    row_packed, row_escapes, original, collisions = _op("rowmajor_pack")(
        packed, escapes, int(exp_base)
    )
    return row_packed, row_escapes, int(original), int(collisions)


def rowmajor_embedding(
    ids: torch.Tensor,
    row_packed: torch.Tensor,
    row_escapes: torch.Tensor,
    exp_base: int,
) -> torch.Tensor:
    """Gather exact BF16 rows from the row-major embedding layout."""

    return _op("rowmajor_embedding")(ids, row_packed, row_escapes, int(exp_base))


__all__ = [
    "SUPPORTED_ARCHS",
    "available",
    "decode",
    "decode_fp32",
    "embedding",
    "encode",
    "last_error",
    "linear",
    "nf12_hip_supported",
    "prewarm_nf12_hip_extension",
    "rowmajor_embedding",
    "rowmajor_pack",
    "sub_fp32",
]
