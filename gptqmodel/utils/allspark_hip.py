# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the AllSpark HIP/RDNA4 W8A16 fused GEMM.

Port of ``gptqmodel/utils/allspark.py`` to AMD RDNA4 (gfx1200 / gfx1201).
The CUDA AllSpark kernel relies on cp.async, ldmatrix and ``mma.sync``
m16n8k16, none of which exist on RDNA4; this module builds the HIP port in
``gptqmodel_ext/allspark_hip`` instead.

Supported configuration: bits=8, fp16 or bf16 activations, and either
per-channel (``group_size=-1``, GPTQ 8-bit neutral zero point) or grouped
AWQ 8-bit with an arbitrary per-group zero point on the K axis.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .cpp import TorchOpsJitExtension, default_jit_cuda_cflags, default_torch_ops_build_root


_ALLSPARK_HIP_OPS_NAME = "gptqmodel_allspark_hip_ops"
_ALLSPARK_HIP_NAMESPACE = "gptqmodel_allspark_hip"

# The HIP repack pads N to a multiple of 128 so every launch block owns a full
# 128-column tile; the CUDA kernel only needed 32.
ALLSPARK_HIP_N_ALIGN = 128
ALLSPARK_HIP_K_ALIGN = 16
ALLSPARK_HIP_MAX_M = 16


def _root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "allspark_hip"


def _sources() -> list[str]:
    root = _root()
    return [
        str(root / "allspark_torch_hip.cpp"),
        str(root / "allspark_hip.cu"),
    ]


def _rocm_root() -> Path | None:
    """Locate a ROCm root that has the development layout hipcc needs.

    The pip ROCm wheels split the toolkit: torch's ``ROCM_HOME`` points at
    ``_rocm_sdk_core``, which carries the runtime soname (``libamdhip64.so.7``)
    but no unversioned ``libamdhip64.so`` symlink, and the device bitcode lives
    under a separate ``core-*`` package. A system install (``/opt/rocm``) has
    both, so prefer the first candidate that looks complete.
    """
    from torch.utils import cpp_extension

    candidates = [
        os.environ.get("ROCM_PATH"),
        os.environ.get("ROCM_HOME"),
        getattr(cpp_extension, "ROCM_HOME", None),
        "/opt/rocm",
    ]
    for candidate in candidates:
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
        include_ptxas_optimizations=True,
    )
    flags += [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        # torch's HIP build defines these; the kernel relies on ordinary
        # float <-> half/bfloat16 conversions.
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
    ]
    # The pip ROCm runtime wheel ships no device bitcode; point hipcc at the
    # full toolkit so the gfx1201 device library resolves.
    rocm_root = _rocm_root()
    if rocm_root is not None:
        flags.append(f"--rocm-path={rocm_root}")
    return flags


def _ldflags() -> list[str]:
    # Torch adds -lamdhip64 for every HIP extension, but ROCM_HOME (the pip
    # runtime wheel) only carries the versioned soname. /opt/rocm/lib has the
    # development symlink.
    rocm_root = _rocm_root()
    return [f"-L{rocm_root / 'lib'}"] if rocm_root is not None else []


_EXTENSION = TorchOpsJitExtension(
    name=_ALLSPARK_HIP_OPS_NAME,
    namespace=_ALLSPARK_HIP_NAMESPACE,
    required_ops=(
        "allspark_w8a16_gemm",
        "rearrange_kn_weight_as_n32k16_order",
        "allspark_hip_unpack",
        "allspark_hip_selftest",
    ),
    sources=_sources,
    build_root_env="GPTQMODEL_ALLSPARK_HIP_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("allspark_hip"),
    display_name="AllSpark W8A16 (HIP/RDNA4)",
    extra_cuda_cflags=_cuda_cflags,
    extra_ldflags=_ldflags,
    force_rebuild_env="GPTQMODEL_ALLSPARK_HIP_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)

_SELFTESTED = False


def prewarm_allspark_hip_extension() -> bool:
    global _SELFTESTED
    if not _EXTENSION.load():
        return False
    if not _SELFTESTED:
        # Proves the device code ran: the kernel stamps a sentinel word. An
        # empty kernel body (the `#if __CUDA_ARCH__` failure mode) would leave
        # it untouched.
        ref = torch.zeros(1, dtype=torch.float16, device='cuda')
        if not torch.ops.gptqmodel_allspark_hip.allspark_hip_selftest(ref):
            raise RuntimeError(
                "allspark_hip: kernel did not execute (sentinel missing) - "
                "device code was compiled empty or the wrong arch was targeted"
            )
        _SELFTESTED = True
    return True


def _device_arch(device: torch.device | None = None) -> str:
    props = torch.cuda.get_device_properties(device)
    return getattr(props, "gcnArchName", "") or ""


def allspark_hip_supported(device: torch.device | None = None) -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    arch = _device_arch(device)
    return arch.startswith("gfx1200") or arch.startswith("gfx1201")


def _n32(n: int) -> int:
    return (n + ALLSPARK_HIP_N_ALIGN - 1) // ALLSPARK_HIP_N_ALIGN * ALLSPARK_HIP_N_ALIGN


def allspark_hip_repack(qweight_codes: torch.Tensor, scales: torch.Tensor, n: int, k: int,
                        zeros: torch.Tensor | None = None):
    """Repack (K, N) uint8 codes into the RDNA4 WMMA fragment order.

    ``scales`` is ``(N,)`` for GPTQ 8-bit or ``(GroupCnt, N)`` for grouped AWQ,
    where ``GroupCnt == k // group_size``. ``zeros`` has the same shape as
    ``scales`` and holds the zero-point **codes** (AWQ qzeros), so the weight
    is ``(code - zero) * scale``.
    """
    n32 = _n32(n)
    packed = torch.empty((n32 // 4, k * 4), dtype=torch.uint8, device=qweight_codes.device)
    group_cnt = scales.numel() // n
    scales_2d = scales.reshape(group_cnt, n)
    scales_packed = torch.empty((group_cnt, n32), dtype=scales.dtype, device=scales.device)
    if zeros is not None:
        zeros_packed = torch.empty((group_cnt, n32), dtype=scales.dtype, device=scales.device)
        zeros_2d = zeros.reshape(group_cnt, n).to(scales.dtype)
    else:
        zeros_packed = None
        zeros_2d = None
    torch.ops.gptqmodel_allspark_hip.rearrange_kn_weight_as_n32k16_order(
        qweight_codes,
        scales_2d,
        zeros_2d,
        zeros_2d is not None,
        packed,
        scales_packed,
        zeros_packed,
        k,
        n,
        n32,
    )
    if zeros_packed is not None:
        zeros_packed = zeros_packed.reshape(-1)
    return packed.view(n32, k), scales_packed.reshape(-1), zeros_packed


def allspark_hip_repack_awq(qweight: torch.Tensor, qzeros: torch.Tensor,
                            scales: torch.Tensor, n: int, k: int,
                            group_size: int):
    """AWQ 8-bit (qweight, qzeros, scales) -> RDNA4 WMMA layout.

    AWQ packs along the output axis (``qweight`` is ``(K, ceil(N*8/32))`` and
    ``qzeros`` is ``(K/group, ceil(N*8/32))``), which is a different layout from
    GPTQ 8-bit. Both are unpacked here with the shared AWQ helpers, so the
    kernel only ever sees ``(K, N)`` codes plus per-group scale/zero rows.
    """
    from ..quantization.awq.utils.packing_utils import (
        AWQ_ORDER,
        reverse_awq_order,
        unpack_awq,
    )

    # bits=8 means ``_pack_contiguous`` was used, which is a plain little-endian
    # 4-codes-per-int32 stream; AWQ_ORDER only applies to the 4-bit layout.
    assert AWQ_ORDER is not None
    iweights, izeros = unpack_awq(qweight, qzeros, 8, n)
    iweights, izeros = reverse_awq_order(iweights, izeros, 8)
    codes = (iweights.to(torch.int64) & 0xFF).to(torch.uint8).contiguous()
    zeros = (izeros.to(torch.int64) & 0xFF).reshape(-1, n)

    scales_2d = scales.reshape(-1, n)
    if group_size > 0 and scales_2d.shape[0] == 1 and k != group_size:
        # A checkpoint can carry per-channel scales while the layer reports a
        # grouped size only when G == K/group_size, so this is a real mismatch.
        raise ValueError(
            f"AWQ scale rows ({scales_2d.shape[0]}) do not match K/group_size "
            f"({k // group_size})"
        )
    return allspark_hip_repack(codes, scales_2d, n, k, zeros=zeros)


def allspark_hip_gemm(a: torch.Tensor, b_qweight: torch.Tensor, b_scales: torch.Tensor,
                      n: int, b_qzeros: torch.Tensor | None = None, group_size: int = -1):
    """W8A16 GEMM against weights already in the HIP packed layout.

    ``b_qzeros`` is the packed zero-point buffer (AWQ) and ``group_size`` is the
    K grouping of ``b_scales`` (-1 for per-channel).
    """
    props = torch.cuda.get_device_properties(a.device)
    k = a.shape[-1]
    m = a.numel() // k
    if m <= ALLSPARK_HIP_MAX_M:
        return torch.ops.gptqmodel_allspark_hip.allspark_w8a16_gemm(
            a,
            b_qweight,
            b_scales,
            b_qzeros,
            n,
            group_size,
            props.multi_processor_count,
            props.major * 10 + props.minor,
            1024,
            b_qzeros is not None,
            True,
        )
    # Prefill fallback: unpack per call and hand off to torch.matmul. The dense
    # tensor is a local temporary; nothing is cached on the module.
    w = torch.ops.gptqmodel_allspark_hip.allspark_hip_unpack(
        b_qweight, b_scales, n, k, b_qzeros, group_size
    )
    return torch.matmul(a, w)


def allspark_hip_gemm_f32(a: torch.Tensor, b_qweight: torch.Tensor, b_scales: torch.Tensor,
                          n: int, b_qzeros: torch.Tensor | None = None, group_size: int = -1):
    """fp32-accumulator output (test/diagnostic path; M<=16)."""
    props = torch.cuda.get_device_properties(a.device)
    return torch.ops.gptqmodel_allspark_hip.allspark_hip_gemm_f32(
        a, b_qweight, b_scales, n, props.multi_processor_count, b_qzeros, group_size
    )


def allspark_hip_dtype_supported(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16)


__all__ = [
    "ALLSPARK_HIP_K_ALIGN",
    "ALLSPARK_HIP_MAX_M",
    "ALLSPARK_HIP_N_ALIGN",
    "allspark_hip_dtype_supported",
    "allspark_hip_gemm",
    "allspark_hip_gemm_f32",
    "allspark_hip_repack",
    "allspark_hip_repack_awq",
    "allspark_hip_supported",
    "prewarm_allspark_hip_extension",
]
