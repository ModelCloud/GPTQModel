# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the AllSpark Ampere W8A16 fused GEMM (bits=8, group_size=-1).

Kernel sources are vendored from vLLM (vllm-project/vllm#12931, Apache-2.0) and
live in ``gptqmodel_ext/allspark``.  Upstream gates the kernel to Ampere
(compute capability 80..89); the CUDA code itself only requires
``__CUDA_ARCH__ >= 800`` and is numerically correct on sm_120, so the runtime
check below allows 80..120. Select it explicitly with ``BACKEND.ALLSPARK``;
the backend routes GPTQ and AWQ checkpoints to their matching linear module.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .cpp import (
    TorchOpsJitExtension,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


_ALLSPARK_OPS_NAME = "gptqmodel_allspark_ops"
_ALLSPARK_NAMESPACE = "gptqmodel_allspark"

# Kernel requires K % 16 == 0 and N % 16 == 0 (see allspark_repack.cu) and the
# GEMM path pads N up to a multiple of 32.
ALLSPARK_N_ALIGN = 16
ALLSPARK_K_ALIGN = 16
ALLSPARK_M_CUBLAS_THRESHOLD = 1024
ALLSPARK_MIN_CAPABILITY = 80
ALLSPARK_MAX_CAPABILITY = 120


def _allspark_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "allspark"


def _allspark_sources() -> list[str]:
    root = _allspark_root()
    return [
        str(root / "allspark_torch.cpp"),
        str(root / "allspark_qgemm_w8a16.cu"),
        str(root / "allspark_repack.cu"),
    ]


def _allspark_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
    )
    # Torch's cpp_extension defines these to keep its own headers strict; the
    # AllSpark sources rely on implicit float -> half/bfloat16 conversions.
    flags += [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    return flags


_ALLSPARK_EXTENSION = TorchOpsJitExtension(
    name=_ALLSPARK_OPS_NAME,
    namespace=_ALLSPARK_NAMESPACE,
    required_ops=("allspark_w8a16_gemm", "rearrange_kn_weight_as_n32k16_order"),
    sources=_allspark_sources,
    build_root_env="GPTQMODEL_ALLSPARK_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("allspark"),
    display_name="AllSpark W8A16",
    extra_cuda_cflags=_allspark_cuda_cflags,
    extra_ldflags=lambda: ["-lcublas"],
    force_rebuild_env="GPTQMODEL_ALLSPARK_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)


def prewarm_allspark_extension() -> bool:
    return _ALLSPARK_EXTENSION.load()


def allspark_supported(device: torch.device | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    # ROCm reports a fake compute capability (gfx1201 -> 12.0), which would fall
    # inside the Ampere..Blackwell window and send the loader to this CUDA-only
    # kernel. The RDNA4 path lives in gptqmodel/utils/allspark_hip.py.
    if torch.version.hip is not None:
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    capability = torch.cuda.get_device_capability(device)
    return ALLSPARK_MIN_CAPABILITY <= capability[0] * 10 + capability[1] <= ALLSPARK_MAX_CAPABILITY


def allspark_repack(qweight: torch.Tensor, scales: torch.Tensor, n: int, k: int,
                    zeros: torch.Tensor | None = None):
    """Rearrange a (K, N) uint8 weight into the kernel's N32K16 order.

    ``zeros`` is the per-output-channel zero-point **code** (AWQ). The kernel
    subtracts it after its bias-128 conversion, i.e. it computes
    ``(code - 128) - zero_buffer``, so the caller passes ``zero_awq - 128``.
    Passing a raw AWQ zero would shift every weight by 128 LSBs.
    """
    n_32align = (n + 32 - 1) // 32 * 32
    qweight_reordered = torch.empty(
        (n_32align // 4, k * 4), dtype=torch.uint8, device=qweight.device
    )
    scales_reordered = torch.empty(
        (n_32align,), dtype=scales.dtype, device=scales.device
    )
    if zeros is not None:
        zeros_2d = zeros.reshape(n).to(scales.dtype)
        zeros_reordered = torch.empty(
            (n_32align,), dtype=scales.dtype, device=scales.device
        )
    else:
        zeros_2d = None
        zeros_reordered = None
    torch.ops.gptqmodel_allspark.rearrange_kn_weight_as_n32k16_order(
        qweight,
        scales,
        zeros_2d,
        zeros_2d is not None,
        qweight_reordered,
        scales_reordered,
        zeros_reordered,
        k,
        n,
        n_32align,
    )
    return qweight_reordered.view(n_32align, k), scales_reordered, zeros_reordered


def allspark_repack_awq(qweight: torch.Tensor, qzeros: torch.Tensor,
                        scales: torch.Tensor, n: int, k: int,
                        group_size: int):
    """AWQ 8-bit (qweight, qzeros, scales) -> N32K16 layout (CUDA/Ampere).

    AWQ packs along the output axis (``qweight`` is ``(K, ceil(N*8/32))`` and
    ``qzeros`` is ``(K/group, ceil(N*8/32))``), which is a different layout from
    GPTQ 8-bit. Both are unpacked here with the shared AWQ helpers, so the
    kernel only ever sees ``(K, N)`` codes plus one scale/zero row per N.

    The kernel computes ``(code - 128) - B_zero``, so the stored buffer is
    ``zero_awq - 128`` converted to the scale dtype. ``group_size`` must be -1
    (one group over all of K); ``k`` is the equivalent normalized spelling.
    """
    from ..quantization.awq.utils.packing_utils import (
        reverse_awq_order,
        unpack_awq,
    )

    if group_size not in (-1, k):
        raise ValueError(f"AWQ AllSpark supports group_size=-1 (or K) only, got {group_size}")
    iweights, izeros = unpack_awq(qweight, qzeros, 8, n)
    iweights, izeros = reverse_awq_order(iweights, izeros, 8)
    codes = (iweights.to(torch.int64) & 0xFF).to(torch.uint8).contiguous()
    zeros = (izeros.to(torch.int64) & 0xFF).reshape(-1, n)
    scales_1d = scales.reshape(-1, n)
    if zeros.shape[0] != 1 or scales_1d.shape[0] != 1:
        raise ValueError(
            "AWQ AllSpark needs one scale/zero row per output channel, got "
            f"{scales_1d.shape[0]} scale rows and {zeros.shape[0]} zero rows"
        )
    # The CUDA kernel subtracts the zero AFTER its bias-128 dequant, so it wants
    # zero_awq - 128 rather than the raw code.
    zero_adjusted = (zeros.reshape(n).to(torch.float32) - 128.0).to(scales.dtype)
    return allspark_repack(codes, scales_1d.reshape(n), n, k, zeros=zero_adjusted)


def allspark_gemm(a: torch.Tensor, b_qweight: torch.Tensor, b_scales: torch.Tensor, n: int,
                  b_qzeros: torch.Tensor | None = None):
    """W8A16 GEMM against weights already in N32K16 order.

    ``b_qzeros`` is the repacked ``zero_awq - 128`` buffer (see
    :func:`allspark_repack_awq`); ``None`` keeps the GPTQ symmetric path.
    """
    device = a.device
    props = torch.cuda.get_device_properties(device)
    sm_version = props.major * 10 + props.minor
    return torch.ops.gptqmodel_allspark.allspark_w8a16_gemm(
        a,
        b_qweight,
        b_scales,
        b_qzeros,
        n,
        -1,
        props.multi_processor_count,
        sm_version,
        ALLSPARK_M_CUBLAS_THRESHOLD,
        b_qzeros is not None,
        True,
    )
