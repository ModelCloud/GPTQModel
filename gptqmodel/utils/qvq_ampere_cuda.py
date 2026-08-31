# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exact continuous-window P32 WMMA kernel for A100-class sm_80 GPUs."""

from __future__ import annotations

from pathlib import Path

import torch

from ..quantization.qvq_rates import qvq_transition_bits
from .cpp import (
    TorchOpsJitExtension,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    is_nvcc_compatible,
)

_QVQ_AMPERE_NAME = "gptqmodel_qvq_ampere_ops"
_QVQ_AMPERE_NAMESPACE = "gptqmodel_qvq_ampere"
_SM80_FLAGS = (
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_80,code=compute_80",
)
_TORCH_NVCC_UNDEFINES = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source() -> list[str]:
    return [str(_project_root() / "gptqmodel_ext" / "qvq" / "qvq_ampere_cuda.cu")]


def _cuda_flags() -> list[str]:
    flags = [
        *_TORCH_NVCC_UNDEFINES,
        *default_jit_cuda_cflags(
            enable_bf16=False,
            include_lineinfo=True,
            include_nvcc_threads=True,
            nvcc_threads="2",
            include_split_compile=True,
            include_ptxas_optimizations=True,
            include_ptxas_verbosity=False,
            include_fatbin_compression=True,
            include_diag_suppress=True,
        ),
        *_SM80_FLAGS,
    ]
    if is_nvcc_compatible():
        flags.insert(0, "-static-global-template-stub=false")
    return flags


_QVQ_AMPERE_EXTENSION = TorchOpsJitExtension(
    name=_QVQ_AMPERE_NAME,
    namespace=_QVQ_AMPERE_NAMESPACE,
    required_ops=("p32_window",),
    sources=_source,
    build_root_env="GPTQMODEL_QVQ_AMPERE_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("qvq_ampere"),
    display_name="QVQ exact-P32 Ampere WMMA kernel",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=False),
    extra_cuda_cflags=_cuda_flags,
    force_rebuild_env="GPTQMODEL_QVQ_AMPERE_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    merge_visible_cuda_arch_override=False,
)


def _auto_split_count(*, in_features: int, out_features: int, k_tiles: int, sm_count: int) -> int:
    """Use measured Qwen3.8 splits, then a live-SM-derived fallback."""

    tuned_split = {
        (5120, 1024): 8,
        (5120, 6144): 8,
        (5120, 10240): 6,
        (5120, 12288): 5,
        (5120, 17408): 8,
        (6144, 5120): 8,
        (17408, 5120): 8,
    }.get((in_features, out_features))
    if tuned_split is not None:
        return min(tuned_split, k_tiles)
    n64_blocks = (out_features + 63) // 64
    target_blocks = max(1, sm_count * 2)
    split_count = max(1, (target_blocks + n64_blocks - 1) // n64_blocks)
    return min(split_count, 8, k_tiles)


def qvq_p32_window_ampere(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    bank_alt_id: int = 3,
    split_count: int = 0,
) -> torch.Tensor:
    """Run exact continuous-window P32 with FP16 WMMA and FP32 accumulation."""

    transition_bits = qvq_transition_bits(bits, vector_size=2)
    if transition_bits not in (4, 5, 6, 7):
        raise ValueError("QVQ P32 Ampere WMMA supports W2 through W3.5")
    if split_count == 0:
        if not input.is_cuda:
            raise ValueError("QVQ P32 Ampere input must be CUDA")
        properties = torch.cuda.get_device_properties(input.device)
        split_count = _auto_split_count(
            in_features=int(input.shape[1]),
            out_features=int(out_features),
            k_tiles=int(input.shape[1]) // 16,
            sm_count=properties.multi_processor_count,
        )
        # The scalar M<=4 kernel groups sixteen N16 tiles per CTA.  Wide
        # projections therefore need a fuller split wave than the WMMA
        # shape table (which was tuned for four-warp/N64 CTAs) to keep all
        # 124 Ampere SMs resident during the short decode.  The same policy
        # applies to M=1, whose scalar route was tuned first.
        if input.shape[0] <= 4 and input.shape[1] <= 6144:
            # Attention-out has enough N64 CTAs that 24 slices beat the
            # reduction overhead of a 32-way wave for the scalar M1-M4
            # route. Other short-K projections retain the fuller 32-way wave.
            small_m_split = (
                24
                if (int(input.shape[1]), int(out_features)) == (6144, 5120)
                else 32
            )
            split_count = min(small_m_split, int(input.shape[1]) // 16)
        elif input.shape[0] == 8 and input.shape[1] <= 6144:
            # The WMMA partial-row path uses four N16 tiles per CTA.  The
            # original shape table was tuned for M16 and leaves short-M8
            # projections under-filled, especially the small-N KV and
            # attention projections. Keep the measured split choices local
            # to M8; M5-M7 retain the conservative generic table.
            m8_split = {
                1024: 32,
                5120: 32,
                6144: 32,
                10240: 16,
                12288: 16,
                17408: 16,
            }.get(int(out_features))
            if m8_split is not None:
                split_count = min(m8_split, int(input.shape[1]) // 16)
        elif input.shape[0] == 16 and input.shape[1] <= 6144:
            # The full-row WMMA path also benefits from a fuller wave on
            # small-N projections. These choices are deliberately shape
            # specific: the wide Q and MLP projections already have enough
            # CTAs at their original measured splits.
            m16_split = {
                1024: 32,
                5120: 12,
                6144: 16,
                10240: 16,
            }.get(int(out_features))
            if m16_split is not None:
                split_count = min(m16_split, int(input.shape[1]) // 16)
        elif (int(input.shape[1]), int(out_features)) == (17408, 5120):
            # MLP-down is the only measured long-K Qwen shape. Its original
            # eight-way wave leaves too few CTAs per SM on the 124-SM A100;
            # a wider wave reduces the per-CTA K span and the split reducer
            # remains cheaper than the additional idle time. M1-M2 use the
            # scalar route and continue to benefit from a wide wave (128 for
            # M1, 96 for M2); M4+ retain the measured 32-way WMMA wave.
            long_k_split = 128 if input.shape[0] == 1 else 96 if input.shape[0] == 2 else 32
            split_count = min(long_k_split, int(input.shape[1]) // 16)
    return _QVQ_AMPERE_EXTENSION.op("p32_window")(
        input,
        trellis,
        levels,
        bank_ids,
        transition_bits,
        out_features,
        bank_alt_id,
        split_count,
    )


__all__ = ["qvq_p32_window_ampere"]
