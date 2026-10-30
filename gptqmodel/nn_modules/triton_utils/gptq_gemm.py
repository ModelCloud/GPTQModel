"""Fused GPTQ GEMM for small-batch (M in 2..16) forward passes.

``TorchLinear`` normally dequantizes the whole weight matrix and then calls
``torch.matmul``. That materializes an ``in_features x out_features``
activation-dtype tensor and then streams it straight back in: for small M the
extra write + read of that buffer costs more than the matmul itself. This
kernel dequantizes the low-bit weights *inside* the GEMM, so the packed weights
are read once and no dequantized weight buffer is ever written.

The single-token (M == 1) case is handled by :mod:`gptq_gemm`'s sibling
``gptq_gemv``; this module covers the small-M band used by speculative decoding
and small-batch serving, where the whole batch fits in one ``BLOCK_M`` tile and
the weights are therefore dequantized exactly once (same dequant work as the
split path, minus the buffer traffic).

Supports 2/4/8-bit GPTQ weights packed into ``int32`` words (the 10-1-10-1-10
3-bit layout is not handled here and falls back).

Measured on gfx950 (MI350X/MI355X) against the dequantize + matmul path with
``BLOCK_M=16, BLOCK_N=16, BLOCK_K=256, num_warps=4, num_stages=2`` (module
forward, M=2..16):

    K=896  N=128   : 1.44x-1.49x      K=4096 N=4096  : 1.46x
    K=896  N=896   : 1.43x-1.47x      K=4096 N=11008 : 1.24x-1.28x
    K=896  N=4864  : 1.45x-1.46x      K=5120 N=5120  : 1.05x-1.07x
    K=4864 N=896   : 1.42x-1.45x      K=5120 N=12288 : 1.17x-1.20x
    K=3584 N=3584  : 1.61x-1.65x      K=5120 N=13824 : 1.08x
    K=3584 N=18944 : 1.26x-1.28x

End to end on Qwen2.5-0.5B-Instruct (4-bit, M=2..16) the forward is 1.51x-1.54x
faster; M=1 and M>``SMALL_M_MAX`` are unchanged.

``BLOCK_N=16`` is deliberately small: it maximizes the grid size so the
memory-bound dequant side saturates the device (wider ``BLOCK_N`` starves
occupancy and was slower on every shape tested).

``MAX_IN_FEATURES`` bounds the shape gate. Past roughly K=6144 the kernel's
serial K-loop becomes latency-bound (there is not enough N-parallelism left to
hide it) and the split path wins again -- e.g. 0.68x at K=11008/N=4096 and
0.81x at K=8192/N=4096. The crossover is not monotonic in K or N, so the gate
uses the conservative bound that was measured to win on every shape at or below
it (1.05x-1.65x across K in 896..5120).

Launch constants are per architecture class. The ``cdna4`` entry is exactly the
gfx950 configuration measured above and must not drift.

``rdna4`` (gfx1200/gfx1201/gfx1250) uses a wider ``(block_m, block_n) = (32,
32)`` tile, measured on an AMD Radeon AI PRO R9700 (``gfx1201``, HIP
7.16.26385, torch 2.16.0.dev20261029+rocm10.1). The sweep covered K in
{896, 2048, 3584, 4096, 5120, 6144} x N in {128, 1280, 2048, 3584, 4096, 5120,
8192, 11008}, M in {2, 8, 16}, ``num_warps`` in {2, 4, 8} and (block_m,
block_n) in {16, 32}^2: (32, 32) had the lowest worst-case regret (1.138x,
mean 1.031x) versus (16, 16)'s 1.365x / 1.116x, and is 1.19x faster at
K=N=4096 (71us vs 87us) and 1.17x at K=N=6144. ``tl.dot`` lowers to
``v_wmma_f32_16x16x16_f16`` on gfx1201 (read from the kernel's ``amdgcn``
assembly), so the win is real matrix-core throughput, not an FMA fallback.
``num_warps=8`` measured 1.4x-1.9x slower.

The activation dtype is a separate gate, applied by ``TorchLinear`` (see
``gptqmodel.utils.arch.gemm_fused_dtypes``): the RDNA4 fused small-M GEMM is a
measured win in FP16 but not in BF16, so BF16 takes the split path there.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from ...utils import arch as _arch
from ...utils.arch import (
    gate_limits as _arch_gate_limits,
)


class GemmLaunch(NamedTuple):
    """Resolved launch tile for :func:`gptq_gemm` on one architecture class."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    small_m_max: int
    max_in_features: int


# Launch table keyed by ``arch_class()``. ``cdna4`` is the gfx950 configuration
# measured for CDNA (see the module docstring); ``rdna4`` was re-measured on
# gfx1201 rather than inherited from the CDNA values.
_LAUNCH_CONFIGS: dict[str, GemmLaunch] = {
    "cdna4": GemmLaunch(
        block_m=16, block_n=16, block_k=256, num_warps=4, num_stages=2,
        small_m_max=16, max_in_features=5120,
    ),
    # RDNA4 (gfx1200/gfx1201/gfx1250): measured on gfx1201 (R9700).  A sweep
    # over K in 896..6144 x N in 128..11008, M in {2, 8, 16} and
    # (block_m, block_n) in {16, 32}^2 with num_warps=4 found (32, 32) has the
    # lowest worst-case regret (1.138x, mean 1.031x) against (16, 16)'s 1.365x /
    # 1.116x, and it is up to 1.19x faster on wave32 (K=4096: 71us vs 87us).
    "rdna4": GemmLaunch(
        block_m=32, block_n=32, block_k=256, num_warps=4, num_stages=2,
        small_m_max=16, max_in_features=17408,
    ),
}

# Default for non-ROCm (CUDA/MPS) callers of the Triton dequant path, and for
# callers that pass no architecture. The small-M gate is disabled there.
_DEFAULT_LAUNCH = GemmLaunch(
    block_m=16, block_n=16, block_k=256, num_warps=4, num_stages=2,
    small_m_max=0, max_in_features=0,
)

# Conservative module-level defaults kept for import compatibility
# (``tests/kernels/test_gptq_rocm_gemm.py`` imports SMALL_M_MAX/MAX_IN_FEATURES).
SMALL_M_MAX = _LAUNCH_CONFIGS["cdna4"].small_m_max
MAX_IN_FEATURES = _LAUNCH_CONFIGS["cdna4"].max_in_features
BLOCK_M = _LAUNCH_CONFIGS["cdna4"].block_m
BLOCK_N = _LAUNCH_CONFIGS["cdna4"].block_n
BLOCK_K = _LAUNCH_CONFIGS["cdna4"].block_k
NUM_WARPS = _LAUNCH_CONFIGS["cdna4"].num_warps
NUM_STAGES = _LAUNCH_CONFIGS["cdna4"].num_stages


def launch_config(arch=None) -> GemmLaunch:
    """Return the GEMM launch tile for an architecture string or device.

    Accepts a normalized ``gfxXXXX``/``sm_XX`` string, a ``torch.device``, or
    ``None``. Unregistered architectures get the conservative default, whose
    ``small_m_max``/``max_in_features`` are ``0`` so the fused path stays off.
    """

    if not isinstance(arch, str):
        return _LAUNCH_CONFIGS.get(_arch.arch_class(arch), _DEFAULT_LAUNCH)

    name = _arch.normalize_arch(arch)
    if not name.startswith("gfx"):
        return _DEFAULT_LAUNCH
    return _LAUNCH_CONFIGS.get(_arch.arch_class(name), _DEFAULT_LAUNCH)


def gate_limits(arch=None) -> tuple[int, int]:
    """Return ``(small_m_max, max_in_features)`` for the fused GEMM gate.

    Delegates to :func:`gptqmodel.utils.arch.gate_limits`; anything that is not
    a ROCm ``gfx*`` architecture string yields ``(0, 0)`` (gate disabled).
    """

    if not isinstance(arch, str):
        return (0, 0)
    name = _arch.normalize_arch(arch)
    if not name.startswith("gfx"):
        return (0, 0)
    return _arch_gate_limits(name)


@triton.jit
def _gptq_gemm_small_m_kernel(
    x_ptr, qweight_ptr, scales_ptr, qzeros_ptr, g_idx_ptr, out_ptr,
    m,
    in_features: tl.constexpr, out_features: tl.constexpr,
    num_groups: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    bits: tl.constexpr,
):
    pack_factor: tl.constexpr = 32 // bits
    maxq: tl.constexpr = (1 << bits) - 1

    rows_m = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    m_mask = rows_m < m
    n_mask = cols < out_features

    accumulator = tl.zeros((block_m, block_n), tl.float32)
    k_offsets = tl.arange(0, block_k)

    for chunk in range(tl.cdiv(in_features, block_k)):
        rows = chunk * block_k + k_offsets
        row_mask = rows < in_features
        groups = tl.load(g_idx_ptr + rows, row_mask, other=0)
        groups = tl.where(groups < 0, groups + num_groups, groups)
        group_mask = row_mask & (groups >= 0) & (groups < num_groups)
        weight_mask = group_mask[:, None] & n_mask[None, :]

        weight_words = tl.load(
            qweight_ptr + (rows[:, None] // pack_factor) * out_features + cols[None, :],
            weight_mask, other=0,
        )
        zero_words = tl.load(
            qzeros_ptr + groups[:, None] * (out_features // pack_factor) + cols[None, :] // pack_factor,
            weight_mask, other=0,
        )
        scales = tl.load(
            scales_ptr + groups[:, None] * out_features + cols[None, :],
            weight_mask, other=0,
        ).to(tl.float32)

        codes = (weight_words >> ((rows[:, None] % pack_factor) * bits)) & maxq
        zeros = (zero_words >> ((cols[None, :] % pack_factor) * bits)) & maxq
        weights = ((codes - zeros).to(tl.float32) * scales).to(x_ptr.dtype.element_ty)

        activations = tl.load(
            x_ptr + rows_m[:, None] * in_features + rows[None, :],
            m_mask[:, None] & row_mask[None, :], other=0,
        )
        accumulator = tl.dot(activations, weights, accumulator)

    tl.store(
        out_ptr + rows_m[:, None] * out_features + cols[None, :],
        accumulator.to(out_ptr.dtype.element_ty),
        m_mask[:, None] & n_mask[None, :],
    )


def gptq_gemm(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
              qzeros: torch.Tensor, g_idx: torch.Tensor,
              out_dtype: torch.dtype = None, bits: int = 4) -> torch.Tensor:
    cfg = launch_config(_arch.rocm_arch(x.device))
    m, in_features = x.shape
    out_features = qweight.shape[1]
    output = torch.empty(
        (m, out_features), device=x.device, dtype=x.dtype if out_dtype is None else out_dtype,
    )
    grid = (triton.cdiv(m, cfg.block_m), triton.cdiv(out_features, cfg.block_n))
    with torch.cuda.device(x.device):
        _gptq_gemm_small_m_kernel[grid](
            x, qweight, scales, qzeros, g_idx, output,
            m, in_features, out_features, scales.shape[0],
            cfg.block_m, cfg.block_n, cfg.block_k, bits,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
        )
    return output
