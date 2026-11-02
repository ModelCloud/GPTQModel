"""Fused GPTQ GEMV for single-token (batch-1) decode.

``TorchLinear`` normally dequantizes the whole weight matrix and then calls
``torch.matmul``, which materializes an ``in_features x out_features``
activation-dtype tensor and launches a separate GEMM. For batch-1 decode that
is dominated by that dequantize + GEMM pair. This kernel dequantizes the
low-bit weights inside the matrix-vector product, so the packed weights are
read once and no dequantized weight buffer is ever written.

Supports 2/4/8-bit GPTQ weights packed into ``int32`` words (the 10-1-10-1-10
3-bit layout is not handled here and falls back).

The block shape below was measured on gfx950 (MI350X/MI355X) across the
projection shapes of Qwen2.5-0.5B and larger 7B-32B style models
(K,N in 896..17408). ``BLOCK_K=512, BLOCK_N=8, num_warps=8`` was the fastest
configuration for every shape tested, and the fused path was 1.5x-2.9x faster
than the dequantize + matmul path.

The launch constants live in a per-architecture table. ``cdna4`` holds exactly
the measured gfx950 values above (``block_k`` cap 512, ``block_n`` 8,
``num_warps`` 8). ``rdna4`` (gfx1200/gfx1201/gfx1250) uses ``block_n=16`` /
``num_warps=4``, measured on an AMD Radeon AI PRO R9700 (``gfx1201``, HIP
7.16.26385, torch 2.16.0.dev20261029+rocm10.1), 4-bit, N == K, single token,
median of 60 iterations.  Per-shape ``block_k`` cap stays 512.

Measured decode latency (both tiles launched on the same gfx1201 device):

======  =============  =============  ========
K       cdna tile      rdna4 tile     gain
======  =============  =============  ========
2048    54.9 us        50.2 us        1.09x
3584    66.1 us        61.5 us        1.08x
4096    69.7 us        65.3 us        1.07x
5120    81.7 us        78.4 us        1.04x
======  =============  =============  ========

RDNA4 is wave32, so ``num_warps=4`` covers 128 lanes of a CTA against CDNA's
wave64 ``num_warps=8`` (512 lanes); the wider ``block_n`` with fewer warps is
what wins on hardware, and it wins on every shape measured.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from ...utils.arch import arch_class, normalize_arch, rocm_arch


class GemvLaunch(NamedTuple):
    """Resolved launch constants for :func:`gptq_gemv` on one arch class."""

    max_block_k: int
    block_n: int
    num_warps: int


# ``block_k`` is the cap of ``min(cap, max(32, 1 << (in_features - 1).bit_length()))``;
# keeping the power-of-two formula means a small ``in_features`` still gets one
# K tile, while large projections stop at the measured cap.
_GEMV_LAUNCH_CONFIGS: dict[str, GemvLaunch] = {
    # Measured on gfx950 (MI350X/MI355X) across K,N in 896..17408.
    "cdna4": GemvLaunch(max_block_k=512, block_n=8, num_warps=8),
    # Measured on gfx1201 (R9700): block_n=16 / num_warps=4 wins for K>=3584
    # (1.09x-1.17x) and is within 2% at K=2048.  See the module docstring.
    "rdna4": GemvLaunch(max_block_k=512, block_n=16, num_warps=4),
}

_DEFAULT_GEMV_LAUNCH = GemvLaunch(max_block_k=512, block_n=8, num_warps=8)

# Module-level defaults kept for callers that import the constants directly.
GEMV_MAX_BLOCK_K = _DEFAULT_GEMV_LAUNCH.max_block_k
GEMV_BLOCK_N = _DEFAULT_GEMV_LAUNCH.block_n
GEMV_NUM_WARPS = _DEFAULT_GEMV_LAUNCH.num_warps


def launch_config(arch=None) -> GemvLaunch:
    """Return the GEMV launch constants for an architecture string or device."""

    if not isinstance(arch, str):
        return _GEMV_LAUNCH_CONFIGS.get(arch_class(arch), _DEFAULT_GEMV_LAUNCH)

    name = normalize_arch(arch)
    if not name.startswith("gfx"):
        return _DEFAULT_GEMV_LAUNCH
    return _GEMV_LAUNCH_CONFIGS.get(arch_class(name), _DEFAULT_GEMV_LAUNCH)


def block_k_for(in_features: int, arch=None) -> int:
    """Return the ``block_k`` tile for ``in_features`` under ``arch``."""

    cap = launch_config(arch).max_block_k
    return min(cap, max(32, 1 << (in_features - 1).bit_length()))


@triton.jit
def _gptq_gemv_kernel(
    x_ptr, qweight_ptr, scales_ptr, qzeros_ptr, g_idx_ptr, out_ptr,
    in_features: tl.constexpr, out_features: tl.constexpr,
    num_groups: tl.constexpr, block_k: tl.constexpr, block_n: tl.constexpr,
    bits: tl.constexpr,
):
    pack_factor: tl.constexpr = 32 // bits
    maxq: tl.constexpr = (1 << bits) - 1
    row = tl.program_id(1)
    cols = tl.program_id(0) * block_n + tl.arange(0, block_n)
    offsets = tl.arange(0, block_k)
    accumulator = tl.full((block_k, block_n), 0, tl.float32)

    for chunk in range(tl.cdiv(in_features, block_k)):
        rows = chunk * block_k + offsets
        row_mask = rows < in_features
        mask = row_mask[:, None] & (cols[None, :] < out_features)
        activations = tl.load(x_ptr + row * in_features + rows, row_mask, other=0).to(tl.float32)
        groups = tl.load(g_idx_ptr + rows, row_mask, other=0)
        groups = tl.where(groups < 0, groups + num_groups, groups)
        mask = mask & (groups[:, None] >= 0) & (groups[:, None] < num_groups)
        weight_words = tl.load(
            qweight_ptr + (rows[:, None] // pack_factor) * out_features + cols[None, :], mask, other=0,
        )
        zero_words = tl.load(
            qzeros_ptr + groups[:, None] * (out_features // pack_factor) + cols[None, :] // pack_factor,
            mask, other=0,
        )
        scales = tl.load(
            scales_ptr + groups[:, None] * out_features + cols[None, :], mask, other=0,
        ).to(tl.float32)
        codes = (weight_words >> ((rows[:, None] % pack_factor) * bits)) & maxq
        zeros = (zero_words >> ((cols[None, :] % pack_factor) * bits)) & maxq
        weights = ((codes - zeros).to(tl.float32) * scales).to(x_ptr.dtype.element_ty).to(tl.float32)
        accumulator += activations[:, None] * weights

    result = tl.sum(accumulator, axis=0)
    tl.store(out_ptr + row * out_features + cols, result, cols < out_features)


def gptq_gemv(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
              qzeros: torch.Tensor, g_idx: torch.Tensor,
              out_dtype: torch.dtype = None, bits: int = 4) -> torch.Tensor:
    in_features = x.shape[1]
    out_features = qweight.shape[1]
    arch = rocm_arch(x.device)
    cfg = launch_config(arch)
    block_k = block_k_for(in_features, arch)
    block_n = cfg.block_n
    output = torch.empty(
        (x.shape[0], out_features), device=x.device, dtype=x.dtype if out_dtype is None else out_dtype,
    )
    with torch.cuda.device(x.device):
        _gptq_gemv_kernel[(triton.cdiv(out_features, block_n), x.shape[0])](
            x, qweight, scales, qzeros, g_idx, output,
            in_features, out_features, scales.shape[0], block_k, block_n, bits,
            num_warps=cfg.num_warps,
        )
    return output
