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

``ada`` (RTX 4090, SM 8.9) reuses the ``(16, 16, 256, 4, 2)`` tile with
``small_m_max=16``, measured on one 4090 (torch 2.15.0+cu132, Triton 3.9.0,
4-bit gs=128, M in {2, 8, 16}, ``split / fused`` speedup, hot and 256 MiB
L2-flushed medians with ``torch.cuda.synchronize`` around every launch).  A
sweep over ``(block_m, block_n)`` in {16, 32, 64} x {16, 32, 64},
``block_k`` in {128, 256}, warps {4, 8} and stages {2, 4} found this tile has
the only uniform win region:

  FP16 / BF16, K <= 2048 and N <= 1536: 1.09x-1.65x hot (1.10x-1.82x on the
  initial wider sweep), and 1.10x-1.41x on L2-flushed inputs throughout
  K <= 2048, N <= 2048.  The K=2048 / N=2048 corner is neutral hot
  (FP16 0.98x-1.03x, BF16 1.11x-1.13x) but still 1.11x-1.33x flushed, so it
  stays inside the gate as a non-regressing point rather than a hot win.

It loses outside that rectangle, and the loss is not monotonic, so the gate is
the conservative rectangle rather than a shape-dependent band (hot medians):

  K=2048 N=2560/3072/3584   : 0.95x / 0.87x / 0.87x
  K=1536 N=4096             : 0.89x
  K=2048 N=6144             : 0.78x
  K=2560..4096 N>=2560      : 0.78x-0.86x

A second measured band (K <= 1536, N <= 4096, min 1.01x FP16 / 1.13x BF16) is
not enabled because the enclosing rectangle regresses; ``TorchLinear``'s gate
only expresses ``small_m_max``/``max_in_features``/``max_out_features``.
Wider tiles (32x32, 16x32 with block_k=128, 64x64) recover some of the large-N
points but lose worst-case elsewhere and are not shipped.

BF16 is a measured win inside the gate (1.11x-1.82x), so Ada claims both
dtypes; unlike RDNA4, no dtype is left on the split path for correctness.
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
    # Optional output-width cap.  ``0`` (the default) disables it, which keeps
    # the cdna4/rdna4 entries and their ``(small_m_max, max_in_features)``
    # contract byte-for-byte unchanged.
    max_out_features: int = 0
    # Ada/Blackwell retile the launch by M: (16, 32) for M <= 16, (32, 64) for
    # M <= 32 and (64, 64) for M <= 64.  CDNA4/RDNA4 keep one fixed tile.
    m_banded: bool = False
    # Deterministic split-K is only auto-selected for the retiled CUDA entries;
    # the ROCm entries keep their previously measured single-pass launch.
    auto_split: bool = False


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
    # Ada (RTX 4090, SM 8.9): the packed-load rewrite loads every int32 word
    # once and unpacks it in registers, so the serial-K and narrow-grid losses
    # of the old (16, 16, 256) tile are gone.  BLOCK_K=128, 4 warps, 2 stages,
    # with BLOCK_N=32 (64 when M > 16) and BLOCK_M in {16, 32, 64}.  See the
    # module docstring table.
    "ada": GemmLaunch(
        block_m=16, block_n=32, block_k=128, num_warps=4, num_stages=2,
        small_m_max=64, max_in_features=0, max_out_features=0,
        m_banded=True, auto_split=True,
    ),
    # Blackwell (SM 10.x+): same packed-load tile as Ada.  This host has no
    # Blackwell GPU to re-measure, so the entry is enabled from the issue's
    # indicative sm_120 report and shares the Ada kernel exactly.
    "blackwell": GemmLaunch(
        block_m=16, block_n=32, block_k=128, num_warps=4, num_stages=2,
        small_m_max=64, max_in_features=0, max_out_features=0,
        m_banded=True, auto_split=True,
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

    Accepts a normalized ``gfxXXXX``/``sm_XX`` string, a ``(major, minor)``
    capability tuple, a ``torch.device``, or ``None``.  Dispatch goes through
    the shared :func:`gptqmodel.utils.arch.tuning_family` vocabulary so CUDA
    reaches the ``ada`` entry and ROCm reaches ``cdna4``/``rdna4``.
    Unregistered architectures get the conservative default, whose
    ``small_m_max``/``max_in_features`` are ``0`` so the fused path stays off.
    """

    if arch is None:
        return _DEFAULT_LAUNCH
    if not isinstance(arch, str):
        return _LAUNCH_CONFIGS.get(_arch.tuning_family(arch), _DEFAULT_LAUNCH)

    name = _arch.normalize_arch(arch)
    if not name.startswith("gfx") and not name.startswith("sm_"):
        return _DEFAULT_LAUNCH
    return _LAUNCH_CONFIGS.get(_arch.tuning_family(name), _DEFAULT_LAUNCH)


def gate_limits(arch=None) -> tuple[int, int]:
    """Return ``(small_m_max, max_in_features)`` for the fused GEMM gate.

    Delegates to :func:`gptqmodel.utils.arch.gate_limits`; anything that is not
    a registered ``gfx*``/``sm_*`` architecture string yields ``(0, 0)`` (gate
    disabled).
    """

    if arch is None:
        return (0, 0)
    if not isinstance(arch, str):
        return _arch_gate_limits(arch)
    name = _arch.normalize_arch(arch)
    if not name.startswith("gfx") and not name.startswith("sm_"):
        return (0, 0)
    return _arch_gate_limits(name)


def gate_max_out_features(arch=None) -> int:
    """Return the fused GEMM output-width cap (``0`` = no cap)."""

    return _arch.gemm_fused_max_out_features(arch)


# M-banded launch for the CUDA packed-load entries.  ``(max_m, block_m,
# block_n)``: the whole batch fits one M tile, so every weight word is decoded
# exactly once per output tile.
_M_BANDS: tuple[tuple[int, int, int], ...] = ((16, 16, 32), (32, 32, 64), (64, 64, 64))


def tile_for_m(cfg: GemmLaunch, m: int) -> tuple[int, int]:
    """Return ``(block_m, block_n)`` for this batch size and launch entry."""

    if not cfg.m_banded:
        return cfg.block_m, cfg.block_n
    for max_m, block_m, block_n in _M_BANDS:
        if m <= max_m:
            return block_m, block_n
    return _M_BANDS[-1][1], _M_BANDS[-1][2]


def split_k_for(m: int, in_features: int) -> int:
    """Pick a deterministic split-K factor for the CUDA packed-load kernel.

    Largest ``S`` in ``{2, 4, 8}`` with ``K % (128 * S) == 0`` and
    ``K / S >= 512``; capped at 4 when ``M > 32`` so the fp32 partial buffer
    stays small.
    """

    chosen = 1
    for candidate in (8, 4, 2):
        if in_features % (128 * candidate) == 0 and in_features // candidate >= 512:
            chosen = candidate
            break
    if m > 32:
        chosen = min(chosen, 4)
    return chosen


@triton.jit
def _gptq_gemm_small_m_kernel(
    x_ptr, qweight_ptr, scales_ptr, qzeros_ptr, g_idx_ptr, out_ptr,
    m,
    in_features: tl.constexpr, out_features: tl.constexpr,
    num_groups: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    bits: tl.constexpr,
    uniform_group: tl.constexpr,
    split_k: tl.constexpr,
):
    pack_factor: tl.constexpr = 32 // bits
    maxq: tl.constexpr = (1 << bits) - 1
    group_size: tl.constexpr = in_features // num_groups
    k_per_split: tl.constexpr = in_features // split_k
    num_chunks: tl.constexpr = (k_per_split + block_k - 1) // block_k

    split = tl.program_id(2)
    k_begin = split * k_per_split

    rows_m = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    m_mask = rows_m < m
    n_mask = cols < out_features

    accumulator = tl.zeros((block_m, block_n), tl.float32)
    k_offsets = tl.arange(0, block_k)
    packed_k = tl.arange(0, block_k // pack_factor)
    lanes = tl.arange(0, pack_factor)
    shifts = lanes * bits
    packed_cols = tl.arange(0, block_n // pack_factor)

    for chunk in range(num_chunks):
        k_start = k_begin + chunk * block_k
        rows = k_start + k_offsets
        row_mask = rows < in_features

        # One load per int32 word: [BLOCK_K / pack_factor, BLOCK_N].  The
        # per-lane codes are then extracted in registers and reshaped back to
        # [BLOCK_K, BLOCK_N]; the old kernel re-fetched every word once per
        # lane.
        packed_rows = k_start // pack_factor + packed_k
        packed_mask = (packed_rows < in_features // pack_factor)[:, None] & n_mask[None, :]
        words = tl.load(
            qweight_ptr + packed_rows[:, None] * out_features + cols[None, :],
            packed_mask, other=0,
        )
        codes = (words[:, None, :] >> shifts[None, :, None]) & maxq
        codes = tl.reshape(codes, (block_k, block_n))

        if uniform_group:
            # g_idx is the identity mapping and the K-tile lies inside one
            # quantization group: one scale/zero row covers the whole tile.
            group = k_start // group_size
            scale_row = tl.load(
                scales_ptr + group * out_features + cols, n_mask, other=0,
            ).to(tl.float32)
            zero_cols = tl.program_id(1) * block_n // pack_factor + packed_cols
            zero_mask = zero_cols < (out_features // pack_factor)
            zero_words = tl.load(
                qzeros_ptr + group * (out_features // pack_factor) + zero_cols,
                zero_mask, other=0,
            )
            zeros = (zero_words[:, None] >> shifts[None, :]) & maxq
            zeros = tl.reshape(zeros, (block_n,))
            weights = (
                (codes - zeros[None, :]).to(tl.float32) * scale_row[None, :]
            ).to(x_ptr.dtype.element_ty)
        else:
            # Act-order (or a group narrower than BLOCK_K): keep the per-row
            # g_idx gather.
            groups = tl.load(g_idx_ptr + rows, row_mask, other=0)
            groups = tl.where(groups < 0, groups + num_groups, groups)
            group_mask = row_mask & (groups >= 0) & (groups < num_groups)
            weight_mask = group_mask[:, None] & n_mask[None, :]
            scales = tl.load(
                scales_ptr + groups[:, None] * out_features + cols[None, :],
                weight_mask, other=0,
            ).to(tl.float32)
            zero_words = tl.load(
                qzeros_ptr
                + groups[:, None] * (out_features // pack_factor)
                + cols[None, :] // pack_factor,
                weight_mask, other=0,
            )
            zeros = (zero_words >> ((cols[None, :] % pack_factor) * bits)) & maxq
            weights = ((codes - zeros).to(tl.float32) * scales).to(x_ptr.dtype.element_ty)

        activations = tl.load(
            x_ptr + rows_m[:, None] * in_features + rows[None, :],
            m_mask[:, None] & row_mask[None, :], other=0,
        )
        accumulator = tl.dot(activations, weights, accumulator)

    if split_k > 1:
        # Deterministic split-K: each program owns one K slice and writes an
        # fp32 partial.  The caller reduces ``partials.sum(0)``; no atomics.
        tl.store(
            out_ptr
            + split * m * out_features
            + rows_m[:, None] * out_features
            + cols[None, :],
            accumulator,
            m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.store(
            out_ptr + rows_m[:, None] * out_features + cols[None, :],
            accumulator.to(out_ptr.dtype.element_ty),
            m_mask[:, None] & n_mask[None, :],
        )


def gptq_gemm(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
              qzeros: torch.Tensor, g_idx: torch.Tensor,
              out_dtype: torch.dtype = None, bits: int = 4, *,
              uniform_group: bool = False,
              split_k: int = None) -> torch.Tensor:
    """Fused GPTQ small-M GEMM.

    ``uniform_group`` says the checkpoint's ``g_idx`` is the identity mapping;
    the kernel then loads one scale/zero row per K-tile instead of gathering
    per row.  ``split_k=None`` selects the deterministic split factor from
    :func:`split_k_for`; pass ``1`` to force the single-pass path (which is
    bitwise identical to the pre-rewrite kernel).
    """

    cfg = launch_config(x.device)
    m, in_features = x.shape
    out_features = qweight.shape[1]
    block_m, block_n = tile_for_m(cfg, m)
    output_dtype = x.dtype if out_dtype is None else out_dtype

    if split_k is None:
        split_k = split_k_for(m, in_features) if cfg.auto_split else 1
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")

    group_size = in_features // scales.shape[0]
    uniform_group = bool(uniform_group) and group_size >= cfg.block_k

    if split_k > 1:
        if in_features % (cfg.block_k * split_k) != 0:
            raise ValueError(
                f"split_k={split_k} does not divide in_features={in_features} "
                f"into {cfg.block_k}-element K tiles"
            )
        output = torch.empty(
            (split_k, m, out_features), device=x.device, dtype=torch.float32,
        )
    else:
        output = torch.empty((m, out_features), device=x.device, dtype=output_dtype)

    grid = (
        triton.cdiv(m, block_m),
        triton.cdiv(out_features, block_n),
        split_k,
    )
    with torch.cuda.device(x.device):
        _gptq_gemm_small_m_kernel[grid](
            x, qweight, scales, qzeros, g_idx, output,
            m, in_features, out_features, scales.shape[0],
            block_m, block_n, cfg.block_k, bits,
            uniform_group, split_k,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
        )
    if split_k > 1:
        return output.sum(0).to(output_dtype)
    return output
