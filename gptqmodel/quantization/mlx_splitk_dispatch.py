# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Route MLX affine matmuls around the split-K regression on Apple GPUs.

MLX's Metal dispatcher sends every transposed weight matmul with
``rows >= get_qmv_batch_limit(K, N, device)`` to ``qmm_splitk``, which picks
``split_k = 512 // (ceil(N / 32) * ceil(M / 32))`` threadgroups and then sums
the partitions in a separate ``steel_gemm_splitk_accum`` pass.

Split-K only earns its extra pass when the unsplit grid cannot fill the GPU.
The 5120-wide projections of a 27B-class model already emit 160 threadgroups at
``M = 12``, so MLX asks for ``split_k = 3`` on a grid that was not starved.
Measured on an M5 Ultra at 5120x6144, ``M = 12``, group size 64, 4 bits, the
same ``bm = 32`` kernel runs at 112 GB/s of weight bandwidth with
``split_k = 3`` and 480 GB/s with ``split_k = 1``.

This module evaluates those rows as a slightly taller matrix so the dispatcher
selects ``split_k = 1``. The padding rows are zero and their outputs are
discarded. Only the K-reduction order changes, which is validated against a
Torch float64 oracle.

Not every split-K run is worth undoing. Narrower outputs such as ``k_proj``
(1024 columns) are cheap enough that padding costs far more than the extra
pass, so the rule is restricted to the measured win region.

Below ``get_qmv_batch_limit`` MLX runs ``qmv_wide`` instead of the matrix
kernel. ``qmv_wide`` streams ``vecs_per_tg`` input vectors per threadgroup and
caps that tile at five, so a batch above five rows is split into
``ceil(rows / 5)`` tiles and the whole weight matrix is re-read once per tile.
That is the same trade split-K makes, only worse: on an M5 Ultra a 5120-wide
projection falls from 945 GB/s of weight bandwidth at one row to 152 GB/s at
eleven rows. Those row counts are also worth evaluating as the smallest
unsplit matrix batch.
"""

from functools import lru_cache

import mlx.core as mx


# This module mirrors the Metal dispatch in
# ``mlx/backend/metal/quantized.cpp`` as of MLX 0.32.4. MLX 0.32.5 reworks that
# dispatch: ``qmv_mma`` replaces ``qmv_wide`` from six rows up, and
# ``qmm_splitk`` first diverts to a NAX kernel with a different split-K
# formula. Neither assumption survives the change, so stop steering the
# dispatcher instead of padding batches the new heuristics already handle.
_MIRRORED_DISPATCH = (0, 32, 4)


@lru_cache(maxsize=1)
def _dispatch_version():
    """Return the installed MLX version as a tuple, or ``None`` if unknown."""
    try:
        parts = getattr(mx, "__version__", "").split(".")[:3]
        return tuple(int(part) for part in parts)
    except (AttributeError, ValueError):
        return None


def _mirrors_installed_dispatch():
    """True when the installed MLX still uses the dispatch mirrored here."""
    version = _dispatch_version()
    return version is not None and version <= _MIRRORED_DISPATCH


# Keep these in sync with mlx/backend/metal/quantized.cpp.
_SPLIT_K_TARGET_THREADGROUPS = 512
_SPLIT_K_TILE = 32
# ``qmv_wide`` caps its input-vector tile at five rows and re-reads the weights
# per tile, so batches above this size pay multiple passes over the weights.
# FP16 activations are weight-traffic bound there, so a single-pass matrix
# kernel wins from six rows up. BF16 decode is ALU bound instead and still
# reaches ~430 GB/s at six rows, so it only wins once the vector kernel's extra
# weight passes outweigh its cheaper arithmetic; measured crossover is nine.
_QMV_WIDE_ESCAPE_ROWS = {mx.float16: 6, mx.bfloat16: 9}
# The escape must never move more weight traffic than the vector kernel it
# replaces. ``qmv_wide`` makes ``ceil(rows / 5)`` passes, which is two or three
# for the row counts above, so the matrix kernel is capped at two 32-row tiles.
_MAX_ESCAPE_ROW_TILES = 2
# Below this output width the split-K pass is cheaper than the padded rows.
_MIN_OUTPUT_DIMS = 3072
# Evaluating more than this many times the requested rows loses more to tile
# work than split-K costs.
_MAX_ROW_GROWTH = 8


@lru_cache(maxsize=8)
def _architecture():
    """Return ``(generation, suffix)`` for the active Metal device.

    MLX reads the generation from the two characters before the suffix, so
    ``applegpu_g17d`` is generation 17, suffix ``'d'``.
    """
    try:
        name = mx.device_info()["architecture"]
    except Exception:
        return 0, ""
    if len(name) < 3:
        return 0, name[-1:] if name else ""
    tens, ones = name[-3], name[-2]
    if not ("0" <= tens <= "9" and "0" <= ones <= "9"):
        return 0, name[-1]
    return 10 * (ord(tens) - ord("0")) + ord(ones) - ord("0"), name[-1]


@lru_cache(maxsize=8)
def _qmv_batch_limit(input_dims, output_dims):
    """Mirror MLX's ``get_qmv_batch_limit`` for the active device."""
    generation, suffix = _architecture()
    if generation >= 17 and suffix != "d":
        if input_dims <= 2048 and output_dims <= 2048:
            return 33
        if input_dims <= 4096 and output_dims <= 4096:
            return 25
        return 13
    if generation >= 15 and suffix != "d":
        if input_dims <= 2048 and output_dims <= 2048:
            return 13
        if input_dims <= 4096 and output_dims <= 4096:
            return 15
        return 13
    if generation >= 13:
        if suffix == "d":
            if input_dims <= 2048 and output_dims <= 2048:
                return 32
            if input_dims <= 4096 and output_dims <= 4096:
                return 18
            return 12
        if input_dims <= 2048 and output_dims <= 2048:
            return 14
        if input_dims <= 4096 and output_dims <= 4096:
            return 10
        return 7 if (generation == 13 and suffix == "s") else 6
    if suffix == "d":
        if input_dims <= 2048 and output_dims <= 2048:
            return 32
        if input_dims <= 4096 and output_dims <= 4096:
            return 18
        return 12
    if input_dims <= 2048 and output_dims <= 2048:
        return 18
    if input_dims <= 4096 and output_dims <= 4096:
        return 12
    return 10


def _split_k(rows, input_dims, output_dims, group_size):
    """Mirror MLX's ``qmm_splitk`` split-K choice for transposed weights."""
    column_tiles = -(-output_dims // _SPLIT_K_TILE)
    row_tiles = -(-rows // _SPLIT_K_TILE)
    split_k = max(1, _SPLIT_K_TARGET_THREADGROUPS // (column_tiles * row_tiles))
    align = max(group_size, 32)
    split_k = min(split_k, max(1, input_dims // align))
    while split_k > 1 and input_dims % (split_k * align):
        split_k -= 1
    return split_k


def _qmv_wide_is_wasteful(rows, generation, dtype):
    """True when the vector path costs more than an unsplit matrix batch.

    Only affine quantization reaches this module, and ``use_qmv_wide`` enables
    the wide vector kernel for affine weights on generation 15 and newer.
    """
    if generation < 15:
        return False
    floor = _QMV_WIDE_ESCAPE_ROWS.get(dtype)
    return floor is not None and rows >= floor


def _padding_target(
    rows, input_dims, output_dims, group_size, floor, max_row_tiles=32,
):
    """Smallest row count at or above ``floor`` that avoids the split-K pass."""
    limit = _MAX_ROW_GROWTH * rows
    if floor > limit:
        return None
    if _split_k(floor, input_dims, output_dims, group_size) == 1:
        return floor
    for row_tiles in range(2, max_row_tiles + 1):
        candidate = row_tiles * _SPLIT_K_TILE - (_SPLIT_K_TILE - 1)
        if candidate < floor:
            continue
        if candidate > limit:
            break
        if _split_k(candidate, input_dims, output_dims, group_size) == 1:
            return candidate
    return None


@lru_cache(maxsize=4096)
def padded_rows(rows, input_dims, output_dims, group_size, dtype):
    """Return the row count that avoids MLX's split-K pass, or ``None``.

    ``None`` means the caller should run the matmul unchanged: either MLX never
    reaches split-K for this shape, split-K is already off, or padding would
    cost more than the pass it removes. Below ``get_qmv_batch_limit`` the same
    applies to the repeated weight passes of ``qmv_wide``.
    """
    if not _mirrors_installed_dispatch():
        # A newer MLX dispatches these shapes itself; do not pad blindly.
        return None
    if output_dims < _MIN_OUTPUT_DIMS:
        # Narrow outputs finish in microseconds; the extra rows dominate.
        return None
    vector_limit = _qmv_batch_limit(input_dims, output_dims)
    if rows < vector_limit:
        # MLX runs its vector kernels below this row count and never splits K.
        # That is one pass over the weights while the batch fits one wide tile;
        # past the dtype's measured crossover the wide kernel's extra weight
        # passes cost more than the rows we add to reach the matrix kernel.
        if not _qmv_wide_is_wasteful(rows, _architecture()[0], dtype):
            return None
        return _padding_target(
            rows,
            input_dims,
            output_dims,
            group_size,
            vector_limit,
            _MAX_ESCAPE_ROW_TILES,
        )
    if _split_k(rows, input_dims, output_dims, group_size) == 1:
        return None
    return _padding_target(rows, input_dims, output_dims, group_size, rows)


def quantized_matmul(x, weight, scales, biases, *, group_size, bits, mode="affine"):
    """Multiply ``x`` by transposed quantized ``weight`` without split-K.

    Falls back to :func:`mlx.core.quantized_matmul` whenever the split-K
    avoidance does not apply, so shapes outside the win region are untouched.
    """
    kwargs = {"transpose": True, "group_size": group_size, "bits": bits, "mode": mode}
    if mode != "affine":
        return mx.quantized_matmul(x, weight, scales=scales, biases=biases, **kwargs)
    input_dims = x.shape[-1]
    if x.size == 0 or x.dtype not in (mx.float16, mx.bfloat16):
        return mx.quantized_matmul(x, weight, scales=scales, biases=biases, **kwargs)
    rows = x.size // input_dims
    output_dims = weight.shape[0]
    target = padded_rows(rows, input_dims, output_dims, group_size, x.dtype)
    if target is None:
        return mx.quantized_matmul(x, weight, scales=scales, biases=biases, **kwargs)
    flat = x.reshape(rows, input_dims)
    padded = mx.zeros((target, input_dims), dtype=x.dtype)
    padded[:rows] = flat
    out = mx.quantized_matmul(padded, weight, scales=scales, biases=biases, **kwargs)
    return out[:rows].reshape(*x.shape[:-1], output_dims)
