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
"""

from functools import lru_cache

import mlx.core as mx


# Keep these in sync with mlx/backend/metal/quantized.cpp.
_SPLIT_K_TARGET_THREADGROUPS = 512
_SPLIT_K_TILE = 32
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


@lru_cache(maxsize=4096)
def padded_rows(rows, input_dims, output_dims, group_size):
    """Return the row count that avoids MLX's split-K pass, or ``None``.

    ``None`` means the caller should run the matmul unchanged: either MLX never
    reaches split-K for this shape, split-K is already off, or padding would
    cost more than the pass it removes.
    """
    if rows < _qmv_batch_limit(input_dims, output_dims):
        # MLX uses its vector kernels below this row count and never splits K.
        return None
    if _split_k(rows, input_dims, output_dims, group_size) == 1:
        return None
    if output_dims < _MIN_OUTPUT_DIMS:
        # Narrow outputs finish in microseconds; the extra rows dominate.
        return None
    limit = _MAX_ROW_GROWTH * rows
    for row_tiles in range(-(-rows // _SPLIT_K_TILE) + 1, 33):
        # The shortest row count that still needs ``row_tiles`` 32-row tiles.
        candidate = row_tiles * _SPLIT_K_TILE - (_SPLIT_K_TILE - 1)
        if candidate > limit:
            break
        if _split_k(candidate, input_dims, output_dims, group_size) == 1:
            return candidate
    return None


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
    target = padded_rows(rows, input_dims, output_dims, group_size)
    if target is None:
        return mx.quantized_matmul(x, weight, scales=scales, biases=biases, **kwargs)
    flat = x.reshape(rows, input_dims)
    padded = mx.zeros((target, input_dims), dtype=x.dtype)
    padded[:rows] = flat
    out = mx.quantized_matmul(padded, weight, scales=scales, biases=biases, **kwargs)
    return out[:rows].reshape(*x.shape[:-1], output_dims)
