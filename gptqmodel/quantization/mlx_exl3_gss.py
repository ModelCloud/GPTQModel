# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Native MLX sampling and optimization for EXL3 global-scale search."""

import math
from functools import lru_cache

from .mlx_exl3 import exl3_quantize_tiles_mlx


@lru_cache(maxsize=1)
def _exl3_gss_sample_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_gss_sample",
        input_names=["weight"],
        output_names=["tiles"],
        source="""
            uint index = thread_position_in_grid.x;
            if (index >= ELEMENTS) return;

            uint sample = index >> 8;
            uint lane = index & 255u;
            uint diagonal = sample / WIDTH;
            uint offset = sample % WIDTH;
            uint tile_row = diagonal % TILE_ROWS;
            uint tile_column = (diagonal + offset) % TILE_COLUMNS;

            // Match EXL3's tensor_core_perm without storing a lookup table.
            uint thread_id = lane >> 3;
            uint item = lane & 7u;
            uint local_row = (thread_id & 3u) * 2u + (item & 1u);
            if (item & 2u) local_row += 8u;
            uint local_column = (thread_id >> 2) + ((item & 4u) ? 8u : 0u);

            uint matrix_index = (tile_row * 16u + local_row) * COLUMNS
                + tile_column * 16u + local_column;
            tiles[index] = weight[matrix_index];
        """,
    )


def exl3_sample_global_scale_tiles_mlx(weight, *, width: int = 3):
    """Sample wrapped-diagonal EXL3 tiles for global-scale search.

    EXL3 stores quantization weights as ``(input_features, output_features)``.
    The result has shape ``(max(tile_rows, tile_columns) * width, 256)`` and
    exactly follows the sample order and tensor-core lane order used by EXL3.
    """
    import mlx.core as mx

    weight = mx.array(weight)
    if weight.dtype != mx.float32:
        raise ValueError("weight must have float32 dtype")
    if weight.ndim != 2 or any(dimension == 0 for dimension in weight.shape):
        raise ValueError("weight must be a nonempty rank-two array")
    if weight.shape[0] % 16 or weight.shape[1] % 16:
        raise ValueError("weight dimensions must be divisible by 16")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")

    rows, columns = weight.shape
    tile_rows = rows // 16
    tile_columns = columns // 16
    sample_count = max(tile_rows, tile_columns) * width
    output_shape = (sample_count, 256)
    output_size = sample_count * 256
    output = _exl3_gss_sample_kernel()(
        inputs=[mx.contiguous(weight)],
        template=[
            ("ELEMENTS", output_size),
            ("WIDTH", width),
            ("TILE_ROWS", tile_rows),
            ("TILE_COLUMNS", tile_columns),
            ("COLUMNS", columns),
        ],
        grid=(output_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(output)
    return output


def _exl3_global_scale_search_tiles_mlx(
    tiles,
    *,
    bits: int,
    codebook: str,
    workspace_bytes: int,
):
    import mlx.core as mx

    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    lower = 0.1
    upper = 1.9
    tolerance = 0.01

    def test_scale(scale: float):
        quantized, _ = exl3_quantize_tiles_mlx(
            tiles * scale,
            bits=bits,
            codebook=codebook,
            workspace_bytes=workspace_bytes,
        )
        mse = mx.mean(mx.square(quantized / scale - tiles))
        mx.eval(mse)
        return float(mse.item())

    x1 = lower + resphi * (upper - lower)
    x2 = upper - resphi * (upper - lower)
    f1 = test_scale(x1)
    f2 = test_scale(x2)
    while abs(upper - lower) > tolerance:
        if f1 < f2:
            upper = x2
            x2 = x1
            f2 = f1
            x1 = lower + resphi * (upper - lower)
            f1 = test_scale(x1)
        else:
            lower = x1
            x1 = x2
            f1 = f2
            x2 = upper - resphi * (upper - lower)
            f2 = test_scale(x2)

    final_mse = (mx.array(f1, dtype=mx.float32) + mx.array(f2, dtype=mx.float32)) / 2
    mx.eval(final_mse)
    return (lower + upper) / 2, float(final_mse.item())


def exl3_global_scale_search_mlx(
    weight,
    *,
    bits: int,
    codebook: str = "mcg",
    width: int = 3,
    workspace_bytes: int = 256 << 20,
):
    """Find EXL3's global weight scale with native MLX quantization.

    This reproduces EXL3's fixed ``[0.1, 1.9]`` golden-section search with a
    ``0.01`` interval tolerance. It returns the selected Python float scale and
    the mean of the final two float32 quantization errors.
    """
    tiles = exl3_sample_global_scale_tiles_mlx(weight, width=width)
    return _exl3_global_scale_search_tiles_mlx(
        tiles,
        bits=bits,
        codebook=codebook,
        workspace_bytes=workspace_bytes,
    )


__all__ = [
    "exl3_global_scale_search_mlx",
    "exl3_sample_global_scale_tiles_mlx",
]
