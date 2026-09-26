# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx

"""Native MLX tile-layout transforms for EXL3 quantization."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _exl3_tile_layout_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_tile_layout",
        input_names=["input"],
        output_names=["output"],
        source="""
            uint index = thread_position_in_grid.x;
            if (index >= ELEMENTS) return;

            uint tile = index >> 8;
            uint lane = index & 255u;
            uint thread_id = lane >> 3;
            uint item = lane & 7u;

            // Match EXL3's tensor_core_perm without storing a lookup table.
            uint local_row = (thread_id & 3u) * 2u + (item & 1u);
            if (item & 2u) local_row += 8u;
            uint local_column = (thread_id >> 2) + ((item & 4u) ? 8u : 0u);

            uint tile_row = tile / TILE_COLUMNS;
            uint tile_column = tile % TILE_COLUMNS;
            uint matrix_index = (tile_row * 16u + local_row) * COLUMNS
                + tile_column * 16u + local_column;

            if (REVERSE) {
                output[matrix_index] = input[index];
            } else {
                output[index] = input[matrix_index];
            }
        """,
    )


def _validate_float32(array, *, name: str):
    import mlx.core as mx

    array = mx.array(array)
    if array.dtype != mx.float32:
        raise ValueError(f"{name} must have float32 dtype")
    return array


def exl3_to_tensor_core_tiles_mlx(weight):
    """Reorder an EXL3 weight matrix into tensor-core 16-by-16 tiles.

    EXL3 quantization stores weights as ``(input_features, output_features)``.
    Both dimensions must be divisible by 16. The returned float32 array has
    shape ``(input_features // 16, output_features // 16, 256)`` and uses the
    exact lane order consumed by EXL3's Viterbi quantizer.
    """
    import mlx.core as mx

    weight = _validate_float32(weight, name="weight")
    if weight.ndim != 2 or any(dimension == 0 for dimension in weight.shape):
        raise ValueError("weight must be a nonempty rank-two array")
    if weight.shape[0] % 16 or weight.shape[1] % 16:
        raise ValueError("weight dimensions must be divisible by 16")

    rows, columns = weight.shape
    tile_rows = rows // 16
    tile_columns = columns // 16
    output_shape = (tile_rows, tile_columns, 256)
    output = _exl3_tile_layout_kernel()(
        inputs=[mx.contiguous(weight)],
        template=[
            ("REVERSE", False),
            ("ELEMENTS", weight.size),
            ("COLUMNS", columns),
            ("TILE_COLUMNS", tile_columns),
        ],
        grid=(weight.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(output)
    return output


def exl3_from_tensor_core_tiles_mlx(tiles):
    """Restore an EXL3 float32 matrix from tensor-core 16-by-16 tiles."""
    import mlx.core as mx

    tiles = _validate_float32(tiles, name="tiles")
    if tiles.ndim != 3 or tiles.shape[-1] != 256:
        raise ValueError("tiles must have shape (tile_rows, tile_columns, 256)")
    if tiles.shape[0] == 0 or tiles.shape[1] == 0:
        raise ValueError("tiles must contain at least one tile")

    rows = tiles.shape[0] * 16
    columns = tiles.shape[1] * 16
    output_shape = (rows, columns)
    output = _exl3_tile_layout_kernel()(
        inputs=[mx.contiguous(tiles)],
        template=[
            ("REVERSE", True),
            ("ELEMENTS", tiles.size),
            ("COLUMNS", columns),
            ("TILE_COLUMNS", tiles.shape[1]),
        ],
        grid=(tiles.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(output)
    return output


__all__ = ["exl3_from_tensor_core_tiles_mlx", "exl3_to_tensor_core_tiles_mlx"]
