# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPTC packing with signed Hadamard transforms and two-sided LDLQ."""

from __future__ import annotations

import math
import zlib

import torch

from ..utils.cptc import native_ops
from ..utils.hadamard import hadamard_available, hadamard_transform
from ..utils.torch import tf32_high_precision_guard
from .cptc import (
    _block_scales,
    _codebook_limits,
    _encode_scaled_blocks,
    _indices,
    _lookup,
    decode_e2m5,
    gaussian_table,
    pack_states,
)
from .rotation.hadamard_utils import hadamard_supported, is_pow2, matmul_hadU


TILE_COLUMNS = 16


def validate_tile_shape(in_features, out_features, block_size):
    tile_rows = block_size // TILE_COLUMNS
    if in_features % TILE_COLUMNS or out_features % tile_rows:
        raise ValueError(f"CPTC YAQA requires projection dimensions divisible by {tile_rows} × {TILE_COLUMNS}")
    if not (hadamard_supported(in_features) and hadamard_supported(out_features)):
        raise ValueError(f"CPTC YAQA needs supported Hadamard widths, got {in_features} and {out_features}")
    return tile_rows


def hadamard(value, *, transpose=False):
    width = value.shape[-1]
    if (value.is_cuda and value.dtype == torch.float32 and is_pow2(width)
            and not (torch.is_grad_enabled() and value.requires_grad) and hadamard_available()):
        return hadamard_transform(value.contiguous(), 1.0 / math.sqrt(width))
    return matmul_hadU(value.contiguous(), transpose=transpose)


def rotate_matrix(matrix, input_signs, output_signs, *, inverse=False):
    if inverse:
        rows = hadamard(matrix, transpose=True) * input_signs
        return (hadamard(rows.T, transpose=True).T * output_signs[:, None]).contiguous()
    rows = hadamard(matrix * input_signs)
    return hadamard(rows.T * output_signs).T.contiguous()


def block_feedback(sketch, signs, tile, damping):
    # Orthogonal congruence preserves the mean diagonal. Damping makes the
    # empirical sketches positive definite even with few calibration rows.
    hessian = sketch.to(device=signs.device, dtype=torch.float64)
    hessian = rotate_matrix(hessian, signs, signs)
    mean = hessian.diagonal().mean()
    if not bool(torch.isfinite(hessian).all()) or mean < 0:
        raise ValueError("CPTC YAQA requires finite Hessian sketches with nonnegative trace")
    if mean == 0:
        return torch.zeros_like(hessian, dtype=torch.float32)
    hessian = (hessian + hessian.T) / (2 * mean)
    hessian.diagonal().add_(damping)
    chol = torch.linalg.cholesky(hessian)
    width = chol.shape[0]
    groups = width // tile
    diagonal = chol.reshape(groups, tile, groups, tile).diagonal(dim1=0, dim2=2).movedim(-1, 0)
    columns = chol.reshape(width, groups, tile).permute(1, 2, 0)
    feedback = torch.linalg.solve_triangular(diagonal.mT, columns, upper=True)
    feedback = feedback.permute(2, 0, 1).contiguous().view(width, width)
    index = torch.arange(groups, device=feedback.device)
    feedback.view(groups, tile, groups, tile)[index, :, index, :] = 0
    return feedback.float()


def _wavefront_tiles(weight, input_feedback, output_feedback, tile_rows, encode):
    """Quantize each tile after its right, lower and lower-right neighbors."""
    n, k = weight.shape
    device = weight.device
    row_count, col_count = n // tile_rows, k // TILE_COLUMNS
    row_offsets = torch.arange(tile_rows, device=device)
    col_offsets = torch.arange(TILE_COLUMNS, device=device)
    error = torch.zeros_like(weight)
    left_error = torch.zeros_like(weight)
    for diagonal in range(row_count + col_count - 2, -1, -1):
        row_tiles = torch.arange(max(0, diagonal - col_count + 1), min(row_count, diagonal + 1), device=device)
        col_tiles = diagonal - row_tiles
        rows = row_tiles[:, None] * tile_rows + row_offsets
        cols = col_tiles[:, None] * TILE_COLUMNS + col_offsets
        indices = (rows[:, :, None], cols[:, None, :])
        source = weight[indices]
        # left_error is L_out.T @ E with a block identity diagonal.
        # The right update therefore includes the bilinear cross term.
        below = torch.bmm(output_feedback[:, rows].permute(1, 2, 0), error[:, cols].permute(1, 0, 2))
        right = torch.bmm(left_error[rows], input_feedback[:, cols].permute(1, 0, 2))
        reconstructed = encode(source + below + right, row_tiles, col_tiles)
        residual = source - reconstructed
        error[indices] = residual
        left_error[indices] = residual + below


@torch.no_grad()
def quantize_calibrated_weight(weight, spec, sketch, *, device, seed, name, damping=0.01, loading=0.9):
    """Return rotated tile streams and two Rademacher sign vectors."""
    n, k = weight.shape
    tile_rows = validate_tile_shape(k, n, spec.block_size)
    generator = torch.Generator(device=device).manual_seed(seed + zlib.crc32(name.encode()))
    input_signs = torch.randint(0, 2, (k,), generator=generator, dtype=torch.int8, device=device).mul_(2).sub_(1)
    output_signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int8, device=device).mul_(2).sub_(1)
    table = gaussian_table(spec, device)
    coefficients = torch.tensor(spec.coefficients(), dtype=torch.int32, device=device)
    decoded = _lookup(_indices(spec, device), decode_e2m5(table))
    limits = _codebook_limits(decoded)
    loading = decoded.square().mean().sqrt() * loading
    native = native_ops(device)
    if native is not None and spec.trellis_depth == 16:
        decoded = decoded.half()

    shape = (n // tile_rows, k // TILE_COLUMNS)
    packed = torch.empty((*shape, spec.block_bytes), dtype=torch.uint8, device=device)
    scales = torch.empty(shape, dtype=torch.float32, device=device)
    valid = torch.ones((), dtype=torch.bool, device=device)

    def encode(target, rows, cols):
        blocks = target.flatten(1)
        scale, finite = _block_scales(blocks, spec.block_size, loading, limits)
        valid.logical_and_(finite)
        paths = _encode_scaled_blocks(blocks, scale, decoded, spec, native)
        packed[rows, cols] = pack_states(paths, spec)
        scales[rows, cols] = scale
        return (decoded[paths.long()].float() * scale[:, None]).reshape_as(target)

    with tf32_high_precision_guard():
        rotated = rotate_matrix(weight.to(device=device, dtype=torch.float32), input_signs, output_signs)
        input_feedback = block_feedback(sketch.input, input_signs, TILE_COLUMNS, damping)
        output_feedback = block_feedback(sketch.output, output_signs, tile_rows, damping)
        _wavefront_tiles(rotated, input_feedback, output_feedback, tile_rows, encode)
    if not bool(valid):
        raise ValueError("CPTC YAQA cannot encode non-finite weights or feedback")
    return packed, table, coefficients, scales, input_signs, output_signs
