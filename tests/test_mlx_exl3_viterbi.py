# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Independent Torch-oracle checks for native MLX EXL3 path search."""

import gc
import sys
from functools import lru_cache

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_codebook import _torch_codebook_oracle

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3 import exl3_quantize_tiles_mlx


@lru_cache(maxsize=24)
def _torch_viterbi_tables(bits, codebook):
    edges = 1 << (16 - bits)
    states = torch.arange(1 << 16, dtype=torch.int64).reshape(1 << bits, edges)
    incoming = states >> bits
    decoded = _torch_codebook_oracle(codebook).to(torch.float16)[states]
    out_edges = torch.arange(edges, dtype=torch.int64)
    return incoming, decoded, out_edges


def _torch_viterbi_oracle_tensors(input_tiles, bits, codebook):
    """Reproduce EXL3's two-pass tail-biting search with independent Torch math."""
    if isinstance(input_tiles, torch.Tensor):
        source = input_tiles.to(dtype=torch.float32).contiguous()
    else:
        source = torch.from_numpy(np.ascontiguousarray(input_tiles, dtype=np.float32))
    original_shape = source.shape
    weights = source.reshape(-1, 256).to(torch.float16)
    incoming, decoded, out_edges = _torch_viterbi_tables(bits, codebook)
    tile_count = weights.shape[0]
    edges = incoming.shape[1]
    state_shift = 16 - bits
    history = torch.empty((tile_count, 256, edges), dtype=torch.int16)

    def forward(roll, fixed_incoming=None):
        costs = None
        for step in range(256):
            ri = (step + roll) & 255
            delta = (decoded.unsqueeze(0) - weights[:, ri, None, None]).to(
                torch.float16
            )
            squared = (delta.to(torch.float64) * delta.to(torch.float64)).to(
                torch.float16
            )
            if step == 0:
                candidates = squared
                if fixed_incoming is not None:
                    allowed = incoming.unsqueeze(0) == fixed_incoming[:, None, None]
                    candidates = torch.where(allowed, candidates, torch.inf)
            else:
                candidates = (
                    delta.to(torch.float64) * delta.to(torch.float64)
                    + costs[:, incoming].to(torch.float64)
                ).to(torch.float16)
            costs, symbols = candidates.min(dim=1)
            predecessors = (
                (symbols.to(torch.int64) << state_shift) + out_edges
            ) >> bits
            history[:, ri, :] = predecessors.to(torch.int16)
        return costs

    costs = forward(128)
    edge = costs.argmin(dim=1).to(torch.int64)
    for step in range(255, -1, -1):
        ri = (step + 128) & 255
        edge = history[:, ri, :].gather(1, edge[:, None]).squeeze(1).to(torch.int64)
        if ri == 0:
            break

    initial_edge = edge
    forward(0, initial_edge)
    indices = torch.empty((tile_count, 256), dtype=torch.int64)
    edge = initial_edge
    for step in range(255, -1, -1):
        previous = (
            history[:, step, :].gather(1, edge[:, None]).squeeze(1).to(torch.int64)
        )
        indices[:, step] = (previous << bits) | edge
        edge = previous

    lut = _torch_codebook_oracle(codebook).to(torch.float32)
    quantized = lut[indices]
    return quantized.reshape(original_shape), indices.to(torch.int16).reshape(
        original_shape
    )


def _torch_viterbi_oracle(input_tiles, bits, codebook):
    quantized, indices = _torch_viterbi_oracle_tensors(input_tiles, bits, codebook)
    return quantized.numpy(), indices.numpy()


def _assert_matches_oracle(
    values, bits, codebook, *, workspace_bytes=256 << 20, err_msg=None
):
    expected_values, expected_indices = _torch_viterbi_oracle(values, bits, codebook)
    actual_values, actual_indices = exl3_quantize_tiles_mlx(
        mx.array(values),
        bits=bits,
        codebook=codebook,
        workspace_bytes=workspace_bytes,
    )
    np.testing.assert_array_equal(
        np.asarray(actual_indices), expected_indices, err_msg=err_msg
    )
    np.testing.assert_array_equal(
        np.asarray(actual_values), expected_values, err_msg=err_msg
    )


@pytest.mark.parametrize("bits", range(1, 9))
def test_exl3_viterbi_all_bit_widths_match_torch_oracle(bits):
    rng = np.random.default_rng(3200 + bits)
    values = rng.normal(0.0, 1.75, (1, 256)).astype(np.float32)
    _assert_matches_oracle(values, bits, "mcg")


@pytest.mark.parametrize("codebook", ["3inst", "mcg", "mul1"])
def test_exl3_viterbi_all_codebooks_match_torch_oracle(codebook):
    rng = np.random.default_rng(4421)
    values = rng.normal(0.0, 1.75, (1, 256)).astype(np.float32)
    _assert_matches_oracle(values, 4, codebook)


def _half_rounding_boundary(codebook):
    values = _torch_codebook_oracle(codebook).to(torch.float16)
    candidate_states = torch.arange(16, dtype=torch.int64) << 12
    candidates = values[candidate_states]
    for left in range(candidates.numel()):
        for right in range(left + 1, candidates.numel()):
            midpoint = (
                (
                    candidates[left].to(torch.float64)
                    + candidates[right].to(torch.float64)
                )
                / 2
            ).to(torch.float16)
            delta_left = (candidates[left] - midpoint).to(torch.float16)
            delta_right = (candidates[right] - midpoint).to(torch.float16)
            error_left = (delta_left.to(torch.float64) ** 2).to(torch.float16)
            error_right = (delta_right.to(torch.float64) ** 2).to(torch.float16)
            if error_left == error_right and error_left != 0:
                return np.float16(midpoint.item())
    raise AssertionError(f"no representable tie found for {codebook}")


@pytest.mark.parametrize("codebook", ["3inst", "mcg", "mul1"])
def test_exl3_viterbi_rounding_boundaries_match_torch_oracle(codebook):
    tie = _half_rounding_boundary(codebook)
    below = np.nextafter(tie, np.float16(-np.inf), dtype=np.float16)
    above = np.nextafter(tie, np.float16(np.inf), dtype=np.float16)
    lut = _torch_codebook_oracle(codebook).numpy()
    boundary_values = np.array(
        [below, tie, above, -0.0, 0.0, lut.min(), lut.max()],
        dtype=np.float32,
    )
    values = np.resize(boundary_values, (1, 256)).astype(np.float32, copy=False)
    _assert_matches_oracle(values, 4, codebook)


def test_exl3_viterbi_reuses_bounded_workspace_exactly():
    bits = 4
    edges = 1 << (16 - bits)
    one_slot_bytes = edges * (2 * 2 + 256 * 2)
    rng = np.random.default_rng(5810)
    values = rng.normal(0.0, 1.75, (3, 256)).astype(np.float32)
    _assert_matches_oracle(values, bits, "mcg", workspace_bytes=one_slot_bytes)


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_viterbi_qwen38_projection_oracle(name, out_features, in_features):
    bits = 4
    codebook = "mcg"
    rng = np.random.default_rng(7100 + out_features + in_features)
    source_bank = rng.normal(0.0, 1.75, (4, 256)).astype(np.float32)
    expected_values, expected_indices = _torch_viterbi_oracle(
        source_bank, bits, codebook
    )
    tile_shape = (in_features // 16, out_features // 16, 256)
    tile_count = tile_shape[0] * tile_shape[1]
    values = np.resize(source_bank, (tile_count, 256)).reshape(tile_shape)

    actual_values, actual_indices = exl3_quantize_tiles_mlx(
        mx.array(values), bits=bits, codebook=codebook
    )
    actual_values = np.asarray(actual_values).reshape(-1, 256)
    actual_indices = np.asarray(actual_indices).reshape(-1, 256)
    for start in range(0, tile_count, 8192):
        stop = min(start + 8192, tile_count)
        expected_rows = np.arange(start, stop) % source_bank.shape[0]
        np.testing.assert_array_equal(
            actual_indices[start:stop], expected_indices[expected_rows], err_msg=name
        )
        np.testing.assert_array_equal(
            actual_values[start:stop], expected_values[expected_rows], err_msg=name
        )

    del values, actual_values, actual_indices
    gc.collect()
    mx.clear_cache()


def test_exl3_viterbi_rejects_invalid_inputs():
    valid = mx.zeros((1, 256), dtype=mx.float32)
    for bits in (0, 9, 4.0, True):
        with pytest.raises(ValueError, match="bits"):
            exl3_quantize_tiles_mlx(valid, bits=bits)
    for codebook in ("unknown", 1, None):
        with pytest.raises(ValueError, match="codebook"):
            exl3_quantize_tiles_mlx(valid, bits=4, codebook=codebook)
    with pytest.raises(ValueError, match="rank"):
        exl3_quantize_tiles_mlx(mx.zeros((256,), dtype=mx.float32), bits=4)
    with pytest.raises(ValueError, match="256 values"):
        exl3_quantize_tiles_mlx(mx.zeros((1, 255), dtype=mx.float32), bits=4)
    with pytest.raises(ValueError, match="nonempty"):
        exl3_quantize_tiles_mlx(mx.zeros((0, 256), dtype=mx.float32), bits=4)
    with pytest.raises(ValueError, match="float32"):
        exl3_quantize_tiles_mlx(mx.zeros((1, 256), dtype=mx.float16), bits=4)
    with pytest.raises(ValueError, match="positive integer"):
        exl3_quantize_tiles_mlx(valid, bits=4, workspace_bytes=0)
    with pytest.raises(ValueError, match="at least"):
        exl3_quantize_tiles_mlx(valid, bits=1, workspace_bytes=1 << 20)
