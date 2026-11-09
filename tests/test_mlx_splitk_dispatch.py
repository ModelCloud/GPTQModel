# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Checks that split-K avoidance keeps MLX affine output accurate."""

import sys

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_splitk_dispatch import (  # noqa: E402
    _architecture,
    _dispatch_version,
    _mirrors_installed_dispatch,
    _qmv_batch_limit,
    _split_k,
    padded_rows,
    quantized_matmul,
)


@pytest.fixture(autouse=True)
def _reset_dispatch_caches():
    yield
    _dispatch_version.cache_clear()
    padded_rows.cache_clear()


def _use_mlx_version(version):
    mx.__version__ = version
    _dispatch_version.cache_clear()
    padded_rows.cache_clear()


def _require_mirrored_dispatch():
    if not _mirrors_installed_dispatch():
        pytest.skip("installed MLX no longer uses the dispatch mirrored here")


def _reference(x, weight, scales, biases, group_size, bits, dtype):
    """Torch float64 oracle over the dequantized weights."""
    decoded = mx.dequantize(
        weight, scales, biases, group_size=group_size, bits=bits,
    ).astype(mx.float32)
    mx.eval(decoded)
    lhs = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    rhs = torch.from_numpy(np.asarray(decoded)).double()
    return lhs @ rhs.T


def _check_shape(rows, input_dims, output_dims, group_size, bits, dtype):
    source = mx.random.normal((output_dims, input_dims)).astype(mx.float16)
    weight, scales, biases = mx.quantize(
        source, group_size=group_size, bits=bits,
    )
    x = mx.random.normal((rows, input_dims)).astype(dtype)
    baseline = mx.quantized_matmul(
        x, weight, scales=scales, biases=biases,
        transpose=True, group_size=group_size, bits=bits,
    )
    actual = quantized_matmul(
        x, weight, scales, biases, group_size=group_size, bits=bits,
    )
    assert actual.dtype == baseline.dtype
    oracle = _reference(x, weight, scales, biases, group_size, bits, dtype)
    observed = torch.from_numpy(np.asarray(actual.astype(mx.float32))).double()
    relative = float((observed - oracle).norm() / oracle.norm())
    assert relative <= 2e-3
    # The padding only reorders the K reduction, so it must stay far closer to
    # the baseline than the acceptance limit.
    reference = torch.from_numpy(np.asarray(baseline.astype(mx.float32))).double()
    assert float((observed - reference).norm() / reference.norm()) <= 2e-3
    return relative


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("group_size,bits", [(64, 4), (128, 4), (32, 4), (64, 8)])
def test_splitk_dispatch_matches_oracle_for_all_projections(dtype, group_size, bits):
    for _, output_dims, input_dims in QWEN38_27B_PROJECTIONS:
        for rows in (12, 16, 32):
            _check_shape(
                rows, input_dims, output_dims, group_size, bits, dtype,
            )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("group_size,bits", [(64, 4), (64, 8)])
def test_splitk_dispatch_matches_oracle_for_qmv_wide_rows(dtype, group_size, bits):
    # Six to eleven rows are the batches qmv_wide splits into multiple weight
    # passes, so they are the rows the padded matrix kernel replaces.
    for _, output_dims, input_dims in QWEN38_27B_PROJECTIONS:
        for rows in (6, 8, 11):
            _check_shape(
                rows, input_dims, output_dims, group_size, bits, dtype,
            )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_splitk_dispatch_preserves_leading_batch_dims(dtype):
    input_dims, output_dims, group_size, bits = 6144, 5120, 64, 4
    source = mx.random.normal((output_dims, input_dims)).astype(mx.float16)
    weight, scales, biases = mx.quantize(source, group_size=group_size, bits=bits)
    x = mx.random.normal((2, 8, input_dims)).astype(dtype)
    actual = quantized_matmul(
        x, weight, scales, biases, group_size=group_size, bits=bits,
    )
    assert actual.shape == (2, 8, output_dims)
    flat = quantized_matmul(
        x.reshape(16, input_dims), weight, scales, biases,
        group_size=group_size, bits=bits,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.reshape(16, output_dims).astype(mx.float32)),
        np.asarray(flat.astype(mx.float32)),
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_splitk_dispatch_is_identity_without_splitk(dtype):
    # gate_proj's split_k is already 1, so the dispatcher must not pad.
    input_dims, output_dims, group_size, bits = 5120, 17408, 64, 4
    assert padded_rows(16, input_dims, output_dims, group_size, dtype) is None
    source = mx.random.normal((output_dims, input_dims)).astype(mx.float16)
    weight, scales, biases = mx.quantize(source, group_size=group_size, bits=bits)
    x = mx.random.normal((16, input_dims)).astype(dtype)
    baseline = mx.quantized_matmul(
        x, weight, scales=scales, biases=biases,
        transpose=True, group_size=group_size, bits=bits,
    )
    actual = quantized_matmul(
        x, weight, scales, biases, group_size=group_size, bits=bits,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(baseline.astype(mx.float32)),
    )


def test_splitk_split_choice_matches_mlx_dispatch():
    # Mirrors mlx/backend/metal/quantized.cpp: qmm_splitk targets 512
    # threadgroups of 32x32 and then rounds down until K divides evenly.
    assert _split_k(12, 6144, 5120, 64) == 3
    assert _split_k(32, 6144, 5120, 64) == 3
    assert _split_k(33, 6144, 5120, 64) == 1
    assert _split_k(12, 17408, 5120, 64) == 2
    assert _split_k(32, 17408, 5120, 64) == 2
    assert _split_k(12, 5120, 17408, 64) == 1
    assert _split_k(12, 5120, 1024, 64) == 16


def test_splitk_padding_target_clears_splitk():
    _require_mirrored_dispatch()
    generation, _ = _architecture()
    if generation < 13:
        pytest.skip("split-K avoidance targets Apple GPU generation 13+")
    for input_dims, output_dims in ((6144, 5120), (17408, 5120)):
        for rows in (12, 16, 24, 32):
            target = padded_rows(rows, input_dims, output_dims, 64, mx.float16)
            assert target is not None
            assert target > rows
            assert _split_k(target, input_dims, output_dims, 64) == 1
            # The pad must stay modest so the extra rows cannot dominate.
            assert target <= 8 * rows


def test_splitk_padding_declines_when_already_unsplit():
    for input_dims, output_dims in ((5120, 17408), (5120, 10240)):
        for rows in (12, 16, 32):
            assert _split_k(rows, input_dims, output_dims, 64) == 1
            assert padded_rows(rows, input_dims, output_dims, 64, mx.float16) is None


def test_splitk_padding_declines_inside_one_qmv_wide_tile():
    # qmv_wide reads the whole weight matrix once while the batch fits a single
    # five-row input-vector tile, so padding would only add work.
    generation, _ = _architecture()
    if generation < 15:
        pytest.skip("qmv_wide handles affine weights on Apple GPU generation 15+")
    for input_dims, output_dims in ((5120, 17408), (6144, 5120), (17408, 5120)):
        for rows in range(1, 6):
            for dtype in (mx.float16, mx.bfloat16):
                assert padded_rows(rows, input_dims, output_dims, 64, dtype) is None


def test_splitk_padding_escapes_multi_tile_qmv_wide():
    _require_mirrored_dispatch()
    # Past five rows qmv_wide re-reads the weights once per tile, which costs
    # more than the rows needed to reach the single-pass matrix kernel.
    generation, _ = _architecture()
    if generation < 15:
        pytest.skip("qmv_wide handles affine weights on Apple GPU generation 15+")
    for input_dims, output_dims in ((5120, 17408), (6144, 5120), (17408, 5120)):
        limit = _qmv_batch_limit(input_dims, output_dims)
        for rows in (6, 7, 8, 9, 10, 11):
            assert rows < limit
            target = padded_rows(rows, input_dims, output_dims, 64, mx.float16)
            assert target is not None
            assert target > rows
            assert _split_k(target, input_dims, output_dims, 64) == 1
            assert target <= 8 * rows


def test_splitk_padding_bf16_crossover_is_later():
    _require_mirrored_dispatch()
    # BF16 decode stays ALU bound in qmv_wide, so the matrix kernel only wins
    # once the vector kernel has accumulated enough extra weight passes.
    generation, _ = _architecture()
    if generation < 15:
        pytest.skip("qmv_wide handles affine weights on Apple GPU generation 15+")
    for input_dims, output_dims in ((5120, 17408), (6144, 5120)):
        for rows in range(6, 9):
            assert padded_rows(rows, input_dims, output_dims, 64, mx.bfloat16) is None
        for rows in (9, 10, 11):
            target = padded_rows(rows, input_dims, output_dims, 64, mx.bfloat16)
            assert target is not None
            assert _split_k(target, input_dims, output_dims, 64) == 1


def test_splitk_padding_never_exceeds_two_row_tiles():
    _require_mirrored_dispatch()
    # qmv_wide makes two or three weight passes at these row counts, so the
    # matrix kernel must not be handed a batch that needs more than two.
    generation, _ = _architecture()
    if generation < 15:
        pytest.skip("qmv_wide handles affine weights on Apple GPU generation 15+")
    # Both shapes only drop split-K at 65 rows (three tiles), so they decline.
    assert padded_rows(11, 4096, 4096, 64, mx.float16) is None
    assert padded_rows(11, 14336, 4096, 64, mx.float16) is None
    # Two tiles are still allowed where they clear split-K.
    assert padded_rows(11, 8192, 8192, 64, mx.float16) == 33


def test_splitk_dispatch_version_gate():
    # This module mirrors the MLX 0.32.4 dispatch. 0.32.5 replaces qmv_wide with
    # qmv_mma and routes split-K through a NAX kernel, so the padding rules stop
    # applying and the dispatcher must be left alone.
    original = mx.__version__
    try:
        _use_mlx_version("0.32.4")
        assert _dispatch_version() == (0, 32, 4)
        assert _mirrors_installed_dispatch() is True
        for version in ("0.32.5", "0.33.0", "1.0.0"):
            _use_mlx_version(version)
            assert _mirrors_installed_dispatch() is False
        _use_mlx_version("nightly")
        assert _dispatch_version() is None
        assert _mirrors_installed_dispatch() is False
    finally:
        _use_mlx_version(original)


def test_splitk_padding_declines_on_newer_mlx():
    original = mx.__version__
    try:
        _use_mlx_version("0.32.5")
        for input_dims, output_dims in ((5120, 17408), (6144, 5120), (8192, 8192)):
            for rows in (6, 8, 11, 16, 32):
                assert padded_rows(rows, input_dims, output_dims, 64, mx.float16) is None
    finally:
        _use_mlx_version(original)


def test_splitk_padding_declines_for_narrow_outputs():
    # k_proj/v_proj: split_k is large but the padded rows cost more than it.
    for rows in (6, 9, 12, 16, 32):
        assert _split_k(rows, 5120, 1024, 64) > 1
        assert padded_rows(rows, 5120, 1024, 64, mx.float16) is None
