# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPTQ format and packing: ModelCloud.ai, Apache-2.0, https://github.com/ModelCloud/GPTQModel
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Validate packed GPTQ group-16 decode on every bit width and Qwen3.8 shape."""

import gc
import sys

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


if sys.platform != "darwin":
    pytest.skip("MLX Metal tests require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from gptqmodel.nn_modules.qlinear.mlx_gptq import (  # noqa: E402
    MlxGPTQGroup16Linear,
    _uses_packed_prefill,
)
from gptqmodel.nn_modules.qlinear.mlx_group16 import MlxGroup16Linear  # noqa: E402
from gptqmodel.utils.mlx_packing import _pack_rows  # noqa: E402


SOURCE_BITS = (2, 3, 4, 5, 6, 7, 8)


def _group16_fixture(out_features, in_features, source_bits, *, bias=True, patterns=2):
    runtime_bits = 8 if source_bits == 7 else source_bits
    positions = np.arange(in_features, dtype=np.uint32)
    phases = np.arange(patterns, dtype=np.uint32)[:, None]
    codes = (
        positions[None, :] * (phases * 2 + 3) + phases * 7 + 1
    ) & ((1 << source_bits) - 1)
    codes = codes.astype(np.uint8)
    groups = np.arange(in_features // 16, dtype=np.float32)[None, :]
    scales = (0.0007 + ((groups + phases) % 11) * 0.00003).astype(np.float32)
    zeros = ((groups.astype(np.uint32) + phases) & ((1 << source_bits) - 1)).astype(np.float32)
    offsets = -zeros * scales
    output_pattern = np.arange(out_features) % patterns

    layer = MlxGPTQGroup16Linear(
        in_features, out_features, bits=runtime_bits, bias=bias,
    )
    layer.weight = mx.array(_pack_rows(codes, runtime_bits)[output_pattern])
    layer.scales_even = mx.array(scales[:, ::2][output_pattern])
    layer.scales_odd = mx.array(scales[:, 1::2][output_pattern])
    layer.biases_even = mx.array(offsets[:, ::2][output_pattern])
    layer.biases_odd = mx.array(offsets[:, 1::2][output_pattern])
    layer_bias = None
    if bias:
        layer_bias = (((output_pattern % 7) - 3) * 0.003).astype(np.float16)
        layer.bias = mx.array(layer_bias)
    reference = codes.astype(np.float64) * np.repeat(scales.astype(np.float64), 16, axis=1)
    reference += np.repeat(offsets.astype(np.float64), 16, axis=1)
    return layer, reference, output_pattern, layer_bias


def _rounded_oracle(x, reference, output_pattern, layer_bias, dtype):
    source = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    values = (source @ torch.from_numpy(reference).double().T)[:, torch.from_numpy(output_pattern)]
    if layer_bias is not None:
        values += torch.from_numpy(layer_bias).double()
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    return values.to(target).float().numpy(), values.numpy()


def _unrounded_output(layer, x):
    rows = x.size // layer.input_dims
    output_dims = layer.weight.shape[0]
    if _uses_packed_prefill(
        x.dtype, rows, layer.input_dims, output_dims,
    ):
        return layer._packed_prefill(
            x, rows, min(rows, 4), output_dtype=mx.float32,
        )
    return layer(x.astype(mx.float32))


def _assert_within_one_output_ulp(actual, rounded, dtype):
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    actual_target = torch.from_numpy(actual).to(target)
    rounded_target = torch.from_numpy(rounded).to(target)
    lower = torch.nextafter(
        rounded_target, torch.full_like(rounded_target, float("-inf")),
    )
    upper = torch.nextafter(
        rounded_target, torch.full_like(rounded_target, float("inf")),
    )
    assert torch.all(
        (actual_target == rounded_target)
        | (actual_target == lower)
        | (actual_target == upper)
    )


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("bias", (False, True), ids=("no_bias", "bias"))
@pytest.mark.parametrize("source_bits", SOURCE_BITS, ids=lambda bits: f"{bits}bit")
def test_gptq_group16_random_decode_matches_independent_torch_oracle(
    source_bits, bias, dtype, record_property,
):
    runtime_bits = 8 if source_bits == 7 else source_bits
    rng = np.random.default_rng(1600 + source_bits + 17 * int(bias))
    in_features, out_features = 256, 96
    codes = rng.integers(
        0, 1 << source_bits, (out_features, in_features), dtype=np.uint8,
    )
    scales = rng.uniform(0.0005, 0.003, (out_features, in_features // 16)).astype(np.float32)
    zeros = rng.integers(
        0, 1 << source_bits, scales.shape, dtype=np.uint8,
    ).astype(np.float32)
    offsets = -zeros * scales
    layer = MlxGPTQGroup16Linear(
        in_features, out_features, bits=runtime_bits, bias=bias,
    )
    layer.weight = mx.array(_pack_rows(codes, runtime_bits))
    layer.scales_even = mx.array(scales[:, ::2])
    layer.scales_odd = mx.array(scales[:, 1::2])
    layer.biases_even = mx.array(offsets[:, ::2])
    layer.biases_odd = mx.array(offsets[:, 1::2])
    layer_bias = None
    if bias:
        layer_bias = rng.uniform(-0.01, 0.01, out_features).astype(np.float16)
        layer.bias = mx.array(layer_bias)
    reference = codes.astype(np.float64) * np.repeat(scales.astype(np.float64), 16, axis=1)
    reference += np.repeat(offsets.astype(np.float64), 16, axis=1)
    x = mx.array(rng.normal(0, 0.2, (1, in_features)).astype(np.float32)).astype(dtype)
    actual = layer(x)
    internal = _unrounded_output(layer, x)
    mx.eval(actual, internal)
    source = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    expected = source @ torch.from_numpy(reference).double().T
    if layer_bias is not None:
        expected += torch.from_numpy(layer_bias).double()
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    rounded = expected.to(target).float().numpy()
    visible = np.asarray(actual.astype(mx.float32))
    internal_values = np.asarray(internal)
    record_property(
        "max_abs_internal_vs_fp64_torch",
        float(np.max(np.abs(internal_values - expected.numpy()))),
    )
    record_property("max_abs_vs_rounded_torch", float(np.max(np.abs(visible - rounded))))
    record_property("max_abs_vs_fp64_torch", float(np.max(np.abs(visible - expected.numpy()))))
    assert actual.dtype == dtype
    assert internal.dtype == mx.float32
    np.testing.assert_allclose(internal_values, expected.numpy(), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(visible, rounded, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("source_bits", SOURCE_BITS, ids=lambda bits: f"{bits}bit")
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_gptq_group16_qwen38_decode_preserves_dtype_and_accuracy(
    name, out_features, in_features, source_bits, dtype, record_property,
):
    layer, reference, output_pattern, layer_bias = _group16_fixture(
        out_features, in_features, source_bits,
    )
    positions = np.arange(in_features, dtype=np.float32)
    source = (np.sin(positions * 0.013) * 0.08 + np.cos(positions * 0.007) * 0.04)[None]
    x = mx.array(source).astype(dtype)
    actual = layer(x)
    internal = _unrounded_output(layer, x)
    mx.eval(actual, internal)
    rounded, raw = _rounded_oracle(x, reference, output_pattern, layer_bias, dtype)
    visible = np.asarray(actual.astype(mx.float32))
    internal_values = np.asarray(internal)
    record_property("projection", name)
    record_property("source_bits", source_bits)
    record_property(
        "max_abs_internal_vs_fp64_torch",
        float(np.max(np.abs(internal_values - raw))),
    )
    record_property("max_abs_vs_rounded_torch", float(np.max(np.abs(visible - rounded))))
    record_property("max_abs_vs_fp64_torch", float(np.max(np.abs(visible - raw))))
    assert actual.dtype == dtype
    assert internal.dtype == mx.float32
    np.testing.assert_allclose(internal_values, raw, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(visible, rounded, rtol=2e-3, atol=2e-3)
    del layer, actual, internal
    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("bias", (False, True), ids=("no_bias", "bias"))
@pytest.mark.parametrize("source_bits", SOURCE_BITS, ids=lambda bits: f"{bits}bit")
@pytest.mark.parametrize("rows", (2, 5, 8, 12, 16), ids=lambda rows: f"rows{rows}")
def test_gptq_group16_small_prefill_matches_boundary_oracle(
    rows, source_bits, dtype, bias, record_property,
):
    layer, reference, output_pattern, layer_bias = _group16_fixture(
        96, 256, source_bits, bias=bias,
    )
    torch_dtype = torch.float16 if dtype == mx.float16 else torch.bfloat16
    base_tensor = torch.linspace(-0.2, 0.2, 256).to(torch_dtype)
    upper = torch.nextafter(base_tensor, torch.full_like(base_tensor, float("inf")))
    base = base_tensor.float().numpy()
    patterns = [base, upper.float().numpy(), -base]
    sources = [patterns[row % len(patterns)].copy() for row in range(rows)]
    sources[0][127] = np.float32(-0.0)
    sources[0][128] = np.float32(0.0)
    x = mx.array(np.stack(sources)).astype(dtype)
    actual = layer._packed_prefill(x, rows, min(rows, 4))
    internal = layer._packed_prefill(
        x, rows, min(rows, 4), output_dtype=mx.float32,
    )
    mx.eval(actual, internal)
    rounded, raw = _rounded_oracle(
        x, reference, output_pattern, layer_bias, dtype,
    )
    visible = np.asarray(actual.astype(mx.float32))
    internal_values = np.asarray(internal)
    record_property(
        "max_abs_internal_vs_fp64_torch",
        float(np.max(np.abs(internal_values - raw))),
    )
    record_property(
        "max_abs_visible_vs_rounded_torch",
        float(np.max(np.abs(visible - rounded))),
    )
    record_property(
        "max_abs_visible_vs_fp64_torch",
        float(np.max(np.abs(visible - raw))),
    )
    assert actual.dtype == dtype
    assert internal.dtype == mx.float32
    np.testing.assert_allclose(internal_values, raw, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(visible, rounded, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("source_bits", SOURCE_BITS, ids=lambda bits: f"{bits}bit")
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_gptq_group16_qwen38_prefill_matches_independent_torch_oracle(
    name, out_features, in_features, source_bits, dtype, record_property,
):
    layer, reference, output_pattern, layer_bias = _group16_fixture(
        out_features, in_features, source_bits,
    )
    positions = np.arange(in_features, dtype=np.float32)
    source = np.stack([
        np.sin(positions * (0.011 + row * 0.0001)) * 0.08
        + np.cos(positions * (0.007 + row * 0.0001)) * 0.04
        for row in range(16)
    ])
    x = mx.array(source).astype(dtype)
    actual = layer(x)
    internal = _unrounded_output(layer, x)
    baseline = MlxGroup16Linear.__call__(layer, x)
    mx.eval(actual, internal, baseline)
    rounded, raw = _rounded_oracle(x, reference, output_pattern, layer_bias, dtype)
    visible = np.asarray(actual.astype(mx.float32))
    internal_values = np.asarray(internal)
    internal_error = float(np.max(np.abs(internal_values - raw)))
    rounded_error = float(np.max(np.abs(visible - rounded)))
    raw_error = float(np.max(np.abs(visible - raw)))
    record_property("projection", name)
    record_property("source_bits", source_bits)
    record_property("max_abs_internal_vs_fp64_torch", internal_error)
    record_property("max_abs_visible_vs_rounded_torch", rounded_error)
    record_property("max_abs_visible_vs_fp64_torch", raw_error)
    assert actual.dtype == dtype
    assert internal.dtype == mx.float32
    np.testing.assert_allclose(internal_values, raw, rtol=2e-3, atol=2e-3)
    native = _uses_packed_prefill(dtype, 16, in_features, out_features)
    if native:
        np.testing.assert_allclose(visible, rounded, rtol=2e-3, atol=2e-3)
    else:
        np.testing.assert_array_equal(
            visible, np.asarray(baseline.astype(mx.float32)),
        )
        within_tolerance = np.allclose(
            visible, rounded, rtol=2e-3, atol=2e-3,
        )
        record_property(
            "fallback_one_ulp_adjudication", not within_tolerance,
        )
        if not within_tolerance:
            _assert_within_one_output_ulp(visible, rounded, dtype)
    del layer, actual, internal
    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize(
    "rows,expected",
    (
        (1, False), (2, True), (8, True), (9, False), (10, False),
        (11, False), (12, True),
        (13, False), (14, False), (15, False), (16, True), (17, False),
    ),
)
def test_gptq_group16_prefill_gate_excludes_slower_partial_tiles(rows, expected):
    assert _uses_packed_prefill(mx.bfloat16, rows, 5120, 1024) is expected
    assert not _uses_packed_prefill(mx.float16, rows, 5120, 1024)
    assert not _uses_packed_prefill(mx.bfloat16, rows, 4096, 1024)
    assert not _uses_packed_prefill(mx.bfloat16, rows, 5120, 2048)


def test_gptq_group16_validates_width_and_empty_output():
    layer = MlxGPTQGroup16Linear(256, 96, bits=4)
    with pytest.raises(ValueError, match="expected input width 256"):
        layer(mx.zeros((1, 128), dtype=mx.float16))
    result = layer(mx.zeros((0, 256), dtype=mx.bfloat16))
    assert result.shape == (0, 96)
    assert result.dtype == mx.bfloat16
