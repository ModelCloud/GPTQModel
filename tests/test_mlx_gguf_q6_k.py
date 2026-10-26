# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Validate the packed MLX GGUF Q6_K decode kernel on Qwen3.8-27B shapes."""

import gc
import sys

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


if sys.platform != "darwin":
    pytest.skip("MLX Metal tests require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from gptqmodel.nn_modules.qlinear.mlx_gguf import MlxGGUFQ6KLinear  # noqa: E402
from gptqmodel.utils.mlx_packing import _pack_rows  # noqa: E402


def _q6_k_fixture(out_features, in_features, *, bias=False, patterns=4):
    positions = np.arange(in_features, dtype=np.uint32)
    phases = np.arange(patterns, dtype=np.uint32)[:, None]
    codes = ((positions[None, :] * (phases * 2 + 5) + phases * 11 + 3) & 63).astype(np.uint8)
    groups = np.arange(in_features // 16, dtype=np.float32)[None, :]
    scales = (0.0007 + ((groups + phases) % 13) * 0.000025).astype(np.float32)
    biases = (-31.5 * scales + ((groups + phases) % 3 - 1) * 0.0001).astype(np.float32)
    output_pattern = np.arange(out_features) % patterns

    layer = MlxGGUFQ6KLinear(in_features, out_features, bias=bias)
    layer.weight = mx.array(_pack_rows(codes, 6)[output_pattern])
    layer.scales_even = mx.array(scales[:, ::2][output_pattern])
    layer.scales_odd = mx.array(scales[:, 1::2][output_pattern])
    layer.biases_even = mx.array(biases[:, ::2][output_pattern])
    layer.biases_odd = mx.array(biases[:, 1::2][output_pattern])
    layer_bias = None
    if bias:
        layer_bias = (((output_pattern % 7) - 3) * 0.003).astype(np.float16)
        layer.bias = mx.array(layer_bias)
    reference = codes.astype(np.float64) * np.repeat(scales.astype(np.float64), 16, axis=1)
    reference += np.repeat(biases.astype(np.float64), 16, axis=1)
    return layer, reference, output_pattern, layer_bias


def _rounded_oracle(x, reference, output_pattern, layer_bias, dtype):
    torch_input = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    values = torch_input @ torch.from_numpy(reference).double().T
    values = values[:, torch.from_numpy(output_pattern)]
    if layer_bias is not None:
        values += torch.from_numpy(layer_bias).double()
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    return values.to(target).float().numpy(), values.numpy()


def _unrounded_output(layer, x):
    rows = x.size // layer.input_dims
    output_dims = layer.weight.shape[0]
    if x.dtype == mx.bfloat16 and 1 < rows <= 16 and output_dims <= 2048:
        return layer._packed_matmul(
            x, rows, min(rows, 4), 32, output_dtype=mx.float32,
        )
    return layer(x.astype(mx.float32))


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("bias", (False, True), ids=("no_bias", "bias"))
@pytest.mark.parametrize("rows", (1, 3), ids=("decode", "small_prefill"))
def test_q6_k_matches_independent_torch_boundary_oracle(
    dtype, bias, rows, record_property,
):
    layer, reference, output_pattern, layer_bias = _q6_k_fixture(96, 256, bias=bias)
    torch_dtype = torch.float16 if dtype == mx.float16 else torch.bfloat16
    base_tensor = torch.linspace(-0.2, 0.2, 256).to(torch_dtype)
    upper = torch.nextafter(base_tensor, torch.full_like(base_tensor, float("inf")))
    base = base_tensor.float().numpy()
    sources = [base, upper.float().numpy(), -base]
    sources[0][127] = np.float32(-0.0)
    sources[0][128] = np.float32(0.0)
    x = mx.array(np.stack(sources[:rows])).astype(dtype)
    actual = layer(x)
    internal = _unrounded_output(layer, x)
    mx.eval(actual, internal)
    rounded, raw = _rounded_oracle(x, reference, output_pattern, layer_bias, dtype)
    visible = np.asarray(actual.astype(mx.float32))
    internal_values = np.asarray(internal)
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


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_q6_k_decode_qwen38_shapes_preserve_dtype_and_accuracy(
    name, out_features, in_features, dtype, record_property,
):
    layer, reference, output_pattern, _ = _q6_k_fixture(
        out_features, in_features, patterns=2,
    )
    positions = np.arange(in_features, dtype=np.float32)
    source = (np.sin(positions * 0.013) * 0.08 + np.cos(positions * 0.007) * 0.04)[None]
    x = mx.array(source).astype(dtype)
    actual = layer(x)
    mx.eval(actual)
    rounded, raw = _rounded_oracle(x, reference, output_pattern, None, dtype)
    visible = np.asarray(actual.astype(mx.float32))
    rounded_error = float(np.max(np.abs(visible - rounded)))
    raw_error = float(np.max(np.abs(visible - raw)))
    record_property("projection", name)
    record_property("max_abs_vs_rounded_torch", rounded_error)
    record_property("max_abs_vs_fp64_torch", raw_error)
    assert actual.dtype == dtype
    np.testing.assert_allclose(visible, rounded, rtol=2e-3, atol=2e-3)
    del layer, actual
    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_q6_k_prefill_qwen38_shapes_matches_independent_torch_oracle(
    name, out_features, in_features, dtype, record_property,
):
    layer, reference, output_pattern, _ = _q6_k_fixture(
        out_features, in_features, patterns=2,
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
    mx.eval(actual, internal)
    rounded, raw = _rounded_oracle(x, reference, output_pattern, None, dtype)
    visible = np.asarray(actual.astype(mx.float32))
    internal_values = np.asarray(internal)
    rounded_error = float(np.max(np.abs(visible - rounded)))
    raw_error = float(np.max(np.abs(visible - raw)))
    internal_error = float(np.max(np.abs(internal_values - raw)))
    record_property("projection", name)
    record_property("max_abs_internal_vs_fp64_torch", internal_error)
    record_property("max_abs_visible_vs_rounded_torch", rounded_error)
    record_property("max_abs_visible_vs_fp64_torch", raw_error)
    assert actual.dtype == dtype
    assert internal.dtype == mx.float32
    np.testing.assert_allclose(internal_values, raw, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(visible, rounded, rtol=2e-3, atol=2e-3)
    del layer, actual, internal
    gc.collect()
    mx.clear_cache()


def test_q6_k_validates_width_and_empty_output():
    layer = MlxGGUFQ6KLinear(256, 96)
    with pytest.raises(ValueError, match="expected input width 256"):
        layer(mx.zeros((1, 128), dtype=mx.float16))
    result = layer(mx.zeros((0, 256), dtype=mx.bfloat16))
    assert result.shape == (0, 96)
    assert result.dtype == mx.bfloat16
