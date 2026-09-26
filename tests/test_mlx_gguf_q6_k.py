# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# MLX Metal runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx
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
from gptqmodel.nn_modules.qlinear.mlx_group16 import MlxGroup16Linear  # noqa: E402
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


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("bias", (False, True), ids=("no_bias", "bias"))
def test_q6_k_decode_matches_independent_torch_oracle(dtype, bias, record_property):
    layer, reference, output_pattern, layer_bias = _q6_k_fixture(96, 256, bias=bias)
    x = mx.array(np.linspace(-0.2, 0.2, 256, dtype=np.float32)[None]).astype(dtype)
    actual = layer(x)
    mx.eval(actual)
    rounded, raw = _rounded_oracle(x, reference, output_pattern, layer_bias, dtype)
    visible = np.asarray(actual.astype(mx.float32))
    record_property("max_abs_vs_rounded_torch", float(np.max(np.abs(visible - rounded))))
    record_property("max_abs_vs_fp64_torch", float(np.max(np.abs(visible - raw))))
    assert actual.dtype == dtype
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
def test_q6_k_prefill_uses_exact_group16_path(dtype):
    candidate, _, _, _ = _q6_k_fixture(96, 256)
    baseline = MlxGroup16Linear(256, 96, bits=6)
    for name in ("weight", "scales_even", "scales_odd", "biases_even", "biases_odd"):
        setattr(baseline, name, getattr(candidate, name))
    x = mx.full((3, 256), 0.03125, dtype=dtype)
    expected, actual = baseline(x), candidate(x)
    mx.eval(expected, actual)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)), np.asarray(expected.astype(mx.float32)),
    )


def test_q6_k_validates_width_and_empty_output():
    layer = MlxGGUFQ6KLinear(256, 96)
    with pytest.raises(ValueError, match="expected input width 256"):
        layer(mx.zeros((1, 128), dtype=mx.float16))
    result = layer(mx.zeros((0, 256), dtype=mx.bfloat16))
    assert result.shape == (0, 96)
    assert result.dtype == mx.bfloat16
