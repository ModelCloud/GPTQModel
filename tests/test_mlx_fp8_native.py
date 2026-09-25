# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Qwen3.8-27B shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.
# FP8 E4M3 reference: PyTorch contributors, BSD-3-Clause, https://github.com/pytorch/pytorch
# MXFP8 matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""Exact packed FP8 transfer and independent Torch inference checks on Metal."""

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


def make_source(out_features, in_features, scale_method="row"):
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear

    source = TorchFP8Linear(
        bits=8, group_size=-1, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features, bias=True,
        dtype=torch.float16, format="float8_e4m3fn",
        weight_scale_method=scale_method,
    )
    # Vary both axes while avoiding an enormous random-number allocation.
    row = torch.arange(out_features, dtype=torch.int32).reshape(-1, 1)
    col = torch.arange(in_features, dtype=torch.int32).reshape(1, -1)
    values = ((row * 17 + col * 13) % 49 - 24).to(torch.float32)
    source.weight.copy_(values.to(torch.float8_e4m3fn))
    if scale_method == "row":
        source.weight_scale_inv.copy_(90 + torch.arange(out_features) % 31)
    else:
        source.weight_scale_inv.fill_(107)
    source.bias.copy_(torch.linspace(-0.02, 0.02, out_features, dtype=torch.float16))
    return source


def torch_oracle(source, x):
    """Decode FP8 and apply inverse checkpoint scales without MLX intermediates."""
    scale = source.weight_scale_inv.double()
    expected = torch.empty((x.shape[0], source.out_features), dtype=torch.float64)
    for first in range(0, source.out_features, 256):
        last = min(first + 256, source.out_features)
        weights = source.weight[first:last].float().double()
        if source.weight_scale_method == "row":
            weights /= scale[first:last, None]
        else:
            weights /= scale
        expected[:, first:last] = x.double() @ weights.T + source.bias[first:last].double()
    return expected


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
@pytest.mark.parametrize("rows", [1, 16])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fp8_native_qwen38_projection(name, out_features, in_features, rows, dtype):
    from gptqmodel.nn_modules.qlinear.mlx import FP8MlxQuantLinear
    from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8Linear

    source = make_source(out_features, in_features)
    assert FP8MlxQuantLinear.native_compatible(source), name
    weight, scales, output_scale, params = FP8MlxQuantLinear.pack_source(source)
    assert params == {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    np.testing.assert_array_equal(weight.view(np.uint8), source.weight.view(torch.uint8).numpy())
    assert np.all(scales == 127)
    native = MlxFP8Linear(in_features, out_features, output_scale, source.bias.numpy())
    native.linear.load_weights([("weight", mx.array(weight)), ("scales", mx.array(scales))])
    rng = np.random.default_rng(389 + out_features + in_features)
    x = rng.normal(0, 0.05, (rows, in_features)).astype(np.float16)
    inputs = mx.array(x).astype(dtype)
    actual = native(inputs)
    mx.eval(actual)
    assert actual.dtype == dtype
    reference_inputs = torch.from_numpy(np.asarray(inputs.astype(mx.float32)))
    expected_torch = torch_oracle(source, reference_inputs)
    expected = expected_torch.numpy()
    expected_rounded = expected_torch.to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16
    ).float().numpy()
    output = np.asarray(actual.astype(mx.float32))
    if dtype == mx.float16:
        np.testing.assert_allclose(output, expected, rtol=0.002, atol=0.002)
    else:
        # BF16's output spacing exceeds 2e-3 for some values. Confirm that
        # the output is at most one representable BF16 step from the rounded
        # independent oracle; the FP32 arithmetic is checked below at 1e-5.
        rounded = torch.from_numpy(expected_rounded).to(torch.bfloat16)
        upper = torch.nextafter(rounded, torch.full_like(rounded, float("inf"))).float().numpy()
        lower = torch.nextafter(rounded, torch.full_like(rounded, float("-inf"))).float().numpy()
        ulp = np.maximum(upper - expected_rounded, expected_rounded - lower)
        tolerance = 0.002 + 0.002 * np.abs(expected_rounded)
        assert np.all(np.abs(output - expected_rounded) <= np.maximum(ulp, tolerance))
    assert np.isfinite(output).all()
    # Measure the arithmetic before the required FP16/BF16 output rounding.
    internal = native._unscaled_dot(inputs) * native.output_scale + native.bias
    mx.eval(internal)
    np.testing.assert_allclose(np.asarray(internal), expected, rtol=1e-5, atol=1e-5)


def test_fp8_native_large_unscaled_dot_uses_fp32():
    from gptqmodel.nn_modules.qlinear.mlx import FP8MlxQuantLinear
    from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8Linear

    source = make_source(64, 17408, "tensor")
    source.weight.fill_(400)
    source.weight_scale_inv.fill_(1000)
    packed, scales, output_scale, _ = FP8MlxQuantLinear.pack_source(source)
    native = MlxFP8Linear(17408, 64, output_scale, source.bias.numpy())
    native.linear.load_weights([("weight", mx.array(packed)), ("scales", mx.array(scales))])
    x = torch.ones((1, 17408), dtype=torch.float16)
    result = native(mx.array(x.numpy()))
    mx.eval(result)
    np.testing.assert_allclose(np.asarray(result), torch_oracle(source, x).numpy(), rtol=0.002, atol=0.002)


def test_mxfp8_transfer_preserves_every_finite_e4m3_code():
    codes = np.arange(256, dtype=np.uint8)
    codes[np.array([127, 255])] = 0
    weight = codes.reshape(8, 32)
    packed = mx.array(weight.view(np.uint32))
    scales = mx.full((8, 1), 127, dtype=mx.uint8)
    actual = mx.dequantize(packed, scales, mode="mxfp8")
    mx.eval(actual)
    expected = torch.from_numpy(weight.copy()).view(torch.float8_e4m3fn).float().numpy()
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), expected)


@pytest.mark.parametrize("scale_method", ["row", "tensor"])
def test_fp8_native_rejects_nonpositive_scales(scale_method):
    from gptqmodel.nn_modules.qlinear.mlx import FP8MlxQuantLinear

    source = make_source(64, 128, scale_method)
    source.weight_scale_inv.reshape(-1)[0] = 0
    assert not FP8MlxQuantLinear.native_compatible(source)


def test_fp8_native_rejects_nan_codes_and_unaligned_width():
    from gptqmodel.nn_modules.qlinear.mlx import FP8MlxQuantLinear

    source = make_source(64, 128)
    source.weight[0, 0] = float("nan")
    assert not FP8MlxQuantLinear.native_compatible(source)
    unaligned = make_source(64, 96 + 1)
    assert not FP8MlxQuantLinear.native_compatible(unaligned)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_fp8_native_input_dtype(dtype):
    from gptqmodel.nn_modules.qlinear.mlx import FP8MlxQuantLinear
    from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8Linear

    source = make_source(64, 128, "tensor")
    packed, scales, output_scale, _ = FP8MlxQuantLinear.pack_source(source)
    native = MlxFP8Linear(128, 64, output_scale, source.bias.numpy())
    native.linear.load_weights([("weight", mx.array(packed)), ("scales", mx.array(scales))])
    x = mx.array(np.full((2, 128), 0.01, np.float32)).astype(dtype)
    actual = native(x)
    mx.eval(actual)
    assert actual.dtype == dtype
    expected = torch_oracle(source, torch.from_numpy(np.asarray(x.astype(mx.float32))))
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), expected.numpy(), rtol=0.002, atol=0.002)


def test_activation_dtype_flows_through_residual_norm_and_linear():
    from gptqmodel.nn_modules.qlinear.mlx import FP8MlxQuantLinear
    from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8Linear

    source = make_source(128, 128)
    packed, scales, output_scale, _ = FP8MlxQuantLinear.pack_source(source)
    native = MlxFP8Linear(128, 128, output_scale, source.bias.numpy())
    native.linear.load_weights([("weight", mx.array(packed)), ("scales", mx.array(scales))])
    norm = nn.RMSNorm(128)
    norm.weight = norm.weight.astype(mx.float16)
    next_layer = nn.Linear(128, 64, bias=False)
    next_layer.weight = next_layer.weight.astype(mx.float16)
    x = mx.ones((2, 128), dtype=mx.float16)
    output = next_layer(norm(native(x) + x))
    mx.eval(output)
    assert output.dtype == mx.float16
    assert np.isfinite(np.asarray(output)).all()
