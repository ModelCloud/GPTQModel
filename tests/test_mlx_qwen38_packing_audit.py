# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# MLX matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Qwen3.8-27B projection checks for merged GPTQ/AWQ to MLX layout packers."""

import sys

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("MLX requires macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from gptqmodel.utils.mlx_packing import repack_awq_4bit, repack_gptq_4bit  # noqa: E402
from gptqmodel.nn_modules.qlinear.mlx_group16 import MlxGroup16Linear  # noqa: E402


@pytest.mark.parametrize("source_format", ("gptq", "awq"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_qwen38_27b_repack_and_inference(source_format, name, out_features, in_features):
    """Check complete 4-bit words, scales, and one-token dense Torch output."""
    del name
    group_size = 128
    rng = np.random.default_rng(3800 + out_features + in_features)
    codes = rng.integers(0, 16, (in_features, out_features), dtype=np.uint8)
    zeros = rng.integers(0, 16, (in_features // group_size, out_features), dtype=np.uint8)
    scales = rng.uniform(0.0005, 0.002, zeros.shape).astype(np.float16)
    shifts = np.arange(8, dtype=np.uint32) * 4
    if source_format == "gptq":
        qweight = np.bitwise_or.reduce(
            codes.reshape(-1, 8, out_features).astype(np.uint32)
            << shifts[None, :, None], axis=1,
        )
        qzeros = np.bitwise_or.reduce(
            zeros.reshape(-1, out_features // 8, 8).astype(np.uint32)
            << shifts[None, None, :], axis=-1,
        )
        repack = repack_gptq_4bit
    else:
        order = [0, 2, 4, 6, 1, 3, 5, 7]
        qweight = np.bitwise_or.reduce(
            codes.reshape(in_features, -1, 8)[:, :, order].astype(np.uint32)
            << shifts[None, None, :], axis=-1,
        )
        qzeros = np.bitwise_or.reduce(
            zeros.reshape(-1, out_features // 8, 8)[:, :, order].astype(np.uint32)
            << shifts[None, None, :], axis=-1,
        )
        repack = repack_awq_4bit
    packed, mlx_scales, biases = repack(
        qweight, qzeros, scales, in_features, out_features,
    )
    expected_packed = np.bitwise_or.reduce(
        codes.T.reshape(out_features, -1, 8).astype(np.uint32)
        << shifts[None, None, :], axis=-1,
    )
    np.testing.assert_array_equal(packed, expected_packed)
    np.testing.assert_array_equal(mlx_scales, scales.T)
    expected_biases = -zeros.T.astype(np.float32) * scales.T.astype(np.float32)
    np.testing.assert_array_equal(biases, expected_biases)

    x = rng.normal(0, 0.01, (1, in_features)).astype(np.float32)
    mlx_x = mx.array(x).astype(mx.bfloat16)
    actual = mx.quantized_matmul(
        mlx_x, mx.array(packed), mx.array(mlx_scales), mx.array(biases),
        group_size=group_size, bits=4,
    )
    mx.eval(actual)
    oracle_weight = (
        torch.from_numpy(codes.astype(np.float32))
        - torch.from_numpy(zeros.astype(np.float32)).repeat_interleave(group_size, dim=0)
    ) * torch.from_numpy(scales.astype(np.float32)).repeat_interleave(group_size, dim=0)
    expected = torch.from_numpy(np.asarray(mlx_x.astype(mx.float32))) @ oracle_weight
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_qwen38_27b_group16_inference(name, out_features, in_features, dtype, record_property):
    """Check the merged two-matmul group-16 kernel on complete projections."""
    del name
    rng = np.random.default_rng(4800 + out_features + in_features)
    codes = rng.integers(0, 16, (in_features, out_features), dtype=np.uint8)
    zeros = rng.integers(0, 16, (in_features // 16, out_features), dtype=np.uint8)
    scales = rng.uniform(0.0005, 0.002, zeros.shape).astype(np.float32)
    shifts = np.arange(8, dtype=np.uint32) * 4
    packed = np.bitwise_or.reduce(
        codes.T.reshape(out_features, -1, 8).astype(np.uint32)
        << shifts[None, None, :], axis=-1,
    )
    group_scales = scales.T.reshape(out_features, -1, 2)
    group_biases = (-zeros.astype(np.float32) * scales).T.reshape(out_features, -1, 2)
    layer = MlxGroup16Linear(in_features, out_features, 4)
    layer.weight = mx.array(packed)
    layer.scales_even = mx.array(group_scales[..., 0])
    layer.scales_odd = mx.array(group_scales[..., 1])
    layer.biases_even = mx.array(group_biases[..., 0])
    layer.biases_odd = mx.array(group_biases[..., 1])
    x = rng.normal(0, 0.01, (1, in_features)).astype(np.float32)
    mlx_x = mx.array(x).astype(dtype)
    actual = layer(mlx_x)
    mx.eval(actual)
    assert actual.dtype == dtype
    oracle_weight = (
        torch.from_numpy(codes.astype(np.float32))
        - torch.from_numpy(zeros.astype(np.float32)).repeat_interleave(16, dim=0)
    ) * torch.from_numpy(scales).repeat_interleave(16, dim=0)
    expected = torch.from_numpy(np.asarray(mlx_x.astype(mx.float32))) @ oracle_weight
    rounded = expected.to(torch.float16 if dtype == mx.float16 else torch.bfloat16).float()
    visible = np.asarray(actual.astype(mx.float32))
    record_property("max_abs_vs_rounded_torch", float(np.max(np.abs(visible - rounded.numpy()))))
    record_property("max_abs_vs_float32_torch", float(np.max(np.abs(visible - expected.numpy()))))
    np.testing.assert_allclose(visible, rounded.numpy(), rtol=2e-3, atol=2e-3)
