# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Torch-oracle checks for native MLX ParoQuant export packing."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

_source = Path(__file__).resolve().parents[1] / "gptqmodel/quantization/mlx_paroquant.py"
_spec = importlib.util.spec_from_file_location("gptqmodel_mlx_paroquant_test", _source)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)


def _torch_paroquant_oracle(weight, scales, group_size):
    out_features, in_features = weight.shape
    groups = in_features // group_size
    runtime_scales = scales.to(torch.float16) if scales.dtype == torch.float32 else scales
    values = weight.reshape(out_features, groups, group_size)
    codes = torch.round(
        (values + (scales * 8)[:, :, None]) / runtime_scales[:, :, None]
    ).to(torch.int64)
    assert bool(((codes >= 0) & (codes <= 15)).all())
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    transposed = codes.reshape(out_features, in_features).T.reshape(
        in_features, out_features // 8, 8
    )
    shifts = (torch.arange(8, dtype=torch.int64) * 4).reshape(1, 1, 8)
    words = torch.bitwise_left_shift(transposed[:, :, order], shifts).sum(dim=-1)
    qweight = words.numpy().astype(np.uint32).view(np.int32)
    qzeros = np.full((groups, out_features // 8), -0x77777778, dtype=np.int32)
    return qweight, qzeros, runtime_scales.T.float().numpy()


def _mlx_inputs(weight, scales):
    dtype = {torch.float16: mx.float16, torch.bfloat16: mx.bfloat16,
             torch.float32: mx.float32}[weight.dtype]
    return mx.array(weight.float().numpy()).astype(dtype), mx.array(scales.float().numpy()).astype(dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_paroquant_pack_kernel_accepts_integral_float_codes(dtype):
    out_features, in_features = 32, 8
    codes = np.arange(out_features * in_features, dtype=np.uint32).reshape(
        out_features, in_features
    ) % 16
    packed = native._paroquant_pack_kernel()(
        inputs=[mx.array(codes).astype(dtype)],
        grid=(in_features * (out_features // 8), 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(in_features, out_features // 8)],
        output_dtypes=[mx.int32],
        template=[("OUT_PACKS", out_features // 8), ("IN_FEATURES", in_features)],
    )[0]
    order = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    grouped = codes.T.reshape(in_features, out_features // 8, 8)[:, :, order]
    shifts = (np.arange(8, dtype=np.uint32) * 4).reshape(1, 1, 8)
    expected = np.bitwise_or.reduce(grouped << shifts, axis=2).view(np.int32)
    np.testing.assert_array_equal(np.asarray(packed), expected)


@pytest.mark.parametrize("out_features,in_features,group_size", [
    (32, 64, 16), (64, 128, 32), (96, 256, 64), (128, 256, 128),
    (32, 64, -1),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_paroquant_packing_matches_torch_oracle(out_features, in_features, group_size, dtype):
    effective_group_size = in_features if group_size == -1 else group_size
    torch.manual_seed(124)
    original = torch.randn(out_features, in_features, dtype=torch.float32).to(dtype)
    scales = (
        original.reshape(out_features, -1, effective_group_size).abs().amax(dim=-1)
        .clamp(min=1e-5) / 7
    )
    weight = (
        torch.round(original.reshape(out_features, -1, effective_group_size) / scales[:, :, None])
        .clamp(-8, 7) * scales[:, :, None]
    ).reshape_as(original)
    mlx_weight, mlx_scales = _mlx_inputs(weight, scales)
    actual = native.paroquant_pack_weight_mlx(mlx_weight, mlx_scales, group_size=group_size)
    expected = _torch_paroquant_oracle(weight, scales, effective_group_size)
    for index in (0, 1):
        np.testing.assert_array_equal(np.asarray(actual[index]), expected[index])
    np.testing.assert_array_equal(np.asarray(actual[2].astype(mx.float32)), expected[2])


def test_paroquant_packing_matches_existing_awq_export_layout():
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    torch.manual_seed(125)
    original = torch.randn(64, 128, dtype=torch.float16)
    scales = original.reshape(64, 2, 64).abs().amax(dim=-1) / 7
    weight = (
        torch.round(original.reshape(64, 2, 64) / scales[:, :, None])
        .clamp(-8, 7) * scales[:, :, None]
    ).reshape(64, 128)
    linear = torch.nn.Linear(128, 64, bias=False, dtype=torch.float16)
    linear.weight.data = weight
    current = AwqTorchLinear(
        bits=4, group_size=64, sym=True, desc_act=False,
        in_features=128, out_features=64, register_buffers=False,
    )
    current.pack(linear, scales, torch.full_like(scales, 8))
    mlx_weight, mlx_scales = _mlx_inputs(weight, scales)
    actual = native.paroquant_pack_weight_mlx(mlx_weight, mlx_scales, group_size=64)
    np.testing.assert_array_equal(np.asarray(actual[0]), current.qweight.numpy())
    np.testing.assert_array_equal(np.asarray(actual[1]), current.qzeros.numpy())
    np.testing.assert_array_equal(np.asarray(actual[2]), current.scales.numpy())


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_paroquant_packing_qwen38_27b_shapes(name, out_features, in_features):
    """Check complete BF16 projection packs against the Torch export oracle."""
    del name
    group_size = 64
    generator = torch.Generator().manual_seed(2718 + in_features + out_features)
    scales = torch.rand(
        (out_features, in_features // group_size), generator=generator,
        dtype=torch.float32,
    ).mul(0.02).add(0.01).to(torch.bfloat16)
    codes = torch.randint(
        -7, 8, (out_features, in_features), generator=generator, dtype=torch.int8,
    )
    weight = (codes.to(torch.bfloat16).reshape(out_features, -1, group_size)
              * scales[:, :, None]).reshape(out_features, in_features)
    del codes
    mlx_weight, mlx_scales = _mlx_inputs(weight, scales)
    actual = native.paroquant_pack_weight_mlx(mlx_weight, mlx_scales, group_size=group_size)
    expected = _torch_paroquant_oracle(weight, scales, group_size)
    np.testing.assert_array_equal(np.asarray(actual[0]), expected[0])
    np.testing.assert_array_equal(np.asarray(actual[1]), expected[1])
    np.testing.assert_array_equal(np.asarray(actual[2].astype(mx.float32)), expected[2])


def test_paroquant_packing_at_code_boundaries():
    weight = np.zeros((32, 64), dtype=np.float32)
    for code in range(15):
        midpoint = np.float32(code - 8 + 0.5)
        for step, direction in enumerate((-np.inf, None, np.inf)):
            weight[0, code * 3 + step] = (
                midpoint if direction is None
                else np.nextafter(midpoint, np.float32(direction))
            )
    weight[1, :4] = [-8.0, 7.0, 0.0, -0.0]
    scales = np.ones((32, 1), dtype=np.float32)
    actual = native.paroquant_pack_weight_mlx(
        mx.array(weight), mx.array(scales), group_size=64
    )
    expected = _torch_paroquant_oracle(torch.from_numpy(weight), torch.from_numpy(scales), 64)
    np.testing.assert_array_equal(np.asarray(actual[0]), expected[0])
    np.testing.assert_array_equal(np.asarray(actual[1]), expected[1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paroquant_packing_at_low_precision_code_boundaries(dtype):
    weight = torch.zeros((32, 64), dtype=dtype)
    midpoint = torch.tensor(0.5, dtype=dtype)
    weight[0, :3] = torch.stack((
        torch.nextafter(midpoint, torch.tensor(-float("inf"), dtype=dtype)),
        midpoint,
        torch.nextafter(midpoint, torch.tensor(float("inf"), dtype=dtype)),
    ))
    scales = torch.ones((32, 1), dtype=dtype)
    mlx_weight, mlx_scales = _mlx_inputs(weight, scales)
    actual = native.paroquant_pack_weight_mlx(mlx_weight, mlx_scales, group_size=64)
    expected = _torch_paroquant_oracle(weight, scales, 64)
    np.testing.assert_array_equal(np.asarray(actual[0]), expected[0])


def test_paroquant_packing_at_fp16_scale_cast_boundaries():
    midpoint = np.float32(1.0 + 2.0**-11)
    scales = np.ones((32, 1), dtype=np.float32)
    scales[:3, 0] = [
        np.nextafter(midpoint, np.float32(-np.inf)),
        midpoint,
        np.nextafter(midpoint, np.float32(np.inf)),
    ]
    weight = np.zeros((32, 64), dtype=np.float32)
    actual = native.paroquant_pack_weight_mlx(
        mx.array(weight), mx.array(scales), group_size=64
    )
    expected = _torch_paroquant_oracle(torch.from_numpy(weight), torch.from_numpy(scales), 64)
    np.testing.assert_array_equal(np.asarray(actual[0]), expected[0])
    np.testing.assert_array_equal(np.asarray(actual[2]), expected[2])


def test_paroquant_packing_rejects_out_of_range_codes():
    weight = np.zeros((32, 64), dtype=np.float32)
    weight[0, 0] = -9.0
    with pytest.raises(ValueError, match="codes must be in"):
        native.paroquant_pack_weight_mlx(mx.array(weight), mx.ones((32, 1)), group_size=64)
    weight[0, 0] = 8.0
    with pytest.raises(ValueError, match="codes must be in"):
        native.paroquant_pack_weight_mlx(mx.array(weight), mx.ones((32, 1)), group_size=64)


def test_paroquant_packing_validates_shape_and_scale():
    with pytest.raises(ValueError, match="divisible by 32"):
        native.paroquant_pack_weight_mlx(mx.zeros((16, 64)), mx.ones((16, 1)), group_size=64)
    with pytest.raises(ValueError, match="shape"):
        native.paroquant_pack_weight_mlx(mx.zeros((32, 64)), mx.ones((32, 2)), group_size=64)
    with pytest.raises(ValueError, match="positive"):
        native.paroquant_pack_weight_mlx(mx.zeros((32, 64)), mx.zeros((32, 1)), group_size=64)
    with pytest.raises(ValueError, match="stored scales"):
        native.paroquant_pack_weight_mlx(mx.zeros((32, 64)), mx.full((32, 1), 1e5), group_size=64)
