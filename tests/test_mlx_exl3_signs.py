# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 sign format: TurboDerp and ExLlamaV3 contributors.

"""Independent Torch-oracle checks for native MLX EXL3 sign packing."""

import gc
import sys

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3_signs import exl3_pack_signs_mlx


def _torch_pack_signs_oracle(signs):
    """Reproduce EXL3's float16 sign-bit packing with Torch only."""
    source = signs.to(dtype=torch.float16).contiguous().reshape(-1)
    raw = source.view(torch.int16).to(torch.int32).reshape(-1, 16)
    shifts = torch.arange(16, dtype=torch.int32, device=raw.device)
    bits = (raw >> 15) & 1
    return torch.sum(bits << shifts, dim=1).to(torch.int16)


def _mlx_dtype(name):
    return {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }[name]


def _torch_dtype(name):
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


@pytest.mark.parametrize("dtype_name", ("float16", "float32", "bfloat16"))
def test_exl3_pack_signs_cast_and_sign_boundaries(dtype_name):
    positive_subnormal = np.nextafter(np.float32(0.0), np.float32(1.0))
    values = np.array(
        [
            -np.inf,
            -np.finfo(np.float32).max,
            -65520.0,
            -65504.0,
            -1.0,
            -np.finfo(np.float32).tiny,
            -positive_subnormal,
            -0.0,
            0.0,
            positive_subnormal,
            np.finfo(np.float32).tiny,
            1.0,
            65504.0,
            65520.0,
            np.finfo(np.float32).max,
            np.inf,
        ],
        dtype=np.float32,
    )
    torch_signs = torch.from_numpy(values).to(_torch_dtype(dtype_name))
    expected = _torch_pack_signs_oracle(torch_signs).numpy()
    mlx_signs = mx.array(values).astype(_mlx_dtype(dtype_name))
    actual = np.asarray(exl3_pack_signs_mlx(mlx_signs))
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == np.int16(255)


def test_exl3_pack_signs_multidimensional_flattening_matches_torch():
    rng = np.random.default_rng(83175)
    signs = rng.choice([-1.0, 1.0], size=(4, 8, 16)).astype(np.float32)
    signs[0, 0, 0] = -0.0
    signs[0, 0, 1] = 0.0
    expected = _torch_pack_signs_oracle(torch.from_numpy(signs)).numpy()
    actual = np.asarray(exl3_pack_signs_mlx(mx.array(signs)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "mlx_storage,mlx_dtype,torch_dtype,positive_nan,negative_nan",
    (
        (mx.uint16, mx.float16, torch.float16, 0x7E00, 0xFE00),
        (mx.uint16, mx.bfloat16, torch.bfloat16, 0x7FC0, 0xFFC0),
        (mx.uint32, mx.float32, torch.float32, 0x7FC00000, 0xFFC00000),
    ),
)
def test_exl3_pack_signs_preserves_nan_sign_bits(
    mlx_storage, mlx_dtype, torch_dtype, positive_nan, negative_nan
):
    storage_dtype = np.uint16 if mlx_storage == mx.uint16 else np.uint32
    raw = np.resize(np.array([positive_nan, negative_nan], dtype=storage_dtype), 16)
    mlx_signs = mx.array(raw, dtype=mlx_storage).view(mlx_dtype)
    torch_signs = torch.from_numpy(raw).view(torch_dtype)
    expected = _torch_pack_signs_oracle(torch_signs).numpy()
    actual = np.asarray(exl3_pack_signs_mlx(mlx_signs))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", ("input", "output"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_pack_signs_qwen38_projection_oracle(
    name, out_features, in_features, mode
):
    channels = in_features if mode == "input" else out_features
    rng = np.random.default_rng(93175 + out_features + in_features + len(mode))
    signs = rng.choice([-1.0, 1.0], size=channels).astype(np.float32)
    torch_signs = torch.from_numpy(signs).to(torch.bfloat16)
    expected = _torch_pack_signs_oracle(torch_signs).numpy()
    mlx_signs = mx.array(signs).astype(mx.bfloat16)
    actual = np.asarray(exl3_pack_signs_mlx(mlx_signs))
    np.testing.assert_array_equal(actual, expected, err_msg=f"{name} {mode}")
    assert actual.size == channels // 16
    del signs, torch_signs, expected, mlx_signs, actual
    gc.collect()
    mx.clear_cache()


def test_exl3_pack_signs_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="dtype"):
        exl3_pack_signs_mlx(mx.ones((16,), dtype=mx.int16))
    with pytest.raises(ValueError, match="nonempty"):
        exl3_pack_signs_mlx(mx.zeros((0,), dtype=mx.float32))
    with pytest.raises(ValueError, match="divisible by 16"):
        exl3_pack_signs_mlx(mx.ones((17,), dtype=mx.float32))
