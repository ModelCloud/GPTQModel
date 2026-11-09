# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Check ParoQuant's affine zero offset before packed-code rounding."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch


_SOURCE = Path(__file__).resolve().parents[1] / "gptqmodel/quantization/mlx_paroquant.py"


def _load_packer():
    spec = importlib.util.spec_from_file_location("mlx_paroquant_scale_test", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _boundary_inputs():
    scales = np.ones((32, 1), dtype=np.float16)
    scales[:3, 0] = [8188, 8192, 9360]
    weight = np.zeros((32, 64), dtype=np.float16)
    for row in range(3):
        scale = scales[row, 0]
        midpoint = np.float16(scale / np.float16(2))
        weight[row, :3] = [
            np.nextafter(midpoint, np.float16(-np.inf)),
            midpoint,
            np.nextafter(midpoint, np.float16(np.inf)),
        ]
        weight[row, 3:7] = [-scale, scale, np.float16(-0.0), np.float16(0.0)]
    weight[3, :2] = [-8, 7]
    return weight, scales


def _independent_oracles(weight, scales):
    weight_t = torch.from_numpy(weight)
    scales_t = torch.from_numpy(scales)
    torch_codes = (
        torch.round(
            (weight_t.float().reshape(32, 1, 64) + scales_t.float()[:, :, None] * 8) / scales_t.float()[:, :, None]
        )
        .to(torch.int64)
        .reshape(32, 64)
    )
    numpy_codes = (
        np.rint(
            (weight.astype(np.float32).reshape(32, 1, 64) + scales.astype(np.float32)[:, :, None] * 8)
            / scales.astype(np.float32)[:, :, None]
        )
        .astype(np.int64)
        .reshape(32, 64)
    )
    np.testing.assert_array_equal(torch_codes.numpy(), numpy_codes)
    assert bool(((torch_codes >= 0) & (torch_codes <= 15)).all())

    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    lanes = torch_codes.T.reshape(64, 4, 8)[:, :, order]
    shifts = (torch.arange(8, dtype=torch.int64) * 4).reshape(1, 1, 8)
    qweight = torch.bitwise_left_shift(lanes, shifts).sum(dim=-1).numpy()
    qweight = qweight.astype(np.uint32).view(np.int32)
    return numpy_codes, qweight


def test_fp16_scale_zero_offset_matches_torch_and_numpy_on_cpu(monkeypatch):
    """Exercise the production expression with NumPy-backed MLX primitives."""
    weight, scales = _boundary_inputs()
    expected_codes, _ = _independent_oracles(weight, scales)
    packer = _load_packer()
    captured = {}

    core = ModuleType("mlx.core")
    core.float16 = np.float16
    core.bfloat16 = object()
    core.float32 = np.float32
    core.int32 = np.int32
    core.round = np.rint
    core.all = np.all
    core.isfinite = np.isfinite
    core.eval = lambda *arrays: None
    core.full = np.full
    core.contiguous = np.ascontiguousarray
    mlx = ModuleType("mlx")
    mlx.core = core

    def capture_codes(*, inputs, output_shapes, **kwargs):
        del kwargs
        captured["codes"] = np.asarray(inputs[0]).copy()
        return [np.zeros(output_shapes[0], dtype=np.int32)]

    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(packer, "_paroquant_pack_kernel", lambda: capture_codes)
    _, qzeros, stored_scales = packer.paroquant_pack_weight_mlx(weight, scales, group_size=64)

    np.testing.assert_array_equal(captured["codes"].reshape(32, 64), expected_codes)
    np.testing.assert_array_equal(qzeros, np.full((1, 4), -0x77777778, dtype=np.int32))
    np.testing.assert_array_equal(stored_scales, scales.T)
    np.testing.assert_array_equal(expected_codes[:3, :3], [[8, 8, 9]] * 3)


def test_fp16_scale_zero_offset_packs_on_metal():
    if sys.platform != "darwin":
        pytest.skip("MLX Metal packer requires macOS")
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("MLX Metal device is unavailable")

    weight, scales = _boundary_inputs()
    _, expected_qweight = _independent_oracles(weight, scales)
    packed, qzeros, stored_scales = _load_packer().paroquant_pack_weight_mlx(
        mx.array(weight), mx.array(scales), group_size=64
    )
    np.testing.assert_array_equal(np.asarray(packed), expected_qweight)
    np.testing.assert_array_equal(np.asarray(qzeros), np.full((1, 4), -0x77777778, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(stored_scales), scales.T)
