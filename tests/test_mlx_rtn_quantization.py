# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Torch-oracle checks for native MLX weight-only RTN quantization."""

import gc
import sys
from unittest import mock

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization import rtn as rtn_module
from gptqmodel.quantization.config import RTNConfig
from gptqmodel.quantization.mlx_rtn import quantize_rtn_weight_mlx
from gptqmodel.quantization.rtn import RTN


def _compare(weight, bits=4, group_size=128, sym=True):
    layer = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype
    )
    layer.weight.data.copy_(weight)
    oracle = RTN(layer, RTNConfig(bits=bits, group_size=group_size, sym=sym))
    with mock.patch.object(
        rtn_module, "_mlx_rtn_quantization_available", return_value=False
    ):
        expected = oracle.quantize()
    mlx_dtype = {
        torch.float32: mx.float32,
        torch.float16: mx.float16,
        torch.bfloat16: mx.bfloat16,
    }[weight.dtype]
    actual = quantize_rtn_weight_mlx(
        mx.array(weight.float().numpy()).astype(mlx_dtype),
        bits=bits,
        group_size=group_size,
        sym=sym,
    )
    assert actual[0].dtype == mlx_dtype
    for index in range(4):
        observed = actual[index].astype(mx.float32) if index == 0 else actual[index]
        reference = expected[index].float() if index == 0 else expected[index]
        np.testing.assert_allclose(
            np.asarray(observed),
            reference.numpy(),
            rtol=0,
            atol=1e-6,
            err_msg=f"RTN tensor {index}",
        )
    groups = actual[1].shape[1]
    effective = weight.shape[1] if group_size == -1 else group_size
    actual_scale = np.repeat(np.asarray(actual[1]), effective, axis=1)[
        :, : weight.shape[1]
    ]
    actual_zero = np.repeat(np.asarray(actual[2]), effective, axis=1)[
        :, : weight.shape[1]
    ]
    expected_scale = expected[1].repeat_interleave(effective, dim=1)[
        :, : weight.shape[1]
    ]
    expected_zero = expected[2].repeat_interleave(effective, dim=1)[
        :, : weight.shape[1]
    ]
    actual_codes = np.rint(
        np.asarray(actual[0].astype(mx.float32)) / actual_scale + actual_zero
    )
    expected_codes = torch.round(
        expected[0].float() / expected_scale + expected_zero
    ).numpy()
    np.testing.assert_array_equal(actual_codes, expected_codes)
    assert groups == (weight.shape[1] + effective - 1) // effective
    return actual, expected


@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("group_size", [-1, 32, 128])
@pytest.mark.parametrize("sym", [False, True])
def test_rtn_torch_oracle_small(bits, group_size, sym):
    source = (
        torch.randn(
            7, 139, generator=torch.Generator().manual_seed(820), dtype=torch.float32
        )
        * 0.03
    )
    source[0] = 0
    source[1, :32] = 0
    _compare(source, bits=bits, group_size=group_size, sym=sym)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("group_size", [-1, 16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("sym", [False, True])
def test_rtn_all_bits_groups_and_low_precision_inputs(dtype, bits, group_size, sym):
    source = torch.randn(3, 2048, generator=torch.Generator().manual_seed(812)).to(
        dtype
    )
    source[0] = 0
    source[1, 0] = -0.0
    _compare(source, bits=bits, group_size=group_size, sym=sym)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_rtn_accepts_transposed_low_precision_weights(dtype):
    source = mx.random.normal((256, 7)).astype(dtype)
    weight = source.T
    actual = quantize_rtn_weight_mlx(weight, bits=4, group_size=128, sym=True)
    expected = quantize_rtn_weight_mlx(
        mx.contiguous(weight), bits=4, group_size=128, sym=True
    )
    for observed, reference in zip(actual, expected):
        np.testing.assert_array_equal(
            np.asarray(observed.astype(mx.float32)),
            np.asarray(reference.astype(mx.float32)),
        )


def test_rtn_rounding_boundaries():
    # The first row's extrema fix scale at 2/15. Adjacent values probe
    # values exactly on and one float32 step around several code boundaries.
    scale = torch.tensor(2.0 / 15, dtype=torch.float32)
    midpoint = torch.arange(-7, 8, dtype=torch.float32) * scale + scale / 2
    samples = torch.cat(
        (
            torch.nextafter(midpoint, torch.full_like(midpoint, -float("inf"))),
            midpoint,
            torch.nextafter(midpoint, torch.full_like(midpoint, float("inf"))),
        )
    )
    source = torch.zeros(3, 128, dtype=torch.float32)
    source[:, 0] = -1
    source[:, 1] = 1
    source[:, 2 : 2 + len(samples)] = samples
    _compare(source, bits=4, group_size=128, sym=True)


def test_rtn_asymmetric_zero_point_tie_and_endpoints():
    source = torch.zeros(3, 32, dtype=torch.float32)
    source[:, 0] = -1
    source[:, 1] = 1
    # With 2 bits, the ideal affine zero point is 1.5. The reference's
    # float32 scale calculation and ties-to-even rounding determine the code.
    scale = torch.tensor(2 / 3, dtype=torch.float32)
    boundaries = torch.arange(-2, 2, dtype=torch.float32) * scale + scale / 2
    neighbors = torch.cat(
        (
            torch.nextafter(boundaries, torch.full_like(boundaries, -float("inf"))),
            boundaries,
            torch.nextafter(boundaries, torch.full_like(boundaries, float("inf"))),
        )
    )
    source[:, 2 : 2 + len(neighbors)] = neighbors
    _compare(source, bits=2, group_size=32, sym=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rtn_processor_uses_mlx_with_torch_oracle(monkeypatch, dtype):
    torch.manual_seed(8237)
    source = torch.randn((7, 139), dtype=torch.float32).to(dtype)
    source[0] = 0
    source[1, 0] = -0.0
    layer = torch.nn.Linear(139, 7, bias=False, dtype=dtype)
    layer.weight.data.copy_(source)
    config = RTNConfig(bits=4, group_size=32, sym=False)

    monkeypatch.setattr(rtn_module, "_mlx_rtn_quantization_available", lambda: False)
    expected = RTN(layer, config).quantize()

    mlx_calls = 0
    original = rtn_module._quantize_rtn_weight_mlx_to_torch

    def counted_mlx(*args, **kwargs):
        nonlocal mlx_calls
        mlx_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(rtn_module, "_mlx_rtn_quantization_available", lambda: True)
    monkeypatch.setattr(rtn_module, "_MLX_RTN_MIN_ELEMENTS", 0)
    monkeypatch.setattr(rtn_module, "_quantize_rtn_weight_mlx_to_torch", counted_mlx)
    actual = RTN(layer, config).quantize()

    assert mlx_calls == 1
    for observed, reference in zip(actual[:4], expected[:4]):
        torch.testing.assert_close(observed, reference, rtol=0, atol=1e-6)


def test_rtn_processor_keeps_torch_below_mlx_crossover(monkeypatch):
    layer = torch.nn.Linear(32, 2, bias=False, dtype=torch.float16)
    config = RTNConfig(bits=4, group_size=32, sym=True)
    monkeypatch.setattr(rtn_module, "_mlx_rtn_quantization_available", lambda: True)
    monkeypatch.setattr(rtn_module, "_MLX_RTN_MIN_ELEMENTS", layer.weight.numel() + 1)

    def reject_mlx(*args, **kwargs):
        raise AssertionError("small RTN weight used the slower MLX path")

    monkeypatch.setattr(rtn_module, "_quantize_rtn_weight_mlx_to_torch", reject_mlx)
    RTN(layer, config).quantize()


def test_rtn_processor_keeps_torch_for_smoothing(monkeypatch):
    from gptqmodel.quantization.config import SmoothMAD

    layer = torch.nn.Linear(32, 2, bias=False, dtype=torch.float16)
    config = RTNConfig(bits=4, group_size=32, sym=True, smooth=SmoothMAD())
    monkeypatch.setattr(rtn_module, "_mlx_rtn_quantization_available", lambda: True)
    monkeypatch.setattr(rtn_module, "_MLX_RTN_MIN_ELEMENTS", 0)

    def reject_mlx(*args, **kwargs):
        raise AssertionError("smoothed RTN weight used the unsupported MLX path")

    monkeypatch.setattr(rtn_module, "_quantize_rtn_weight_mlx_to_torch", reject_mlx)
    RTN(layer, config).quantize()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires a Torch MPS device"
)
def test_rtn_processor_mps_bridge_matches_cpu_torch_oracle(monkeypatch):
    torch.manual_seed(8238)
    source = torch.randn((7, 139), dtype=torch.float16)
    cpu_layer = torch.nn.Linear(139, 7, bias=False, dtype=torch.float16)
    cpu_layer.weight.data.copy_(source)
    config = RTNConfig(bits=4, group_size=32, sym=True)

    monkeypatch.setattr(rtn_module, "_mlx_rtn_quantization_available", lambda: False)
    expected = RTN(cpu_layer, config).quantize()

    mps_layer = torch.nn.Linear(139, 7, bias=False, dtype=torch.float16, device="mps")
    mps_layer.weight.data.copy_(source.to("mps"))
    monkeypatch.setattr(rtn_module, "_mlx_rtn_quantization_available", lambda: True)
    monkeypatch.setattr(rtn_module, "_MLX_RTN_MIN_ELEMENTS", 0)
    actual = RTN(mps_layer, config).quantize()

    for observed, reference in zip(actual[:4], expected[:4]):
        torch.testing.assert_close(observed.cpu(), reference, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name,rows,cols", QWEN38_27B_PROJECTIONS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sym", [False, True])
def test_rtn_qwen38_projection_oracle(name, rows, cols, sym, dtype):
    rng = np.random.default_rng(380027 + rows + cols)
    source = torch.from_numpy(rng.normal(0, 0.025, (rows, cols)).astype(np.float32)).to(
        dtype
    )
    actual, expected = _compare(source, bits=4, group_size=128, sym=sym)
    assert actual[0].shape == (rows, cols), name
    del source, actual, expected
    gc.collect()
