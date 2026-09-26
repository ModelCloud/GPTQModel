# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Torch-oracle checks for native MLX weight-only RTN quantization."""

import gc
import sys

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.config import RTNConfig  # noqa: E402
from gptqmodel.quantization.mlx_rtn import quantize_rtn_weight_mlx  # noqa: E402
from gptqmodel.quantization.rtn import RTN  # noqa: E402


def _compare(weight, bits=4, group_size=128, sym=True):
    layer = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype
    )
    layer.weight.data.copy_(weight)
    oracle = RTN(layer, RTNConfig(bits=bits, group_size=group_size, sym=sym))
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
    source = torch.randn(3, 2048, generator=torch.Generator().manual_seed(812)).to(dtype)
    source[0] = 0
    source[1, 0] = -0.0
    _compare(source, bits=bits, group_size=group_size, sym=sym)


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


@pytest.mark.parametrize("name,rows,cols", QWEN38_27B_PROJECTIONS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sym", [False, True])
def test_rtn_qwen38_projection_oracle(name, rows, cols, sym, dtype):
    rng = np.random.default_rng(380027 + rows + cols)
    source = torch.from_numpy(
        rng.normal(0, 0.025, (rows, cols)).astype(np.float32)
    ).to(dtype)
    actual, expected = _compare(source, bits=4, group_size=128, sym=sym)
    assert actual[0].shape == (rows, cols), name
    del source, actual, expected
    gc.collect()
