# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""MLX FOEM must apply the #4046 block-boundary beta correction (#4548).

The Linux CI host has no MLX/Metal runtime, so the shipped MLX host loop is
executed against stand-in MLX primitives: numpy arrays plus a recorder kernel
that returns a known constant error.  The assertion is on the *next* group's
first input column, compared against an independent float64 oracle that never
imports the implementation under test.  Device numerics remain an Apple-host
qualification.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import gptqmodel.quantization.mlx_foem as mlx_foem
import gptqmodel.quantization.mlx_native as mlx_native


_GROUP = 16
_COLUMNS = 32
_ROWS = 4
_ERROR = 0.5
_HINV_OFFDIAG = 0.05


class _BFloat16:
    """Sentinel dtype: numpy has no bfloat16, and only float32 is exercised."""


class _RecorderKernel:
    def __init__(self, launches, error_value):
        self._launches = launches
        self._error_value = error_value

    def __call__(
        self,
        *,
        inputs,
        template,
        grid,
        threadgroup,
        output_shapes,
        output_dtypes,
    ):
        del grid, threadgroup, output_dtypes
        self._launches.append(
            {
                "inputs": [np.array(value, dtype=np.float32, copy=True) for value in inputs],
                "template": dict(template),
                "output_shapes": tuple(output_shapes),
            }
        )
        rows, width = output_shapes[0]
        outputs = [
            np.zeros((rows, width), dtype=np.float32),
            np.full((rows, width), self._error_value, dtype=np.float32),
            np.ones((rows, 1), dtype=np.float32),
            np.zeros((rows, 1), dtype=np.float32),
        ]
        if len(output_shapes) == 5:
            outputs.append(np.zeros((rows, width), dtype=np.float32))
        return outputs


class _FakeMlxCore:
    """The subset of mlx.core used by ``foem_quantize_weight_mlx``."""

    float16 = np.float16
    bfloat16 = _BFloat16
    float32 = np.float32

    def __init__(self, launches, error_value):
        self.fast = SimpleNamespace(
            metal_kernel=lambda **spec: _RecorderKernel(launches, error_value)
        )

    @staticmethod
    def isfinite(value):
        return np.isfinite(value)

    @staticmethod
    def diag(value):
        return np.diag(value)

    @staticmethod
    def all(value):
        return np.all(value)

    @staticmethod
    def array(value, dtype=None):
        return np.array(value, dtype=dtype)

    @staticmethod
    def contiguous(value):
        return value

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def expand_dims(value, axis):
        return np.expand_dims(value, axis)

    @staticmethod
    def eval(*values):
        return None


def _weight():
    generator = np.random.default_rng(7)
    return generator.standard_normal((_ROWS, _COLUMNS)).astype(np.float32)


def _hessian():
    return (np.eye(_COLUMNS, dtype=np.float32) + np.float32(_HINV_OFFDIAG)).astype(
        np.float32
    )


def _oracle_boundary_column(weight, hessian, beta):
    """Independent float64 oracle for the first column of group 2.

    The kernel reports a constant error, so the host propagation gives
    ``raw[:, G] - error * sum_k hessian[k, G]`` for column ``G``.  #4046 then
    pulls that column toward its raw value by ``beta``.
    """

    raw = weight.astype(np.float64)
    factor = hessian.astype(np.float64)
    propagated = raw[:, _GROUP] - _ERROR * factor[:_GROUP, _GROUP].sum()
    return propagated - beta * (propagated - raw[:, _GROUP])


def _run(monkeypatch, beta):
    launches = []
    core = _FakeMlxCore(launches, _ERROR)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(
        mlx_native,
        "_accurate_matmul_mlx",
        lambda left, right: left @ right,
    )
    mlx_foem._foem_group_kernel.cache_clear()
    try:
        result = mlx_foem.foem_quantize_weight_mlx(
            _weight(),
            _hessian(),
            bits=4,
            group_size=_GROUP,
            beta=beta,
            sym=True,
        )
    finally:
        mlx_foem._foem_group_kernel.cache_clear()
    return launches, result


@pytest.mark.parametrize("beta", [0.2, 1.0, 0.0])
def test_mlx_foem_applies_block_boundary_beta(monkeypatch, beta):
    launches, result = _run(monkeypatch, beta)

    assert len(launches) == _COLUMNS // _GROUP
    assert len(result) == 3
    assert result[0].shape == (_ROWS, _COLUMNS)
    assert launches[1]["template"]["G"] == _GROUP

    next_group = launches[1]["inputs"][0]
    assert next_group.shape == (_ROWS, _GROUP)

    expected = _oracle_boundary_column(_weight(), _hessian(), beta)
    assert np.allclose(next_group[:, 0], expected, rtol=1e-6, atol=1e-6)

    # beta=0 must leave the propagated column untouched; a nonzero beta must
    # move it, so this test fails on the pre-#4548 MLX loop.
    propagated = _oracle_boundary_column(_weight(), _hessian(), 0.0)
    if beta == 0.0:
        assert np.allclose(next_group[:, 0], propagated, rtol=1e-6, atol=1e-6)
    else:
        assert not np.allclose(next_group[:, 0], propagated, rtol=1e-6, atol=1e-6)
