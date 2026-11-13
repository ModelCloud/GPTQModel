# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import ttbfp
from ttbfp import torch_ops


@pytest.mark.parametrize("fmt", [ttbfp.TTBF8, ttbfp.TTBF4])
def test_torch_pack_unpack_matches_numpy(fmt):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((7, 33)).astype(np.float32)
    e_np, d_np = ttbfp.pack(w, fmt)
    e_t, d_t = torch_ops.pack(torch.from_numpy(w), fmt)
    np.testing.assert_array_equal(e_t.numpy(), e_np)
    np.testing.assert_array_equal(d_t.numpy(), d_np)
    y_t = torch_ops.unpack(e_t, d_t, fmt, cols=33)
    y_np = ttbfp.unpack(e_np, d_np, fmt, cols=33)
    np.testing.assert_array_equal(y_t.numpy(), y_np)


@pytest.mark.parametrize("fmt", [ttbfp.TTBF8, ttbfp.TTBF4])
def test_torch_matmul_matches_numpy(fmt):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 64)).astype(np.float32)
    w = rng.standard_normal((64, 48)).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)
    c_np = ttbfp.matmul(a, e, d, fmt, cols=48)
    c_t = torch_ops.matmul(torch.from_numpy(a), torch.from_numpy(e),
                           torch.from_numpy(d), fmt, cols=48)
    np.testing.assert_array_equal(c_t.numpy(), c_np)


@pytest.mark.parametrize("fmt", [ttbfp.TTBF8, ttbfp.TTBF4])
def test_torch_prepacked_matches_numpy(fmt):
    rng = np.random.default_rng(2)
    a = rng.standard_normal((5, 64)).astype(np.float32)
    w = rng.standard_normal((64, 32)).astype(np.float32)
    et, dt = ttbfp.pack_transposed(w, fmt)
    et_t, dt_t = torch_ops.pack_transposed(torch.from_numpy(w), fmt)
    np.testing.assert_array_equal(et_t.numpy(), et)
    np.testing.assert_array_equal(dt_t.numpy(), dt)
    c_np = ttbfp.matmul_prepacked(a, et, dt, fmt, cols=32)
    c_t = torch_ops.matmul_prepacked(torch.from_numpy(a), et_t, dt_t, fmt,
                                     cols=32)
    np.testing.assert_array_equal(c_t.numpy(), c_np)


@pytest.mark.parametrize("fmt", [ttbfp.TTBF8, ttbfp.TTBF4])
def test_torch_prepacked_s8_matches_numpy(fmt):
    rng = np.random.default_rng(4)
    a = rng.standard_normal((3, 64)).astype(np.float32)
    w = rng.standard_normal((64, 32)).astype(np.float32)
    et, s8 = ttbfp.prepack_s8(w, fmt)
    et_t, s8_t = torch_ops.prepack_s8(torch.from_numpy(w), fmt)
    np.testing.assert_array_equal(et_t.numpy(), et)
    np.testing.assert_array_equal(s8_t.numpy(), s8)
    c_np = ttbfp.matmul_prepacked_s8(a, et, s8, fmt, cols=32)
    c_t = torch_ops.matmul_prepacked_s8(torch.from_numpy(a), et_t, s8_t, fmt,
                                        cols=32)
    np.testing.assert_array_equal(c_t.numpy(), c_np)
