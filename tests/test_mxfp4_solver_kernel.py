# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from test_format_reference import scalar_decode
from test_format_solver_reference import independent_gptq

from gptqmodel import solvers
from gptqmodel.nn_modules.qlinear.mxfp4 import MXFP4Linear
from gptqmodel.quantization.format_reference import quantize_rtn
from gptqmodel.quantization.solver_format import Hessian, solve_gptq_hessian
from scripts.prove_format_solvers import problem


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("block_size", [1, 7, 32, 128])
@pytest.mark.parametrize("desc_act", [False, True])
def test_blocked_gptq_matches_independent_normal_equations(block_size, desc_act):
    weight, inputs, _ = problem(13)
    weight = weight[:2]
    hessian = inputs.T @ inputs / len(inputs)
    original = hessian.clone()
    actual = solve_gptq_hessian(weight, hessian, solvers.GPTQ(block_size=block_size, desc_act=desc_act))
    expected = independent_gptq(weight, inputs, actual.grid, desc_act)
    assert torch.equal(actual.codes(), expected)
    assert torch.equal(hessian, original)
    assert not torch.equal(actual.payload, quantize_rtn(weight, "mxfp4").payload)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("assign", [False, True])
def test_kernel_independent_payload_decode_and_storage_lifetime(dtype, assign):
    torch.manual_seed(5)
    rows, cols = 131, 64
    module = MXFP4Linear(cols, rows, torch.randn(rows, dtype=dtype))
    module.qweight.random_(0, 256)
    module.scales.random_(122, 130)
    weights = torch.tensor([
        [scalar_decode(int(module.qweight[row, column // 2] >> (4 * (column % 2))) & 15, "e2m1")
         * 2. ** (int(module.scales[row, column // 32]) - 127) for column in range(cols)]
        for row in range(rows)
    ], dtype=torch.float64 if dtype == torch.float64 else torch.float32)
    inputs = torch.randn(2, 3, cols * 2, dtype=dtype)[..., ::2]
    expected = (inputs.to(weights.dtype) @ weights.T + module.bias.to(weights.dtype)).to(dtype)
    actual = module(inputs)
    # The ragged output tile can select a different BLAS reduction order.
    tolerance = 2e-5 if dtype == torch.float32 else 1e-12 if dtype == torch.float64 else 0
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    fresh = MXFP4Linear(cols, rows, torch.zeros_like(module.bias))
    for _ in range(2):
        fresh.load_state_dict(module.state_dict(), assign=assign)
        torch.testing.assert_close(fresh(inputs), actual, rtol=0, atol=0)
    assert set(fresh.state_dict()) == {"qweight", "scales", "bias"}
    assert sum(t.numel() * t.element_size() for t in fresh.buffers()) == rows * (cols // 2 + cols // 32) + rows * inputs.element_size()
    assert not list(fresh.parameters())
    # No packed or floating weight cache may survive a forward or a reload.
    assert all(not isinstance(value, torch.Tensor) for value in vars(fresh).values())
    module.qweight.bitwise_xor_(8)
    fresh.load_state_dict(module.state_dict(), assign=assign)
    assert not torch.equal(fresh(inputs), expected)


def test_hessian_excludes_padding_and_owns_no_inputs():
    inputs = torch.tensor([[[1., 3.], [100., 200.], [2., -1.]]])
    accumulator = Hessian(2)
    accumulator.mask = torch.tensor([[1, 0, 1]])
    accumulator.capture(None, (inputs,))
    expected = torch.tensor([[5., 1.], [1., 10.]], dtype=torch.float64)
    assert torch.equal(accumulator.sum, expected)
    assert accumulator.tokens == 2
    assert not any(value is inputs for value in vars(accumulator).values())
