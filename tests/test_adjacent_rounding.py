# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.quantization.adjacent import (
    adjacent_dequantize,
    adjacent_exact,
    adjacent_exact_blocks,
    adjacent_hessian_error,
    adjacent_ising_energy,
    adjacent_problem_payload,
    adjacent_qubo_energy,
    adjacent_qubo_to_ising,
    adjacent_round_to_nearest_state,
    build_adjacent_rounding_qubo,
    enumerate_adjacent_states,
    quantize_adjacent_rows,
)
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.quantizer import Quantizer
from scripts.quantum_quantization.adjacent_group32 import problem_from_payload


def _spd(size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1701 + size)
    activations = torch.randn(4 * size, size, generator=generator, dtype=torch.float64)
    return activations.mT @ activations / activations.shape[0] + 0.05 * torch.eye(
        size, dtype=torch.float64
    )


def _quantizer(weight: torch.Tensor, bits: int, *, sym: bool = False) -> Quantizer:
    config = QuantizeConfig(
        bits=bits,
        group_size=32,
        sym=sym,
        desc_act=False,
        mse=0.0,
        scale_search=None,
    )
    quantizer = Quantizer(qcfg=config, name="adjacent-test")
    quantizer.configure(perchannel=True)
    quantizer.find_params(weight, weight=True)
    return quantizer


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("sym", [False, True])
def test_adjacent_rtn_reconstructs_existing_quantizer(bits: int, sym: bool):
    generator = torch.Generator().manual_seed(100 * bits + int(sym))
    weight = torch.randn(3, 11, generator=generator, dtype=torch.float32)
    hessian = _spd(weight.shape[1])
    quantizer = _quantizer(weight, bits, sym=sym)
    expected = quantizer.quantize(weight)

    actual = torch.empty_like(weight)
    for row in range(weight.shape[0]):
        problem = build_adjacent_rounding_qubo(
            weight[row],
            hessian,
            scale=quantizer.scale[row],
            zero=quantizer.zero[row],
            bits=bits,
        )
        state = adjacent_round_to_nearest_state(problem)
        actual[row] = adjacent_dequantize(problem, state, dtype=weight.dtype)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_adjacent_direct_qubo_and_ising_energies_match(bits: int):
    generator = torch.Generator().manual_seed(2000 + bits)
    weight = torch.randn(8, generator=generator, dtype=torch.float64)
    hessian = _spd(weight.numel())
    quantizer = _quantizer(weight.unsqueeze(0).float(), bits)
    problem = build_adjacent_rounding_qubo(
        weight,
        hessian,
        scale=quantizer.scale[0],
        zero=quantizer.zero[0],
        bits=bits,
    )
    states = enumerate_adjacent_states(problem.size)
    direct = adjacent_hessian_error(problem, states)

    torch.testing.assert_close(
        adjacent_qubo_energy(problem, states), direct, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        adjacent_ising_energy(adjacent_qubo_to_ising(problem), states),
        direct,
        rtol=1e-12,
        atol=1e-12,
    )
    bridged = problem_from_payload(adjacent_problem_payload(problem))
    torch.testing.assert_close(
        torch.from_numpy(bridged.qubo_linear),
        problem.linear.cpu(),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_adjacent_exact_never_worse_than_rtn(bits: int):
    generator = torch.Generator().manual_seed(3000 + bits)
    weight = torch.randn(8, generator=generator, dtype=torch.float32)
    hessian = _spd(weight.numel())
    quantizer = _quantizer(weight.unsqueeze(0), bits)
    problem = build_adjacent_rounding_qubo(
        weight,
        hessian,
        scale=quantizer.scale[0],
        zero=quantizer.zero[0],
        bits=bits,
    )

    rtn_cost = float(
        adjacent_hessian_error(problem, adjacent_round_to_nearest_state(problem))[0]
    )
    exact = adjacent_exact(problem)

    assert exact.states_checked == 256
    assert exact.cost <= rtn_cost + 1e-12


def _block_calibration() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260724)
    calibration = torch.zeros(256, 32, dtype=torch.float32)
    for block in range(4):
        rows = slice(64 * block, 64 * (block + 1))
        columns = slice(8 * block, 8 * (block + 1))
        latent = torch.randn(64, 3, generator=generator)
        mixing = torch.randn(3, 8, generator=generator)
        calibration[rows, columns] = latent @ mixing + 0.25 * torch.randn(
            64, 8, generator=generator
        )
    return calibration


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_adjacent_rows_use_real_gptq_hessian_and_return_packable_values(bits: int):
    torch.manual_seed(20260724)
    module = torch.nn.Linear(32, 2, bias=False, dtype=torch.float32).eval()
    config = QuantizeConfig(
        bits=bits,
        group_size=32,
        sym=False,
        desc_act=False,
        mse=0.0,
        scale_search=None,
    )
    task = GPTQ(module=module, qcfg=config)
    task.quantizer.configure(perchannel=True)

    calibration = _block_calibration()
    task.add_batch(calibration, module(calibration))
    hessian = task.finalize_hessian(target_device=torch.device("cpu")).clone()
    weight = task.clone_module(device=torch.device("cpu"))
    task.quantizer.find_params(weight, weight=True, hessian=hessian)

    def solver(problem):
        return adjacent_exact_blocks(problem).state

    quantized, states = quantize_adjacent_rows(
        weight,
        hessian,
        scales=task.quantizer.scale,
        zeros=task.quantizer.zero,
        bits=bits,
        solver=solver,
    )
    rtn = task.quantizer.quantize(weight)

    assert quantized.shape == module.weight.shape
    assert states.shape == module.weight.shape
    assert torch.isfinite(quantized).all()
    codes = quantized / task.quantizer.scale + task.quantizer.zero
    rounded_codes = torch.round(codes)
    code_tolerance = 1e-5 if bits == 8 else 1e-6
    torch.testing.assert_close(codes, rounded_codes, rtol=0.0, atol=code_tolerance)
    assert bool(((rounded_codes >= 0) & (rounded_codes <= (1 << bits) - 1)).all())
    for row in range(weight.shape[0]):
        problem = build_adjacent_rounding_qubo(
            weight[row],
            hessian,
            scale=task.quantizer.scale[row],
            zero=task.quantizer.zero[row],
            bits=bits,
        )
        adjacent_cost = adjacent_hessian_error(problem, states[row])[0]
        rtn_cost = adjacent_hessian_error(
            problem, adjacent_round_to_nearest_state(problem)
        )[0]
        assert adjacent_cost <= rtn_cost + 1e-12
        torch.testing.assert_close(
            quantized[row].double(),
            adjacent_dequantize(problem, states[row]),
            atol=1e-7,
            rtol=0,
        )
        torch.testing.assert_close(
            rtn[row].double(),
            adjacent_dequantize(problem, adjacent_round_to_nearest_state(problem)),
            atol=1e-7,
            rtol=0,
        )
