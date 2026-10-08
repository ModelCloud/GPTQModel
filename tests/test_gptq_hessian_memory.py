"""Numerical checks for Hessian inversion buffer reuse."""

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def _linear_grid_oracle(hessian, damp_percent, damp_auto_increment):
    """Reference the former recovery loop without using the implementation under test."""

    damp = damp_percent
    while 0 < damp < 1:
        candidate = hessian.clone()
        candidate.diagonal().add_(damp * hessian.diagonal().mean())
        try:
            chol = torch.linalg.cholesky(candidate)
        except torch._C._LinAlgError:
            damp += damp_auto_increment
            continue
        torch.cholesky_inverse(chol, out=chol)
        return torch.linalg.cholesky(chol, upper=True), damp
    return None, 1.0


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("size", [3, 129, 512])
def test_hessian_inverse_reuses_buffer_with_exact_factor(device_type, size):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    device = torch.device(device_type)
    torch.manual_seed(733)
    matrix = torch.randn(size, size, device=device)
    hessian = matrix.T @ matrix + torch.eye(size, device=device) * size
    original = hessian.clone()
    config = QuantizeConfig(damp_percent=0.05)
    layer = torch.nn.Linear(size, 1, bias=False, device=device)
    task = GPTQ(layer, qcfg=config)

    factor, used_damp = task.hessian_inverse(hessian)
    reference_hessian = original.clone()
    reference_hessian.diagonal().add_(used_damp * original.diagonal().mean())
    chol = torch.linalg.cholesky(reference_hessian)
    reference = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)

    assert used_damp == config.damp_percent
    torch.testing.assert_close(factor, reference, rtol=0, atol=0)
    torch.testing.assert_close(hessian, original, rtol=0, atol=0)


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_hessian_inverse_retry_preserves_original_matrix(device_type):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    device = torch.device(device_type)
    original = torch.tensor([[1.0, 1.4], [1.4, 1.0]], device=device)
    hessian = original.clone()
    config = QuantizeConfig(damp_percent=0.05, damp_auto_increment=0.25)
    task = GPTQ(torch.nn.Linear(2, 1, bias=False, device=device), qcfg=config)

    factor, used_damp = task.hessian_inverse(hessian)
    reference_hessian = original.clone()
    reference_hessian.diagonal().add_(used_damp * original.diagonal().mean())
    chol = torch.linalg.cholesky(reference_hessian)
    reference = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)

    assert used_damp == pytest.approx(0.55)
    torch.testing.assert_close(factor, reference, rtol=0, atol=0)
    torch.testing.assert_close(hessian, original, rtol=0, atol=0)


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_hessian_inverse_binary_search_matches_default_linear_grid(device_type, monkeypatch):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    device = torch.device(device_type)
    original = torch.tensor([[1.0, 1.7], [1.7, 1.0]], device=device)
    config = QuantizeConfig(damp_percent=0.01, damp_auto_increment=0.0015)
    reference, reference_damp = _linear_grid_oracle(
        original,
        config.damp_percent,
        config.damp_auto_increment,
    )

    real_cholesky_ex = torch.linalg.cholesky_ex
    cholesky_attempts = 0

    def counted_cholesky_ex(*args, **kwargs):
        nonlocal cholesky_attempts
        cholesky_attempts += 1
        return real_cholesky_ex(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", counted_cholesky_ex)
    hessian = original.clone()
    task = GPTQ(torch.nn.Linear(2, 1, bias=False, device=device), qcfg=config)

    factor, used_damp = task.hessian_inverse(hessian)

    assert used_damp == reference_damp
    assert cholesky_attempts <= 12
    torch.testing.assert_close(factor, reference, rtol=0, atol=0)
    torch.testing.assert_close(hessian, original, rtol=0, atol=0)


def test_hessian_inverse_diagonal_floor_recovery_is_preserved():
    hessian = torch.tensor([[0.0, 0.01], [0.01, 0.0]])
    config = QuantizeConfig(damp_percent=0.05, damp_auto_increment=0.0)
    task = GPTQ(torch.nn.Linear(2, 1, bias=False), qcfg=config)

    factor, used_damp = task.hessian_inverse(hessian)

    assert factor is not None
    assert used_damp == config.damp_percent
    torch.testing.assert_close(
        hessian.diagonal(),
        torch.full((2,), 0.01),
        rtol=0,
        atol=1e-7,
    )


def test_hessian_inverse_compatible_search_preserves_quantized_output():
    hessian = torch.tensor([[1.0, 1.7], [1.7, 1.0]])
    weight = torch.tensor(
        [
            [-0.81, 0.37],
            [0.22, 0.93],
            [-0.44, 0.61],
        ]
    )

    def make_task():
        config = QuantizeConfig(
            bits=4,
            group_size=2,
            damp_percent=0.01,
            damp_auto_increment=0.0015,
        )
        layer = torch.nn.Linear(2, 3, bias=False)
        layer.weight.data.copy_(weight)
        task = GPTQ(layer, qcfg=config)
        task.quantizer.configure(perchannel=True)
        task.H = hessian.clone()
        task.nsamples = 1
        return task

    reference_task = make_task()
    reference_task.hessian_inverse = lambda matrix: _linear_grid_oracle(
        matrix,
        reference_task.qcfg.damp_percent,
        reference_task.qcfg.damp_auto_increment,
    )
    reference = reference_task.quantize(blocksize=2)

    actual = make_task().quantize(blocksize=2)

    for actual_tensor, reference_tensor in zip(actual[:4], reference[:4]):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)
    assert actual[6] == reference[6]
