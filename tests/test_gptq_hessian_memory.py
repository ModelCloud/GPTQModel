"""Numerical checks for Hessian inversion buffer reuse."""

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


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
