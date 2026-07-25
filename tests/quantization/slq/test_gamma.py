# GPU=-1
import math

import pytest
import torch

from gptqmodel.quantization.slq.gamma import (
    centering_inefficiency,
    gamma_squared_variance_law,
    quantize_uniform,
    quantization_noise_variance_asym,
    quantization_noise_variance_sym,
    step_size_asym,
    step_size_sym,
)


def test_centering_inefficiency_symmetric():
    w = torch.linspace(-1.0, 1.0, 100)
    gamma = centering_inefficiency(w)
    assert torch.allclose(gamma, torch.tensor(1.0), atol=1e-4)


def test_centering_inefficiency_skewed():
    w = torch.tensor([-0.8, 0.0, 1.2])
    gamma = centering_inefficiency(w)
    # R = 2.0, M = 1.2, gamma = 2*1.2/2.0 = 1.2
    assert math.isclose(gamma.item(), 1.2, rel_tol=1e-5)


def test_step_size_ratio_equals_gamma():
    w = torch.tensor([-0.8, 0.0, 1.2])
    max_abs = torch.maximum(w.min().abs(), w.max().abs())
    d_sym = step_size_sym(4, max_abs)
    d_asym = step_size_asym(4, w.min(), w.max())
    assert torch.allclose(d_sym / d_asym, torch.tensor(1.2), atol=1e-5)


def test_noise_variance_ratio_equals_gamma_squared():
    w = torch.tensor([-0.8, 0.0, 1.2])
    max_abs = torch.maximum(w.min().abs(), w.max().abs())
    var_sym = quantization_noise_variance_sym(4, max_abs)
    var_asym = quantization_noise_variance_asym(4, w.min(), w.max())
    gamma = centering_inefficiency(w)
    ratio = var_sym / var_asym
    assert torch.allclose(ratio, gamma ** 2, atol=1e-5)


def test_gamma_squared_variance_law():
    torch.manual_seed(42)
    w = torch.randn(1000) * 0.5 + 0.3  # skewed distribution
    result = gamma_squared_variance_law(w, bits=6)
    gamma = result["gamma"]
    assert gamma.item() >= 1.0
    law_error = result["law_error"].item()
    # Identity should hold exactly for the analytical step-size formulas.
    assert abs(law_error) < 1e-4


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_law_holds_across_bitwidths(bits):
    w = torch.randn(500) * 0.4 + 0.25
    result = gamma_squared_variance_law(w, bits=bits)
    assert abs(result["law_error"].item()) < 1e-4


def test_quantize_uniform_symmetric():
    w = torch.linspace(-1.0, 1.0, 100)
    q = quantize_uniform(w, bits=4, symmetric=True)
    assert q.shape == w.shape
    assert (q.abs() <= w.abs().max()).all()


def test_quantize_uniform_asymmetric_covers_range():
    w = torch.tensor([-0.8, 0.0, 1.2])
    q = quantize_uniform(w, bits=4, symmetric=False)
    assert q.min() <= w.min() + 1e-4
    assert q.max() >= w.max() - 1e-4


def test_asymmetric_smaller_error_for_skewed_tensor():
    torch.manual_seed(0)
    w = torch.randn(5000) * 0.5 + 0.4
    q_sym = quantize_uniform(w, bits=4, symmetric=True)
    q_asym = quantize_uniform(w, bits=4, symmetric=False)
    err_sym = (w - q_sym).abs().mean()
    err_asym = (w - q_asym).abs().mean()
    # For skewed data asymmetric should not be worse, and is usually better.
    assert err_asym <= err_sym * 1.01
