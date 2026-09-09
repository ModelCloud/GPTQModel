"""Contract checks for the signed scalar prior; not model-quality evidence."""

import pytest
import torch

from gptqmodel.quantization.gsq_initialization import signed_scalar_range_search


def test_signed_endpoints_and_zero_teacher():
    weights = torch.tensor([[-1.5, 0., 1.5, 3.], [-3., -1.5, 0., 1.5], [0., 0., 0., 0.]])
    reconstructed, scales, zeros = signed_scalar_range_search(weights, bits=2)
    assert torch.equal(reconstructed, weights)
    assert scales[0] < 0 < scales[1]
    assert torch.isfinite(scales).all() and (scales != 0).all()
    assert (zeros == 2).all()
    codes = torch.round(reconstructed/scales)+zeros
    assert (codes >= 0).all() and (codes <= 3).all()


@pytest.mark.parametrize('bits', [True, 1, 2.5, 8])
def test_unsupported_grid_rejects(bits):
    with pytest.raises(ValueError, match='W2/W3/W4'):
        signed_scalar_range_search(torch.ones(2, 8), bits=bits)


@pytest.mark.parametrize('weight', [torch.empty(2, 0), torch.tensor([[float('nan')]]), torch.ones(2, 3).long()])
def test_invalid_teacher_rejects(weight):
    with pytest.raises(ValueError, match='finite nonempty floating'):
        signed_scalar_range_search(weight, bits=2)


@pytest.mark.parametrize('bits', [2, 3, 4])
def test_signed_initial_scales_preserve_assignment_and_gradients(bits):
    from gptqmodel.quantization.gsq_training import GSQScalarTrainingModule

    scales = torch.tensor([[-.5, .25], [.5, -.25]])
    codes = torch.tensor([[-2., -1., 0., 1.], [1., 0., -1., -2.]])
    weights = codes*scales.repeat_interleave(2, dim=1)
    count = 4 if bits == 2 else 5
    module = GSQScalarTrainingModule(weights, scales, 2, bits=bits, noise=torch.zeros(count, 2, 4))
    assert torch.equal(module.hard_weight(), weights)
    output = module(uniform=torch.full_like(module.logits, .5), temperature=.7, multiplier=12.)
    output.square().sum().backward()
    assert torch.isfinite(module.logits.grad).all() and module.logits.grad.abs().sum() > 0
    assert torch.isfinite(module.scales.grad).all() and module.scales.grad.abs().sum() > 0


def test_signed_prior_rejects_conflicting_scale_objective():
    from gptqmodel.quantization.config import GPTQConfig
    from gptqmodel.quantization.gsq_initialization import SignedGSQQuantizer

    quantizer = SignedGSQQuantizer(GPTQConfig(bits=2, group_size=128, sym=True))
    quantizer.configure(perchannel=True)
    with pytest.raises(ValueError, match='explicit MSE'):
        quantizer.find_params(torch.ones(2, 128), weight=True)
