import pytest
import torch

from gptqmodel.quantization import GSQConfig
from gptqmodel.quantization.gsq_scalar import asymmetric_error_term, refine_affine_scalar


@pytest.mark.parametrize("tokens", [3, 19])  # singular and full-rank moments
@pytest.mark.parametrize("alpha", [0.0, 0.25, 1.0])
def test_asymmetric_term_matches_paired_activation_loss_and_gradient(tokens, alpha):
    rng = torch.Generator().manual_seed(7)
    x = torch.randn(8, tokens, generator=rng, dtype=torch.float64)
    native = x + torch.randn(8, tokens, generator=rng, dtype=torch.float64) * .3
    teacher = torch.randn(5, 8, generator=rng, dtype=torch.float64)
    candidate = torch.randn(5, 8, generator=rng, dtype=torch.float64, requires_grad=True)
    baseline = teacher + .05
    h = x @ x.T / tokens
    cross = (native-x) @ x.T / tokens

    def moment_loss(w):
        error = w-teacher
        return ((error @ h)*error).sum() + asymmetric_error_term(error, teacher, cross, alpha)

    def explicit_loss(w):
        return (w @ x-teacher @ (x+alpha*(native-x))).square().sum()/tokens

    torch.testing.assert_close(moment_loss(candidate)-moment_loss(baseline),
                               explicit_loss(candidate)-explicit_loss(baseline), rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(moment_loss(candidate), candidate)[0]
    expected_grad = torch.autograd.grad(explicit_loss(candidate), candidate)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("invalid", ["shape", "cross", "nan", "alpha", "integer"])
def test_asymmetric_term_rejects_invalid_inputs(invalid):
    error, teacher, cross = torch.ones(2, 4), torch.ones(2, 4), torch.eye(4)
    alpha = 1.0
    if invalid == "shape":
        error = error[0]
    elif invalid == "cross":
        cross = cross[:2]
    elif invalid == "nan":
        cross[0, 0] = torch.nan
    elif invalid == "alpha":
        alpha = float("inf")
    else:
        cross = cross.long()
    with pytest.raises(ValueError, match="asymmetric GSQ"):
        asymmetric_error_term(error, teacher, cross, alpha)


def test_asymmetric_hard_checkpoint_matches_native_output_target():
    teacher = torch.ones(2, 4)
    x = torch.eye(4)
    native = 2*x
    scales = torch.ones(2, 1)
    zeros = torch.zeros_like(scales)
    groups = torch.zeros(4, dtype=torch.int32)
    result = refine_affine_scalar(teacher, scales, zeros, groups, target=teacher, bits=2,
                                  inputs=x, cross_moment=(native-x).T @ x,
                                  config=GSQConfig(enabled=True, steps=100, seed=7))
    assert torch.equal(result.weight, 2*teacher)
    # The omitted constant is ||W (X_native-X)||² / ||W X||² = 1.
    assert result.before == 0
    assert result.after == -1
    assert ((result.weight @ x.T-teacher @ native.T).square().sum() == 0)
