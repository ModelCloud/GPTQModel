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


@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("method,alpha,beta", [("gptaq", .5, 0), ("foem", 0, .2),
                                               ("foem", .5, .2), ("foem", .5, 0)])
def test_gptaq_gsq_original_column_objective(desc_act, method, alpha, beta):
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.quantization import GPTQConfig, QuantizeConfig
    from gptqmodel.quantization.gptaq import GPTAQ
    from gptqmodel.quantization.foem import FOEM
    from gptqmodel.quantization.gsq_scalar import affine_codes

    rng = torch.Generator().manual_seed(7)
    layer = torch.nn.Linear(32, 16, bias=False, dtype=torch.float16)
    layer.weight.data.copy_(torch.randn(16, 32, generator=rng)*.1)
    teacher = layer.weight.detach().float().clone()
    x = torch.randn(1, 64, 32, generator=rng)
    native = x + torch.randn(1, 64, 32, generator=rng)*.2
    named = NamedModule(layer, name="proj", full_name="model.proj", layer_index=0)
    named.state["native_inp"] = [native.clone()]
    method_config = {"alpha": alpha} if method == "gptaq" else {"alpha": alpha, "beta": beta}
    cfg = GPTQConfig(bits=4, group_size=16, desc_act=desc_act, act_group_aware=False,
                     **{method: method_config}, gsq=GSQConfig(enabled=True, steps=10))
    cfg = QuantizeConfig.from_quant_config(cfg.to_dict())
    task = (GPTAQ if method == "gptaq" else FOEM)(named, cfg)
    task.quantizer.configure(perchannel=True)
    task.add_batch(x.clone(), None)
    weight, scales, zeros, groups, *_ = task.quantize()
    code = affine_codes(weight, scales, zeros, groups, 4, packing="gptq")
    decoded = scales.half().float()[:, groups]*(code-zeros[:, groups])
    residual = (native-x)[0] @ teacher.T * alpha
    error = x[0] @ decoded.T - (x[0] @ teacher.T + residual)
    expected = (error.square().sum()-residual.square().sum())/(x[0] @ teacher.T).square().sum()
    expected_objective = "asymmetric_quadratic_without_constant" if alpha else "calibration_hessian"
    assert task.gsq_diagnostics["objective"] == expected_objective
    if method == "foem":
        assert task.gsq_diagnostics["initializer_beta"] == beta
    assert task.gsq_diagnostics["after"] == pytest.approx(expected.item(), abs=1e-7, rel=1e-4)
    assert task.gsq_diagnostics["after"] <= task.gsq_diagnostics["before"]


@pytest.mark.parametrize("control", [None, GSQConfig(enabled=False),
                                    GSQConfig(enabled=True, modules=("unmatched",))])
@pytest.mark.parametrize("method", ["gptaq", "foem"])
def test_gptaq_disabled_matches_original_quantizer(control, monkeypatch, method):
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.quantization import GPTQConfig
    from gptqmodel.quantization.gptaq import GPTAQ
    from gptqmodel.quantization.foem import FOEM

    monkeypatch.setattr("gptqmodel.quantization.gptq.refine_affine_scalar",
                        lambda *a, **kw: pytest.fail("disabled GSQ reached fitting"))
    gen = torch.Generator().manual_seed(7)
    weight = torch.randn(16, 32, generator=gen).half()
    x, native = [torch.randn(1, 64, 32, generator=gen) for _ in range(2)]

    def make():
        layer = torch.nn.Linear(32, 16, bias=False, dtype=torch.float16)
        layer.weight.data.copy_(weight)
        named = NamedModule(layer, name="proj", full_name="model.proj", layer_index=0)
        named.state["native_inp"] = [native.clone()]
        cls = GPTAQ if method == "gptaq" else FOEM
        task = cls(named, GPTQConfig(bits=4, group_size=16, **{method: {"alpha": .5}}, gsq=control))
        task.quantizer.configure(perchannel=True)
        task.add_batch(x.clone(), None)
        return task

    expected = make()._quantize_impl()
    actual = make().quantize()
    assert all(torch.equal(a, b) for a, b in zip(actual[:4], expected[:4], strict=True))
