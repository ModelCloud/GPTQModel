import pytest
import torch

from gptqmodel.quantization import GSQConfig
from gptqmodel.quantization.gsq_scalar import affine_codes, refine_affine_scalar


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("packing", ["gptq", "awq_gemm"])
def test_scalar_assignment_fit_on_grid(bits, packing):
    scale = torch.tensor([[0.125, 0.25], [0.5, 0.125]])
    zero = torch.full_like(scale, 2**(bits-1))
    groups = torch.tensor([1, 0, 1, 0], dtype=torch.int32)  # noncontiguous ownership after act-order
    base_codes = torch.full((2, 4), 2**(bits-1)).float()
    target_codes = base_codes + torch.tensor([[1, -1, 1, -1], [-1, 1, -1, 1]])
    baseline = scale[:, groups] * (base_codes-zero[:, groups])
    target = scale[:, groups] * (target_codes-zero[:, groups])
    result = refine_affine_scalar(baseline, scale, zero, groups, target=target, bits=bits, packing=packing,
                                  inputs=torch.eye(4), config=GSQConfig(enabled=True, steps=100, seed=7))
    assert result.after == 0
    assert torch.equal(result.weight, target)
    assert torch.equal(result.scales, scale)
    assert torch.equal(affine_codes(result.weight, result.scales, zero, groups, bits, packing=packing), target_codes)


def test_scalar_scale_learning_and_storage_guard():
    scale = torch.full((2, 1), 0.125)
    zeros, groups = torch.full_like(scale, 2), torch.zeros(8, dtype=torch.int32)
    baseline = torch.full((2, 8), 0.125, dtype=torch.float16)
    target = torch.full((2, 8), 0.2)
    fixed = refine_affine_scalar(baseline, scale, zeros, groups, target=target, bits=2,
                                 config=GSQConfig(enabled=True, steps=50))
    learned = refine_affine_scalar(baseline, scale, zeros, groups, target=target, bits=2,
                                   config=GSQConfig(enabled=True, steps=50, learn_scales=True))
    assert learned.after < fixed.after
    assert torch.equal(learned.scales, learned.scales.half().float())
    assert learned.weight.dtype == baseline.dtype
    assert (learned.scales > 0).all()


def test_scalar_hessian_matches_real_activation_objective():
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(3, 8, generator=gen)  # intentionally rank deficient
    scale, zeros = torch.ones(2, 2), torch.ones(2, 2)
    groups = torch.arange(8, dtype=torch.int32) % 2
    baseline = torch.zeros(2, 8)
    target = torch.randn(2, 8, generator=gen)
    cfg = GSQConfig(enabled=True, steps=1)
    direct = refine_affine_scalar(baseline, scale, zeros, groups, target=target, bits=2, config=cfg, inputs=x)
    factored = refine_affine_scalar(baseline, scale, zeros, groups, target=target, bits=2, config=cfg, hessian=x.T @ x)
    assert factored.before == pytest.approx(direct.before, rel=1e-5)
    assert factored.history == pytest.approx(direct.history, rel=1e-5)


def test_scalar_default_is_exact_bypass(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled GSQ performed fitting")
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    w, s, z, g = torch.zeros(2, 4), torch.ones(2, 1), torch.ones(2, 1), torch.zeros(4, dtype=torch.int32)
    before = torch.random.get_rng_state()
    r = refine_affine_scalar(w, s, z, g, target=None, bits=4)
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip((r.weight, r.scales, r.zeros, r.g_idx), (w, s, z, g)))
    assert r.before is r.after is None


def test_scalar_rejects_indefinite_metric():
    with pytest.raises(ValueError, match="positive-semidefinite"):
        refine_affine_scalar(torch.zeros(2, 4), torch.ones(2, 1), torch.ones(2, 1), torch.zeros(4, dtype=torch.int32),
                              target=torch.ones(2, 4), bits=4, hessian=-torch.eye(4)*1e-8,
                              config=GSQConfig(enabled=True, steps=1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_awq_gemm_codes_match_actual_packer(dtype):
    from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.awq.utils.packing_utils import reverse_awq_order, unpack_awq

    rng = torch.Generator().manual_seed(7)
    linear = torch.nn.Linear(64, 32, bias=False, dtype=dtype)
    scales = (torch.rand(32, 2, generator=rng) * 0.12 + 0.001).to(dtype)
    zeros = torch.randint(0, 16, (32, 2), generator=rng).to(dtype)
    groups = torch.arange(64, dtype=torch.int32) // 32
    # Include off-grid values and endpoint saturation, not just exact powers of two.
    linear.weight.data.copy_(torch.randn(32, 64, generator=rng))
    packed = AwqTorchLinear(bits=4, group_size=32, sym=False, desc_act=False,
                           in_features=64, out_features=32, bias=False)
    # Exercise the GEMM CPU packing method using a CPU-capable container;
    # constructing the CUDA-only GEMM runtime would require a GPU lease.
    AwqGEMMLinear.pack(packed, linear, scales, zeros, groups)
    actual, _ = reverse_awq_order(*unpack_awq(packed.qweight, packed.qzeros, 4), 4)
    expected = affine_codes(linear.weight, scales, zeros, groups, 4, packing="awq_gemm",
                             scale_dtype=dtype if dtype != torch.float32 else torch.float16)
    assert torch.equal(actual.bitwise_and(15).T, expected)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_rtn_gsq_lifecycle_and_packed_objective(bits):
    from gptqmodel import BACKEND
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization import QuantizeConfig, RTNConfig
    from gptqmodel.quantization.rtn import RTN

    torch.manual_seed(7)
    layer = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
    cfg = RTNConfig(bits=bits, group_size=32, gsq=GSQConfig(enabled=True, steps=10, learn_scales=True))
    restored = QuantizeConfig.from_quant_config(cfg.to_dict())
    assert isinstance(restored, RTNConfig)
    assert restored.gsq == cfg.gsq
    quantizer = RTN(layer, restored)
    weight, scales, zeros, groups, *_ = quantizer.quantize()
    assert quantizer.gsq_diagnostics["after"] <= quantizer.gsq_diagnostics["before"]
    output = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
    output.weight.data.copy_(weight)
    packed = TorchLinear(bits=bits, group_size=32, sym=cfg.sym, desc_act=False,
                          in_features=64, out_features=32, bias=False, backend=BACKEND.TORCH)
    packed.pack_original(output, scales, zeros, groups)
    codes, packed_zeros = packed._unpack_continuous_codes()
    # The objective is the FP32 canonical reconstruction from stored metadata.
    # Native FP16 dequantization/forward rounding is a separate runtime gate.
    decoded = (packed.scales.float()[packed.g_idx] * (codes.float()-packed_zeros.float()[packed.g_idx])).T
    target = layer.weight.detach().float()
    loss = float((decoded-target).square().sum() / target.square().sum())
    assert loss == pytest.approx(quantizer.gsq_diagnostics["after"], rel=1e-5)


@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_gptq_gsq_uses_original_column_calibration(bits, desc_act):
    from gptqmodel.quantization import GPTQConfig, QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    rng = torch.Generator().manual_seed(7)
    layer = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
    layer.weight.data.copy_(torch.randn(32, 64, generator=rng)*0.05)
    inputs = torch.randn(128, 64, generator=rng) * torch.linspace(0.2, 2, 64)
    cfg = GPTQConfig(bits=bits, group_size=32, desc_act=desc_act,
                      act_group_aware=False, gsq=GSQConfig(enabled=True, steps=10))
    restored = QuantizeConfig.from_quant_config(cfg.to_dict())
    assert restored.gsq == cfg.gsq
    quantizer = GPTQ(layer, qcfg=restored)
    quantizer.quantizer.configure(perchannel=True)
    quantizer.add_batch(inputs, None)
    weight, scales, zeros, groups, *_ = quantizer.quantize()
    codes = affine_codes(weight, scales, zeros, groups, bits, packing="gptq")
    decoded = scales.half().float()[:, groups] * (codes-zeros[:, groups])
    target_output = layer.weight.detach().float() @ inputs.T
    loss = float(((decoded @ inputs.T)-target_output).square().sum() / target_output.square().sum())
    assert loss == pytest.approx(quantizer.gsq_diagnostics["after"], rel=1e-5)
    assert quantizer.gsq_diagnostics["after"] <= quantizer.gsq_diagnostics["before"]


@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_awq_gsq_preclip_teacher_and_packed_loss(sym, dtype):
    from test_adjacent_awq import _TestAWQProcessor

    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization import AWQConfig, QuantizeConfig

    cfg = AWQConfig(bits=4, group_size=32, sym=sym, gsq=GSQConfig(enabled=True, steps=10))
    restored = QuantizeConfig.from_quant_config(cfg.to_dict())
    assert restored.gsq == cfg.gsq
    processor = _TestAWQProcessor(restored)
    rng = torch.Generator().manual_seed(7)
    layer = torch.nn.Linear(64, 32, bias=False, dtype=dtype)
    layer.weight.data.copy_(torch.randn(32, 64, generator=rng)*0.2)
    named = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)
    references = processor._capture_adjacent_reference_weights({"proj": named})
    layer.weight.data.clamp_(-0.15, 0.15)
    inputs = torch.randn(97, 64, generator=rng)
    processor.apply_quant({"proj": named}, [], input_features={"proj": inputs}, adjacent_references=references)
    named.stream_sync()
    packed = AwqTorchLinear(bits=4, group_size=32, sym=sym, desc_act=False,
                             in_features=64, out_features=32, bias=False)
    packed.pack(layer, named.state["q_scales"], named.state["q_zeros"])
    # Decode stored values in FP32 for the canonical objective, retaining the
    # exact stored half/bfloat scale values.
    packed.scales = packed.scales.float()
    decoded = packed.dequantize_weight().T
    target_output = references["proj"].float() @ inputs.T
    loss = float(((decoded @ inputs.T)-target_output).square().sum() / target_output.square().sum())
    stats = named.state["gsq_diagnostics"]
    assert stats["objective"] == "activation_mse"
    assert loss == pytest.approx(stats["after"], rel=1e-5)
    assert stats["after"] <= stats["before"]


@pytest.mark.parametrize("method", ["gptq", "rtn"])
def test_scalar_lifecycle_disabled_preserves_outputs_and_rng(method):
    from gptqmodel.quantization import GPTQConfig, RTNConfig
    from gptqmodel.quantization.gptq import GPTQ
    from gptqmodel.quantization.rtn import RTN

    torch.manual_seed(7)
    layer = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
    inputs = torch.randn(128, 64)
    cls, config_cls = (GPTQ, GPTQConfig) if method == "gptq" else (RTN, RTNConfig)
    outputs = []
    for gsq in (None, GSQConfig(enabled=False)):
        quantizer = cls(layer, qcfg=config_cls(bits=4, group_size=32, gsq=gsq))
        if method == "gptq":
            quantizer.quantizer.configure(perchannel=True)
            quantizer.add_batch(inputs, None)
        before = torch.random.get_rng_state()
        outputs.append(quantizer.quantize()[:4])
        assert torch.equal(before, torch.random.get_rng_state())
        assert not hasattr(quantizer, "gsq_diagnostics")
    assert all(torch.equal(a, b) for a, b in zip(*outputs))


def test_scalar_diagonal_metric_matches_dense():
    scale = torch.ones(2, 1)
    zeros = torch.ones_like(scale)
    groups = torch.zeros(4, dtype=torch.int32)
    h = torch.tensor([0., 0.2, 0.5, 1.])
    kwargs = dict(target=torch.ones(2, 4), bits=2, config=GSQConfig(enabled=True, steps=5))
    diagonal = refine_affine_scalar(torch.zeros(2, 4), scale, zeros, groups, hessian=h, **kwargs)
    dense = refine_affine_scalar(torch.zeros(2, 4), scale, zeros, groups, hessian=h.diag(), **kwargs)
    assert diagonal.history == pytest.approx(dense.history)


@pytest.mark.parametrize("invalid", [
    "bits", "weight_shape", "scale_shape", "group_shape", "group_bounds", "nan_weight",
    "zero_scale", "fractional_zero", "both_metrics", "hessian_shape", "negative_diagonal",
    "empty_inputs", "candidate_budget", "storage_underflow", "packing", "baseline_overflow",
    "relaxed_overflow", "hard_overflow", "device", "gradient_overflow",
])
def test_scalar_rejects_invalid_or_nonfinite_fit(invalid):
    weight, target = torch.zeros(2, 4), torch.ones(2, 4)
    scales, zeros = torch.ones(2, 1), torch.ones(2, 1)
    groups = torch.zeros(4, dtype=torch.int32)
    kw = dict(bits=2, config=GSQConfig(enabled=True, steps=1))
    if invalid == "bits":
        kw["bits"] = True
    elif invalid == "weight_shape":
        target = torch.ones(4, 2)
    elif invalid == "scale_shape":
        scales = scales.flatten()
    elif invalid == "group_shape":
        groups = groups.float()
    elif invalid == "group_bounds":
        groups[0] = 1
    elif invalid == "nan_weight":
        weight[0, 0] = torch.nan
    elif invalid == "zero_scale":
        scales.zero_()
    elif invalid == "fractional_zero":
        zeros.fill_(0.5)
    elif invalid == "both_metrics":
        kw.update(hessian=torch.eye(4), inputs=torch.eye(4))
    elif invalid == "hessian_shape":
        kw["hessian"] = torch.eye(3)
    elif invalid == "negative_diagonal":
        kw["hessian"] = -torch.ones(4)
    elif invalid == "empty_inputs":
        kw["inputs"] = torch.ones(0, 4)
    elif invalid == "candidate_budget":
        kw["config"] = GSQConfig(enabled=True, max_candidate_bytes=1)
    elif invalid == "storage_underflow":
        scales.fill_(1e-20)
    elif invalid == "packing":
        kw["packing"] = "unverified"
    elif invalid == "baseline_overflow":
        weight.fill_(1)
        target.fill_(1e-30)
    elif invalid == "relaxed_overflow":
        target.fill_(1e-30)
        scales.fill_(16)
    elif invalid == "gradient_overflow":
        target.fill_(1e-30)
    elif invalid == "device":
        target = target.to("meta")
    elif invalid == "hard_overflow":
        weight.fill_(0.125)
        scales.fill_(0.125)
        zeros.fill_(2)
        target.fill_(0.2)
        kw.update(scale_dtype=torch.float32,
                  config=GSQConfig(enabled=True, steps=1, learn_scales=True, learning_rate=80))
    message = {"storage_underflow": "checkpoint dtype", "baseline_overflow": "baseline objective",
               "relaxed_overflow": "relaxed objective", "gradient_overflow": "gradient",
               "hard_overflow": "hard objective", "device": "share a device"}.get(invalid)
    with pytest.raises(ValueError, match=message):
        refine_affine_scalar(weight, scales, zeros, groups, target=target, **kw)
