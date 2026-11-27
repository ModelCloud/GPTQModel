# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math

import pytest
import torch
from torch import nn

from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module
from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import GSQConfig, QuantizeConfig
from gptqmodel.quantization.config import FORMAT
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.gsq_scalar import (
    gsq_enabled_for,
    refine_affine_scalar,
    refine_gptq_scalar,
)


def test_gsq_config_round_trip_and_scope():
    config = QuantizeConfig(
        bits=4,
        group_size=16,
        gsq=GSQConfig(enabled=True, steps=3, modules=(r"q_proj$",)),
    )
    restored = QuantizeConfig.from_quant_config(config.to_dict())
    assert restored.gsq == config.gsq
    assert restored.gsq.modules == (r"q_proj$",)

    with pytest.raises(ValueError, match="mock quantization"):
        QuantizeConfig(bits=4, mock_quantization=True, gsq=GSQConfig(enabled=True))
    with pytest.raises(ValueError, match="checkpoint format"):
        QuantizeConfig(bits=4, format=FORMAT.MARLIN, gsq=GSQConfig(enabled=True))

    assert gsq_enabled_for(restored.gsq, "model.layers.0.self_attn.q_proj")
    assert not gsq_enabled_for(restored.gsq, "model.layers.0.self_attn.k_proj")

    dynamic = QuantizeConfig(
        bits=4, group_size=16,
        dynamic={r".*q_proj$": {"gsq": {"enabled": True, "steps": 2}}},
    )
    selected = clone_gptq_config_for_module(dynamic, "model.layers.0.self_attn.q_proj")
    unselected = clone_gptq_config_for_module(dynamic, "model.layers.0.self_attn.k_proj")
    assert selected.gsq.enabled and selected.gsq.steps == 2
    assert unselected.gsq is None


def test_gsq_refinement_chunks_rows_and_keeps_baseline_if_no_better_export():
    torch.manual_seed(11)
    target = torch.randn(5, 8, dtype=torch.float16)
    scales = torch.full((5, 2), 0.25, dtype=torch.float16)
    zeros = torch.full_like(scales, 8)
    groups = torch.arange(8, dtype=torch.int32) // 4
    codes = (target / scales[:, groups] + zeros[:, groups]).round().clamp(0, 15)
    baseline = scales[:, groups] * (codes - zeros[:, groups])
    x = torch.randn(18, 8)
    hessian = x.T @ x
    result = refine_gptq_scalar(
        baseline, scales, zeros, groups,
        target=target, hessian=hessian, bits=4,
        config=GSQConfig(enabled=True, steps=3, candidates=5, learn_scales=True,
                         max_candidate_bytes=8 * 5 * 4 * 2),
    )
    assert result.weight.shape == baseline.shape
    assert result.scales.shape == scales.shape
    assert result.after <= result.before + 1e-7
    assert len(result.history) == 4
    assert torch.equal(result.g_idx, groups)
    assert torch.equal(result.zeros, zeros)


def test_gsq_rank_deficient_hessian():
    target = torch.tensor([[0.15, -0.20, 0.35, 0.10]], dtype=torch.float16)
    scales = torch.full((1, 1), 0.1, dtype=torch.float16)
    zeros = torch.full_like(scales, 8)
    groups = torch.zeros(4, dtype=torch.int32)
    baseline = torch.zeros_like(target)
    inputs = torch.tensor([[1.0, 2.0, 1.0, 0.0], [2.0, 4.0, 2.0, 0.0]])
    hessian = inputs.T @ inputs
    result = refine_gptq_scalar(
        baseline, scales, zeros, groups,
        target=target, hessian=hessian, bits=4,
        config=GSQConfig(enabled=True, steps=4, candidates=4),
    )
    assert math.isfinite(result.after)
    assert result.after <= result.before


def test_gsq_moves_integer_codes_when_reconstruction_improves():
    baseline = torch.zeros((1, 4), dtype=torch.float16)
    scales = torch.full((1, 1), 0.1, dtype=torch.float16)
    zeros = torch.full_like(scales, 8)
    groups = torch.zeros(4, dtype=torch.int32)
    target = torch.tensor([[0.35, -0.15, 0.50, -0.20]], dtype=torch.float16)
    result = refine_gptq_scalar(
        baseline, scales, zeros, groups,
        target=target, hessian=torch.eye(4), bits=4,
        config=GSQConfig(enabled=True, steps=30, candidates=16, seed=7),
    )
    assert result.after < result.before
    assert not torch.equal(result.weight, baseline)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("desc_act,group_size", [(False, 16), (True, 16), (False, -1)])
def test_gsq_gptq_pack_and_reconstruction(bits, desc_act, group_size):
    torch.manual_seed(20 + bits)
    layer = nn.Linear(32, 32, bias=False, dtype=torch.float16).eval()
    target = layer.weight.detach().float().clone()
    config = QuantizeConfig(
        bits=bits, group_size=group_size, desc_act=desc_act,
        act_group_aware=False if group_size == -1 else None,
        gsq=GSQConfig(enabled=True, steps=2, candidates=4, learn_scales=True,
                      max_candidate_bytes=32 * 4 * 4 * 8),
    )
    gptq = GPTQ(layer, qcfg=config)
    gptq.quantizer.configure(perchannel=True)
    activations = torch.randn(1, 64, 32, dtype=torch.float16)
    gptq.add_batch(activations, None)
    hessian = gptq.finalize_hessian().float().clone()
    quantized, scales, zeros, groups, *_ = gptq.quantize(blocksize=32)
    diagnostics = gptq.gsq_diagnostics
    assert diagnostics["after"] <= diagnostics["before"] + 1e-7

    with torch.no_grad():
        layer.weight.copy_(quantized)
    packed = TorchLinear(
        bits=bits, group_size=group_size, sym=config.sym, desc_act=config.desc_act,
        in_features=32, out_features=32, format=config.format,
    )
    packed.pack_original(layer.cpu(), scales.cpu(), zeros.cpu(), groups.cpu())
    decoded = packed.dequantize_weight().T.float()
    error = decoded - target
    energy = ((target @ torch.linalg.cholesky(hessian)).square().sum()).item()
    measured = (error @ hessian * error).sum().item() / energy
    assert math.isclose(measured, diagnostics["after"], rel_tol=2e-2, abs_tol=2e-4)


def test_gsq_disabled_matches_original_gptq():
    torch.manual_seed(42)
    layer = nn.Linear(16, 16, bias=False, dtype=torch.float16).eval()
    samples = torch.randn(1, 32, 16, dtype=torch.float16)
    outputs = []
    for gsq in (None, GSQConfig(enabled=False)):
        gptq = GPTQ(layer, qcfg=QuantizeConfig(bits=4, group_size=16, gsq=gsq))
        gptq.quantizer.configure(perchannel=True)
        gptq.add_batch(samples, None)
        outputs.append(gptq.quantize(blocksize=16))
    for left, right in zip(outputs[0][:4], outputs[1][:4]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_gsq_weight_replays_in_inference_mode():
    with torch.inference_mode():
        layer = nn.Linear(16, 16, bias=False, dtype=torch.float16).eval()
        inputs = torch.randn(1, 32, 16, dtype=torch.float16)
    hooked = HookedLinear.from_linear(layer)
    gptq = GPTQ(
        hooked,
        qcfg=QuantizeConfig(
            bits=4, group_size=16,
            gsq=GSQConfig(enabled=True, steps=2, candidates=4),
        ),
    )
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(inputs, None)
    weight, *_ = gptq.quantize(blocksize=16)
    assert weight.is_inference()
    hooked.weight.data = weight
    with torch.inference_mode():
        assert hooked(inputs).shape == (1, 32, 16)


def _packed_codes(weight, scales, zeros, g_idx, bits):
    scale = scales.float()[:, g_idx]
    return ((weight.float() + zeros.float()[:, g_idx] * scale) / scale).round().clamp(0, 2**bits - 1)


def _hessian_metric(weight, target, hessian):
    factor = torch.linalg.cholesky((hessian.float() + hessian.float().T) * 0.5)
    error = (weight.float() - target.float()) @ factor
    teacher = target.float() @ factor
    return (error.square().sum() / teacher.square().sum()).item()


def test_gsq_moves_codes_away_from_a_gptq_local_optimum():
    """#4592: the initializer is a local optimum of the same Gram.

    A GPTQ weight is already optimal for its own codes, so a refiner that only
    follows the relaxed code expectation returns it unchanged. The hard search
    has to leave that point and lower the exported score.
    """

    torch.manual_seed(4592)
    rows, columns, bits, group_size = 24, 32, 4, 8
    layer = nn.Linear(columns, rows, bias=False, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.mul_(0.5)
    config = QuantizeConfig(bits=bits, group_size=group_size, gsq=None)
    gptq = GPTQ(layer, qcfg=config)
    gptq.quantizer.configure(perchannel=True)
    samples = torch.randn(1, 96, columns, dtype=torch.float16)
    for sample in samples:
        gptq.add_batch(sample, None)
    hessian = gptq.finalize_hessian().float().clone()
    weight, scales, zeros, groups, *_ = gptq.quantize(blocksize=32)

    result = refine_gptq_scalar(
        weight, scales, zeros, groups.to(torch.int32), target=layer.weight.detach().float(),
        bits=bits, hessian=hessian,
        config=GSQConfig(enabled=True, steps=8, candidates=5, seed=3),
    )

    assert result.after < result.before
    assert not torch.equal(result.weight, weight)
    codes = _packed_codes(result.weight, result.scales, result.zeros, result.g_idx.int(), bits)
    assert _hessian_metric(result.weight, layer.weight, hessian) <= result.after * (1 + 1e-3)
    assert torch.equal(codes, _packed_codes(
        result.weight, result.scales, result.zeros, result.g_idx.int(), bits))


def test_gsq_accepts_per_row_and_keeps_worse_rows_at_the_initializer():
    """A chunk is accepted row by row, not as one all-or-nothing block."""

    torch.manual_seed(7)
    rows, columns, bits = 12, 16, 4
    groups = (torch.arange(columns) // 4).to(torch.int32)
    scales = torch.full((rows, columns // 4), 0.1, dtype=torch.float16)
    zeros = torch.full_like(scales, 8)
    target = torch.randn(rows, columns, dtype=torch.float16) * 0.4
    # One row is already exact at the initializer; its codes must not move.
    target[0] = 0.0
    weight = scales.float()[:, groups] * (
        target.float() / scales.float()[:, groups]
    ).round().clamp(0, 2**bits - 1)

    result = refine_gptq_scalar(
        weight, scales, zeros, groups, target=target, bits=bits,
        hessian=torch.eye(columns),
        config=GSQConfig(enabled=True, steps=6, candidates=5, seed=11),
    )

    assert result.after <= result.before
    assert torch.equal(result.weight[0], weight[0])


def test_gsq_does_not_accept_a_worse_export():
    """The relaxed phase may never ship a worse hard export."""

    torch.manual_seed(5)
    columns, bits = 8, 4
    groups = torch.zeros(columns, dtype=torch.int32)
    scales = torch.full((1, 1), 0.5, dtype=torch.float16)
    zeros = torch.full_like(scales, 8)
    target = torch.tensor([[3.7, -3.7, 3.7, -3.7, 3.7, -3.7, 3.7, -3.7]], dtype=torch.float16)
    weight = torch.zeros_like(target)

    result = refine_gptq_scalar(
        weight, scales, zeros, groups, target=target, bits=bits,
        hessian=torch.eye(columns),
        config=GSQConfig(enabled=True, steps=3, candidates=4),
    )

    assert result.after <= result.before


def test_gsq_history_matches_the_configured_step_budget():
    torch.manual_seed(13)
    columns, rows, bits = 8, 3, 4
    groups = torch.zeros(columns, dtype=torch.int32)
    scales = torch.full((rows, 1), 0.2, dtype=torch.float16)
    zeros = torch.full_like(scales, 8)
    target = torch.randn(rows, columns, dtype=torch.float16)
    weight = torch.zeros_like(target)
    config = GSQConfig(enabled=True, steps=5, candidates=4)

    result = refine_gptq_scalar(
        weight, scales, zeros, groups, target=target, bits=bits,
        hessian=torch.eye(columns), config=config,
    )

    assert len(result.history) == config.steps + 1
    assert result.history[0] == pytest.approx(result.before)
    assert result.after <= result.before


def test_gsq_hard_search_score_matches_the_packed_export():
    """The reported ``after`` must be measurable on the packed checkpoint."""

    torch.manual_seed(21)
    rows, columns, bits, group_size = 10, 24, 4, 6
    layer = nn.Linear(columns, rows, bias=False, dtype=torch.float16)
    samples = torch.randn(1, 64, columns, dtype=torch.float16)
    gptq = GPTQ(layer, qcfg=QuantizeConfig(bits=bits, group_size=group_size))
    gptq.quantizer.configure(perchannel=True)
    for sample in samples:
        gptq.add_batch(sample, None)
    hessian = gptq.finalize_hessian().float().clone()
    weight, scales, zeros, groups, *_ = gptq.quantize(blocksize=24)

    result = refine_gptq_scalar(
        weight, scales, zeros, groups.to(torch.int32), target=layer.weight.detach().float(),
        bits=bits, hessian=hessian,
        config=GSQConfig(enabled=True, steps=6, candidates=5),
        scale_dtype=scales.dtype,
    )

    codes = ((result.weight.float() + result.zeros.float()[:, result.g_idx] * result.scales.float()[:, result.g_idx])
             / result.scales.float()[:, result.g_idx]).round().clamp(0, 2**bits - 1)
    decoded = result.scales.float()[:, result.g_idx] * (codes - result.zeros.float()[:, result.g_idx])
    measured = _hessian_metric(decoded, layer.weight, hessian)
    assert measured == pytest.approx(result.after, rel=2e-4, abs=2e-6)


def _unrepresentable_endpoint_fixture(columns=16):
    """A finite FP16 grid whose negative endpoint reconstructs ``-65536``."""
    scale = torch.tensor([[8192.0]], dtype=torch.float16)
    zero = torch.tensor([[8.0]], dtype=torch.float16)
    groups = torch.zeros(columns, dtype=torch.int32)
    weight = torch.full((1, columns), -57344.0, dtype=torch.float16)
    target = torch.full((1, columns), -62000.0, dtype=torch.float32)
    config = GSQConfig(
        enabled=True, steps=60, candidates=16, seed=7, learning_rate=0.1,
        temperature_start=1.0, temperature_end=0.05, learn_scales=False,
    )
    return scale, zero, groups, weight, target, config


def test_gsq_scalar_never_accepts_a_reconstruction_outside_the_replay_dtype():
    """A code that narrows to ``-inf`` must not win on its FP32 score.

    ``scale=8192, zero=8`` makes code 0 reconstruct ``-65536``: finite in
    FP32 but ``-inf`` in FP16, where the packer recovers code 0 from it. The
    target is closer to that endpoint than to the baseline code 1, so an
    unguarded search accepts the narrowed ``-inf`` candidate. Both scalar
    refiners must keep the finite initializer.
    """

    scale, zero, groups, weight, target, config = _unrepresentable_endpoint_fixture()
    baseline = weight.float()
    endpoint = scale.float() * (0.0 - zero.float())
    baseline_error = (baseline - target).square().sum()
    endpoint_error = (endpoint - target).square().sum()
    # Trigger: the unrepresentable endpoint is finite in FP32, becomes -inf in
    # the replay dtype, and scores better than the baseline.
    assert torch.isfinite(endpoint).all()
    assert not torch.isfinite(endpoint.to(weight.dtype)).all()
    assert endpoint_error < baseline_error

    affine = refine_affine_scalar(
        weight, scale, zero, groups, target=target, bits=4, config=config,
        hessian=torch.ones(weight.shape[1]), packing="gptq", scale_dtype=torch.float16,
    )
    assert torch.isfinite(affine.weight).all()
    assert torch.equal(affine.weight, weight)
    assert affine.after <= affine.before

    gptq = refine_gptq_scalar(
        weight, scale, zero, groups, target=target, bits=4, config=config,
        hessian=torch.eye(weight.shape[1]), scale_dtype=torch.float16,
    )
    assert torch.isfinite(gptq.weight).all()
    assert torch.equal(gptq.weight, weight)
    assert gptq.after <= gptq.before


def test_gsq_scalar_keeps_a_finite_initializer_at_the_grid_endpoint():
    """A finite initializer whose own recovered code overflows stays finite."""

    scale, zero, groups, _, target, config = _unrepresentable_endpoint_fixture()
    weight = torch.full((1, 16), -65504.0, dtype=torch.float16)
    # The grid recovers code 0 from this finite weight; its FP32 decode is
    # -65536 and narrows to -inf. The initializer itself must be returned.
    result = refine_gptq_scalar(
        weight, scale, zero, groups, target=target, bits=4, config=config,
        hessian=torch.eye(weight.shape[1]), scale_dtype=torch.float16,
    )
    assert torch.isfinite(result.weight).all()
    assert torch.equal(result.weight, weight)
    assert result.after <= result.before
