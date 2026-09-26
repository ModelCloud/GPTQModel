# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import GSQConfig, QuantizeConfig
from gptqmodel.quantization.config import FORMAT
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.gsq_scalar import gsq_enabled_for, refine_gptq_scalar
from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module


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
