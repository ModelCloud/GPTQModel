# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from tests.models.w4a_token_energy import install_token_energy_diagnostic, token_energy_gain


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [128, 2048, 8192])
def test_gain_matches_independent_scalar_double_oracle(dtype, width):
    generator = torch.Generator().manual_seed(783)
    source = torch.randn(2, 3, width, generator=generator).to(dtype)
    decoded = (source.float() + .08 * torch.randn(source.shape, generator=generator)).to(dtype)
    expected = torch.tensor([
        math.sqrt(math.fsum(float(x) ** 2 for x in left) / math.fsum(float(x) ** 2 for x in right))
        for left, right in zip(source.reshape(-1, width), decoded.reshape(-1, width), strict=True)
    ])
    actual = token_energy_gain(source, decoded)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    corrected = decoded.double().reshape(-1, width) * actual.double()[:, None]
    torch.testing.assert_close(corrected.norm(dim=-1), source.double().reshape(-1, width).norm(dim=-1),
                               rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("magnitude", [1e-35, 1e30])
def test_extreme_finite_values_do_not_overflow_or_underflow(magnitude):
    source = torch.full((3, 128), magnitude)
    gain = token_energy_gain(source, source * .5)
    torch.testing.assert_close(gain, torch.full((3,), 2.), rtol=1e-6, atol=1e-6)


def test_zero_tokens_and_empty_batch():
    assert torch.equal(token_energy_gain(torch.zeros(2, 128), torch.zeros(2, 128)), torch.ones(2))
    assert torch.equal(token_energy_gain(torch.zeros(2, 128), torch.ones(2, 128)), torch.zeros(2))
    assert token_energy_gain(torch.empty(0, 128), torch.empty(0, 128)).shape == (0,)
    with pytest.raises(ValueError, match="entirely zero"):
        token_energy_gain(torch.ones(2, 128), torch.zeros(2, 128))


@pytest.mark.parametrize("source,decoded", [
    (torch.ones(2, 128), torch.ones(3, 128)),
    (torch.ones(128), torch.ones(128)),
    (torch.ones(2, 128).long(), torch.ones(2, 128)),
    (torch.full((2, 128), float("nan")), torch.ones(2, 128)),
    (torch.ones(2, 128), torch.full((2, 128), float("inf"))),
])
def test_invalid_inputs_fail(source, decoded):
    with pytest.raises(ValueError):
        token_energy_gain(source, decoded)


@pytest.mark.parametrize("scope,count", [("all", 5), ("residual", 3)])
def test_hooks_preserve_carrier_storage_and_are_removable(monkeypatch, scope, count):
    from gptqmodel.nn_modules.qlinear import w4a_boundary
    from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation

    keys = ["model.layers.0.input", "model.layers.0.self_attn.o_proj.input",
            "model.layers.0.post_attention_residual", "model.layers.0.mlp.down_proj.input",
            "model.layers.0.output"]
    core = torch.nn.ModuleList([w4a_boundary.NVFP4BoundaryQuantizer(key, "cpu") for key in keys])
    codes = torch.zeros((2, 64), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scales = torch.ones(1, dtype=torch.float32)
    operand = W4AActivation("w4a_nvfp4", codes, scales, (2, 128), torch.bfloat16,
                            torch.tensor(1.), token_scale=torch.ones(2))
    monkeypatch.setattr(w4a_boundary, "pack_activation", lambda *args, **kwargs: operand)
    monkeypatch.setattr(W4AActivation, "decode", lambda self, dtype: torch.full(self.shape, 2.))
    handles, stats = install_token_energy_diagnostic(core, scope)
    assert len(handles) == count
    try:
        outputs = [module(torch.full((2, 128), 4.), "w4a_nvfp4") for module in core]
        assert len(stats) == count
        for module, result in zip(core, outputs, strict=True):
            assert result.codes is codes and result.scales is scales
            expected = 2. if module.key in stats else 1.
            assert torch.equal(result.token_scale, torch.full((2,), expected))
        assert torch.equal(operand.token_scale, torch.ones(2))
    finally:
        for handle in handles:
            handle.remove()
    assert all(not module._forward_hooks for module in core)


def test_install_rejects_missing_producers_and_unknown_scope():
    with pytest.raises(ValueError, match="producers"):
        install_token_energy_diagnostic(torch.nn.Identity(), "all")
    with pytest.raises(ValueError, match="scope"):
        install_token_energy_diagnostic(torch.nn.Identity(), "unknown")
