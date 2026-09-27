# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent numeric checks for scale-aware activation transport."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation, pack_activation
from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
@pytest.mark.parametrize("mode", ["w4afp8"])
@pytest.mark.parametrize("rows,width", [(1, 128), (7, 256), (129, 128)])
def test_encoded_activation_matches_independent_torch_oracle(mode, rows, width):
    generator = torch.Generator(device="cuda").manual_seed(222 + rows)
    x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16, generator=generator)
    x[0, :16] = 0
    if rows > 1:
        x[1, 0] = 400
    encoded = pack_activation(x, mode)
    assert isinstance(encoded, W4AActivation)
    assert encoded.shape == x.shape
    assert encoded.codes.dtype == torch.float8_e4m3fn
    amax = x.float().abs().amax(dim=-1)
    reference_scale = torch.where(amax > 0, amax / 448.0, 1.0)
    reference_codes = (x.float() / reference_scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(encoded.scales, reference_scale, rtol=1e-6, atol=1e-6)
    assert torch.equal(encoded.codes.view(torch.uint8), reference_codes.view(torch.uint8))
    expected = reference_codes.float() * reference_scale[:, None]
    torch.cuda.synchronize()
    torch.testing.assert_close(encoded.decode(torch.float32), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
@pytest.mark.parametrize("mode,kernel", [
    ("w4afp8", W4AFP8Linear),
])
def test_prepacked_linear_matches_independent_torch_oracle(mode, kernel):
    k, n, rows = 256, 128, 3
    layer = kernel(bits=4, group_size=128, sym=True, desc_act=False,
                   in_features=k, out_features=n, bias=False).cuda()
    raw = (torch.arange(k, device="cuda")[:, None] + torch.arange(n, device="cuda")[None, :]) % 16
    shifts = (torch.arange(8, device="cuda", dtype=torch.int32) * 4)[None, :, None]
    layer.qweight.copy_((raw.reshape(k // 8, 8, n).to(torch.int32) << shifts).sum(dim=1))
    layer.qzeros.fill_(0x77777777)
    layer.scales[0].fill_(0.125)
    layer.scales[1].fill_(0.25)
    layer.post_init()

    generator = torch.Generator(device="cuda").manual_seed(486)
    x = torch.randn((1, rows, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    x[0, 0, :16] = 0
    encoded = pack_activation(x, mode)
    actual = layer(encoded)
    assert isinstance(actual, W4AActivation) and actual.mode == mode
    assert actual.shape == (1, rows, n)

    blocks = x.float().reshape(rows, k)
    maxima = blocks.abs().amax(dim=-1, keepdim=True)
    token_scale = torch.where(maxima > 0, maxima / 448.0, torch.ones_like(maxima))
    xq = (blocks / token_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * token_scale

    weight = raw.float() - 8
    reference = torch.zeros((rows, n), device="cuda", dtype=torch.float32)
    for group in range(2):
        sl = slice(group * 128, (group + 1) * 128)
        reference += (xq[:, sl] @ weight[sl]) * layer.scales[group].float()
    maxima = reference.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maxima > 0, maxima / 448.0, torch.ones_like(maxima))
    reference = (reference / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.decode(torch.float32).reshape(rows, n), reference, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
@pytest.mark.parametrize("mode,kernel", [
    ("w4afp8", W4AFP8Linear),
])
def test_rotated_prepacked_input_is_reencoded_before_gemm(monkeypatch, mode, kernel):
    """Online rotation consumes a carrier and produces a fresh hardware operand."""
    k = n = 128
    layer = kernel(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=False,
    ).cuda()
    raw = (torch.arange(k, device="cuda")[:, None] + torch.arange(n, device="cuda")[None, :]) % 16
    shifts = (torch.arange(8, device="cuda", dtype=torch.int32) * 4)[None, :, None]
    layer.qweight.copy_((raw.reshape(k // 8, 8, n).to(torch.int32) << shifts).sum(dim=1))
    layer.qzeros.fill_(0x77777777)
    layer.scales.fill_(0.125)
    layer.post_init()

    x = torch.randn((1, 2, k), device="cuda", dtype=torch.bfloat16)
    encoded = pack_activation(x, mode)
    rotated_dense = encoded.decode(torch.float32).flip(-1)
    expected_input = pack_activation(rotated_dense, mode, model_dtype=encoded.model_dtype)

    monkeypatch.setattr(type(layer), "_apply_rotation_to_input", lambda _self, value: value.flip(-1))
    layer.online_full_had = True
    actual = layer(encoded)
    layer.online_full_had = False
    expected = layer(expected_input)

    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.decode(torch.float32), expected.decode(torch.float32), rtol=2e-3, atol=2e-3
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
@pytest.mark.parametrize("mode,kernel", [
    ("w4afp8", W4AFP8Linear),
])
def test_pre_rotated_carrier_is_consumed_without_second_rounding(monkeypatch, mode, kernel):
    """The MLP producer may rotate before creating the hardware operand."""
    k = n = 128
    layer = kernel(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=False,
    ).cuda()
    raw = (torch.arange(k, device="cuda")[:, None] + torch.arange(n, device="cuda")[None, :]) % 16
    shifts = (torch.arange(8, device="cuda", dtype=torch.int32) * 4)[None, :, None]
    layer.qweight.copy_((raw.reshape(k // 8, 8, n).to(torch.int32) << shifts).sum(dim=1))
    layer.qzeros.fill_(0x77777777)
    layer.scales.fill_(0.125)
    layer.post_init()

    rotated = torch.randn((1, 2, k), device="cuda", dtype=torch.bfloat16)
    pre_rotated = pack_activation(rotated, mode, rotation_applied=True)
    ordinary = pack_activation(rotated, mode)
    monkeypatch.setattr(
        type(layer), "_apply_rotation_to_input",
        lambda _self, _value: (_ for _ in ()).throw(AssertionError("rotation repeated")),
    )
    layer.online_full_had = True
    actual = layer(pre_rotated)
    layer.online_full_had = False
    expected = layer(ordinary)

    torch.cuda.synchronize()
    assert actual.rotation_applied is False
    torch.testing.assert_close(
        actual.decode(torch.float32), expected.decode(torch.float32), rtol=2e-3, atol=2e-3
    )
