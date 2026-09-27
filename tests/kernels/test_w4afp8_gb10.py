# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.looper.gptq_processor import enable_w4afp8_replay
from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear
from gptqmodel.nn_modules.qlinear.w4a_triton import _token_fp8_quant
from gptqmodel.quantization.activation_floatx import fp8_token_qdq
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.backend import BACKEND, backend_for_activation


def _packed_linear(device="cpu", k=128, n=32):
    module = W4AFP8Linear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=False,
    ).to(device)
    codes = torch.arange(k, device=device, dtype=torch.int32).remainder(16)
    words = (codes.reshape(k // 8, 8) << (
        torch.arange(8, device=device, dtype=torch.int32) * 4
    )).sum(dim=1).to(torch.int32)
    module.qweight.copy_(words[:, None].expand(k // 8, n))
    # GPTQ v1 stores the logical symmetric zero point 8 as nibble 7.
    module.qzeros.fill_(0x77777777)
    module.scales.copy_(torch.arange(1, k // 128 + 1, device=device, dtype=torch.float16)[:, None].expand(k // 128, n) / 8)
    module.post_init()
    return module


def test_activation_policy_round_trip(tmp_path):
    cfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        offload_to_disk=False, activation="w4afp8",
    )
    assert cfg.activation_mode == "w4afp8"
    assert backend_for_activation(cfg.activation_mode, BACKEND.AUTO) == BACKEND.GPTQ_W4AFP8
    cfg.save_pretrained(str(tmp_path))
    loaded = QuantizeConfig.from_pretrained(str(tmp_path))
    assert loaded.activation == {"version": 3, "mode": "w4afp8"}
    assert loaded.activation_version == 3
    assert loaded.pack_dtype == torch.int32
    with pytest.raises(ValueError, match="Version 1 used input-only rounding"):
        QuantizeConfig(
            bits=4, group_size=128, sym=True, desc_act=False,
            offload_to_disk=False, activation={"version": 1, "mode": "w4afp8"},
        )


def test_exact_fp8_weight_cache_and_native_checkpoint():
    module = _packed_linear()
    expected = (torch.arange(128, dtype=torch.float32).remainder(16) - 8)
    torch.testing.assert_close(module._weight_e4m3.float()[:, 0], expected, rtol=0, atol=0)
    assert set(module.state_dict()) == {"qweight", "qzeros", "scales", "g_idx"}
    assert module.qweight.dtype == torch.int32
    original = module._weight_e4m3.clone()
    module.post_init()
    torch.testing.assert_close(module._weight_e4m3.float(), original.float(), rtol=0, atol=0)


def test_replay_quantizes_only_selected_linear_input():
    linear = torch.nn.Linear(128, 32, bias=False, dtype=torch.float32)
    x = torch.randn(2, 3, 128)
    expected = linear(fp8_token_qdq(x))
    enable_w4afp8_replay(linear)
    torch.testing.assert_close(linear(x), expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
def test_token_fp8_codes_match_independent_torch_oracle():
    generator = torch.Generator(device="cuda").manual_seed(97)
    x = torch.randn((3, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    x[0].zero_()
    x[1, 0] = 400
    x[2, 1] = -500
    codes = torch.empty(x.shape, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((3,), device="cuda", dtype=torch.float32)
    _token_fp8_quant[(3,)](x, codes, scales, 128, 128, num_warps=4)
    amax = x.float().abs().amax(dim=-1)
    reference_scales = torch.where(amax > 0, amax / 448.0, 1.0)
    reference_codes = (x.float() / reference_scales[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.cuda.synchronize()
    torch.testing.assert_close(scales, reference_scales, rtol=1e-6, atol=1e-6)
    assert torch.equal(codes.view(torch.uint8), reference_codes.view(torch.uint8))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
@pytest.mark.parametrize("rows", [1, 16, 33])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gb10_native_fp8_group_gemm(rows, dtype):
    module = _packed_linear("cuda", k=256, n=64)
    generator = torch.Generator(device="cuda").manual_seed(1843)
    x = torch.randn(rows, 256, device="cuda", dtype=dtype, generator=generator)
    x[0].zero_()
    y = module(x)
    assert y.dtype == dtype
    # Independent Torch oracle: derive logical INT4 codes and per-token FP8
    # values from the checkpoint, then sum each GPTQ group in FP32.
    codes = torch.arange(256, device="cuda", dtype=torch.float32).remainder(16) - 8
    max_per_token = x.float().abs().amax(dim=1, keepdim=True)
    token_scales = torch.where(max_per_token == 0, 1.0, max_per_token / 448.0)
    rounded = (x.float() / token_scales).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    reference = torch.zeros((rows, 64), device="cuda", dtype=torch.float32)
    for group in range(2):
        sl = slice(group * 128, (group + 1) * 128)
        reference += (rounded[:, sl] @ codes[sl, None].expand(128, 64)) * module.scales[group].float()
    reference = (reference * token_scales).to(x.dtype)
    torch.testing.assert_close(y, reference, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
def test_gb10_fp8_random_columns_and_partial_tile():
    generator = torch.Generator(device="cuda").manual_seed(2029)
    k, n, rows = 256, 96, 7
    module = _packed_linear("cuda", k=k, n=n)
    logical_codes = torch.randint(0, 16, (k, n), generator=generator, device="cuda", dtype=torch.int32)
    shifts = torch.arange(8, device="cuda", dtype=torch.int32) * 4
    module.qweight.copy_((logical_codes.reshape(k // 8, 8, n) << shifts[None, :, None]).sum(dim=1))
    module.scales.copy_((0.02 + 0.18 * torch.rand((k // 128, n), generator=generator, device="cuda")).half())
    module.post_init()
    x = torch.randn((rows, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    actual = module(x)

    amax = x.float().abs().amax(dim=-1, keepdim=True)
    token_scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    xq = (x.float() / token_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    weight = logical_codes.float() - 8
    oracle = torch.zeros((rows, n), device="cuda", dtype=torch.float32)
    for group in range(k // 128):
        sl = slice(group * 128, (group + 1) * 128)
        oracle += (xq[:, sl] @ weight[sl]) * module.scales[group].float()
    oracle = (oracle * token_scale).to(actual.dtype)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_token_fp8_rounding_thresholds_and_neighbors(dtype):
    # Pin amax=448 so the scale is exactly one. Include ties between normal
    # and subnormal FP8 values, both signs, and input-dtype nextafter neighbors.
    thresholds = torch.tensor(
        [2.0**-10, 3.0 * 2.0**-10, 1.0625, 1.1875, 15.5, 248.0, 432.0],
        device="cuda", dtype=dtype,
    )
    values = torch.cat((
        torch.nextafter(thresholds, torch.full_like(thresholds, -torch.inf)),
        thresholds,
        torch.nextafter(thresholds, torch.full_like(thresholds, torch.inf)),
    ))
    x = torch.zeros((1, 128), device="cuda", dtype=dtype)
    x[0, :values.numel()] = values
    x[0, values.numel():2 * values.numel()] = -values
    x[0, -4:] = torch.tensor([0.0, -0.0, 448.0, -448.0], device="cuda", dtype=dtype)
    codes = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((1,), device="cuda", dtype=torch.float32)
    _token_fp8_quant[(1,)](x, codes, scales, 128, 128, num_warps=4)
    expected = x.float().clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.cuda.synchronize()
    torch.testing.assert_close(scales, torch.ones_like(scales), rtol=0, atol=0)
    assert torch.equal(codes.view(torch.uint8), expected.view(torch.uint8))
