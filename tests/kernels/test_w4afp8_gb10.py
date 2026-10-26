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
from tests.w4a_hardware_marks import FP8_HARDWARE


def _assert_native_gemm_close(actual, expected, magnitude):
    """Compare a native FP8/NVFP4 MMA result against the independent FP32 oracle.

    Hopper/Blackwell FP8 tensor cores accumulate each ``mma.sync`` k=32 product
    set in full FP32, so the result must match the oracle to ``rtol=atol=2e-3``.
    Ada Lovelace (SM 8.9) does not: the hardware accumulator only keeps a
    reduced-precision fixed-point window and truncates the products against the
    largest term of the same instruction. A raw
    ``mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`` probe with no Triton
    involved reproduces the kernel output bit for bit on RTX 4090 (an
    alternating 448/1 row sums to 28712 instead of 28736), so the drift is a
    hardware property, not a kernel defect. It stays bounded by the magnitude
    of the summed products (measured worst case ~3e-5 of ``sum |a*b|``), which
    is what this Ada-specific bound asserts instead of the strict equality.
    """
    ordinal = actual.device.index
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    if torch.cuda.get_device_capability(ordinal) >= (9, 0):
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-3, atol=2e-3)
        return
    delta = (actual.float() - expected.float()).abs()
    bound = 1e-2 * expected.float().abs() + 2e-4 * magnitude
    slack = delta - bound
    if bool((slack <= 0).all()):
        return
    index = int(torch.argmax(slack))
    raise AssertionError(
        "Ada FP8 MMA drift exceeds the documented hardware bound: "
        f"|delta|={delta.flatten()[index].item():.3e} > "
        f"bound={bound.flatten()[index].item():.3e} "
        f"at ({index // delta.shape[1]}, {index % delta.shape[1]})"
    )


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
    assert loaded.activation == {"mode": "w4afp8"}
    assert loaded.pack_dtype == torch.int32
    with pytest.raises(ValueError, match="unsupported field"):
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


@FP8_HARDWARE
def test_token_fp8_codes_match_independent_torch_oracle():
    x = torch.randn((3, 128), device="cuda", dtype=torch.bfloat16)
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


@FP8_HARDWARE
@pytest.mark.parametrize("rows", [1, 16, 33])
def test_gb10_native_fp8_group_gemm(rows):
    module = _packed_linear("cuda", k=256, n=64)
    generator = torch.Generator(device="cuda").manual_seed(1843)
    x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    x[0].zero_()
    y = module(x)
    # Independent Torch oracle: derive logical INT4 codes and per-token FP8
    # values from the checkpoint, then sum each GPTQ group in FP32.
    codes = torch.arange(256, device="cuda", dtype=torch.float32).remainder(16) - 8
    max_per_token = x.float().abs().amax(dim=1, keepdim=True)
    token_scales = torch.where(max_per_token == 0, 1.0, max_per_token / 448.0)
    rounded = (x.float() / token_scales).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    reference = torch.zeros((rows, 64), device="cuda", dtype=torch.float32)
    magnitude = torch.zeros((rows, 64), device="cuda", dtype=torch.float32)
    for group in range(2):
        sl = slice(group * 128, (group + 1) * 128)
        reference += (rounded[:, sl] @ codes[sl, None].expand(128, 64)) * module.scales[group].float()
        magnitude += (rounded[:, sl].abs() @ codes[sl, None].expand(128, 64).abs()) * module.scales[group].float().abs()
    reference = (reference * token_scales).to(x.dtype)
    magnitude = magnitude * token_scales
    _assert_native_gemm_close(y, reference, magnitude)


@FP8_HARDWARE
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
    magnitude = torch.zeros((rows, n), device="cuda", dtype=torch.float32)
    for group in range(k // 128):
        sl = slice(group * 128, (group + 1) * 128)
        oracle += (xq[:, sl] @ weight[sl]) * module.scales[group].float()
        magnitude += (xq[:, sl].abs() @ weight[sl].abs()) * module.scales[group].float().abs()
    oracle = (oracle * token_scale).to(actual.dtype)
    magnitude = magnitude * token_scale
    torch.cuda.synchronize()
    _assert_native_gemm_close(actual, oracle, magnitude)


def _round_toward_zero_to_fp16(value: torch.Tensor) -> torch.Tensor:
    """Emulate ``cvt.rz.f16.f32`` for values inside the normal fp16 range."""
    return (value.view(torch.int32) & ~0x1FFF).view(torch.float32)


@FP8_HARDWARE
def test_token_fp8_rounding_holds_at_e4m3_midpoint_neighbours():
    """Values just above an E4M3 midpoint must round like the Torch oracle.

    ``_token_fp8_quant`` divides by a non-power-of-two token scale, so the value
    handed to the E4M3 conversion is a general fp32 number. Lowering
    ``.to(tl.float8e4nv)`` to ``cvt.rz.f16.f32`` + ``cvt.rn.satfinite.e4m3x2.f16x2``
    truncates through fp16 first and then mis-rounds every value that sits within
    one fp16 step above a midpoint; SM 8.9 has the direct
    ``cvt.rn.satfinite.e4m3x2.f32`` and must use it.

    Candidates are searched against the oracle instead of being assumed: each row
    is a bf16 input the old double-rounding path provably disagrees on, so the
    coverage cannot go vacuous if the lowering changes.
    """
    width = 128
    # 300 / 448 is not a power of two, so x / scale lands off the fp16 grid.
    amax = torch.tensor(300.0, dtype=torch.float32)
    row_scale = (amax / 448.0).to(torch.float32)
    grid = torch.linspace(0.05, 2.0, 8192, dtype=torch.float32).to(torch.bfloat16)
    pre_round = (grid.float() / row_scale).clamp(-448, 448)
    direct = pre_round.to(torch.float8_e4m3fn)
    through_fp16 = _round_toward_zero_to_fp16(pre_round).to(torch.float8_e4m3fn)
    discriminating = grid[direct.view(torch.uint8) != through_fp16.view(torch.uint8)]
    assert discriminating.numel() >= 16, (
        "the swept inputs no longer separate the direct E4M3 conversion from the "
        "fp16-truncating one, so this test would be vacuous"
    )
    candidates = list(discriminating[:64])

    padded = torch.zeros((len(candidates), width), dtype=torch.bfloat16)
    for row, encoded in enumerate(candidates):
        padded[row, 0] = encoded
        padded[row, 1] = amax.to(torch.bfloat16)

    x = padded.to(device="cuda")
    codes = torch.empty(x.shape, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0],), device="cuda", dtype=torch.float32)
    _token_fp8_quant[(x.shape[0],)](x, codes, scales, width, width, num_warps=4)
    torch.cuda.synchronize()

    row_amax = x.float().abs().amax(dim=-1, keepdim=True)
    reference_scales = row_amax / 448.0
    pre_round = (x.float() / reference_scales).clamp(-448, 448)
    oracle = pre_round.to(torch.float8_e4m3fn)
    torch.testing.assert_close(scales, reference_scales.squeeze(1), rtol=0, atol=0)
    assert torch.equal(codes.view(torch.uint8), oracle.view(torch.uint8))
