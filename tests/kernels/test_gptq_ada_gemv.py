# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for the mocked launch-table cases; the kernel-oracle cases carry
# ``pytest.mark.cuda`` and skip unless the device is an Ada (SM 8.9) GPU.
"""Ada (RTX 4090, SM 8.9) fused GPTQ GEMV: launch table + Torch-oracle A/B.

The oracle is independent: it decodes the packed int32 words with Torch integer
shifts and accumulates in FP64.  It never calls the Triton kernel.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.triton_utils.gptq_gemv import (
    _DEFAULT_GEMV_LAUNCH,
    _GEMV_LAUNCH_CONFIGS,
    GemvLaunch,
    block_k_for,
    gptq_gemv,
    launch_config,
)
from gptqmodel.utils import arch


TOL = 2e-3


@pytest.fixture(autouse=True)
def _arch_cache():
    arch.clear_arch_cache()
    yield
    arch.clear_arch_cache()


def _is_ada_host() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and arch.is_ada(0)
    )


requires_ada = pytest.mark.skipif(not _is_ada_host(), reason="Ada (SM 8.9) CUDA GPU required")


# --------------------------------------------------------------------------- #
# CPU-only: launch-table dispatch through the shared tuning-family vocabulary.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("arch_id", "expected"),
    [
        ("sm_89", GemvLaunch(max_block_k=256, block_n=8, num_warps=8)),
        ((8, 9), GemvLaunch(max_block_k=256, block_n=8, num_warps=8)),
        ("gfx1201", GemvLaunch(max_block_k=512, block_n=16, num_warps=4)),
        ("gfx950", GemvLaunch(max_block_k=512, block_n=8, num_warps=8)),
        ("gfx942", _DEFAULT_GEMV_LAUNCH),
        ("sm_80", _DEFAULT_GEMV_LAUNCH),
        ("sm_90", _DEFAULT_GEMV_LAUNCH),
        ("", _DEFAULT_GEMV_LAUNCH),
    ],
)
def test_gemv_launch_routes_ada(arch_id, expected):
    assert launch_config(arch_id) == expected


def test_gemv_ada_is_an_explicit_entry_not_the_default():
    # The user asked for an explicit table entry: it must exist even though the
    # shipped values differ from the A100 default here.
    assert "ada" in _GEMV_LAUNCH_CONFIGS
    assert "rdna4" in _GEMV_LAUNCH_CONFIGS and "cdna4" in _GEMV_LAUNCH_CONFIGS
    assert launch_config("sm_89") == _GEMV_LAUNCH_CONFIGS["ada"]
    # No-arch keeps the A100-tested default rather than inheriting the device.
    assert launch_config(None) == _DEFAULT_GEMV_LAUNCH


def test_gemv_ada_block_k_cap_is_256():
    assert block_k_for(4096, "sm_89") == 256
    assert block_k_for(256, "sm_89") == 256
    assert block_k_for(128, "sm_89") == 128
    assert block_k_for(5120, (8, 9)) == 256


# --------------------------------------------------------------------------- #
# Kernel-oracle cases (real hardware only).
# --------------------------------------------------------------------------- #

def _make_module(in_features, out_features, dtype, group_size=128,
                 desc_act=False, bias=False, bits=4):
    module = TorchLinear(
        bits=bits, group_size=group_size, sym=True, desc_act=desc_act,
        in_features=in_features, out_features=out_features,
        pack_dtype=torch.int32, bias=bias, register_buffers=True,
    ).to(device="cuda", dtype=dtype).eval()
    module.optimize = lambda *args, **kwargs: None
    module.qzero_format(format=2)
    with torch.no_grad():
        module.qweight.copy_(torch.randint(-2**31, 2**31, module.qweight.shape, device="cuda", dtype=torch.int32))
        module.qzeros.copy_(torch.randint(-2**31, 2**31, module.qzeros.shape, device="cuda", dtype=torch.int32))
        module.scales.uniform_(0.001, 0.02)
        if desc_act:
            module.g_idx.copy_(module.g_idx[torch.randperm(in_features, device="cuda")])
            module.g_idx[0] = -1
        if bias:
            module.bias.uniform_(-0.05, 0.05)
    module.post_init()
    return module


def _oracle_fp64(x, module):
    """Independent Torch decode + FP64 accumulation (excludes bias)."""
    pack_factor = 32 // module.bits
    maxq = (1 << module.bits) - 1
    group_index = module.g_idx.to(torch.int64)
    shifts = torch.arange(pack_factor, device=x.device, dtype=torch.int64) * module.bits
    codes = ((module.qweight.to(torch.int64).unsqueeze(1) >> shifts[None, :, None]) & maxq)
    codes = codes.reshape(module.in_features, module.out_features)
    zeros = ((module.qzeros.to(torch.int64).unsqueeze(-1) >> shifts[None, None, :]) & maxq)
    zeros = zeros.reshape(module.scales.shape[0], module.out_features)
    weights = ((codes - zeros[group_index]).float() * module.scales[group_index].float()).to(x.dtype)
    return x.double() @ weights.double()


def _dense_weight_candidates(module):
    shape = (module.in_features, module.out_features)
    found = []
    for name, value in vars(module).items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        if value.dim() == 2 and tuple(value.shape) in (shape, (shape[1], shape[0])):
            found.append(f"{name}{tuple(value.shape)}")
    for name, value in module.named_buffers():
        if value.is_floating_point() and value.dim() == 2 and tuple(value.shape) in (shape, (shape[1], shape[0])):
            found.append(f"buffer:{name}{tuple(value.shape)}")
    return found


def _assert_gemv_case(dtype, shape, group_size, desc_act, bias, bits):
    in_features, out_features = shape
    torch.manual_seed(812)
    module = _make_module(in_features, out_features, dtype, group_size, desc_act, bias, bits)
    x = torch.randn(1, in_features, device="cuda", dtype=dtype)
    with torch.inference_mode():
        accumulation = _oracle_fp64(x, module)
        visible = module(x)
        raw = gptq_gemv(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=torch.float32, bits=bits,
        )
    torch.cuda.synchronize()
    assert visible.dtype == dtype
    assert raw.shape == (1, out_features)

    # Storage rule: the fused path consumes packed weights only.
    assert _dense_weight_candidates(module) == [], "forward retained a dense weight"
    with torch.inference_mode():
        again = module(x)
    assert again.dtype == dtype
    assert _dense_weight_candidates(module) == [], "repeated forward retained a dense weight"

    drift = (raw.double() - accumulation).abs()
    torch.testing.assert_close(raw, accumulation.float(), rtol=TOL, atol=TOL)
    # FP32 arithmetic bound: report separately from visible rounding.
    assert drift.max().item() <= 1e-3, f"FP32 accumulation drift {drift.max().item():.3e} exceeds 1e-3"

    raw_rounded = raw.to(dtype)
    accumulation_rounded = accumulation.to(dtype)
    expected_visible = raw_rounded + module.bias if module.bias is not None else raw_rounded
    assert torch.equal(visible, expected_visible), "visible output is not round(matmul) + bias"

    differs = raw_rounded != accumulation_rounded
    if differs.any():
        midpoint = (raw_rounded.double() + accumulation_rounded.double()) / 2
        boundary_distance = (accumulation - midpoint).abs()
        accumulation_error = (raw.double() - accumulation).abs()
        unexplained = differs & (boundary_distance > accumulation_error)
        assert not unexplained.any(), (
            f"{int(unexplained.sum())} visible element(s) differ without the FP32 accumulation "
            f"error crossing the rounding boundary; max raw-vs-fp64 drift={drift.max().item():.3e}"
        )


_SHAPES = [
    ((896, 128), 128, False, False, 4),
    ((896, 1280), 128, False, False, 4),
    ((2048, 1024), 128, False, False, 4),
    ((2048, 2048), 128, False, True, 4),
    ((4096, 4096), 256, False, False, 4),
    ((5120, 12288), 128, True, False, 4),
    ((2048, 1024), 128, False, False, 8),
    ((2048, 1024), 128, True, True, 8),
    ((2048, 1024), 128, False, False, 2),
    ((896, 1280), 128, True, False, 2),
]


@requires_ada
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,group_size,desc_act,bias,bits", _SHAPES)
def test_ada_gptq_gemv_matches_independent_torch_oracle(
    dtype, shape, group_size, desc_act, bias, bits,
):
    _assert_gemv_case(dtype, shape, group_size, desc_act, bias, bits)


@requires_ada
@pytest.mark.cuda
def test_ada_gptq_gemv_dispatch_and_fallbacks():
    module = _make_module(2048, 1024, torch.float16)
    x1 = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    assert module._can_use_rocm_gemv(x1) is True
    assert module._can_use_rocm_gemv(torch.randn(2, 2048, device="cuda", dtype=torch.float16)) is False
    assert module._can_use_rocm_gemv(
        torch.randn(1, 2048, device="cuda", dtype=torch.float16, requires_grad=True)
    ) is False
    module.train()
    assert module._can_use_rocm_gemv(x1) is False
    module.eval()

    # 3-bit uses a different layout and must stay on the fallback.
    three = _make_module(2048, 1024, torch.float16, bits=3)
    assert three._can_use_rocm_gemv(x1) is False
    for bits in (2, 8):
        assert _make_module(2048, 1024, torch.float16, bits=bits)._can_use_rocm_gemv(x1) is True
    # Narrow projections are fused too.
    narrow = _make_module(896, 128, torch.float16)
    assert narrow._can_use_rocm_gemv(torch.randn(1, 896, device="cuda", dtype=torch.float16)) is True
