# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for the mocked gate/dispatch cases; the kernel-oracle cases carry
# ``pytest.mark.cuda`` and skip unless the device is an Ada (SM 8.9) GPU.
"""Ada (RTX 4090, SM 8.9) fused small-M GPTQ GEMM: gates + Torch-oracle A/B.

The oracle decodes the packed int32 words with Torch integer shifts and
accumulates in FP64; it never calls the Triton kernel.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import torch as torch_linear_mod
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.triton_utils.gptq_gemm import (
    _DEFAULT_LAUNCH,
    _LAUNCH_CONFIGS,
    GemmLaunch,
    gate_limits,
    gate_max_out_features,
    gptq_gemm,
    launch_config,
    split_k_for,
    tile_for_m,
)
from gptqmodel.utils import arch
from tests.kernels._gptq_gemm_pre_rewrite import pre_rewrite_gptq_gemm


TOL = 2e-3
ADA_GATE = (64, 0)
BLACKWELL_GATE = (64, 0)


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
# CPU-only: launch table, gate constants and dispatch predicates.
# --------------------------------------------------------------------------- #

def test_ada_launch_entry_is_measured_tile():
    cfg = launch_config("sm_89")
    assert cfg == launch_config((8, 9))
    assert (cfg.block_m, cfg.block_n, cfg.block_k) == (16, 32, 128)
    assert (cfg.num_warps, cfg.num_stages) == (4, 2)
    assert cfg.small_m_max == 64
    assert cfg.max_in_features == 0
    assert cfg.max_out_features == 0
    assert cfg.m_banded is True
    assert cfg.auto_split is True
    assert cfg == _LAUNCH_CONFIGS["ada"]
    assert launch_config("sm_100") == _LAUNCH_CONFIGS["blackwell"]
    assert launch_config("sm_120") == _LAUNCH_CONFIGS["blackwell"]


@pytest.mark.parametrize(
    ("arch_id", "expected"),
    [
        ("sm_89", GemmLaunch(16, 32, 128, 4, 2, 64, 0, 0, True, True)),
        ((8, 9), GemmLaunch(16, 32, 128, 4, 2, 64, 0, 0, True, True)),
        ("sm_100", GemmLaunch(16, 32, 128, 4, 2, 64, 0, 0, True, True)),
        ("sm_120", GemmLaunch(16, 32, 128, 4, 2, 64, 0, 0, True, True)),
        ("gfx950", GemmLaunch(16, 16, 256, 4, 2, 16, 5120, 0)),
        ("gfx1201", GemmLaunch(32, 32, 256, 4, 2, 16, 17408, 0)),
        ("gfx942", _DEFAULT_LAUNCH),
        ("sm_80", _DEFAULT_LAUNCH),
        ("sm_90", _DEFAULT_LAUNCH),
        ("", _DEFAULT_LAUNCH),
        (None, _DEFAULT_LAUNCH),
    ],
)
def test_gemm_launch_routes_every_family(arch_id, expected):
    assert launch_config(arch_id) == expected


def test_ada_gate_and_dtypes():
    assert gate_limits("sm_89") == ADA_GATE
    assert gate_limits((8, 9)) == ADA_GATE
    assert gate_max_out_features("sm_89") == 0
    assert set(arch.gemm_fused_dtypes("sm_89")) == {"float16", "bfloat16"}
    assert arch.is_cuda_fused("sm_89") is True
    assert gate_limits("sm_100") == BLACKWELL_GATE
    assert gate_limits("sm_120") == BLACKWELL_GATE
    assert gate_max_out_features("sm_100") == 0
    assert set(arch.gemm_fused_dtypes("sm_100")) == {"float16", "bfloat16"}
    assert arch.is_cuda_fused("sm_100") is True
    assert arch.is_cuda_fused("sm_120") is True
    # Unchanged for the AMD families (the output cap stays disabled).
    assert gate_limits("gfx950") == (16, 5120)
    assert gate_max_out_features("gfx950") == 0
    assert gate_limits("gfx1201") == (16, 17408)
    assert gate_max_out_features("gfx1201") == 0
    # Unregistered targets still keep the fused path off.
    for arch_id in ("sm_80", "sm_90", "gfx942", ""):
        assert gate_limits(arch_id) == (0, 0)
        assert gate_max_out_features(arch_id) == 0
    # ``None`` is "the current device" in ``utils.arch`` (same convention as
    # ``is_ada``/``tuning_family``/``cuda_capability``), so it tracks the host.
    host_family = arch.tuning_family(None)
    # Registered non-Ada hosts (e.g. an RDNA4 GPU running the ROCm suite) still
    # resolve their own gate; only truly unregistered families stay disabled.
    _KNOWN_HOST_GATES = {
        "ada": ADA_GATE,
        "blackwell": BLACKWELL_GATE,
        "cdna4": (16, 5120),
        "rdna4": (16, 17408),
    }
    assert gate_max_out_features(None) == 0
    assert arch.gate_limits(None) == _KNOWN_HOST_GATES.get(host_family, (0, 0))
    # The kernel-module helpers keep ``None`` = "unknown caller" -> disabled,
    # and only ``gemm_fused_max_out_features`` follows the shared device default.
    assert gate_limits(None) == (0, 0)


def _cpu_module(in_features: int = 2048, out_features: int = 1024) -> TorchLinear:
    module = TorchLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        pack_dtype=torch.int32, bias=False, register_buffers=True,
    )
    module.optimize = lambda *args, **kwargs: None
    module.qzero_format(format=2)
    module.post_init()
    return module


def _fake_cuda_ada(monkeypatch, capability=(8, 9)):
    # Simulate an Ada CUDA host without touching the real device: HIP is off and
    # capability reads report the requested SM.
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch_linear_mod, "IS_ROCM", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    return torch.device("cuda:0")


@pytest.mark.parametrize(
    ("capability", "expect_fused", "expect_gate", "expect_out_cap", "expect_dtypes"),
    [
        ((8, 9), True, ADA_GATE, 0, {"float16", "bfloat16"}),
        ((8, 0), False, (0, 0), 0, set()),
        ((8, 6), False, (0, 0), 0, set()),
        ((9, 0), False, (0, 0), 0, set()),
        ((10, 0), True, BLACKWELL_GATE, 0, {"float16", "bfloat16"}),
        ((12, 0), True, BLACKWELL_GATE, 0, {"float16", "bfloat16"}),
    ],
)
def test_module_dispatch_follows_capability(
    monkeypatch, capability, expect_fused, expect_gate, expect_out_cap, expect_dtypes,
):
    device = _fake_cuda_ada(monkeypatch, capability)
    module = _cpu_module()
    x_stub = type("X", (), {"device": device})()

    assert module._rocm_arch_is_fused(x_stub) is expect_fused
    assert module._rocm_gemm_gate(x_stub) == expect_gate
    assert torch_linear_mod._fused_arch_key(device) == (
        "sm_" + str(capability[0]) + str(capability[1])
    )
    for dtype in (torch.float16, torch.bfloat16):
        x_dtype = type("X", (), {"device": device, "dtype": dtype})()
        assert module._rocm_gemm_dtype_supported(x_dtype) is (
            str(dtype).removeprefix("torch.") in expect_dtypes
        )
    # The output cap must follow the same simulated capability as the gate.
    assert arch.gemm_fused_max_out_features(capability) == expect_out_cap
    if expect_out_cap:
        assert module._rocm_gemm_out_features_supported(
            type("X", (), {"device": device})()
        ) is True


def test_ada_has_no_output_width_cap(monkeypatch):
    device = _fake_cuda_ada(monkeypatch)
    narrow = _cpu_module(out_features=2048)
    wide = _cpu_module(out_features=4096)
    x_stub = type("X", (), {"device": device})()
    assert narrow._rocm_gemm_out_features_supported(x_stub) is True
    assert wide._rocm_gemm_out_features_supported(x_stub) is True


def test_ada_has_no_in_features_cap(monkeypatch):
    device = _fake_cuda_ada(monkeypatch)
    x_stub = type("X", (), {"device": device})()
    at_cap = _cpu_module(in_features=2048, out_features=1024)
    over = _cpu_module(in_features=2560, out_features=1024)
    assert at_cap._rocm_gemm_gate(x_stub)[1] == 0
    assert over._rocm_gemm_gate(x_stub)[1] == 0
    for module in (at_cap, over):
        # The shape gate accepts both widths; the real dtype/device gates are
        # exercised by the kernel-oracle cases.
        assert module._rocm_gemm_gate(x_stub)[0] == 64


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


def _decoded_weights(module):
    pack_factor = 32 // module.bits
    maxq = (1 << module.bits) - 1
    group_index = module.g_idx.to(torch.int64)
    shifts = torch.arange(pack_factor, device=module.qweight.device, dtype=torch.int64) * module.bits
    codes = ((module.qweight.to(torch.int64).unsqueeze(1) >> shifts[None, :, None]) & maxq)
    codes = codes.reshape(module.in_features, module.out_features)
    zeros = ((module.qzeros.to(torch.int64).unsqueeze(-1) >> shifts[None, None, :]) & maxq)
    zeros = zeros.reshape(module.scales.shape[0], module.out_features)
    return (codes - zeros[group_index]).float() * module.scales[group_index].float()


def _oracle_fp64(x, module):
    return x.double() @ _decoded_weights(module).to(x.dtype).double()


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


def _assert_gemm_case(dtype, shape, group_size, desc_act, bias, bits, m):
    torch.manual_seed(813)
    in_features, out_features = shape
    module = _make_module(in_features, out_features, dtype, group_size, desc_act, bias, bits)
    x = torch.randn(m, in_features, device="cuda", dtype=dtype)
    with torch.inference_mode():
        accumulation = _oracle_fp64(x, module)
        visible = module(x)
        raw = gptq_gemm(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=torch.float32, bits=bits,
        )
    torch.cuda.synchronize()
    assert visible.dtype == dtype
    assert raw.shape == (m, out_features)
    assert _dense_weight_candidates(module) == [], "forward retained a dense weight"
    with torch.inference_mode():
        again = module(x)
    assert again.dtype == dtype
    assert _dense_weight_candidates(module) == [], "repeated forward retained a dense weight"

    drift = (raw.double() - accumulation).abs()
    torch.testing.assert_close(raw, accumulation.float(), rtol=TOL, atol=TOL)
    max_abs = drift.max().item()
    max_ref = accumulation.abs().max().item()
    limit = 1e-3 + 1e-5 * max_ref
    assert max_abs <= limit, (
        f"FP32 accumulation drift {max_abs:.3e} exceeds {limit:.3e} "
        f"(reference max |out|={max_ref:.3e})"
    )

    fused_dispatch = module._can_use_rocm_gemm(x)
    bias = module.bias
    if fused_dispatch:
        # The fused kernel keeps the bias in the FP32 epilogue before the
        # output is narrowed, so the oracle must add it before rounding too.
        wide_kernel = raw if bias is None else raw + bias.float()
        path = "fused GEMM"
    else:
        split_weight = _decoded_weights(module).to(dtype)
        # The CUDA split path keeps low-precision partial sums in FP32
        # (``matmul_fp32_reduce``) and fuses the bias into that FP32 epilogue
        # before rounding once, so the oracle must use the same ``addmm``
        # expression instead of a narrow ``matmul`` followed by a second add.
        flat_x = x.reshape(-1, x.shape[-1])
        if bias is None:
            wide_kernel = torch.mm(flat_x, split_weight, out_dtype=torch.float32)
        else:
            wide_kernel = torch.addmm(
                bias.float(), flat_x, split_weight, out_dtype=torch.float32
            )
        wide_kernel = wide_kernel.reshape(x.shape[:-1] + (split_weight.shape[-1],))
        path = "dequantize + matmul"
    expected_visible = wide_kernel.to(dtype)
    assert torch.equal(visible, expected_visible), f"visible output does not match the {path} contract"

    # Adjudicate against the FP64 oracle with the same fused-bias epilogue.
    wide_reference = accumulation if bias is None else accumulation + bias.double()
    reference_rounded = wide_reference.to(dtype)
    differs = visible != reference_rounded
    if differs.any():
        midpoint = (visible.double() + reference_rounded.double()) / 2
        boundary_distance = (wide_reference - midpoint).abs()
        accumulation_error = (wide_kernel.double() - wide_reference).abs()
        unexplained = differs & (boundary_distance > accumulation_error)
        assert not unexplained.any(), (
            f"{int(unexplained.sum())} visible element(s) differ without the FP32 accumulation "
            f"error crossing the rounding boundary; max raw-vs-fp64 drift={drift.max().item():.3e}"
        )


# Shapes covering the widened gate: small and 7B-style projections, K/N tails,
# 2/4/8-bit, gs 128/256 and act-order.
_SHAPES = [
    ((896, 128), 128, False, False, 4),
    ((896, 1280), 128, False, False, 4),
    ((2048, 1024), 128, False, False, 4),
    ((2048, 2048), 128, False, True, 4),
    ((2048, 1024), 128, True, True, 8),
    ((2048, 1024), 128, False, False, 2),
    ((2048, 4096), 128, False, False, 4),
    ((4096, 4096), 256, False, True, 4),
    ((5120, 12288), 128, True, False, 4),
]


@requires_ada
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,group_size,desc_act,bias,bits", _SHAPES)
@pytest.mark.parametrize("m", [2, 16, 32, 64])
def test_ada_gptq_gemm_matches_independent_torch_oracle(
    dtype, shape, group_size, desc_act, bias, bits, m,
):
    _assert_gemm_case(dtype, shape, group_size, desc_act, bias, bits, m)


@requires_ada
@pytest.mark.cuda
def test_ada_gptq_gemm_gate_band():
    module = _make_module(2048, 1024, torch.float16)

    def x(m):
        return torch.randn(m, 2048, device="cuda", dtype=torch.float16)

    assert not module._can_use_rocm_gemm(x(1))
    assert module._can_use_rocm_gemv(x(1))
    for m in (2, 4, 8, 16, 17, 32, 33, 64):
        assert module._can_use_rocm_gemm(x(m)), f"M={m} should use the fused GEMM"
    assert not module._can_use_rocm_gemm(x(65))

    # The packed-load rewrite removed the old K/N caps: both wide-N and long-K
    # projections are inside the gate now.
    wide = _make_module(2048, 4096, torch.float16)
    assert wide._can_use_rocm_gemm(x(8))
    long_k = _make_module(4096, 1024, torch.float16)
    assert long_k._can_use_rocm_gemm(
        torch.randn(8, 4096, device="cuda", dtype=torch.float16)
    )
    # BF16 is a measured in-gate win, so both dtypes take the fused path.
    for dtype in (torch.float16, torch.bfloat16):
        probe = _make_module(2048, 1024, dtype)
        x_m = torch.randn(8, 2048, device="cuda", dtype=dtype)
        assert probe._can_use_rocm_gemm(x_m) is True
        x_1 = torch.randn(1, 2048, device="cuda", dtype=dtype)
        assert probe._can_use_rocm_gemv(x_1) is True


def test_ada_dtypes_are_the_measured_set():
    """Only dtypes actually measured on the 4090 may enable the fused GEMM.

    Both were measured on the 4090.  Blackwell ships the same kernel and is
    enabled from the issue's indicative sm_120 report; this host has no
    Blackwell GPU to re-measure it.
    """

    assert arch.gemm_fused_dtypes("sm_89") == frozenset({"float16", "bfloat16"})
    assert arch.gemm_fused_dtypes((8, 9)) == frozenset({"float16", "bfloat16"})
    assert arch.gate_limits("sm_89") == (64, 0)
    assert arch.gemm_fused_max_out_features("sm_89") == 0
    assert arch.gemm_fused_dtypes("sm_100") == frozenset({"float16", "bfloat16"})
    assert arch.gemm_fused_dtypes("sm_120") == frozenset({"float16", "bfloat16"})
    assert arch.gate_limits("sm_100") == (64, 0)
    assert arch.gate_limits("sm_120") == (64, 0)
    # Unregistered CUDA families keep the fused path fully disabled.
    for arch_id in ("sm_80", "sm_86", "sm_90"):
        assert arch.gemm_fused_dtypes(arch_id) == frozenset()
        assert arch.gate_limits(arch_id) == (0, 0)


@requires_ada
@pytest.mark.cuda
def test_ada_forward_retains_no_dense_weight_at_scale():
    """Storage rule at benchmark scale, after one and repeated forwards."""

    for dtype in (torch.float16, torch.bfloat16):
        module = _make_module(4096, 4096, dtype, bits=4)
        assert _dense_weight_candidates(module) == []
        x = torch.randn(8, 4096, device="cuda", dtype=dtype)
        with torch.inference_mode():
            first = module(x)
        torch.cuda.synchronize()
        assert _dense_weight_candidates(module) == [], "single forward retained a dense weight"
        with torch.inference_mode():
            for _ in range(3):
                again = module(x)
        torch.cuda.synchronize()
        assert _dense_weight_candidates(module) == [], "repeated forwards retained a dense weight"
        assert torch.equal(first, again)
        assert first.dtype == dtype


# --------------------------------------------------------------------------- #
# Bitwise / split-K contracts.
# --------------------------------------------------------------------------- #

def test_ada_tile_for_m_bands():
    cfg = launch_config("sm_89")
    assert tile_for_m(cfg, 1) == (16, 32)
    assert tile_for_m(cfg, 16) == (16, 32)
    assert tile_for_m(cfg, 17) == (32, 64)
    assert tile_for_m(cfg, 32) == (32, 64)
    assert tile_for_m(cfg, 33) == (64, 64)
    assert tile_for_m(cfg, 64) == (64, 64)
    # CDNA4/RDNA4 keep one fixed tile.
    for arch_id in ("gfx950", "gfx1201"):
        fixed = launch_config(arch_id)
        assert tile_for_m(fixed, 2) == (fixed.block_m, fixed.block_n)
        assert tile_for_m(fixed, 64) == (fixed.block_m, fixed.block_n)


def test_ada_split_k_selection():
    assert split_k_for(2, 4096) == 8
    assert split_k_for(16, 14336) == 8
    assert split_k_for(64, 4096) == 4
    assert split_k_for(64, 14336) == 4
    # K/S must stay >= 512 and the split must divide the K tiles.
    assert split_k_for(2, 1024) == 2
    assert split_k_for(2, 512) == 1
    assert split_k_for(2, 4090) == 1


@requires_ada
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("m", [2, 64])
def test_ada_split_off_is_bitwise_equal_to_pre_rewrite_kernel(
    dtype, desc_act, bits, m,
):
    torch.manual_seed(3982)
    module = _make_module(2048, 1024, dtype, group_size=128,
                          desc_act=desc_act, bits=bits)
    x = torch.randn(m, 2048, device="cuda", dtype=dtype)

    with torch.inference_mode():
        old_fp32 = pre_rewrite_gptq_gemm(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=torch.float32, bits=bits,
        )
        new_fp32 = gptq_gemm(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=torch.float32, bits=bits,
            uniform_group=module._gptq_gemm_uniform_group, split_k=1,
        )
        old_visible = pre_rewrite_gptq_gemm(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=None, bits=bits,
        )
        new_visible = gptq_gemm(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=None, bits=bits,
            uniform_group=module._gptq_gemm_uniform_group, split_k=1,
        )

    assert torch.equal(old_fp32, new_fp32), "split-off FP32 output is not bitwise equal"
    assert torch.equal(old_visible, new_visible), "split-off visible output is not bitwise equal"
    assert new_visible.dtype == dtype


@requires_ada
@pytest.mark.cuda
def test_ada_auto_split_k_is_deterministic():
    module = _make_module(4096, 1024, torch.float16, group_size=128)
    x = torch.randn(16, 4096, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        first = gptq_gemm(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=torch.float32, bits=4,
            uniform_group=module._gptq_gemm_uniform_group,
        )
        second = gptq_gemm(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=torch.float32, bits=4,
            uniform_group=module._gptq_gemm_uniform_group,
        )
    assert torch.equal(first, second)


@requires_ada
@pytest.mark.cuda
@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("m", [16, 64])
def test_ada_auto_split_k_is_no_worse_than_single_pass(desc_act, m):
    """Split-K must stay inside the FP32 drift bound and not increase it."""
    torch.manual_seed(3982)
    module = _make_module(14336, 1024, torch.float16, group_size=128,
                          desc_act=desc_act)
    x = torch.randn(m, 14336, device="cuda", dtype=torch.float16)
    reference = _oracle_fp64(x, module)

    with torch.inference_mode():
        single = gptq_gemm(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=torch.float32, bits=4,
            uniform_group=module._gptq_gemm_uniform_group, split_k=1,
        )
        split = gptq_gemm(
            x,
            module.qweight, module.scales,
            module.qzeros, module.g_idx, out_dtype=torch.float32, bits=4,
            uniform_group=module._gptq_gemm_uniform_group, split_k=None,
        )

    single_drift = (single.double() - reference).abs().max().item()
    split_drift = (split.double() - reference).abs().max().item()
    limit = 1e-3 + 1e-5 * reference.abs().max().item()
    assert split_drift <= limit, f"split-K drift {split_drift:.3e} exceeds {limit:.3e}"
    assert split_drift <= single_drift + 1e-6, (
        f"split-K drift {split_drift:.3e} is worse than single-pass {single_drift:.3e}"
    )
    assert split_k_for(m, 14336) == (8 if m <= 32 else 4)


@requires_ada
@pytest.mark.cuda
@pytest.mark.parametrize("m", [2, 64])
@pytest.mark.parametrize(
    "shape",
    [(14336, 4096), (4096, 14336), (4096, 11008)],
)
def test_ada_large_projection_matches_independent_oracle(shape, m):
    _assert_gemm_case(torch.float16, shape, 128, False, False, 4, m)
