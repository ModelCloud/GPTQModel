# GPU=-1 for the mocked architecture/gate cases below; the kernel-oracle cases
# carry ``pytest.mark.cuda`` and skip without a matching ROCm device.
import pytest
import torch

from gptqmodel.nn_modules.qlinear import torch as torch_linear_mod
from gptqmodel.nn_modules.qlinear.torch import TorchLinear, _rocm_arch_name
from gptqmodel.nn_modules.triton_utils.gptq_gemm import (
    MAX_IN_FEATURES,
    SMALL_M_MAX,
    gate_limits,
    gptq_gemm,
    launch_config,
)
from gptqmodel.utils import arch
from gptqmodel.utils.arch import gemm_fused_dtypes


# Registered fused families. Each kernel case is skipped unless the running
# device reports that exact architecture, so a gfx950 host still runs the
# original cases and a gfx1201 host (R9700) runs the new ones.
FUSED_ARCHES = ("gfx950", "gfx1201")


def _device_arch() -> str:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return ""
    props = torch.cuda.get_device_properties(0)
    return arch.normalize_arch(getattr(props, "gcnArchName", ""))


def _require_gpu_arch(arch_name: str) -> None:
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("ROCm GPU required")
    detected = _device_arch()
    if detected != arch_name:
        pytest.skip(f"{arch_name} GPU required (found {detected or 'none'})")


@pytest.fixture(autouse=True)
def _reset_arch_cache():
    arch.clear_arch_cache()
    yield
    arch.clear_arch_cache()


# --------------------------------------------------------------------------- #
# CPU-only mocked coverage: launch table and gate dispatch.
# --------------------------------------------------------------------------- #

def _fake_rocm_device(monkeypatch, name):
    # Simulate a ROCm host: ``torch.version.hip`` drives ``arch.rocm_arch`` and
    # the module-level ``IS_ROCM`` snapshot must agree.
    monkeypatch.setattr(torch.version, "hip", "7.16.26385")
    monkeypatch.setattr(torch_linear_mod, "IS_ROCM", True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index=0: type("P", (), {"gcnArchName": name, "warp_size": 32})(),
    )
    return torch.device("cuda:0")


@pytest.mark.parametrize(
    ("arch_name", "expected"),
    [
        ("gfx950", (16, 16, 256, 4, 2, 16, 5120)),
        ("gfx942", None),
        ("gfx1100", None),
        ("gfx1200", (32, 32, 256, 4, 2, 16, 17408)),
        ("gfx1201:sramecc+:xnack-", (32, 32, 256, 4, 2, 16, 17408)),
        ("gfx1250", (32, 32, 256, 4, 2, 16, 17408)),
        ("sm_89", None),
        ("gfx9999", None),
    ],
)
def test_launch_config_table(arch_name, expected):
    cfg = launch_config(arch_name)
    if expected is None:
        # Unregistered architectures must not enable the fused gate.
        assert (cfg.small_m_max, cfg.max_in_features) == (0, 0)
        assert gate_limits(arch_name) == (0, 0)
        return
    actual = (
        cfg.block_m, cfg.block_n, cfg.block_k, cfg.num_warps, cfg.num_stages,
        cfg.small_m_max, cfg.max_in_features,
    )
    assert actual == expected
    assert gate_limits(arch_name) == (expected[5], expected[6])


def test_cdna4_launch_constants_unchanged():
    # The gfx950 tuning is a compatibility contract: do not drift these values.
    cfg = launch_config("gfx950")
    assert (cfg.block_m, cfg.block_n, cfg.block_k) == (16, 16, 256)
    assert (cfg.num_warps, cfg.num_stages) == (4, 2)
    # Module-level symbols imported by this file must keep the gfx950 values.
    assert SMALL_M_MAX == 16
    assert MAX_IN_FEATURES == 5120


def test_rdna4_launch_is_measured_not_a_cdna_placeholder():
    # gfx120* has its own measured tile: the wider (32, 32) CTA wins on wave32
    # and must not silently collapse back to the CDNA (16, 16) values.
    cdna = launch_config("gfx950")
    rdna = launch_config("gfx1201")
    assert rdna != cdna
    assert (rdna.block_m, rdna.block_n) == (32, 32)
    assert rdna.num_warps == 4 and rdna.num_stages == 2
    assert rdna.small_m_max == cdna.small_m_max == 16
    assert rdna.max_in_features == 17408
    for name in ("gfx1200", "gfx1201", "gfx1250"):
        assert launch_config(name) == rdna


def test_gate_limits_reject_non_gfx_arch():
    assert gate_limits("sm_89") == (0, 0)
    assert gate_limits(None) == (0, 0)
    assert gate_limits("") == (0, 0)


def _cpu_module(in_features: int = 256, out_features: int = 128) -> TorchLinear:
    module = TorchLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        pack_dtype=torch.int32, bias=False, register_buffers=True,
    )
    module.optimize = lambda *args, **kwargs: None
    module.qzero_format(format=2)
    module.post_init()
    return module


@pytest.mark.parametrize(
    ("arch_name", "expect_fused", "expect_gate"),
    [
        ("gfx950", True, (16, 5120)),
        ("gfx1201:sramecc+", True, (16, 17408)),
        ("gfx1250", True, (16, 17408)),
        ("gfx942", False, (0, 0)),
        ("gfx1100", False, (0, 0)),
        ("gfx9999", False, (0, 0)),
    ],
)
def test_module_gate_dispatch_follows_arch(monkeypatch, arch_name, expect_fused, expect_gate):
    device = _fake_rocm_device(monkeypatch, arch_name)
    module = _cpu_module()
    x_stub = type("X", (), {"device": device})()

    assert module._rocm_arch_is_fused(x_stub) is expect_fused
    assert module._rocm_gemm_gate(x_stub) == expect_gate


def test_unregistered_arch_keeps_fused_path_disabled(monkeypatch):
    # Regression lock: before RDNA4 support the gate was `gfx950 only`; an
    # unknown gfx target must behave exactly like that (fused off, fallback on).
    device = _fake_rocm_device(monkeypatch, "gfx9999")
    module = _cpu_module()
    x_stub = type("X", (), {"device": device})()

    assert module._rocm_arch_is_fused(x_stub) is False
    assert module._rocm_gemm_gate(x_stub) == (0, 0)
    assert _rocm_arch_name(device) == "gfx9999"


@pytest.mark.parametrize(
    ("arch_name", "allowed"),
    [
        ("gfx950", {"float16", "bfloat16"}),
        ("gfx1200", {"float16"}),
        ("gfx1201:sramecc+", {"float16"}),
        ("gfx1250", {"float16"}),
        ("gfx942", set()),
        ("sm_89", set()),
        ("gfx9999", set()),
    ],
)
def test_gemm_fused_dtype_gate(monkeypatch, arch_name, allowed):
    # RDNA4's fused small-M GEMM is not a uniform BF16 win on gfx1201
    # (0.43x-1.42x, median 0.85x), so only FP16 takes it there. CDNA keeps both.
    monkeypatch.setattr(torch.version, "hip", "7.16.26385")
    monkeypatch.setattr(torch_linear_mod, "IS_ROCM", True)
    device = _fake_rocm_device(monkeypatch, arch_name)
    assert set(gemm_fused_dtypes(arch_name)) == allowed
    module = _cpu_module()
    for dtype in (torch.float16, torch.bfloat16):
        x_stub = type("X", (), {"device": device, "dtype": dtype})()
        expect = str(dtype).removeprefix("torch.") in allowed
        assert module._rocm_gemm_dtype_supported(x_stub) is expect, (arch_name, dtype)


def test_module_arch_name_returns_empty_without_hip(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    _cpu_module()
    assert _rocm_arch_name(torch.device("cuda:0")) == ""


def _dense_weight_candidates(module: TorchLinear) -> list[str]:
    """Names of floating tensors large enough to be a full dense weight."""
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


def test_no_dense_weight_retained_after_init_and_forward_paths():
    # Storage rule (AGENTS.md): the fused kernels must consume packed weights
    # only. ``GPTQ_TORCH_CACHE_WEIGHTS`` stays off, and no fallback dequant
    # result may be installed on the module. Toggling the legacy cache shim
    # must not resurrect a dense copy.
    import warnings

    module = _cpu_module(in_features=512, out_features=256)
    assert module._cache_enabled is False
    assert _dense_weight_candidates(module) == []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        module.enable_weight_cache(True)
        module.enable_lookahead(True)
    assert module._cache_enabled is False
    assert _dense_weight_candidates(module) == []

    # Packed storage is what remains resident.
    assert module.qweight.dtype == torch.int32
    assert module.scales.dtype == torch.float16 or module.scales.dtype == torch.float32
    packed_bytes = sum(
        t.numel() * t.element_size()
        for t in (module.qweight, module.qzeros, module.scales, module.g_idx)
    )
    assert packed_bytes < module.in_features * module.out_features * 2


# --------------------------------------------------------------------------- #
# Kernel-oracle cases (real hardware only).
# --------------------------------------------------------------------------- #

def _oracle_fp64(x: torch.Tensor, module: TorchLinear) -> torch.Tensor:
    """Independent oracle: Torch integer decode + FP64 accumulation (excludes bias)."""
    return x.double() @ _decoded_weights(module).to(x.dtype).double()


def _decoded_weights(module: TorchLinear) -> torch.Tensor:
    """Decode packed weights independently of the module's own dequant code."""
    pack_factor = 32 // module.bits
    maxq = (1 << module.bits) - 1
    group_index = module.g_idx.to(torch.int64)
    shifts = torch.arange(pack_factor, device=module.qweight.device, dtype=torch.int64) * module.bits
    codes = ((module.qweight.to(torch.int64).unsqueeze(1) >> shifts[None, :, None]) & maxq)
    codes = codes.reshape(module.in_features, module.out_features)
    zeros = ((module.qzeros.to(torch.int64).unsqueeze(-1) >> shifts[None, None, :]) & maxq)
    zeros = zeros.reshape(module.scales.shape[0], module.out_features)
    return (codes - zeros[group_index]).float() * module.scales[group_index].float()


def _make_module(in_features: int, out_features: int, dtype: torch.dtype,
                 group_size: int, desc_act: bool, bias: bool, bits: int = 4) -> TorchLinear:
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


def _assert_gemm_case(dtype, shape, group_size, desc_act, bias, bits, m):
    torch.manual_seed(811)
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

    # Arithmetic accuracy: raw FP32 accumulation against the independent FP64 oracle.
    drift = (raw.double() - accumulation).abs()
    torch.testing.assert_close(raw, accumulation.float(), rtol=2e-3, atol=2e-3)

    # The absolute bound is scale-aware. A fixed 1e-3 is only meaningful for
    # outputs of order ~1e2: the kernel sums K=2048 products in its own tree
    # order, while torch's FP32 matmul sums them in another, so both differ from
    # an FP64 accumulation by pure FP32 rounding. Measured on gfx1201
    # (2048x1024, 8-bit, fp16, m=8): kernel 1.254e-3 vs FP64 at output scale
    # 338 => 3.7e-6 relative, against torch FP32 matmul's own 1.1e-6 relative.
    # Keep the tight 1e-3 absolute floor, plus a 1e-5 relative allowance that
    # covers reduction-order noise at large magnitudes.
    max_abs = drift.max().item()
    max_ref = accumulation.abs().max().item()
    relative = max_abs / max_ref if max_ref > 0 else 0.0
    limit = 1e-3 + 1e-5 * max_ref
    assert max_abs <= limit, (
        f"FP32 accumulation drift {max_abs:.3e} exceeds {limit:.3e} "
        f"(reference max |out|={max_ref:.3e}, relative={relative:.3e})"
    )

    # The visible output must be the module's own rounded matmul plus bias.
    # Which matmul that is depends on dispatch: the fused kernel when it is
    # enabled for this (arch, dtype, shape), otherwise the module's dequantize +
    # dtype matmul path.  BF16 on gfx1201 is gated off the fused GEMM (see
    # ``gemm_fused_dtypes``), so that case must match the split path instead.
    fused_dispatch = module._can_use_rocm_gemm(x)
    if fused_dispatch:
        visible_ref = raw.to(dtype)
        path = "fused GEMM"
    else:
        # The dtype/arch gate keeps BF16 on gfx1201 off the fused kernel, so the
        # visible output must be reproducible from an independent decode of the
        # packed weights followed by the same dtype matmul.
        split_weight = _decoded_weights(module).to(dtype)
        visible_ref = (x @ split_weight).to(dtype)
        path = "dequantize + matmul"
    expected_visible = visible_ref + module.bias if module.bias is not None else visible_ref
    assert torch.equal(visible, expected_visible), f"visible output is not round({path}) + bias"

    if fused_dispatch:
        # Any rounded difference must be explained by the FP32 accumulation
        # crossing the rounding boundary.
        raw_rounded = raw.to(dtype)
        accumulation_rounded = accumulation.to(dtype)
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


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,group_size,desc_act,bias,bits", [
    ((896, 128), 128, False, False, 4),
    ((896, 4864), 128, False, False, 4),
    ((2048, 1024), 128, False, False, 4),
    ((4096, 4096), 256, False, True, 4),
    ((5120, 12288), 128, True, False, 4),
    ((2048, 1024), 128, False, False, 8),
    ((2048, 1024), 128, True, True, 8),
    ((2048, 1024), 128, False, False, 2),
    ((896, 4864), 128, True, False, 2),
])
@pytest.mark.parametrize("m", [2, 4, 8, SMALL_M_MAX])
def test_gfx950_gptq_gemm_matches_independent_torch_oracle(
    dtype: torch.dtype, shape: tuple[int, int], group_size: int,
    desc_act: bool, bias: bool, bits: int, m: int,
):
    _require_gpu_arch("gfx950")
    _assert_gemm_case(dtype, shape, group_size, desc_act, bias, bits, m)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,group_size,desc_act,bias,bits", [
    ((896, 128), 128, False, False, 4),
    ((2048, 1024), 128, True, True, 4),
    ((4096, 4096), 256, False, True, 4),
    ((2048, 1024), 128, False, False, 8),
    ((896, 4864), 128, True, False, 2),
])
@pytest.mark.parametrize("m", [2, 4, 8, SMALL_M_MAX])
def test_gfx1201_gptq_gemm_matches_independent_torch_oracle(
    dtype: torch.dtype, shape: tuple[int, int], group_size: int,
    desc_act: bool, bias: bool, bits: int, m: int,
):
    _require_gpu_arch("gfx1201")
    _assert_gemm_case(dtype, shape, group_size, desc_act, bias, bits, m)


@pytest.mark.cuda
def test_gfx950_gptq_gemm_gate_band():
    _require_gpu_arch("gfx950")
    _assert_gemm_gate_band()


@pytest.mark.cuda
def test_gfx1201_gptq_gemm_gate_band():
    _require_gpu_arch("gfx1201")
    _assert_gemm_gate_band()


def _assert_gemm_gate_band():
    module = _make_module(2048, 1024, torch.float16, 128, False, False)

    def x(m):
        return torch.randn(m, 2048, device="cuda", dtype=torch.float16)

    # Single token belongs to the GEMV kernel, not the GEMM.
    assert not module._can_use_rocm_gemm(x(1))
    assert module._can_use_rocm_gemv(x(1))
    for m in (2, 4, 8, SMALL_M_MAX):
        assert module._can_use_rocm_gemm(x(m)), f"M={m} should use the fused GEMM"
        assert not module._can_use_rocm_gemv(x(m))
    # Above the small-M band the weight would be dequantized per BLOCK_M tile.
    for m in (SMALL_M_MAX + 1, 32, 64):
        assert not module._can_use_rocm_gemm(x(m))

    assert not module._can_use_rocm_gemm(x(4).requires_grad_(True))
    module.train()
    assert not module._can_use_rocm_gemm(x(4))

    for bits in (2, 8):
        wide = _make_module(2048, 1024, torch.float16, 128, False, False, bits)
        assert wide._can_use_rocm_fused(x(4))
    three_bit = _make_module(2048, 1024, torch.float16, 128, False, False, 3)
    assert not three_bit._can_use_rocm_fused(x(4))

    # Past the measured crossover the split path wins, so those shapes must
    # fall back. The bound is per architecture: CDNA4 stops at 5120 because its
    # serial K-loop goes latency bound, while RDNA4's wider tile kept winning up
    # to 17408 (see the launch-table comment in gptq_gemm.py).
    fused_max_in_features = gate_limits(_device_arch())[1]
    assert fused_max_in_features > 0
    wide_k = _make_module(fused_max_in_features + 512, 4096, torch.float16, 128, False, False)
    assert not wide_k._can_use_rocm_gemm(
        torch.randn(4, fused_max_in_features + 512, device="cuda", dtype=torch.float16)
    )
    edge = _make_module(fused_max_in_features, 4096, torch.float16, 128, False, False)
    assert edge._can_use_rocm_gemm(torch.randn(4, fused_max_in_features, device="cuda", dtype=torch.float16))

    # The fused small-M GEMM is a measured win only for the dtypes listed by
    # ``gemm_fused_dtypes``. On gfx1201 BF16 measured 0.43x-1.42x of the split
    # path (median 0.85x), so it must stay on dequantize+matmul there while FP16
    # (and CDNA BF16) keep the fused kernel.
    allowed = gemm_fused_dtypes(_device_arch())
    assert "float16" in allowed
    for dtype in (torch.float16, torch.bfloat16):
        probe = _make_module(2048, 1024, dtype, 128, False, False)
        x_m = torch.randn(4, 2048, device="cuda", dtype=dtype)
        assert probe._can_use_rocm_gemm(x_m) is (str(dtype).removeprefix("torch.") in allowed)
        # M=1 GEMV is a different kernel and stays enabled in both dtypes.
        x_1 = torch.randn(1, 2048, device="cuda", dtype=dtype)
        assert probe._can_use_rocm_gemv(x_1) is True
