# GPU=-1 for the mocked launch-constant cases; the kernel-oracle cases carry
# ``pytest.mark.cuda`` and skip without a matching ROCm device.
import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.triton_utils.gptq_gemv import (
    GEMV_BLOCK_N,
    GEMV_MAX_BLOCK_K,
    GEMV_NUM_WARPS,
    block_k_for,
    gptq_gemv,
    launch_config,
)
from gptqmodel.utils import arch


@pytest.fixture(autouse=True)
def _arch_cache():
    arch.clear_arch_cache()
    yield
    arch.clear_arch_cache()


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


# --------------------------------------------------------------------------- #
# CPU-only mocked coverage: launch constants are module-level and arch-tabled.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("arch_name", "expected"),
    [
        ("gfx950", (512, 8, 8)),
        ("gfx1200", (512, 16, 4)),
        ("gfx1201:sramecc+:xnack-", (512, 16, 4)),
        ("gfx1250", (512, 16, 4)),
        ("gfx942", (512, 8, 8)),
        ("sm_89", (512, 8, 8)),
    ],
)
def test_gemv_launch_constants_are_module_level(arch_name, expected):
    # The former literals inside gptq_gemv() must exist as module constants so a
    # future autotune sweep can change them in one place. The module-level
    # symbols are the non-ROCm/CDNA defaults; the per-arch table supplies the
    # rest. Both must agree for gfx950, which is the compatibility contract.
    assert (GEMV_MAX_BLOCK_K, GEMV_BLOCK_N, GEMV_NUM_WARPS) == (512, 8, 8)
    cfg = launch_config(arch_name)
    assert (cfg.max_block_k, cfg.block_n, cfg.num_warps) == expected
    if arch_name.startswith("gfx950"):
        assert (GEMV_MAX_BLOCK_K, GEMV_BLOCK_N, GEMV_NUM_WARPS) == expected


def test_gemv_block_k_keeps_power_of_two_formula():
    # min(cap, max(32, 1 << (in_features - 1).bit_length()))
    assert block_k_for(896) == 512
    assert block_k_for(5120) == 512
    assert block_k_for(4096) == 512
    assert block_k_for(256) == 256
    assert block_k_for(48) == 64
    assert block_k_for(17) == 32
    assert block_k_for(1) == 32
    assert block_k_for(896, "gfx1201") == 512


def test_gemv_block_k_respects_cap(monkeypatch):
    from gptqmodel.nn_modules.triton_utils import gptq_gemv as gemv_mod

    monkeypatch.setitem(gemv_mod._GEMV_LAUNCH_CONFIGS, "rdna4", gemv_mod.GemvLaunch(256, 8, 8))
    assert block_k_for(4096, "gfx1201") == 256
    assert block_k_for(256, "gfx1201") == 256
    assert block_k_for(128, "gfx1201") == 128


def test_gemv_cdna_values_unchanged():
    cfg = launch_config("gfx950")
    assert cfg == (512, 8, 8)


def _oracle_fp64(x: torch.Tensor, module: TorchLinear) -> torch.Tensor:
    """Independent oracle.

    Decodes the packed codes with Torch integer shifts and accumulates in FP64.
    The dequantized weights are rounded to the activation dtype first, which is
    the arithmetic the reference Torch and Triton dequant paths both specify.
    The returned tensor excludes bias.
    """
    pack_factor = 32 // module.bits
    maxq = (1 << module.bits) - 1
    group_index = module.g_idx.to(torch.int64)
    shifts = torch.arange(pack_factor, device=x.device, dtype=torch.int64) * module.bits
    codes = ((module.qweight.to(torch.int64).unsqueeze(1) >> shifts[None, :, None]) & maxq)
    codes = codes.reshape(module.in_features, module.out_features)
    zeros = ((module.qzeros.to(torch.int64).unsqueeze(-1) >> shifts[None, None, :]) & maxq)
    zeros = zeros.reshape(module.scales.shape[0], module.out_features)
    weights = ((codes - zeros[group_index]).float() * module.scales[group_index].float()).to(x.dtype)
    weights = weights.double()
    output = x.double() @ weights
    return output


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


def _assert_gemv_case(dtype, shape, group_size, desc_act, bias, bits):
    torch.manual_seed(810)
    in_features, out_features = shape
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

    # Arithmetic accuracy: raw FP32 accumulation against the independent FP64 oracle.
    drift = (raw.double() - accumulation).abs()
    torch.testing.assert_close(raw, accumulation.float(), rtol=2e-3, atol=2e-3)
    assert drift.max().item() <= 1e-3, f"FP32 accumulation drift {drift.max().item():.3e} exceeds 1e-3"

    # The visible output must be the module's own rounded matmul plus bias. The bias add
    # rounds in the activation dtype, so this is checked structurally rather than by a
    # blanket relative tolerance (one bf16 ULP is already 0.5% near 1.5).
    raw_rounded = raw.to(dtype)
    accumulation_rounded = accumulation.to(dtype)
    expected_visible = raw_rounded + module.bias if module.bias is not None else raw_rounded
    assert torch.equal(visible, expected_visible), "visible output is not round(matmul) + bias"

    # Visible accuracy: each remaining difference between the kernel's rounded result and
    # the FP64 oracle's rounded result is adjudicated. It is accepted only when the FP64
    # and FP32 accumulations straddle the rounding boundary, i.e. the FP32 accumulation
    # error is at least the distance from that boundary.
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
    ((4096, 4096), 256, False, True, 4),
    ((5120, 12288), 128, True, False, 4),
    ((2048, 1024), 128, False, False, 8),
    ((2048, 1024), 128, True, True, 8),
    ((2048, 1024), 128, False, False, 2),
    ((896, 1280), 128, True, False, 2),
]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,group_size,desc_act,bias,bits", _SHAPES)
def test_gfx950_gptq_gemv_matches_independent_torch_oracle(
    dtype: torch.dtype, shape: tuple[int, int], group_size: int,
    desc_act: bool, bias: bool, bits: int,
):
    _require_gpu_arch("gfx950")
    _assert_gemv_case(dtype, shape, group_size, desc_act, bias, bits)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,group_size,desc_act,bias,bits", _SHAPES)
def test_gfx1201_gptq_gemv_matches_independent_torch_oracle(
    dtype: torch.dtype, shape: tuple[int, int], group_size: int,
    desc_act: bool, bias: bool, bits: int,
):
    _require_gpu_arch("gfx1201")
    _assert_gemv_case(dtype, shape, group_size, desc_act, bias, bits)


@pytest.mark.cuda
def test_gfx950_gptq_gemv_falls_back_for_multi_token_or_grad():
    _require_gpu_arch("gfx950")
    _assert_gemv_fallbacks()


@pytest.mark.cuda
def test_gfx1201_gptq_gemv_falls_back_for_multi_token_or_grad():
    _require_gpu_arch("gfx1201")
    _assert_gemv_fallbacks()


def _assert_gemv_fallbacks():
    module = _make_module(2048, 1024, torch.float16, 128, False, False)
    assert module._can_use_rocm_gemv(torch.randn(1, 2048, device="cuda", dtype=torch.float16))
    assert not module._can_use_rocm_gemv(torch.randn(2, 2048, device="cuda", dtype=torch.float16))
    assert not module._can_use_rocm_gemv(torch.randn(1, 2048, device="cuda", dtype=torch.float16, requires_grad=True))
    module.train()
    assert not module._can_use_rocm_gemv(torch.randn(1, 2048, device="cuda", dtype=torch.float16))

    # Narrow projections (e.g. attention k/v) are still faster fused.
    narrow = _make_module(896, 128, torch.float16, 128, False, False)
    assert narrow._can_use_rocm_gemv(torch.randn(1, 896, device="cuda", dtype=torch.float16))

    # 2/8-bit packed weights are fused too; 3-bit uses a different layout.
    for bits in (2, 8):
        wide = _make_module(2048, 1024, torch.float16, 128, False, False, bits)
        assert wide._can_use_rocm_gemv(torch.randn(1, 2048, device="cuda", dtype=torch.float16))
    three_bit = _make_module(2048, 1024, torch.float16, 128, False, False, 3)
    assert not three_bit._can_use_rocm_gemv(torch.randn(1, 2048, device="cuda", dtype=torch.float16))
