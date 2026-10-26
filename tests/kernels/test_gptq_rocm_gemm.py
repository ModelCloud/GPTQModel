import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.triton_utils.gptq_gemm import (
    MAX_IN_FEATURES,
    SMALL_M_MAX,
    gptq_gemm,
)


def _oracle_fp64(x: torch.Tensor, module: TorchLinear) -> torch.Tensor:
    """Independent oracle: Torch integer decode + FP64 accumulation (excludes bias)."""
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


@pytest.fixture(autouse=True)
def require_gfx950():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("ROCm GPU required")
    if not getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").startswith("gfx950"):
        pytest.skip("gfx950 GPU required")


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
    assert drift.max().item() <= 1e-3, f"FP32 accumulation drift {drift.max().item():.3e} exceeds 1e-3"

    # The visible output must be the module's own rounded matmul plus bias.
    raw_rounded = raw.to(dtype)
    accumulation_rounded = accumulation.to(dtype)
    expected_visible = raw_rounded + module.bias if module.bias is not None else raw_rounded
    assert torch.equal(visible, expected_visible), "visible output is not round(matmul) + bias"

    # Any rounded difference must be explained by the FP32 accumulation crossing the
    # rounding boundary.
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
def test_gfx950_gptq_gemm_gate_band():
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

    # Past the measured crossover the serial K-loop goes latency bound and the
    # split path is faster, so those shapes must fall back.
    wide_k = _make_module(MAX_IN_FEATURES + 512, 4096, torch.float16, 128, False, False)
    assert not wide_k._can_use_rocm_gemm(torch.randn(4, MAX_IN_FEATURES + 512, device="cuda", dtype=torch.float16))
    edge = _make_module(MAX_IN_FEATURES, 4096, torch.float16, 128, False, False)
    assert edge._can_use_rocm_gemm(torch.randn(4, MAX_IN_FEATURES, device="cuda", dtype=torch.float16))
