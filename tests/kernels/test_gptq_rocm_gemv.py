import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.triton_utils.gptq_gemv import gptq_gemv


def _oracle_fp64(x: torch.Tensor, module: TorchLinear) -> torch.Tensor:
    """Independent oracle.

    Decodes the packed codes with Torch integer shifts and accumulates in FP64.
    The dequantized weights are rounded to the activation dtype first, which is
    the arithmetic the reference Torch and Triton dequant paths both specify.
    The returned tensor excludes bias.
    """
    group_index = module.g_idx.to(torch.int64)
    shifts = torch.arange(8, device=x.device, dtype=torch.int64) * 4
    codes = ((module.qweight.to(torch.int64).unsqueeze(1) >> shifts[None, :, None]) & 15)
    codes = codes.reshape(module.in_features, module.out_features)
    zeros = ((module.qzeros.to(torch.int64).unsqueeze(-1) >> shifts[None, None, :]) & 15)
    zeros = zeros.reshape(module.scales.shape[0], module.out_features)
    weights = ((codes - zeros[group_index]).float() * module.scales[group_index].float()).to(x.dtype)
    weights = weights.double()
    output = x.double() @ weights
    return output


def _make_module(in_features: int, out_features: int, dtype: torch.dtype,
                 group_size: int, desc_act: bool, bias: bool) -> TorchLinear:
    module = TorchLinear(
        bits=4, group_size=group_size, sym=True, desc_act=desc_act,
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
@pytest.mark.parametrize("shape,group_size,desc_act,bias", [
    ((896, 1280), 128, False, False),
    ((2048, 1024), 128, False, False),
    ((4096, 4096), 256, False, True),
    ((5120, 12288), 128, True, False),
])
def test_gfx950_gptq_gemv_matches_independent_torch_oracle(
    dtype: torch.dtype, shape: tuple[int, int], group_size: int,
    desc_act: bool, bias: bool,
):
    torch.manual_seed(810)
    in_features, out_features = shape
    module = _make_module(in_features, out_features, dtype, group_size, desc_act, bias)
    x = torch.randn(1, in_features, device="cuda", dtype=dtype)
    with torch.inference_mode():
        accumulation = _oracle_fp64(x, module)
        visible = module(x)
        raw = gptq_gemv(x, module.qweight, module.scales, module.qzeros, module.g_idx, out_dtype=torch.float32)
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


@pytest.mark.cuda
def test_gfx950_gptq_gemv_falls_back_for_multi_token_or_grad():
    module = _make_module(2048, 1024, torch.float16, 128, False, False)
    assert module._can_use_rocm_gemv(torch.randn(1, 2048, device="cuda", dtype=torch.float16))
    assert not module._can_use_rocm_gemv(torch.randn(2, 2048, device="cuda", dtype=torch.float16))
    assert not module._can_use_rocm_gemv(torch.randn(1, 2048, device="cuda", dtype=torch.float16, requires_grad=True))
    module.train()
    assert not module._can_use_rocm_gemv(torch.randn(1, 2048, device="cuda", dtype=torch.float16))
