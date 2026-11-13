# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CUDA/ROCm quantized linear output rounding and wide-GEMM gradients."""

import weakref

import pytest
import torch
from test_torch_weight_storage import state_tensors

from gptqmodel.nn_modules.qlinear._numerics import (
    _can_fuse_hip_bias,
    linear_fp32_reduce,
    matmul_fp32_reduce,
)
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.quantization import FORMAT


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for FP32 bias epilogue",
)


def _gptq(dtype, streaming, *, fused=False):
    k = 32
    n = 192 if streaming else 32
    linear = torch.nn.Linear(k, n, bias=True)
    linear.weight.data.fill_(41.25)
    linear.bias.data.fill_(-1000)
    module = TorchLinear(
        bits=4, group_size=32, sym=False, desc_act=False,
        in_features=k, out_features=n, bias=True, format=FORMAT.GPTQ_V2,
    )
    module.pack_block(
        linear, torch.full((n, 1), 2.75), torch.zeros((n, 1)),
        torch.zeros(k, dtype=torch.int32),
    )
    if fused:
        # The packed zero points are literally zero, which is already the v2
        # encoding of zero point 0, so flipping the flag opens the fused-kernel
        # gate without re-encoding any packed word. Assert the dequantized
        # weight still matches the fixture's logical weight.
        module.qzero_format(2)
        assert torch.equal(module.dequantize_weight(), torch.full((n, k), 41.25))
    module._init_wf_unsqueeze_buffers()
    module = module.eval().to(device="cuda", dtype=dtype)
    if streaming:
        module.enable_streaming(True, tile_cols=64)
    return module, 41.25, 50, -1000


def _awq(dtype):
    k = n = 32
    module = AwqTorchLinear(
        bits=4, group_size=16, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=True, dtype=dtype,
        register_buffers=True,
    )
    # Each packed nibble is 15; the zero point is 0, giving 15 * 2.75.
    module.qweight.fill_(-1)
    module.qzeros.zero_()
    module.scales.fill_(2.75)
    module.bias.fill_(-1000)
    module.post_init()
    return module.eval().to(device="cuda", dtype=dtype), 41.25, 50, -1000


def _tritonv2(dtype):
    pytest.importorskip("triton")
    k = n = 32
    module = TritonV2Linear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=True,
        format=FORMAT.GPTQ_V2, register_buffers=True,
    )
    # Each packed nibble is 15; the zero point is 0, giving 15 * 2.75.
    module.qweight.fill_(-1)
    module.qzeros.zero_()
    module.scales.fill_(2.75)
    module.bias.fill_(-1000)
    module.post_init()
    return module.eval().to(device="cuda", dtype=dtype), 41.25, 50, -1000


def _fp8(dtype, scaled):
    k = n = 32
    module = TorchFP8Linear(
        bits=8, group_size=-1, desc_act=False, sym=True,
        in_features=k, out_features=n, bias=True, format="float8_e4m3fn",
        weight_scale_method="tensor" if scaled else "row",
    )
    linear = torch.nn.Linear(k, n, bias=True)
    linear.weight.data.fill_(41.25)
    linear.bias.data.fill_(-8000)
    module.pack_original(linear, scales=None, zeros=None)
    module = module.eval().to(device="cuda")
    module._scaled_mm_hard_disabled = not scaled
    return module, 41.25, 52, -8000


def _gguf(dtype):
    k = n = 32
    module = GGUFTorchLinear(
        bits="q4_0", group_size=-1, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=True, register_buffers=True,
    )
    linear = torch.nn.Linear(k, n, bias=True)
    linear.weight.data.fill_(38.5)
    linear.bias.data.fill_(-5000)
    module.pack_original(linear, scales=None, zeros=None)
    return module.eval().to(device="cuda", dtype=dtype), 38.5, 55, -5000


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_fp32_reduce_fuses_bias_before_rounding(dtype):
    x = torch.full((129, 128), 516, device="cuda", dtype=dtype)
    weight = torch.ones((128, 128), device="cuda", dtype=dtype)
    bias = torch.full((128,), -1024, device="cuda", dtype=dtype)
    expected = (x.double() @ weight.double() + bias.double()).to(dtype)

    actual = matmul_fp32_reduce(x, weight, bias)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_matmul_fp32_reduce_keeps_bias_range_under_autocast():
    """Autocast must keep the bias addition outside the narrowed epilogue.

    BF16 is the case that regresses: autocast narrows the GEMM to FP16, so a
    BF16 bias inside the fused ``addmm`` is cast to FP16 and overflows, while
    the standalone addition kept it through type promotion.
    """
    x = torch.full((129, 32), 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    bias = torch.full((32,), -65536, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda"):
        # The expression the helper replaced: the matmul narrows, the
        # standalone addition keeps the bias range through type promotion.
        expected = torch.matmul(x, weight) + bias
        actual = matmul_fp32_reduce(x, weight, bias)

    assert torch.isfinite(actual).all()
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    oracle = (x.double() @ weight.double() + bias.double()).to(actual.dtype)
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


def test_matmul_fp32_reduce_matches_broadcast_bias_expression():
    """A bias that needs broadcasting keeps the unfused expression."""
    generator = torch.Generator().manual_seed(7)
    x = torch.randint(-8, 9, (2, 3, 32), generator=generator).to(device="cuda", dtype=torch.float16) / 16
    weight = torch.randint(-8, 9, (32, 48), generator=generator).to(device="cuda", dtype=torch.float16) / 16
    bias = torch.randint(-8, 9, (2, 1, 48), generator=generator).to(device="cuda", dtype=torch.float16) / 16
    expected = torch.matmul(x, weight) + bias

    actual = matmul_fp32_reduce(x, weight, bias)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hip_bias_fusion_scope():
    """Only ROCm low-precision calls without autocast take the fused epilogue."""
    from gptqmodel.utils.rocm import IS_ROCM

    x16 = torch.zeros((4, 32), device="cuda", dtype=torch.float16)
    weight16 = torch.zeros((32, 48), device="cuda", dtype=torch.float16)
    bias16 = torch.zeros(48, device="cuda", dtype=torch.float16)

    # FP32 never rounds before the add, on any platform.
    assert _can_fuse_hip_bias(x16.float(), weight16.float(), bias16.float()) is False
    assert _can_fuse_hip_bias(x16, weight16, bias16) is IS_ROCM
    assert _can_fuse_hip_bias(x16, weight16, None) is False
    with torch.autocast("cuda"):
        assert _can_fuse_hip_bias(x16, weight16, bias16) is False

    # Broadcasting that the flattened 2D `addmm` input cannot express.
    x3 = torch.zeros((2, 3, 32), device="cuda", dtype=torch.float16)
    bias3 = torch.zeros((2, 1, 48), device="cuda", dtype=torch.float16)
    assert _can_fuse_hip_bias(x3, weight16, bias3) is False
    # ... while a bias that already matches the output shape still fuses.
    assert _can_fuse_hip_bias(x3, weight16, torch.zeros((2, 3, 48), device="cuda", dtype=torch.float16)) is IS_ROCM


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 2, 16])
def test_rocm_fused_gptq_bias_rounds_once(dtype, rows):
    """The fused GPTQ GEMV/GEMM epilogues must not narrow the partial sum first."""
    module, weight, value, bias = _gptq(dtype, streaming=False, fused=True)
    x = torch.full((rows, 32), value, device="cuda", dtype=dtype)
    gemv = rows == 1 and module._can_use_rocm_gemv(x)
    gemm = rows > 1 and module._can_use_rocm_gemm(x)
    if not (gemv or gemm):
        pytest.skip(f"fused GPTQ kernel not eligible for {dtype} M={rows} on this device")
    assert gemv != gemm, "exactly one fused kernel handles this shape"

    expected = torch.full(
        (rows, module.out_features), 32 * value * weight + bias,
        device="cuda", dtype=torch.float64,
    )
    with torch.inference_mode():
        actual = module(x)

    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "backend",
    ["gptq", "gptq_streaming", "awq", "tritonv2", "fp8_dense", "fp8_scaled", "gguf_dense"],
)
def test_cuda_bias_cancellation_rounds_once(backend, dtype):
    if backend == "gptq":
        module, weight, value, bias = _gptq(dtype, streaming=False)
    elif backend == "gptq_streaming":
        module, weight, value, bias = _gptq(dtype, streaming=True)
    elif backend == "awq":
        module, weight, value, bias = _awq(dtype)
    elif backend == "tritonv2":
        module, weight, value, bias = _tritonv2(dtype)
    elif backend == "fp8_dense":
        module, weight, value, bias = _fp8(dtype, scaled=False)
    elif backend == "fp8_scaled":
        module, weight, value, bias = _fp8(dtype, scaled=True)
    else:
        module, weight, value, bias = _gguf(dtype)

    x = torch.full((2, 32), value, device="cuda", dtype=dtype)
    if backend == "fp8_scaled" and not module._can_use_scaled_mm(x):
        pytest.skip("Native FP8 scaled matmul unavailable on this GPU")
    if backend == "gptq_streaming":
        assert module._should_use_streaming(x)
    if backend == "gguf_dense":
        assert not module._should_use_fused_k_forward(x)

    # The independent FP64 oracle uses logical constant weights, not the
    # changed inference matmul or any of its dequantized intermediates.
    expected = torch.full(
        (2, module.out_features), 32 * value * weight + bias,
        device="cuda", dtype=torch.float64,
    )
    assert torch.isfinite(expected.to(dtype)).all()
    before = {tensor.data_ptr() for tensor in state_tensors(module)}
    stream = None
    if backend == "gptq_streaming":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(2):
        if stream is not None:
            with torch.inference_mode(), torch.cuda.stream(stream):
                actual = module(x)
            stream.synchronize()
        else:
            with torch.inference_mode():
                actual = module(x)
        assert actual.dtype == dtype
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
        assert not any(
            tensor.data_ptr() not in before
            and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and tensor.numel() >= 32 * module.out_features
            for tensor in state_tensors(module)
        ), "inference retained an expanded dense weight"
    if backend == "fp8_scaled":
        assert not module._scaled_mm_hard_disabled


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("linear", [False, True])
def test_cuda_wide_matmul_backward_matches_independent_fp32(dtype, rank, biased, linear):
    generator = torch.Generator().manual_seed(91)
    shape = (2, 3, 32) if rank == 3 else (6, 32)
    x_values = torch.randint(-4, 5, shape, generator=generator).to(device="cuda", dtype=dtype) / 16
    weight_values = torch.randint(-4, 5, (32, 48), generator=generator).to(device="cuda", dtype=dtype) / 16
    bias_values = torch.randint(-4, 5, (48,), generator=generator).to(device="cuda", dtype=dtype) / 16
    x = x_values.detach().requires_grad_()
    weight = weight_values.detach().requires_grad_()
    bias = bias_values.detach().requires_grad_() if biased else None

    actual = (
        linear_fp32_reduce(x, weight.t(), bias)
        if linear else matmul_fp32_reduce(x, weight, bias)
    )
    actual.float().sum().backward()

    ref_x = x_values.detach().requires_grad_()
    ref_weight = weight_values.detach().requires_grad_()
    ref_bias = bias_values.detach().requires_grad_() if biased else None
    expected = ref_x.float() @ ref_weight.float()
    if ref_bias is not None:
        expected = expected + ref_bias.float()
    expected = expected.to(dtype)
    expected.float().sum().backward()

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=0, atol=0)
    if bias is not None:
        torch.testing.assert_close(bias.grad, ref_bias.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_bias_only_backward_does_not_hold_dense_weight(dtype):
    x = torch.ones((2, 32), device="cuda", dtype=dtype)
    weight = torch.ones((32, 48), device="cuda", dtype=dtype)
    bias = torch.zeros(48, device="cuda", dtype=dtype, requires_grad=True)
    weight_ref = weakref.ref(weight)
    output = matmul_fp32_reduce(x, weight, bias)
    del weight
    assert weight_ref() is None
    output.float().sum().backward()
    torch.testing.assert_close(bias.grad, torch.full_like(bias, 2), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_wide_backward_empty_rows(dtype):
    x = torch.empty((0, 32), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.ones((32, 48), device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.ones(48, device="cuda", dtype=dtype, requires_grad=True)
    output = matmul_fp32_reduce(x, weight, bias)
    assert output.shape == (0, 48)
    output.float().sum().backward()
    assert x.grad.shape == x.shape
    assert torch.count_nonzero(weight.grad) == 0
    assert torch.count_nonzero(bias.grad) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_native_bias_only_backward_keeps_scaled_matmul(dtype):
    module, _, _, _ = _fp8(dtype, scaled=True)
    module.bias.requires_grad_(True)
    x = torch.ones((2, 32), device="cuda", dtype=dtype)
    if not module._can_use_scaled_mm(x):
        pytest.skip("Native FP8 scaled matmul unavailable on this GPU")

    module(x).float().sum().backward()
    assert not module._scaled_mm_hard_disabled
    torch.testing.assert_close(module.bias.grad, torch.full_like(module.bias, 2), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["gptq", "gptq_streaming", "awq", "fp8_dense", "fp8_scaled", "gguf_dense"])
def test_quantized_cuda_backward_uses_differentiable_path(backend, dtype):
    if backend == "gptq":
        module, weight, _, _ = _gptq(dtype, streaming=False)
    elif backend == "gptq_streaming":
        module, weight, _, _ = _gptq(dtype, streaming=True)
    elif backend == "awq":
        module, weight, _, _ = _awq(dtype)
    elif backend == "fp8_dense":
        module, weight, _, _ = _fp8(dtype, scaled=False)
    elif backend == "fp8_scaled":
        module, weight, _, _ = _fp8(dtype, scaled=True)
    else:
        module, weight, _, _ = _gguf(dtype)

    if backend != "gguf_dense":
        module.train()
    module.bias.requires_grad_(True)
    x = torch.arange(64, device="cuda", dtype=dtype).remainder(7).reshape(2, 32).div(16).requires_grad_()
    if backend == "gptq_streaming":
        assert not module._should_use_streaming(x)
    if backend == "fp8_scaled":
        assert not module._can_use_scaled_mm(x)
    if backend == "gguf_dense":
        module.gguf_fused_cuda_max_rows = 16
        module.gguf_fused_cuda_min_matrix_elements = 0
        assert module._is_fused_k_forward_candidate(x.detach())
        assert not module._is_fused_k_forward_candidate(x)

    ref_x = x.detach().clone().requires_grad_()
    ref_bias = module.bias.detach().clone().requires_grad_()
    ref_weight = torch.full((32, module.out_features), weight, device="cuda", dtype=torch.float32)
    expected = (ref_x.float() @ ref_weight + ref_bias.to(dtype).float()).to(dtype)

    actual = module(x)
    actual.float().mean().backward()
    expected.float().mean().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(module.bias.grad, ref_bias.grad, rtol=2e-3, atol=2e-3)
    if backend == "fp8_scaled":
        assert not module._scaled_mm_hard_disabled
