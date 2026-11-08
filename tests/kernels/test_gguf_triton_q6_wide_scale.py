# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Q6_K GPU paths must preserve finite dots for wide signed scales."""

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import (
    GGUFTorchLinear,
    _dequantize_gguf_tensor_numpy,
)
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel, fused_q6_k_matmul


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


def _packed_q6(*, code: int, subscale: int = 127, bias: bool = False):
    source = GGUFTorchLinear(
        bits="q6_k", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).eval()
    blocks = np.zeros((64, source.gguf_type_size), dtype=np.uint8)
    # Q6_K stores (code + 32) in low nibbles and two-bit high fields.
    encoded = code + 32
    low, high = encoded & 0x0F, encoded >> 4
    blocks[:, :128] = low | (low << 4)
    blocks[:, 128:192] = high * 0x55
    blocks[:, 192:208] = np.array([subscale], dtype=np.int8).view(np.uint8)
    blocks[:, 208:210] = np.array([1000], dtype=np.float16).view(np.uint8)
    source.qweight.copy_(torch.from_numpy(blocks))
    if bias:
        source.bias.fill_(-16)
    return source, blocks


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("code", "subscale"), [(0, 127), (1, 127), (-32, 127), (31, 127), (1, -127)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_wide_q6_scale_matches_independent_fp64_oracle(code, subscale, dtype, rows, bias):
    source, blocks = _packed_q6(code=code, subscale=subscale, bias=bias)
    module = GGUFTritonKernel(
        bits="q6_k", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    inputs = torch.full((rows, 256), 2**-16, device="cuda", dtype=dtype)
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(blocks, "Q6_K")
    ).to(device="cuda", dtype=torch.float64)
    assert torch.equal(dense, torch.full_like(dense, code * subscale * 1000))
    oracle = inputs.to(torch.float16).to(torch.float64) @ dense.T
    if bias:
        oracle += module.bias.to(torch.float64)
    expected = oracle.to(dtype)

    with torch.inference_mode():
        first = module(inputs)
        repeated = module(inputs)
        cache = module._get_triton_cache(inputs.device)
        direct = fused_q6_k_matmul(
            inputs.to(torch.float16), cache["ql"], cache["qh"], cache["scale"],
            output_dtype=torch.float32,
        )
    torch.cuda.synchronize()

    assert cache["scale"].dtype == torch.float32
    assert first.dtype == dtype
    assert torch.isfinite(first).all()
    assert torch.equal(first, repeated)
    torch.testing.assert_close(first, expected, rtol=2e-3, atol=2e-3)
    direct_reference = oracle - (module.bias.to(torch.float64) if bias else 0)
    torch.testing.assert_close(
        direct.to(dtype), direct_reference.to(dtype), rtol=2e-3, atol=2e-3,
    )
    assert all(
        value.numel() < module.in_features * module.out_features
        for value in cache.values() if torch.is_tensor(value)
    )


@requires_gpu
@pytest.mark.cuda
def test_ordinary_q6_scale_keeps_half_metadata_and_reselects_on_update():
    source = GGUFTorchLinear(
        bits="q6_k", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False,
    ).eval()
    blocks = np.zeros((64, source.gguf_type_size), dtype=np.uint8)
    blocks[:, :128] = 0x11
    blocks[:, 128:192] = 0xAA
    blocks[:, 192:208] = 1
    blocks[:, 208:210] = np.array([1], dtype=np.float16).view(np.uint8)
    source.qweight.copy_(torch.from_numpy(blocks))
    module = GGUFTritonKernel(
        bits="q6_k", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    inputs = torch.ones((1, 256), device="cuda", dtype=torch.float16)

    with torch.no_grad():
        ordinary = module(inputs)
        assert module._get_triton_cache(inputs.device)["scale"].dtype == torch.float16
        wide_source, _ = _packed_q6(code=1)
        module.qweight.copy_(wide_source.qweight.to("cuda"))
        wide = module(inputs * 2**-16)
        assert module._get_triton_cache(inputs.device)["scale"].dtype == torch.float32

    assert ordinary[0, 0].item() == 256
    assert torch.isfinite(wide).all()
    assert wide[0, 0].item() == 496


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("path", ["dense", "chunked"])
@pytest.mark.parametrize("code", [0, 1, -32])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_torch_q6_wide_weight_keeps_finite_output(path, code, rows, bias):
    source, blocks = _packed_q6(code=code, bias=bias)
    module = GGUFTorchLinear(
        bits="q6_k", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    if path == "dense":
        module.gguf_fused_cuda_max_rows = 0
    else:
        module.gguf_fused_cuda_min_matrix_elements = 0
        module.autotune_enabled = False
    inputs = torch.full((rows, 256), 2**-16, device="cuda", dtype=torch.float16)
    assert module._should_use_fused_k_forward(inputs) is (path == "chunked")
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(blocks, "Q6_K")
    ).to(device="cuda", dtype=torch.float64)
    oracle = inputs.double() @ dense.T
    if bias:
        oracle += module.bias.double()

    with torch.inference_mode():
        first = module(inputs)
        repeated = module(inputs)
    torch.cuda.synchronize()

    assert first.dtype == inputs.dtype
    assert torch.isfinite(first).all()
    assert torch.equal(first, repeated)
    torch.testing.assert_close(first, oracle.to(inputs.dtype), rtol=2e-3, atol=2e-3)
    assert list(module._k_scale_overflow_cache.values()) == [True]
    assert not any(
        torch.is_tensor(value) and value.numel() >= module.in_features * module.out_features
        for value in vars(module).values()
    )


@requires_gpu
@pytest.mark.cuda
def test_torch_q6_scale_guard_reselects_after_weight_replacement():
    source, _ = _packed_q6(code=1)
    module = GGUFTorchLinear(
        bits="q6_k", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    blocks, _, _ = module._reshape_blocks(device="cuda")
    assert module._k_scale_overflows(blocks, torch.float16)

    with torch.no_grad():
        module.qweight[:, 208:210] = torch.from_numpy(
            np.array([1], dtype=np.float16).view(np.uint8)
        ).to("cuda")
    blocks, _, _ = module._reshape_blocks(device="cuda")
    assert not module._k_scale_overflows(blocks, torch.float16)
    assert list(module._k_scale_overflow_cache.values()) == [False]


@requires_gpu
@pytest.mark.cuda
def test_torch_q6_wide_dense_fallback_preserves_input_gradient():
    source, _ = _packed_q6(code=1)
    module = GGUFTorchLinear(
        bits="q6_k", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    module.gguf_fused_cuda_max_rows = 0
    inputs = torch.full(
        (1, 256), 2**-16, device="cuda", dtype=torch.float16, requires_grad=True,
    )

    loss = module(inputs).float().sum() / 128
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    expected = torch.full_like(inputs.grad, 64 * 127000 / 128)
    torch.testing.assert_close(inputs.grad, expected, rtol=2e-3, atol=2e-3)


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_torch_q6_dense_bf16_signed_subscale_cancellation(rows, bias):
    source = GGUFTorchLinear(
        bits="q6_k", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).eval()
    blocks = np.zeros((64, source.gguf_type_size), dtype=np.uint8)
    blocks[:, :128] = 0x11
    blocks[:, 128:192] = 0xAA
    blocks[:, 192:208] = np.array([127, -126] * 8, dtype=np.int8).view(np.uint8)
    blocks[:, 208:210] = np.array([1000], dtype=np.float16).view(np.uint8)
    source.qweight.copy_(torch.from_numpy(blocks))
    if bias:
        source.bias.fill_(-16)
    module = GGUFTorchLinear(
        bits="q6_k", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    module.gguf_fused_cuda_max_rows = 0
    inputs = torch.full((rows, 256), 2**-8, device="cuda", dtype=torch.bfloat16)
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(blocks, "Q6_K")
    ).to(device="cuda", dtype=torch.float64)
    oracle = inputs.double() @ dense.T
    if bias:
        oracle += module.bias.double()
    assert oracle[0, 0].item() == (484 if bias else 500)

    with torch.inference_mode():
        first = module(inputs)
        repeated = module(inputs)
    torch.cuda.synchronize()

    assert first.dtype == torch.bfloat16
    assert torch.isfinite(first).all()
    assert torch.equal(first, repeated)
    torch.testing.assert_close(first, oracle.to(torch.bfloat16), rtol=2e-3, atol=2e-3)
    assert not any(
        torch.is_tensor(value) and value.numel() >= module.in_features * module.out_features
        for value in vars(module).values()
    )
