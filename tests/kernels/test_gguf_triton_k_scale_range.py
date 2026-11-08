# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GGUF Triton K-quant scales must stay wide until the weight is reconstructed."""

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import (
    GGUFTorchLinear,
    _dequantize_gguf_tensor_numpy,
)
from gptqmodel.nn_modules.qlinear.gguf_triton import (
    GGUFTritonKernel,
    fused_q4_k_matmul,
    fused_q5_k_matmul,
)


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


def _packed_source(bits: str, *, bias: bool):
    code_max = 15 if bits == "q4_k" else 31
    d = 74.0
    dmin = float(round((d * 63 * code_max - 30000.0) / 63))
    source = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).eval()
    blocks = np.zeros((64, source.gguf_type_size), dtype=np.uint8)
    blocks[:, :2] = np.array([d], dtype=np.float16).view(np.uint8)
    blocks[:, 2:4] = np.array([dmin], dtype=np.float16).view(np.uint8)
    blocks[:, 4:8] = 63
    blocks[:, 8:12] = 63
    if bits == "q4_k":
        blocks[:, 16:] = 0xFF
    else:
        blocks[:, 16:48] = 0xFF
        blocks[:, 48:] = 0xFF
    source.qweight.copy_(torch.from_numpy(blocks))
    if bias:
        source.bias.fill_(-16)
    return source, blocks, d, dmin, code_max


def _wide_weight_source(bits: str, *, bias: bool):
    source = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).eval()
    blocks = np.zeros((64, source.gguf_type_size), dtype=np.uint8)
    blocks[:, :2] = np.array([1000], dtype=np.float16).view(np.uint8)
    blocks[:, 4:8] = 63
    if bits == "q4_k":
        blocks[:, 16:] = 0xFF
    else:
        blocks[:, 16:48] = 0xFF
        blocks[:, 48:] = 0xFF
    source.qweight.copy_(torch.from_numpy(blocks))
    if bias:
        source.bias.fill_(-16)
    return source, blocks


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q4_k", "q5_k"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_finite_k_weight_after_out_of_range_intermediate(bits, dtype, rows, bias):
    source, blocks, d, dmin, code_max = _packed_source(bits, bias=bias)
    assert d * 63 * code_max > torch.finfo(torch.float16).max
    assert abs(d * 63 * code_max - dmin * 63) < torch.finfo(torch.float16).max

    module = GGUFTritonKernel(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    inputs = torch.full((rows, 256), 2**-16, device="cuda", dtype=dtype)
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(blocks, module.gguf_tensor_qtype)
    ).to(device="cuda", dtype=torch.float64)
    oracle = inputs.to(torch.float16).to(torch.float64) @ dense.T
    if bias:
        oracle += module.bias.to(torch.float64)
    expected = oracle.to(dtype)

    with torch.inference_mode():
        first = module(inputs)
        second = module(inputs)
    torch.cuda.synchronize()

    assert first.dtype == dtype
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, expected, rtol=2e-3, atol=2e-3)
    assert torch.equal(first, second)
    cache = module._get_triton_cache(inputs.device)
    assert cache["scale"].dtype == torch.float32
    assert cache["min"].dtype == torch.float32
    assert all(
        value.numel() < module.in_features * module.out_features
        for value in cache.values() if torch.is_tensor(value)
    )


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q4_k", "q5_k"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_out_of_fp16_weight_still_has_finite_dot_product(bits, dtype):
    source, blocks = _wide_weight_source(bits, bias=False)
    module = GGUFTritonKernel(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    inputs = torch.full((1, 256), 2**-16, device="cuda", dtype=dtype)
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(blocks, module.gguf_tensor_qtype)
    ).to(device="cuda", dtype=torch.float64)
    assert dense.abs().max() > torch.finfo(torch.float16).max
    oracle = inputs.to(torch.float16).to(torch.float64) @ dense.T
    assert oracle.abs().max() < torch.finfo(torch.float16).max

    actual = module(inputs)
    cache = module._get_triton_cache(inputs.device)
    x_half = inputs.to(torch.float16)
    if bits == "q4_k":
        direct = fused_q4_k_matmul(
            x_half, cache["qs"], cache["scale"], cache["min"], output_dtype=torch.float32,
        )
    else:
        direct = fused_q5_k_matmul(
            x_half, cache["qs"], cache["qh"], cache["scale"], cache["min"],
            output_dtype=torch.float32,
        )

    assert cache["scale"].dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, oracle.to(dtype), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(direct, oracle.to(torch.float32), rtol=2e-3, atol=2e-3)


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q4_k", "q5_k"])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_torch_dense_wide_k_weight_still_has_finite_dot_product(bits, rows, bias):
    source, blocks = _wide_weight_source(bits, bias=bias)
    module = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    module.gguf_fused_cuda_max_rows = 0
    inputs = torch.full((rows, 256), 2**-16, device="cuda", dtype=torch.float16)
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(blocks, module.gguf_tensor_qtype)
    ).to(device="cuda", dtype=torch.float64)
    oracle = inputs.double() @ dense.T
    if bias:
        oracle += module.bias.double()

    with torch.inference_mode():
        first = module(inputs)
        repeated = module(inputs)
    torch.cuda.synchronize()

    assert torch.isfinite(first).all()
    assert torch.equal(first, repeated)
    torch.testing.assert_close(first, oracle.to(inputs.dtype), rtol=2e-3, atol=2e-3)
    assert any(
        dtype == torch.float16 and overflow
        for (_, dtype), overflow in module._k_scale_overflow_cache.items()
    )
    assert not any(
        torch.is_tensor(value) and value.numel() >= module.in_features * module.out_features
        for value in vars(module).values()
    )


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q4_k", "q5_k"])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_torch_dense_bf16_k_cancellation_matches_fp64_oracle(bits, rows, bias):
    source, blocks, _, _, _ = _packed_source(bits, bias=bias)
    module = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    module.gguf_fused_cuda_max_rows = 0
    inputs = torch.full((rows, 256), 2**-16, device="cuda", dtype=torch.bfloat16)
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(blocks, module.gguf_tensor_qtype)
    ).to(device="cuda", dtype=torch.float64)
    oracle = inputs.double() @ dense.T
    if bias:
        oracle += module.bias.double()

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


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q4_k", "q5_k"])
def test_ordinary_k_scale_keeps_compact_fp16_metadata(bits):
    linear = torch.nn.Linear(256, 64, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(1)
    source = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False, register_buffers=False,
    )
    source.pack_original(linear, scales=None, zeros=None)
    source.post_init()
    module = GGUFTritonKernel(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    inputs = torch.ones((1, 256), device="cuda", dtype=torch.float16)
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(
            source.qweight.detach().cpu().numpy(), module.gguf_tensor_qtype
        )
    ).to(device="cuda", dtype=torch.float64)

    actual = module(inputs)
    oracle = inputs.to(torch.float64) @ dense.T
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, oracle.to(torch.float16), rtol=2e-3, atol=2e-3)
    cache = module._get_triton_cache(inputs.device)
    assert cache["scale"].dtype == torch.float16
    assert cache["min"].dtype == torch.float16


@requires_gpu
@pytest.mark.cuda
def test_scale_width_reselects_after_packed_weight_replacement():
    wide_source, _, _, _, _ = _packed_source("q5_k", bias=False)
    linear = torch.nn.Linear(256, 64, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(1)
    ordinary_source = GGUFTorchLinear(
        bits="q5_k", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False, register_buffers=False,
    )
    ordinary_source.pack_original(linear, scales=None, zeros=None)
    ordinary_source.post_init()
    module = GGUFTritonKernel(
        bits="q5_k", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False,
    ).to("cuda").eval()
    module.load_state_dict(ordinary_source.state_dict(), strict=True)
    inputs = torch.full((1, 256), 2**-16, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        module(inputs)
        assert module._get_triton_cache(inputs.device)["min"].dtype == torch.float16
        module.qweight.copy_(wide_source.qweight.to("cuda"))
        actual = module(inputs)
        assert module._get_triton_cache(inputs.device)["min"].dtype == torch.float32

    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(
            wide_source.qweight.detach().cpu().numpy(), "Q5_K"
        )
    ).to(device="cuda", dtype=torch.float64)
    oracle = inputs.to(torch.float64) @ dense.T
    torch.testing.assert_close(actual, oracle.to(torch.float16), rtol=2e-3, atol=2e-3)
