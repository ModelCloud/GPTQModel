# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Q4_0 and Q8_0 CUDA paths must preserve finite dots with wide weights."""

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear, _dequantize_gguf_tensor_numpy


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


def _wide_simple_source(bits: str, code: int, d: int, *, bias: bool):
    source = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).eval()
    blocks = np.zeros((64, 256 // source.gguf_block_size, source.gguf_type_size), dtype=np.uint8)
    blocks[..., :2] = np.array([d], dtype=np.float16).view(np.uint8)
    if bits == "q4_0":
        blocks[..., 2:] = ((code + 8) << 4) | (code + 8)
    else:
        blocks[..., 2:] = np.array([code], dtype=np.int8).view(np.uint8)
    packed = blocks.reshape(64, -1)
    source.qweight.copy_(torch.from_numpy(packed))
    if bias:
        source.bias.fill_(-16)
    return source, packed


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("bits", "code", "d", "input_value"),
    [
        ("q4_0", 7, 10000, 2**-9),
        ("q4_0", -8, 10000, 2**-10),
        ("q8_0", 127, 1000, 2**-10),
        ("q8_0", -128, 1000, 2**-10),
    ],
)
@pytest.mark.parametrize("path", ["dense", "chunked"])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_wide_simple_weight_keeps_finite_fp16_dot(bits, code, d, input_value, path, rows, bias):
    source, packed = _wide_simple_source(bits, code, d, bias=bias)
    module = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    if path == "dense":
        module.gguf_fused_cuda_max_rows = 0
    else:
        module.gguf_fused_cuda_min_matrix_elements = 0
        module.autotune_enabled = False
    inputs = torch.full((rows, 256), input_value, device="cuda", dtype=torch.float16)
    assert module._should_use_fused_k_forward(inputs) is (path == "chunked")
    decoded = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(packed, module.gguf_tensor_qtype)
    ).to(device="cuda", dtype=torch.float64)
    assert decoded.abs().max() > torch.finfo(torch.float16).max
    oracle = inputs.double() @ decoded.T
    if bias:
        oracle += module.bias.double()
    assert oracle.abs().max() < torch.finfo(torch.float16).max

    with torch.inference_mode():
        first = module(inputs)
        repeated = module(inputs)
    torch.cuda.synchronize()

    assert first.dtype == torch.float16
    assert torch.isfinite(first).all()
    assert torch.equal(first, repeated)
    torch.testing.assert_close(first, oracle.to(torch.float16), rtol=2e-3, atol=2e-3)
    assert list(module._k_scale_overflow_cache.values()) == [True]
    assert not any(
        torch.is_tensor(value) and value.numel() >= module.in_features * module.out_features
        for value in vars(module).values()
    )


@requires_gpu
@pytest.mark.cuda
def test_simple_scale_guard_reselects_after_packed_weight_update():
    source, _ = _wide_simple_source("q8_0", 127, 1000, bias=False)
    module = GGUFTorchLinear(
        bits="q8_0", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    blocks, _, _ = module._reshape_blocks(device="cuda")
    assert module._k_scale_overflows(blocks, torch.float16)

    with torch.no_grad():
        module.qweight.view(64, -1, module.gguf_type_size)[..., :2] = torch.from_numpy(
            np.array([1], dtype=np.float16).view(np.uint8)
        ).to("cuda")
    blocks, _, _ = module._reshape_blocks(device="cuda")
    assert not module._k_scale_overflows(blocks, torch.float16)
    assert list(module._k_scale_overflow_cache.values()) == [False]
