# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Less common GGUF CUDA formats keep finite dots with wide decoded weights."""

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear, _dequantize_gguf_tensor_numpy


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


def _packed_source(bits: str, code: int, *, bias: bool):
    source = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).eval()
    blocks = np.zeros((64, 256 // source.gguf_block_size, source.gguf_type_size), dtype=np.uint8)
    if bits == "q2_0":
        blocks[..., :2] = np.array([50000], dtype=np.float16).view(np.uint8)
        packed_code = code + 1
        blocks[..., 2:] = packed_code * 0x55
    elif bits == "q8_k":
        blocks[..., :4] = np.array([1000], dtype=np.float32).view(np.uint8)
        blocks[..., 4:260] = np.array([code], dtype=np.int8).view(np.uint8)
    else:
        # E8M0 byte 143 decodes to 32768; FP4 code 7 decodes to 12.
        blocks[..., 0] = 143
        blocks[..., 1:] = 0x77
    packed = blocks.reshape(64, -1)
    source.qweight.copy_(torch.from_numpy(packed))
    if bias:
        source.bias.fill_(-16)
    return source, packed


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize(("bits", "code", "input_value"), [
    ("q2_0", 2, 2**-9),
    ("q8_k", 127, 2**-10),
    ("q8_k", -128, 2**-10),
    ("mxfp4", 7, 2**-12),
])
@pytest.mark.parametrize("path", ["dense", "chunked"])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_other_wide_gguf_weight_has_finite_fp16_dot(bits, code, input_value, path, rows, bias):
    source, packed = _packed_source(bits, code, bias=bias)
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
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(packed, module.gguf_tensor_qtype)
    ).to(device="cuda", dtype=torch.float64)
    assert dense.abs().max() > torch.finfo(torch.float16).max
    oracle = inputs.double() @ dense.T
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
    if bits == "mxfp4":
        assert not module._k_scale_overflow_cache
    else:
        assert list(module._k_scale_overflow_cache.values()) == [True]
    assert not any(
        torch.is_tensor(value) and value.numel() >= module.in_features * module.out_features
        for value in vars(module).values()
    )


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q2_0", "q8_k"])
def test_other_scale_bound_reselects_after_packed_weight_update(bits):
    source, _ = _packed_source(bits, 2 if bits == "q2_0" else 127, bias=False)
    module = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    blocks, _, _ = module._reshape_blocks(device="cuda")
    assert module._k_scale_overflows(blocks, torch.float16)

    scale_dtype = np.float16 if bits == "q2_0" else np.float32
    low_scale = torch.from_numpy(np.array([1], dtype=scale_dtype).view(np.uint8)).to("cuda")
    with torch.no_grad():
        module.qweight.view(64, -1, module.gguf_type_size)[..., : low_scale.numel()] = low_scale
    blocks, _, _ = module._reshape_blocks(device="cuda")
    assert not module._k_scale_overflows(blocks, torch.float16)
    assert list(module._k_scale_overflow_cache.values()) == [False]
