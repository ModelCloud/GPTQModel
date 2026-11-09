# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GGUF BF16 dense fallback must retain weight precision through cancellation."""

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear, _dequantize_gguf_tensor_numpy


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


def _cancelling_blocks(bits: str, *, bias: bool):
    source = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).eval()
    num_blocks = 256 // source.gguf_block_size
    blocks = np.zeros((64, num_blocks, source.gguf_type_size), dtype=np.uint8)
    midpoint = num_blocks // 2
    blocks[:, :midpoint, :2] = np.array([1000], dtype=np.float16).view(np.uint8)
    blocks[:, midpoint:, :2] = np.array([999], dtype=np.float16).view(np.uint8)
    if bits == "q4_0":
        blocks[:, :midpoint, 2:] = 0xFF  # decoded code +7
        blocks[:, midpoint:, 2:] = 0x11  # decoded code -7
    elif bits == "q8_0":
        blocks[:, :midpoint, 2:] = np.array([7], dtype=np.int8).view(np.uint8)
        blocks[:, midpoint:, 2:] = np.array([-7], dtype=np.int8).view(np.uint8)
    else:
        blocks[:, :midpoint, 2:] = 0xFF  # decoded sign +1
        blocks[:, midpoint:, 2:] = 0x00  # decoded sign -1
    packed = blocks.reshape(64, -1)
    source.qweight.copy_(torch.from_numpy(packed))
    if bias:
        source.bias.fill_(-4)
    return source, packed


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q1_0", "q4_0", "q8_0"])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_bf16_dense_fallback_matches_independent_fp64_oracle(bits, rows, bias):
    source, packed = _cancelling_blocks(bits, bias=bias)
    module = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=bias,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    module.gguf_fused_cuda_max_rows = 0
    value = 2**-7 if bits == "q1_0" else 2**-5
    inputs = torch.full((rows, 256), value, device="cuda", dtype=torch.bfloat16)
    decoded = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(packed, module.gguf_tensor_qtype)
    ).to(device="cuda", dtype=torch.float64)
    oracle = inputs.double() @ decoded.T
    if bias:
        oracle += module.bias.double()
    assert oracle[0, 0].item() == ((1 if bits == "q1_0" else 28) - (4 if bias else 0))

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
def test_bf16_dense_fallback_preserves_input_gradient():
    source, packed = _cancelling_blocks("q4_0", bias=False)
    module = GGUFTorchLinear(
        bits="q4_0", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False,
    ).to("cuda").eval()
    module.load_state_dict(source.state_dict(), strict=True)
    module.gguf_fused_cuda_max_rows = 0
    inputs = torch.full(
        (1, 256), 2**-5, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    expected = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(packed, "Q4_0")[0]
    ).to(device="cuda", dtype=torch.bfloat16).unsqueeze(0)

    (module(inputs).float().sum() / 64).backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    torch.testing.assert_close(inputs.grad, expected, rtol=2e-3, atol=2e-3)
