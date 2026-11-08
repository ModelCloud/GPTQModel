# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GGUF K-scale overflow checks handle inference tensors and weight updates."""

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear, _dequantize_gguf_tensor_numpy


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


def _packed_module(bits: str):
    k, n = 256, 64
    linear = torch.nn.Linear(k, n, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(1)
    module = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=False, register_buffers=False,
    )
    module.pack_original(linear, scales=None, zeros=None)
    module.post_init()
    return module


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q4_k", "q5_k"])
def test_inference_packed_weight_runs_fused_cuda_and_dequantizes(bits):
    module = _packed_module(bits).cuda().eval()
    packed = module.qweight.detach().cpu().numpy()
    inputs = torch.ones((1, module.in_features), device="cuda", dtype=torch.float16)
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(packed, module.gguf_tensor_qtype)
    ).to(device="cuda", dtype=torch.float64)
    reference = inputs.to(torch.float64) @ dense.T

    with torch.inference_mode():
        module.qweight = module.qweight.clone()
        assert module.qweight.is_inference()
        first = module._forward_fused_k(inputs)
        second = module._forward_fused_k(inputs)
        dequantized = module.dequantize_weight(device="cuda", dtype=torch.float16)

    assert first.dtype == torch.float16
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    error = (first.to(torch.float64) - reference).abs().max().item()
    assert error / reference.abs().max().item() <= 2e-3
    assert torch.allclose(dequantized.T.to(torch.float64), dense, rtol=2e-3, atol=0)
    assert len(module._k_scale_overflow_cache) == 1
    assert all(isinstance(value, bool) for value in module._k_scale_overflow_cache.values())
    assert not any(
        torch.is_tensor(value) and value.numel() >= module.in_features * module.out_features
        for value in vars(module).values()
    )
    assert all(
        value.dtype == torch.uint8 or value.numel() < module.in_features * module.out_features
        for value in module._buffers.values() if torch.is_tensor(value)
    )


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q4_k", "q5_k"])
def test_versioned_scale_cache_replaces_stale_answers(bits):
    module = _packed_module(bits).cuda().eval()
    low_scale = torch.from_numpy(np.array([1], dtype=np.float16).view(np.uint8)).to("cuda")
    high_scale = torch.from_numpy(np.array([100], dtype=np.float16).view(np.uint8)).to("cuda")

    with torch.no_grad():
        for index in range(12):
            high = bool(index % 2)
            module.qweight[:, :2].copy_(high_scale if high else low_scale)
            blocks, _, _ = module._reshape_blocks(device="cuda")
            assert module._k_scale_overflows(blocks, torch.float16) is high
            assert module._k_scale_overflows(blocks, torch.float16) is high
            assert len(module._k_scale_overflow_cache) == 1

        module.load_state_dict(_packed_module(bits).state_dict(), strict=True)
        assert not module._k_scale_overflow_cache
        assert module._k_scale_overflow_cache_source_key is None
        module._k_scale_overflows(module._reshape_blocks(device="cuda")[0], torch.float16)
        module.cpu().cuda()
        assert not module._k_scale_overflow_cache


@requires_gpu
@pytest.mark.cuda
def test_inference_tensor_update_uses_explicit_clear():
    module = _packed_module("q4_k").cuda().eval()
    high_scale = torch.from_numpy(np.array([100], dtype=np.float16).view(np.uint8)).to("cuda")

    with torch.inference_mode():
        module.qweight = module.qweight.clone()
        blocks, _, _ = module._reshape_blocks(device="cuda")
        assert module._k_scale_overflows(blocks, torch.float16) is False
        module.qweight[:, :2].copy_(high_scale)
        module.clear_weight_cache()
        blocks, _, _ = module._reshape_blocks(device="cuda")
        assert module._k_scale_overflows(blocks, torch.float16) is True
        assert len(module._k_scale_overflow_cache) == 1
