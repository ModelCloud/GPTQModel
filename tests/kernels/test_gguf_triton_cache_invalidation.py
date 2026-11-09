# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Packed GGUF Triton metadata follows weight writes and lifecycle events."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear, _dequantize_gguf_tensor_numpy
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


def _packed_source(bits: str, in_features: int, out_features: int, weight_value: float):
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(weight_value)
    source = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        bias=False, register_buffers=False,
    )
    source.pack_original(linear, scales=None, zeros=None)
    source.post_init()
    return source


def _reference(inputs: torch.Tensor, packed: torch.Tensor, qtype: str) -> torch.Tensor:
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(packed.detach().cpu().numpy(), qtype)
    ).to(device=inputs.device, dtype=torch.float64)
    return inputs.to(torch.float64) @ dense.T


def _assert_matches(actual: torch.Tensor, expected: torch.Tensor):
    assert actual.dtype == torch.float16
    error = (actual.to(torch.float64) - expected).abs().max().item()
    assert error / expected.abs().max().item() <= 2e-3


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q1_0", "q1_0_g128", "q4_k", "q5_k", "q6_k"])
def test_in_place_weight_update_rebuilds_packed_metadata(bits):
    k, n = 256, 64
    old = _packed_source(bits, k, n, 1)
    new = _packed_source(bits, k, n, 2)
    module = GGUFTritonKernel(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=False, register_buffers=True,
    ).to("cuda").eval()
    module.load_state_dict(old.state_dict(), strict=True)
    inputs = torch.ones((1, k), device="cuda", dtype=torch.float16)

    with torch.no_grad():
        first = module(inputs)
        first_cache = module._get_triton_cache(inputs.device)
        assert module._get_triton_cache(inputs.device) is first_cache
        module.qweight.copy_(new.qweight.to("cuda"))
        second = module(inputs)
        second_cache = module._get_triton_cache(inputs.device)

    _assert_matches(first, _reference(inputs, old.qweight, module.gguf_tensor_qtype))
    _assert_matches(second, _reference(inputs, new.qweight, module.gguf_tensor_qtype))
    assert (second - first).abs().max().item() > 100
    assert second_cache is not first_cache
    assert second_cache["qweight_key"] != first_cache["qweight_key"]
    for value in second_cache.values():
        if torch.is_tensor(value):
            assert value.numel() < k * n


@requires_gpu
@pytest.mark.cuda
def test_replacement_state_load_and_device_move_clear_cache():
    k, n = 256, 64
    old = _packed_source("q4_k", k, n, 1)
    new = _packed_source("q4_k", k, n, 2)
    module = GGUFTritonKernel(
        bits="q4_k", group_size=-1, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=False, register_buffers=True,
    ).to("cuda").eval()
    module.load_state_dict(old.state_dict(), strict=True)
    inputs = torch.ones((1, k), device="cuda", dtype=torch.float16)
    with torch.no_grad():
        module(inputs)
        module.qweight = new.qweight.to("cuda")
        replaced = module(inputs)
        _assert_matches(replaced, _reference(inputs, new.qweight, "Q4_K"))

        module.load_state_dict(old.state_dict(), strict=True)
        assert not module._gguf_triton_cache
        loaded = module(inputs)
        _assert_matches(loaded, _reference(inputs, old.qweight, "Q4_K"))

        module.cpu().cuda()
        assert not module._gguf_triton_cache
        moved = module(inputs)
        _assert_matches(moved, _reference(inputs, old.qweight, "Q4_K"))


@requires_gpu
@pytest.mark.cuda
def test_inference_tensor_can_be_refreshed_by_explicit_clear():
    k, n = 256, 64
    old = _packed_source("q4_k", k, n, 1)
    new = _packed_source("q4_k", k, n, 2)
    module = GGUFTritonKernel(
        bits="q4_k", group_size=-1, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=False, register_buffers=True,
    ).to("cuda").eval()
    inputs = torch.ones((1, k), device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        module.qweight = old.qweight.to("cuda").clone()
        assert module.qweight.is_inference()
        module(inputs)
        module.qweight.copy_(new.qweight.to("cuda"))
        module.clear_weight_cache()
        actual = module(inputs)

    _assert_matches(actual, _reference(inputs, new.qweight, "Q4_K"))
