# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GGUF Triton epilogue checks against an independent GGUF block decoder."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import gguf_triton as gguf_triton_module
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear, _dequantize_gguf_tensor_numpy
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


def _packed_pair(bits: str, in_features: int, out_features: int = 64):
    linear = torch.nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(1)
        linear.bias.fill_(-16384)

    source = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        bias=True, register_buffers=False,
    )
    source.pack_original(linear, scales=None, zeros=None)
    source.post_init()

    module = GGUFTritonKernel(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        bias=True, register_buffers=True,
    )
    module.load_state_dict(source.state_dict(), strict=True)
    return source, module.to("cuda").eval()


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize(
    ("bits", "in_features", "input_value"),
    [
        ("q4_k", 256, 300),
        ("q5_k", 256, 300),
        ("q6_k", 256, 300),
        ("q1_0", 1024, 75),
        ("q1_0", 2048, 37.5),
    ],
)
def test_biased_result_is_finite_after_wide_kernel_output(
    bits, in_features, input_value, rows, dtype
):
    source, module = _packed_pair(bits, in_features)
    inputs = torch.full((rows, in_features), input_value, device="cuda", dtype=dtype)

    # NumPy unpacks the original GGUF bytes independently of Triton's cached
    # scales, codes, and dot product. FP64 Torch then supplies the biased oracle.
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(
            source.qweight.detach().cpu().numpy(), module.gguf_tensor_qtype
        )
    ).to(device="cuda", dtype=torch.float64)
    reference = (
        inputs.to(torch.float16).to(torch.float64) @ dense.T
        + module.bias.to(torch.float64)
    )
    assert reference.abs().max() < torch.finfo(torch.float16).max
    assert (reference + 16384).abs().min() > torch.finfo(torch.float16).max

    with torch.inference_mode():
        actual = module(inputs)
        repeated = module(inputs)

    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, repeated)
    error = (actual.to(torch.float64) - reference).abs().max().item()
    assert error / reference.abs().max().item() <= 2e-3

    # Only packed blocks and their compact metadata may persist.
    for entry in module._gguf_triton_cache.values():
        for value in entry.values():
            if torch.is_tensor(value):
                assert value.numel() < in_features * module.out_features


@requires_gpu
@pytest.mark.cuda
def test_fixed_launch_writes_fp32_before_bias(monkeypatch):
    source, module = _packed_pair("q4_k", 256)
    inputs = torch.full((1, 256), 300, device="cuda", dtype=torch.float16)
    cache = module._get_triton_cache(inputs.device)
    fixed_config = {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32,
        "num_warps": 4, "num_stages": 2,
    }
    monkeypatch.setattr(
        gguf_triton_module, "_select_q4_k_fixed_launch_config", lambda **kwargs: fixed_config
    )

    direct = gguf_triton_module._launch_q4_k_fixed_matmul(
        inputs, cache["qs"], cache["scale"], cache["min"],
        fixed_config=fixed_config, output_dtype=torch.float32,
    )
    actual = module(inputs)
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(source.qweight.cpu().numpy(), "Q4_K")
    ).to(device="cuda", dtype=torch.float64)
    reference = (inputs.to(torch.float64) @ dense.T + module.bias.to(torch.float64)).half()

    assert direct.dtype == torch.float32
    assert torch.isfinite(direct).all()
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, reference, rtol=2e-3, atol=0)


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(0, 256), (2, 0, 256)])
def test_empty_input_does_not_launch_triton(shape):
    _, module = _packed_pair("q4_k", 256)
    inputs = torch.empty(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    result = module(inputs)
    assert result.shape == shape[:-1] + (module.out_features,)
    assert result.dtype == inputs.dtype
    result.sum().backward()
    assert inputs.grad is not None
    assert inputs.grad.shape == inputs.shape

    with pytest.raises(RuntimeError, match="only supports CUDA/ROCm inference"):
        module(torch.empty(shape, dtype=torch.float16))
