# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A finite quantized linear result must not overflow before bias cancellation."""

import pytest
import torch
from test_torch_weight_storage import state_tensors

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required for FP32 bias epilogue",
)


def _gptq(dtype, streaming):
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
@pytest.mark.parametrize("backend", ["gptq", "gptq_streaming", "awq", "fp8_dense", "fp8_scaled", "gguf_dense"])
def test_cuda_bias_cancellation_rounds_once(backend, dtype):
    if backend == "gptq":
        module, weight, value, bias = _gptq(dtype, streaming=False)
    elif backend == "gptq_streaming":
        module, weight, value, bias = _gptq(dtype, streaming=True)
    elif backend == "awq":
        module, weight, value, bias = _awq(dtype)
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
