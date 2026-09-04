# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq_activation import fake_quantize_qvq_fp8_activation
from gptqmodel.utils.qvq_cuda import qvq_cuda_hadamard, qvq_cuda_supported

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not qvq_cuda_supported(),
        reason="requires NVIDIA CUDA compute capability >= 8.0",
    ),
]


@pytest.mark.parametrize(
    ("width", "scale_mode", "source_dtype", "input_rounding_mode"),
    (
        (128, 1, torch.float16, 0),
        (128, 1, torch.bfloat16, 1),
        (2048, 2, torch.float16, 0),
        (2048, 2, torch.bfloat16, 1),
    ),
)
@pytest.mark.parametrize("pad_to_16", (False, True))
def test_qvq_native_fp8_input_hadamard_matches_portable_dequantization(
    width, scale_mode, source_dtype, input_rounding_mode, pad_to_16
):
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("native E4M3 input requires NVIDIA compute capability 8.9 or newer")
    torch.manual_seed(17)
    source = (
        torch.randn(3, width, device="cuda", dtype=source_dtype) * 2.5
    ).contiguous()
    pre_scale = (
        torch.rand(width, device="cuda", dtype=torch.float16) + 0.5
    ).contiguous()
    quantized, scale, dequantized = fake_quantize_qvq_fp8_activation(source)

    expected = qvq_cuda_hadamard(
        dequantized.to(torch.float16).contiguous(),
        pre_scale=pre_scale,
        scale_mode=scale_mode,
        pad_to_16=pad_to_16,
    )
    actual = qvq_cuda_hadamard(
        quantized,
        input_scale=scale,
        input_rounding_mode=input_rounding_mode,
        pre_scale=pre_scale,
        scale_mode=scale_mode,
        pad_to_16=pad_to_16,
    )

    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("bits", (2, 3))
@pytest.mark.parametrize("source_dtype", (torch.float16, torch.bfloat16))
def test_qvq_p32_a8_full_dispatch_matches_explicit_dequantization(bits, source_dtype):
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("native E4M3 input requires NVIDIA compute capability 8.9 or newer")
    torch.manual_seed(29 + int(bits))
    layer = (
        QVQLinear(
            bits=bits,
            in_features=2048,
            out_features=16,
            bank_count=2,
            v2b2_p32=True,
            # This is the explicit legacy pre-linear fake-quantization
            # control. Boolean True now selects the recommended deployed
            # P32-operand target, whose quantization point is intentionally
            # different from fake_quantize_qvq_fp8_activation(source).
            activation_quantization={"target": "linear_input", "replay_passes": 0},
            dtype=source_dtype,
        )
        .eval()
        .cuda()
    )
    layer.post_init()
    source = (
        torch.randn(3, 2048, device="cuda", dtype=source_dtype) * 1.75
    ).contiguous()

    actual = layer(source)
    _, _, dequantized = fake_quantize_qvq_fp8_activation(source)
    layer.activation_quantization = None
    expected = layer(dequantized)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
