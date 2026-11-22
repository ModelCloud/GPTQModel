# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Portable Torch checks on CPU and the active CUDA/ROCm runtime device."""

import pytest
import torch
from ricursive_test_utils import make_model


@pytest.fixture(params=["cpu", "cuda:0"])
def runtime_device(request):
    # This selects the Torch execution namespace, not a vendor-specific kernel.
    # Both NVIDIA CUDA and AMD HIP use this namespace for these Torch operations.
    if request.param != "cpu" and not torch.cuda.is_available():
        pytest.skip("No CUDA/ROCm runtime device")
    return torch.device(request.param)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_heads,kv_heads", [(8, 2), (80, 8)])
@torch.inference_mode()
def test_attention_weights_preserve_logits_and_dtype(runtime_device, dtype, query_heads, kv_heads):
    model = make_model(num_attention_heads=query_heads, num_key_value_heads=kv_heads).to(
        device=runtime_device, dtype=dtype
    )
    ids = torch.arange(3, 16, device=runtime_device).unsqueeze(0)
    expected = model(ids, use_cache=False).logits
    output = model(ids, use_cache=False, output_attentions=True)
    assert output.logits.dtype == dtype
    assert all(weights.dtype == dtype for weights in output.attentions)
    torch.testing.assert_close(output.logits, expected, atol=2e-3, rtol=2e-3)
    assert torch.equal(output.logits.argmax(-1), expected.argmax(-1))
