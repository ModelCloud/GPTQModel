# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Portable Torch checks on CPU and the active CUDA/ROCm runtime device."""

import pytest
import torch
from ricursive_test_utils import compatibility_module, make_model


@pytest.fixture(params=["cpu", "cuda:0"])
def runtime_device(request):
    # This selects the Torch execution namespace, not a vendor-specific kernel.
    # Both NVIDIA CUDA and AMD HIP use this namespace for these Torch operations.
    if request.param != "cpu" and not torch.cuda.is_available():
        pytest.skip("No CUDA/ROCm runtime device")
    return torch.device(request.param)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_heads,kv_heads", [(8, 2), (80, 8)])
@pytest.mark.parametrize("legacy", [False, True])
@torch.inference_mode()
def test_attention_weights_preserve_logits_and_dtype(runtime_device, dtype, query_heads, kv_heads, legacy):
    model = make_model(num_attention_heads=query_heads, num_key_value_heads=kv_heads).to(
        device=runtime_device, dtype=dtype
    )
    if legacy:
        type(model)._ricursive_tf5_native_fix = "2026-11-05"
        compatibility_module().apply_ricursive_generate_patch(model)
    ids = torch.arange(3, 16, device=runtime_device).unsqueeze(0)
    expected = model(ids, use_cache=False).logits
    output = model(ids, use_cache=False, output_attentions=True)
    assert output.logits.dtype == dtype
    assert all(weights.dtype == dtype for weights in output.attentions)
    torch.testing.assert_close(output.logits, expected, atol=2e-3, rtol=2e-3)
    assert torch.equal(output.logits.argmax(-1), expected.argmax(-1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_norm_against_independent_fp32_oracle(runtime_device, dtype):
    model = make_model().to(device=runtime_device, dtype=dtype)
    normalization = model.model.layers[0].input_layernorm
    generator = torch.Generator(device=runtime_device).manual_seed(42)
    normalization.weight.copy_(torch.rand(32, generator=generator, device=runtime_device, dtype=dtype) + 0.5)
    inputs = torch.randn(3, 17, 32, generator=generator, device=runtime_device, dtype=dtype)
    reference = (
        inputs.float()
        * torch.rsqrt(inputs.float().square().mean(-1, keepdim=True) + normalization.eps)
        * normalization.weight.float()
    )
    compatibility_module()._patch_fused_rms_norm(model)
    actual = normalization(inputs)
    assert actual.dtype == dtype
    # Compare with the rounded oracle: BF16's representable output spacing is
    # larger than 2e-3 at this magnitude. No arithmetic tolerance is widened.
    torch.testing.assert_close(actual, reference.to(dtype), atol=2e-3, rtol=2e-3)
