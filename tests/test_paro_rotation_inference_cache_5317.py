# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paro's bf16 rotation cache must not be built out of inference tensors (#5317).

``ParoLinear._ensure_rotation_runtime_dtype`` caches dtype-converted copies of
``theta``/``channel_scales`` for the bf16 auto-cache. Before #5317 those copies
were built in the caller's context, so the first bf16 forward under
``torch.inference_mode()`` (``generate()``, an eval warm-up, a server prefill)
cached inference tensors. The cache is then reused on every later forward,
including grad-mode ones, where the fused rotation saves ``theta`` for backward
and raises ``RuntimeError: Inference tensors cannot be saved for backward``.

The regression is the transition, not construction: warm the cache with one
inference-mode forward, then assert the cached entries are ordinary tensors and
run a grad-mode forward+backward.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.paro import ParoLinear


IN_FEATURES = 256
OUT_FEATURES = 8
GROUP_SIZE = 128
KROT = 8


def _module(device: str) -> ParoLinear:
    torch.manual_seed(5317)
    module = ParoLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
        krot=KROT,
    ).to(device).eval()
    with torch.no_grad():
        module.qweight.fill_(0x33333333)
        module.qzeros.fill_(0x22222222)
        module.scales.fill_(1)
        # A real (non-identity) rotation is what makes the runtime cache
        # load-bearing; the identity shortcut returns before building it.
        module.theta.normal_(0, 0.1)
    module.post_init()
    module.clear_weight_cache()
    assert not module._rotation_identity
    return module


def _activation(device: str) -> torch.Tensor:
    return torch.randn(2, 3, IN_FEATURES, device=device, dtype=torch.bfloat16)


def _assert_cache_is_normal(module: ParoLinear) -> None:
    assert module._runtime_theta is not None
    assert module._runtime_channel_scales is not None
    assert not module._runtime_theta.is_inference(), (
        "the bf16 rotation cache outlives the inference-mode build"
    )
    assert not module._runtime_channel_scales.is_inference(), (
        "the bf16 rotation cache outlives the inference-mode build"
    )


def test_inference_forward_does_not_cache_inference_tensors():
    """Mechanism guard: the reported order must leave an ordinary cache."""

    module = _module("cpu")
    with torch.inference_mode():
        module(_activation("cpu"))
    _assert_cache_is_normal(module)


def test_grad_forward_after_inference_forward_keeps_the_warm_cache():
    """A grad-mode forward after an eval forward must reuse the normal cache."""

    module = _module("cpu")
    with torch.inference_mode():
        module(_activation("cpu"))
    _assert_cache_is_normal(module)
    cached_theta = module._runtime_theta

    x = _activation("cpu").requires_grad_(True)
    out = module(x)
    out.float().sum().backward()
    assert x.grad is not None
    # The warm cache is still the one in use, so eval and training do not
    # convert the rotation on every call.
    assert module._runtime_theta is cached_theta


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the fused Paro rotation autograd wrapper is CUDA-only",
)
def test_cuda_bf16_backward_after_inference_forward():
    """End-to-end: the fused autograd wrapper saves the cached rotation."""

    device = "cuda:0"
    module = _module(device)
    with torch.inference_mode():
        module(_activation(device))
    _assert_cache_is_normal(module)

    x = _activation(device).requires_grad_(True)
    module(x).float().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
