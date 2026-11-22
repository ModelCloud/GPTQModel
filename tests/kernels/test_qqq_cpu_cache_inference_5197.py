# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A QQQ CPU packed cache must not be built out of inference tensors (#5197).

``QQQTorchLinear`` lazily builds its AVX512 ``qqq_int4`` operands on the first
forward and keeps ``(packed, s_group, s_channel)`` on the module. Every
``generate()``/eval loop runs that first forward inside ``torch.inference_mode``
and the module then dropped the checkpoint ``B``/scales, so the cache was the
only resident weight -- made of inference tensors. The next grad-mode forward
raised ``RuntimeError: Inference tensors cannot be saved for backward`` at
``accum * s_channel``.

The builder now runs under ``torch.inference_mode(False)`` + ``torch.no_grad()``
so the cached operands are normal tensors for either ordering and for a module
packed inside the quantization loop's ``inference_mode``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear, _qqq_cpu_kernel


IN_FEATURES = 256
OUT_FEATURES = 128
GROUP_SIZE = 128
DTYPE = torch.float16

pytestmark = [pytest.mark.cpu]


def _build_module() -> QQQTorchLinear:
    torch.manual_seed(5197)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=DTYPE)
    linear.weight.data.normal_(0, 0.08)
    groups = IN_FEATURES // GROUP_SIZE
    scales = torch.rand((OUT_FEATURES, groups), dtype=DTYPE) * 0.02 + 0.01
    s_extra = torch.rand(OUT_FEATURES, dtype=torch.float32) * 0.001 + 0.001

    module = QQQTorchLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    module.pack(linear, scales, s_extra)
    module.post_init()
    return module.eval()


def _require_packed_cpu_kernel() -> None:
    """The packed path (and the bug) needs the AVX512 extension build."""

    if _qqq_cpu_kernel() is None:
        pytest.skip("QQQ packed AVX512 CPU kernel is unavailable on this host")


def _activation() -> torch.Tensor:
    return torch.randn(4, IN_FEATURES, dtype=DTYPE, requires_grad=True)


def test_inference_forward_then_grad_forward_matches():
    """The reported order: ``generate()`` first, then a training forward."""

    _require_packed_cpu_kernel()
    module = _build_module()
    x = _activation()

    with torch.inference_mode():
        inference_out = module(x).clone()
    assert module._torch_packed_cache is not None
    assert module._torch_dropped_checkpoint is not None

    grad_out = module(x)
    assert grad_out.requires_grad
    grad_out.sum().backward()
    assert x.grad is not None

    torch.testing.assert_close(grad_out.detach(), inference_out, rtol=0, atol=0)


def test_inference_built_cache_holds_normal_tensors():
    """Mechanism guard: a cache built inside ``inference_mode`` is normal."""

    module = _build_module()

    with torch.inference_mode():
        module._packed_weight_for_cpu()

    assert module._torch_dropped_checkpoint is not None
    packed, s_group, s_channel = module._torch_packed_cache
    assert not any(tensor.is_inference() for tensor in (packed, s_group, s_channel)), (
        "the cached CPU kernel operands outlive the inference-mode build"
    )


def test_grad_forward_after_in_session_quantize_matches():
    """Quantize packs under ``inference_mode``; a later grad forward must work."""

    _require_packed_cpu_kernel()
    with torch.inference_mode():
        module = _build_module()
    assert module.B.is_inference(), "fixture did not pack under inference mode"
    assert module.s_channel.is_inference()

    x = _activation()
    grad_out = module(x)
    grad_out.sum().backward()
    assert x.grad is not None

    with torch.inference_mode():
        inference_out = module(x).clone()

    torch.testing.assert_close(grad_out.detach(), inference_out, rtol=0, atol=0)
