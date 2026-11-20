# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #5305: the fused NF12 CUDA/RDNA4 kernels register no
Autograd formula.

``nf12::linear`` / ``nf12_hip::linear`` still return a tensor whose
``grad_fn`` is ``torch::autograd::WarnNotImplemented``, so a grad-mode forward
looked differentiable but produced ``None`` for ``x.grad``. The layer must
route any activation that needs a gradient through the differentiable
decode + matmul path, while inference keeps the fused kernel.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
from gptqmodel.utils import nf12 as nf12_runtime
from gptqmodel.utils.backend import BACKEND


_IN_FEATURES = 64
_OUT_FEATURES = 16
_BITS = 12
_EXP_BASE = 54


def _module() -> NF12Linear:
    return NF12Linear(
        bits=_BITS,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        bias=True,
        backend=BACKEND.NF12,
        pack_dtype=torch.int32,
        exp_base=_EXP_BASE,
    )


def _pack(module: NF12Linear) -> None:
    torch.manual_seed(5305)
    linear = torch.nn.Linear(_IN_FEATURES, _OUT_FEATURES, bias=True).to(torch.bfloat16)
    module.pack_original(linear)


def _cuda_module() -> NF12Linear:
    module = _module()
    _pack(module)
    module = module.cuda()
    module.post_init()
    return module


cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_required
def test_cuda_bf16_forward_keeps_the_input_gradient():
    module = _cuda_module()
    x = torch.randn(4, _IN_FEATURES, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    out = module(x)

    assert out.grad_fn is not None
    assert "WarnNotImplemented" not in str(out.grad_fn)
    out.float().sum().backward()
    assert x.grad is not None

    # d(sum(out))/dx is grad_out @ W with grad_out = ones.
    weight = module.dequantize_weight(device="cuda", dtype=torch.float32)
    expected = torch.ones_like(out, dtype=torch.float32) @ weight
    torch.testing.assert_close(x.grad.float(), expected, rtol=2e-2, atol=2e-2)


@cuda_required
@pytest.mark.skipif(torch.version.hip is not None, reason="CUDA dispatch only")
def test_native_linear_is_skipped_only_when_a_gradient_is_needed(monkeypatch):
    module = _cuda_module()
    calls: list[bool] = []

    def fake_linear(x, packed, escapes, base, bias):
        calls.append(torch.is_grad_enabled())
        return torch.zeros(x.shape[:-1] + (module.out_features,), dtype=x.dtype, device=x.device)

    # Pretend the fused kernel is usable; `decode*` raise so `dequantize_weight`
    # takes its Torch-codec fallback instead of a missing compiled op.
    def unavailable_decode(*args, **kwargs):
        raise RuntimeError("native decode disabled by test")

    monkeypatch.setattr(nf12_runtime, "available", lambda *args, **kwargs: True)
    monkeypatch.setattr(nf12_runtime, "linear", fake_linear)
    monkeypatch.setattr(nf12_runtime, "decode", unavailable_decode)
    monkeypatch.setattr(nf12_runtime, "decode_fp32", unavailable_decode)

    x = torch.randn(2, _IN_FEATURES, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = module(x)
    assert calls == []
    out.float().sum().backward()
    assert x.grad is not None

    with torch.no_grad():
        module(x)
    assert calls == [False]
