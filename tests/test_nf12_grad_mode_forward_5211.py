# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Regression for #5211: NF12 must forward with autograd enabled.

``NF12Linear.dequantize_weight()`` used to be decorated with
``torch.inference_mode()`` and ``pack_original()`` / ``load_quant_result()``
installed the packed buffers inside one. The decoded matrix was therefore an
inference tensor, and the wide forward path's ``torch.mm`` raised
``Inference tensors cannot be saved for backward`` as soon as the activation
required grad. CPU always takes the wide path, so every CPU forward in grad
mode failed; CUDA fp16 did too.

The fix keeps the installed buffers and the decoded operand normal tensors, so
a plain (not ``no_grad``) forward works and matches the inference result
exactly.
"""

from __future__ import annotations

import torch

from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
from gptqmodel.quantization.nf12 import quantize_nf12_weight
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


def _pack(module: NF12Linear) -> torch.Tensor:
    torch.manual_seed(5211)
    linear = torch.nn.Linear(_IN_FEATURES, _OUT_FEATURES, bias=True).to(torch.bfloat16)
    module.pack_original(linear)
    return linear.weight.detach().clone()


def _inputs() -> torch.Tensor:
    generator = torch.Generator().manual_seed(5211)
    return torch.randn(4, _IN_FEATURES, dtype=torch.bfloat16, generator=generator)


def test_packing_leaves_normal_buffers_and_allows_a_grad_mode_forward():
    module = _module()
    _pack(module)

    # Packing runs under InferenceMode; the installed buffers must still be
    # usable from a graph-tracking forward.
    for name in ("nf12_packed", "nf12_escapes", "nf12_exp_base", "bias"):
        assert not getattr(module, name).is_inference(), name

    x = _inputs().requires_grad_(True)
    with torch.inference_mode():
        expected = module(x.detach())

    actual = module(x)

    assert torch.equal(actual, expected)
    assert actual.requires_grad
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_buffers_installed_inside_inference_mode_still_forward_in_grad_mode():
    """A loader can install a checkpoint under ``inference_mode`` (#5211)."""

    module = _module()
    weight = _pack(module)
    result = quantize_nf12_weight(weight, exp_base=_EXP_BASE)

    with torch.inference_mode():
        # `clone()` inside InferenceMode is what a loader installing a
        # checkpoint there hands the module.
        module.nf12_packed = result.packed.clone()
        module.nf12_escapes = result.escapes.clone()
        module.nf12_exp_base = torch.tensor([result.exp_base], dtype=torch.uint8)
        module.bias = torch.zeros(_OUT_FEATURES, dtype=torch.bfloat16)
        module.clear_weight_cache()

    assert module.nf12_packed.is_inference()

    x = _inputs().requires_grad_(True)
    with torch.inference_mode():
        expected = module(x.detach())

    actual = module(x)

    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dequantize_weight_is_not_an_inference_tensor():
    module = _module()
    _pack(module)

    weight = module.dequantize_weight(dtype=torch.float32)

    assert not weight.is_inference()
    x = torch.randn(2, _IN_FEATURES, dtype=torch.float32, requires_grad=True)
    # This is the operation that used to raise in the forward.
    out = torch.mm(x, weight.t())
    out.sum().backward()
    assert x.grad is not None
