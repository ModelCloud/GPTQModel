# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""NF12's forward must not query ``Tensor.is_inference()``.

Dynamo cannot trace ``tensor.is_inference()`` ("This graph break is
fundamental"), so the per-forward ``_as_normal_tensor`` wrapper that #5230 put
on ``dequantize_weight`` graph-broke every compiled CPU forward and made
``fullgraph=True`` fail, even though ``NF12Linear`` declares
``COMPILE_LEVEL.INDUCTOR`` for CPU (#5299).

The buffers are normalised when they are installed (``pack`` /
``load_quant_result``, and ``post_init`` for a checkpoint installed under
``InferenceMode``), so the forward does not need to ask.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
from gptqmodel.quantization.nf12 import quantize_nf12_weight


IN_FEATURES = 64
OUT_FEATURES = 16
EXP_BASE = 54

pytestmark = [pytest.mark.cpu]


def _module(bias: bool = True) -> NF12Linear:
    return NF12Linear(
        bits=12,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias,
        register_buffers=True,
    )


def _pack(module: NF12Linear) -> torch.Tensor:
    generator = torch.Generator().manual_seed(5299)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16, generator=generator)
    module.pack_original(torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True), None, None)
    module.clear_weight_cache()
    return weight


def _inputs() -> torch.Tensor:
    generator = torch.Generator().manual_seed(5299)
    return torch.randn(4, IN_FEATURES, dtype=torch.bfloat16, generator=generator)


def _forbid_inference_probe(monkeypatch) -> None:
    """Fail if anything asks a tensor whether it is an inference tensor."""

    def _boom(self):
        raise AssertionError(
            "tensor.is_inference() is not dynamo-traceable; NF12's forward must "
            "not call it (#5299)"
        )

    monkeypatch.setattr(torch.Tensor, "is_inference", _boom)


def test_forward_does_not_query_is_inference(monkeypatch):
    module = _module()
    _pack(module)
    module.post_init()
    module.eval()

    # Installed buffers are normal by now; the forward must not re-check.
    _forbid_inference_probe(monkeypatch)
    with torch.inference_mode():
        module(_inputs())


def test_dequantize_weight_does_not_query_is_inference(monkeypatch):
    module = _module()
    _pack(module)

    _forbid_inference_probe(monkeypatch)
    weight = module.dequantize_weight(dtype=torch.float32)

    assert weight.dtype == torch.float32
    assert weight.shape == (OUT_FEATURES, IN_FEATURES)


def test_post_init_normalises_buffers_installed_under_inference_mode(monkeypatch):
    module = _module()
    weight = torch.randn(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16)
    result = quantize_nf12_weight(weight, exp_base=EXP_BASE)

    with torch.inference_mode():
        module.nf12_packed = result.packed.clone()
        module.nf12_escapes = result.escapes.clone()
        module.nf12_exp_base = torch.tensor([result.exp_base], dtype=torch.uint8)
        module.bias = torch.zeros(OUT_FEATURES, dtype=torch.bfloat16)

    assert module.nf12_packed.is_inference()

    module.post_init()

    for name in ("nf12_packed", "nf12_escapes", "nf12_exp_base", "bias"):
        assert not getattr(module, name).is_inference(), name

    # A grad-mode forward still works from the normalised buffers.
    x = _inputs().requires_grad_(True)
    out = module(x)
    out.sum().backward()
    assert x.grad is not None
