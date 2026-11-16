# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression for #4370: ``dequantize_model`` must use the model compute dtype.

GPTQ checkpoints always store FP16 scales, and the loader restores them at
that storage dtype even when the model was loaded as BF16. Deriving the dense
dtype from ``scales.dtype`` therefore produced FP16 Linear weights inside a
BF16 model, and the first forward raised a dtype-mismatch error.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.nn_modules.qlinear import torch_aten_kernel as aten_module
from gptqmodel.nn_modules.qlinear import torch_fused as fused_module
from gptqmodel.nn_modules.qlinear.torch import TorchLinear, dequantize_model
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import (
    TorchAtenLinear,
    _has_local_int4pack_cpu_ops,
)
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.quantization import FORMAT


pytestmark = [pytest.mark.cpu]


def _aten_available() -> bool:
    return _has_local_int4pack_cpu_ops() and TorchAtenLinear.cached_validate_once()[0]


def _make_layer(cls, *, in_features=512, out_features=64, group_size=128):
    layer = cls(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
        compile=False,
    )
    layer.post_init()
    layer.qweight.zero_()
    layer.qzeros.zero_()
    # The checkpoint storage dtype is FP16 regardless of the model dtype.
    layer.scales.fill_(0.01)
    layer.g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    return layer


class _Holder(nn.Module):
    """A quantized projection preceded by a non-quantized layer at ``dtype``."""

    def __init__(self, layer, dtype):
        super().__init__()
        self.config = types.SimpleNamespace(quantization_config={"quant_method": "gptq"})
        self.dense = nn.Linear(layer.in_features, layer.in_features, bias=False, dtype=dtype)
        self.proj = layer

    def forward(self, x):
        return self.proj(self.dense(x))


_BACKENDS = [
    pytest.param(TorchLinear, dequantize_model, id="torch"),
    pytest.param(TorchFusedLinear, fused_module.dequantize_model, id="torch_fused"),
    pytest.param(
        TorchAtenLinear,
        aten_module.dequantize_model,
        id="torch_aten",
        marks=pytest.mark.skipif(
            not _aten_available(), reason="ATen CPU int4pack ops are unavailable"
        ),
    ),
]


@pytest.mark.parametrize(("kernel_cls", "dequantize"), _BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dequantize_model_uses_model_compute_dtype(kernel_cls, dequantize, dtype):
    layer = _make_layer(kernel_cls)
    assert layer.scales.dtype is torch.float16
    model = _Holder(layer, dtype)

    dequantize(model)

    assert type(model.proj) is nn.Linear
    assert model.proj.weight.dtype is dtype
    assert model.proj.bias is None

    x = torch.randn(3, layer.in_features, dtype=dtype)
    out = model(x)
    assert out.dtype is dtype
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(("kernel_cls", "dequantize"), _BACKENDS)
def test_dequantize_model_explicit_dtype_overrides_model(kernel_cls, dequantize):
    layer = _make_layer(kernel_cls)
    model = _Holder(layer, torch.bfloat16)

    dequantize(model, dtype=torch.float16)

    assert model.proj.weight.dtype is torch.float16


def _tiny_bf16_gptq_model():
    """A tiny BF16 Llama whose packed layers keep FP16 checkpoint scales."""

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    for name, linear in list(model.named_modules()):
        if not isinstance(linear, nn.Linear) or name == "lm_head":
            continue
        quant = TorchLinear(
            bits=4,
            group_size=32,
            sym=True,
            desc_act=False,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=False,
            format=FORMAT.GPTQ_V2,
            compile=False,
        )
        scales = torch.full((linear.out_features, linear.in_features // 32), 0.01)
        zeros = torch.full_like(scales, 8)
        quant.pack_original(linear, scales, zeros)
        model.set_submodule(name, quant)
    model.config.quantization_config = {"quant_method": "gptq"}
    return model


def test_dequantize_model_bf16_tiny_llama_forward():
    model = _tiny_bf16_gptq_model()
    assert model.model.layers[0].self_attn.q_proj.scales.dtype is torch.float16

    dequantize_model(model)

    q_proj = model.model.layers[0].self_attn.q_proj
    assert type(q_proj) is nn.Linear
    assert q_proj.weight.dtype is torch.bfloat16
    logits = model(input_ids=torch.tensor([[1, 3, 4, 5]]), use_cache=False).logits
    assert logits.dtype is torch.bfloat16
    assert torch.isfinite(logits).all()
