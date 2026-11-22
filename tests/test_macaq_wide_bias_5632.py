# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A wide MACaQ bias must reach the FP32 reduction unrounded (#5632).

``-257`` is exact in FP32 but rounds to ``-256`` in BF16, and ``-256`` no
longer cancels the packed unit weights' ``256`` product.  The bias therefore
keeps its stored dtype through packing and checkpoint loading, and the forward
adds it after the FP32 reduction instead of narrowing it to the activation
dtype first.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear
from gptqmodel.quantization.macaq import macaq_quantize_weight
from gptqmodel.utils import macaq_triton
from gptqmodel.utils.model import load_checkpoint_in_model_then_tie_weights


IN_FEATURES = 32
OUT_FEATURES = 1
WIDE_BIAS = -257.0


def _quantize(weight: float, bias: float):
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)
    linear.weight.data.fill_(weight)
    linear.bias.data.fill_(bias)
    result = macaq_quantize_weight(
        linear.weight.detach(),
        torch.eye(IN_FEATURES),
        IN_FEATURES,
        rotation="none",
        pre_mode="none",
        do_flip=False,
        exp_screen=False,
    )
    return linear, result


def _layer(dtype: torch.dtype) -> MacaqLinear:
    return MacaqLinear(
        bits=4,
        group_size=32,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        rotation="none",
        pre_mode="none",
        activation_precision="w4a16",
        dtype=dtype,
        bias=True,
    )


def _pack(dtype: torch.dtype):
    linear, result = _quantize(1.0, WIDE_BIAS)
    layer = _layer(dtype)
    layer.pack(linear, macaq_result=result)
    return layer, linear


def _cancelling_input() -> torch.Tensor:
    x = torch.zeros(1, IN_FEATURES, dtype=torch.bfloat16)
    x[0, 0] = 256
    return x


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pack_keeps_the_source_bias_dtype(dtype):
    layer, linear = _pack(dtype)
    x = _cancelling_input()

    assert layer.bias.dtype == torch.float32
    assert layer.bias.item() == WIDE_BIAS
    with torch.inference_mode():
        assert layer(x).item() == -1.0
    assert linear(x.float()).item() == -1.0


class _Model(nn.Module):
    def __init__(self, layer: MacaqLinear):
        super().__init__()
        self.proj = layer

    def tie_weights(self):
        pass


def test_checkpoint_round_trip_restores_the_wide_bias(tmp_path):
    source, _ = _pack(torch.bfloat16)
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {name: tensor.contiguous() for name, tensor in _Model(source).state_dict().items()},
        str(checkpoint),
        metadata={"format": "pt"},
    )

    target = _Model(_layer(torch.bfloat16))
    load_checkpoint_in_model_then_tie_weights(
        target,
        checkpoint=str(checkpoint),
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )

    assert target.proj.bias.dtype == torch.float32
    assert target.proj.bias.item() == WIDE_BIAS
    with torch.inference_mode():
        assert target.proj(_cancelling_input()).item() == -1.0


def test_triton_w4a16_tail_adds_a_wide_bias_in_fp32(monkeypatch):
    weight = torch.ones(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16)
    # Exercise the Torch epilogue on CPU with the Triton decode result supplied.
    monkeypatch.setattr(macaq_triton, "TRITON_AVAILABLE", True)
    monkeypatch.setattr(macaq_triton, "dequant_w4_to_bf16", lambda packed, scales: weight)
    bias = torch.tensor([WIDE_BIAS], dtype=torch.float32)

    out = macaq_triton.mxfp4_w4a16_linear(
        _cancelling_input(),
        torch.zeros((OUT_FEATURES, IN_FEATURES // 2), dtype=torch.uint8),
        torch.zeros((OUT_FEATURES, IN_FEATURES // 32), dtype=torch.uint8),
        bias=bias,
        compute_dtype=torch.bfloat16,
    )

    assert out.item() == -1.0
