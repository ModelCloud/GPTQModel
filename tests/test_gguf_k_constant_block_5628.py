# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #5628: constant K-quant blocks must survive packing.

A nonzero block whose FP32 superblock base rounds to zero in FP16 used to be
stored with scale 0, erasing the whole block. The packer now clamps the base to
the smallest positive FP16 subnormal and rebuilds the scale/min codes against
the stored base.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import (
    GGUFTorchLinear,
    _dequantize_gguf_tensor_numpy,
    _fallback_gguf_quantize,
)


K_QTYPES = ("Q4_K", "Q5_K", "Q6_K")
K_BITS = {"Q4_K": "q4_k_m", "Q5_K": "q5_k_m", "Q6_K": "q6_k"}


@pytest.mark.parametrize("qtype", K_QTYPES)
@pytest.mark.parametrize("value", [2.0**-16, 2.0**-20])
def test_constant_k_block_keeps_decoded_bytes(qtype: str, value: float) -> None:
    weight = np.full((1, 256), value, dtype=np.float32)

    packed = _fallback_gguf_quantize(weight, qtype)
    assert np.isfinite(packed.view(np.uint8)).all()

    decoded = _dequantize_gguf_tensor_numpy(packed, qtype)
    np.testing.assert_array_equal(decoded, np.full_like(decoded, value))


@pytest.mark.parametrize("qtype", K_QTYPES)
@pytest.mark.parametrize("value", [2.0**-16, 2.0**-20])
def test_constant_k_block_pack_forward(qtype: str, value: float) -> None:
    linear = torch.nn.Linear(256, 1, bias=False)
    linear.weight.data.fill_(value)

    module = GGUFTorchLinear(
        bits=K_BITS[qtype],
        in_features=256,
        out_features=1,
        group_size=-1,
        sym=True,
        desc_act=False,
    )
    module.pack(linear, None, None)

    x = torch.zeros(1, 256)
    x[:, 0] = 1.0 / value
    assert module(x).item() == pytest.approx(1.0, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("qtype", K_QTYPES)
def test_zero_k_block_control(qtype: str) -> None:
    weight = np.zeros((1, 256), dtype=np.float32)

    packed = _fallback_gguf_quantize(weight, qtype)
    assert np.isfinite(packed.view(np.uint8)).all()

    decoded = _dequantize_gguf_tensor_numpy(packed, qtype)
    assert np.count_nonzero(decoded) == 0
