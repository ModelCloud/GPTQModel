# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #5630: Q8_K must not erase tiny FP32-scaled weights.

Q8_K stores an FP32 block scale, so a block whose weights are below the FP16
subnormal floor survives packing but rounds to zero when the decoded weights
are narrowed to FP16. The FP16 dispatch guard must decode Q8_K wide for such a
block (an overflow-only check misses the erasure), while all-zero and ordinary
blocks keep the narrow path.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


def _build(value: float, *, bias: bool = False) -> tuple[GGUFTorchLinear, torch.nn.Linear]:
    src = torch.nn.Linear(256, 1, bias=bias)
    src.weight.data.fill_(value)
    if bias:
        src.bias.data.zero_()
    module = GGUFTorchLinear(
        bits="q8_k",
        in_features=256,
        out_features=1,
        group_size=-1,
        sym=True,
        desc_act=False,
        bias=bias,
    )
    module.pack_original(src, None, None)
    return module, src


def _activation() -> torch.Tensor:
    x = torch.zeros(1, 256, dtype=torch.float16)
    x[0, 0] = 32768
    return x


@pytest.mark.parametrize("bias", [False, True])
def test_q8_k_fp16_keeps_tiny_weights(bias: bool) -> None:
    module, src = _build(2.0**-26, bias=bias)
    x = _activation()
    expected = src(x.float()).item()

    assert module(x).item() == pytest.approx(expected, rel=1e-6)
    # One activation dtype per module (contracts/kernel/forward-dtype.md):
    # check FP32 activations on a separately built module.
    fp32_module, _ = _build(2.0**-26, bias=bias)
    assert fp32_module(x.float()).item() == pytest.approx(expected, rel=1e-6)


def test_q8_k_fp16_guard_widens_only_for_tiny_nonzero_blocks() -> None:
    tiny, _ = _build(2.0**-26)
    zero, _ = _build(0.0)
    normal, _ = _build(1.0)

    assert tiny._use_fp32_gguf_weight(torch.float16) is True
    assert zero._use_fp32_gguf_weight(torch.float16) is False
    assert normal._use_fp32_gguf_weight(torch.float16) is False
    # FP32 activations never need the wide temporary.
    assert tiny._use_fp32_gguf_weight(torch.float32) is False


def test_q8_k_fp16_fused_dispatch_matches_dense() -> None:
    module, _ = _build(2.0**-26)
    module.eval()
    module.gguf_fused_cpu_max_rows = 1
    module.gguf_fused_cpu_min_matrix_elements = 0
    x = _activation()

    assert module._should_use_fused_k_forward(x) is True
    assert module(x).item() == pytest.approx(0.00048828125, rel=1e-6)


def test_q8_k_fp16_zero_block_stays_zero() -> None:
    module, _ = _build(0.0)
    assert module(_activation()).item() == 0.0
