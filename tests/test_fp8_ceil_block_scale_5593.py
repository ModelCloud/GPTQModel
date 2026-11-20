# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""FP8 ceil-blocked checkpoints must keep the compact per-block scale (#5593).

A checkpoint whose 128x128 block grid does not divide the weight shape was
expanded to a dense ``N x K`` FP32 ``weight_scale_inv`` with a ``(1, 1)`` decode
plan, keeping an FP32 tensor four times the size of the FP8 weight for the
module's whole lifetime. The wrapper now stores the checkpoint's
``ceil(N / br) x ceil(K / bc)`` grid and decodes with the real block size.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import gptqmodel.models.base as base_module
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear


def _harness(*, block_size, semantics="multiplier"):
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = SimpleNamespace(
        config=SimpleNamespace(
            quantization_config={
                "quant_method": "fp8",
                "format": "float8_e4m3fn",
                "weight_block_size": list(block_size),
                "weight_scale_semantics": semantics,
            }
        )
    )
    return harness


def _build(n, k, scale_shape, *, block=(128, 128), semantics="multiplier"):
    torch.manual_seed(0)
    weight = (torch.randn(n, k) * 0.25).to(torch.float8_e4m3fn)
    scale = torch.full(scale_shape, 0.5, dtype=torch.float32)
    submodule = nn.Linear(k, n, bias=False)
    module = base_module.BaseQModel._build_fp8_forward_module(
        _harness(block_size=block, semantics=semantics),
        target_submodule=submodule,
        checkpoint_tensors={"weight": weight, "weight_scale_inv": scale},
        device=torch.device("cpu"),
        target_dtype=torch.bfloat16,
    )
    assert isinstance(module, TorchFP8Linear)
    return weight, scale, module


def _reference_weight(weight, scale, n, k, *, block=(128, 128), semantics="multiplier"):
    """Independent per-element decode from the checkpoint grid, in FP64."""

    expanded = scale.double()
    expanded = expanded.repeat_interleave(block[0], dim=0)
    expanded = expanded.repeat_interleave(block[1], dim=1)[:n, :k]
    source = weight.double()
    return source * expanded if semantics == "multiplier" else source / expanded


@pytest.mark.parametrize(
    ("n", "k", "scale_shape"),
    [
        (576, 256, (5, 2)),
        # DeepSeek-V3 kv_a_proj_with_mqa: 5 * 128 >= 576 and 56 * 128 == 7168.
        (576, 7168, (5, 56)),
    ],
)
def test_ceil_blocked_scale_stays_compact(n, k, scale_shape):
    weight, scale, module = _build(n, k, scale_shape)

    assert module.weight_block_size == (128, 128)
    assert module.smooth_block_size() == 128
    # The checkpoint's own grid, not an N x K expansion with a (1, 1) plan.
    assert tuple(module.weight_scale_inv.shape) == scale_shape
    assert module.weight_scale_inv.dtype is torch.float32
    scale_bytes = module.weight_scale_inv.numel() * module.weight_scale_inv.element_size()
    assert scale_bytes == scale.numel() * 4
    assert scale_bytes * 4 < weight.numel(), "scale must not rival the FP8 weight"

    expected = _reference_weight(weight, scale, n, k)
    decoded = module.dequantize_weight(device="cpu", dtype=torch.float32).T
    torch.testing.assert_close(decoded.double(), expected, rtol=1e-6, atol=0)

    x = torch.randn(2, k, dtype=torch.float32)
    out = module(x)
    # The forward must consume exactly the verified decoded weight.
    torch.testing.assert_close(out, torch.matmul(x, decoded.T), rtol=0, atol=0)


def test_divisible_control_keeps_its_compact_grid():
    """The exact-divisible layout must not change while ceil grids are added."""

    n, k, scale_shape = 512, 256, (4, 2)
    weight, scale, module = _build(n, k, scale_shape)

    assert module.weight_block_size == (128, 128)
    assert tuple(module.weight_scale_inv.shape) == scale_shape
    expected = _reference_weight(weight, scale, n, k)
    decoded = module.dequantize_weight(device="cpu", dtype=torch.float32).T
    torch.testing.assert_close(decoded.double(), expected, rtol=1e-6, atol=0)


def test_ceil_blocked_inverse_semantics_divides_the_scale():
    n, k, scale_shape = 576, 256, (5, 2)
    weight, scale, module = _build(n, k, scale_shape, semantics="inverse")

    assert module.weight_scale_semantics == "inverse"
    assert tuple(module.weight_scale_inv.shape) == scale_shape
    expected = _reference_weight(weight, scale, n, k, semantics="inverse")
    decoded = module.dequantize_weight(device="cpu", dtype=torch.float32).T
    torch.testing.assert_close(decoded.double(), expected, rtol=1e-6, atol=0)


def test_ceil_blocked_construction_allocates_the_ceil_grid():
    """A 576-row block layer is constructible without a pre-expanded scale."""

    module = TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=7168,
        out_features=576,
        bias=False,
        pack_dtype=torch.int32,
        format="float8_e4m3fn",
        weight_scale_method="block",
        weight_block_size=(128, 128),
    )

    assert tuple(module.weight_scale_inv.shape) == (5, 56)
    assert tuple(module.weight.shape) == (576, 7168)
