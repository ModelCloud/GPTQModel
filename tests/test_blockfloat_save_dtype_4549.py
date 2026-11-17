# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``save_blockfloat_checkpoint`` must not narrow dense tensors to FP16 (#4549).

Only tensors that are actually block-float encoded may change representation.
Off-grid decoder linears are written as dense floats and their biases (on both
the dense and the encoded path) are stored as-is, so a bf16/fp32 value above
the fp16 range survives the save instead of becoming ``inf``.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.utils import tt_blockfloat_quant as bq


# Larger than torch.finfo(torch.float16).max (65504), and exactly representable
# in both bfloat16 and float32.
OFF_GRID_VALUE = 99840.0


def _model(dtype: torch.dtype) -> torch.nn.Module:
    """A one-block decoder model with an off-grid and a grid linear."""

    block = torch.nn.Module()
    block.proj = torch.nn.Linear(64, 32, bias=True).to(dtype)
    block.grid = torch.nn.Linear(64, 32, bias=True).to(dtype)
    with torch.no_grad():
        # Random values are not fixed points of the block-float grid, so this
        # layer takes the dense fallback.
        weight = (torch.randn(32, 64) * 0.02).to(dtype)
        weight[0, 0] = OFF_GRID_VALUE
        block.proj.weight.copy_(weight)
        block.proj.bias.fill_(OFF_GRID_VALUE)
        # This one is on the grid and must still be encoded as codes.
        block.grid.weight.copy_(bq.quantize_bf4(torch.randn(32, 64) * 0.02).to(dtype))
        block.grid.bias.fill_(OFF_GRID_VALUE)

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.decoder = torch.nn.Module()
    model.model.decoder.layers = torch.nn.ModuleList([block])
    return model


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
def test_off_grid_dense_weight_and_bias_keep_storage_dtype(tmp_path, dtype):
    """The dense fallback keeps the module dtype and payload values."""

    model = _model(dtype)
    proj = model.model.decoder.layers[0].proj
    source_weight = proj.weight.detach().clone()
    source_bias = proj.bias.detach().clone()

    out = str(tmp_path / f"off-grid-{dtype}")
    bq.save_blockfloat_checkpoint(model, out, bits=4)
    payload = bq._load_blockfloat_payload(out)

    # The off-grid layer is dense, not encoded.
    assert "0.proj.blockfloat_codes" not in payload

    saved_weight = payload["0.proj.weight"]
    assert saved_weight.dtype is dtype
    assert torch.equal(saved_weight, source_weight)
    assert torch.isfinite(saved_weight).all()
    assert saved_weight.max().item() == OFF_GRID_VALUE

    # The old writer dropped this key entirely after `continue`.
    saved_bias = payload["0.proj.bias"]
    assert saved_bias.dtype is dtype
    assert torch.equal(saved_bias, source_bias)
    assert torch.isfinite(saved_bias).all()
    assert saved_bias.max().item() == OFF_GRID_VALUE


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
def test_grid_layer_bias_roundtrips_in_source_dtype(tmp_path, dtype):
    """Control: encoded layers still work and their bias reloads unchanged."""

    model = _model(dtype)
    grid = model.model.decoder.layers[0].grid
    source_weight = grid.weight.detach().clone()
    source_bias = grid.bias.detach().clone()

    out = str(tmp_path / f"grid-{dtype}")
    bq.save_blockfloat_checkpoint(model, out, bits=4)
    payload = bq._load_blockfloat_payload(out)

    # The on-grid layer is still stored as codes, never as a dense float.
    assert "0.grid.blockfloat_codes" in payload
    assert "0.grid.weight" not in payload
    assert payload["0.grid.bias"].dtype is dtype
    assert torch.equal(payload["0.grid.bias"], source_bias)

    fresh = _model(dtype)
    bq.load_native_checkpoint_into_dense_model(fresh, out)
    restored = fresh.model.decoder.layers[0].grid
    assert restored.weight.dtype is dtype
    assert torch.equal(restored.weight.detach(), source_weight)
    assert restored.bias.dtype is dtype
    assert torch.equal(restored.bias.detach(), source_bias)
    assert torch.isfinite(restored.bias).all()
