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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
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


# Below the smallest positive FP16 subnormal (2 ** -24). A writer that stored
# the bias at FP16 flushed this to zero.
SUBNORMAL_BIAS = 2.0**-30

# The bias cancels a dot product of 16 * 4375 = 70000. Both terms are above the
# FP16 range, so a bias narrowed to FP16 became ``inf`` and the output became
# non-finite even though every source value was finite.
CANCEL_INPUT = 4375.0
CANCEL_BIAS = -69999.0


def _single_linear_model(dtype: torch.dtype, weight: torch.Tensor, bias: float):
    block = torch.nn.Module()
    block.proj = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=True).to(dtype)
    with torch.no_grad():
        block.proj.weight.copy_(weight.to(dtype))
        block.proj.bias.fill_(bias)

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.decoder = torch.nn.Module()
    model.model.decoder.layers = torch.nn.ModuleList([block])
    return model


def _reload(dtype: torch.dtype, weight: torch.Tensor, out: str):
    fresh = _single_linear_model(dtype, torch.zeros_like(weight), 0.0)
    bq.load_native_checkpoint_into_dense_model(fresh, out)
    return fresh.model.decoder.layers[0].proj


# An off-grid weight for the dense-fallback payload assertions. Only the
# payload is checked for that path: a dense layer is not yet listed in
# ``blockfloat.json``, so it is not reachable through the reader.
OFF_GRID_WEIGHT = torch.full((1, 16), 1.0001)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_bias_below_fp16_subnormal_survives(tmp_path, dtype):
    """A tiny non-zero bias must not be flushed to zero by storage.

    ``2 ** -30`` is representable in both bf16 and fp32 but not in FP16, where
    the smallest positive subnormal is ``2 ** -24``.
    """

    weight = bq.quantize_bf4(torch.ones(1, 16))
    model = _single_linear_model(dtype, weight, SUBNORMAL_BIAS)
    source_bias = model.model.decoder.layers[0].proj.bias.detach().clone()
    assert source_bias.abs().min().item() > 0.0

    out = str(tmp_path / f"subnormal-{dtype}")
    bq.save_blockfloat_checkpoint(model, out, bits=4)
    saved = bq._load_blockfloat_payload(out)["0.proj.bias"]
    assert saved.dtype is dtype
    assert torch.equal(saved, source_bias)
    assert saved.abs().min().item() > 0.0

    restored = _reload(dtype, weight, out).bias.detach()
    assert torch.equal(restored, source_bias)
    assert restored.abs().min().item() > 0.0


def test_dense_fallback_bias_is_not_narrowed(tmp_path):
    """The dense path stores the bias at source precision, not FP16.

    Reachability of a dense layer through the manifest is out of scope here; the
    assertion is that the tensor the writer emits is not narrowed.
    """

    model = _single_linear_model(torch.float32, OFF_GRID_WEIGHT, SUBNORMAL_BIAS)
    out = str(tmp_path / "dense-subnormal")
    bq.save_blockfloat_checkpoint(model, out, bits=4)
    payload = bq._load_blockfloat_payload(out)

    assert "0.proj.blockfloat_codes" not in payload
    assert payload["0.proj.weight"].dtype is torch.float32
    assert payload["0.proj.bias"].dtype is torch.float32
    assert payload["0.proj.bias"].abs().min().item() > 0.0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_bias_cancellation_keeps_forward_finite(tmp_path, dtype):
    """A reloaded model must reproduce the saved model's finite output.

    Every source value is finite: the dot product is ``16 * 4375 = 70000`` and
    the bias is ``-69999``, so the layer output is near zero even though both
    terms exceed the FP16 range. With a narrowed bias the output was ``-inf``.
    """

    weight = bq.quantize_bf4(torch.ones(1, 16))
    model = _single_linear_model(dtype, weight, CANCEL_BIAS)
    source = model.model.decoder.layers[0].proj
    x = torch.full((1, 16), CANCEL_INPUT, dtype=dtype)
    with torch.no_grad():
        expected = source(x)
    assert torch.isfinite(expected).all()
    # Cancellation, not magnitude: far below the 70000 the terms reach.
    assert expected.abs().max().item() < 0.01 * 16 * CANCEL_INPUT

    out = str(tmp_path / f"cancel-{dtype}")
    bq.save_blockfloat_checkpoint(model, out, bits=4)
    saved_bias = bq._load_blockfloat_payload(out)["0.proj.bias"]
    assert saved_bias.dtype is dtype
    assert torch.isfinite(saved_bias).all()
    assert torch.equal(saved_bias, source.bias.detach())

    restored = _reload(dtype, weight, out)
    assert torch.equal(restored.bias.detach(), source.bias.detach())
    with torch.no_grad():
        actual = restored(x)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def test_reader_loads_a_stored_bias_at_the_target_dtype(tmp_path):
    """A stored bias is converted straight to the declared target dtype.

    The payload keeps the source precision; a target model built at another
    precision lands there without a round trip through FP16.
    """

    weight = bq.quantize_bf4(torch.ones(1, 16))
    source = _single_linear_model(torch.float32, weight, 70000.0)
    out = str(tmp_path / "declared-dtype")
    bq.save_blockfloat_checkpoint(source, out, bits=4)

    payload = bq._load_blockfloat_payload(out)
    assert payload["0.proj.bias"].dtype is torch.float32
    assert torch.isfinite(payload["0.proj.bias"]).all()

    restored = _reload(torch.bfloat16, weight, out)
    assert restored.weight.dtype is torch.bfloat16
    assert restored.bias.dtype is torch.bfloat16
    assert torch.equal(restored.bias.detach(), payload["0.proj.bias"].to(torch.bfloat16))
    assert torch.isfinite(restored.bias.detach()).all()
