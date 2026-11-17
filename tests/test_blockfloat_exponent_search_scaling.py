# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The block-float exponent search must be scale invariant.

``quantize_blockfloat`` picks the shared exponent of each 16-value block by
comparing the squared error of ``floor(log2(block_max))`` and its two
neighbours.  That comparison used to run on the raw block, so its magnitude
decided whether the search worked at all: for a block near ``2**80`` every
candidate error summed to ``inf`` and no candidate compared ``< inf``, while for
a block near ``2**-80`` every error summed to ``0`` and the strict ``<`` never
replaced the base candidate.  Both cases kept ``floor(log2(block_max))`` even
when a neighbour was measurably closer, so the same 16 values placed at a
different scale quantized to a different exponent and different weights.

The oracle here enumerates every allowed code for every candidate exponent in
float64, independently of the implementation's arithmetic, so it is not a
restatement of the code under test.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.tt_blockfloat_quant import (
    BLOCKFLOAT_BLOCK,
    decompose_blockfloat,
    quantize_blockfloat,
    recompose_blockfloat,
    stored_mantissa_bits,
)


BITS = (2, 4, 8)
# Two orders of magnitude below and above the range where float32 squared
# errors are representable: ``2**80`` overflows the sum, ``2**-80`` underflows.
SCALES = (-80, -40, 0, 40, 80)


def _blocks() -> dict:
    return {
        "constant": [1.9] * 16,
        "constant_negative": [-1.9] * 16,
        "mixed_signs": [1.9, -1.5, 0.3, -0.1, 0.0, 1.0, -1.0, 0.25,
                        1.8, -1.7, 0.7, -0.4, 1.2, -1.1, 0.9, -0.2],
        "max_positive": [1.9, 0.4, -0.2, 0.1, 0.0, 0.5, -0.5, 1.5,
                         0.7, -0.9, 1.1, -1.3, 0.2, -0.6, 1.4, -1.8],
        "zero_block": [0.0] * 16,
        "signed_zero": [-0.0] + [0.0] * 15,
    }


def oracle_block(values: torch.Tensor, bits: int):
    """Enumerate every code for each candidate exponent, in float64.

    Returns ``(exponent, codes, tied)`` where ``codes`` are the signed grid
    indices (``code - offset``) chosen element-wise and ``tied`` marks elements
    that sit exactly half-way between two grid points. A tie is resolved by the
    format's round-to-nearest-even rule, so callers compare those elements by
    achieved error instead of by code.
    """

    assert values.numel() == BLOCKFLOAT_BLOCK
    block = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    levels = 1 << stored_mantissa_bits(bits)
    offset = 2 * levels - 1
    signed_codes = torch.arange(-offset, offset + 1, dtype=torch.int64)
    units = signed_codes.to(torch.float64) / levels

    safe = max(float(block.abs().max()), float(torch.finfo(torch.float32).tiny))
    base = min(int(torch.floor(torch.log2(torch.tensor(safe))).item()), 127)

    best = None
    for exponent in (base, base - 1, base + 1):
        if exponent > 127:
            continue
        grid = (units * (2.0**exponent)).reshape(-1, 1)
        distance = (grid - block.reshape(1, -1)).abs()
        chosen = distance.argmin(dim=0)
        error = float((grid[chosen, 0] - block).pow(2).sum().item())
        if best is None or error < best[0]:
            best = (error, exponent, grid, distance)

    _, exponent, grid, distance = best
    chosen = distance.argmin(dim=0)
    minimum = distance.min(dim=0).values
    # More than one code at the minimum distance is an exact tie.
    tied = (distance <= minimum.unsqueeze(0)).sum(dim=0) > 1
    codes = signed_codes[chosen]
    return exponent, codes, tied, grid[chosen, 0].reshape(-1)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("name", sorted(_blocks()))
def test_exponent_matches_float64_oracle_at_every_scale(bits, scale, name):
    values = torch.tensor(_blocks()[name], dtype=torch.float32)
    block = values * torch.tensor(2.0**scale, dtype=torch.float32)
    got, got_exponent = quantize_blockfloat(block.reshape(1, -1), bits)
    expected_exponent, expected_codes, tied, expected_values = oracle_block(block, bits)

    assert int(got_exponent.reshape(-1)[0]) == expected_exponent

    levels = 1 << stored_mantissa_bits(bits)
    scale = torch.pow(2.0, torch.tensor(float(expected_exponent)))
    got_codes = torch.round(got.reshape(-1) / scale * levels).to(torch.int64)
    # Exact ties are resolved by the format's round-to-nearest-even rule, so
    # compare the achieved error there instead of the chosen code.
    assert torch.equal(got_codes[~tied], expected_codes[~tied])
    got_error = (got.reshape(-1).double() - block.to(torch.float64)).pow(2).sum()
    oracle_error = (expected_values - block.to(torch.float64)).pow(2).sum()
    assert torch.equal(got_error, oracle_error)


def _top_quarter_block() -> torch.Tensor:
    """A block whose maximum sits in the top quarter of its binade.

    That is the regime where ``floor(log2(block_max))`` is not the winner: the
    grid only reaches ``1.75 * 2**exp`` for bf4_b, so ``base + 1`` is closer.
    """

    generator = torch.Generator().manual_seed(20261117)
    block = torch.rand(1, BLOCKFLOAT_BLOCK, generator=generator) + 1.0
    assert float(block.max()) > 1.75
    return block


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("pattern", ["top_quarter", "constant"])
def test_search_is_scale_invariant(bits, pattern):
    """The same pattern at ``2**p`` must produce exactly ``2**p``-scaled output."""

    block = (
        _top_quarter_block()
        if pattern == "top_quarter"
        else torch.full((1, BLOCKFLOAT_BLOCK), 1.9, dtype=torch.float32)
    )

    reference, reference_exponent = quantize_blockfloat(block, bits)
    assert torch.isfinite(reference).all()

    for exponent in (-80, -60, -40, -20, 20, 40, 60, 80):
        factor = torch.tensor(2.0**exponent, dtype=torch.float32)
        got, got_exponent = quantize_blockfloat(block * factor, bits)
        assert int(got_exponent.reshape(-1)[0]) == int(reference_exponent.reshape(-1)[0]) + exponent
        assert torch.equal(got.reshape(-1), reference.reshape(-1) * factor)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("magnitude", ["tiny", "max", "one_ulp_below_max"])
def test_extreme_magnitudes_stay_finite_and_storable(bits, magnitude):
    """A block at the float32 edge must still produce a storable exponent."""

    info = torch.finfo(torch.float32)
    value = {
        "tiny": float(info.tiny),
        "max": float(info.max),
        "one_ulp_below_max": float(torch.nextafter(
            torch.tensor(info.max, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )),
    }[magnitude]
    block = torch.full((1, BLOCKFLOAT_BLOCK), value, dtype=torch.float32)

    got, got_exponent = quantize_blockfloat(block, bits)
    exponent = int(got_exponent.reshape(-1)[0])

    assert torch.isfinite(got).all(), got
    # ``2**exponent`` must be finite and the exponent must fit int8 storage.
    assert -128 <= exponent <= 127
    assert torch.isfinite(torch.pow(2.0, torch.tensor(float(exponent))))
    # The stored grid value must survive decompose/recompose unchanged.
    codes, recovered = decompose_blockfloat(got, bits)
    assert torch.equal(recompose_blockfloat(codes, recovered, bits), got)


def test_pack_path_carries_the_corrected_exponent():
    """The production CPU pack path must produce the oracle's exponent."""

    weight = torch.full((16, 16), 1.9 * 2.0**80, dtype=torch.float32)
    layer = TTBlockfloatLinear(
        bits=4,
        in_features=16,
        out_features=16,
        bias=False,
        backend=BACKEND.TT_BLOCKFLOAT,
        dtype=torch.float32,
    )
    source = torch.nn.Linear(16, 16, bias=False)
    with torch.no_grad():
        source.weight.copy_(weight)
    layer.pack(source)

    expected_exponent, expected_codes, tied, _ = oracle_block(weight[0], 4)
    assert expected_exponent == 81
    assert not bool(tied.any())
    assert torch.equal(
        layer.exponents.to(torch.int64),
        torch.full_like(layer.exponents, 81, dtype=torch.int64),
    )
    from gptqmodel.utils.tt_blockfloat_quant import _unpack_codes

    got_codes = _unpack_codes(layer.codes, 4, (16, 16))
    assert torch.equal(got_codes[0].to(torch.int64), expected_codes + 7)
