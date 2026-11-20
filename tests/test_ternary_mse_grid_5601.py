# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Ternary MSE must shrink the two levels, not the affine grid (#5601).

``configure(trits=True)`` stores the positive and negative levels in
``scale``/``zero`` and sets ``maxq = -1`` as a sentinel. The MSE candidate grid
built the affine ``(xmax - xmin) / maxq`` scale from that sentinel, so every
candidate was negative: an exactly representable ``[-1, 0, 1]`` input became
``[-1, -0.42, -0.42]`` with ``mse=2``. The candidate endpoints must come from
the shrunk extrema, with the unshrunk endpoints as the incumbent.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.qqq import Quantizer as QQQQuantizer
from gptqmodel.quantization.quantizer import Quantizer as GPTQQuantizer


GRID = 100
MAXSHRINK = 0.8
NORM = 2.0

QUANTIZERS = [GPTQQuantizer, QQQQuantizer]


def _brute(x, sym, grid=GRID, maxshrink=MAXSHRINK, norm=NORM):
    """Independent three-level reference for one channel."""

    xmin = min(x)
    xmax = max(x)
    if sym:
        xmax = max(abs(xmin), xmax)
        xmin = -xmax
    if xmin == 0 and xmax == 0:
        xmin = xmin - 1
        xmax = xmax + 1
    best = float("inf")
    scale = torch.tensor(xmax, dtype=torch.float32)
    zero = torch.tensor(xmin, dtype=torch.float32)
    xr = torch.tensor(x, dtype=torch.float32)
    for i in range(int(maxshrink * grid)):
        p = torch.tensor(1 - i / grid, dtype=torch.float32)
        s = p * xmax
        z = p * xmin
        q = torch.where(xr > s / 2, s, torch.where(xr < z / 2, z, torch.tensor(0.0)))
        err = ((q - xr).abs() ** norm).sum()
        if err < best:
            best = err
            scale = s
            zero = z
    return float(scale), float(zero)


def _quantizer(cls, sym, mse=NORM):
    if cls is GPTQQuantizer:
        q = GPTQQuantizer(QuantizeConfig(bits=4, group_size=-1, sym=sym, mse=mse))
        q.configure(perchannel=True, trits=True)
        return q
    q = QQQQuantizer()
    q.configure(bits=4, perchannel=True, sym=sym, groupsize=-1, mse=True, norm=mse, trits=True)
    return q


@pytest.mark.parametrize("cls", QUANTIZERS)
def test_exact_levels_survive_mse(cls):
    """The unshrunk endpoints reconstruct ``[-1, 0, 1]`` exactly."""

    x = [-1.0, 0.0, 1.0]
    for sym in (True, False):
        q = _quantizer(cls, sym)
        q.find_params(torch.tensor([x]), weight=True)
        assert float(q.scale.flatten()[0]) == 1.0
        assert float(q.zero.flatten()[0]) == -1.0
        assert [round(float(v), 4) for v in q.quantize(torch.tensor([x])).flatten()] == x


@pytest.mark.parametrize("cls", QUANTIZERS)
def test_brute_three_level_reference(cls):
    """The selected endpoints match the brute-force three-level argmin."""

    x = [-4.0, 1.5, 1.5, 1.5, 1.5, 1.5, 4.0]
    for sym in (True, False):
        scale, zero = _brute(x, sym)
        q = _quantizer(cls, sym)
        q.find_params(torch.tensor([x]), weight=True)
        assert abs(float(q.scale.flatten()[0]) - scale) < 1e-6
        assert abs(float(q.zero.flatten()[0]) - zero) < 1e-6


@pytest.mark.parametrize("cls", QUANTIZERS)
def test_candidates_are_three_levels(cls):
    """A shrunk candidate stays a valid positive/negative level pair."""

    x = [-4.0, 1.5, 1.5, 1.5, 1.5, 1.5, 4.0]
    for sym in (True, False):
        q = _quantizer(cls, sym)
        q.find_params(torch.tensor([x]), weight=True)
        scale = float(q.scale.flatten()[0])
        zero = float(q.zero.flatten()[0])
        assert scale > 0
        assert zero < 0
        assert scale <= 4.0
        assert zero >= -4.0
