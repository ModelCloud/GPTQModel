#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Validate ttbfp's quantizer against tt-metal's actual tt-llk helpers.

The functions below are copied verbatim from

  tt_metal/tt-llk/tests/python_tests/helpers/bfp_format_utils.py
  tt_metal/tt-llk/tests/python_tests/helpers/pack.py

(SPDX-FileCopyrightText: (c) 2025-2026 Tenstorrent AI ULC, Apache-2.0).
Requires torch; run with:

    TTBFP_LIB=./libttbfp.so PYTHONPATH=./python \
        python tools/verify_against_tt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import ttbfp  # noqa: E402
from ttbfp import reference as ref  # noqa: E402

BFP_BLOCK = 16
BFP_MAX_ALIGNMENT_SHIFT = 8


def _bf16_sign_exp_mantissa(flat: torch.Tensor):
    bf16_bits = (flat.view(torch.int32) >> 16) & 0xFFFF
    signs = bf16_bits >> 15
    exps = (bf16_bits >> 7) & 0xFF
    mants = ((bf16_bits & 0x7F) >> 1) | 0x40
    return signs, exps, mants


def _bfp8_mantissas_per_block(mants_blocks: torch.Tensor, exps_blocks: torch.Tensor):
    shared_exps = exps_blocks.max(dim=1, keepdim=True).values
    deltas = shared_exps - exps_blocks
    shifts = deltas.clamp(max=BFP_MAX_ALIGNMENT_SHIFT)
    guard_shifts = (deltas - 1).clamp(min=0, max=BFP_MAX_ALIGNMENT_SHIFT)
    guard_bits = (mants_blocks >> guard_shifts) & 1
    guard = torch.where(deltas > 0, guard_bits, torch.zeros_like(mants_blocks))
    bfp8 = ((mants_blocks >> shifts) + guard) & 0x7F
    return shared_exps, bfp8


def upstream_pack(values: np.ndarray, mb: int):
    """Byte-exact upstream packer for a whole 1024-element tile."""
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    assert values.numel() == 1024
    bits = values.view(torch.int32)
    values = torch.where(torch.isnan(values), bits | (1 << 22), bits).view(torch.float32)
    signs, exps, mants = _bf16_sign_exp_mantissa(values)
    shared, q8 = _bfp8_mantissas_per_block(
        mants.reshape(64, BFP_BLOCK), exps.reshape(64, BFP_BLOCK))
    mags = q8 >> (7 - mb)
    dat = torch.where(mags != 0, (signs.reshape(64, BFP_BLOCK) << mb) | mags, 0)
    dat = dat.flatten().to(torch.uint8).numpy()
    if mb == 3:
        dat = (dat[0::2] | (dat[1::2] << 4)).astype(np.uint8)
    elif mb == 1:
        dat = (dat[0::4] | (dat[1::4] << 2) | (dat[2::4] << 4)
               | (dat[3::4] << 6)).astype(np.uint8)
    return shared.flatten().to(torch.uint8).numpy(), dat


def make_tile(rng, mode):
    if mode == 0:
        return rng.standard_normal((32, 32)).astype(np.float32)
    if mode == 1:
        return (rng.standard_normal((32, 32)) *
                np.float32(10.0) ** rng.uniform(-30, 30, (32, 32))).astype(np.float32)
    if mode == 2:
        e = rng.integers(-30, 30, (32, 32))
        t = np.ldexp(np.ones((32, 32), np.float32), e.astype(np.int32))
        return (t + np.ldexp(np.ones((32, 32), np.float32),
                             (e - rng.integers(1, 10, (32, 32))).astype(np.int32))
                ).astype(np.float32)
    t = np.zeros((32, 32), np.float32)
    t.flat[::5] = -0.0
    t[0, :8] = np.array([np.inf, -np.inf, np.nan, 0.0, 1.0, -1.0, 2.0, -2.0],
                        np.float32)
    return t


def main():
    rng = np.random.default_rng(0)
    failures = 0
    for seed in range(200):
        for mode in range(4):
            tile = make_tile(rng, mode)
            faces = ref.tilize(tile)
            for mb in (7, 3, 1):
                fmt = 8 if mb == 7 else (4 if mb == 3 else 2)
                ue, ud = upstream_pack(faces, mb)
                ce, cd = ttbfp.pack(faces.reshape(64, 16), fmt)
                ok = np.array_equal(ue, ce.reshape(-1)) and np.array_equal(ud, cd.reshape(-1))
                if not ok:
                    failures += 1
                    print(f"FAIL seed={seed} mode={mode} mb={mb}")
                    if failures > 5:
                        return 1
                blob_u = np.concatenate([ue, ud])
                blob_c = ttbfp.tile_pack(tile, fmt)
                if not np.array_equal(blob_u, blob_c):
                    failures += 1
                    print(f"TILE FAIL seed={seed} mode={mode} mb={mb}")
                    if failures > 5:
                        return 1
    print(f"checked 200 seeds x 4 distributions x 3 formats: "
          f"{'OK' if failures == 0 else str(failures) + ' failures'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
