# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paro bf16 rotation must round once, not after every Givens stage (#4221).

The reference and vectorized CPU paths used to cast ``cos``/``sin`` and the
activation to bf16 inside each of the ``krot`` stages, which is roughly three
times the error of computing the whole rotation in fp32 and rounding the result
once.  fp16 follows the same single-rounding contract (#5634): it has the
narrower exponent range, so per-stage narrowing could even overflow a partial
sum whose final value is representable.
"""

import torch

from gptqmodel.utils.paro import (
    apply_paro_rotation_reference,
    apply_paro_rotation_vectorized,
)


def _rotation_fixture(rows=64, hidden=256, group_size=128, krot=8, dtype=torch.bfloat16):
    torch.manual_seed(20261116)
    groups = hidden // group_size
    half_group = group_size // 2
    pair_rows = []
    for _ in range(krot):
        group_pairs = []
        for _ in range(groups):
            permutation = torch.randperm(group_size)
            group_pairs.append(
                torch.stack([permutation[:half_group], permutation[half_group:]], dim=-1).reshape(-1)
            )
        pair_rows.append(torch.cat(group_pairs))
    pairs = torch.stack(pair_rows).to(torch.int16)
    theta = torch.randn(krot, hidden // 2) * 0.5
    x = torch.randn(rows, hidden, dtype=dtype)
    return x, pairs, theta


def _rotate_per_stage(x, pairs, theta, group_size):
    """Emulate the old per-stage activation-dtype rounding contract."""

    original_shape = x.shape
    x2d = x.reshape(-1, original_shape[-1])
    out = x2d.clone()
    hidden = x2d.shape[-1]
    num_groups = hidden // group_size
    half_group = group_size // 2
    pairs = pairs.to(device=x2d.device, dtype=torch.int64)
    theta = theta.to(device=x2d.device, dtype=torch.float32)
    for rot_idx in range(pairs.shape[0]):
        next_out = out.clone()
        pair_row = pairs[rot_idx].reshape(num_groups, half_group, 2)
        theta_row = theta[rot_idx].reshape(num_groups, half_group)
        for group_idx in range(num_groups):
            offset = group_idx * group_size
            idx_i = offset + pair_row[group_idx, :, 0]
            idx_j = offset + pair_row[group_idx, :, 1]
            cos_t = torch.cos(theta_row[group_idx]).to(dtype=out.dtype)
            sin_t = torch.sin(theta_row[group_idx]).to(dtype=out.dtype)
            xi = out[:, idx_i]
            xj = out[:, idx_j]
            next_out[:, idx_i] = xi * cos_t + xj * sin_t
            next_out[:, idx_j] = -xi * sin_t + xj * cos_t
        out = next_out
    return out.reshape(original_shape)


def test_paro_bf16_rotation_rounds_once():
    """Both CPU paths must match the fp32 oracle to within one bf16 rounding."""

    x, pairs, theta = _rotation_fixture(dtype=torch.bfloat16)
    oracle = apply_paro_rotation_reference(
        x.float(), pairs, theta, group_size=128
    ).to(torch.bfloat16)
    per_stage = _rotate_per_stage(x, pairs, theta, group_size=128)

    per_stage_error = (per_stage.float() - oracle.float()).abs().max().item()
    assert per_stage_error > 0

    for rotate in (apply_paro_rotation_reference, apply_paro_rotation_vectorized):
        rotated = rotate(x, pairs, theta, group_size=128)
        assert rotated.dtype == torch.bfloat16
        error = (rotated.float() - oracle.float()).abs().max().item()
        # One final rounding is at most half an ulp; the old path rounded after
        # every stage and is measurably farther from the oracle.
        assert error <= per_stage_error / 2


def test_paro_bf16_reference_and_vectorized_stay_identical():
    x, pairs, theta = _rotation_fixture(dtype=torch.bfloat16)

    reference = apply_paro_rotation_reference(x, pairs, theta, group_size=128)
    vectorized = apply_paro_rotation_vectorized(x, pairs, theta, group_size=128)

    torch.testing.assert_close(reference, vectorized, rtol=0, atol=0)


def test_paro_fp16_rotation_rounds_once():
    """fp16 must match the fp32 oracle within one rounding (#5634).

    fp16 used to keep the historical per-stage arithmetic; that narrowing both
    accumulated error and could overflow an intermediate value whose final
    result is representable, so it now shares the single-rounding contract.
    """

    x, pairs, theta = _rotation_fixture(dtype=torch.float16)
    oracle = apply_paro_rotation_reference(
        x.float(), pairs, theta, group_size=128
    ).to(torch.float16)
    per_stage = _rotate_per_stage(x, pairs, theta, group_size=128)

    per_stage_error = (per_stage.float() - oracle.float()).abs().max().item()
    assert per_stage_error > 0

    for rotate in (apply_paro_rotation_reference, apply_paro_rotation_vectorized):
        rotated = rotate(x, pairs, theta, group_size=128)
        assert rotated.dtype == torch.float16
        torch.testing.assert_close(rotated, oracle, rtol=0, atol=0)
