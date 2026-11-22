# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paro fp16 rotation must not overflow on an intermediate Givens stage (#5634).

A rotation whose final value is representable can still push a partial sum past
the fp16 maximum (65504).  Rounding the running activation to fp16 after every
stage then stores ``inf`` and the following stage turns it into ``nan``.  The
reference path and the fused kernel both accumulate in fp32 and round once, so
the intermediate 84853 below collapses back to a finite 60000.
"""

import math

import pytest
import torch

from gptqmodel.utils import paro as paro_module
from gptqmodel.utils.paro import (
    apply_paro_rotation,
    apply_paro_rotation_reference,
    apply_paro_rotation_vectorized,
)


def _transient_overflow_fixture(rows=4, hidden=128):
    """Two +/-45 degree stages that overflow only between the stages."""

    half_group = hidden // 2
    x = torch.full((rows, hidden), 60000.0, dtype=torch.float16)
    pairs_row = torch.stack([torch.arange(half_group), torch.arange(half_group) + half_group], dim=-1)
    pairs = pairs_row.reshape(1, -1).repeat(2, 1).to(torch.int16)
    theta = torch.tensor(
        [math.pi / 4] * half_group + [-math.pi / 4] * half_group, dtype=torch.float16
    ).reshape(2, half_group)
    return x, pairs, theta


def _pad_rotation_stages(pairs, theta, stages=8):
    """Append identity stages so the fused kernel is eligible.

    The overflow fixture only needs two stages to trigger the bug, but
    ``_rotation_kernel_ready`` accepts just ``krot in {1, 8}``.  An unpadded
    two-stage fixture therefore takes the reference path, which made the fused
    comparison below pass without ever running the kernel (issue #5690).
    """

    half_group = theta.shape[1]
    extra = stages - pairs.shape[0]
    assert extra >= 0
    if extra == 0:
        return pairs, theta
    identity_pairs = (
        torch.stack([torch.arange(half_group), torch.arange(half_group) + half_group], dim=-1)
        .reshape(1, -1)
        .repeat(extra, 1)
        .to(pairs.dtype)
    )
    identity_theta = torch.zeros((extra, half_group), dtype=theta.dtype)
    return torch.cat([pairs, identity_pairs]), torch.cat([theta, identity_theta])


def _rotate_per_stage(x, pairs, theta, group_size=128):
    """Emulate the old per-stage fp16 workspace that overflows."""

    original_shape = x.shape
    x2d = x.reshape(-1, original_shape[-1])
    out = x2d.clone()
    hidden = x2d.shape[-1]
    num_groups = hidden // group_size
    half_group = group_size // 2
    pairs = pairs.to(dtype=torch.int64)
    theta = theta.to(dtype=torch.float32)
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


def test_per_stage_fp16_workspace_overflows_the_fixture():
    """Keep the regression honest: the old contract really does overflow here."""

    x, pairs, theta = _transient_overflow_fixture()
    per_stage = _rotate_per_stage(x, pairs, theta)
    assert not torch.isfinite(per_stage).all()


def test_reference_fp16_transient_overflow_stays_finite():
    x, pairs, theta = _transient_overflow_fixture()
    out = apply_paro_rotation_reference(x, pairs, theta, group_size=128)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.full_like(out, 60000.0))


def test_vectorized_fp16_transient_overflow_stays_finite():
    x, pairs, theta = _transient_overflow_fixture()
    out = apply_paro_rotation_vectorized(x, pairs, theta, group_size=128)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.full_like(out, 60000.0))


def test_reference_fp16_rounds_once_like_fp32():
    """fp16 must match fp32 arithmetic rounded once at the end."""

    torch.manual_seed(20261120)
    hidden, group_size = 128, 128
    half_group = group_size // 2
    permutation = torch.randperm(group_size)
    pairs = (
        torch.stack([permutation[:half_group], permutation[half_group:]], dim=-1)
        .reshape(1, -1)
        .repeat(8, 1)
        .to(torch.int16)
    )
    theta = torch.randn(8, hidden // 2) * 0.5
    x = (torch.randn(64, hidden) * 2000.0).to(torch.float16)

    actual = apply_paro_rotation_reference(x, pairs, theta, group_size=group_size)
    expected = apply_paro_rotation_reference(
        x.to(torch.float32), pairs, theta.to(torch.float32), group_size=group_size
    ).to(torch.float16)
    assert torch.equal(actual, expected)


def _fused_ready():
    return (
        torch.cuda.is_available()
        and paro_module._rotation_kernel_ready(
            torch.empty(0, dtype=torch.float16, device="cuda"),
            torch.empty(0, dtype=torch.int16, device="cuda"),
            torch.empty((1, 0), dtype=torch.float16, device="cuda"),
            None,
            128,
        )
        and paro_module._load_rotation_extension()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Paro fused rotation requires CUDA.")
def test_fused_fp16_matches_reference(monkeypatch):
    x, pairs, theta = _transient_overflow_fixture()
    pairs, theta = _pad_rotation_stages(pairs, theta)
    device = torch.device("cuda", torch.cuda.current_device())
    x, pairs, theta = x.to(device), pairs.to(device), theta.to(device)
    if not _fused_ready():
        pytest.skip("fused Paro rotation extension is unavailable on this device")
    # Guard the fixture itself: without an eligible stage count the call below
    # silently degrades to the reference path and the comparison is vacuous.
    assert paro_module._rotation_kernel_ready(x, pairs, theta, None, 128)

    expected = apply_paro_rotation_reference(x, pairs, theta, group_size=128)

    def _no_reference_fallback(*args, **kwargs):
        raise AssertionError("apply_paro_rotation fell back to the reference path")

    monkeypatch.setattr(paro_module, "apply_paro_rotation_reference", _no_reference_fallback)
    actual = apply_paro_rotation(x, pairs, theta, group_size=128)
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Paro fused rotation requires CUDA.")
def test_fused_fp16_identity_round_trip():
    device = torch.device("cuda", torch.cuda.current_device())
    hidden = 128
    x = torch.arange(hidden, dtype=torch.float16, device=device).reshape(1, hidden)
    pairs = torch.zeros((1, hidden), dtype=torch.int16, device=device)
    theta = torch.zeros((1, hidden // 2), dtype=torch.float16, device=device)
    if not _fused_ready():
        pytest.skip("fused Paro rotation extension is unavailable on this device")

    out = apply_paro_rotation(x, pairs, theta, group_size=128)
    assert torch.equal(out, x)
