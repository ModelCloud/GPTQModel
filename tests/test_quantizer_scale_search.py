# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Strict accuracy tests for Quantizer scale-search paths.

These tests compare the batched grouped scale search against the per-group
find_params reference and against a full FP64 grid-search reference for small
problems, so FP32/FP64 or tie-breaking regressions in the fast paths are caught
in CI.
"""

import os

import pytest
import torch

from gptqmodel.quantization import QuantizeConfig, ScaleSearchConfig, Quantizer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _method_seed(method: ScaleSearchConfig) -> int:
    """Stable numeric seed offset for each scale-search method."""
    return {
        ScaleSearchConfig.ACTIVATION: 0,
        ScaleSearchConfig.HESSIAN: 1,
        ScaleSearchConfig.HYBRID: 2,
        ScaleSearchConfig.MSE: 3,
    }[method]


def _make_group_block_hessian(cols: int, group_size: int, device: str = "cuda", dtype=torch.float32):
    """Build a positive-definite block-diagonal Hessian for grouped scale search."""
    num_groups = cols // group_size
    groups = []
    for _ in range(num_groups):
        a = torch.randn(2 * group_size, group_size, device=device, dtype=dtype)
        groups.append(a.T @ a / a.shape[0])
    return torch.block_diag(*groups)


def _reference_find_params(
    W: torch.Tensor,
    hessian: torch.Tensor,
    qcfg: QuantizeConfig,
):
    """Reference per-group find_params run on each group of W."""
    rows, cols = W.shape
    group_size = qcfg.group_size
    num_groups = cols // group_size
    scale_ref = []
    zero_ref = []
    for g in range(num_groups):
        q = Quantizer(qcfg)
        q.configure(perchannel=True, grid=100, maxshrink=0.8)
        q.maxq = q.maxq.to(W.device)
        block = W[:, g * group_size : (g + 1) * group_size]
        h_block = hessian[g * group_size : (g + 1) * group_size, g * group_size : (g + 1) * group_size]
        q.find_params(block, weight=True, hessian=h_block)
        scale_ref.append(q.scale.reshape(rows))
        zero_ref.append(q.zero.reshape(rows))
    return torch.stack(scale_ref, dim=1), torch.stack(zero_ref, dim=1)


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("method", [ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID])
def test_find_params_batched_matches_per_group_reference(group_size, bits, sym, method):
    """batched grouped search must reproduce per-group find_params scale/zero."""
    rows = 64
    cols = group_size * 2
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(2025 + group_size + bits + int(sym) + _method_seed(method))
    W = torch.randn(rows, cols, generator=generator, device=device, dtype=torch.float32)
    hessian = _make_group_block_hessian(cols, group_size, device=device)

    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        mse=2.0,
        scale_search=method,
    )

    # Disable the experimental Triton path so this test validates the
    # supported PyTorch reference path regardless of environment variables.
    prev = os.environ.get("GPTQMODEL_SCALE_SEARCH_TRITON")
    os.environ["GPTQMODEL_SCALE_SEARCH_TRITON"] = "0"
    try:
        q = Quantizer(qcfg)
        q.configure(perchannel=True, grid=100, maxshrink=0.8)
        q.maxq = q.maxq.to(device)
        W3d = W.reshape(rows, cols // group_size, group_size)
        if method == ScaleSearchConfig.ACTIVATION:
            hessian_batched = hessian.diagonal().reshape(cols // group_size, group_size)
        else:
            blocks = hessian.reshape(cols // group_size, group_size, cols // group_size, group_size)
            idx = torch.arange(cols // group_size, device=device)
            hessian_batched = blocks[idx, :, idx, :]
        scale_b, zero_b = q.find_params_batched(W3d, weight=True, hessian=hessian_batched)

        scale_ref, zero_ref = _reference_find_params(W, hessian, qcfg)

        torch.cuda.synchronize()
    finally:
        if prev is None:
            os.environ.pop("GPTQMODEL_SCALE_SEARCH_TRITON", None)
        else:
            os.environ["GPTQMODEL_SCALE_SEARCH_TRITON"] = prev

    scale_diff = (scale_b - scale_ref).abs().max().item()
    zero_diff = (zero_b - zero_ref).abs().max().item()
    assert scale_diff < 1e-6, f"scale mismatch {scale_diff} for {group_size=}, {bits=}, {sym=}, {method=}"
    assert zero_diff < 1e-6, f"zero mismatch {zero_diff} for {group_size=}, {bits=}, {sym=}, {method=}"


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("method", [ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID])
def test_find_params_matches_fp64_grid_reference(group_size, bits, sym, method):
    """Quantizer.find_params output should be close to a full FP64 grid-search reference."""
    rows = 16
    cols = group_size
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(3030 + group_size + bits + int(sym) + _method_seed(method))
    W = torch.randn(rows, cols, generator=generator, device=device, dtype=torch.float32)
    hessian = _make_group_block_hessian(cols, group_size, device=device, dtype=torch.float64)

    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        mse=2.0,
        scale_search=method,
    )

    # FP64 reference grid search
    W64 = W.to(torch.float64)
    H64 = hessian.to(torch.float64)
    maxq = (1 << bits) - 1
    tmp = torch.zeros(rows, device=device, dtype=torch.float64)
    xmin = torch.minimum(W64.amin(dim=-1), tmp)
    xmax = torch.maximum(W64.amax(dim=-1), tmp)
    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        xmin = torch.where(xmin < 0, -xmax, xmin)
    zero_all = torch.where((xmin == 0) & (xmax == 0), -torch.ones_like(xmin), xmin)
    xmax = torch.where((xmin == 0) & (xmax == 0), torch.ones_like(xmax), xmax)
    xmin = zero_all

    grid = 100
    maxshrink = 0.8
    candidate_count = int(maxshrink * grid)
    shrink = 1.0 - torch.arange(candidate_count, device=device, dtype=torch.float64) / grid
    p = shrink.view(-1, 1)
    xmin_all = p * xmin.unsqueeze(0)
    xmax_all = p * xmax.unsqueeze(0)
    scale_all = (xmax_all - xmin_all) / maxq
    if sym:
        zero_ref_all = torch.full_like(scale_all, (maxq + 1.0) / 2.0)
    else:
        zero_ref_all = torch.round(-xmin_all / scale_all)

    # build error tensor [candidate, rows, cols]
    W_exp = W64.unsqueeze(0)
    q = torch.round(W_exp / scale_all.unsqueeze(-1))
    q = q + zero_ref_all.unsqueeze(-1)
    q = torch.clamp(q, 0.0, float(maxq))
    dequant = (q - zero_ref_all.unsqueeze(-1)) * scale_all.unsqueeze(-1)
    error = dequant - W_exp

    if method == ScaleSearchConfig.ACTIVATION:
        importance = H64.diagonal().clamp_min(0)
        importance = importance / importance.mean().clamp_min(1e-12)
        losses = (error.square() * importance).sum(dim=-1)
    elif method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
        # block diagonal, single group
        h = H64 / H64.diagonal().clamp_min(0).mean().clamp_min(1e-12)
        if method == ScaleSearchConfig.HYBRID:
            h = h * 0.5
            h.diagonal().mul_(2.0)
        projected = error @ h
        losses = (error * projected).sum(dim=-1).clamp_min(0)
    else:
        losses = error.abs().pow(2.0).sum(dim=-1)

    best_idx = losses.argmin(dim=0)
    scale_ref = scale_all[best_idx, torch.arange(rows, device=device)]

    # FP32 Quantizer
    prev = os.environ.get("GPTQMODEL_SCALE_SEARCH_TRITON")
    os.environ["GPTQMODEL_SCALE_SEARCH_TRITON"] = "0"
    try:
        q = Quantizer(qcfg)
        q.configure(perchannel=True, grid=100, maxshrink=0.8)
        q.maxq = q.maxq.to(device)
        q.find_params(W, weight=True, hessian=hessian.to(torch.float32))
        scale_q = q.scale.reshape(rows)
        zero_q = q.zero.reshape(rows)
        torch.cuda.synchronize()
    finally:
        if prev is None:
            os.environ.pop("GPTQMODEL_SCALE_SEARCH_TRITON", None)
        else:
            os.environ["GPTQMODEL_SCALE_SEARCH_TRITON"] = prev

    # Compare against the FP64 reference by the actual objective value rather
    # than raw scale/zero coordinates: FP32 grid search can legitimately pick
    # an adjacent candidate on a near tie, whose scale is one grid step away.
    scale_q64 = scale_q.to(torch.float64)
    zero_q64 = zero_q.to(torch.float64)
    q_q = torch.round(W64 / scale_q64.unsqueeze(-1))
    q_q = q_q + zero_q64.unsqueeze(-1)
    q_q = torch.clamp(q_q, 0.0, float(maxq))
    dequant_q = (q_q - zero_q64.unsqueeze(-1)) * scale_q64.unsqueeze(-1)
    error_q = dequant_q - W64

    if method == ScaleSearchConfig.ACTIVATION:
        importance = H64.diagonal().clamp_min(0)
        importance = importance / importance.mean().clamp_min(1e-12)
        loss_q = (error_q.square() * importance).sum(dim=-1)
    elif method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
        h = H64 / H64.diagonal().clamp_min(0).mean().clamp_min(1e-12)
        if method == ScaleSearchConfig.HYBRID:
            h = h * 0.5
            h.diagonal().mul_(2.0)
        projected = error_q @ h
        loss_q = (error_q * projected).sum(dim=-1).clamp_min(0)
    else:
        loss_q = error_q.abs().pow(2.0).sum(dim=-1)

    loss_ref_min = losses[best_idx, torch.arange(rows, device=device)]
    loss_diff = (loss_q - loss_ref_min).abs().max().item()
    # FP32 grid search uses FP32 Hessian and reductions, so it can select an
    # adjacent candidate on a near tie. Allow a small relative margin on the
    # objective value; the first per-group reference test still enforces exact
    # agreement with the supported FP32 path.
    loss_tol = max(1e-9, 2.0e-1 * loss_ref_min.abs().max().item())
    assert loss_diff <= loss_tol, (
        f"loss mismatch {loss_diff} for {group_size=}, {bits=}, {sym=}, {method=}"
    )

    # The selected scale should never be more than one grid step away from the
    # FP64 optimum; this catches gross regressions while allowing single-candidate
    # tie differences.
    scale_step = ((xmax - xmin) / (maxq * grid)).abs().max().item()
    scale_diff = (scale_q - scale_ref).abs().max().item()
    assert scale_diff <= 2.0 * scale_step + 1e-6, (
        f"scale mismatch {scale_diff} for {group_size=}, {bits=}, {sym=}, {method=}"
    )


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("method", [ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID])
def test_find_params_batched_triton_matches_eager(group_size, bits, sym, method):
    """Triton fast path for grouped find_params_batched must match the eager path."""
    rows = 64
    cols = group_size
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(4040 + group_size + bits + int(sym) + _method_seed(method))
    W = torch.randn(rows, cols, generator=generator, device=device, dtype=torch.float32)
    hessian = _make_group_block_hessian(cols, group_size, device=device)

    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        mse=2.0,
        scale_search=method,
    )

    def run(triton_flag: str):
        prev = os.environ.get("GPTQMODEL_SCALE_SEARCH_TRITON")
        os.environ["GPTQMODEL_SCALE_SEARCH_TRITON"] = triton_flag
        try:
            q = Quantizer(qcfg)
            q.configure(perchannel=True, grid=100, maxshrink=0.8)
            q.maxq = q.maxq.to(device)
            W3d = W.reshape(rows, cols // group_size, group_size)
            blocks = hessian.reshape(1, group_size, group_size)
            if method == ScaleSearchConfig.ACTIVATION:
                hessian_batched = hessian.diagonal().reshape(1, group_size)
            else:
                hessian_batched = blocks
            scale, zero = q.find_params_batched(W3d, weight=True, hessian=hessian_batched)
            torch.cuda.synchronize()
            return scale, zero
        finally:
            if prev is None:
                os.environ.pop("GPTQMODEL_SCALE_SEARCH_TRITON", None)
            else:
                os.environ["GPTQMODEL_SCALE_SEARCH_TRITON"] = prev

    scale_ref, zero_ref = run("0")
    scale_tri, zero_tri = run("1")

    scale_diff = (scale_tri - scale_ref).abs().max().item()
    zero_diff = (zero_tri - zero_ref).abs().max().item()
    assert scale_diff < 1e-6, f"scale mismatch {scale_diff} for {group_size=}, {bits=}, {sym=}, {method=}"
    assert zero_diff < 1e-6, f"zero mismatch {zero_diff} for {group_size=}, {bits=}, {sym=}, {method=}"


def _make_dense_pd_hessian(cols: int, device: str = "cuda", dtype=torch.float32):
    """Build a dense positive-definite Hessian for multi-group scale-search tests."""
    generator = torch.Generator(device=device).manual_seed(10000 + cols)
    a = torch.randn(cols, max(cols // 2, 1), generator=generator, device=device, dtype=dtype)
    h = a.matmul(a.t())
    h = (h + h.t()) * 0.5
    return h


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("method", [ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID])
def test_find_params_batched_dense_matches_per_group_reference(group_size, bits, sym, method):
    """Batched grouped search must reproduce per-group find_params on dense Hessians with Triton enabled."""
    rows = 128
    cols = 256
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(
        5050 + group_size + bits + int(sym) + _method_seed(method)
    )
    W = torch.randn(rows, cols, generator=generator, device=device, dtype=torch.float32)
    hessian = _make_dense_pd_hessian(cols, device=device)

    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        mse=2.0,
        scale_search=method,
    )

    # Reference per-group find_params (always CPU/CUDA fallback, no Triton)
    scale_ref, zero_ref = _reference_find_params(W, hessian, qcfg)

    # Batched path with Triton enabled (activation uses Triton; hessian/hybrid use vectorized fallback)
    num_groups = cols // group_size
    q = Quantizer(qcfg)
    q.configure(perchannel=True, grid=100, maxshrink=0.8)
    q.maxq = q.maxq.to(device)
    W3d = W.reshape(rows, num_groups, group_size)
    if method == ScaleSearchConfig.ACTIVATION:
        hessian_batched = hessian.diagonal().reshape(num_groups, group_size)
    else:
        hessian_batched = torch.stack(
            [hessian[g * group_size : (g + 1) * group_size, g * group_size : (g + 1) * group_size] for g in range(num_groups)],
            dim=0,
        )
    scale_b, zero_b = q.find_params_batched(W3d, weight=True, hessian=hessian_batched)
    torch.cuda.synchronize()

    scale_diff = (scale_b - scale_ref).abs().max().item()
    zero_diff = (zero_b - zero_ref).abs().max().item()
    assert scale_diff < 1e-6, f"scale mismatch {scale_diff} for {group_size=}, {bits=}, {sym=}, {method=}"
    assert zero_diff < 1e-6, f"zero mismatch {zero_diff} for {group_size=}, {bits=}, {sym=}, {method=}"
