#!/usr/bin/env python3
"""Verify find_params_batched matches per-group find_params."""

from __future__ import annotations

import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization import QuantizeConfig, Quantizer, ScaleSearchConfig  # noqa: E402


def build_hessians(columns: int, group_size: int, device: str):
    generator = torch.Generator(device=device).manual_seed(123)
    H = torch.randn(columns, max(columns // 2, 1), generator=generator, device=device, dtype=torch.float32)
    H = H.matmul(H.t())
    H = (H + H.t()) * 0.5
    diagonals = []
    blocks = []
    for g in range(0, columns, group_size):
        ge = min(g + group_size, columns)
        blocks.append(H[g:ge, g:ge])
        diagonals.append(H.diagonal()[g:ge])
    return H, torch.stack(blocks, dim=0), torch.stack(diagonals, dim=0)


def test_group_size(group_size: int, method: ScaleSearchConfig, sym: bool = False):
    device = "cuda:0"
    rows = 64
    columns = 256
    generator = torch.Generator(device=device).manual_seed(42)
    W = torch.randn((rows, columns), generator=generator, device=device, dtype=torch.float32)
    H, H_blocks, H_diag = build_hessians(columns, group_size, device)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=group_size,
        sym=sym,
        desc_act=False,
        offload_to_disk=False,
        mse=2.0,
        scale_search=method,
    )
    q = Quantizer(qcfg)
    q.configure(perchannel=True, grid=20, maxshrink=0.8)
    q.maxq = q.maxq.to(device)

    # Per-group reference
    ref_scales = []
    ref_zeros = []
    def hessian_for_group(g):
        if method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
            return H_blocks[g // group_size]
        return H_diag[g // group_size]

    for g in range(0, columns, group_size):
        ge = min(g + group_size, columns)
        h = hessian_for_group(g)
        q.find_params(W[:, g:ge], weight=True, hessian=h)
        ref_scales.append(q.scale.squeeze().clone())
        ref_zeros.append(q.zero.squeeze().clone())
    ref_scale = torch.stack(ref_scales, dim=1)
    ref_zero = torch.stack(ref_zeros, dim=1)

    # Batched
    W_3d = W.reshape(rows, columns // group_size, group_size)
    if method == ScaleSearchConfig.ACTIVATION:
        batched_hessian = H_diag
    elif method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
        batched_hessian = H_blocks
    else:
        batched_hessian = None
    batched_scale, batched_zero = q.find_params_batched(W_3d, weight=True, hessian=batched_hessian)

    print(f"group_size={group_size} method={method.value} sym={sym}")
    print(" scale close:", torch.allclose(batched_scale, ref_scale, atol=1e-5, rtol=1e-5))
    print(" zero close:", torch.allclose(batched_zero, ref_zero, atol=1e-5, rtol=1e-5))
    if not torch.allclose(batched_scale, ref_scale, atol=1e-5, rtol=1e-5):
        diff = (batched_scale - ref_scale).abs()
        print(" max scale diff:", diff.max().item(), "at", (diff == diff.max()).nonzero().tolist())
    if not torch.allclose(batched_zero, ref_zero, atol=1e-5, rtol=1e-5):
        diff = (batched_zero - ref_zero).abs()
        print(" max zero diff:", diff.max().item())


def main():
    for group_size in (32, 64, 128):
        for method in (ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID):
            test_group_size(group_size, method, sym=False)
    print("OK")


if __name__ == "__main__":
    main()
