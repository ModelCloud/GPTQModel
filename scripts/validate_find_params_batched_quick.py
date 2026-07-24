#!/usr/bin/env python3
"""Quick targeted strict check for known failing configurations."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization import QuantizeConfig, Quantizer, ScaleSearchConfig  # noqa: E402


def build_hessians(columns: int, group_size: int, device: str, generator: torch.Generator):
    H = torch.randn(columns, max(columns // 2, 1), generator=generator, device=device, dtype=torch.float32)
    H = H.matmul(H.t())
    H = (H + H.t()) * 0.5
    blocks = []
    diagonals = []
    for g in range(0, columns, group_size):
        ge = min(g + group_size, columns)
        blocks.append(H[g:ge, g:ge])
        diagonals.append(H.diagonal()[g:ge])
    return torch.stack(blocks, dim=0), torch.stack(diagonals, dim=0)


def check(rows: int, columns: int, group_size: int, method: ScaleSearchConfig, sym: bool, bits: int, seed: int):
    device = "cuda:0"
    generator = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn((rows, columns), generator=generator, device=device, dtype=torch.float32)
    H_blocks, H_diag = build_hessians(columns, group_size, device, generator)

    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=False,
        offload_to_disk=False,
        mse=2.0,
        scale_search=method,
    )
    q = Quantizer(qcfg)
    q.configure(perchannel=True, grid=100, maxshrink=0.8)
    q.maxq = q.maxq.to(device)

    ref_scales, ref_zeros = [], []
    for g in range(0, columns, group_size):
        ge = min(g + group_size, columns)
        if method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
            h = H_blocks[g // group_size]
        else:
            h = H_diag[g // group_size]
        q.find_params(W[:, g:ge], weight=True, hessian=h)
        ref_scales.append(q.scale.squeeze().clone())
        ref_zeros.append(q.zero.squeeze().clone())
    ref_scale = torch.stack(ref_scales, dim=1)
    ref_zero = torch.stack(ref_zeros, dim=1)

    W_3d = W.reshape(rows, columns // group_size, group_size)
    if method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
        batched_h = H_blocks
    else:
        batched_h = H_diag
    batched_scale, batched_zero = q.find_params_batched(W_3d, weight=True, hessian=batched_h)

    sdiff = (batched_scale - ref_scale).abs()
    zdiff = (batched_zero - ref_zero).abs()
    return sdiff.max().item(), zdiff.max().item()


def main():
    failed = False
    configs = [
        (4096, 512, 32, False, 2),
        (4096, 512, 64, False, 2),
        (4096, 512, 128, False, 2),
        (4096, 512, 32, True, 2),
        (4096, 512, 64, True, 2),
        (4096, 512, 128, True, 2),
        (4096, 512, 64, False, 4),
        (4096, 512, 64, True, 4),
    ]
    for rows, cols, gs, sym, bits in configs:
        for method in (ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID):
            smax, zmax = check(rows, cols, gs, method, sym, bits, 999)
            ok = smax < 1e-4 and zmax < 1e-4
            print(f"rows={rows} cols={cols} gs={gs} sym={sym} bits={bits} method={method.value}: s={smax:.3e} z={zmax:.3e} {'OK' if ok else 'FAIL'}", flush=True)
            if not ok:
                failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
