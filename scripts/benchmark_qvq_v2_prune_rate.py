# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Measure achievable Cauchy-Schwarz bound-prune rates for the QVQ Viterbi recurrence.

Simulates the coupled-bank G-recurrence with the kernel's prefix-major scan
order over production-style [tiles, 128, 2] sequences built from a real
checkpoint, counting candidates whose valid lower bound
``G_prev[pred] + (sqrt(tn) - sqrt(norm_s))^2`` exceeds the running best.
Skipping such candidates preserves the exact argmin, so the reported rate is
the exact-pruning headroom for a bound-pruned kernel.

Usage:
    python scripts/benchmark_qvq_v2_prune_rate.py --model /path/to/safetensors/dir
"""

from __future__ import annotations



import glob
import os

import torch
from safetensors.torch import load_file

from gptqmodel.quantization.qvq_codecs import pgc16_codebook_v2_bank


def load_weight_matrices(snapshot_dir: str, max_matrices: int = 8):
    matrices = []
    for path in sorted(glob.glob(os.path.join(snapshot_dir, "*.safetensors"))):
        tensors = load_file(path)
        for name, t in tensors.items():
            if t.ndim == 2 and "embed" not in name and "lm_head" not in name and t.size(0) >= 16:
                matrices.append((name, t))
                if len(matrices) >= max_matrices:
                    return matrices
    return matrices


def build_sequences(w: torch.Tensor, slab_rows: int = 16) -> torch.Tensor:
    """Production-style tiles: [ntiles, 128, 2] from contiguous 16-row slabs."""
    device = w.device if w.is_cuda else "cuda"
    w = w.to(device=device, dtype=torch.float32)
    rows = min(slab_rows, w.size(0))
    cols = (w.size(1) // 256) * 256
    if rows < 16 or cols < 256:
        return None
    max_tiles = 4096
    slabs = []
    for start in range(0, w.size(0) - rows + 1, rows):
        block = w[start : start + rows, :cols]
        tiles = block.reshape(rows, cols // 16, 16).permute(1, 0, 2)
        slabs.append(tiles.reshape(-1, 128, 2))
        if sum(s.size(0) for s in slabs) >= max_tiles:
            break
    seq = torch.cat(slabs)[:max_tiles]
    return seq.contiguous()


def simulate_prune(seq: torch.Tensor, cb: torch.Tensor, shift: int, steps: int = 12):
    """Vectorized scan-order prune simulation. seq [T,128,2], cb [banks,65536,2]."""
    banks = cb.size(0)
    c = cb.float()
    norms = c[..., 0] ** 2 + c[..., 1] ** 2
    prefix_count = 1 << shift
    suffix_count = 1 << (16 - shift)

    h = torch.arange(prefix_count, device=seq.device).view(-1, 1)
    x = torch.arange(suffix_count, device=seq.device).view(1, -1)
    pred = (h * suffix_count + x) >> shift  # [prefix, suffix]

    total = 0
    skipped = 0
    batch_chunks = torch.split(seq, 64, dim=0)
    G = None
    for chunk in batch_chunks:
        T = chunk.size(0)
        # per-bank frontiers [banks*T, suffix]; treat each sequence independently
        G = torch.zeros((banks, T, suffix_count), device=seq.device)
        for step in range(steps):
            t = chunk[:, step]  # [T,2]
            tn = (t * t).sum(-1)  # [T]
            d = ((c.unsqueeze(2) - t.view(1, 1, T, 2)) ** 2).sum(-1)  # [b,65536,T]
            d = d.view(banks, prefix_count, suffix_count, T)
            best = torch.full((banks, T, suffix_count), float("inf"), device=seq.device)
            for hi in range(prefix_count):
                gp = G.gather(2, pred[hi].view(1, 1, -1).expand(banks, T, -1))
                rs = norms[:, hi * suffix_count : (hi + 1) * suffix_count].sqrt()
                bound = gp + (
                    torch.sqrt(tn).view(1, T, 1) - rs.unsqueeze(1)
                ).pow(2)
                skip = bound > best
                dv = d[:, hi]  # [b, suffix, T]
                val = gp + dv.transpose(1, 2)  # [b, T, suffix]
                # do it cleanly below instead
                val = None
                dv = d[:, hi]  # [b, suffix, T]
                val = gp + dv.transpose(1, 2)  # [b, T, suffix]
                best = torch.where(skip, best, torch.minimum(best, val))
                total += skip.numel()
                skipped += int(skip.sum())
            # advance G exactly
            acc = None
            for hi in range(prefix_count):
                gp2 = G.gather(2, pred[hi].view(1, 1, -1).expand(banks, T, -1))
                dv = d[:, hi]
                v = gp2 + dv.transpose(1, 2)
                acc = v if acc is None else torch.minimum(acc, v)
            G = acc
        del G
    return skipped / max(total, 1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str,
        default="/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
                "c1899de289a04d12100db370d81485cdf75e47ca",
        help="Directory containing *.safetensors weight shards",
    )
    parser.add_argument("--max-matrices", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    snap = args.model
    matrices = load_weight_matrices(snap, args.max_matrices)
    print(f"loaded {len(matrices)} matrices")
    configs = ((6, 4, 3.0), (5, 2, 2.5))
    results = {}
    for name, w in matrices:
        seq = build_sequences(w)
        if seq is None:
            print(f"skip {name}: too small")
            continue
        row = []
        for shift, banks, bits in configs:
            cb = torch.stack(
                tuple(pgc16_codebook_v2_bank(b, bits=bits, dtype=torch.float32) for b in range(banks))
            ).to("cuda")
            rate = simulate_prune(seq, cb, shift, steps=args.steps)
            row.append(f"s{shift}:{100 * rate:.1f}%")
            results.setdefault(shift, []).append(rate)
            del cb
        print(f"{name:<28} {seq.size(0):>5} tiles  " + "  ".join(row))
    print("\nsummary (mean over matrices):")
    for shift, rates in results.items():
        print(f"  shift {shift}: {100 * sum(rates) / len(rates):.1f}% prunable")


if __name__ == "__main__":
    main()
