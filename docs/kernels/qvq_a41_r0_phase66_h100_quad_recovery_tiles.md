# Phase 66: rejected H100 four-output recovery tiles

Phase 66 extended Phase 64's exact paired-output recovery tree from two to
four output tiles. Tiles `t`, `t+8`, `t+16`, and `t+24` share their first
three 32-point high-transform stages. This reduces four independent selected
trees from 128 loads and 124 rounded operations to 32 loads and 36 rounded
operations.

The candidate was bit exact and spill-free, but every measured M regressed.
All candidate CUDA, Python, schema, and test code was removed; production
remains Phase 64.

## Exact tree

After the first three ascending stages, four rounded values remain:

```text
a0 = R(v0 + v1)    a1 = R(v2 + v3)
d0 = R(v0 - v1)    d1 = R(v2 - v3)

output(t)      = R(a0 + a1)
output(t + 8)  = R(d0 + d1)
output(t + 16) = R(a0 - a1)
output(t + 24) = R(d0 - d1)
```

This is exactly the same balanced ascending-stage tree as four Phase-63
selected outputs. Each resulting column then independently executes the
unchanged recovery scale/bias, FP16 SiLU/product/down-scale boundary, and
packed-FP16 low transform.

## Same-binary H100 result

Timing used 30 warmups, 300 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample after three spaced 0% utilization / 0 MiB H100 admission
samples. M16 control is Phase 64's two-output pairing; smaller rows use the
Phase-63 middle kernel. Every candidate output was bit exact.

| M | M/K/N gate/up x2 | Phase 64 | Four-output tiles | vs Phase 64 | Better |
|--:|:--|--:|--:|--:|:--:|
| 1 | 1/2048/8192 x2 | 7.613 us | 9.322 us | 0.8167x | No |
| 2 | 2/2048/8192 x2 | 7.722 us | 9.649 us | 0.8003x | No |
| 4 | 4/2048/8192 x2 | 7.933 us | 9.978 us | 0.7950x | No |
| 8 | 8/2048/8192 x2 | 8.670 us | 10.354 us | 0.8374x | No |
| 16 | 16/2048/8192 x2 | 9.771 us | 11.139 us | 0.8772x | No |

CUDA `cuobjdump` reports 48 registers/thread, zero stack bytes, zero local
bytes, and 2 KiB of user shared memory. The M16 grid is 128 blocks, nearly
one block per H100 SM, but every CTA serializes four nonlinear and low-
transform output paths. The additional per-CTA dependency depth costs more
than the shared recovery subtree saves.

## Decision

- No Phase-66 candidate remains in production source.
- Two-output pairing is the measured sharing limit for the current fused
  middle-kernel ownership model.
- Quantization, checkpoints, workspaces, persistent VRAM, and non-H100 paths
  remain unchanged.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.
