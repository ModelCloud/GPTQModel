# QVQ P32 Hopper large-M optimization, third pass

## Scope and baseline

This pass starts from merged `origin/main` commit `f74ec0cc`, which contains
PR #112. It targets the complete Llama 3.2 1B MLP on the physical 132-SM
NVIDIA H100. All timings use warmed CUDA Graph replay and CUDA events. Marlin
and Machete use W4 and remain figurative latency/efficiency baselines for the
lower-rate QVQ W2-W3.5 kernels.

The MLP contains two `M x 2048 x 8192` gate/up projections, exact FP16
recovery and SwiGLU/down preconditioning, and one `M x 8192 x 2048` down
projection.

## Phase 1: reuse one decoded P32 fragment across M128

The merged kernel owns four independent M16 activation tiles per CTA. For one
P32 fragment `Q_f`, it computes

```text
A_f = decode_p32(Q_f)
C_0 += A_f @ B_0
C_1 += A_f @ B_1
C_2 += A_f @ B_2
C_3 += A_f @ B_3
```

and repeats the complete compressed-weight decode for the next M64 slab. The
new kernel retains eight independent FP32 accumulators and eight TMA-staged
M16 activation tiles:

```text
A_f = decode_p32(Q_f)
for r in 0..7:
    C_r += A_f @ B_r
```

Thus the exact same decoded register fragment feeds M128 rather than M64.
There is no change to the P32 checkpoint, selector stream, codebook, input
transform, K traversal, FP32 accumulation within an output element, or FP16
recovery boundary. Runtime dispatch is restricted to the measured physical
H100 Llama gate/up geometry, unsplit children, `M >= 128`, and `M % 128 == 0`.
Other shapes retain reuse-4 or the established fallback.

The named low-level operator is
`p32_window_m128_tma_grouped_reuse8`. The Python wrapper rejects split-K
children and invalid row geometry before launch. Production telemetry records
`h100_reuse8_gate_up_launches` independently.

## Correctness and CUDA Graph safety

The real `2048 -> 8192 + 8192` low-level test covers W2, W2.5, W3, and W3.5.
Reuse-8 must be bit-exact to reuse-4, and the captured/replayed CUDA Graph
must remain bit-exact. The runtime test separately checks H100 dispatch,
telemetry, and the absence of plain fallback. All five focused tests pass.

Across the full production benchmark the maximum absolute error against the
dense FP32 P32 Torch oracle is `1.073e-6`, below the `2e-3` acceptance bound.

## H100 production benchmark

The executable kernel is commit `3d670c08`. `Better than last` is the strict
median comparison with the fresh merged-main artifact. Ratios above one in
the baseline columns mean QVQ is faster.

| Weight | MLP MKN (gate/up; down) | Merged main | Reuse-8 | Speedup | vs Marlin W4 | vs Machete W4 | Better than last |
|---|---|---:|---:|---:|---:|---:|:---:|
| W2 | 128x2048x8192; 128x8192x2048 | 122.352 us | 115.046 us | 1.064x | 0.691x | 0.638x | Yes |
| W2 | 512x2048x8192; 512x8192x2048 | 410.857 us | 381.370 us | 1.077x | 0.434x | 0.307x | Yes |
| W2 | 4096x2048x8192; 4096x8192x2048 | 3156.976 us | 2968.358 us | 1.064x | 0.471x | 0.308x | Yes |
| W2.5 | 128x2048x8192; 128x8192x2048 | 123.860 us | 112.987 us | 1.096x | 0.703x | 0.649x | Yes |
| W2.5 | 512x2048x8192; 512x8192x2048 | 422.785 us | 388.721 us | 1.088x | 0.426x | 0.301x | Yes |
| W2.5 | 4096x2048x8192; 4096x8192x2048 | 3204.343 us | 2918.758 us | 1.098x | 0.479x | 0.313x | Yes |
| W3 | 128x2048x8192; 128x8192x2048 | 122.546 us | 116.418 us | 1.053x | 0.683x | 0.630x | Yes |
| W3 | 512x2048x8192; 512x8192x2048 | 420.238 us | 385.756 us | 1.089x | 0.430x | 0.303x | Yes |
| W3 | 4096x2048x8192; 4096x8192x2048 | 3183.490 us | 2975.374 us | 1.070x | 0.470x | 0.307x | Yes |
| W3.5 | 128x2048x8192; 128x8192x2048 | 123.464 us | 114.809 us | 1.075x | 0.692x | 0.639x | Yes |
| W3.5 | 512x2048x8192; 512x8192x2048 | 433.610 us | 396.394 us | 1.094x | 0.418x | 0.295x | Yes |
| W3.5 | 4096x2048x8192; 4096x8192x2048 | 3283.414 us | 3028.471 us | 1.084x | 0.461x | 0.302x | Yes |

All twelve cells improve. The geometric-mean speedup over merged main is
`1.0792x`; the Marlin and Machete geometric ratios are `0.5180x` and
`0.3897x`, respectively.

## Exact-commit Nsight Compute and SASS audit

The W3 M512 gate/up kernel at commit `3d670c08` was collected with Nsight
Compute 2026.2.1 using SpeedOfLight, LaunchStats, Occupancy, SchedulerStats,
WarpStateStats, InstructionStats, and MemoryWorkloadAnalysis. The matched
merged-main report used the same command and physical H100.

| Metric | M64 reuse-4 | M128 reuse-8 | Change |
|---|---:|---:|---:|
| NCU duration | 178.30 us | 154.24 us | 1.156x |
| Executed warp instructions | 102.93 M | 62.01 M | -39.76% |
| Grid blocks | 1024 | 512 | -50% |
| Registers/thread | 91 | 137 | +46 |
| Achieved occupancy | 26.63% | 13.81% | -12.82 points |
| Eligible warps/scheduler | 1.54 | 0.63 | -0.91 |
| DRAM throughput | 6.24% | 7.39% | +1.15 points |

The emitted dynamic opcode mix demonstrates algebraic/representation reuse,
not a timing-only effect. `IMAD` falls from 19.93M to 9.99M, `PRMT` from
12.68M to 6.34M, `LOP3` from 11.20M to 5.64M, shared halfword/word loads from
8.39M each to 4.19M each, and 64-bit funnel shifts from 4.19M to 2.10M.
Tensor-core `HGMMA` remains exactly 4.19M because useful matrix work is
unchanged.

The higher register footprint and lower occupancy are real costs, but the
40% instruction reduction wins. The next phase should attempt to recover
scheduler readiness without giving up M128 decode reuse, preferably by
sharing the same eight input tiles across more independent N64 consumer
warpgroups. Any wider-CTA candidate must stay within H100 shared-memory and
register limits and is rejected on spills or full-MLP regression.
