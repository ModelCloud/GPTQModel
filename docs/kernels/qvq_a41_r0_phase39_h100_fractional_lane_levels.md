# Phase 39: H100 fractional-rate lane-pair level tables

Phase 39 promotes a block-local level-table layout for W2.5 and W3.5. It
removes approximately 44% of their random shared-load bank conflicts and
improves every focused gate/up cell and every complete-MLP cell at both rates.
The canonical P32 payload and checkpoint storage are unchanged.

## Profile diagnosis

Matched Nsight Compute 2026.2.1 captures of the production W2.5 and W3.5
K2048 x N8192 grouped gate/up launch showed the same decoder stream:

| Rate | Shared loads | Shared-load bank conflicts | Shared-load wavefronts | Executed instructions | Long scoreboard | Eligible warps/cycle |
|:--|--:|--:|--:|--:|--:|--:|
| W2.5 | 2,228,224 | 1,871,010 | 4,175,437 | 10,663,520 | 1.39 | 0.51 |
| W3.5 | 2,228,224 | 1,871,494 | 4,164,789 | 10,667,406 | 1.44 | 0.51 |

The two rates also execute essentially identical opcode counts. W3.5's extra
bit width changes staged trellis capacity but not the number of state, PGC, or
level operations. The dominant actionable difference from the faster W3 path
was therefore the compact random level table, not the lossless window payload.

## Layout math

Let `L[i]` be FP16 codebook level `i`, `l` the consumer lane, and

\[
p=\left\lfloor l/2\right\rfloor.
\]

The old fractional-rate layout stored one low and one high-permuted 256-entry
table. Random indices from a warp therefore repeatedly selected the same
32-bit shared banks.

The promoted table reuses the W3 lane-pair layout:

\[
T[i,2p]=L[i],
\qquad
T[i,2p+1]=L[i\mathbin{\oplus}(i\gg7)].
\]

Its byte address is

\[
A(i,l)=A_0+(i\ll6)+(l\ll1),
\]

and its bank is

\[
bank(i,l)=\left(16i+\left\lfloor l/2\right\rfloor\right)\bmod32.
\]

The even lane slot supplies the low-byte view and the odd lane slot supplies
the exact PGC high-byte permutation. Adjacent consumer pairs receive distinct
bank ownership while still sharing one 32-bit word. Decoder state extraction,
bank selection, PGC arithmetic, WGMMA order, and FP16 recovery are unchanged.

This is transient shared memory only. It adds no checkpoint bits per weight,
no persistent VRAM, no workspace, and no launch. W2.5 uses 44,800 bytes per
block. W3.5 uses 48,896 bytes, 256 bytes below the default 48 KiB Hopper block
limit, so neither rate needs opt-in dynamic shared memory and both retain the
two-stage TMA pipeline.

## Matched isolated H100 result

The physical 132-SM H100 passed the strict 0% utilization / 0 MiB admission
gate. Timing uses 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample for grouped gate/up inner P32 plus paired recovery.

| Rate | M/K/N per child | Production | Lane-pair | vs production | Better than last benchmark |
|:--|:--|--:|--:|--:|:--:|
| W2.5 | 1/2048/8192 x2 | 32.632 us | 30.941 us | 1.0546x | Yes |
| W2.5 | 2/2048/8192 x2 | 32.218 us | 30.533 us | 1.0552x | Yes |
| W2.5 | 4/2048/8192 x2 | 32.590 us | 30.882 us | 1.0553x | Yes |
| W2.5 | 8/2048/8192 x2 | 32.967 us | 31.299 us | 1.0533x | Yes |
| W2.5 | 16/2048/8192 x2 | 33.731 us | 32.165 us | 1.0487x | Yes |
| W3.5 | 1/2048/8192 x2 | 31.826 us | 30.264 us | 1.0516x | Yes |
| W3.5 | 2/2048/8192 x2 | 32.081 us | 30.129 us | 1.0648x | Yes |
| W3.5 | 4/2048/8192 x2 | 32.196 us | 30.413 us | 1.0586x | Yes |
| W3.5 | 8/2048/8192 x2 | 32.566 us | 30.883 us | 1.0545x | Yes |
| W3.5 | 16/2048/8192 x2 | 33.364 us | 31.601 us | 1.0558x | Yes |

Geometric means are **1.0534x** for W2.5 and **1.0570x** for W3.5.

## Matched profiler result

| Rate | Metric | Production | Lane-pair | Change |
|:--|:--|--:|--:|--:|
| W2.5 | Shared-load bank conflicts | 1,871,010 | 1,051,853 | -43.78% |
| W2.5 | Shared-load wavefronts | 4,175,437 | 3,337,238 | -20.07% |
| W2.5 | Executed instructions | 10,663,520 | 10,800,426 | +1.28% |
| W2.5 | Eligible warps/cycle | 0.51 | 0.58 | +13.7% |
| W3.5 | Shared-load bank conflicts | 1,871,494 | 1,050,858 | -43.85% |
| W3.5 | Shared-load wavefronts | 4,164,789 | 3,363,274 | -19.24% |
| W3.5 | Executed instructions | 10,667,406 | 10,800,468 | +1.25% |
| W3.5 | Eligible warps/cycle | 0.51 | 0.59 | +15.7% |

The result validates the bandwidth/conflict priority: a small increase in
address arithmetic wins because far fewer shared transactions serialize.

## Complete Llama 3.2 1B MLP

The full path includes shared input transform, grouped gate/up P32, exact
recovery, fused SiLU/product/down preconditioning, split-16 down P32, and down
recovery. Effective throughput uses logical dense-equivalent FLOPs. Marlin and
Machete are figurative W4 baselines and do not imply equal compressed work or
quantization quality. `Better` compares with accepted Phase 36.

| Rate | MKN: gate/up x2; down | QVQ | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|:--|--:|--:|--:|--:|:--:|
| W2 | 1/2048/8192 x2; 1/8192/2048 | 52.399 us | 1.921 | 0.556x | 0.936x | No |
| W2 | 2/2048/8192 x2; 2/8192/2048 | 52.660 us | 3.823 | 0.592x | 0.934x | No |
| W2 | 4/2048/8192 x2; 4/8192/2048 | 53.350 us | 7.547 | 0.588x | 0.928x | No |
| W2 | 8/2048/8192 x2; 8/8192/2048 | 53.794 us | 14.970 | 0.545x | 0.925x | No |
| W2 | 16/2048/8192 x2; 16/8192/2048 | 55.125 us | 29.217 | 0.590x | 0.901x | No |
| W2.5 | 1/2048/8192 x2; 1/8192/2048 | 51.661 us | 1.949 | 0.564x | 0.949x | Yes |
| W2.5 | 2/2048/8192 x2; 2/8192/2048 | 51.827 us | 3.885 | 0.601x | 0.949x | Yes |
| W2.5 | 4/2048/8192 x2; 4/8192/2048 | 52.637 us | 7.650 | 0.596x | 0.940x | Yes |
| W2.5 | 8/2048/8192 x2; 8/8192/2048 | 53.205 us | 15.136 | 0.551x | 0.935x | Yes |
| W2.5 | 16/2048/8192 x2; 16/8192/2048 | 54.249 us | 29.689 | 0.599x | 0.916x | Yes |
| W3 | 1/2048/8192 x2; 1/8192/2048 | 50.555 us | 1.991 | 0.576x | 0.970x | Yes |
| W3 | 2/2048/8192 x2; 2/8192/2048 | 50.906 us | 3.955 | 0.612x | 0.966x | No |
| W3 | 4/2048/8192 x2; 4/8192/2048 | 51.457 us | 7.825 | 0.610x | 0.962x | No |
| W3 | 8/2048/8192 x2; 8/8192/2048 | 52.056 us | 15.470 | 0.563x | 0.956x | Yes |
| W3 | 16/2048/8192 x2; 16/8192/2048 | 53.180 us | 30.286 | 0.611x | 0.934x | No |
| W3.5 | 1/2048/8192 x2; 1/8192/2048 | 51.719 us | 1.946 | 0.563x | 0.948x | Yes |
| W3.5 | 2/2048/8192 x2; 2/8192/2048 | 51.836 us | 3.884 | 0.601x | 0.948x | Yes |
| W3.5 | 4/2048/8192 x2; 4/8192/2048 | 52.101 us | 7.728 | 0.602x | 0.950x | Yes |
| W3.5 | 8/2048/8192 x2; 8/8192/2048 | 52.817 us | 15.247 | 0.555x | 0.942x | Yes |
| W3.5 | 16/2048/8192 x2; 16/8192/2048 | 53.779 us | 29.949 | 0.604x | 0.924x | Yes |

The targeted full-MLP geometric means are **1.0490x** for W2.5 and
**1.0560x** for W3.5. The complete 20-cell matrix improves **1.0259x** and
12/20 strict cells improve; all unchanged W2/W3 movements are sub-percent
cross-run variance.

## Correctness and scope

- All ten focused outputs are exact and CUDA Graph stable.
- The full MLP is bit-exact to the unfused reference for every rate and M.
- 126 Hopper P32/grouped tests pass, covering all W2-W3.5 rates, ordered
  reduction, exact plain-child parity, dense-oracle bounds, and graph replay.
- No kernel math, checkpoint bytes, output ordering, persistent VRAM, or
  architecture legality changed.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.

Artifacts:

- `artifacts/a41_phase39_h100/production_mlp_fractional_lane_levels_vs_phase36.json`
- `artifacts/a41_phase39_h100/fractional_lane_level_profile.json`
- binary Nsight reports outside Git under
  `/root/qvq-profiler-artifacts/phase39-fractional-decode/`

## Next phase

The compact fractional table was the remaining large bank-conflict source.
Phase 40 should profile W2, which still uses the compact table, and test the
same lane-pair mapping only if conflict counters and matched timing justify its
15 KiB block-local cost. If W2 is already limited elsewhere, return to the
remaining window-address/PGC dependency chain rather than increasing storage.
