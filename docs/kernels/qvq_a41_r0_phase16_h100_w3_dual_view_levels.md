# QVQ A41/R0 Phase 16: dual-view W3 level table

Phase 16 uses the two FP16 lane slots already allocated by Phase 15 as two
exact views of the W3 PGC level table. The even slot stores the canonical
level and the odd slot stores the fixed high-byte permutation. A low-byte
decode addresses the even slot; a high-byte decode addresses the odd slot.
This deletes four high-index permutations from every decoded fragment while
retaining Phase 15's shared-memory bank mapping and 16-KiB table footprint.

On the physical 132-SM NVIDIA H100, grouped W3 gate/up improves **1.0272x**
geometric mean over Phase 15 and the complete Llama 3.2 1B W3 MLP improves
**1.0233x**, with all five M values positive. Matched Nsight Compute profiling
shows 1.001M fewer executed warp instructions and no return of the shared-bank
conflicts removed in Phase 15.

## Exact dual-view math

Let `L[i]` be canonical FP16 level `i`, `b` an unpermuted high byte, and `l`
the consumer lane. Phase 15 stored the same value in both half-word slots of
each lane pair and transformed every high index in the decode loop:

\[
h(b)=b\oplus(b\gg7),\qquad y_{high}=L[h(b)].
\]

Phase 16 stores two views in the same index-major table:

\[
T[i,l]=
\begin{cases}
L[i], & l\bmod2=0,\\
L[i\oplus(i\gg7)], & l\bmod2=1.
\end{cases}
\]

For physical consumer lane `l`, define:

\[
l_{low}=l\mathbin{\&}\sim1,
\qquad
l_{high}=l\mathbin{|}1.
\]

The two lookups are therefore:

\[
y_{low}=T[i_{low},l_{low}]=L[i_{low}],
\]

\[
y_{high}=T[b,l_{high}]=L[b\oplus(b\gg7)].
\]

The result is identical to Phase 15 for every byte value. The high-byte
permutation moves from every hot-loop decode to the one-time shared-table
preload. No PGC state, selector, FP16 level bit, WGMMA operand, or reduction
order changes.

## Address and storage mapping

The byte address remains:

\[
A(i,l)=A_0+(i\ll6)+(l\ll1).
\]

The shared bank therefore remains:

\[
bank(i,l)=\left(16i+\left\lfloor l/2\right\rfloor\right)\bmod32.
\]

Low and high lookups select the even and odd half words of the same 32-bit
bank word. The table remains `256*32` FP16 values, or exactly 16 KiB. W3 total
static shared memory remains 45.824 KiB per block, below Hopper's default
48-KiB block limit. Checkpoint bytes, canonical/window payloads, global
workspace, CUDA Graph nodes, and persistent VRAM are unchanged.

During preload, one 32-bit pair contains:

```text
low 16 bits  = bits(L[index])
high 16 bits = bits(L[index xor (index >> 7)])
```

Four identical `uint4` stores replicate that pair across the 32 lane slots.
The conflict-aware flattened Phase-15 initialization order is retained.

## Grouped gate/up A/B

The Phase-15 and Phase-16 artifacts use the same physical H100, payload/input
seeds, strict zero-MiB idle admission, CUDA Graph replay, and exactness checks.
Timed work contains grouped W3 gate/up P32 plus exact paired recovery.

| M | Phase 15 us | Dual-view us | Speedup | Better |
|---:|---:|---:|---:|:---:|
| 1 | 30.853 | 30.019 | 1.0278x | Yes |
| 2 | 30.943 | 30.074 | 1.0289x | Yes |
| 4 | 31.349 | 30.531 | 1.0268x | Yes |
| 8 | 31.683 | 30.862 | 1.0266x | Yes |
| 16 | 32.474 | 31.661 | 1.0257x | Yes |

The grouped gate/up geometric-mean speedup is **1.0272x**. Every output is
FP16 bit-exact to Phase 15, repeatable over ordinary launches, and CUDA Graph
stable.

## Matched Nsight Compute and SASS

Nsight Compute 2026.2.1 profiled the same W3 M1 grouped gate/up kernel before
and after the dual-view change. Reports and raw/source CSVs are intentionally
kept outside Git under:

```text
/root/qvq-profiler-artifacts/phase16-dual-view-levels/
```

| Metric | Phase 15 | Phase 16 | Change |
|:--|--:|--:|--:|
| NCU replay duration | 28.992 us | 28.192 us | **1.0284x** |
| executed warp instructions | 11,814,964 | 10,813,792 | **-8.47%** |
| `LOP3` | 1,913,856 | 1,405,952 | -507,904 |
| `SHF` | 1,892,608 | 1,376,512 | -516,096 |
| `LDS` | 2,228,224 | 2,228,224 | unchanged |
| `PRMT` | 1,589,248 | 1,589,248 | unchanged |
| shared-load bank conflicts | 1,052,722 | 1,050,937 | effectively unchanged |
| shared-load wavefronts | 3,343,923 | 3,350,885 | effectively unchanged |
| conflict share | 31.48% | 31.36% | effectively unchanged |
| eligible warps/cycle | 0.634 | 0.587 | lower |
| WGMMA stall / issue-active cycle | 0.218 | 0.272 | higher |
| wait stall | 0.958 | 0.923 | lower |
| long-scoreboard stall | 1.330 | 1.402 | higher |
| active warps | 14.763% | 15.101% | +0.338 points |
| registers/thread | 58 | 60 | +2 |
| static shared memory | 45.824 KiB | 45.824 KiB | unchanged |
| local/shared spills | 0 / 0 | 0 / 0 | unchanged |

The two half-million-instruction reductions match the four removed high-byte
permutations per decoded fragment. Scheduler ratios do not all improve, but
the candidate retains the conflict reduction and shortens the dependent
decode chain enough to reduce kernel duration by 2.84%.

## Complete Llama 3.2 1B MLP

The formal post-commit artifact executes production SHA `6dc033b8` with 30
warmups, 200 CUDA-event samples, and 50 CUDA Graph replays per sample. `vs` is
comparator latency divided by QVQ latency, so values below one mean the W4
comparator is faster. `Better` compares with the committed Phase-15 artifact;
`No` records a regression.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 15 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 67.489 | 1.492 | 0.431x | 0.741x | 1.0051x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 67.622 | 2.977 | 0.461x | 0.736x | 1.0004x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 68.377 | 5.889 | 0.459x | 0.735x | 1.0008x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 68.981 | 11.674 | 0.425x | 0.732x | 1.0009x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.400 | 25.010 | 0.505x | 0.784x | 1.0007x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.683 | 1.466 | 0.423x | 0.728x | 0.9976x | No |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.748 | 2.928 | 0.454x | 0.723x | 1.0004x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.146 | 5.823 | 0.454x | 0.726x | 1.0013x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.937 | 11.515 | 0.420x | 0.722x | 1.0018x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.140 | 24.726 | 0.500x | 0.775x | 1.0009x | Yes |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 65.664 | 1.533 | 0.443x | 0.762x | 1.0246x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 65.781 | 3.061 | 0.474x | 0.756x | 1.0228x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 66.330 | 6.070 | 0.473x | 0.757x | 1.0212x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 66.896 | 12.038 | 0.439x | 0.755x | 1.0230x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 62.215 | 25.888 | 0.523x | 0.812x | 1.0250x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.236 | 1.475 | 0.426x | 0.733x | 0.9999x | No |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.918 | 2.921 | 0.453x | 0.722x | 1.0010x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.499 | 5.794 | 0.452x | 0.723x | 0.9977x | No |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.741 | 11.547 | 0.421x | 0.724x | 0.9981x | No |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.027 | 24.768 | 0.501x | 0.776x | 1.0007x | Yes |

The targeted W3 cells improve **5/5**, with **1.0233x** geometric-mean
complete-MLP speedup. All-rate geometric mean is **1.0061x** versus Phase 15,
**2.0797x** versus ordinary per-module QVQ, **0.4559x** versus Marlin W4,
and **0.7456x** versus Machete W4. Unchanged W2/W2.5/W3.5 code shows small
cross-run movement; it is telemetry, not a Phase-16 effect. The W4 comparisons
are figurative dense-equivalent throughput baselines, not equal-work or
equal-quantization-quality claims.

## Correctness and promotion gates

- The 59-test grouped Hopper suite passes on the physical H100 across
  W2/W2.5/W3/W3.5 and M=1/2/4/8/16.
- Tests cover dense-oracle error, bit-exact plain-child equivalence, ordered
  split reduction, repeatability, CUDA Graph stability, and real Llama shapes.
- Grouped gate/up A/B output is FP16 bit-exact at all five M values.
- The complete MLP artifact passes its existing accuracy/reference checks.
- Formal measurement admitted the H100 only after three 0%-utilization,
  zero-MiB-memory samples.
- Builds use at most four Ninja jobs, one NVCC host thread, and one CUDA split
  compile partition.

Artifacts:

- `artifacts/a41_phase16_h100/production_mlp_w3_dual_view_vs_phase15.json`
- `artifacts/a41_phase16_h100/w3_single_view_baseline_all_m.json`
- `artifacts/a41_phase16_h100/w3_dual_view_candidate_all_m.json`

## Next experiment

The decoder experiments that followed Phase 16 did not satisfy the complete
MLP promotion gate. Phase 17 therefore moved to the next measurable operation
boundary and made the down precondition write its final M16-padded input
directly. See
`docs/kernels/qvq_a41_r0_phase17_h100_direct_padding.md`.
