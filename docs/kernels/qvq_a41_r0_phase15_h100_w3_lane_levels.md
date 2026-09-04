# QVQ A41/R0 Phase 15: bank-conflict-reduced W3 level lookup

Phase 15 replaces W3's single shared PGC level table with an index-major,
lane-interleaved table. Each of the 256 FP16 codebook levels is replicated
across 32 lane slots inside the thread block's shared memory. A consumer lane
loads only its own slot, restricting an arbitrary level lookup to its lane
pair's two shared banks instead of letting all 32 lanes contend across one
ordinary 256-entry table.

The accepted implementation deliberately spends more integer instructions
and 15.488 KiB more shared memory per active W3 block. Matched H100 profiling
shows why this is profitable: shared-load bank conflicts fall **43.8%** and
shared-load wavefronts fall **20.2%**, while grouped gate/up improves
**1.0348x** geometric mean. No model/checkpoint/global VRAM is added.

## Exact table and bank mapping

Let `L[i]` be canonical FP16 level `i`, and let `l` be consumer lane 0--31.
The W3 shared table stores the exact replication:

\[
L_{lane}[i,l]=L[i],\qquad 0\leq i<256.
\]

Its byte address is:

\[
A(i,l)=A_0+2(32i+l)=A_0+(i\ll6)+(l\ll1).
\]

With 32 four-byte shared banks, the bank number is:

\[
bank(i,l)=\left(16i+\left\lfloor l/2\right\rfloor\right)\bmod32.
\]

Therefore lane pair `floor(l/2)` owns bank `p` for even indices and bank
`p+16` for odd indices. Different lane pairs cannot collide with each other;
only the two lanes in one pair can request the same bank. This bounds a level
load to two wavefronts while preserving the original value bit-for-bit.

The previous high-byte view stored:

\[
L_{high}[b]=L[b\oplus(b\gg7)].
\]

W3 now computes the same index explicitly:

\[
h=b\oplus(b\gg7)
\]

and loads `L_lane[h,l]`. The low-index helper already produces the exact byte
offset:

\[
o=2((p\oplus(p\gg7))\mathbin{\&}255).
\]

The lane-interleaved address is then `A0 + 32*o + 2*l`. No PGC state,
codebook value, FP16 bit, WGMMA operand, or accumulation order changes.

## Conflict-aware initialization

One W3 block needs `256*32` half values, or 16 KiB. Four identical `uint4`
vectors represent the 32 lane copies of one level. The initial prototype let
one thread write all four vectors for an index; adjacent threads were 64 bytes
apart and NCU reported a 16-way initialization-store conflict.

The promoted path flattens the 1,024 vectors:

```cpp
for (entry = thread; entry < 256 * 4; entry += 160) {
    index = entry >> 2;
    vectors[entry] = replicated_level(index);
}
```

Consecutive lanes now write consecutive 128-bit vectors. The final memory
profile no longer reports the initialization-store warning, and its M1 event
median improves from 31.355 us in the first prototype to 31.113 us in the
focused run.

## Candidate history

Before changing the table layout, Phase 15 tested four W3 register fragments
in flight. It remained exact but regressed M1 grouped gate/up from the accepted
depth-three 32.059 us to 33.138 us (0.967x). The fourth fragment was reverted;
production retains Phase 14's depth three.

## Matched grouped gate/up A/B

The depth-three baseline and lane-table candidate use the same physical H100,
payload/input seeds, strict zero-MiB idle admission, 30 warmups, 300
CUDA-event samples, and 50 CUDA Graph replays per sample. Timed work includes
grouped W3 gate/up inner P32 and exact paired recovery. Every result is
bit-exact, repeatable over ten launches, and graph-stable.

| M | Ordinary shared table us | Lane table us | Speedup | Better |
|---:|---:|---:|---:|:---:|
| 1 | 32.059 | 30.853 | 1.0391x | Yes |
| 2 | 32.084 | 30.943 | 1.0369x | Yes |
| 4 | 32.366 | 31.349 | 1.0324x | Yes |
| 8 | 32.739 | 31.683 | 1.0333x | Yes |
| 16 | 33.517 | 32.474 | 1.0321x | Yes |

The geometric-mean grouped gate/up improvement is **1.0348x**.

## Matched Nsight Compute / SASS

Nsight Compute 2026.2.1 captured the accepted Phase-14 W3 M1 depth-three
kernel and the promoted lane-table kernel. Separate `MemoryWorkloadAnalysis`
table captures measure shared conflicts; the standard profile set measures
instructions, resources, and scheduler state. Reports and raw/source CSVs are
outside Git under:

```text
/root/qvq-profiler-artifacts/phase15-level-conflicts/
```

| Metric | Phase 14 | Phase 15 | Change |
|:--|--:|--:|--:|
| shared-load requests | 2,228,224 | 2,228,224 | unchanged |
| shared-load bank conflicts | 1,871,630 | 1,052,722 | **-43.8%** |
| shared-load wavefronts | 4,191,896 | 3,343,923 | **-20.2%** |
| average shared-load conflict | 1.9-way | 1.5-way | lower |
| conflict share of load wavefronts | 44.65% | 31.48% | **-13.17 points** |
| NCU replay duration | 29.472 us | 28.992 us | **1.0166x** |
| executed warp instructions | 10,677,982 | 11,814,964 | +10.65% |
| eligible warps/cycle | 0.536 | 0.634 | **+18.15%** |
| WGMMA stall / issue-active cycle | 0.377 | 0.218 | **-42.1%** |
| long-scoreboard stall | 1.459 | 1.330 | **-8.9%** |
| active warps | 15.174% | 14.763% | -0.411 points |
| registers/thread | 56 | 58 | +2 |
| static shared memory | 30.336 KiB | 45.824 KiB | +15.488 KiB |
| shared-memory block limit | 7 | 4 | lower |
| local/shared spills | 0 / 0 | 0 / 0 | unchanged |
| `LDS` | 2,228,224 | 2,228,224 | unchanged |
| `LOP3` | 1,383,424 | 1,913,856 | +530,432 |
| `SHF` | 1,362,176 | 1,892,608 | +530,432 |
| `HGMMA` | 131,072 | 131,072 | unchanged |

This is direct evidence that instruction count is not the governing metric for
this decoder. The lane table executes 1.137M more warp instructions and lowers
the shared-memory residency limit, yet it issues useful work more often and
runs faster because the dependent level loads need 848K fewer wavefronts.

## Complete Llama 3.2 1B MLP

The post-commit artifact executes production SHA `d09e2b33` with 30 warmups,
200 CUDA-event samples, and 50 CUDA Graph replays per sample. It contains the
complete grouped gate/up, recovery, exact SiLU/down precondition, down P32,
and down recovery path. `vs` is comparator latency divided by QVQ latency;
values below one mean the W4 comparator is faster. `Better` compares with the
committed Phase-14 artifact and records `No` for a regression.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 14 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 67.830 | 1.484 | 0.431x | 0.745x | 0.9962x | No |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 67.651 | 2.976 | 0.463x | 0.735x | 0.9980x | No |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 68.433 | 5.884 | 0.461x | 0.732x | 0.9996x | No |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.040 | 11.664 | 0.427x | 0.727x | 1.0001x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.442 | 24.993 | 0.506x | 0.778x | 0.9985x | No |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.516 | 1.469 | 0.426x | 0.738x | 1.0015x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.775 | 2.927 | 0.456x | 0.723x | 0.9994x | No |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.234 | 5.816 | 0.456x | 0.723x | 1.0005x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 70.062 | 11.494 | 0.421x | 0.717x | 1.0015x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.201 | 24.702 | 0.501x | 0.769x | 0.9999x | No |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 67.279 | 1.496 | 0.434x | 0.751x | 1.0125x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 67.283 | 2.992 | 0.466x | 0.739x | 1.0125x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 67.735 | 5.945 | 0.466x | 0.739x | 1.0135x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 68.435 | 11.768 | 0.431x | 0.734x | 1.0113x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 63.772 | 25.256 | 0.512x | 0.787x | 1.0131x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.228 | 1.475 | 0.428x | 0.741x | 0.9994x | No |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.991 | 2.918 | 0.454x | 0.721x | 0.9934x | No |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.337 | 5.807 | 0.455x | 0.722x | 0.9964x | No |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.607 | 11.569 | 0.423x | 0.722x | 0.9984x | No |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.073 | 24.751 | 0.502x | 0.771x | 0.9988x | No |

The targeted W3 cells improve **5/5**, with **1.0126x** geometric-mean
complete-MLP speedup versus Phase 14. Across all rates, including unchanged
rates and cross-run noise, geometric mean is **1.0022x**. Other all-cell
geometric means are **2.0769x** versus ordinary per-module QVQ, **0.4550x**
versus Marlin W4, and **0.7405x** versus Machete W4. The W4 comparisons are
figurative dense-equivalent efficiency baselines, not equal-work or
equal-quantization-quality claims.

## Correctness, storage, and validation

- The 59-test grouped Hopper suite passes across every supported rate and
  M=1/2/4/8/16, including dense-oracle bounds, independent-child equality,
  ordered reduction, repeatability, CUDA Graph replay, and real Llama shapes.
- The matched W3 A/B output is FP16 bit-exact at all five M values and stable
  over ten ordinary launches plus graph replay.
- The complete MLP matrix passes its existing accuracy/reference checks.
- Canonical/window payload bytes, checkpoint size, global transient workspace,
  CUDA Graph nodes, and persistent VRAM are unchanged.
- W3 shared memory rises by 15.488 KiB per resident block only; a compile-time
  assertion keeps total static shared storage below Hopper's 48-KiB default
  per-block limit.
- Builds use at most four Ninja jobs, one NVCC host thread, and one CUDA split
  compile partition.

Artifacts:

- `artifacts/a41_phase15_h100/production_mlp_w3_lane_levels_vs_phase14.json`
- `artifacts/a41_phase15_h100/w3_depth3_baseline_all_m.json`
- `artifacts/a41_phase15_h100/w3_lane_levels_candidate_all_m.json`
- `artifacts/a41_phase15_h100/rejected_depth4_w3_m1.json`

## Next experiment

Phase 16 keeps this exact 16-KiB lane table and uses its two half-word slots as
canonical and high-permuted views. That removes the hot-loop high-index
permutation without changing the table footprint or returning the level loads
to global/L1. See
`docs/kernels/qvq_a41_r0_phase16_h100_w3_dual_view_levels.md`.
