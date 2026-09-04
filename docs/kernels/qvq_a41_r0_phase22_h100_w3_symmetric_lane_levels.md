# A41/R0 Phase 22: H100 W3 symmetric lane levels

## Decision

Phase 22 promotes no executable change. The frozen PGC16 codebook's exact
FP16 sign symmetry can compress a conflict-free one-word-per-lane table into
the same 16 KiB shared footprint as production. The layout eliminated every
shared-load bank conflict and improved scheduler eligibility, but exact
sign/magnitude address reconstruction increased instructions and register
pressure enough to regress the full W3 MLP by 19.7%.

All experimental CUDA was reverted. End-of-phase production source is
byte-identical to commit `59771878`.

## Exact symmetry and layout

The canonical codebook has the following bit identity for every
\(0\le i<128\):

\[
bits(level[i])=bits(level[255-i])\oplus 0x8000.
\]

Therefore only the positive half needs physical storage. Given an original
level index \(i\), define

\[
d=i-128,
\qquad
s=d\gg31,
\qquad
m=d\oplus s.
\]

Here \(m\in[0,127]\) is the positive magnitude index and `s` is all ones only
for a negative result. The exact FP16 output bits are

\[
bits(level[128+m])\oplus(s\mathbin{\&}0x8000).
\]

The candidate shared layout was

```text
positive_level_word[128 magnitudes][32 lanes]
```

where each 32-bit word replicated one positive FP16 bit pattern. Its size is

\[
128\times32\times4=16384\text{ bytes},
\]

exactly the same as production's W3 lane-pair table. The address stride is
128 bytes per magnitude and four bytes per lane, so every lane addresses a
distinct Hopper shared-memory bank. Checkpoint bytes, persistent VRAM, TMA
stage count, WGMMA order, and graph topology were unchanged.

## Candidate evolution

Three exact implementations were measured at grouped W3 gate/up M1:

| Variant | MKN per child | CUDA-event us | Static SASS | Registers/thread | Better than production |
|:--|:--|--:|--:|--:|:---:|
| Production lane-pair table | 1x2048x8192 (x2) | 31.693 | 1,947 | 58 | baseline |
| C++ sign/magnitude expressions | 1x2048x8192 (x2) | 52.920 | 2,880 | 75 | No |
| Seven-instruction PTX mapping | 1x2048x8192 (x2) | 44.622 | 2,552 | 69 | No |
| Five-instruction `BFE`/`LOP3` PTX mapping | 1x2048x8192 (x2) | 47.326 | 2,552 | 65 | No |

The seven-instruction form was the fastest isolated candidate. The nominally
shorter five-instruction form shifted work onto sign-extension/logic issue
paths and was slower, so it was retained only long enough to produce the
required full-MLP matrix.

## Nsight result

The source-expression candidate confirms that the physical layout worked as
designed:

| W3 M1 metric | Production | Symmetric lane words | Change |
|:--|--:|--:|--:|
| Replay duration | 28.192 us | 40.640 us | 44.2% slower |
| Executed warp instructions | 10,792,258 | 18,621,514 | +72.5% |
| Shared-load bank conflicts | 1,050,805 | 0 | eliminated |
| LSU wavefronts | 3,350,926 | 2,581,327 | -23.0% |
| Eligible warps/scheduler/cycle | 0.588 | 0.736 | +25.2% |
| Registers/thread | 58 | 75 | +17 |

Removing conflicts and wavefronts did improve scheduler eligibility. The
candidate still lost because every one of the eight level values needed an
index-centering/sign operation, magnitude fold, wider address calculation,
32-bit load, and sign restoration. Even hand-written PTX left 2,552 static
instructions, 31.1% above production, and at least 65 registers/thread.

## Full Llama 3.2 1B MLP benchmark

All timings below are warmed CUDA Graph replays measured with CUDA events on
the physical 132-SM H100 UUID
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`. Admission required zero MiB in
use. W4 Marlin and Machete were freshly measured in the same process. The
comparison is figurative: QVQ is W3 while the baselines are W4. Effective
TFLOP/s is logical dense-equivalent work divided by latency.

| Rate | M | MKN (gate/up; down) | Candidate us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs last | Better than last |
|---:|---:|:--|---:|---:|---:|---:|---:|:---:|
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 76.456 | 1.317 | 0.390x | 0.667x | 0.807x | No |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 75.798 | 2.656 | 0.424x | 0.661x | 0.799x | No |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 75.065 | 5.364 | 0.426x | 0.672x | 0.803x | No |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 75.315 | 10.692 | 0.397x | 0.667x | 0.801x | No |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 76.096 | 21.166 | 0.436x | 0.658x | 0.806x | No |

Geometric means were **0.414x Marlin W4**, **0.665x Machete W4**, and
**0.803x the Phase-19 production benchmark**. All five cells regressed. Every
full-MLP output remained bit-exact to the ordinary QVQ control.

## Validation and artifacts

The focused kernel checks covered dense-oracle error, repeatability, recovered
FP16 bit equality, and CUDA Graph replay. Maximum inner error was within the
existing `2e-3` dense-P32 gate. The full MLP script required exact output
equality before recording every row.

- `artifacts/a41_phase22_h100/rejected_w3_symmetric_lane_full_mlp.json`
- `artifacts/a41_phase22_h100/w3_symmetric_lane_summary.json`

The binary Nsight report remains outside Git at
`/root/qvq-profiler-artifacts/phase22-gateup/w3_m1_symmetric_lane_words.ncu-rep`.

Compilation was capped at Ninja `-j4`, one NVCC host thread, and one
split-compile partition.

## Constraint learned

Codebook antisymmetry can buy either half the storage or one full 32-bit bank
per lane, but the required per-value index fold costs more than the remaining
1.5-way shared conflict on this kernel. A future level-layout experiment must
make the sign/magnitude mapping free at payload generation time or reuse one
mapped index across multiple outputs. Recomputing it for each decoded scalar
is closed by this phase.
