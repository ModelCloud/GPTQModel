# QVQ A41/R0 Phase 13: H100 W3 lane-static decode plan

Phase 13 removes repeated W3 lane/window address construction from the grouped
Hopper P32 decode loop. The accepted specialization computes each consumer
lane's two window locations, shifts, and selector shift once before the K loop
and reuses them for every K16 tile. W3.5 already used this representation; W3
now does as well. W2 and W2.5 retain the existing compact decoder.

The change is exact and format-neutral. It does not alter canonical P32 or
window payloads, selector bytes, level tables, PGC math, WGMMA accumulation,
split-K, output recovery, CUDA Graph topology, transient workspace, or any
non-H100 fallback.

## Decode math

For consumer lane `l`, define:

\[
n_p=l\mathbin{\gg}2,\qquad k_p=l\mathbin{\&}3,
\]

\[
p_0=16k_p+n_p,\qquad p_1=p_0+8.
\]

For transition width `E`, the two bit positions are:

\[
b_0=(63-p_0)E,\qquad b_1=b_0-8E.
\]

Each position defines a first word, possible wrapped next word, and shift:

\[
w_i=b_i\mathbin{\gg}5,\qquad s_i=b_i\mathbin{\&}31.
\]

The paired K state is the same continuous-window extraction used previously:

\[
u_i=W[w_i]\;|\;(W[next(w_i)]\ll32),
\]

\[
v_i=W[w_i-2E]\;|\;(W[w_i-2E+1]\ll32),
\]

\[
state_{0i}=u_i\gg s_i,\qquad state_{1i}=v_i\gg s_i.
\]

Before Phase 13, the W3 hot loop reconstructed `lane`, `n_pair`, `k_pair`,
`pair00`, `pair01`, both bit positions, word pairs, and shifts for every
decoded K block. The accepted path constructs this lane-static descriptor
once:

```cpp
plan = {
    first_word0, first_next_word0, shift0,
    first_word1, first_next_word1, shift1,
    bank_shift,
};
```

Every K iteration then performs only the four planned word-pair extractions
and the unchanged selector/PGC/level lookup. There is no approximation and no
new rounding or accumulation boundary.

## Candidate history

Three larger alternatives were tested first and rejected.

| Candidate, W3 M1 | Baseline us | Candidate us | Speedup | Exact vs split-1 | Extra transient bytes | Decision |
|:--|--:|--:|--:|:--:|--:|:--|
| ordered split-2 | 33.401 | 33.213 | 1.006x | No, 2,608 FP16 values changed | 2,097,152 | rejected |
| ordered split-4 | 33.401 | 37.382 | 0.893x | No, 4,018 changed | 4,194,304 | rejected |
| ordered split-8 | 33.401 | 38.825 | 0.860x | No, 3,863 changed | 8,388,608 | rejected |
| two-consumer N128 WGMMA | 33.233 | 34.231 | 0.971x | Yes | 0 | rejected |
| global/L1 level lookup | 33.144 | 34.256 | 0.968x | Yes | 0 | rejected |

The split-K candidates are repeatable and bounded by the dense oracle, but
they change the FP32 reduction order and therefore fail the required
split-1 recovered-output equality. The exact N128 candidate shares activation
staging between two WGMMA consumer groups but loses to the simpler N64 grid.
The global-level candidate avoids shared level-table traffic but loses the
low-latency shared lookup. All rejected production/API source was reverted.

## Matched gate/up A/B

The baseline and candidate use the same physical H100, input/payload seeds,
strict zero-MiB idle admission, 30 warmups, 200 CUDA-event samples, and 50
CUDA Graph replays per sample. The measured operation includes grouped W3
gate/up inner P32 and exact paired recovery. Every output is bit-exact, ten
ordinary repeats are stable, and graph replay is stable.

| M | Baseline us | Lane-plan us | Speedup | Better |
|---:|---:|---:|---:|:---:|
| 1 | 32.748 | 32.556 | 1.0059x | Yes |
| 2 | 32.668 | 32.363 | 1.0094x | Yes |
| 4 | 32.960 | 32.720 | 1.0073x | Yes |
| 8 | 33.379 | 33.000 | 1.0115x | Yes |
| 16 | 34.098 | 33.818 | 1.0083x | Yes |

The geometric-mean gate/up improvement is **1.0085x**.

## Matched Nsight Compute / SASS

Nsight Compute 2026.2.1 captured one retained W3 M1 grouped gate/up WGMMA
launch for the Phase-12 source and one for production commit `e0a76930`. The
reports, raw metric exports, and source-correlated SASS exports are outside
Git at:

```text
/root/qvq-profiler-artifacts/phase13-gateup-decode/
```

Both captures used the physical H100 UUID, strict zero-MiB idle admission,
`--profile-from-start off`, one named launch, and the `SpeedOfLight`,
`LaunchStats`, `Occupancy`, `SchedulerStats`, `WarpStateStats`, and
`InstructionStats` section sets.

| Metric/opcode | Phase 12 | Phase 13 | Change |
|:--|--:|--:|--:|
| total executed warp instructions | 10,744,642 | 10,687,611 | **-57,031 (-0.53%)** |
| `IMAD` | 2,727,894 | 2,695,201 | **-32,693** |
| `LEA` | 70,400 | 54,016 | **-16,384** |
| `ISETP` | 68,096 | 59,904 | **-8,192** |
| `SEL` | 44,032 | 35,840 | **-8,192** |
| `LOP3` | 1,391,616 | 1,383,424 | **-8,192** |
| `VIADD` | 177,920 | 186,112 | +8,192 |
| `SHF` | 1,353,984 | 1,362,176 | +8,192 |
| `LDS` | 2,228,224 | 2,228,224 | unchanged |
| registers/thread | 55 | 55 | unchanged |
| static shared memory | 30.336 KiB | 30.336 KiB | unchanged |
| local/shared spills | 0 / 0 | 0 / 0 | unchanged |

The source transformation removes address and predicate work rather than
decode loads. Active warps are unchanged (15.17%), long-scoreboard stall is
unchanged within measurement noise (1.428), and wait stall rises from 0.938
to 1.015 instructions per issue-active cycle. Multi-pass NCU replay duration
therefore is not used as the latency acceptance metric; warmed graph replay
measured by CUDA events is the canonical timing.

## Complete Llama 3.2 1B MLP

The post-commit artifact runs executable SHA `e0a76930` with 30 warmups, 200
CUDA-event samples, and 50 CUDA Graph replays per sample. It includes grouped
gate/up P32, paired recovery, exact SiLU/down precondition, down P32, and down
recovery. `vs` is comparator latency divided by QVQ latency, so values below
one mean the W4 comparator is faster. `Better` compares each cell with the
committed Phase-12 artifact and records `No` for a regression.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 12 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 67.675 | 1.487 | 0.432x | 0.752x | 1.0046x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 67.551 | 2.980 | 0.462x | 0.750x | 1.0025x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 68.486 | 5.879 | 0.459x | 0.750x | 0.9993x | No |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 68.968 | 11.676 | 0.426x | 0.742x | 1.0028x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.359 | 25.025 | 0.508x | 0.796x | 1.0015x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.569 | 1.468 | 0.426x | 0.743x | 1.0021x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.695 | 2.931 | 0.454x | 0.738x | 1.0022x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.208 | 5.818 | 0.454x | 0.742x | 1.0006x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.974 | 11.509 | 0.420x | 0.732x | 1.0010x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.245 | 24.685 | 0.501x | 0.785x | 0.9998x | No |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.703 | 1.465 | 0.425x | 0.741x | 1.0073x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.674 | 2.932 | 0.455x | 0.738x | 1.0055x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.134 | 5.824 | 0.454x | 0.743x | 1.0054x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.828 | 11.533 | 0.421x | 0.733x | 1.0054x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.314 | 24.660 | 0.500x | 0.785x | 1.0017x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.384 | 1.472 | 0.427x | 0.745x | 1.0011x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.537 | 2.938 | 0.455x | 0.739x | 1.0037x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.387 | 5.803 | 0.453x | 0.740x | 1.0000x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.881 | 11.524 | 0.421x | 0.733x | 0.9963x | No |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.090 | 24.744 | 0.502x | 0.787x | 0.9972x | No |

The targeted W3 cells improve **5/5**, with **1.0051x** geometric-mean
complete-MLP speedup versus Phase 12. Across all rates, including unchanged
rates and cross-run noise, 16/20 cells improve and the geometric mean is
**1.0020x**. Other all-cell geometric means are **2.0703x** versus ordinary
per-module QVQ, **0.4518x** versus Marlin W4, and **0.7504x** versus Machete
W4. The W4 comparisons are figurative dense-equivalent efficiency baselines;
they do not assert equal compressed work or quantization quality.

## Correctness, VRAM, and validation

- The 59-test grouped Hopper suite passes across W2/W2.5/W3/W3.5 and
  M=1/2/4/8/16, including dense-oracle bounds, independent-child equality,
  deterministic ordered reduction, repeatability, CUDA Graph stability, and
  real Llama QKV/gate-up shapes.
- The matched W3 A/B sweep is bit-exact for all five M values and stable over
  ten ordinary launches plus graph replay.
- The complete MLP matrix passes its existing dense/reference and graph
  validation.
- Ruff, Python compilation, and changed-file whitespace checks pass.
- CUDA compilation is capped at four Ninja jobs, one NVCC host thread, and
  one split-compile partition.
- Persistent and transient VRAM are unchanged. The accepted specialization
  adds only lane-local integer plan values already present for W3.5; it adds
  no checkpoint state, workspace, packed payload, graph node, or global
  allocation.

Artifacts and drivers:

- `artifacts/a41_phase13_h100/production_mlp_w3_lane_plan_vs_phase12.json`
- `artifacts/a41_phase13_h100/w3_lane_plan_baseline_all_m.json`
- `artifacts/a41_phase13_h100/w3_lane_plan_candidate_all_m.json`
- `artifacts/a41_phase13_h100/rejected_ordered_split_w3_m1.json`
- `artifacts/a41_phase13_h100/rejected_n128_w3_m1.json`
- `artifacts/a41_phase13_h100/rejected_global_levels_w3_m1.json`
- `scripts/benchmark_qvq_phase13_gateup_split.py`
- `scripts/profile_qvq_phase13_gateup_wgmma.py`

## Next experiment

The accepted change removes only 0.53% of the grouped W3 decoder's executed
instructions, so more lane-address algebra alone will not close the Machete
gap. W2/W2.5 should not inherit the plan without a separate same-binary A/B
gate. The next phase should target a larger representation-level source of
the remaining `IMAD`, `PRMT`, `LOP3`, `SHF`, and shared level-lookup stream,
while preserving the split-1 accumulation order and N64 launch geometry that
won the rejected-candidate comparison.

Phase 14 retained that exact geometry and increased only the W3
decode/WGMMA pending depth from two to three. See
`docs/kernels/qvq_a41_r0_phase14_h100_w3_decode_overlap.md`.
