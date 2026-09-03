# Phase 28: H100 W2.5 lane-static decode plan

Phase 28 promotes a transition-width-five specialization for the grouped
Hopper P32 decoder. It hoists W2.5 continuous-window word locations, shifts,
and selector position out of the hot K loop. The physical P32 layout,
quantized values, selectors, level table, WGMMA operation, accumulation
order, launch geometry, and storage remain unchanged.

Production commit: `baf71282`.

## Exact state extraction

For consumer lane `l`, define the output pair and K pair

```text
n_pair = l >> 2
k_pair = l & 3
pair0 = 16 * k_pair + n_pair
pair1 = pair0 + 8
```

For W2.5, transition width `E=5`, the circular-window positions are

```text
bit0 = (127 - pair0) * E
bit1 = bit0 - 8 * E
word_i = bit_i >> 5
shift_i = bit_i & 31
```

The paired K state lives `2E` words behind each first state. For each of the
two output pairs, the exact extraction is

```text
u_i = word[word_i] | (word[next(word_i)] << 32)
v_i = word[word_i - 2E] | (word[word_i - 2E + 1] << 32)

state_0i = u_i >> shift_i
state_1i = v_i >> shift_i
```

Before Phase 28, the inlined width-five path rebuilt `pair0`, `pair1`, both
bit positions, word positions, wrapped-next positions, shifts, and selector
shift for each of the sixteen K16 blocks in every K256 stage. Phase 28 builds
one lane-local descriptor before the stage loop:

```text
{word0, next0, shift0, word1, next1, shift1, selector_shift}
```

Every hot-loop iteration then performs only the four unchanged circular
window loads/funnel shifts. W2 keeps the compact generic implementation;
W3/W3.5 already used the same exact descriptor from Phase 13.

## Isolated grouped gate/up result

Both children have K=2048 and N=8192, retain distinct P32 payloads and bank
selectors, and execute in the existing 256-CTA grouped split-one launch.
Timing includes grouped gate/up inner P32 plus paired output recovery. The
physical 132-SM H100 was admitted only after three 0% utilization / 0 MiB
samples. Each result uses 30 warmups, 200 CUDA-event samples, and 50 warmed
CUDA Graph replays per sample.

| M/K/N per child | Baseline | Phase 28 | vs baseline | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 32.903 us | 32.489 us | 1.0127x | N/A | N/A | Yes |
| 2/2048/8192 x2 | 32.431 us | 32.186 us | 1.0076x | N/A | N/A | Yes |
| 4/2048/8192 x2 | 32.785 us | 32.631 us | 1.0047x | N/A | N/A | Yes |
| 8/2048/8192 x2 | 33.150 us | 32.930 us | 1.0067x | N/A | N/A | Yes |
| 16/2048/8192 x2 | 33.977 us | 33.764 us | 1.0063x | N/A | N/A | Yes |

The isolated geometric-mean speedup is **1.0076x**, and all five cells are
bit-exact, repeatable, CUDA-Graph stable, and within the 2e-3 dense P32 gate.
Marlin/Machete do not expose the recovery-inclusive grouped sub-operation, so
their meaningful comparison is the complete MLP below.

## Complete Llama 3.2 1B MLP

The formal post-commit matrix includes the two gate/up projections, exact
paired recovery, SiLU/product/down preconditioning, split-16 down P32, and
fused down reduction/recovery. Effective TFLOP/s uses logical dense-equivalent
work. W4 Marlin and Machete are figurative efficiency baselines; they neither
perform the same compressed work nor imply equal quantization quality.

| Rate | M | MKN: gate/up x2; down | QVQ us | Eff. TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 26 | Better than last benchmark |
|---:|---:|:--|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1/2048/8192 x2; 1/8192/2048 | 53.514 | 1.881 | 0.545x | 0.941x | 0.998x | No |
| W2 | 2 | 2/2048/8192 x2; 2/8192/2048 | 53.732 | 3.747 | 0.583x | 0.931x | 0.999x | No |
| W2 | 4 | 4/2048/8192 x2; 4/8192/2048 | 54.263 | 7.420 | 0.581x | 0.932x | 1.001x | Yes |
| W2 | 8 | 8/2048/8192 x2; 8/8192/2048 | 54.742 | 14.711 | 0.538x | 0.925x | 1.000x | Yes |
| W2 | 16 | 16/2048/8192 x2; 16/8192/2048 | 56.041 | 28.740 | 0.582x | 0.906x | 1.001x | Yes |
| W2.5 | 1 | 1/2048/8192 x2; 1/8192/2048 | 54.162 | 1.859 | 0.539x | 0.929x | 1.006x | Yes |
| W2.5 | 2 | 2/2048/8192 x2; 2/8192/2048 | 54.290 | 3.708 | 0.577x | 0.921x | 1.007x | Yes |
| W2.5 | 4 | 4/2048/8192 x2; 4/8192/2048 | 55.036 | 7.316 | 0.572x | 0.919x | 1.006x | Yes |
| W2.5 | 8 | 8/2048/8192 x2; 8/8192/2048 | 55.479 | 14.516 | 0.531x | 0.913x | 1.009x | Yes |
| W2.5 | 16 | 16/2048/8192 x2; 16/8192/2048 | 56.689 | 28.412 | 0.576x | 0.896x | 1.006x | Yes |
| W3 | 1 | 1/2048/8192 x2; 1/8192/2048 | 51.555 | 1.953 | 0.566x | 0.976x | 1.002x | Yes |
| W3 | 2 | 2/2048/8192 x2; 2/8192/2048 | 51.593 | 3.902 | 0.607x | 0.970x | 1.005x | Yes |
| W3 | 4 | 4/2048/8192 x2; 4/8192/2048 | 52.244 | 7.707 | 0.603x | 0.968x | 1.000x | No |
| W3 | 8 | 8/2048/8192 x2; 8/8192/2048 | 52.651 | 15.295 | 0.560x | 0.962x | 1.005x | Yes |
| W3 | 16 | 16/2048/8192 x2; 16/8192/2048 | 53.810 | 29.932 | 0.607x | 0.944x | 1.004x | Yes |
| W3.5 | 1 | 1/2048/8192 x2; 1/8192/2048 | 54.298 | 1.854 | 0.538x | 0.927x | 1.006x | Yes |
| W3.5 | 2 | 2/2048/8192 x2; 2/8192/2048 | 54.341 | 3.705 | 0.576x | 0.921x | 1.005x | Yes |
| W3.5 | 4 | 4/2048/8192 x2; 4/8192/2048 | 55.036 | 7.316 | 0.572x | 0.919x | 1.001x | Yes |
| W3.5 | 8 | 8/2048/8192 x2; 8/8192/2048 | 55.550 | 14.497 | 0.530x | 0.911x | 0.995x | No |
| W3.5 | 16 | 16/2048/8192 x2; 16/8192/2048 | 56.437 | 28.538 | 0.578x | 0.900x | 1.001x | Yes |

The targeted W2.5 rows improve **5/5** with a **1.0069x** geometric mean.
Across all rates, including unchanged-rate cross-run variance, 16/20 rows
improve and the geometric mean is **1.0027x** versus Phase 26. Other
geometric means are **0.5676x Marlin W4**, **0.9303x Machete W4**, and
**2.5792x ordinary per-module QVQ**.

## Nsight Compute and SASS

Nsight Compute 2026.2.1 captured the exact parent `90a2d214` and promoted
`baf71282` with 19 replay passes and source-correlated assembly.

| W2.5 M1 metric | Baseline | Phase 28 | Change |
|:--|--:|--:|--:|
| Executed warp instructions | 10,721,020 | 10,663,544 | **-57,476 (-0.54%)** |
| `IMAD` | 2,731,142 | 2,698,406 | **-32,736** |
| `LEA` | 70,400 | 54,016 | **-16,384** |
| `ISETP` | 68,096 | 59,904 | **-8,192** |
| `SEL` | 44,032 | 35,840 | **-8,192** |
| `LOP3` | 1,391,616 | 1,383,424 | **-8,192** |
| `VIADD` | 180,480 | 188,672 | +8,192 |
| `SHF` | 1,353,984 | 1,362,176 | +8,192 |
| Shared level loads (`LDS`) | 2,228,224 | 2,228,224 | unchanged |
| Registers/thread | 55 | 55 | unchanged |
| Static shared memory | 28.288 KiB | 28.288 KiB | unchanged |
| Long scoreboard / issue-active | 1.384 | 1.381 | unchanged |
| DRAM throughput | 14.73% | 14.76% | unchanged |

Replay duration is essentially neutral (29.760 versus 29.728 us), as expected
for multi-pass profiling of a small kernel. The canonical CUDA-event A/B is
the promotion timing. The SASS proves that the source change removed address
and predicate work without changing decode memory traffic or launch resources.

Profiler reports remain outside Git:

```text
/root/qvq-profiler-artifacts/phase28-w25-lane-plan/baseline_w25_m1.ncu-rep
/root/qvq-profiler-artifacts/phase28-w25-lane-plan/candidate_w25_m1.ncu-rep
```

## Validation and storage

- all 59 grouped Hopper tests pass across W2/W2.5/W3/W3.5 and M=1,2,4,8,16;
- the isolated and complete-MLP probes require exact recovered outputs,
  repeatability, CUDA Graph stability, and the existing 2e-3 dense P32 gate;
- build parallelism remained capped at four Ninja jobs, one NVCC host thread,
  and one split-compile partition;
- no checkpoint, packed payload, transient workspace, persistent allocation,
  graph node, or runtime dispatch changed.

Artifacts:

- `artifacts/a41_phase28_h100/w25_lane_plan_baseline.json`
- `artifacts/a41_phase28_h100/w25_lane_plan_candidate.json`
- `artifacts/a41_phase28_h100/w25_lane_plan_full_mlp_candidate.json`
- `artifacts/a41_phase28_h100/production_mlp_w25_lane_plan_vs_phase26.json`
- `artifacts/a41_phase28_h100/w25_lane_plan_ncu_summary.json`

## Next phase

Phase 29 should test the same lane-static plan at transition width four (W2)
as a separate compile-and-measure gate. W2's word geometry is simpler and
may already be optimized by `ptxas`, so it must not inherit the W2.5 policy
without an exact matched A/B and complete-MLP win.
