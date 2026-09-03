# QVQ A41/R0 Phase 24: vectorized fused down reduction

## Decision

Phase 24 promotes no executable change. An exact `float4` implementation
reduced the fused Phase-23 recovery kernel's executed warp instructions by
6.1%, but lowered eligible scheduler work and made the matched Nsight Compute
replay 2.1% slower. The complete W3 MLP geometric mean was **0.9993x** versus
Phase 23, with two strict regressions.

All experimental CUDA, Python dispatch, and operator-schema changes were
reverted. Production source remains byte-identical to Phase-23 commit
`c861b0a0`.

## Candidate math

Phase 23 assigns each of 1,024 recovery threads two output columns. Every
scalar executes the deterministic split reduction

\[
r_j=((((0+P_{0,j})+P_{1,j})+\cdots)+P_{15,j}).
\]

The candidate assigned the first 512 threads one aligned `float4`. Each
thread retained four independent accumulator chains:

```cpp
float4 partial = reinterpret_cast<const float4*>(plane)[vector_index];
value0 += partial.x;
value1 += partial.y;
value2 += partial.z;
value3 += partial.w;
```

Each chain still began at positive zero and visited split planes 0 through 15
in the same order. The four results were then narrowed, normalized, and
written to the identical bank-padded shared-memory locations. After one block
barrier, all 1,024 threads executed the unchanged Hadamard, scale, bias, and
FP16 store.

The candidate was therefore bit-exact; it changed load width and independent
instruction scheduling only.

## Matched W3 timing

The scalar and vectorized variants were compiled into one binary and timed
with warmed CUDA Graph replay and CUDA events on physical H100 UUID
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`. Admission required zero MiB in
use. `site` includes the split-16 WGMMA launch and fused recovery.

| M | MKN | Scalar site us | Vector site us | Site speedup | Recovery-only speedup | Better |
|---:|:---|---:|---:|---:|---:|:---:|
| 1 | 1x8192x2048 | 20.052 | 19.751 | 1.0152x | 1.0004x | Yes |
| 2 | 2x8192x2048 | 19.637 | 19.652 | 0.9993x | 1.0066x | No |
| 4 | 4x8192x2048 | 19.435 | 19.673 | 0.9879x | 0.9997x | No |
| 8 | 8x8192x2048 | 19.708 | 19.693 | 1.0008x | 0.9990x | Yes |
| 16 | 16x8192x2048 | 19.628 | 19.743 | 0.9942x | 1.0074x | No |

Geometric means are **0.9994x** for the complete down site and **1.0026x**
for recovery in isolation. The candidate does not provide a stable operation
level improvement.

## Measured SASS and scheduler result

Nsight Compute 2026.4.1 captured one scalar and one vectorized M1 kernel from
the same experimental source using 19 hardware-counter replay passes.

| Metric | Scalar | `float4` | Change |
|:--|--:|--:|:--|
| Executed warp instructions | 29,664 | 27,840 | **-6.1%** |
| NCU replay duration | 9.088 us | 9.280 us | **2.1% slower** |
| Eligible warps/scheduler/cycle | 1.308 | 1.109 | **-15.2%** |
| Active warps | 47.96% | 47.85% | unchanged |
| Registers/thread | 32 | 32 | unchanged |
| Dynamic shared memory | 8.448 KiB | 8.448 KiB | unchanged |
| DRAM throughput | 0.717% | 0.710% | unchanged |
| Combined memory throughput | 3.686% | 3.681% | unchanged |

The load-width change removes instructions, but it creates four 16-deep FP32
addition chains in half of the block. Hopper exposes less eligible work while
those chains resolve. Neither form approaches HBM saturation, so fewer load
instructions do not translate into useful latency reduction.

Binary reports remain outside Git:

```text
/root/qvq-profiler-artifacts/phase24-down-reduction/
```

## Required full-MLP comparison

The rejected candidate was measured with the same 30 warmups, 200 samples,
and 50 CUDA Graph replays per sample as Phase 23. Marlin W4 and Machete W4
were freshly measured in the same process. `Better` is the strict median
comparison against the Phase-23 production artifact.

| Rate | M | MKN (gate/up; down) | Candidate us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 23 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 59.620 | 1.688 | 0.488x | 0.826x | 1.0016x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 58.960 | 3.415 | 0.528x | 0.829x | 0.9940x | No |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 58.617 | 6.869 | 0.534x | 0.843x | 0.9994x | No |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 58.852 | 13.683 | 0.497x | 0.843x | 1.0006x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 59.572 | 27.036 | 0.545x | 0.835x | 1.0007x | Yes |

Geometric means are **0.9993x** versus Phase 23, **0.5180x** versus Marlin
W4, and **0.8353x** versus Machete W4. W4 is a figurative dense-equivalent
throughput baseline, not an equal-rate or equal-quality comparison.

## Validation and artifacts

- scalar and vectorized outputs were bit-exact for M1/M2/M4/M8/M16;
- both paths were CUDA-Graph stable;
- no persistent or transient storage size changed;
- compilation stayed within Ninja `-j4` and one CUDA split-compile partition;
- production source was restored before this document was committed.

Artifacts:

- `artifacts/a41_phase24_h100/vectorized_down_reduction_w3_probe.json`
- `artifacts/a41_phase24_h100/rejected_vectorized_down_reduction_full_mlp.json`

## Constraint learned

Widening loads is not sufficient when the saved load instructions are already
hidden and the replacement exposes fewer independent ready warps. A future
reduction optimization must shorten or overlap the 16-step dependency chain,
or remove another global synchronization/materialization boundary. Merely
packing four scalar loads into one instruction is closed by this phase.
