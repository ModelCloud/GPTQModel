# Phase 45: H100 W2 deferred fragment-reuse wait

Phase 45 extends Phase 44's last-safe-point fragment wait to W2, the other
rate with a depth-two register-fragment schedule. The isolated grouped
gate/up site improves 1.0623x and the complete Llama 3.2 1B MLP improves
1.0361x. W2 is now faster than Machete W4 at every measured M.

## Schedule and exactness

W2 selects one of two register fragments for K-block `j`:

\[
F_j=F_{j\bmod 2}.
\]

The state/window extraction, selector-mask construction, and four PGC
products do not access `F_j`. Only the subsequent level lookup writes decoded
FP16 values into it. The reuse dependency is therefore protected exactly at
the first fragment write:

```text
state/window extraction
selector-bank masks
four PGC products
wait_group<1>
level lookup -> write F[j mod 2]
WGMMA(F[j mod 2])
```

This overlaps independent scalar decode work with the prior register-sourced
WGMMA. It does not change the P32 state, codebook index, level, WGMMA issue
order, accumulator order, or numerical boundary. W2.5/W3 keep their measured
depth-three schedule; W3.5 retains the independently promoted Phase-44 form.

## Isolated grouped gate/up result

Timing uses CUDA Graph replay and CUDA events on the physical 132-SM H100.
The output is exact and graph-stable for every row.

| M/K/N per child | Before | W2 deferred wait | Speedup | Better than last benchmark |
|:--|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 29.195 us | 27.442 us | 1.0639x | Yes |
| 2/2048/8192 x2 | 28.851 us | 27.177 us | 1.0616x | Yes |
| 4/2048/8192 x2 | 29.255 us | 27.564 us | 1.0613x | Yes |
| 8/2048/8192 x2 | 29.539 us | 27.787 us | 1.0630x | Yes |
| 16/2048/8192 x2 | 30.377 us | 28.614 us | 1.0616x | Yes |

The isolated geometric mean is **1.0623x**.

## Complete Llama 3.2 1B MLP

The formal matrix acquired three spaced 0% utilization / 0 MiB idle samples,
then used 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph replays
per sample. Effective throughput counts logical dense-equivalent FLOPs.
Marlin and Machete are figurative W4 baselines; a ratio above one means QVQ
is faster. `Better` compares with Phase 44.

| MKN: gate/up x2; down | QVQ W2 | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2; 1/8192/2048 | 47.898 us | 2.102 | 0.609x | 1.053x | Yes |
| 2/2048/8192 x2; 2/8192/2048 | 48.074 us | 4.188 | 0.648x | 1.044x | Yes |
| 4/2048/8192 x2; 4/8192/2048 | 48.836 us | 8.245 | 0.641x | 1.036x | Yes |
| 8/2048/8192 x2; 8/8192/2048 | 49.471 us | 16.278 | 0.593x | 1.025x | Yes |
| 16/2048/8192 x2; 16/8192/2048 | 50.632 us | 31.810 | 0.643x | 1.006x | Yes |

W2 improves **1.0361x** geometrically versus Phase 44. Its geometric ratios
are **0.6262x versus Marlin W4** and **1.0324x versus Machete W4**. Across all
rates, including unchanged-rate controls, the matrix improves **1.0068x**.

## Matched Nsight Compute evidence

Both reports use the same W2 M1 grouped gate/up geometry, payload layout,
physical H100, and metric set. The candidate profiler launch required three
immediate 0% utilization samples because spaced preflight attempts repeatedly
intersected an external 2% utilization pulse; the formal performance matrix
above used the normal spaced idle gate.

| Metric | Before | Deferred wait | Change |
|:--|--:|--:|--:|
| Kernel duration | 26.976 us | 25.184 us | -6.64% |
| Executed instructions | 10,394,898 | 10,394,892 | unchanged |
| Shared-load instructions | 1,966,080 | 1,966,080 | unchanged |
| Shared-load bank conflicts | 1,051,197 | 1,052,585 | +0.13% |
| Shared-load wavefronts | 3,343,802 | 3,394,571 | +1.52% |
| Eligible warps/cycle | 0.589 | 0.656 | +11.30% |
| Long-scoreboard stalls | 1.356 | 1.312 | -3.24% |
| Registers/thread | 48 | 53 | +5 |
| Shared memory/block | 42,752 B | 42,752 B | unchanged |

The candidate executes the same instruction and shared-load counts. More
eligible warps and fewer long-scoreboard stalls match the intended overlap.
The small shared wavefront/conflict increase is not the source of the win and
does not offset the 6.64% kernel-duration reduction.

## Correctness and scope

- 126 Hopper P32 and grouped-P32 tests pass across W2--W3.5.
- Dense-oracle bounds, exact child parity, ordered split reductions,
  repeatability, and CUDA Graph replay remain covered.
- No checkpoint bytes, persistent VRAM, shared-memory allocation, external
  workspace, split policy, decoded value, or accumulation order changed.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one CUDA
  split-compile partition.

Artifacts:

- `artifacts/a41_phase45_h100/production_mlp_w2_deferred_wait_vs_phase44.json`
- `artifacts/a41_phase45_h100/w2_deferred_fragment_wait_profile.json`
- binary report outside Git under
  `/root/qvq-profiler-artifacts/phase45-w2-deferred-wait/`

## Next phase

Phase 46 should test last-safe-point waits for the depth-three W2.5/W3
specializations. These require `wait_group<2>` rather than the depth-two
`wait_group<1>` used here, so the helper must select the compile-time wait
count without introducing a runtime branch or extra instructions.
