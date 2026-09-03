# Phase 46: H100 depth-three deferred fragment waits

Phase 46 extends last-safe-point fragment waits to the depth-three W2.5 and
W3 specializations. The wait count is selected at compile time: depth-three
rates use `wait_group<2>`, while the already promoted depth-two W2/W3.5 rates
use `wait_group<1>`. W2.5 improves the complete MLP 1.0203x and W3 improves
it 1.0160x, with ten of ten focused cells improving.

## Compile-time schedule

For decode depth `d`, K-block `j` selects:

\[
F_j=F_{j\bmod d}.
\]

The reusable fragment must not be overwritten until no more than `d-1`
committed register-sourced WGMMA groups remain:

\[
\operatorname{wait\_group}\langle d-1\rangle.
\]

For W2.5/W3, `d=3`. Phase 46 executes the fragment-independent work first:

```text
state/window extraction
selector-bank masks
four PGC products
wait_group<2>
level lookup -> write F[j mod 3]
WGMMA(F[j mod 3])
```

The helper's `TransitionBits` is a template argument, so the W2.5/W3 branch
and wait count are compile-time constants. There is no runtime rate branch in
the generated decoder. All P32 math, shared addresses, WGMMA issue order,
accumulator order, and FP16/FP32 boundaries remain unchanged.

## Isolated grouped gate/up result

Timing uses CUDA Graph replay and CUDA events on the physical 132-SM H100.
All outputs are exact and graph-stable.

| M/K/N per child | W2.5 before | W2.5 deferred | Better | W3 before | W3 deferred | Better |
|:--|--:|--:|:--:|--:|--:|:--:|
| 1/2048/8192 x2 | 30.067 us | 29.202 us | Yes | 29.651 us | 28.872 us | Yes |
| 2/2048/8192 x2 | 29.683 us | 28.853 us | Yes | 29.416 us | 28.923 us | Yes |
| 4/2048/8192 x2 | 30.068 us | 29.256 us | Yes | 29.724 us | 29.092 us | Yes |
| 8/2048/8192 x2 | 30.432 us | 29.682 us | Yes | 30.112 us | 29.580 us | Yes |
| 16/2048/8192 x2 | 31.243 us | 30.472 us | Yes | 30.715 us | 30.149 us | Yes |

The geometric gains are **1.0273x for W2.5** and **1.0205x for W3**.

## Complete Llama 3.2 1B MLP

The formal run acquired three spaced 0% utilization / 0 MiB idle samples and
uses 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph replays per
sample. Each row performs two gate/up projections plus down. Effective
throughput counts logical dense-equivalent FLOPs. Marlin/Machete are
figurative W4 baselines; ratios above one mean QVQ is faster. `Better`
compares with Phase 45.

| MKN: gate/up x2; down | QVQ W2.5 | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | Better | QVQ W3 | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | Better |
|:--|--:|--:|--:|--:|:--:|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2; 1/8192/2048 | 49.902 us | 2.017 | 0.583x | 0.983x | Yes | 50.086 us | 2.010 | 0.581x | 0.979x | Yes |
| 2/2048/8192 x2; 2/8192/2048 | 49.924 us | 4.033 | 0.623x | 0.981x | Yes | 50.199 us | 4.011 | 0.619x | 0.976x | Yes |
| 4/2048/8192 x2; 4/8192/2048 | 50.687 us | 7.944 | 0.617x | 0.977x | Yes | 50.683 us | 7.945 | 0.617x | 0.977x | Yes |
| 8/2048/8192 x2; 8/8192/2048 | 51.325 us | 15.690 | 0.570x | 0.968x | Yes | 51.404 us | 15.666 | 0.569x | 0.966x | Yes |
| 16/2048/8192 x2; 16/8192/2048 | 52.255 us | 30.822 | 0.622x | 0.951x | Yes | 52.417 us | 30.727 | 0.620x | 0.948x | Yes |

W2.5 improves **1.0203x** geometrically and W3 improves **1.0160x**. Their
Machete-relative geometric means are **0.9717x** and **0.9691x**. The all-rate
matrix, including W2/W3.5 controls, improves **1.0094x**.

## Matched Nsight Compute evidence

| Rate | Metric | Before | Deferred wait | Change |
|:--|:--|--:|--:|--:|
| W2.5 | Kernel duration | 27.776 us | 26.624 us | -4.15% |
| W2.5 | Executed instructions | 10,793,252 | 10,793,246 | unchanged |
| W2.5 | Eligible warps/cycle | 0.612 | 0.622 | +1.49% |
| W2.5 | Long-scoreboard stalls | 1.363 | 1.298 | -4.77% |
| W2.5 | Registers/thread | 59 | 61 | +2 |
| W3 | Kernel duration | 28.000 us | 27.040 us | -3.43% |
| W3 | Executed instructions | 10,639,754 | 10,639,670 | unchanged |
| W3 | Eligible warps/cycle | 0.587 | 0.596 | +1.46% |
| W3 | Long-scoreboard stalls | 1.394 | 1.375 | -1.38% |
| W3 | Registers/thread | 54 | 57 | +3 |

W2.5 shared-load instructions remain exactly 2,228,224; its conflicts and
wavefronts move by less than 1.3%. The older W3 baseline report did not collect
the shared-memory metric set, but total executed instructions are unchanged.
The duration/eligibility/stall changes support latency overlap rather than
removed math.

## Correctness and scope

- 126 Hopper P32 and grouped-P32 tests pass across W2--W3.5.
- Dense-oracle bounds, child-exact grouping, ordered reductions,
  repeatability, and CUDA Graph replay remain covered.
- No checkpoint bytes, persistent VRAM, shared-memory allocation, workspace,
  split policy, decoded value, or accumulation order changed.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one CUDA
  split-compile partition.

Artifacts:

- `artifacts/a41_phase46_h100/production_mlp_depth3_deferred_wait_vs_phase45.json`
- `artifacts/a41_phase46_h100/depth3_deferred_fragment_wait_profile.json`
- binary reports outside Git under
  `/root/qvq-profiler-artifacts/phase46-depth3-deferred-wait/`

## Next phase

Phase 47 should re-sweep decode depth after the wait relocation. The old
depth-two versus depth-three comparisons were made with waits before all
decoder arithmetic, so they do not prove that depth two remains optimal for
W2/W3.5 or that depth three remains optimal for W2.5/W3 under the new
last-safe-point schedule.
