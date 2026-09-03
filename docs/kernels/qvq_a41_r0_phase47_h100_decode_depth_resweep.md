# Phase 47: H100 decode-depth re-sweep after deferred waits

Phase 47 re-measures fragment depth after Phases 44--46 moved reuse waits to
the first fragment overwrite. Depth three remains a regression for W2/W3.5.
Depth four improves W2.5/W3 across gate/up, grouped QKV, split-16 down, and
the complete Llama 3.2 1B MLP.

## Schedule

The depth-`d` decoder rotates register fragments as

\[
F_j=F_{j\bmod d}
\]

and protects reuse with

\[
\operatorname{wait\_group}\langle d-1\rangle
\]

immediately before level lookup overwrites `F_j`. Phase 47's accepted rate
mapping is:

| Rate | Decode depth | Reuse wait |
|:--|--:|:--|
| W2 | 2 | `wait_group<1>` |
| W2.5 | 4 | `wait_group<3>` |
| W3 | 4 | `wait_group<3>` |
| W3.5 | 2 | `wait_group<1>` |

The fourth fragment is a register tensor with the same eight FP16 values as
the other fragments. It adds no shared memory, workspace, persistent VRAM, or
checkpoint bytes. Decode and WGMMA order remain ascending in K; only the
number of independent in-flight fragments changes.

## Rejected depth-three extension

W2/W3.5 depth three was re-tested under the new wait placement and rejected.

| M/K/N per child | W2 depth 2 | W2 depth 3 | Better | W3.5 depth 2 | W3.5 depth 3 | Better |
|:--|--:|--:|:--:|--:|--:|:--:|
| 1/2048/8192 x2 | 27.442 us | 27.498 us | No | 28.998 us | 29.602 us | No |
| 2/2048/8192 x2 | 27.177 us | 27.222 us | No | 28.842 us | 29.448 us | No |
| 4/2048/8192 x2 | 27.564 us | 27.581 us | No | 29.193 us | 29.607 us | No |
| 8/2048/8192 x2 | 27.787 us | 27.847 us | No | 29.603 us | 30.194 us | No |
| 16/2048/8192 x2 | 28.614 us | 28.606 us | Yes | 30.317 us | 30.876 us | No |

W2 depth three is **0.9988x** geometrically; W3.5 is **0.9815x**. Neither
candidate entered production.

## Accepted depth-four gate/up result

| M/K/N per child | W2.5 depth 3 | W2.5 depth 4 | Better | W3 depth 3 | W3 depth 4 | Better |
|:--|--:|--:|:--:|--:|--:|:--:|
| 1/2048/8192 x2 | 29.202 us | 28.967 us | Yes | 28.872 us | 28.780 us | Yes |
| 2/2048/8192 x2 | 28.853 us | 28.431 us | Yes | 28.923 us | 28.726 us | Yes |
| 4/2048/8192 x2 | 29.256 us | 28.849 us | Yes | 29.092 us | 28.909 us | Yes |
| 8/2048/8192 x2 | 29.682 us | 29.277 us | Yes | 29.580 us | 29.353 us | Yes |
| 16/2048/8192 x2 | 30.472 us | 29.991 us | Yes | 30.149 us | 30.124 us | Yes |

Geometric gains are **1.0134x for W2.5** and **1.0050x for W3**. A second
400-sample W3 confirmation remained positive geometrically; individual cells
showed the expected sub-percent run variation.

## Complete Llama 3.2 1B MLP

The formal run acquired three spaced 0% utilization / 0 MiB idle samples and
uses 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph replays per
sample. Effective throughput counts logical dense-equivalent FLOPs. Marlin
and Machete are figurative W4 baselines; ratios above one mean QVQ is faster.
`Better` compares with Phase 46.

| MKN: gate/up x2; down | QVQ W2.5 | vs Marlin W4 | vs Machete W4 | Better | QVQ W3 | vs Marlin W4 | vs Machete W4 | Better |
|:--|--:|--:|--:|:--:|--:|--:|--:|:--:|
| 1/2048/8192 x2; 1/8192/2048 | 49.700 us | 0.588x | 1.000x | Yes | 49.933 us | 0.585x | 0.995x | Yes |
| 2/2048/8192 x2; 2/8192/2048 | 49.794 us | 0.629x | 0.989x | Yes | 50.070 us | 0.625x | 0.983x | Yes |
| 4/2048/8192 x2; 4/8192/2048 | 50.431 us | 0.625x | 0.987x | Yes | 50.543 us | 0.623x | 0.985x | Yes |
| 8/2048/8192 x2; 8/8192/2048 | 51.094 us | 0.576x | 0.979x | Yes | 51.113 us | 0.576x | 0.979x | Yes |
| 16/2048/8192 x2; 16/8192/2048 | 52.094 us | 0.626x | 0.961x | Yes | 52.207 us | 0.625x | 0.959x | Yes |

W2.5 improves **1.0039x** and W3 improves **1.0036x** across the complete
MLP, each with five of five wins. Their Machete-relative geometric means are
**0.9831x** and **0.9802x**.

## Shape audit

Depth four was compared with a freshly rebuilt depth-three executable using
identical payloads and CUDA timing parameters.

| Site | W2.5 depth-4 speedup | Wins | W3 depth-4 speedup | Wins |
|:--|--:|--:|--:|--:|
| Grouped QKV split `(8,8,8)` | 1.0094x | 5/5 | 1.0054x | 5/5 |
| Down split 16 | 1.0092x | 5/5 | 1.0091x | 5/5 |

All 20 shape-audit rows are repeatable, CUDA Graph stable, and inside the
dense-P32 error bound. The added fragment therefore does not create a hidden
register-pressure regression on either the narrow grouped site or the
512-block down launch.

## Nsight Compute evidence

| Rate | Metric | Depth 3 | Depth 4 | Change |
|:--|:--|--:|--:|--:|
| W2.5 | Duration | 26.624 us | 26.304 us | -1.20% |
| W2.5 | Instructions | 10,793,246 | 10,785,072 | -0.08% |
| W2.5 | Eligible warps/cycle | 0.622 | 0.635 | +2.14% |
| W2.5 | Registers/thread | 61 | 65 | +4 |
| W3 | Duration | 27.040 us | 27.136 us | +0.36% |
| W3 | Instructions | 10,639,670 | 10,631,478 | -0.08% |
| W3 | Eligible warps/cycle | 0.596 | 0.594 | -0.39% |
| W3 | Registers/thread | 57 | 59 | +2 |

Depth four removes one dynamic wait instruction per fragment cycle, visible
as roughly 8,192 fewer executed instructions. The single replay of W3 under
Nsight moves against the CUDA-Graph result by 0.36%; the higher-sample
gate/up run, complete MLP, QKV, and down matrices all favor depth four. W3 is
retained based on 18 positive CUDA-Graph cells across the primary and
confirmation runs, not the single instrumented replay.

## Correctness and scope

- 126 Hopper P32/grouped-P32 tests pass with depth four.
- Numerical math, child-local reduction order, split policies, and storage
  formats are unchanged.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one CUDA
  split-compile partition.

Artifacts:

- `artifacts/a41_phase47_h100/production_mlp_w25_w3_depth4_vs_phase46.json`
- `artifacts/a41_phase47_h100/depth_resweep_shape_audit.json`
- `artifacts/a41_phase47_h100/depth4_profile.json`
- binary reports outside Git under
  `/root/qvq-profiler-artifacts/phase47-depth-resweep/`

## Next phase

Phase 48 should profile the new per-rate steady state and attack the next
largest dependency chain. Fragment depth and wait placement are now measured
across every rate; further depth increases would add more registers for less
remaining overlap and are lower priority than reducing or rescheduling the
four PGC products and level-load dependency chain itself.
