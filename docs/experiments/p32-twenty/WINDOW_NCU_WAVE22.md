# Expanded window Nsight Compute wave 22

This wave profiled the exact production fused-window inner kernel on the F6
seed-7 snapshot for eight real projections at M=1, 16, and 2048. Each run used
BM64, BN64, split-K=1, exact K0 FP32 accumulation, and the existing
`--profile-fused` warmup path. All eight jobs completed and all 24 profiled
cases passed the local teacher and window correctness gates.

The expanded sections were `ComputeWorkloadAnalysis`, `InstructionStats`,
`LaunchStats`, `MemoryWorkloadAnalysis`, `MemoryWorkloadAnalysis_Tables`,
`Occupancy`, `SchedulerStats`, `SpeedOfLight`,
`SpeedOfLight_HierarchicalTensorRooflineChart`, `WarpStateStats`, and
`SourceCounters`.

## Kernel metrics

The rows below are reported by Nsight for the `_gemm` kernel. `Inst` is total
executed instructions for the profiled launch. DRAM and SM Busy are reported
percentages. The input and output transforms are outside this kernel profile.

| Projection | M | Reg/thread | Achieved occupancy | SM Busy | Inst | L2 hit | DRAM | Warp cycles/inst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| l0down | 1 | 115 | 6.25% | 6.24% | 19,363,456 | 42.71% | 0.44% | 4.63 |
| l0down | 16 | 115 | 6.24% | 6.24% | 19,363,456 | 65.29% | 0.46% | 4.63 |
| l0down | 2048 | 115 | 21.80% | 53.77% | 619,642,880 | 91.32% | 2.35% | 6.60 |
| l0gate | 1 | 115 | 6.49% | 20.47% | 19,454,464 | 46.21% | 1.43% | 4.71 |
| l0gate | 16 | 115 | 6.49% | 20.34% | 19,454,464 | 68.86% | 1.44% | 4.72 |
| l0gate | 2048 | 115 | 24.10% | 65.99% | 622,592,000 | 98.59% | 1.47% | 6.61 |
| l0q | 1 | 107 | 6.28% | 5.90% | 5,223,552 | 56.68% | 0.28% | 4.64 |
| l0q | 16 | 116 | 6.25% | 6.06% | 4,895,744 | 65.96% | 0.31% | 4.75 |
| l0q | 2048 | 116 | 22.33% | 54.50% | 156,676,096 | 96.48% | 0.86% | 6.52 |
| l0up | 1 | 115 | 6.50% | 20.60% | 19,454,464 | 37.39% | 1.68% | 4.71 |
| l0up | 16 | 115 | 6.49% | 20.62% | 19,454,464 | 61.02% | 1.70% | 4.71 |
| l0up | 2048 | 115 | 24.10% | 65.46% | 622,592,000 | 98.56% | 1.49% | 6.66 |
| l1down | 1 | 115 | 6.25% | 6.24% | 19,363,456 | 42.63% | 0.44% | 4.62 |
| l1down | 16 | 115 | 6.26% | 6.24% | 19,363,456 | 65.84% | 0.46% | 4.63 |
| l1down | 2048 | 115 | 21.80% | 53.76% | 619,642,880 | 91.23% | 2.34% | 6.59 |
| l1gate | 1 | 115 | 6.49% | 20.42% | 19,454,464 | 46.18% | 1.43% | 4.71 |
| l1gate | 16 | 115 | 6.49% | 20.23% | 19,454,464 | 68.49% | 1.43% | 4.72 |
| l1gate | 2048 | 115 | 24.11% | 65.85% | 622,592,000 | 98.78% | 1.48% | 6.61 |
| l1q | 1 | 107 | 6.27% | 5.89% | 5,223,552 | 57.43% | 0.28% | 4.64 |
| l1q | 16 | 116 | 6.26% | 6.06% | 4,895,744 | 65.86% | 0.31% | 4.75 |
| l1q | 2048 | 116 | 22.28% | 54.26% | 156,676,096 | 96.45% | 0.84% | 6.53 |
| l1up | 1 | 115 | 6.49% | 20.67% | 19,454,464 | 36.85% | 1.69% | 4.70 |
| l1up | 16 | 115 | 6.48% | 20.69% | 19,454,464 | 61.64% | 1.70% | 4.71 |
| l1up | 2048 | 115 | 24.12% | 65.44% | 622,592,000 | 98.61% | 1.49% | 6.65 |

All variants use 8,192 bytes dynamic shared memory and 25% theoretical
occupancy. The Nsight recommendations report roughly 33% shared-load
wavefront conflicts for the BM64/BN64 tile and register pressure as the
theoretical occupancy limit. The raw CSVs contain the full shared-memory,
source-counter, scheduler, and launch-statistic rows.

The low-M kernels are severely under-occupied: achieved occupancy is about
6.2–6.5% and SM Busy is 5.9–20.7%. At M=2048, achieved occupancy rises to
21.8–24.1% and SM Busy to 53.8–66.0%, while warp cycles per instruction rise
to about 6.5–6.7. This supports prioritizing cross-row reuse and producer /
consumer scheduling at low M, and reducing register/shared-memory pressure at
large M.

Nsight emitted no eligible values for the Tensor-Core hierarchical roofline
section. This is an instrumentation limitation for this Triton `_gemm` profile,
not evidence that the model uses no Tensor Core work. Decoder/GEMM overlap is
also not directly measured by this single-kernel capture.

Raw reports and CSVs:

- [l0 q](results/window-ncu-wave22/l0q.json) / [CSV](results/window-ncu-wave22/l0q.csv)
- [l1 q](results/window-ncu-wave22/l1q.json) / [CSV](results/window-ncu-wave22/l1q.csv)
- [l0 gate](results/window-ncu-wave22/l0gate.json) / [CSV](results/window-ncu-wave22/l0gate.csv)
- [l1 gate](results/window-ncu-wave22/l1gate.json) / [CSV](results/window-ncu-wave22/l1gate.csv)
- [l0 up](results/window-ncu-wave22/l0up.json) / [CSV](results/window-ncu-wave22/l0up.csv)
- [l1 up](results/window-ncu-wave22/l1up.json) / [CSV](results/window-ncu-wave22/l1up.csv)
- [l0 down](results/window-ncu-wave22/l0down.json) / [CSV](results/window-ncu-wave22/l0down.csv)
- [l1 down](results/window-ncu-wave22/l1down.json) / [CSV](results/window-ncu-wave22/l1down.csv)
