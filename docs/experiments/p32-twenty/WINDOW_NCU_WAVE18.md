# Fused window Nsight wave 18

Wave 18 profiled the actual BM64/BN64/split-1 fused decode/MMA kernel on the
immutable F6 seed-7 checkpoint. Eight distinct projections were profiled at
`M=1,16,2048`, one projection per UUID-pinned GPU. The fused report checks
passed all 24/24 local teacher and window-reference cases.

The profiler captured `_gemm` with 128 threads per block. Query and down
projections use a 32-wide N grid; gate and up projections use a 128-wide N
grid. The M=1 and M=16 calls have the same grid shape because the fused tile
height is 64, so their profiler IDs are separated by launch order.

## Kernel scheduler measurements

The values are from the fused `_gemm` kernel. `SM Busy`, `Issue Slots Busy`,
and `No Eligible` are percentages; IPC values are instructions per cycle.

| Projection | M | Grid | SM busy | Issue slots | Executed IPC active | Executed IPC elapsed | No eligible |
|---|---:|---|---:|---:|---:|---:|---:|
| l0 self-attn q | 1 | 1x32 | 5.90 | 5.50 | 0.86 | 0.22 | 78.42 |
| l0 self-attn q | 16 | 1x32 | 6.05 | 5.32 | 0.84 | 0.21 | 78.93 |
| l0 self-attn q | 2048 | 32x32 | 54.16 | 47.63 | 2.18 | 1.90 | 45.50 |
| l1 self-attn q | 1 | 1x32 | 5.88 | 5.48 | 0.86 | 0.22 | 78.43 |
| l1 self-attn q | 16 | 1x32 | 6.03 | 5.30 | 0.84 | 0.21 | 78.89 |
| l1 self-attn q | 2048 | 32x32 | 54.29 | 47.74 | 2.18 | 1.91 | 45.50 |
| l0 mlp down | 1 | 1x32 | 6.23 | 5.41 | 0.86 | 0.22 | 78.41 |
| l0 mlp down | 16 | 1x32 | 6.24 | 5.41 | 0.86 | 0.22 | 78.46 |
| l0 mlp down | 2048 | 32x32 | 53.83 | 46.72 | 2.11 | 1.87 | 47.15 |
| l1 mlp down | 1 | 1x32 | 6.24 | 5.42 | 0.87 | 0.22 | 78.34 |
| l1 mlp down | 16 | 1x32 | 6.24 | 5.42 | 0.86 | 0.22 | 78.39 |
| l1 mlp down | 2048 | 32x32 | 53.79 | 46.69 | 2.11 | 1.87 | 47.14 |
| l0 mlp gate | 1 | 1x128 | 20.38 | 17.70 | 0.88 | 0.71 | 77.94 |
| l0 mlp gate | 16 | 1x128 | 20.39 | 17.71 | 0.88 | 0.71 | 77.95 |
| l0 mlp gate | 2048 | 32x128 | 66.09 | 57.39 | 2.34 | 2.30 | 41.59 |
| l1 mlp gate | 1 | 1x128 | 20.43 | 17.74 | 0.88 | 0.71 | 77.92 |
| l1 mlp gate | 16 | 1x128 | 20.35 | 17.67 | 0.88 | 0.71 | 77.95 |
| l1 mlp gate | 2048 | 32x128 | 65.86 | 57.19 | 2.33 | 2.29 | 41.67 |
| l0 mlp up | 1 | 1x128 | 20.60 | 17.89 | 0.88 | 0.72 | 77.90 |
| l0 mlp up | 16 | 1x128 | 20.74 | 18.02 | 0.88 | 0.72 | 77.91 |
| l0 mlp up | 2048 | 32x128 | 65.42 | 56.81 | 2.32 | 2.27 | 42.07 |
| l1 mlp up | 1 | 1x128 | 20.81 | 18.08 | 0.88 | 0.72 | 77.90 |
| l1 mlp up | 16 | 1x128 | 20.68 | 17.96 | 0.88 | 0.72 | 77.91 |
| l1 mlp up | 2048 | 32x128 | 65.49 | 56.87 | 2.32 | 2.27 | 42.10 |

The M=1/16 cases spend about 78% of cycles with no eligible warps and expose
the low parallelism that makes a producer/consumer schedule worth testing.
At M=2048 the grid supplies more work, but no-eligible cycles remain about
42–47% and SM busy is only about 54–66%.

The requested Nsight sections emitted scheduler, warp-state, compute-workload,
and source-counter rows. This run did not emit tensor-core workload,
register/thread, occupancy, shared-memory, L2-traffic, or decoder/GEMM-overlap
fields for the fused kernel, so those scorecard fields remain open rather
than being inferred from SM busy. The raw CSVs are retained for a follow-up
metric selection pass.

## Reproducibility

Accuracy reports and raw Nsight CSVs:

- [l0 q GPU0 report](results/window-ncu-wave18/l0q-gpu0)
- [l1 q GPU1 report](results/window-ncu-wave18/l1q-gpu1)
- [l0 gate GPU2 report](results/window-ncu-wave18/l0gate-gpu2)
- [l1 gate GPU3 report](results/window-ncu-wave18/l1gate-gpu3)
- [l0 up GPU4 report](results/window-ncu-wave18/l0up-gpu4)
- [l1 up GPU5 report](results/window-ncu-wave18/l1up-gpu5)
- [l0 down GPU6 report](results/window-ncu-wave18/l0down-gpu6)
- [l1 down GPU7 report](results/window-ncu-wave18/l1down-gpu7)
