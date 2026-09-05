# Decoder decomposition Nsight Compute wave 4

Four queue jobs collected Nsight Compute CSV data for representative production
window projections on separate A100 GPUs. The jobs covered layer-0 q, v, up,
and down projections at M=1, 16, and 2048, with the planar comparator and
window kernels captured under the CUDA profiler markers.

The selected sections were `SpeedOfLight`, `Occupancy`,
`MemoryWorkloadAnalysis`, `InstructionStats`, and `LaunchStats`. The raw CSVs
therefore include kernel duration, registers per thread, theoretical and
achieved occupancy, SM/memory/L2 throughput, executed and issued instructions,
spilling, and launch-tail diagnostics. The reports also retain the three
correctness rows per projection; all 12/12 candidate rows passed the local
gate.

Representative window-kernel observations from the raw counters include:

| Projection | M | Duration | Registers/thread | Achieved occupancy | Executed instructions |
|---|---:|---:|---:|---:|---:|
| layer-0 q | 1 | 10.848 us | 64 | 30.94% | 2,295,552 |
| layer-0 q | 16 | 35.968 us | 64 | 6.48% | 2,702,464 |
| layer-0 q | 2048 | 976.576 us | 64 | 46.54% | 335,093,760 |

These are kernel counters from one representative projection, not a model
speedup claim. The launch records show severe small-grid occupancy/tail effects
for some M=1/16 kernels, while large-M window kernels have substantially higher
achieved occupancy. The CSVs should be used for the full per-kernel comparison
across q/v/up/down.

Raw artifacts:

- [q projection](results/decomposition-ncu-wave4/q0.csv) and [JSON](results/decomposition-ncu-wave4/q0.json)
- [v projection](results/decomposition-ncu-wave4/v0.csv) and [JSON](results/decomposition-ncu-wave4/v0.json)
- [up projection](results/decomposition-ncu-wave4/up0.csv) and [JSON](results/decomposition-ncu-wave4/up0.json)
- [down projection](results/decomposition-ncu-wave4/down0.csv) and [JSON](results/decomposition-ncu-wave4/down0.json)

This is the first collected instruction/occupancy/memory-counter wave for the
window decomposition. Tensor-Core-specific workload counters, warp-state
stall attribution, and concurrent decoder/MMA overlap still need a focused
follow-up collection.
