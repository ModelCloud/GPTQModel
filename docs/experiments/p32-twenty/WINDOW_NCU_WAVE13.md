# Window Nsight wave 13

The queue ran eight UUID-pinned Nsight Compute jobs concurrently on distinct
layer-0/layer-1 q, gate, up, and down projections. Each job captured M=1, 16,
and 2048 with the production window kernel and retained the local teacher
comparison. All 24 JSON cases passed the local correctness gate.

The capture sections were `ComputeWorkloadAnalysis`, `SchedulerStats`,
`WarpStateStats`, `SpeedOfLight_HierarchicalTensorRooflineChart`, and
`SourceCounters`. The raw CSVs contain SM/issue busy, eligible/no-eligible
warp percentages, warp cycles per instruction, branch efficiency, source
instruction counters, and the tensor-roofline section output where supported.

For the layer-0 q projection, the window kernel reported 64 registers/thread
at all three shapes. Achieved occupancy was 30.94% at M=1, 6.48% at M=16,
and 46.54% at M=2048; executed instructions were 2,295,552, 2,702,464, and
335,093,760 respectively. Across the deeper wave, the window kernels showed
large no-eligible-warp fractions for several M=16 launches, while M=2048
launches generally had about 59–62% SM busy. These counters support focusing
on launch/grid and producer-consumer scheduling work before more decoder
algebra changes.

Raw paired JSON/CSV reports:

- [layer-0 q, GPU0](results/window-ncu-wave13/l0q-gpu0.json) / [CSV](results/window-ncu-wave13/l0q-gpu0.csv)
- [layer-1 q, GPU1](results/window-ncu-wave13/l1q-gpu1.json) / [CSV](results/window-ncu-wave13/l1q-gpu1.csv)
- [layer-0 gate, GPU2](results/window-ncu-wave13/l0gate-gpu2.json) / [CSV](results/window-ncu-wave13/l0gate-gpu2.csv)
- [layer-1 gate, GPU3](results/window-ncu-wave13/l1gate-gpu3.json) / [CSV](results/window-ncu-wave13/l1gate-gpu3.csv)
- [layer-0 down, GPU4](results/window-ncu-wave13/l0down-gpu4.json) / [CSV](results/window-ncu-wave13/l0down-gpu4.csv)
- [layer-1 down, GPU5](results/window-ncu-wave13/l1down-gpu5.json) / [CSV](results/window-ncu-wave13/l1down-gpu5.csv)
- [layer-0 up, GPU6](results/window-ncu-wave13/l0up-gpu6.json) / [CSV](results/window-ncu-wave13/l0up-gpu6.csv)
- [layer-1 up, GPU7](results/window-ncu-wave13/l1up-gpu7.json) / [CSV](results/window-ncu-wave13/l1up-gpu7.csv)
