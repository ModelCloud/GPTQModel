# Decoder decomposition wave 2

This wave used the immutable F6 seed-7 snapshot, captured real FP32-teacher
activations, and `layer_baseline.py` on four assigned GPUs. It measured the
existing planar P32 path against the Ampere window path for 12 projections at

`M = 1, 2, 4, 8, 16, 32, 128, 512, 2048`.

All 108 cases completed and passed the local reconstruction gate. The median
full-layer speedup (planar time divided by window time) across the 12
projections was:

| M | Median speedup | Median planar full (ms) | Median window full (ms) |
|---:|---:|---:|---:|
| 1 | 1.035x | 1.8043 | 1.7746 |
| 16 | 1.019x | 1.7884 | 1.7572 |
| 128 | 1.507x | 2.7822 | 1.8094 |
| 512 | 3.497x | 7.5878 | 2.0224 |
| 2048 | 5.549x | 27.3940 | 4.7278 |

The inner P32 median at M=1 was 0.4157 ms versus 0.0466 ms for the window
path, but input scaling and Hadamard work dominate the full-layer result at
this row count. The M=1 standalone materialization stages were also recorded:
state extraction was about 0.51–0.86 ms and bank lookup about 1.30–3.07 ms,
depending on projection shape. These are separate diagnostic timings and are
not an end-to-end model speedup claim.

The raw reports are [worker 0](results/decomposition-wave2/worker0.json),
[worker 1](results/decomposition-wave2/worker1.json),
[worker 2](results/decomposition-wave2/worker2.json), and
[worker 3](results/decomposition-wave2/worker3.json). Each report has
`complete: true` and 27 rows.

## Queue bookkeeping

The commands returned exit code 0 and wrote the complete reports, but the
dispatcher recorded four failures because the manifest `result` paths ended in
`/report.json` while `layer_baseline.py` writes to the exact path passed to
`--output` (the paths ending in `worker0` through `worker3`). The failed queue
records are retained as history; the archived files above are the authoritative
collected results. Future entries use the exact output path.

This wave advances experiment 1 and the exact-format performance evidence.
Profiler counters, decoder/GEMM overlap, and full-model timing remain open.
