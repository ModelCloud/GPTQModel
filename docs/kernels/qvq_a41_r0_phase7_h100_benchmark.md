# QVQ A41/R0 Phase 7 H100 benchmark

The benchmark uses only the physical NVIDIA H100 at PCI address
`00000000:44:00.0`; the H200 is hidden.  It requires three idle-device samples
before Torch import.  Warmed CUDA Graph replays are timed by CUDA events so
CPU and container scheduling gaps are excluded.

## W3 M1 policy probe

The first probe measures the grouped P32 inner query/key/value operation at

\[
M=1,\quad K=2048,\quad (N_Q,N_K,N_V)=(2048,512,512).
\]

| Query split | Key/value split | Useful blocks | Median (us) | Speedup vs existing grouped split 1 | Better |
|---:|---:|---:|---:|---:|:---:|
| 1 | 1 | 48 | 24.819 | 0.852x | No |
| 2 | 2 | 96 | 16.386 | 1.291x | Yes |
| 4 | 4 | 192 | 14.054 | 1.505x | Yes |
| 8 | 8 | 384 | 13.611 | 1.554x | Yes |

The ordered split-1 path is slower because it pays three copy-reduction
launches without gaining decoder parallelism.  Split 8 fills the H100 and wins
the W3/M1 probe.  All 16 query versus key/value split combinations passed
repeatability, CUDA Graph, and dense accuracy gates.

The raw record is
`artifacts/a41_phase7_h100/qkv_w3_m1_probe.json`.
