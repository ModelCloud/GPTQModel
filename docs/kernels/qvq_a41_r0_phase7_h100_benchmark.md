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

## Full rate and row sweep

The formal sweep evaluates every combination of query split
\(\{1,2,4,8\}\) and shared key/value split \(\{1,2,4,8\}\).  Split 8 for all
three children is the winner in every one of the 20 rate/row cases.

| M | W | K | Q/K/V N | Query split | Key/value split | Median (us) | vs grouped split 1 | Better | Repeatable | Graph | Max abs |
|---:|---:|---:|:---|---:|---:|---:|---:|:---:|:---:|:---:|---:|
| 1 | 2 | 2048 | 2048/512/512 | 8 | 8 | 13.216 | 1.724x | Yes | Yes | Yes | 1.43e-06 |
| 2 | 2 | 2048 | 2048/512/512 | 8 | 8 | 13.234 | 1.664x | Yes | Yes | Yes | 1.91e-06 |
| 4 | 2 | 2048 | 2048/512/512 | 8 | 8 | 13.201 | 1.662x | Yes | Yes | Yes | 1.67e-06 |
| 8 | 2 | 2048 | 2048/512/512 | 8 | 8 | 13.301 | 1.648x | Yes | Yes | Yes | 1.91e-06 |
| 16 | 2 | 2048 | 2048/512/512 | 8 | 8 | 13.123 | 1.674x | Yes | Yes | Yes | 1.91e-06 |
| 1 | 2.5 | 2048 | 2048/512/512 | 8 | 8 | 13.407 | 1.561x | Yes | Yes | Yes | 1.43e-06 |
| 2 | 2.5 | 2048 | 2048/512/512 | 8 | 8 | 13.406 | 1.525x | Yes | Yes | Yes | 1.67e-06 |
| 4 | 2.5 | 2048 | 2048/512/512 | 8 | 8 | 13.414 | 1.523x | Yes | Yes | Yes | 1.67e-06 |
| 8 | 2.5 | 2048 | 2048/512/512 | 8 | 8 | 13.619 | 1.505x | Yes | Yes | Yes | 1.67e-06 |
| 16 | 2.5 | 2048 | 2048/512/512 | 8 | 8 | 13.654 | 1.502x | Yes | Yes | Yes | 2.38e-06 |
| 1 | 3 | 2048 | 2048/512/512 | 8 | 8 | 13.782 | 1.544x | Yes | Yes | Yes | 1.67e-06 |
| 2 | 3 | 2048 | 2048/512/512 | 8 | 8 | 13.520 | 1.525x | Yes | Yes | Yes | 1.43e-06 |
| 4 | 3 | 2048 | 2048/512/512 | 8 | 8 | 13.622 | 1.504x | Yes | Yes | Yes | 1.91e-06 |
| 8 | 3 | 2048 | 2048/512/512 | 8 | 8 | 13.483 | 1.522x | Yes | Yes | Yes | 1.79e-06 |
| 16 | 3 | 2048 | 2048/512/512 | 8 | 8 | 13.505 | 1.517x | Yes | Yes | Yes | 1.67e-06 |
| 1 | 3.5 | 2048 | 2048/512/512 | 8 | 8 | 13.278 | 1.594x | Yes | Yes | Yes | 1.43e-06 |
| 2 | 3.5 | 2048 | 2048/512/512 | 8 | 8 | 13.481 | 1.513x | Yes | Yes | Yes | 1.67e-06 |
| 4 | 3.5 | 2048 | 2048/512/512 | 8 | 8 | 13.241 | 1.554x | Yes | Yes | Yes | 1.91e-06 |
| 8 | 3.5 | 2048 | 2048/512/512 | 8 | 8 | 13.198 | 1.559x | Yes | Yes | Yes | 1.91e-06 |
| 16 | 3.5 | 2048 | 2048/512/512 | 8 | 8 | 13.349 | 1.529x | Yes | Yes | Yes | 1.91e-06 |

This is an inner-operation policy sweep, so Marlin and Machete are not
applicable to this table.  The next production benchmark applies the winning
policy to the complete grouped projections and reports both baselines.

The full 320-row record is
`artifacts/a41_phase7_h100/qkv_all_rates_ordered_sweep.json`.

## Production W3 M1 probe

The production path includes the one shared input scale/Hadamard transform,
the grouped ordered inner kernel, three independent child output recoveries,
and the sibling coordinator.  The prior grouped result is the committed Phase
4 artifact at `566cc427`.

| M | K | Q/K/V N | QVQ W3 (us) | Previous QVQ (us) | vs previous | Marlin W4 (us) | vs Marlin | Machete W4 (us) | vs Machete | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 2048 | 2048/512/512 | 38.002 | 43.873 | 1.154x | 49.880 | 1.313x | 35.311 | 0.929x | Yes |

The result is deterministic and its maximum absolute difference from the old
split-1 output is `0.001953125`.  The grouped output is bit-exact to three
independent ordered split-8 child kernels; the difference from split 1 is the
expected FP32 reduction parenthesization change.

The raw probe is
`artifacts/a41_phase7_h100/production_qkv_w3_m1_probe.json`.
