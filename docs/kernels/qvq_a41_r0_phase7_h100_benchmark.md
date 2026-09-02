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

## Production query/key/value matrix

This is the complete production projection path at all requested rates and
rows.  `vs previous` compares against the committed Phase 4 grouped split-1
artifact.  A ratio above one means QVQ is faster.

| W | M | K | Q/K/V N | QVQ (us) | vs previous | Better | Marlin W4 (us) | vs Marlin | Machete W4 (us) | vs Machete |
|---:|---:|---:|:---|---:|---:|:---:|---:|---:|---:|---:|
| 2 | 1 | 2048 | 2048/512/512 | 36.230 | 1.249x | Yes | 49.868 | 1.376x | 35.098 | 0.969x |
| 2 | 2 | 2048 | 2048/512/512 | 36.371 | 1.241x | Yes | 58.725 | 1.615x | 34.793 | 0.957x |
| 2 | 4 | 2048 | 2048/512/512 | 36.648 | 1.238x | Yes | 59.188 | 1.615x | 34.652 | 0.946x |
| 2 | 8 | 2048 | 2048/512/512 | 37.067 | 1.235x | Yes | 51.114 | 1.379x | 34.136 | 0.921x |
| 2 | 16 | 2048 | 2048/512/512 | 34.972 | 1.241x | Yes | 55.172 | 1.578x | 34.270 | 0.980x |
| 2.5 | 1 | 2048 | 2048/512/512 | 36.611 | 1.195x | Yes | 49.868 | 1.362x | 35.098 | 0.959x |
| 2.5 | 2 | 2048 | 2048/512/512 | 36.566 | 1.195x | Yes | 58.725 | 1.606x | 34.793 | 0.951x |
| 2.5 | 4 | 2048 | 2048/512/512 | 37.002 | 1.191x | Yes | 59.188 | 1.600x | 34.652 | 0.937x |
| 2.5 | 8 | 2048 | 2048/512/512 | 37.117 | 1.194x | Yes | 51.114 | 1.377x | 34.136 | 0.920x |
| 2.5 | 16 | 2048 | 2048/512/512 | 35.453 | 1.182x | Yes | 55.172 | 1.556x | 34.270 | 0.967x |
| 3 | 1 | 2048 | 2048/512/512 | 36.843 | 1.191x | Yes | 49.868 | 1.354x | 35.098 | 0.953x |
| 3 | 2 | 2048 | 2048/512/512 | 36.469 | 1.196x | Yes | 58.725 | 1.610x | 34.793 | 0.954x |
| 3 | 4 | 2048 | 2048/512/512 | 36.741 | 1.191x | Yes | 59.188 | 1.611x | 34.652 | 0.943x |
| 3 | 8 | 2048 | 2048/512/512 | 37.302 | 1.180x | Yes | 51.114 | 1.370x | 34.136 | 0.915x |
| 3 | 16 | 2048 | 2048/512/512 | 35.143 | 1.196x | Yes | 55.172 | 1.570x | 34.270 | 0.975x |
| 3.5 | 1 | 2048 | 2048/512/512 | 36.367 | 1.206x | Yes | 49.868 | 1.371x | 35.098 | 0.965x |
| 3.5 | 2 | 2048 | 2048/512/512 | 36.510 | 1.195x | Yes | 58.725 | 1.608x | 34.793 | 0.953x |
| 3.5 | 4 | 2048 | 2048/512/512 | 36.680 | 1.204x | Yes | 59.188 | 1.614x | 34.652 | 0.945x |
| 3.5 | 8 | 2048 | 2048/512/512 | 36.682 | 1.205x | Yes | 51.114 | 1.393x | 34.136 | 0.931x |
| 3.5 | 16 | 2048 | 2048/512/512 | 34.994 | 1.200x | Yes | 55.172 | 1.577x | 34.270 | 0.979x |

Geometric means across the 20 cases are:

| Comparison | QVQ ratio |
|:---|---:|
| vs previous grouped QVQ | 1.206x |
| vs ordinary per-child QVQ | 2.680x |
| vs Marlin W4 | 1.503x |
| vs Machete W4 | 0.951x |

Thus Phase 7 removes 17.1% of the previous grouped latency on the geometric
mean (`1 - 1/1.206`) and leaves a 5.2% geometric-mean latency gap to Machete
(`1/0.951 - 1`).  Every case is deterministic and better than the previous
benchmark.  The largest recorded difference from the old split-1 output is
`0.00390625`; correctness is defined by bit-exact equality to independent
ordered split-8 children plus the existing dense-inner accuracy gate.

The raw record is
`artifacts/a41_phase7_h100/production_qkv_vs_baselines.json`.
