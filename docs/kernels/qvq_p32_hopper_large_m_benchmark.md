# QVQ P32 Hopper large-M benchmark

## Method

The benchmark uses the physical 132-SM NVIDIA H100 exclusively.  It rejects
foreign compute processes, requires three consecutive zero-utilization
samples before timing, captures the complete projection site in a CUDA Graph,
and records replay latency with CUDA events.  Each row uses 20 warmups, 40
samples, and 20 graph replays per sample.

`vs Marlin` and `vs Machete` are latency speedup ratios against their W4
kernel.  A value above one means QVQ is faster.  `Better than last` compares
against the preceding merged-main or committed-stage QVQ path; a regression is
reported as `No`.

## M16 and M32

| Rate | Site | M x K x aggregate N | QVQ us | vs merged main | vs Marlin W4 | vs Machete W4 | Better than last | Mean error | Max error |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| W2 | QKV | 16 x 2048 x 3072 | 28.285 | 3.005x | 1.928x | 1.222x | Yes | 1.717e-6 | 1.032e-5 |
| W2 | QKV | 32 x 2048 x 3072 | 40.654 | 5.864x | 2.232x | 0.911x | Yes | 1.731e-6 | 1.222e-5 |
| W2 | gate/up | 16 x 2048 x 16384 | 30.984 | 2.891x | 0.505x | 0.937x | Yes | 1.875e-6 | 1.306e-5 |
| W2 | gate/up | 32 x 2048 x 16384 | 62.730 | 17.048x | 0.410x | 0.505x | Yes | 1.887e-6 | 1.354e-5 |
| W2.5 | QKV | 16 x 2048 x 3072 | 28.315 | 3.014x | 1.926x | 1.221x | Yes | 1.730e-6 | 1.397e-5 |
| W2.5 | QKV | 32 x 2048 x 3072 | 40.470 | 5.917x | 2.242x | 0.915x | Yes | 1.730e-6 | 1.417e-5 |
| W2.5 | gate/up | 16 x 2048 x 16384 | 31.944 | 2.787x | 0.490x | 0.909x | Yes | 1.884e-6 | 1.334e-5 |
| W2.5 | gate/up | 32 x 2048 x 16384 | 62.534 | 19.326x | 0.411x | 0.506x | Yes | 1.885e-6 | 1.285e-5 |
| W3 | QKV | 16 x 2048 x 3072 | 28.282 | 2.948x | 1.928x | 1.222x | Yes | 1.731e-6 | 1.169e-5 |
| W3 | QKV | 32 x 2048 x 3072 | 40.057 | 6.277x | 2.265x | 0.925x | Yes | 1.723e-6 | 1.155e-5 |
| W3 | gate/up | 16 x 2048 x 16384 | 31.263 | 2.829x | 0.501x | 0.929x | Yes | 1.873e-6 | 1.255e-5 |
| W3 | gate/up | 32 x 2048 x 16384 | 63.703 | 17.119x | 0.404x | 0.497x | Yes | 1.890e-6 | 1.441e-5 |
| W3.5 | QKV | 16 x 2048 x 3072 | 28.470 | 2.942x | 1.916x | 1.214x | Yes | 1.739e-6 | 1.213e-5 |
| W3.5 | QKV | 32 x 2048 x 3072 | 40.470 | 5.902x | 2.242x | 0.915x | Yes | 1.731e-6 | 1.555e-5 |
| W3.5 | gate/up | 16 x 2048 x 16384 | 31.688 | 2.802x | 0.494x | 0.916x | Yes | 1.886e-6 | 1.232e-5 |
| W3.5 | gate/up | 32 x 2048 x 16384 | 65.198 | 17.974x | 0.394x | 0.486x | Yes | 1.887e-6 | 1.381e-5 |

## M64

| Rate | Site | M x K x aggregate N | QVQ us | vs previous M64 fallback | vs Marlin W4 | vs Machete W4 | Better than last | Mean error | Max error |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| W2 | QKV | 64 x 2048 x 3072 | 49.610 | 8.256x | 2.794x | 0.773x | Yes | 1.740e-6 | 1.378e-5 |
| W2 | gate/up | 64 x 2048 x 16384 | 94.587 | 20.317x | 0.411x | 0.371x | Yes | 1.900e-6 | 1.336e-5 |
| W2.5 | QKV | 64 x 2048 x 3072 | 49.918 | 8.290x | 2.776x | 0.768x | Yes | 1.737e-6 | 1.213e-5 |
| W2.5 | gate/up | 64 x 2048 x 16384 | 94.985 | 21.218x | 0.409x | 0.370x | Yes | 1.895e-6 | 1.564e-5 |
| W3 | QKV | 64 x 2048 x 3072 | 49.661 | 8.475x | 2.791x | 0.772x | Yes | 1.739e-6 | 1.383e-5 |
| W3 | gate/up | 64 x 2048 x 16384 | 96.098 | 20.095x | 0.404x | 0.366x | Yes | 1.893e-6 | 1.341e-5 |
| W3.5 | QKV | 64 x 2048 x 3072 | 50.000 | 8.221x | 2.772x | 0.767x | Yes | 1.737e-6 | 1.277e-5 |
| W3.5 | gate/up | 64 x 2048 x 16384 | 99.538 | 20.141x | 0.390x | 0.353x | Yes | 1.895e-6 | 1.269e-5 |

## Interpretation

The row-grid promotion removes the M greater than 16 planar fallback and is a
large end-to-end win.  It does not yet reuse decoded weights across row tiles.
That is why QKV remains within roughly 23 percent of Machete at M64 while the
wide gate/up group is still 2.7 to 2.8 times slower.  The next M64 kernel
experiment must amortize P32 decode over multiple independent M16 WGMMA input
fragments rather than only increasing the row-grid size.
