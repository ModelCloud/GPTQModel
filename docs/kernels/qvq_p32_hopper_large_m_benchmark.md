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

## M128 and M256

| Rate | Site | M x K x aggregate N | QVQ us | vs previous fallback | vs Marlin W4 | vs Machete W4 | Better than last | Mean error | Max error |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| W2 | QKV | 128 x 2048 x 3072 | 69.390 | 11.552x | 1.734x | 0.586x | Yes | 1.724e-6 | 1.337e-5 |
| W2 | QKV | 256 x 2048 x 3072 | 117.099 | 12.513x | 0.967x | 0.388x | Yes | 1.731e-6 | 1.479e-5 |
| W2 | gate/up | 128 x 2048 x 16384 | 170.697 | 21.742x | 0.224x | 0.212x | Yes | 1.887e-6 | 1.544e-5 |
| W2 | gate/up | 256 x 2048 x 16384 | 324.782 | 22.448x | 0.165x | 0.134x | Yes | 1.885e-6 | 1.741e-5 |
| W2.5 | QKV | 128 x 2048 x 3072 | 69.495 | 12.543x | 1.731x | 0.585x | Yes | 1.730e-6 | 1.293e-5 |
| W2.5 | QKV | 256 x 2048 x 3072 | 117.806 | 12.823x | 0.961x | 0.386x | Yes | 1.729e-6 | 1.403e-5 |
| W2.5 | gate/up | 128 x 2048 x 16384 | 171.586 | 21.926x | 0.223x | 0.211x | Yes | 1.887e-6 | 1.497e-5 |
| W2.5 | gate/up | 256 x 2048 x 16384 | 327.305 | 22.472x | 0.164x | 0.133x | Yes | 1.886e-6 | 1.438e-5 |
| W3 | QKV | 128 x 2048 x 3072 | 69.568 | 11.911x | 1.729x | 0.585x | Yes | 1.734e-6 | 1.261e-5 |
| W3 | QKV | 256 x 2048 x 3072 | 118.449 | 12.544x | 0.956x | 0.384x | Yes | 1.726e-6 | 1.377e-5 |
| W3 | gate/up | 128 x 2048 x 16384 | 176.604 | 21.029x | 0.217x | 0.205x | Yes | 1.886e-6 | 1.398e-5 |
| W3 | gate/up | 256 x 2048 x 16384 | 329.926 | 22.071x | 0.163x | 0.132x | Yes | 1.886e-6 | 1.607e-5 |
| W3.5 | QKV | 128 x 2048 x 3072 | 71.074 | 12.080x | 1.693x | 0.572x | Yes | 1.729e-6 | 1.592e-5 |
| W3.5 | QKV | 256 x 2048 x 3072 | 120.398 | 12.495x | 0.940x | 0.378x | Yes | 1.730e-6 | 1.314e-5 |
| W3.5 | gate/up | 128 x 2048 x 16384 | 180.295 | 20.782x | 0.212x | 0.201x | Yes | 1.886e-6 | 1.636e-5 |
| W3.5 | gate/up | 256 x 2048 x 16384 | 335.638 | 21.843x | 0.160x | 0.130x | Yes | 1.886e-6 | 1.640e-5 |

All sixteen M128/M256 cells improve over the preceding planar fallback.  The
largest maximum absolute error is `1.741e-5`; the M128 and M256 CUDA Graph
tests are bit-exact to the corresponding sequence of M16 row tiles.

## M32/M64 decoded-weight reuse candidate

The reuse-2 CTA loads two M16 activation tiles, decodes each P32 fragment once,
and issues two independent WGMMA operations before overwriting the fragment.
The first W3 complete-site acceptance run is:

| Rate | Site | M x K x aggregate N | QVQ us | vs prior row grid | vs Marlin W4 | vs Machete W4 | Better than last |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| W3 | QKV | 32 x 2048 x 3072 | 35.775 | 1.120x | 2.561x | 1.041x | Yes |
| W3 | QKV | 64 x 2048 x 3072 | 44.474 | 1.117x | 3.129x | 0.856x | Yes |
| W3 | gate/up | 32 x 2048 x 16384 | 54.819 | 1.162x | 0.472x | 0.573x | Yes |
| W3 | gate/up | 64 x 2048 x 16384 | 82.642 | 1.163x | 0.466x | 0.426x | Yes |

The candidate is bit-exact to the ordinary row grid and tiled-M16 reference
for W2, W2.5, W3, and W3.5 at M32, M64, M128, and M256.  CUDA Graph replay is
also exact.  The first W3 timing promotes reuse-2; the full-rate matrix remains
the next benchmark gate.

## Promoted reuse-2 matrix

All 32 complete-site cells improve over the prior row grid.  Geometric mean
speedup is `1.213x`, with individual improvements from `1.123x` through
`1.444x`.

| Rate | Site | M x K x aggregate N | QVQ us | vs prior row grid | vs Marlin W4 | vs Machete W4 | Better than last |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| W2 | QKV | 32 x 2048 x 3072 | 35.215 | 1.154x | 2.637x | 1.063x | Yes |
| W2 | QKV | 64 x 2048 x 3072 | 43.543 | 1.139x | 3.208x | 0.879x | Yes |
| W2 | QKV | 128 x 2048 x 3072 | 57.935 | 1.198x | 2.084x | 0.704x | Yes |
| W2 | QKV | 256 x 2048 x 3072 | 93.960 | 1.246x | 1.206x | 0.488x | Yes |
| W2 | gate/up | 32 x 2048 x 16384 | 49.995 | 1.255x | 0.518x | 0.633x | Yes |
| W2 | gate/up | 64 x 2048 x 16384 | 71.574 | 1.322x | 0.541x | 0.490x | Yes |
| W2 | gate/up | 128 x 2048 x 16384 | 120.841 | 1.413x | 0.318x | 0.303x | Yes |
| W2 | gate/up | 256 x 2048 x 16384 | 224.937 | 1.444x | 0.238x | 0.192x | Yes |
| W2.5 | QKV | 32 x 2048 x 3072 | 35.898 | 1.127x | 2.587x | 1.042x | Yes |
| W2.5 | QKV | 64 x 2048 x 3072 | 44.240 | 1.128x | 3.158x | 0.866x | Yes |
| W2.5 | QKV | 128 x 2048 x 3072 | 60.842 | 1.142x | 1.984x | 0.670x | Yes |
| W2.5 | QKV | 256 x 2048 x 3072 | 100.795 | 1.169x | 1.124x | 0.455x | Yes |
| W2.5 | gate/up | 32 x 2048 x 16384 | 54.982 | 1.137x | 0.471x | 0.576x | Yes |
| W2.5 | gate/up | 64 x 2048 x 16384 | 81.991 | 1.158x | 0.472x | 0.428x | Yes |
| W2.5 | gate/up | 128 x 2048 x 16384 | 133.172 | 1.288x | 0.289x | 0.275x | Yes |
| W2.5 | gate/up | 256 x 2048 x 16384 | 251.038 | 1.304x | 0.213x | 0.172x | Yes |
| W3 | QKV | 32 x 2048 x 3072 | 35.660 | 1.123x | 2.604x | 1.049x | Yes |
| W3 | QKV | 64 x 2048 x 3072 | 44.134 | 1.125x | 3.165x | 0.868x | Yes |
| W3 | QKV | 128 x 2048 x 3072 | 60.508 | 1.150x | 1.995x | 0.674x | Yes |
| W3 | QKV | 256 x 2048 x 3072 | 101.191 | 1.171x | 1.120x | 0.453x | Yes |
| W3 | gate/up | 32 x 2048 x 16384 | 55.234 | 1.153x | 0.469x | 0.573x | Yes |
| W3 | gate/up | 64 x 2048 x 16384 | 82.964 | 1.158x | 0.466x | 0.423x | Yes |
| W3 | gate/up | 128 x 2048 x 16384 | 133.443 | 1.323x | 0.288x | 0.275x | Yes |
| W3 | gate/up | 256 x 2048 x 16384 | 251.850 | 1.310x | 0.213x | 0.171x | Yes |
| W3.5 | QKV | 32 x 2048 x 3072 | 35.629 | 1.136x | 2.607x | 1.050x | Yes |
| W3.5 | QKV | 64 x 2048 x 3072 | 44.021 | 1.136x | 3.173x | 0.870x | Yes |
| W3.5 | QKV | 128 x 2048 x 3072 | 61.188 | 1.162x | 1.973x | 0.667x | Yes |
| W3.5 | QKV | 256 x 2048 x 3072 | 100.030 | 1.204x | 1.133x | 0.459x | Yes |
| W3.5 | gate/up | 32 x 2048 x 16384 | 54.035 | 1.207x | 0.479x | 0.586x | Yes |
| W3.5 | gate/up | 64 x 2048 x 16384 | 81.626 | 1.219x | 0.474x | 0.430x | Yes |
| W3.5 | gate/up | 128 x 2048 x 16384 | 130.856 | 1.378x | 0.294x | 0.280x | Yes |
| W3.5 | gate/up | 256 x 2048 x 16384 | 248.102 | 1.353x | 0.216x | 0.174x | Yes |

Across this matrix, mean absolute error remains below `1.91e-6` and maximum
absolute error remains below `1.75e-5` versus the dense P32 Torch oracle.

## Reuse-4 acceptance

Reuse-4 keeps four independent M16 accumulator fragments in one CTA and uses
each decoded P32 weight fragment four times.  The first W3 gate is:

| Rate | Site | M x K x aggregate N | QVQ us | vs reuse-2 | vs Marlin W4 | vs Machete W4 | Better than last |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| W3 | QKV | 64 x 2048 x 3072 | 42.349 | 1.042x | 3.298x | 0.917x | Yes |
| W3 | QKV | 128 x 2048 x 3072 | 51.632 | 1.172x | 2.330x | 0.802x | Yes |
| W3 | gate/up | 64 x 2048 x 16384 | 54.491 | 1.522x | 0.711x | 0.656x | Yes |
| W3 | gate/up | 128 x 2048 x 16384 | 95.114 | 1.403x | 0.407x | 0.386x | Yes |

All W2 through W3.5 low-level cases at M64, M128, and M256 are bit-exact to
reuse-2, the ordinary row grid, and tiled M16.  CUDA Graph replay is exact.
Reuse-4 is therefore promoted for row counts divisible by 64; reuse-2 remains
the M32 path.

### Full reuse-4 matrix

| Rate | Site | M x K x aggregate N | QVQ us | vs reuse-2 | vs Marlin W4 | vs Machete W4 | Better than last |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| W2 | QKV | 64 x 2048 x 3072 | 41.670 | 1.045x | 3.362x | 0.906x | Yes |
| W2 | QKV | 128 x 2048 x 3072 | 51.231 | 1.131x | 2.346x | 0.798x | Yes |
| W2 | QKV | 256 x 2048 x 3072 | 84.418 | 1.113x | 1.345x | 0.543x | Yes |
| W2 | gate/up | 64 x 2048 x 16384 | 53.674 | 1.334x | 0.720x | 0.663x | Yes |
| W2 | gate/up | 128 x 2048 x 16384 | 92.419 | 1.308x | 0.416x | 0.393x | Yes |
| W2 | gate/up | 256 x 2048 x 16384 | 174.541 | 1.289x | 0.304x | 0.248x | Yes |
| W2.5 | QKV | 64 x 2048 x 3072 | 42.325 | 1.045x | 3.310x | 0.892x | Yes |
| W2.5 | QKV | 128 x 2048 x 3072 | 51.734 | 1.176x | 2.323x | 0.791x | Yes |
| W2.5 | QKV | 256 x 2048 x 3072 | 85.322 | 1.181x | 1.331x | 0.537x | Yes |
| W2.5 | gate/up | 64 x 2048 x 16384 | 54.245 | 1.512x | 0.712x | 0.656x | Yes |
| W2.5 | gate/up | 128 x 2048 x 16384 | 95.118 | 1.400x | 0.404x | 0.382x | Yes |
| W2.5 | gate/up | 256 x 2048 x 16384 | 176.417 | 1.423x | 0.301x | 0.245x | Yes |
| W3 | QKV | 64 x 2048 x 3072 | 42.310 | 1.043x | 3.311x | 0.892x | Yes |
| W3 | QKV | 128 x 2048 x 3072 | 51.414 | 1.177x | 2.338x | 0.795x | Yes |
| W3 | QKV | 256 x 2048 x 3072 | 85.169 | 1.188x | 1.334x | 0.538x | Yes |
| W3 | gate/up | 64 x 2048 x 16384 | 54.649 | 1.518x | 0.707x | 0.651x | Yes |
| W3 | gate/up | 128 x 2048 x 16384 | 94.990 | 1.405x | 0.405x | 0.382x | Yes |
| W3 | gate/up | 256 x 2048 x 16384 | 180.027 | 1.399x | 0.295x | 0.240x | Yes |
| W3.5 | QKV | 64 x 2048 x 3072 | 42.550 | 1.035x | 3.293x | 0.887x | Yes |
| W3.5 | QKV | 128 x 2048 x 3072 | 52.124 | 1.174x | 2.306x | 0.785x | Yes |
| W3.5 | QKV | 256 x 2048 x 3072 | 85.617 | 1.168x | 1.327x | 0.535x | Yes |
| W3.5 | gate/up | 64 x 2048 x 16384 | 57.835 | 1.411x | 0.668x | 0.615x | Yes |
| W3.5 | gate/up | 128 x 2048 x 16384 | 102.139 | 1.281x | 0.376x | 0.355x | Yes |
| W3.5 | gate/up | 256 x 2048 x 16384 | 194.393 | 1.276x | 0.273x | 0.223x | Yes |

All 24 cells improve over reuse-2.  Geometric mean speedup is `1.243x`,
with a `1.035x` to `1.518x` range.

## Complete MLP through M4096

The complete workload includes grouped gate/up, SiLU, product, and down.  The
Llama down split staircase improves the M32-M256 matrix by `1.237x` geometric
mean versus ordinary per-module QVQ.  All M512-M4096 rows use split 1 because
the reuse-4 row grid already exposes at least 256 down CTAs.

| Rate | M; gate/up K,N; down K,N | QVQ us | vs Marlin W4 | vs Machete W4 | Better than ordinary QVQ | Max error |
| ---: | :--- | ---: | ---: | ---: | :---: | ---: |
| W2 | 512; 2048,8192; 8192,2048 | 524.091 | 0.315x | 0.226x | Yes | 1.013e-6 |
| W2 | 1024; 2048,8192; 8192,2048 | 1031.271 | 0.324x | 0.217x | Yes | 1.013e-6 |
| W2 | 2048; 2048,8192; 8192,2048 | 2013.989 | 0.346x | 0.226x | Yes | 1.132e-6 |
| W2 | 4096; 2048,8192; 8192,2048 | 3898.226 | 0.361x | 0.236x | Yes | 1.073e-6 |
| W2.5 | 512; 2048,8192; 8192,2048 | 525.933 | 0.314x | 0.225x | Yes | 9.537e-7 |
| W2.5 | 1024; 2048,8192; 8192,2048 | 1040.114 | 0.322x | 0.215x | Yes | 1.132e-6 |
| W2.5 | 2048; 2048,8192; 8192,2048 | 2045.082 | 0.341x | 0.222x | Yes | 9.537e-7 |
| W2.5 | 4096; 2048,8192; 8192,2048 | 3964.218 | 0.355x | 0.232x | Yes | 1.073e-6 |
| W3 | 512; 2048,8192; 8192,2048 | 530.256 | 0.311x | 0.223x | Yes | 9.537e-7 |
| W3 | 1024; 2048,8192; 8192,2048 | 1049.795 | 0.319x | 0.213x | Yes | 9.537e-7 |
| W3 | 2048; 2048,8192; 8192,2048 | 2046.111 | 0.341x | 0.222x | Yes | 1.013e-6 |
| W3 | 4096; 2048,8192; 8192,2048 | 3990.118 | 0.352x | 0.231x | Yes | 1.013e-6 |
| W3.5 | 512; 2048,8192; 8192,2048 | 564.892 | 0.292x | 0.209x | Yes | 8.941e-7 |
| W3.5 | 1024; 2048,8192; 8192,2048 | 1096.604 | 0.305x | 0.204x | Yes | 1.013e-6 |
| W3.5 | 2048; 2048,8192; 8192,2048 | 2165.379 | 0.322x | 0.210x | Yes | 1.073e-6 |
| W3.5 | 4096; 2048,8192; 8192,2048 | 4290.026 | 0.328x | 0.215x | Yes | 1.073e-6 |

The extended matrix improves ordinary QVQ by `1.076x` geometric mean, but is
only `0.327x` Marlin and `0.220x` Machete.  This near-linear scaling identifies
repeated P32 decode per M64 slab as the main redesign target.

## Rejected reuse-8 experiment

An exact M128 CTA stored eight input tiles and eight accumulator fragments
while decoding each P32 fragment once.  It compiled, replayed in a CUDA Graph,
and was bit-exact, but its shared-memory/register footprint lost to reuse-4:

| Rate | M x K x N (down) | Reuse-4 best us | Reuse-8 best us | Better than last |
| ---: | ---: | ---: | ---: | :---: |
| W3 | 128 x 8192 x 2048 | 31.229 | 30.918 | Yes, 1.010x |
| W3 | 256 x 8192 x 2048 | 53.408 | 53.603 | No |
| W3 | 512 x 8192 x 2048 | about 66 | 92.416 | No |

Reuse-8 is not present in production.  The next design must share decoded
weights without retaining eight independent FP32 accumulator fragments in one
CTA—for example, a persistent/shared decoded-weight tile across smaller row
consumer groups.
