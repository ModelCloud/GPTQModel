# Pangolin Apple 2--8-bit final performance snapshot

Measured on an Apple M4 Max with PyTorch `2.14.0.dev20260806`, FP16 activations
and scales, int32 packed GPTQ-P weights, group size 128, and the default
non-caching `TorchLinear` from `origin/main` (`e0aab819`) as the baseline. The
baseline implementation is unchanged on this branch. Each sample is the median
of synchronized batches after warmup; both implementations use the complete
`QuantLinear.forward` path. Lower latency is better.

The fused Pangolin path does not materialize or retain a dense decoded weight.
A 1,000-call M=32 stress run had a zero-byte change in
`torch.mps.current_allocated_memory()` after synchronization and output release.

## K=N=1024

### Speedup versus `origin/main`

| bits | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M16 | M32 | median |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5.27x | 4.18x | 3.27x | 3.26x | 3.28x | 3.27x | 3.23x | 3.25x | 1.81x | 1.62x | 3.27x |
| 3 | 10.49x | 8.50x | 6.96x | 5.95x | 6.10x | 6.05x | 5.88x | 5.98x | 3.26x | 2.92x | 6.02x |
| 4 | 6.16x | 6.10x | 5.59x | 3.22x | 3.25x | 3.23x | 3.28x | 3.17x | 1.81x | 1.58x | 3.24x |
| 5 | 10.45x | 8.80x | 6.95x | 6.16x | 6.05x | 6.02x | 5.87x | 5.85x | 3.30x | 2.82x | 6.03x |
| 6 | 10.59x | 9.03x | 7.27x | 6.10x | 6.25x | 6.14x | 6.15x | 5.95x | 3.29x | 2.89x | 6.14x |
| 7 | 12.49x | 10.21x | 8.40x | 6.70x | 6.64x | 6.85x | 6.92x | 6.75x | 3.70x | 3.16x | 6.80x |
| 8 | 6.29x | 6.46x | 5.65x | 3.20x | 3.15x | 3.15x | 3.15x | 3.14x | 1.77x | 1.53x | 3.15x |

### Absolute latency (main / PR, milliseconds)

| bits | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M16 | M32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | .280/.053 | .295/.070 | .296/.091 | .299/.092 | .296/.090 | .295/.090 | .293/.091 | .295/.091 | .305/.168 | .286/.176 |
| 3 | .684/.065 | .691/.081 | .701/.101 | .678/.114 | .700/.115 | .702/.116 | .688/.117 | .701/.117 | .706/.216 | .653/.224 |
| 4 | .300/.049 | .295/.048 | .295/.053 | .292/.091 | .295/.091 | .298/.093 | .301/.092 | .299/.094 | .310/.171 | .288/.182 |
| 5 | .685/.066 | .705/.080 | .698/.101 | .701/.114 | .695/.115 | .703/.117 | .692/.118 | .693/.118 | .716/.217 | .643/.228 |
| 6 | .676/.064 | .722/.080 | .718/.099 | .716/.117 | .715/.114 | .715/.117 | .716/.116 | .714/.120 | .726/.221 | .677/.234 |
| 7 | .882/.071 | .905/.089 | .900/.107 | .880/.131 | .884/.133 | .898/.131 | .904/.131 | .898/.133 | .915/.248 | .840/.266 |
| 8 | .308/.049 | .300/.046 | .302/.053 | .299/.093 | .303/.096 | .301/.096 | .304/.097 | .300/.096 | .312/.176 | .295/.193 |

## K=N=4096

### Speedup versus `origin/main`

| bits | M1 | M4 | M8 | M16 | M32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 21.82x | 3.93x | 3.78x | 1.95x | 1.37x |
| 3 | 22.92x | 7.59x | 7.26x | 3.63x | 2.82x |
| 4 | 22.53x | 3.79x | 3.66x | 1.87x | 1.35x |
| 5 | 22.86x | 7.30x | 7.08x | 3.50x | 2.86x |
| 6 | 23.40x | 7.46x | 7.15x | 3.56x | 2.87x |
| 7 | 27.48x | 8.62x | 8.33x | 4.16x | 3.34x |
| 8 | 23.39x | 3.69x | 3.58x | 1.81x | 1.37x |

### Absolute latency (main / PR, milliseconds)

| bits | M1 | M4 | M8 | M16 | M32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 3.414/.157 | 3.422/.871 | 3.449/.912 | 3.471/1.778 | 3.516/2.562 |
| 3 | 9.426/.411 | 9.436/1.244 | 9.452/1.301 | 9.426/2.598 | 9.452/3.349 |
| 4 | 3.432/.152 | 3.466/.915 | 3.490/.955 | 3.518/1.883 | 3.526/2.606 |
| 5 | 9.421/.412 | 9.503/1.303 | 9.523/1.346 | 9.474/2.710 | 9.493/3.319 |
| 6 | 9.691/.414 | 9.769/1.309 | 9.812/1.372 | 9.738/2.738 | 9.810/3.424 |
| 7 | 12.687/.462 | 12.699/1.473 | 12.716/1.527 | 12.703/3.056 | 12.616/3.779 |
| 8 | 3.554/.152 | 3.586/.973 | 3.586/1.003 | 3.625/2.000 | 3.637/2.660 |

## Accuracy and memory context

All 2--8-bit results were checked against direct FP32 dequantize-and-matmul
references and passed `rtol=1e-3, atol=1e-3`, including K=4096 and M=32. The
largest absolute differences occur at large-magnitude FP16 outputs and are one
FP16 rounding step; no comparison exceeded the combined tolerance.

An opt-in, predecoded FP16 weight cache makes later calls a dense matmul and is
faster than an on-the-fly packed kernel, especially as M grows. It also retains
an additional dense matrix equal to `16 / bits` times the qweight storage (8x
at 2-bit, 4x at 4-bit, and 2x at 8-bit). That is not the default main baseline
above and is intentionally not used by Pangolin.

An output-major/int64 repack was also benchmarked. Although a duplicate layout
helped isolated 4-bit decode in some shapes, it regressed 2-bit M1 by about 2.9x
and would retain a second packed-weight allocation. It was rejected.
