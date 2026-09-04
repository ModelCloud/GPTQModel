# Phase 80: fixed H100 Qwen linear-input segment mapper

Phase 80 replaces runtime segment search and runtime-width division in the
Qwen3.8-27B grouped linear-attention input kernel with compile-time arithmetic
for its exact two-child plan:

$$
(5120\rightarrow10240,\;5120\rightarrow6144).
$$

All 15 W2--W3 by M1--M16 cells improve, with unchanged dense-P32 error and
bit-exact CUDA Graph replay.

## Work mapping

The two children contain 160 and 96 N64 blocks.  Their measured ordered split
plans are:

| Rate | first child | second child | useful work items |
|:--|--:|--:|--:|
| W2/W2.5 | 10 | 20 | 3520 |
| W3 | 4 | 20 | 2560 |

The generic grouped kernel scans runtime segment boundaries, loads runtime
N-tile widths, and divides by a runtime N64 count for every thread.  The
specialization knows the first child's work boundary at compile time, chooses
one of two children with a single comparison, and divides/remainders by the
compile-time constants 160 or 96.  It computes the same global trellis block,
child-local split, output offset, and ordered-partial offset.

P32 state, selector, pseudo-random-code, level lookup, WGMMA issue, reduction,
and recovery math do not change.  The host gate requires transition width
four through six, physical `NVIDIA H100`, K=5120, child widths `(10240,6144)`,
and the exact measured per-rate splits.  All other calls retain the generic
mapper.  Telemetry exposes `h100_qwen_fixed_linear_grid_launches`.

## H100 benchmark

The physical device UUID was
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`.  Timings use 20 warmups, 60
samples, and 50 CUDA Graph replays per sample, measured with CUDA events.
Marlin and Machete are figurative W4 projection-sum baselines.  `last/new`
compares with the fresh Phase-79 same-source baseline.

| W | M | MKN: linear QKV; linear Z | new us | effective TFLOP/s | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x10240; 1x5120x6144 | 155.775 | 1.077 | 0.1496x | 0.3198x | 1.0085x | yes |
| 2 | 2 | 2x5120x10240; 2x5120x6144 | 162.263 | 2.068 | 0.1487x | 0.3033x | 1.0094x | yes |
| 2 | 4 | 4x5120x10240; 4x5120x6144 | 168.352 | 3.986 | 0.1444x | 0.2918x | 1.0090x | yes |
| 2 | 8 | 8x5120x10240; 8x5120x6144 | 181.542 | 7.393 | 0.1278x | 0.2703x | 1.0096x | yes |
| 2 | 16 | 16x5120x10240; 16x5120x6144 | 193.013 | 13.908 | 0.1398x | 0.2559x | 1.0073x | yes |
| 2.5 | 1 | 1x5120x10240; 1x5120x6144 | 155.883 | 1.076 | 0.1495x | 0.3196x | 1.0009x | yes |
| 2.5 | 2 | 2x5120x10240; 2x5120x6144 | 162.896 | 2.060 | 0.1481x | 0.3021x | 1.0056x | yes |
| 2.5 | 4 | 4x5120x10240; 4x5120x6144 | 169.271 | 3.965 | 0.1436x | 0.2902x | 1.0066x | yes |
| 2.5 | 8 | 8x5120x10240; 8x5120x6144 | 182.208 | 7.366 | 0.1273x | 0.2693x | 1.0063x | yes |
| 2.5 | 16 | 16x5120x10240; 16x5120x6144 | 193.602 | 13.865 | 0.1393x | 0.2552x | 1.0074x | yes |
| 3 | 1 | 1x5120x10240; 1x5120x6144 | 151.828 | 1.105 | 0.1535x | 0.3281x | 1.0079x | yes |
| 3 | 2 | 2x5120x10240; 2x5120x6144 | 159.365 | 2.106 | 0.1514x | 0.3088x | 1.0079x | yes |
| 3 | 4 | 4x5120x10240; 4x5120x6144 | 165.429 | 4.057 | 0.1469x | 0.2970x | 1.0052x | yes |
| 3 | 8 | 8x5120x10240; 8x5120x6144 | 178.094 | 7.536 | 0.1303x | 0.2755x | 1.0069x | yes |
| 3 | 16 | 16x5120x10240; 16x5120x6144 | 189.532 | 14.163 | 0.1423x | 0.2606x | 1.0056x | yes |

Geometric improvement is **1.00693x**, with 15 of 15 wins.  Per-rate gains
are 1.00877x for W2, 1.00534x for W2.5, and 1.00670x for W3.  Mean absolute
error is at most `2.480e-6`; maximum absolute error is `1.774e-5` against the
same-payload dense-P32 Torch oracle.

The all-rate micro-test uses calibrated output scales, verifies each measured
split plan, exact repeated graph replay, bounded plain-child parity, and the
fixed-grid telemetry counter.  The distilled benchmark is
`artifacts/a41_phase80_h100/qwen38_27b_fixed_linear_input_grid.json`.
