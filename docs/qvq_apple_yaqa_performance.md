# QVQ B2-P32 + YAQA performance on Apple silicon

This file records accepted Apple quantization optimizations, measured results, and remaining performance work. It
is intentionally separate from quality-promotion evidence: synthetic tensors may validate kernel algebra and
throughput, but codec/quality decisions require real model weights and real calibration/evaluation rows.

## Numerical contracts

- Quantization/search math: discrete states and selectors must match exactly. Floating results used to validate a
  quantization optimization must agree within `1e-6` unless a stricter exact comparison is available.
- Inference reconstruction/output: absolute error up to `2e-3` is acceptable, subject to the existing relative-error
  and model-level gates.
- A performance change that violates the applicable accuracy contract is rejected regardless of speed.

## 2026-08-17 accepted MLX tail-search changes

Commit `768678e2` made four exact changes to the Apple B2-P32/B4-P64 YAQA path:

1. Validate and stage each immutable bank stack once, then reuse its FP32 state norms.
2. Fuse provisional and constrained tail-biting recurrences into one Metal command. The provisional half does not
   materialize unused first-half traceback, and rotation is indexed directly in the kernel.
3. Reuse equal-shaped MLX recurrence allocations while bounding cached memory to 512 MiB.
4. Select the measured Metal tile batch by rate. W1--W2 and W3 use 32 tiles on M4 Max; W2.5 and W3.5 retain the
   128-tile window. Explicitly larger caller batches remain respected.

All-zero, weighted, and banked recurrence semantics are unchanged. The fused path matched the independent Torch
oracle for B2-P32 and B4-P64 at W1, W1.5, W2, W2.5, W3, and W3.5. States and selectors were exact; losses met the
existing oracle tolerance. The complete MLX gate passed `309/309` tests after rebasing onto remote tip.

### Kernel microbenchmarks

Host: Apple M4 Max. Tests used all 12 performance cores through `taskpolicy -t 5 -l 5` and capped native thread
pools at 12. Shape: B2-P32, W2, 128 transitions per tile, FP32 search.

| Measurement | Before | After | Speedup |
|---|---:|---:|---:|
| 128 tiles, previous single batch | 97.42 ms | - | - |
| 128 tiles as four 32-tile batches, bounded cache | - | 59.88 ms | 1.63x |
| Fused versus two-command tail search at batch 32 | 15.24 ms | 14.38 ms | 1.06x |
| Combined representative kernel change | 97.42 ms | about 57.5--59.9 ms | about 1.63--1.69x |

Prepared codebook norms alone improved a representative batch by about 2.5%. They are retained because they are
exact and remove repeated immutable work, but they are not treated as a major speed result.

The bounded cache stabilized at about 128 MiB for repeated 32-tile W2 calls; 50 consecutive calls did not grow the
cache. The measured peak when intentionally mixing a 128-tile call was about 512 MiB.

## Real Llama 3.2 1B timing gate

Contract:

- Model: real Llama 3.2 1B Instruct weights.
- Scope: layer 0 Q/K/V/O.
- Rate/codec: W2 B2-P32 + YAQA family reselection.
- Calibration: `neuralmagic/calibration` rows `[0,64)`, 27,455 valid tokens.
- Evaluation: disjoint rows `[64,128)`, 20,384 valid tokens.
- YAQA Sketch-B: further-disjoint rows `[128,192)`, 22,342 valid tokens.
- Full row lengths, batch 1 for calibration/evaluation, no concatenation.
- Device: MPS model shell, CPU Accelerate feedback, native MLX segmented recurrence.
- All 12 M4 Max performance cores enabled.

Artifact: `/tmp/qvq_b2_yaqa_mlx_accel_w2_layer1.json` (ephemeral local timing artifact).

| Projection | Previous matched layer-0 run | Optimized run | Speedup |
|---|---:|---:|---:|
| q_proj, 2048 x 2048 | 49.73 s | 39.95 s | 1.24x |
| k_proj, 512 x 2048 | 9.70 s | 9.41 s | 1.03x |
| v_proj, 512 x 2048 | 9.83 s | 9.40 s | 1.05x |
| o_proj, 2048 x 2048 | 43.72 s | 41.39 s | 1.06x |
| Four-projection sum | 112.98 s | 100.15 s | 1.13x |

The optimized run spent 99.81 seconds in `baseline_encode` out of 146.09 seconds total. Quantization therefore used
about 68.3% of the one-layer arm; the remaining 46.28 seconds covered replay, evaluation, and diagnostics. In the
previous four-layer gate, module quantization consumed 506.78 of 579.77 seconds (87.4%), so optimizing evaluation
first would not materially improve full-model wall time.

Nested optimized telemetry:

| Region | Time |
|---|---:|
| B2 family candidates | 77.28 s |
| Segmented MLX Viterbi | 65.22 s |
| Canonical V2 YAQA | 22.41 s |
| CPU YAQA feedback | 14.08 s |

These regions overlap and must not be summed.

The real-model output remained coherent: final-logit KL `0.003401`, Top-1 agreement `99.71%`, Top-5 overlap
`92.06%`, and Top-10 overlap `91.90%` for the one-layer diagnostic. These values validate the run, not a codec
quality promotion against a different layer count.

## Current conclusion and next bottleneck

The accepted changes produce a substantial kernel improvement but only a 1.13x matched layer-0 quantization gain;
the requested 2--4x end-to-end quantization target is not yet met. The next target is algorithmic scheduling of the
three complete B2 alternative-family YAQA passes. They account for 77.28 seconds and repeat the same immutable
factors and source geometry. A valid optimization must preserve all three family candidates and the independent V2
oracle; silently switching to fixed-family selection changes the search space and is not an equivalent speedup.

Promising exact directions, in priority order:

1. Batch the three family feedback states and issue family-aware segmented recurrences without changing reduction
   order or per-family winner semantics.
2. Pipeline CPU feedback for one family/anti-diagonal with Metal recurrence for another while retaining serialized
   MLX command ownership.
3. Reduce provisional recurrence memory traffic further, provided exact states/selectors and `1e-6` loss parity are
   maintained.
4. Offer fixed-family YAQA only as an explicitly different fast-quality mode after a matched real-model quality
   comparison; do not report it as an exact acceleration of family reselection.

## Exact-FP16 prepared bank tables

The canonical PGC16 V2 bank tables are generated from frozen FP16 bit patterns. An exhaustive W1--W3.5 check
confirmed that every FP32 table value round-trips through FP16 exactly. The Apple tail-biting kernel now stores
those prepared tables as FP16 and converts each loaded scalar to FP32 before the unchanged emission arithmetic.
This halves bank-table traffic without changing the representable codebook, state winner, or selector winner.
Tables containing any value that is not exactly FP16-representable remain on the FP32 kernel path.

For W2 B2-P32 with 32 tiles per launch on the M4 Max, the same prepared PGC tables measured 13.831 ms in FP32 and
11.451 ms in FP16, a 1.208x kernel speedup. States, selectors, and the MLX loss result were bit-exact between the two
storage paths. Against the independent Torch oracle, states and selectors were exact for every half-step from W1
through W3.5; the diagnostic accumulated loss differed by at most `7.63e-6` because the existing Metal and Torch
reduction orders differ. The serialized quantization result is determined by the exact state and selector winners,
so the quantized weights remain exact rather than merely within tolerance.

Validation on the M4 Max: all 310 MLX tests pass, including an explicit guard that canonical PGC banks take the
FP16 path while arbitrary non-FP16 banks fall back to FP32. Inference accuracy remains governed by the separate
`2e-3` output tolerance; this change affects quantization-time table storage only.
