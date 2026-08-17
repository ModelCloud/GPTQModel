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

### Real-Llama validation of FP16 prepared tables

The same one-layer Llama 3.2 1B W2 B2-P32+YAQA contract above was rerun after merging remote schedule reuse and
enabling exact-FP16 prepared tables. The dataset slices, seed, full-row batch-1 execution, factors, candidate set,
and evaluation contract were unchanged.

| Region | Previous | Current | Speedup |
|---|---:|---:|---:|
| Four Q/K/V/O module quantization | 99.810 s | 89.685 s | 1.113x |
| Segmented MLX Viterbi | 65.224 s | 56.691 s | 1.151x |
| Three B2 family candidates | 77.282 s | 67.344 s | 1.148x |
| YAQA feedback | 14.083 s | 12.611 s | 1.117x |
| Complete quantize-and-evaluate arm | 146.094 s | 135.494 s | 1.078x |

Final-logit KL remained `0.003401`; Top-1, Top-5, and Top-10 agreement remained `99.71%`, `92.06%`, and `91.90%`.
The result confirms that the table-traffic optimization survives realistic execution, but also confirms that
micro-kernel tuning alone cannot reach the 2--4x end-to-end target. The three independent family histories now
account for 67.344 seconds and remain the dominant exact-search target.

## Implicit PGC search decoder

Expanded banks are unnecessary for the frozen PGC graph portfolio. Each scalar pair is exactly

```text
state XOR rate-keyed bank mask
  -> PGC16 xor/multiply/xor bijection
  -> two indices into the frozen 256-entry FP16 compander
```

The Apple search kernel now performs that decode directly. It replaces two 65,536-vector bank tables and their
FP32 norm tables with one 256-entry FP16 level table plus two 16-bit masks. Candidate costs remain FP32. Before
enabling the path, preparation compares every supplied bank against the frozen portfolio and requires one unique
mask identity; learned or modified banks automatically use the expanded FP16/FP32 fallback.

At W2 B2-P32 and 32 tiles per launch, the implicit kernel measured 9.135 ms versus 11.507 ms for expanded FP16,
an additional 1.260x speedup. Relative to the original 13.831 ms FP32-table path, the combined table changes are
about 1.51x. B2-P32 and B4-P64 tests at every W1--W3.5 half-step produced bit-exact states, selectors, and reported
losses between implicit and expanded search. The complete MLX suite remains `310/310` passing.

This is a quantization-only optimization. It does not alter checkpoint layout, bank identity, inference decode, or
the separately permitted `2e-3` inference tolerance. The quantized artifact is bit-exact, satisfying the stricter
quantization contract rather than relying on its `1e-6` numerical allowance.

### Real-Llama validation of implicit PGC search

The matched one-layer W2 B2-P32+YAQA run produced the same weights and quality metrics while shifting the profile:

| Region | Expanded FP16 | Implicit PGC | Speedup |
|---|---:|---:|---:|
| Four Q/K/V/O module quantization | 89.685 s | 78.825 s | 1.138x |
| Segmented MLX Viterbi | 56.691 s | 45.383 s | 1.249x |
| Three B2 family candidates | 67.344 s | 56.302 s | 1.196x |
| Complete quantize-and-evaluate arm | 135.494 s | 124.535 s | 1.088x |

Against the 112.98-second matched module baseline before the accepted Apple recurrence changes, current module
quantization is 1.43x faster. Final-logit KL remained `0.003401`; Top-1/5/10 remained `99.71%`, `92.06%`, and
`91.90%`. CPU feedback is now 12.992 seconds and canonical V2 YAQA is 22.420 seconds, while the three family
candidates still consume 56.302 seconds. The exact 2--4x target therefore remains open, but expanded bank-table
bandwidth is no longer the dominant family-search cost.

## Canonical V2 host-to-MLX tail search

The independent V2+YAQA oracle previously moved every CPU-corrected anti-diagonal into Torch MPS and invoked the
older tail path. Canonical V2 is exactly the one-bank special case of the segmented recurrence. Apple YAQA now
keeps those tiles on the host, uses the same guarded staging arena, runs the fused provisional/constrained implicit
PGC kernel with one bank, and returns only traceback results. A synthetic segment boundary at the half-tile point
does not alter the recurrence when only one bank exists.

At W2 and 32 tiles, the old Torch-MPS path measured 7.213 ms and the fused host-MLX path measured 4.300 ms, a
1.677x speedup. States were exact. The reported accumulated loss differed by at most `9.54e-6` between those two
backend reduction orders, while the independent CPU-oracle tests for all W1--W3.5 half-steps pass the required
`1e-6` loss tolerance and exact state gate. Since serialization depends on the exact states, the quantized weights
are unchanged.

The dispatch is restricted to Apple host feedback, L16/V2 codebooks, and `tail_biting_candidates=1`. Dual-V2,
V4, widened tail candidates, and unsupported codebooks retain their existing paths. The complete MLX gate passes
`316/316` tests.

### Real-Llama validation of canonical fusion

The same real-Llama contract retained identical final metrics and produced the following profile change:

| Region | Before canonical fusion | After | Speedup |
|---|---:|---:|---:|
| Canonical V2 YAQA | 22.420 s | 11.020 s | 2.034x |
| Canonical V2 tail recurrence | 18.807 s | 7.351 s | 2.558x |
| Four Q/K/V/O module quantization | 78.825 s | 66.936 s | 1.178x |
| Complete quantize-and-evaluate arm | 124.535 s | 112.748 s | 1.105x |

Relative to the earlier matched 112.98-second module baseline, module quantization is now 1.69x faster. Relative
to the 146.094-second complete-arm measurement, the current arm is 1.30x faster. Final-logit KL remains
`0.003401`, with Top-1/5/10 at `99.71%`, `92.06%`, and `91.90%`.

The profile is now concentrated: the three B2 family candidates consume 55.824 seconds, including 45.373 seconds
of segmented Metal recurrence, while all canonical V2 work consumes 11.020 seconds and CPU feedback across all
four passes consumes 12.592 seconds. A 2x module target requires the family region to fall from 55.824 seconds to
roughly 22 seconds or less; canonical and feedback-only tuning cannot provide that bound by themselves.

## Reused left-side YAQA feedback projection

Each corrected tile contains both `L E R` and `L E_tile`. The former already materializes `L E` across the full
live output suffix, so its leading tile columns are bit-identical to a second skinny `L E_tile` GEMM. YAQA now
reuses that slice. This removes one Accelerate GEMM per tile without changing the addition order, factors, error
history, corrected target, trellis search, or checkpoint format.

Direct FP32 checks over suffix widths through 2048 found bit-exact corrected targets (`max_abs = 0`). The focused
real-artifact Apple parity test also retained exact states and selectors, satisfying the `1e-6` quantization gate.
The matched real-Llama W2 B2-P32+YAQA contract measured:

| Region | Before reuse | After | Speedup |
|---|---:|---:|---:|
| YAQA feedback | 12.592 s | 10.273 s | 1.226x |
| Four Q/K/V/O module quantization | 66.936 s | 64.668 s | 1.035x |
| Complete quantize-and-evaluate arm | 112.748 s | 110.458 s | 1.021x |

Final-logit KL remained `0.003401`; Top-1/5/10 remained `99.71%`, `92.06%`, and `91.90%`. The result is a free
exact win, but it also reinforces the profile conclusion: segmented Metal recurrence remains 45.418 seconds and
requires cross-family batching or a deeper exact-kernel change to reach the requested 2--4x module speedup.

## 512-thread implicit recurrence

The implicit PGC recurrence assigns independent states and suffixes to Metal threads. Increasing its threadgroup
from 256 to 512 halves each thread's state loop while preserving every predecessor scan, FP32 operation, strict
comparison, tie rule, and traceback. The expanded/custom-table fallback remains at 256 threads; the new default
applies only to the verified implicit PGC portfolio.

W1--W3.5 tests compare the 256- and 512-thread outputs directly and require bit-exact states, selectors, and FP32
losses. Separate CPU-oracle tests also pass at every supported half-step, which is stronger than the `1e-6`
quantization tolerance. The W2 kernel microbench improved by 1.31--1.38x for anti-diagonal batches below 32 and by
1.41x at batch 128. W2.5 and W3.5 batch-32 recurrences improved by approximately 1.84x.

The matched real-Llama W2 B2-P32+YAQA run retained every reported quality metric and measured:

| Region | 256 threads | 512 threads | Speedup |
|---|---:|---:|---:|
| Segmented Metal recurrence | 45.418 s | 39.088 s | 1.162x |
| Three B2 family candidates | 54.232 s | 47.615 s | 1.139x |
| Canonical V2 YAQA | 10.333 s | 8.636 s | 1.197x |
| Four Q/K/V/O module quantization | 64.668 s | 56.762 s | 1.139x |
| Complete quantize-and-evaluate arm | 110.458 s | 102.293 s | 1.080x |

Final-logit KL remained `0.003401`; Top-1/5/10 remained `99.71%`, `92.06%`, and `91.90%`. Cumulatively, module
quantization is now 1.99x faster than the original matched 112.98-second Apple baseline, essentially reaching the
lower end of the requested 2--4x target. The full arm is 1.43x faster than the earlier 146.094-second measurement
because dataset capture and model evaluation are intentionally unchanged.

### Rejected scheduling variants

- Cross-family coalescing was bit-exact at W1--W3.5, but combining three family workspaces crossed the M4
  occupancy knee. At 32 and 64 tiles per family it ran at only `0.689x` and `0.654x` the speed of three serial
  family launches. It was removed rather than hidden behind a dispatch heuristic.
- A 1,024-thread recurrence was also bit-exact. It was 2--6% slower than 512 threads for batches 8--64, which
  dominate real W2 execution; its batch-128 win did not compensate under the bounded batch-32 policy.
- Forced full unrolling of bank, prefix, and edge loops preserved all reported quality metrics, but increased the
  matched real-Llama module time from 56.762 seconds to approximately 57.86 seconds. Metal's default optimizer is
  better balanced for this kernel, so the explicit unroll directives were removed.

## Sampled family strategies: another 2.20x module gain

The exact three-family YAQA ceiling spends three complete feedback/recurrence histories to select one module-level
alternative family. `YaqaConfig.sample_strategy` optionally evaluates all three families on 32, 64, 96, 128, or 256
evenly spaced real 16x16 weight tiles using the matching diagonal input/output Hessian blocks. The default remains
`full`:

```text
N real module tiles, N in {32, 64, 96, 128, 256}
  -> score family 1, 2, 3 with tr(E H_I,block E^T H_O,block)
  -> choose one family for the module
  -> run canonical V2+YAQA independently
  -> run one complete B2-P32+YAQA family candidate
  -> compare both under the original full-module Kronecker proxy
```

This is deliberately a different candidate-generation policy, not an approximation inside the selected YAQA pass.
The chosen complete family artifact is bit-exact to an independent fixed-family run; states, selectors, feedback,
and final proxy arithmetic are unchanged. The canonical V2+YAQA oracle remains independently encoded and is restored
on a tie, non-finite result, or proxy regression.

Matched contract: real Llama 3.2 1B Instruct, layer-0 Q/K/V/O, W2 B2-P32+YAQA, `neuralmagic/calibration` rows
`[0,64)` (27,455 valid tokens), disjoint evaluation rows `[64,128)` (20,384 tokens), further-disjoint YAQA rows
`[128,192)` (22,342 tokens), full rows, batch 1, MPS model shell, native MLX recurrence, and all 12 M4 Max P cores.

| Measurement | Full three-family reselect | 64-tile sampled proxy | Change |
|---|---:|---:|---:|
| Four-module quantization | 56.714 s | 25.778 s | **2.200x faster** |
| Complete arm (capture excluded, evaluation included) | 102.704 s | 71.348 s | 1.440x faster |
| Sampled family choice | - | 0.557 s | 0.54% of old arm |
| Complete family histories | 3 per module | 1 per module | 3x -> 1x |
| Final-logit KL | 0.003401 | 0.003251 | **-4.42%** |
| Top-1 agreement | 99.706% | 99.715% | +0.010 pp |
| Top-5 overlap | 92.058% | 92.011% | -0.047 pp |
| Top-10 overlap | 91.904% | 91.821% | -0.083 pp |
| Layer KL | 0.040268 | 0.039665 | -1.50% |

The follow-up matched sweep compared every explicit strategy using the same model, row splits, factors, seed, and
runtime contract:

| Strategy | Module quantization | Complete arm | Final KL | Layer KL | Top-1 | Top-5 | Top-10 | Family histogram |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 56.714 s | 102.704 s | 0.003401 | 0.040268 | 99.706% | 92.058% | 91.904% | all 3 complete candidates |
| `32_16x16` | 26.285 s | 72.287 s | 0.003336 | **0.038784** | 99.696% | 91.913% | 91.894% | 0 / 1 / 2 / 1 |
| `64_16x16` | 25.669 s | 70.948 s | 0.003251 | 0.039665 | **99.715%** | 92.011% | 91.821% | 0 / 1 / 2 / 1 |
| `96_16x16` | 26.063 s | 71.199 s | 0.003370 | 0.039941 | 99.676% | 91.985% | 91.810% | 0 / 2 / 1 / 1 |
| `128_16x16` | 27.647 s | 73.163 s | **0.003194** | 0.040867 | 99.701% | **92.016%** | **91.843%** | 0 / 2 / 0 / 2 |
| `256_16x16` | 28.393 s | 74.160 s | 0.003258 | 0.041306 | 99.696% | 91.953% | 91.807% | 0 / 1 / 2 / 1 |

The strategy is not monotonic in sample count. At this seed, 128 tiles gives the best final KL, 32 gives the best
layer KL, 64 gives the best Top-1, and 96 lies between 64 and 128 in cost but not in quality. The 256-tile screen
loses to both 64 and 128 despite selecting the same family-count histogram as 32 and 64. Histograms do not
identify which projection received each family, and equal family IDs would still not prove equal paths if the
selected modules differ. This is expected from a proposal screen: its diagonal-block proxy omits cross-tile terms,
while the accepted artifact is produced by complete sequential YAQA feedback. No sampled strategy is the default;
`full` remains the conservative ceiling reference until multi-layer, multi-rate, and multi-seed propagated evidence
supports a different policy.

Mathematically, `256_16x16` is the closest sampled estimator to `full` in coverage and sampling variance, while
`full` is the only algebraically exact search mode. If tile contributions were independent with finite variance,
the standard error of a sample mean would decrease approximately as `1/sqrt(N)`; deterministic evenly spaced tiles
and correlated Hessian geometry make that only a useful scaling heuristic here. Increasing `N` cannot remove the
structural bias from omitted off-diagonal Hessian blocks or from not running complete sequential YAQA feedback for
the discarded families. Because family selection is an argmin, even a smaller score-estimation error can cross a
near-tie and produce a discontinuously different module family, V2 path, selector schedule, and downstream error
direction. The observed best final KL at 128 rather than 256 is therefore possible without contradicting the
variance argument and must be confirmed across seeds and deeper propagated execution.

The small Top-5/10 movements are noise-scale guardrails, while final KL and layer KL improve. Local reconstruction
does not uniformly improve: mean relative weight L2 is 0.690088 versus 0.689651 for full reselection. That is
acceptable evidence for escalation because post-quant propagated recovery is the objective; it is not evidence for
making this mode the default yet. Confirm W1.5/W2/W2.5 across more layers and seeds before promotion.

Relative to the original matched 112.98-second Apple module baseline, sampled-family quantization is **4.38x
faster**. The next profile is now canonical V2+YAQA (8.80 s), the selected B2 family (16.02 s), and shared feedback
(5.28 s, overlapping those passes). Another exact 2x from this point requires sharing or concurrently scheduling the
canonical and selected-family histories; further tuning of the 0.56-second sampler cannot materially move the bound.
