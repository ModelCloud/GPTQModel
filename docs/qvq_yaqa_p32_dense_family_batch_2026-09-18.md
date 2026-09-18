# QVQ + YAQA P32 dense family batching

This quantization-stage optimization batches canonical V2 and the two
diversity-selected B2 families for attention-sized CUDA projections. It does
not alter post-quant inference, the candidate set, the YAQA recurrence, or the
authoritative full Kronecker-Fisher selection objective.

## Method

Attention projections use YAQA's dense transformed-error recurrence. Before
this change, canonical V2 and both complementary families traversed the same
anti-diagonal schedule independently. The optimized path:

1. creates one family dimension containing canonical V2 and the selected
   complementary families;
2. submits every anti-diagonal to the existing exact family-grid Viterbi
   operator;
3. preserves the original per-family FP32 `addmm` feedback-update order;
4. scores the completed artifacts with the unchanged full objective; and
5. maps the winning candidate index back to its actual sampled family ID.

Canonical V2 is represented by two identical canonical banks. Existing tie
precedence therefore keeps bank zero and emits the same all-zero selector
payload as the single-bank path.

Wide MLP projections retain the existing factored-feedback implementation.
Their canonical history already overlaps the two-family candidate batch on a
separate CUDA stream, so combining all three histories would increase the
critical-path family-grid work.

## Matched H100 results

Layer 0 of Llama 3.2 1B, 32 independent YAQA sequences, 11,992 valid tokens,
batch 8, exact pruning policy, 256-tile diversity sampling, and the opt-in
FP64 oracle in both arms:

| Rate | Previous layer | Dense batch layer | Layer speedup | Previous attention | Dense batch attention | Attention speedup |
|---|---:|---:|---:|---:|---:|---:|
| W2.5 | 26.613 s | 25.110 s | 1.060x | 6.627 s | 4.912 s | 1.349x |
| W3 | 32.268 s | 29.413 s | 1.097x | 8.357 s | 5.635 s | 1.483x |
| W3.5 | 29.623 s | 27.453 s | 1.079x | 7.792 s | 5.618 s | 1.387x |

All five serialized P32 tensors for all seven projections are bit-for-bit
identical at every rate. Per-module FP32 and FP64 Kronecker-Fisher objectives
are also exactly identical. The previously validated disjoint GSM8K Platinum
held-out results therefore carry over without a statistical recheck.

## Next bottleneck

The three wide MLP projections now account for most layer quantization time.
Their dominant phase is the exact two-family segmented Viterbi recurrence.
The current SM90 implementation launches one grid-parallel kernel per P32
segment plus traceback for each anti-diagonal and tail-biting pass.

## W3 midpoint-only tail biting

The follow-up profiled and tested the proposed persistent replacement before
shipping it. On H100 the existing grid kernel is already compute/L2 bound
(`78.14%` SM throughput, `89.62%` active warps, and negligible DRAM traffic).
The persistent prototype was 1.23--2.90x slower over production family-batch
sizes and was rejected.

The safe remaining redundancy was the provisional circular pass. Its caller
consumes only state 63, but the former family-grid API materialized 128 states,
one loss, and eight selectors per sequence. The W3 path now invokes an exact
midpoint-only family operator and skips the unused traceback products. Dispatch
is intentionally restricted to W3: CUDA-event A/B measurements showed
1.154--1.173x at family batches 1--32 and 1.015--1.022x at batches 64--128,
whereas W2.5 regressed and W3.5 was neutral.

On the same matched Llama layer-0 protocol, total segmented-Viterbi GPU time
fell from `17,898.308 ms` to `17,597.927 ms` (1.017x), and quantization process
time fell from `29.302 s` to `29.162 s` (1.005x). This is an incremental exact
win rather than a new 2x phase. All 174 saved tensors, including all 21
quantized payload tensors, match bit-for-bit. Every per-module FP32 and FP64
Kronecker-Fisher objective also matches exactly, so no held-out rerun is needed.

Telemetry reports midpoint-only production directly: one provisional state is
produced and consumed per family sequence, with no discarded provisional loss
or selector payload. W2.5 and W3.5 retain the previous full provisional solve.

## Wide-MLP grouped feedback and opt-in TF32

Wide projections form YAQA's corrected tiles with grouped FP32 GEMMs. The
exact path now groups the two equal-geometry family matrices together for each
tile instead of submitting two singleton cuBLAS groups. This changes only the
submission layout and preserves the FP32 result and payload exactly. On the
production `2048x8192`, two-family, 128-tile feedback case it improves the
operator from `1.331 ms` to `1.258 ms` (1.058x).

H100 can optionally execute those factored-feedback GEMMs in TF32 tensor-core
mode while retaining FP32 cache/output storage:

```bash
GPTQMODEL_QVQ_YAQA_FAST_TF32=1 python scripts/qvq_quantize.py ...
```

The switch applies only to the wide CUDA factored-feedback path and is off by
default because it deliberately changes rounding and may change the selected
P32 payload. The representative corrected-tile operator measured `1.266 ms`
in exact FP32 and `0.309 ms` in TF32 (4.10x). Use
`scripts/benchmark_qvq_yaqa_feedback.py` to reproduce the operator gate.

Matched layer-0 Llama 3.2 1B results use the same 32-sequence/11,992-token
Sketch-B factors and the same strict disjointness manifest as above:

| Rate | Exact quantization | TF32 quantization | Speedup | FP64 Fisher delta | Disjoint held-out MSE delta |
|---|---:|---:|---:|---:|---:|
| W2.5 | 25.001 s | 23.516 s | 1.063x | +0.03001% | -0.04381% |
| W3 | 29.162 s | 27.745 s | 1.051x | -0.00392% | +0.02369% |
| W3.5 | 27.344 s | 26.069 s | 1.049x | -0.05206% | +0.01186% |

Negative deltas improve the metric. FP32 and FP64 Fisher deltas agree to the
reported precision. The held-out gate contains 256 GSM8K Platinum questions
and 352,926,720 projection elements; it is disjoint from ordinary calibration
rows `[0,32)` and YAQA rows `[64,96)`. W2.5 is a speed/held-out-quality double
win despite a small Fisher-proxy regression. W3 and W3.5 improve aggregate
Fisher while accepting bounded held-out changes of 0.024% and 0.012%.

TF32 changed only the three MLP payloads; all four attention payloads remained
bit-identical. Running only the corrected-tile GEMM in TF32 while retaining
FP32 cache updates was tested as a mitigation, but worsened W3 held-out drift
from `+0.02369%` to `+0.05951%` with no speed benefit. The paired TF32 mode is
therefore the validated option. Telemetry records
`yaqa_fast_tf32_feedback_calls` when it is active.

## Provisional direct-distance Viterbi

The family-grid recurrence normally expands squared distance as
`||t||^2 + ||c||^2 - 2 t.c`. An opt-in CUDA path, validated on SM90, evaluates the same metric as
`(t0-c0)^2 + (t1-c1)^2` in FP32. This removes the packed FP32 codebook-norm
load and shortens the hot instruction sequence, but changes FP32 operation
ordering and can resolve near-ties differently. The safe validated mode uses
it only for the provisional tail-biting pass; the constrained final solve,
path costs, tie precedence, and traceback remain on the exact formulation:

```bash
GPTQMODEL_QVQ_YAQA_FAST_VITERBI_DISTANCE=provisional python scripts/qvq_quantize.py ...
```

On H100, the direct recurrence improves a two-family, 128-tile call by 1.22x
at W2.5, 1.12x at W3, and 1.17x at W3.5. Provisional-only matched Llama 3.2
1B layer-0 results are:

| Rate | Exact layer | Provisional-direct layer | Speedup | FP64 Fisher range | Disjoint held-out MSE delta |
|---|---:|---:|---:|---:|---:|
| W2.5 | 25.001 s | 23.497 s | 1.064x | -1.084% to 0.000% | +0.08494% |
| W3 | 29.257 s | 27.522 s | 1.063x | -0.140% to +0.083% | +0.02034% |
| W3.5 | 27.344 s | 25.553 s | 1.070x | -0.413% to +0.073% | -0.03620% |

Negative deltas improve quality. FP32 and FP64 Fisher deltas agree closely.
Held-out validation uses the same 256 disjoint GSM8K Platinum questions and
352,926,720 projection elements as the feedback gate. W2.5 and W3 accept small
held-out regressions in exchange for speed; W3.5 is a speed/held-out-quality
double win. Telemetry reports `viterbi_provisional_direct_distance`.

Setting the variable to `1` applies direct distance to both passes. It is an
aggressive diagnostic mode, not the recommended configuration: at W3 it was
1.080x faster end-to-end but increased held-out MSE by 0.0987%. The
provisional-only mitigation retained most of the layer speedup while reducing
that drift by 4.85x. `final` is also available for controlled A/B experiments.

## Recommended H100 fast-quality profile

The two independently gated rounding optimizations can be enabled together:

```bash
GPTQMODEL_QVQ_YAQA_FAST_TF32=1 \
GPTQMODEL_QVQ_YAQA_FAST_VITERBI_DISTANCE=provisional \
python scripts/qvq_quantize.py ...
```

This is the recommended opt-in H100 profile for QVQ + YAQA + contiguous P32.
It keeps the final constrained Viterbi solve exact, uses direct FP32 distance
only to choose the provisional tail-biting state, and uses TF32 only for the
wide factored-feedback GEMMs. Defaults remain exact and both switches remain
independent so deployments can gate them separately.

The interaction was measured end to end rather than inferred by adding the
individual results. The two rounding changes partially cancel harmful payload
drift at W2.5 and W3.5:

| Rate | Exact layer | Paired profile | Speedup | Per-module FP64 Fisher delta | Disjoint held-out MSE delta |
|---|---:|---:|---:|---:|---:|
| W2.5 | 25.001 s | 21.946 s | 1.139x | -1.084% to +0.415% | -0.01297% |
| W3 | 29.257 s | 26.089 s | 1.121x | -0.477% to +0.056% | +0.00744% |
| W3.5 | 27.344 s | 24.146 s | 1.132x | -0.747% to +0.336% | -0.11730% |

Negative held-out deltas improve quality. FP32 and FP64 Fisher deltas agree
closely for every projection. The held-out gate uses the same strictly
disjoint 256-question GSM8K Platinum set and 352,926,720 projection elements
as the independent gates. W2.5 and W3.5 are speed/quality double wins; W3
accepts a bounded 0.00744% held-out change for a 1.121x layer speedup.

Do not substitute the aggressive all-pass direct-distance mode: its W3
held-out drift is materially larger. Also do not predict the paired profile
from isolated microbenchmarks—the selected P32 payload depends on interactions
between both rounding boundaries. Re-run the dual FP32/FP64 Fisher gate and a
strictly disjoint held-out gate when changing model, architecture, calibration
data, CUDA/PyTorch stack, or GPU generation.
