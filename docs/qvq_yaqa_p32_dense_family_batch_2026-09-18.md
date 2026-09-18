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
