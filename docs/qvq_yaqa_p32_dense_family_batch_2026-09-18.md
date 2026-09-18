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
segment plus traceback for each anti-diagonal and tail-biting pass. A future
phase should prototype persistent exact W3/W3.5 family-grid kernels, gated by
bitwise payload parity first and the FP32/FP64 plus disjoint held-out oracles
if operation ordering changes.
