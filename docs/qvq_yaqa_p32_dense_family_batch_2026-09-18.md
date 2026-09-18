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

### Optional large-final extension

An additional H100 speed/quality point applies direct FP32 distance to the
constrained final solve only when one family contains at least 64 tiles:

```bash
GPTQMODEL_QVQ_YAQA_FAST_TF32=1 \
GPTQMODEL_QVQ_YAQA_FAST_VITERBI_DISTANCE=provisional_large_final \
python scripts/qvq_quantize.py ...
```

Provisional solves still use direct distance at every size. Final solves below
64 tiles retain the expanded FP32 expression. This shape gate avoids the
small-batch region where direct final distance is neutral or slower and limits
its changed rounding boundary to large anti-diagonals. It is a more aggressive
opt-in than `provisional`, not the default fast-quality profile.

Matched Llama 3.2 1B layer-0 results compare this extension with the paired
profile above. The dual-oracle ranges are candidate-versus-paired-profile:

| Rate | Paired profile | Large-final | Additional speedup | Cumulative vs exact | FP64 Fisher range | Disjoint held-out MSE delta |
|---|---:|---:|---:|---:|---:|---:|
| W2.5 | 21.946 s | 20.673 s | 1.062x | 1.209x | -0.0848% to +0.1136% | +0.02436% |
| W3 | 26.089 s | 25.424 s | 1.026x | 1.151x | -0.1526% to +0.4942% | +0.05815% |
| W3.5 | 24.146 s | 22.834 s | 1.057x | 1.198x | -0.1256% to +0.6462% | +0.16256% |

FP32 and FP64 Fisher movements agree closely for every projection. Held-out
validation again covers 256 strictly disjoint GSM8K Platinum questions and
352,926,720 projection elements. The changes are bounded but not zero, so use
this mode only when those rate-specific quality movements meet the deployment
budget.

A 96-tile mitigation was also evaluated rather than assuming that fewer
changed solves must improve quality. At W3.5 it retained a 1.041x additional
speedup but worsened held-out drift to +0.20708%; it was rejected. Payload
selection is non-monotonic across anti-diagonals, so threshold changes require
fresh end-to-end dual-oracle and held-out gates.

### Optional W3 wide-family unification

Wide MLP projections normally run canonical YAQA and the sampled complementary
families concurrently on separate CUDA streams. The W3-only opt-in below puts
the canonical family and both diversity-selected alternatives into one
factored-feedback schedule:

```bash
GPTQMODEL_QVQ_YAQA_W3_UNIFIED_FAMILIES=1 \
GPTQMODEL_QVQ_YAQA_FAST_TF32=1 \
GPTQMODEL_QVQ_YAQA_FAST_VITERBI_DISTANCE=provisional_large_final \
python scripts/qvq_quantize.py ...
```

This shares the initial factor products, anti-diagonal dispatch, and feedback
updates. It also changes FP32 operation ordering and serializes a larger
family-grid Viterbi batch, so it is deliberately restricted to W3. The switch
does nothing at W2.5 and W3.5.

Matched one-layer H100 validation against the large-final profile found:

| Rate | Concurrent schedule | Unified schedule | Speedup | Decision |
|---|---:|---:|---:|---|
| W2.5 | 20.673 s | 21.009 s | 0.984x | reject; retain concurrent |
| W3 | 25.424 s | 24.185 s | 1.051x | accept as opt-in |
| W3.5 | 22.834 s | 23.665 s | 0.965x | reject; retain concurrent |

At W3, all four attention projections are tensor-identical. Eight of 37 saved
layer tensors change, all in the three MLP projections. FP32 and FP64 Fisher
movements agree: gate improves by 0.000126%/0.000059%, up regresses by
0.081396%/0.081393%, and down improves by 0.499124%/0.498878%. On the same
strictly disjoint 256-question GSM8K Platinum gate, aggregate projection MSE
changes by +0.010259% over 352,926,720 elements. The candidate checkpoint is
bit-identical with and without FP64 diagnostic scoring.

The rejected cross-rate trials still provide a useful scheduling result.
Unified W2.5 improved all three MLP Fisher objectives but was 1.6% slower;
unified W3.5 improved up and down but was 3.5% slower and slightly regressed
gate. Saved arithmetic is not sufficient when the larger family-grid batch
loses overlap, so do not broaden this gate based on objective movement alone.

### Optional device-resident Sketch-B factors

Exact CUDA Sketch-B collection normally copies every completed FP32 factor to
CPU for bounded long-lived storage. A one-layer Llama 3.2 1B run then copies
those same factors back to the same H100 as each projection is quantized. When
the complete exact factor set fits in one CUDA collection pass, this round trip
can be removed:

```bash
GPTQMODEL_QVQ_YAQA_RETAIN_DEVICE_FACTORS=1 python scripts/qvq_quantize.py ...
```

The opt-in fails closed unless collection uses CUDA, the exact dense Gram
strategy, and one factor pass. CPU, MPS, projected, streaming-projected, and
memory-bounded multi-pass collection retain their existing host-storage path.
Factors are released module by module after their final quantization consumer.
Telemetry reports `factor_device` and `retained_device_factors`.

Matched H100 validation retained 958,398,464 bytes of factors and reduced the
final host-transfer region from 1.08--1.34 seconds to about 0.003 seconds:

| Rate | Host-factor prepare + quantize | Device-resident | Speedup |
|---|---:|---:|---:|
| W2.5 | 26.976 s | 25.575 s | 1.055x |
| W3 | 30.511 s | 28.958 s | 1.054x |
| W3.5 | 28.940 s | 27.731 s | 1.044x |

All 37 saved layer tensors were bit-for-bit identical at every rate. Every
per-projection FP32 and FP64 Fisher objective was also numerically identical.
The W3 comparison includes the accepted W3 unified-family schedule in both
arms. Since serialized payloads are exact, the previously recorded strictly
disjoint held-out results are unchanged.

Batch-size changes are not a substitute for this optimization. Increasing
Sketch-B batch size from 8 to 16 slowed capture from 3.49 to 3.75 seconds and
changed 19 layer tensors because stochastic sampling is consumed in a
different grouping. Device retention leaves batch size, seed consumption,
Gram accumulation, and quantizer ordering unchanged.

### Exact norm-band pruning for family batches

The exact norm-band segmented-Viterbi kernel originally excluded
family-batched YAQA calls, leaving the default `viterbi_pruning=auto` policy as
a no-op for the dominant B2 family histories. The family grid now maps each
logical `(sequence, bank)` CTA to its physical family codebook tables while
keeping the same FP32 expanded-distance arithmetic, candidate order, tie
precedence, frontiers, backpointers, and final traceback.

Only unconstrained, unweighted W2.5/W3 FP16-codebook calls are eligible.
W3.5 and every constrained final pass remain on the baseline. Explicit
direct-distance modes also retain their existing recurrence because their
rounding order is intentionally different. For exact W3, the full norm-band
provisional recurrence replaces midpoint-only traceback when pruning is
enabled; H100 microbenchmarks found it 1.13--1.41x faster across family batch
widths 1--128.

Matched one-layer Llama 3.2 1B runs used CUDA-resident exact factors, no TF32
or direct-distance rounding, FP32 and FP64 objective scoring, and the same
disjoint calibration rows `[0,32)` and YAQA rows `[64,96)`:

| Rate | Pristine quantization | Family norm-band | Speedup | Candidate reduction |
|---|---:|---:|---:|---:|
| W2.5 | 24.424 s | 24.230 s | 1.008x | 0.829% |
| W3 | 28.647 s | 27.693 s | 1.034x | 1.833% |
| W3.5 | unchanged | ineligible | 1.000x | 0% |

All 17 serialized model shards have identical SHA-256 hashes between the
enabled and pristine arms at W2.5 and W3. Consequently every saved tensor and
both Fisher oracle values are exactly unchanged, and the existing strictly
disjoint held-out results carry through without rerunning inference.

An inference-tensor cache experiment raised retained norm-rank table entries
from zero to seven but moved W3 from 27.693 to 27.744 seconds. It was rejected
to avoid persistent GPU memory with no measured speed benefit.

### Exact norm-band pruning for the constrained final pass

The final tail-biting recurrence differs from its provisional pass only at
step zero: it permits states whose shifted index equals the provisional
overlap. In the norm-band kernel's skewed predecessor frontier this is exactly
one zero-cost suffix and infinity everywhere else. Initializing that frontier
directly preserves the baseline FP32 candidate expression, ascending-prefix
tie order, bank selection, backpointers, and traceback while allowing every
later step to use the same proven norm interval.

On real Llama 3.2 1B tiles, the constrained kernel improved W3 by 2.5--3.1x
over family-batch widths 1--256. W2.5 improved by 1.3--1.6x above the existing
small-batch cooperative region. Matched one-layer exact-profile runs found:

| Rate | Previous process quantization | Constrained norm-band | Speedup | Prepare + quantize speedup |
|---|---:|---:|---:|---:|
| W2.5 | 24.230 s | 23.917 s | 1.013x | 29.408 s -> 28.740 s (1.023x) |
| W3 | 27.693 s | 26.441 s | 1.047x | 32.845 s -> 31.390 s (1.046x) |
| W3.5 | unchanged | ineligible | 1.000x | unchanged |

Eligible exact norm-band dispatches doubled from 2,712 to 5,424 at W2.5 and
from 2,752 to 5,504 at W3 because each tail-biting solve can now accelerate
both passes. Small cooperative W2.5 calls remain on their faster specialized
kernel.

All 17 serialized checkpoint shards are SHA-256 identical to the merged
family-pruning checkpoints at both W2.5 and W3. Every per-module FP32 and FP64
Fisher oracle value is therefore unchanged, as are the carried-forward
strictly disjoint GSM8K Platinum held-out results. Calibration rows `[0,32)`
and YAQA rows `[64,96)` remained disjoint in both matched runs.

### Exact Shift-7 norm-band pruning for W3.5

“Shift 7” is the W3.5 V2 trellis transition width (`2 * 3.5`), not a 7-bit
weight format. The 16-bit trellis state partitions into 128 predecessor
prefixes and 512 suffix/frontier states. Quantization scores those 128
predecessors for every frontier state at each step, while P32 selects the
codebook bank for each contiguous 32-weight window.

The exact W3.5 specialization uses one CUDA thread per suffix instead of two,
512 threads per CTA, sorted codebook-norm chunks of four candidates, and exact
norm bounds. This doubles CTA residency while retaining the baseline FP32
expanded-distance expression, ascending-prefix tie order, state and selector
payloads, backpointers, and traceback. Both unconstrained and constrained
passes are eligible for B2/P32 and B4/P64 family grids.

On real Llama 3.2 1B tiles, the constrained recurrence improved by
3.25--3.45x over family-batch widths 1--256. The matched one-layer W3.5 run
measured:

| Metric | Pristine | Shift-7 norm band | Speedup |
|---|---:|---:|---:|
| Process quantization | 26.811 s | 25.844 s | 1.037x |
| Prepare + quantize | 31.713 s | 30.684 s | 1.034x |

All 17 serialized checkpoint shards have identical SHA-256 hashes. Every
per-module FP32 and FP64 Fisher oracle value is identical, so the
carried-forward strictly disjoint held-out result is unchanged. The matched
run used calibration rows `[0,32)` and YAQA rows `[64,96)`.

This optimization is quantization-only. Post-quant inference reads the
already-selected trellis states and P32 selectors; it does not execute the
128-way predecessor search. The checkpoint layout and inference kernels are
unchanged. Inference-side work should instead target fused state/selector
decode, codebook lookup, and quantized GEMM.

### Low-overhead exact pruning telemetry

An SM90 Nsight Systems capture of the matched W3.5 layer found that the exact
Shift-7 recurrence accounted for 67.1% of GPU kernel time (22.794 s across
43,920 segment launches). Norm-table sorting and bounds construction together
accounted for only 0.55%, ruling out table construction as the next bottleneck.

Production telemetry formerly reduced two uint64 candidate counters through a
full shared-memory block tree after every segment. The replacement performs
register shuffle reductions within each warp, writes one pair per warp, and
uses warp zero for the final reduction. It retains the same two global atomics
per CTA and bit-exact counter totals, while reducing the telemetry workspace
from `2 * block_threads * 8` bytes to `2 * warps * 8` bytes and replacing the
logarithmic barrier tree with one block barrier.

Warm real-tile H100 medians with telemetry enabled improved across every target
rate and measured batch:

| Rate | Batch range | Previous | Warp reduction | Speedup |
|---|---:|---:|---:|---:|
| W2.5 | 32--256 | 0.640--2.049 ms | 0.630--2.030 ms | 1.009--1.021x |
| W3 | 32--256 | 0.396--1.351 ms | 0.393--1.336 ms | 1.008--1.014x |
| W3.5 | 32--256 | 0.278--0.856 ms | 0.275--0.843 ms | 1.008--1.016x |

The exact telemetry test still reports the analytical possible-candidate count
and the measured evaluated-candidate count, and unconstrained/constrained
family-grid payloads remain bit-for-bit identical to the pristine recurrence.
This is quantization-only observer work and has no inference-side analogue.

Two alternatives were rejected during this phase. A per-chunk predecessor
floor was mathematically exact but added shared-frontier loads and control flow,
regressing real W3.5 microbenchmarks by 12.9--15.7%. A 256-thread Shift-7
multiwave launch improved isolated batches 128--256 by about 1.15x, but its
first pipeline measurement was invalidated by an unrelated orphan GPU workload.
Without a clean end-to-end promotion result, the isolated result was not treated
as production evidence. Neither rejected candidate remains in the source.

### Adaptive Shift-7 CTA geometry

The clean follow-up retained both exact Shift-7 kernels and selects their CTA
shape from launch occupancy. SASS resource metadata reports 96 registers per
thread for the 512-thread kernel and 110 for the 256-thread kernel. On H100,
that permits one 512-thread CTA or two 256-thread CTAs per SM. The 512-thread
form therefore remains selected while the grid is smaller than one SM wave;
at `batch * bank_count >= multiprocessor_count`, the 256-thread form processes
two suffix columns per thread and exposes the second resident CTA.

Warm real Llama 3.2 1B tile medians on the 132-SM H100 were:

| Shape | Batch | 512 threads | Adaptive | Speedup |
|---|---:|---:|---:|---:|
| B2/P32 | 72 | 440.7 us | 384.0 us | 1.148x |
| B2/P32 | 128 | 453.2 us | 396.7 us | 1.142x |
| B2/P32 | 256 | 835.2 us | 711.1 us | 1.175x |
| B4/P64 | 64 | 461.3 us | 407.4 us | 1.132x |
| B4/P64 | 128 | 821.2 us | 697.8 us | 1.177x |
| B4/P64 | 256 | 1570.9 us | 1407.2 us | 1.116x |

Underfilled B2/P32 batches 32 and 64 retain the 512-thread kernel and remain
within 1.4% of the control. The same geometry rule applies to unconstrained and
tail-biting passes and leaves W2.5/W3 dispatch unchanged.

A matched one-layer run reduced exact segmented-Viterbi GPU time from 16.957 s
to 16.437 s (1.032x). Inclusive projection quantization moved from 25.867 s to
25.778 s; the recurrence gain is intentionally reported separately because
other layer phases dominate the small wall-time difference.

Quality was gated on a frozen 2048x2048 Q-projection solver input so CUDA
factor-capture allocation differences could not contaminate the A/B. The
baseline and adaptive branches produced identical reconstructed inner weights,
trellis states, and P32 selectors (matching SHA-256 hashes for all three), so
the independent FP32 and FP64 Fisher objective deltas are exactly zero. Large
grid B2/P32 and B4/P64 tests additionally compare both constrained and
unconstrained results bit-for-bit with the pristine recurrence.

This occupancy switch does not transfer directly to inference. The SM90 P32
inference path is warp-specialized around fixed WGMMA tile geometry rather than
one CTA per trellis suffix grid; changing its CTA width would change the MMA
mapping instead of merely assigning another suffix column to each thread. No
inference code or checkpoint layout changes in this phase.

### One-sqrt exact norm enclosure

Each exact norm-band step previously evaluated the target norm with
`sqrt_rd`, `sqrt_ru`, and `sqrt_rn`. The first two values only enclosed the
real square root for pruning; they never participated in candidate scoring.
The optimized kernel computes one correctly rounded `sqrt_rn` and uses the
immediately adjacent FP32 values as its lower and upper endpoints. Correct
rounding guarantees that the real square root lies between those neighbors,
so the interval remains conservative. Candidate distance arithmetic, scan
order, tie precedence, and traceback are unchanged.

Warm real Llama 3.2 1B B2/P32 medians on H100 were:

| Rate | Batch range | Previous | One sqrt | Speedup range |
|---|---:|---:|---:|---:|
| W2.5 | 32--256 | 0.622--2.014 ms | 0.623--2.000 ms | 0.998--1.009x |
| W3 | 32--256 | 0.389--1.321 ms | 0.382--1.300 ms | 1.014--1.019x |
| W3.5 | 32--256 | 0.271--0.711 ms | 0.266--0.706 ms | 1.007--1.020x |

The W2.5 batch-64 movement is 0.18% negative and within run-to-run noise; all
other listed cells improve. Constrained W3.5 improves 1.017x, 1.022x, 1.000x,
and 1.018x at batches 32, 64, 128, and 256 respectively.

In the matched one-layer W3.5 run, exact segmented-Viterbi GPU time fell from
16.957 s to 16.370 s (1.036x), inclusive projection quantization from 25.867 s
to 25.649 s (1.009x), and prepare plus quantize from 30.791 s to 30.568 s
(1.007x). A frozen 2048x2048 solver input produced identical reconstructed
weights, trellis states, and selectors against merged main. Its independent
objectives were identical in both arms: FP32 `0.22860166430473328` and FP64
`0.22860163687284052`. The full norm-rank/pruning-policy suite passed all 113
tests, including adversarial ties, tiny/large values, constrained family
batches, and exact candidate telemetry.

This is quantization-only. Post-quant inference does not construct norm bands
or evaluate this square root, so no inference code or checkpoint format changes.

### Fast conservative pruning radius

The norm-band radius is a search enclosure, not part of the candidate score.
PTX specifies a maximum relative error of `2^-23` for `sqrt.approx.f32` over
positive finite FP32 inputs. The kernel already multiplies the radius upward
by exactly `1 + 2^-19`, a factor sixteen times larger than that error bound.
Replacing the correctly rounded radius square root with `sqrt.approx.f32`
therefore preserves a strict upper radius bound. Candidate distances still use
the original FP32 FMA/add/subtract sequence, and prefix scan order, ties,
backpointers, and traceback are unchanged. Zero retains radius zero; non-finite
enclosures continue to fall back to the exact full scan.

Warm real Llama 3.2 1B B2/P32 medians on H100 improved in every unconstrained
cell measured:

| Rate | Batch range | One-sqrt control | Fast radius | Speedup range |
|---|---:|---:|---:|---:|
| W2.5 | 32--256 | 0.623--2.000 ms | 0.614--1.973 ms | 1.004--1.015x |
| W3 | 32--256 | 0.382--1.300 ms | 0.379--1.285 ms | 1.008--1.015x |
| W3.5 | 32--256 | 0.266--0.706 ms | 0.264--0.696 ms | 1.007--1.015x |

The constrained pass also improved all 12 measured cells: W2.5 by
1.015--1.026x, W3 by 1.004--1.013x, and W3.5 by 1.002--1.008x. The matched
one-layer W3.5 run was correctly treated as pipeline-neutral: segmented
Viterbi moved from 16.3700 s to 16.3611 s and prepare plus quantize from
30.5684 s to 30.5656 s, while process quantization moved from 25.649 s to
25.656 s (+7 ms noise). No layer-level speedup is claimed.

A frozen 2048x2048 W3.5 solver A/B selected family 2 and produced identical
SHA-256 hashes for reconstructed weights, trellis states, P32 selectors, and
the family ID. Both independent objectives were bit-identical: FP32
`0.22860166430473328` and FP64 `0.22860163687284052`. The 113-test exact
norm-rank and pruning-policy suite passed, including constrained grids,
rounding edges, ties, tiny/large values, and telemetry.

Two alternatives were rejected before this result. Computing the target norm
once per warp and broadcasting six values regressed all rates by roughly
3--6%; retaining only two root-bound shuffles still regressed by 2--4%.
Locally folding next-frontier shared atomics was also negative: W3/W3.5 lost
3--5%, while W2.5 gained only about 0.5% at large batches and regressed small
batches. None of those experiments remains in production source.

This change is quantization-only. Inference consumes the selected states and
P32 selectors and never constructs this pruning radius, so there is no safe
inference-side transplant and no checkpoint-format change.

### Rate-aware approximate target-root enclosure

An SM90 Nsight Compute sample of the 256-thread Shift-7 recurrence measured
55.1% SM throughput, 0.2% DRAM throughput, and a 95.6% L1 hit rate. Warp issue
stalls were led by fixed-latency dependencies (26.18%) and long scoreboards
(14.0%), so additional global-memory packing was not indicated. SASS showed
that `sqrt_rn` for the target-root enclosure expanded into reciprocal-square-
root and correction instructions, whereas the already-proven
`sqrt.approx.f32` radius uses one MUFU instruction.

PTX bounds `sqrt.approx.f32` to `2^-23 = 2u` relative error. For W2.5 and the
256-thread W3.5 specialization, the target estimate is contracted and expanded
with directed rounding by `1 - 2^-22` and `1 + 2^-22`: a 4u correction, twice
the maximum approximation error. The upper endpoint then receives the original
`1 + 2^-19` proof expansion. These endpoints therefore enclose the true target
root before the unchanged norm-band derivation. Candidate-score arithmetic,
prefix order, ties, backpointers, and traceback remain untouched.

The optimization is intentionally rate and geometry aware. W3 and 512-thread
W3.5 retain the correctly rounded target root because their cheaper candidate
scans could not amortize the slightly wider approximate enclosure. Matched,
sequential H100 B2/P32 medians were:

| Pass | Rate/geometry | Batch range | Merged main | Candidate | Speedup |
|---|---|---:|---:|---:|---:|
| Unconstrained | W2.5 | 32--256 | 0.617--1.978 ms | 0.606--1.957 ms | 1.008--1.012x |
| Unconstrained | W3.5, 256 threads | 128--256 | 0.386--0.701 ms | 0.383--0.692 ms | 1.008--1.013x |
| Constrained | W2.5 | 32--256 | 0.651--2.130 ms | 0.644--2.115 ms | 1.007--1.015x |
| Constrained | W3.5, 256 threads | 128--256 | 0.567--1.039 ms | 0.563--1.030 ms | 1.006--1.009x |

W3 and 512-thread W3.5 compile to the merged-main path. SASS resource usage is
unchanged: W2.5 uses 32 registers, W3 uses 64, and Shift-7 uses 96 registers at
512 threads and 110 at 256 threads, with no local or shared spilling.

A clean matched W2.5 layer rerun reduced segmented-Viterbi GPU time from
14.6092 s to 14.5352 s (1.005x) and prepare plus quantize from 28.6264 s to
28.5440 s (1.003x). Inclusive process-quantization accounting moved from
23.517 s to 23.730 s, so no process-level speedup is claimed. An earlier
candidate arm incurred an isolated 104-second sampled-family stall and was
discarded rather than included in the timing comparison. Both clean arms used
calibration rows `[0,32)` and disjoint YAQA rows `[64,96)`. Selected families
and both FP32 and FP64 oracle values were identical for all seven modules; all
174 serialized checkpoint tensors also had identical shapes, dtypes, and
SHA-256 payload hashes.

Frozen 2048x2048 solver A/Bs were payload-identical at both promoted rates.
W2.5 selected family 1 with FP32/FP64 objectives `1.2377052307128906` and
`1.2377053068101906`; W3.5 selected family 2 with objectives
`0.22860166430473328` and `0.22860163687284052`. Reconstructed weights,
trellis states, P32 selectors, and family IDs had matching SHA-256 hashes.
All 113 norm-rank/pruning-policy tests passed.

Two SASS-guided alternatives were rejected. Pairwise Shift-7 argmin shortened
the winner dependency chain but raised live state and regressed 0.4--1.0%.
Replacing the fully unrolled reseed rank with a 3--5-load binary lift reduced
loads and registers but exposed dependent-load latency, regressing 1--2%.
Neither experiment remains in source.

This is quantization-only. Post-quant inference does not evaluate target-root
norm bands, so there is no inference-side kernel or checkpoint-format change.
