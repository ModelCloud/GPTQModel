# QVQ YAQA Sketch-B telemetry and Apple batching fix

This report records the investigation that stopped a full 16-layer Llama 3.2 1B run after more than three hours
without completing YAQA Sketch-B collection. The run had not reached quantization: the failure was collection
throughput, not V2B2-P32 Viterbi or evaluation.

## Root cause

The ordinary calibration/evaluation contract intentionally used batch 1. That batch size was also being reused for
YAQA, although Sketch-B retains a distinct gradient sample for every sequence when independent rows are collated.
This caused hundreds of full-model forward/backward launches. On MPS, the collector additionally ran Python GC,
stream synchronization, and `empty_cache()` after every batch. Its finite checks also converted accelerator
reductions to Python booleans inside every module hook, synchronizing the command stream repeatedly.

The accepted lifecycle therefore separates the controls:

```text
ordinary calibration/evaluation
    -> batch 1, unchanged

YAQA Sketch-B
    -> batch 8 by default
    -> batch 4 for the profiled 16-layer/all-linear Apple geometry
    -> stable descending-length bucketing before batching
    -> retain one Fisher sample per independent sequence
    -> checkpoint decoder layers
    -> accumulate finite flags on device
    -> GC/synchronize/empty-cache every 8 batches and at the final batch
```

Batching and length bucketing change launch and padding efficiency, not the estimator definition:

\[
H_I = \frac{1}{S d_{out}}\sum_{s=1}^{S}G_s^\top G_s,
\qquad
H_O = \frac{1}{S d_{in}}\sum_{s=1}^{S}G_s G_s^\top.
\]

Each `G_s` is still formed independently. No cross-sequence Gram term is introduced. Cleanup scheduling occurs
after backward and therefore cannot change `G_s` or either accumulated factor.

Length bucketing consumes the same rows and masks. In exact arithmetic it only changes their evaluation order.
Because MPS categorical sampling consumes pseudorandom values in batched order, it should be treated as a different
valid Monte Carlo realization rather than as a bit-identical rescheduling of an existing artifact. Reproducibility
therefore requires recording the sort mode in factor-cache provenance.

## Real-model telemetry

Host: Apple M4 Max, all 12 performance cores, MPS, FP32 Sketch factors. Model: real Llama 3.2 1B Instruct.
Dataset: `neuralmagic/calibration`, full rows beginning at offset 512, no concatenation or truncation. Scope:
Q/K/V/O/gate/up/down. Activation checkpointing remained enabled.

### Batch-size gate, 16 rows / 5,831 valid tokens

| Layers | Batch | Capture time | Relative speed |
|---:|---:|---:|---:|
| 1 | 1 | 12.978 s | 1.00x |
| 1 | 8 | 4.528 s | 2.87x |
| 2 | 1 | 22.243 s | 1.00x |
| 2 | 8 | 6.878 s | 3.23x |

Batch 16 was slower than batch 8 because variable-length padding increased useless token work. Disabling activation
checkpointing was also rejected: it was 13.5% slower in the one-layer gate and raised the process footprint to
approximately 25 GiB.

### Full 16-layer/all-linear geometry, 16 rows / 5,831 valid tokens

This gate targeted all 112 Q/K/V/O/gate/up/down projections and materialized 14.28 GiB of FP32 factors. It exposed
the M4-specific batch-size knee that is invisible in one- and two-layer tests.

| Ordering | Batch | Capture | Backward + Sketch | Loss | Final transfer | Relative to native B8 |
|---|---:|---:|---:|---:|---:|---:|
| Native | 8 | 47.150 s | 41.154 s | 3.292 s | 1.138 s | 1.00x |
| Length-descending | 2 | 40.815 s | 36.823 s | 2.017 s | 0.849 s | 1.16x |
| Length-descending | 4 | **38.519 s** | **34.137 s** | 2.395 s | 0.898 s | **1.22x** |
| Length-descending | 8 | 44.240 s | 38.560 s | 2.952 s | 0.987 s | 1.07x |
| Length-descending | 16 | 108.750 s | 91.113 s | 5.039 s | 9.127 s | 0.43x |

Batch 4 is the selected full-model Apple setting. Batch 8 remains the general default because it won the smaller
one- and two-layer gates; callers quantizing the complete Llama 3.2 1B linear set should explicitly request batch 4.
The batch-16 cliff is unified-memory/temporary-workspace pressure, not insufficient CPU threads.

### Cleanup and phase gate, 64 rows / 23,482 valid tokens

| Layers | Cleanup interval | Capture | Forward | Loss | Backward + Sketch | GC + MPS cleanup | Factors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 16.806 s | 0.183 s | 3.886 s | 10.955 s | 1.483 s | 0.893 GiB |
| 1 | 8 | 14.646 s | 0.151 s | 3.192 s | 10.772 s | 0.353 s | 0.893 GiB |
| 2 | 8 | 24.569 s | 0.174 s | 3.896 s | 19.754 s | 0.484 s | 1.785 GiB |

Interval 8 is 1.15x faster than draining every batch. Every saved input/output factor was bit-identical. The MPS
finite-flag change is also bit-identical because it evaluates the same predicates and changes only when the host
observes their OR reduction.

The first patch gave 3.5x in the measured one-layer 64-row contract. The full-geometry batch-4 and bucketing gate
reduces the measured 16-row capture to 38.519 seconds. Linear extrapolation to 512 rows predicts about 20.5 minutes
of Sketch-B collection, versus the stopped run already exceeding three hours. That is an **at-least 8.8x projected
lower bound** because the old job had not completed. A measured 10x claim remains gated on the restarted 512-row
run; the projection is not substituted for that measurement.

## Rejected experiments

| Candidate | Result | Decision |
|---|---|---|
| Batch 16 | Slower than batch 8 on variable full rows | Reject as default |
| Disable decoder checkpointing | 13.5% slower and much higher memory | Reject |
| Keep complete factors resident on MPS | Only about 1.07x on layer 1; unacceptable full-model memory risk | Reject |
| MPS accumulators on full 112-module geometry | 45.479 s versus 28.093 s on CPU for one batch; 6.543 s final transfer | Reject |
| Four-operand official-style MPS einsum | Attempted an impossible petabyte-scale intermediate | Reject |
| Adaptive token-space associative Gram | Worst relative factor drift `8.92e-8`, but 3.8% slower | Reject |
| Batched token-space reassociation on real projection shapes | Up to `3.45e-6` relative drift and only ~1.2x micro speed | Reject: exceeds `1e-6` quantization tolerance |

## Remaining bottleneck and next work

The first restarted 512-row run exposed a full-population memory cliff that the 16-row gate could not: the first
6,014-token batch completed, but all 112 modules kept 14.28 GiB of persistent FP32 accumulators while MPS also held
large transient weight-gradient and Gram workspaces. Free memory fell to roughly 68 MiB and the next batch made no
progress for more than three minutes. That run was stopped rather than allowed to swap for hours.

The accepted exact remedy streams collection in whole-layer groups on MPS. The default 4 GiB factor budget produces
four passes of four Llama layers each:

```text
same dense model + same rows + same seed
    +-- full backward, target layers 0--3   -> retain factors on CPU
    +-- full backward, target layers 4--7   -> retain factors on CPU
    +-- full backward, target layers 8--11  -> retain factors on CPU
    `-- full backward, target layers 12--15 -> retain factors on CPU
```

This is not a layer-local Fisher approximation. Every pass still traverses the complete dense model from final loss
to the first decoder layer. Resetting the same sampling seed regenerates the same sampled token targets, and only
gradient hooks outside the current target group are omitted. A two-layer exact oracle verified bit-identical input
and output factors between one pass and two one-layer passes (`rtol=0`, `atol=0`).

Peak target-factor workspace falls from 14.28 GiB to approximately 3.57 GiB. The cost is four full-model backward
traversals, but model traversal is cheaper than repeatedly paging tens of GiB. `max_factor_bytes_per_pass` can set an
explicit positive byte budget; `None` selects 4 GiB on MPS and an unchunked pass elsewhere.

The remaining native-kernel opportunity is a workspace-bounded MLX/Metal Sketch operator that consumes each
sequence's activation and output gradient, accumulates both symmetric factors directly, and preserves the `1e-6`
factor contract. The materialized PyTorch implementation remains the oracle until exact factors, quantized
states/selectors, and held-out final metrics validate such a kernel.

## Accepted symmetric-Gram packing

The safe first symmetric optimization keeps the established MPS `bmm` reductions unchanged. Both Sketch-B updates
are Gram matrices and MPS produced them bit-symmetric across the tested 512, 2,048, 4,096, and 8,192 feature
geometries. A native MLX/Metal kernel now packs the lower triangle before the host handoff; CPU accumulation uses
that packed representation, and a second native kernel expands it once when the final factor is materialized.

This changes storage and transfer, not the estimator:

\[
N^2\ \text{FP32 elements}
\quad\longrightarrow\quad
\frac{N(N+1)}{2}\ \text{FP32 elements}.
\]

The real Llama 3.2 1B all-linear factor population falls from 14.28 GiB to 7.14 GiB while accumulating. Under the
same 4 GiB lifecycle budget, whole-layer streaming falls from four 28-module passes to two 56-module passes. The
standalone comparison harness can safely collect all 112 modules in one packed pass.

Validation on real model weights and natural calibration rows:

| Scope | Rows / valid tokens | Full accumulator | Packed accumulator | Result |
|---|---:|---:|---:|---|
| 2 layers, all 14 linear projections | 16 / 5,831 | 5.985 s | 6.025 s | Every input/output factor bit-exact |
| 16 layers, all 112 linear projections | 16 / 5,831 | 38.519 s historical | 40.132 s | One packed pass completed without memory pressure |

The isolated one-pass cost is approximately 0.7% on two layers and 4.2% against the historical full-accumulator
16-layer run. That small cost is accepted because the full accumulator cannot survive the 512-row production run.
Relative to the safe four-pass plan, the one-pass 16-layer gate is about 3.84x faster at the Sketch collection
level; the normal 4 GiB lifecycle conservatively uses two passes and should approach a 2x reduction in repeated
full-model backward work. Larger factors and packed round trips were exact at `rtol=0, atol=0`, which is stronger
than the required `1e-6` quantization tolerance.

Rejected alternatives remain rejected: flattening or reassociating the Gram contraction changed FP32 reduction
order by up to approximately `3.45e-6` and provided little speedup. Symmetric packing is accepted specifically
because it preserves every arithmetic operation and only removes redundant storage of the mirrored triangle.
