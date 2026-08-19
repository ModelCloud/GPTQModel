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
    -> retain one Fisher sample per independent sequence
    -> checkpoint decoder layers
    -> accumulate finite flags on device
    -> GC/synchronize/empty-cache every 8 batches and at the final batch
```

Batching changes launch and padding efficiency, not the estimator definition:

\[
H_I = \frac{1}{S d_{out}}\sum_{s=1}^{S}G_s^\top G_s,
\qquad
H_O = \frac{1}{S d_{in}}\sum_{s=1}^{S}G_s G_s^\top.
\]

Each `G_s` is still formed independently. No cross-sequence Gram term is introduced. Cleanup scheduling occurs
after backward and therefore cannot change `G_s` or either accumulated factor.

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

### Cleanup and phase gate, 64 rows / 23,482 valid tokens

| Layers | Cleanup interval | Capture | Forward | Loss | Backward + Sketch | GC + MPS cleanup | Factors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 16.806 s | 0.183 s | 3.886 s | 10.955 s | 1.483 s | 0.893 GiB |
| 1 | 8 | 14.646 s | 0.151 s | 3.192 s | 10.772 s | 0.353 s | 0.893 GiB |
| 2 | 8 | 24.569 s | 0.174 s | 3.896 s | 19.754 s | 0.484 s | 1.785 GiB |

Interval 8 is 1.15x faster than draining every batch. Every saved input/output factor was bit-identical. The MPS
finite-flag change is also bit-identical because it evaluates the same predicates and changes only when the host
observes their OR reduction.

The combined expected improvement over the stopped batch-1/every-batch-cleanup run is approximately 3.5x in the
measured one-layer 64-row contract. Extrapolating the measured two-layer marginal cost to 16 layers and 512 rows
predicts roughly 20--25 minutes of Sketch-B collection rather than more than three hours. This estimate must be
reported as a projection until the restarted full run completes; it is close to, but does not by itself prove, a
10x end-to-end speedup.

## Rejected experiments

| Candidate | Result | Decision |
|---|---|---|
| Batch 16 | Slower than batch 8 on variable full rows | Reject as default |
| Disable decoder checkpointing | 13.5% slower and much higher memory | Reject |
| Keep complete factors resident on MPS | Only about 1.07x on layer 1; unacceptable full-model memory risk | Reject |
| Four-operand official-style MPS einsum | Attempted an impossible petabyte-scale intermediate | Reject |
| Adaptive token-space associative Gram | Worst relative factor drift `8.92e-8`, but 3.8% slower | Reject |

## Remaining bottleneck and next work

The telemetry shows that backward plus per-sequence gradient/Gram construction dominates. The next credible large
gain requires a workspace-bounded native MLX/Metal Sketch operator that consumes each sequence's activation and
output gradient, accumulates both symmetric factors directly, and preserves the `1e-6` factor contract. A mere
reassociation in eager MPS is insufficient. The materialized PyTorch implementation remains the oracle until exact
factors, quantized states/selectors, and held-out final metrics validate such a kernel.
