# QVQ + YAQA P32 quantization speed gate

This phase optimizes checkpoint construction only. It does not alter post-quant inference kernels.

## Optimization

Full YAQA family reselection already evaluates canonical V2 and every legal complementary family under the authoritative Kronecker-Fisher objective. The former production path nevertheless ran a complete preliminary Block-LDLQ family search solely to populate comparison diagnostics. Its winner could not affect the full-reselection payload unless spectral refinement or spectral push consumed that diagnostic baseline.

The production path now skips that reporting-only search for ordinary full reselection. Fixed-family, sampled-family, spectral-refinement, spectral-push, and direct diagnostic calls retain the search. Telemetry records `yaqa_v2b2_block_family_diagnostics_skipped` when it is omitted.

## Matched H100 result

All runs quantized layer 0 of Llama 3.2 1B with QVQ V2B2-P32, YAQA full family reselection, 32 disjoint Sketch-B sequences, batch 8, exact pruning, and the same seed and source checkpoint. Timings are the inclusive sum of seven projection `process_quant` regions.

| Rate | Baseline | Optimized | Speedup | Exact checkpoint tensors | Module losses |
|---|---:|---:|---:|---:|---:|
| W2.5 | 61.812 s | 34.987 s | 1.767x | 174/174 | identical |
| W3 | 71.532 s | 41.113 s | 1.740x | 174/174 | identical |
| W3.5 | 68.604 s | 38.909 s | 1.763x | 174/174 | identical |

The checkpoint comparison includes all five P32 payload tensors for all seven projections: `SU`, `SV`, `trellis`, `bank_ids`, and `bank_alt_id`. No quantized module or non-quantized checkpoint tensor changed in the measured rate matrix.

## Dual-precision quality oracle

`scripts/validate_qvq_yaqa_p32_dual_oracle.py` independently reconstructs the selected P32 payload and evaluates

`trace(E.T @ H_input @ E @ H_output)`

in both FP32 and FP64 from a frozen solver-input snapshot. TF32 is disabled. The default acceptance budget is zero objective regression; explicit absolute or relative budgets may be supplied for later optimizations that deliberately trade a bounded amount of objective quality for speed.

On the real W3 layer-0 Q projection:

- FP32: baseline = candidate = `0.016224926337599754`
- FP64: baseline = candidate = `0.016224940796133396`
- Baseline/candidate delta: zero in both precisions
- FP32-to-FP64 relative gap: `8.91e-7`
- Serialized module payload: exact in all five fields

The FP32/FP64 difference is the expected accumulation-order precision gap. It is not optimization drift because baseline and candidate are identical within each oracle.

## Acceptance policy

A quantization optimization is accepted only when:

1. matched controls use the same model, rate, seed, YAQA data, and quantization settings;
2. speed improves in the inclusive quantization region, not merely in an isolated helper;
3. FP32 and FP64 oracle regressions both remain inside the configured budget;
4. serialized payload differences are reported explicitly;
5. dataset identity and row ranges remain disjoint from post-quant evaluation data.

Bitwise payload identity remains the strongest result, but it is not the only admissible result. A payload-changing optimization may proceed when both precision oracles and held-out quality stay inside an explicitly recorded bounded-loss budget.
