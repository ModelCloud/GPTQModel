# QVQ P4 signed fixed-boundary gate — 2026-08-17

## Question

Test whether a localized spectral refinement can preserve the signed paired residual direction

\[
U_r \Sigma_r V_r^\top
\]

while changing only one fixed-entry/fixed-exit V2B2-P32 trellis segment, and accept the serialized candidate only
after original-YAQA diagnostics and disjoint full-model propagated replay.

## Contract

- Model: real Llama 3.2 1B Instruct, complete 16-layer forward horizon.
- Rate and target: W2, `model.layers.2.self_attn.q_proj`.
- Live packed prefix: layer-0 Q/K/V/O plus layer-1 Q/K, all V2B2-P32+YAQA W2.
- YAQA factors: 512 independent full rows `[1024,1536)`, 163,324 valid tokens, Sketch-B seed 1.
- Search: rows `[1826,1858)`, 12,477 valid tokens, split into two replay folds.
- Confirmation: rows `[1858,1890)`, 10,788 valid tokens.
- Untouched evaluation: rows `[1890,1954)`, 23,494 valid tokens.
- Batch 1, full row lengths, ranks 8/16/32, alphas 0.25/0.5/1.0.
- At most eight proposed segments and one accepted fixed-boundary change; four candidates receive full-horizon replay.
- Confirmation requires at least 0.1% relative KL improvement and no Top-1/5/10 regression above 0.25 points.
- Device: Apple M4 Max MPS, all 12 P cores available to the host process.

The fixed-boundary unit gate passed 14/14 tests. It covers exact canonical recovery, entry/exit-state preservation,
interior-only state changes, composed segments, configuration roundtrip, and serialization neutrality.

## Candidate result

Positive original-YAQA values mean lower original Kronecker proxy loss. Positive module-search values mean lower
held-out target-module output loss. Final-KL values are relative to the exact serialized YAQA rollback; negative is
better. A candidate must improve both propagated folds, so the replay score is the worst relative fold ratio rather
than the token-weighted mean.

| Candidate | Original YAQA | Module search | Mean final KL | Fold 0 KL | Fold 1 KL | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r16_a1_t92_s5` | -0.048662% | -0.007441% | -0.347373% | +0.215369% | -0.670590% | -0.0080 pp | +0.0064 pp | +0.0096 pp |
| `r16_a1_t50_s5` | +0.030496% | -0.000455% | -0.013969% | +0.166480% | -0.117612% | -0.0080 pp | +0.0032 pp | -0.0040 pp |
| `r32_a1_t50_s5` | +0.039435% | -0.006042% | -0.044596% | +0.204044% | -0.187405% | +0.0241 pp | +0.0112 pp | +0.0056 pp |
| `r32_a1_t4352_s5` | +0.058368% | +0.003109% | +0.027754% | +0.144107% | -0.039075% | -0.0241 pp | -0.0080 pp | +0.0008 pp |

No candidate improved both propagated folds. The confirmation callback was therefore not invoked, no selector or
state change was selected, and the exact ordinary V2B2-P32+YAQA artifact was serialized.

The locked diagnostic rerun reproduced the complete search and evaluation metric trees exactly. Quantization time
was 170.47 seconds initially and 168.82 seconds on the diagnostic rerun. The report-only change retained each
candidate's exact original-YAQA and module-search losses; it did not change candidate generation or selection.

## Interpretation

The mechanism is mathematically and operationally valid, but this fresh real-model gate is a negative selection
result:

1. Preserving the signed spectral tensor is sufficient to cross discrete local trellis boundaries.
2. Improving the original YAQA quadratic is not sufficient to improve final logits. The candidate with the largest
   original-YAQA gain worsened aggregate final KL.
3. Aggregate final KL alone is not sufficient for selection. The apparent `-0.347%` mean-KL candidate improved one
   fold by `0.671%` but regressed the other by `0.215%`.
4. Fixed boundaries and atomic serialized rollback prevented this search-split instability from entering the
   checkpoint.

Keep P4 default-off. The next credible change must improve candidate diversity or ranking enough to produce a
material, seed-stable gain on every search fold before independent confirmation and untouched evaluation. Merely
widening ranks, alphas, or replay count is not supported by this result.

## Artifacts

- `artifacts/qvq_p4_signed_fixed_boundary_gate/layer2_q_w2_rows1826_1954_with_yaqa_diagnostics.json`
- `artifacts/qvq_p4_signed_fixed_boundary_gate/layer2_q_w2_rows1826_1954_with_yaqa_diagnostics.safetensors`
