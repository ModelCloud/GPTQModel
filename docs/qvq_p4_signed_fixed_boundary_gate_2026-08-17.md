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

## Disjoint teacher-KL gradient ranking follow-up

The next matched experiment tested whether P4 failed only because local YAQA/module-output ordering discarded the
right fixed-boundary candidates. It used the same model, packed prefix, W2 layer-2 Q target, ranks, alphas, and four
candidate replay budget. The 94 remaining untouched dataset rows were assigned without overlap:

- gradient generation: `[1954,1970)`, 4,812 valid tokens;
- two-fold search: `[1970,1986)`;
- confirmation: `[1986,2002)`;
- untouched evaluation: `[2002,2048)`.

The finite full-model teacher-KL gradient had L2 norm `0.0378667`, maximum magnitude `0.00181752`, and required
14.44 seconds. It ranked a portfolio containing three signed spectral candidates and one direct fixed-boundary
candidate by the first-order term

\[
\langle \nabla_W KL,\Delta W\rangle,
\]

then retained nonlinear two-fold replay and independent confirmation as the selection authorities.

The selected `r32_a1_t4352_s5` candidate improved search KL by `0.4123%`; both folds improved independently by
`0.6011%` and `0.1760%`. Search Top-1 and Top-5 improved by `0.0152` and `0.0243` points while Top-10 changed by
`-0.0061` points. Independent confirmation reversed direction:

| Confirmation metric | Gradient-ranked proposal versus YAQA rollback |
|---|---:|
| Final KL | +0.3550% |
| JSD | +0.3635% |
| Top-1 | +0.0000 pp |
| Top-5 overlap | +0.0103 pp |
| Top-10 overlap | +0.0180 pp |

The confirmation gate rejected the proposal and serialized the exact rollback.

A matched control omitted only the disjoint gradient and used identical search, confirmation, and evaluation rows.
It selected the exact same spectral candidate, produced the exact same search and confirmation metrics, and also
rolled back. Gradient ranking changed the other three shortlisted candidates but did not change the winner.
Quantization took 194.91 seconds with gradient ranking and 168.54 seconds without it, a `15.6%` increase.

This rejects **ranking-only P9 for this gate**. The next candidate generator must change the spectral direction
rather than reorder the existing portfolio. A mathematically distinct option is to project the downstream gradient
into the YAQA-whitened residual atom basis. For

\[
M=U\Sigma V^\top,\qquad
A_i=C_I^{-T}u_i v_i^\top C_O^{-1},
\]

compute each mode's first-order downstream coefficient

\[
c_i=\sigma_i\langle \nabla_W KL,A_i\rangle.
\]

Generate fixed-boundary targets from only modes with `c_i < 0`, ordered by `-c_i`, while retaining each selected
atom's signed `u_i sigma_i v_i^T` contribution. This is materially different from P9: propagation changes candidate
generation instead of only shortlist order. It must remain default-off and retain the independent ordinary-YAQA
oracle, exact serialization check, two-fold replay, confirmation, and untouched evaluation gates.

## Artifacts

- `artifacts/qvq_p4_signed_fixed_boundary_gate/layer2_q_w2_rows1826_1954_with_yaqa_diagnostics.json`
- `artifacts/qvq_p4_signed_fixed_boundary_gate/layer2_q_w2_rows1826_1954_with_yaqa_diagnostics.safetensors`
- `artifacts/qvq_p4_gradient_ranked_fixed_boundary_gate/layer2_q_w2_rows1954_2048.json`
- `artifacts/qvq_p4_gradient_ranked_fixed_boundary_gate/layer2_q_w2_rows1970_2048_nogradient.json`
