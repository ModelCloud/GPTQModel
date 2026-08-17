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

## Propagation-shaped signed-spectrum follow-up (P10)

P10 implemented that distinct generator without changing the checkpoint or inference format. It used the exact
same real Llama model, packed prefix, W2 layer-2 Q target, 512-row YAQA factors, ranks, alphas, and disjoint P9
splits. For the YAQA-whitened residual

\[
M=U\Sigma V^\top,
\]

it evaluated every oversampled signed atom with

\[
c_i=\left\langle \nabla_W KL,
C_I^{-T}(\sigma_i u_i v_i^\top)C_O^{-1}\right\rangle,
\]

discarded modes with `c_i >= 0`, and reordered the remaining complete atoms from most to least favorable. It did
not alter singular-vector signs, singular values, or atom magnitudes. The existing ranks therefore became prefixes
of the propagation-favorable signed spectrum rather than prefixes of descending local spectral energy.

The disjoint gradient reproduced the P9 norm and magnitude. The randomized rank-32 SVD exposed 64 oversampled
modes; 46 had a favorable negative first-order coefficient and the 32 most favorable were retained. This changed
candidate generation materially: P10 selected `r32_a1_t384_s0`, rather than P9's `r32_a1_t4352_s5`.

| Stage or metric | P10 proposal versus exact YAQA rollback |
|---|---:|
| Search fold 0 final KL | -0.4676% |
| Search fold 1 final KL | -0.2000% |
| Token-weighted search final KL | -0.3487% |
| Independent-confirmation final KL | +0.0240% |
| Independent-confirmation JSD | +0.0468% |
| Independent-confirmation Top-1 | -0.0515 pp |
| Independent-confirmation Top-5 overlap | +0.0000 pp |
| Independent-confirmation Top-10 overlap | +0.0206 pp |

The candidate improved both search folds and therefore reached confirmation, but confirmation reversed direction.
The gate rejected it and serialized the exact rollback. Untouched evaluation consequently matches the rollback;
the reported packed-versus-dense differences remain ordinary backend numerical drift, not an accepted P10 change.

One diagnostic is especially important. Although every retained continuous spectral atom had `c_i < 0`, the final
discrete fixed-boundary segment replacement had a slightly positive propagated first-order product
(`1.20915e-6`). Projection into a constrained trellis segment is not sign preserving: a target assembled from
favorable continuous atoms can cross to a representable discrete delta pointing in a different direction. Search
replay rescued this particular candidate on the search rows, but did not generalize to confirmation.

P10 is therefore **rejected as a default** on this matched gate. It is a valid experimental generator, remains
default-off, and keeps exact rollback. The next iteration should constrain or rerank the *realized serialized
delta*, not only its continuous spectral teacher—for example, require a favorable realized first-order product and
then use cross-fitted replay across more independent rows/seeds before confirmation. Merely widening ranks or
alphas is not supported.

## Realized serialized-gradient gate (P11)

P11 tested that next boundary directly. For every legal fixed-boundary candidate it computed

\[
g_{\mathrm{realized}}=
\left\langle\nabla_W KL,
Q_{\mathrm{candidate}}-Q_{\mathrm{baseline}}\right\rangle
\]

on the actual reconstructed dense weight that would be serialized. A candidate with
`g_realized >= 0` was rejected before expensive full-model replay. To avoid merely deleting P10's sole changed
path, the matched run retained ranks 8/16/32 and used stronger shaped pushes `alpha in {1,2,4}`. All model, prefix,
factor, gradient, search, confirmation, and evaluation rows otherwise remained identical to P9/P10.

The stronger pushes produced four changed candidates and all four had favorable realized products. Their local
metrics were allowed to regress because downstream recovery is the primary objective. The selected
`r32_a4_t7_s5` candidate is the clearest example:

- realized propagated first-order product: `-6.24844e-6`;
- original YAQA proxy: `+1.5826%` worse;
- held-out target-module output proxy: `+0.0769%` worse;
- search fold final KL: `-0.2372%` and `-0.2127%`;
- token-weighted search final KL: `-0.2263%`.

Thus P11 proves that a locally worse candidate can be directionally useful after full propagation, and the
localized YAQA/module proxies must not veto it. However, the independent confirmation split reversed again:

| Confirmation metric | P11 proposal versus exact YAQA rollback |
|---|---:|
| Final KL | +0.3391% |
| JSD | +0.3410% |
| Top-1 | +0.0773 pp |
| Top-5 overlap | +0.0000 pp |
| Top-10 overlap | +0.0026 pp |

The Top-N changes are flat-to-positive, but the predeclared primary final-KL gate materially regressed, so the
proposal was rejected and the exact rollback was serialized. P11 remains default-off.

This changes the diagnosis. P10 lacked useful realized directions; P11 found several and replayed four of them.
The remaining failure is generalization across small prompt splits: one gradient split plus two search folds can
still select a candidate whose final-KL direction reverses on confirmation. The next experiment should therefore
change the estimation contract rather than widen the same candidate sweep: use cross-fitted gradients and replay,
require a candidate to remain favorable across independent gradient/search folds, and reserve a fresh dataset or
task-like split for confirmation. No additional format or inference work is justified before that gate.

## Cross-fitted gradient consensus (P12)

P12 split the same 16 disjoint gradient rows into two interleaved folds. A continuous signed spectral atom survived
only when both fold products were negative, and modes were ranked by their worse (largest) fold product. The actual
serialized candidate delta then had to satisfy the same all-fold rule before full replay. The token-weighted mean
gradient remained available only as the core shortlist ordering signal; it could not override a fold disagreement.

The two folds contained 3,161 and 1,651 valid tokens. Their gradient L2 norms were `0.05372` and `0.03816`. Cross-fit
consensus retained only 16 signed modes, versus 46 modes with a negative pooled coefficient in P11. Of the four
shortlisted serialized candidates, three were rejected before replay because the first gradient fold was favorable
but the second was adverse. The sole all-fold candidate, `r16_a4_t7425_s0`, produced:

| Stage or metric | P12 proposal versus exact YAQA rollback |
|---|---:|
| Search fold 0 final KL | -0.5680% |
| Search fold 1 final KL | -0.1878% |
| Token-weighted search final KL | -0.3992% |
| Independent-confirmation final KL | +0.1832% |
| Independent-confirmation JSD | +0.1721% |
| Independent-confirmation Top-1 | +0.0000 pp |
| Independent-confirmation Top-5 overlap | +0.0052 pp |
| Independent-confirmation Top-10 overlap | +0.0000 pp |

The robust estimator therefore halved P11's confirmation KL regression (`+0.3391%` to `+0.1832%`) and removed its
confirmation Top-1 movement, but it did not reverse the primary metric. The locked gate rejected the proposal and
serialized the exact rollback. P12 remains default-off.

This is evidence that fold consensus filters unstable directions, not evidence that eight-row gradient folds are
sufficient. Candidate capacity is present and the all-fold search result is reproducibly positive; estimator sample
size is now the limiting variable. The next matched test should give each gradient and search fold more rows while
retaining a separate confirmation set. Previously reported rollback-only evaluation rows may be reassigned for this
development test, but any eventual promotion still requires a genuinely fresh dataset/task split.

## Expanded cross-fit evidence (P13)

P13 tested the sample-size hypothesis before changing the algorithm again. It doubled gradient generation to 32
rows and search to 32 rows, yielding 16 rows per gradient fold and 16 rows per search fold. It retained 30 separate
confirmation rows and moved untouched evaluation to a disjoint 64-row block. The codec, packed prefix, YAQA factors,
ranks, alphas, candidate count, and acceptance rules remained unchanged.

The gradient folds now contained 6,866 and 4,529 valid tokens. Consensus retained only 12 signed modes. Four changed
serialized candidates reached the bounded shortlist:

- two were rejected before replay because their realized first-order product changed sign across gradient folds;
- the two all-gradient-fold candidates both failed nonlinear search replay;
- their worst-fold KL ratios were `1.0009404` and `1.0013051`, corresponding to regressions of `+0.0940%` and
  `+0.1305%`;
- no candidate was selected, confirmation was not invoked, and the exact YAQA rollback was serialized.

This invalidates the interpretation that P12 merely needed modestly more rows. The small P11/P12 search gains were
selection effects that disappeared under the doubled evidence budget. Cross-fitted signed-spectrum P32 refinement
is therefore **rejected for promotion and should not receive a wider rank/alpha sweep**. Keep the implementation as
a default-off research diagnostic, but move the next accuracy effort to a materially different degree of freedom.

The most credible next target is complete-module propagation-aware bank-family selection. It should independently
encode the canonical V2+YAQA oracle and each legal B2 alternate-family artifact, replay those complete serialized
modules through the live quantized prefix, and choose by cross-fitted final-logit loss before independent
confirmation. This avoids the fragile one-segment spectral projection while testing whether B2's coarse family
diversity supplies a stable downstream error direction.

## Artifacts

- `artifacts/qvq_p4_signed_fixed_boundary_gate/layer2_q_w2_rows1826_1954_with_yaqa_diagnostics.json`
- `artifacts/qvq_p4_signed_fixed_boundary_gate/layer2_q_w2_rows1826_1954_with_yaqa_diagnostics.safetensors`
- `artifacts/qvq_p4_gradient_ranked_fixed_boundary_gate/layer2_q_w2_rows1954_2048.json`
- `artifacts/qvq_p4_gradient_ranked_fixed_boundary_gate/layer2_q_w2_rows1970_2048_nogradient.json`
- `artifacts/qvq_p10_propagation_shaped_spectral/layer2_q_w2_rows1954_2048.json`
- `artifacts/qvq_p11_realized_gradient_gate/layer2_q_w2_rows1954_2048.json`
- `artifacts/qvq_p12_crossfit_gradient_gate/layer2_q_w2_rows1954_2048.json`
- `artifacts/qvq_p13_crossfit_expanded_gate/layer2_q_w2_expanded.json`
