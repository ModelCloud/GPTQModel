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

## Complete serialized family selection (P14)

P14 tested that different degree of freedom. It independently encoded four complete module artifacts for
`model.layers.2.self_attn.q_proj` at W2:

1. canonical V2+YAQA represented by all-zero B2 selectors;
2. V2B2-P32+YAQA with fixed alternate family 1;
3. V2B2-P32+YAQA with fixed alternate family 2;
4. V2B2-P32+YAQA with fixed alternate family 3.

Every candidate was packed and reconstructed exactly before use (`max_abs_error = 0`). More importantly, the final
gate installed each candidate as a native packed `QVQLinear` during search and confirmation. This exposed the small
FP16 transform/backend drift that a dense reconstructed-weight replay does not see. The packed path changed family
2 from a near-tie (`0.999960` through a dense reconstructed weight) to a regression (`1.003042`), validating the
production-path requirement.

The expanded P13 data contract was retained: two 16-row search folds, a disjoint 30-row confirmation split, and a
disjoint 64-row untouched evaluation split. The complete packed search was:

| Family | Worst-fold final-KL ratio | Selector nonzero fraction | Decision |
|---:|---:|---:|---|
| canonical V2 | 1.000000 | 0.00% | rollback oracle |
| 1 | **0.962960** | 49.97% | confirm |
| 2 | 1.003042 | 49.80% | reject |
| 3 | 1.024526 | 50.13% | reject |

Family 1 improved both search folds by at least `3.70%`. It then passed the independent packed confirmation gate:

| Metric | 30-row confirmation delta versus packed V2+YAQA |
|---|---:|
| Final KL | -0.5892% |
| JSD | -0.6643% |
| Top-1 | -0.0772 pp |
| Top-5 overlap | +0.0540 pp |
| Top-10 overlap | +0.0772 pp |

The small confirmation Top-1 loss is noise-scale relative to the token count and is outweighed by consistent primary
distribution and Top-5/10 improvements. It reversed on the larger untouched evaluation split, where every measured
endpoint improved:

| Metric | 64-row untouched evaluation delta versus packed V2+YAQA |
|---|---:|
| Final KL | **-2.0749%** |
| JSD | **-2.0359%** |
| Top-1 | **+0.0395 pp** |
| Top-5 overlap | **+0.0582 pp** |
| Top-10 overlap | **+0.0309 pp** |

P14 is therefore a **validated positive direction**, unlike P10-P13. It is not yet an unconditional default: this
is one module, rate, and seed, and the winning family must be selected rather than hard-coded. The next gate is a
replication across additional Q/K/V/O modules and at least one independent YAQA/RHT seed. If that confirms the
effect, integrate complete packed family replay as an optional propagation-aware quantization stage, retaining
canonical V2+YAQA as the atomic fallback.

## Sequential K-projection replication (P15)

P15 installed P14's accepted layer-2 Q artifact into the live prefix and repeated the complete packed family gate
for `model.layers.2.self_attn.k_proj`. No alternate improved both search folds:

| Family | Worst-fold final-KL ratio | Selector nonzero fraction | Decision |
|---:|---:|---:|---|
| canonical V2 | 1.000000 | 0.00% | selected rollback |
| 1 | 1.022334 | 49.67% | reject |
| 2 | 1.005045 | 50.21% | reject |
| 3 | 1.021583 | 50.00% | reject |

The exact packed V2+YAQA oracle was therefore retained without confirmation. This is not a failure of the P14
strategy: it is evidence that useful bank-family direction is module- and prefix-dependent. Family 1 was strongly
positive for Q and strongly negative for K under the same rate, YAQA factors, prompts, and seed. A global family ID
or local-proxy choice is therefore unsupported. The production design must remain optional per module, score the
actual packed runtime, and preserve atomic canonical rollback.

## Sequential V-projection replication (P16)

P16 added the canonical K rollback to the live prefix and repeated the packed gate for layer-2 `v_proj`. Again, no
alternate family improved both search folds:

| Family | Worst-fold final-KL ratio | Selector nonzero fraction | Decision |
|---:|---:|---:|---|
| canonical V2 | 1.000000 | 0.00% | selected rollback |
| 1 | 1.028622 | 49.59% | reject |
| 2 | 1.024808 | 49.73% | reject |
| 3 | 1.018731 | 49.76% | reject |

The alternative banks were used heavily, but the added discrete capacity produced the wrong propagated directions.
The optimized negative path skipped duplicate selected-artifact endpoint passes and completed in `95.60 s`. P15 and
P16 together reinforce that occupancy and local capacity are not promotion evidence; only the live propagated gate
can decide whether a family is useful for a specific module.

## Sequential O-projection replication (P17)

P17 completed the layer-2 attention quartet by installing the selected Q and canonical K/V artifacts before
quantizing `o_proj`. Family 3 had a strong two-fold search result (`0.955928`, at least `4.41%` better in each fold),
while family 1 was a sub-threshold near-tie and families 2/3 otherwise showed the same high selector occupancy as
earlier modules. The independent confirmation split reversed the selected family:

| Metric | Family 3 versus packed canonical V2+YAQA |
|---|---:|
| Confirmation final KL | **+2.0130%** |
| Confirmation JSD | **+2.0394%** |
| Confirmation Top-1 | +0.0096 pp |
| Confirmation Top-5 overlap | +0.0328 pp |
| Confirmation Top-10 overlap | -0.0347 pp |

The material KL/JSD regression blocked the candidate and the packed canonical artifact was serialized. The complete
quartet result is therefore: Q validated positive; K/V rejected during search; O rejected during confirmation.
Complete-family replay is a useful candidate mechanism, but its search estimator can still overfit and every module
requires independent confirmation. The next decisive test is Q under another YAQA/RHT seed; no production lifecycle
promotion should precede that seed-stability gate.

## Q-projection codec-seed replication (P18)

P18 repeated P14's layer-2 Q gate with codec/RHT seed 0 while holding the 512-row YAQA factor estimate and all prompt
splits fixed. Family 1 again won. Its two-fold search margin was smaller (`0.995143`, at least `0.486%` better in both
folds) but remained above the locked `0.1%` minimum. Family 2 regressed; family 3 improved only `0.050%` in its worst
fold and was excluded as sub-threshold.

Family 1 passed packed confirmation and untouched evaluation:

| Metric | 30-row confirmation | 64-row untouched evaluation |
|---|---:|---:|
| Final KL | **-1.4351%** | **-0.2846%** |
| JSD | **-1.4824%** | **-0.6162%** |
| Top-1 | -0.0772 pp | effectively 0 pp |
| Top-5 overlap | +0.0733 pp | +0.0395 pp |
| Top-10 overlap | +0.0550 pp | +0.0345 pp |

The confirmation Top-1 change is noise-scale and disappears on the larger untouched split, while the primary
distribution metrics and Top-5/10 improve on both. Together P14 and P18 establish codec-seed stability for Q-family
1. The magnitude is seed-dependent, so the lifecycle must still search and confirm rather than cache a role-wide
winner. This clears implementation of an **optional per-module complete-family propagation stage** with canonical
V2+YAQA rollback. It does not yet clear default enablement; a new Fisher sampling seed and broader module/depth sweep
remain required.

## Independent Fisher-seed replication (P19)

P19 recollected the same 512 full YAQA rows with Monte Carlo seed 0: 163,324 valid tokens across 512 independent
sequences. Collection completed in `138.60 s` and produced a separate 392 MiB-class factor cache. The Q-family gate
then used codec/RHT seed 1, matching P14, so only the Fisher sampling seed changed.

No alternate improved both packed search folds:

| Family | Fold 0 final KL | Fold 1 final KL | Worst-fold ratio | Decision |
|---:|---:|---:|---:|---|
| canonical V2 | 0.00193776 | 0.00281087 | 1.000000 | selected rollback |
| 1 | 0.00184288 | 0.00291467 | 1.036928 | reject |
| 2 | 0.00182964 | 0.00287259 | 1.021957 | reject |
| 3 | 0.00185431 | 0.00302310 | 1.075504 | reject |

Every family improved fold 0 but regressed fold 1. Thus P14/P18 establish codec-seed stability under one factor
estimate, but the winning direction is **not Fisher-seed stable**. Default enablement is rejected. The complete-family
stage remains safe and potentially useful only as an optional guarded search because the packed all-fold gate returns
exactly to canonical V2+YAQA when the estimator proposes a harmful family.

Before lifecycle integration, stabilize the YAQA estimator rather than hard-code family 1. The next efficient test
should combine multiple Monte Carlo samples per sequence (or average factor accumulators before factorization), then
repeat the same packed two-fold/confirmation gate. The locked family-selection threshold and canonical rollback must
remain unchanged.

## Two-seed YAQA factor ensemble (P20)

P20 averaged the seed-0 and seed-1 input/output Gram-factor estimators before factorization. Because both caches use
the same 512 sequences and normalization, this is exactly the two-Monte-Carlo-sample estimator for each factor:

\[
\bar H_I=\frac{H_I^{(0)}+H_I^{(1)}}{2},\qquad
\bar H_O=\frac{H_O^{(0)}+H_O^{(1)}}{2}.
\]

The discrete winner did not interpolate between the single-seed winners. Families 1/2 regressed, while family 3
improved both packed search folds with a worst-fold ratio of `0.982316`. This is expected: YAQA factorization,
feedback, and trellis argmins are nonlinear in factor geometry. The family ID is not the stable object; exhaustive
family search under the current estimator is.

Family 3 passed confirmation and untouched evaluation:

| Metric | 30-row confirmation | 64-row untouched evaluation |
|---|---:|---:|
| Final KL | **-4.1327%** | **-1.1657%** |
| JSD | **-4.3264%** | **-1.2478%** |
| Top-1 | +0.0096 pp | -0.0503 pp |
| Top-5 overlap | +0.0579 pp | -0.0050 pp |
| Top-10 overlap | -0.0058 pp | +0.0079 pp |

The untouched Top-1/5 movements are noise-scale, while the primary KL/JSD gains are material and repeat across both
disjoint gates. Under the repository's uncertainty-aware policy, P20 is a validated positive. The recommended
optional stage is therefore:

1. collect or combine at least two YAQA Monte Carlo samples per sequence;
2. independently encode canonical V2+YAQA and every fixed B2 family;
3. replay the actual packed modules across at least two search folds;
4. confirm the best material all-fold improvement on disjoint prompts;
5. atomically retain canonical V2+YAQA on any tie, regression, non-finite value, or callback failure.

This stage remains default-off until broader layer/module evidence exists because its cost is roughly four complete
YAQA encodes plus full-model replay. The mathematical and lifecycle direction is now validated; fixed family IDs and
single-seed factors are rejected.

## Supported fixed-family encoding contract (P21)

The complete-family harness originally forced each candidate by monkeypatching `yaqa_inner_v2b2_p32`. P21 replaces
that research shortcut with the explicit `quantize_qvq_linear(..., yaqa_v2b2_fixed_family_id=...)` contract:

- family `0` independently runs canonical V2+YAQA, emits all-zero P32 selectors, and uses a legal inactive family byte;
- families `1` through `3` independently run the requested fixed B2-P32 family under full YAQA scoring;
- the control is legal only for V2B2-P32 YAQA with `fixed_block_ldlq` family mode and full scoring;
- every candidate still passes the production planar pack/reconstruct equality check before replay or serialization.

The family-0 micro-gate was bit-exact with standalone V2+YAQA for the trellis, transformed weight, and reconstructed
weight. Families 1/2/3 preserved their requested serialized family byte and active selector payload. The focused gate
passed 22 tests: 10 new fixed-family/replay API checks and 12 existing YAQA family, fallback, serialization, and
spectral-interaction checks.

This is an integration and correctness improvement, not a promotion decision. It makes the four independent
candidates reproducible without process-global mutation. Propagated complete-family search remains explicit and
default-off under the P20 two-seed, cross-fold, disjoint-confirmation policy.

## W1.5 rate replication with the two-seed ensemble (P22)

P22 changed only the target Q projection from W2 to W1.5. It retained the P20 real Llama 3.2 1B model, packed W2
live prefix, layer-2 Q target, codec seed 1, two-seed 512-row YAQA factor ensemble, two search folds, 30-row
confirmation split, 64-row untouched evaluation split, full row lengths, and native packed MPS runtime.

| Family | Worst-fold search ratio | Nonzero selectors | Search decision |
|---:|---:|---:|---|
| canonical V2 | 1.000000 | 0.00% | rollback baseline |
| 1 | 1.006063 | 49.98% | reject |
| 2 | **0.966471** | 50.14% | provisional winner |
| 3 | 1.007704 | 50.35% | reject |

Family 2's approximately 3.35% worst-fold search improvement did not generalize to the disjoint confirmation rows:

| Confirmation metric | Family 2 versus canonical V2+YAQA |
|---|---:|
| Final KL | **+1.178% regression** |
| JSD | **+1.104% regression** |
| Top-1 agreement | +0.116 pp |
| Top-5 overlap | effectively unchanged |
| Top-10 overlap | -0.038 pp |

The KL/JSD regressions are material and directionally consistent; the mixed Top-N changes are small. The atomic gate
therefore serialized canonical family 0. The untouched evaluation measured the rollback artifact, not the rejected
family, and must not be presented as evidence for family 2.

This is a confirmed W1.5 rejection for this module and evidence contract, not a global B2-P32 rejection. It shows
that the P20 W2 gain is rate-specific and that neither high selector entropy nor cross-fold search improvement is
sufficient without disjoint confirmation. Complete-family selection remains optional and guarded; no family or rate
should be promoted from another rate's result.

## W2.5 rate replication with the two-seed ensemble (P23)

P23 repeated the P22 contract at W2.5. Family 1 was the only alternate to improve both packed search folds:

| Family | Worst-fold search ratio | Nonzero selectors | Search decision |
|---:|---:|---:|---|
| canonical V2 | 1.000000 | 0.00% | rollback baseline |
| 1 | **0.988515** | 49.89% | provisional winner |
| 2 | 1.014544 | 49.89% | reject |
| 3 | 1.005774 | 49.93% | reject |

Family 1 passed disjoint confirmation, and its small confirmation Top-N losses reversed on the larger untouched split:

| Metric | 30-row confirmation | 64-row untouched evaluation |
|---|---:|---:|
| Final KL | **-2.608%** | **-0.811%** |
| JSD | **-2.469%** | **-0.777%** |
| Top-1 agreement | -0.048 pp | +0.022 pp |
| Top-5 overlap | -0.015 pp | +0.014 pp |
| Top-10 overlap | -0.009 pp | +0.013 pp |

Under the uncertainty-aware policy this is a validated positive: the primary distribution metrics improve on both
disjoint splits, the confirmation Top-N losses are noise-scale, and every Top-N metric improves on the larger untouched
split. P20/P23 now support the optional guarded stage at W2 and W2.5 for this Q-module context. P22 still forbids
assuming the same result at W1.5, and the winning family remains rate-dependent (`3` at W2, `1` at W2.5).

## K-projection role replication at W2 (P24)

P24 returned to W2 but changed the layer-2 target from `q_proj` to `k_proj`, retaining the P20 two-seed factors,
packed prefix, seed, row splits, two-fold search, and full-horizon runtime. Every alternate regressed at least one
search fold:

| Family | Fold 0 final KL | Fold 1 final KL | Worst-fold ratio | Nonzero selectors |
|---:|---:|---:|---:|---:|
| canonical V2 | 0.00188420 | 0.00276094 | 1.000000 | 0.00% |
| 1 | 0.00190226 | 0.00279032 | 1.010641 | 49.37% |
| 2 | 0.00190319 | 0.00273569 | 1.010081 | 49.89% |
| 3 | 0.00191289 | 0.00280961 | 1.017627 | 49.92% |

The gate therefore skipped unnecessary candidate confirmation and serialized canonical V2+YAQA. The final packed
rollback evaluation was finite (`KL=0.00307153`, Top-1 `97.97%`, Top-5 `96.75%`, Top-10 `96.63%`). This is a
confirmed rejection for K/W2 under this evidence contract. It also proves that the Q/W2 positive is role-specific:
neither its winning family nor the decision to use an alternate may be transferred to K.

## V-projection role replication at W2 (P25)

P25 changed the layer-2 target to `v_proj` under the otherwise identical P24 contract. Families 1 and 3 improved
both search folds; family 1 won the minimax comparison:

| Family | Worst-fold ratio | Nonzero selectors | Search decision |
|---:|---:|---:|---|
| canonical V2 | 1.000000 | 0.00% | rollback baseline |
| 1 | **0.980618** | 49.84% | provisional winner |
| 2 | 1.033598 | 49.39% | reject |
| 3 | 0.982620 | 49.96% | eligible, not best |

Family 1 passed confirmation and improved the primary distribution metrics materially on untouched evaluation:

| Metric | 30-row confirmation | 64-row untouched evaluation |
|---|---:|---:|
| Final KL | **-4.755%** | **-3.227%** |
| JSD | **-5.116%** | **-3.500%** |
| Top-1 agreement | +0.183 pp | -0.072 pp |
| Top-5 overlap | +0.056 pp | +0.047 pp |
| Top-10 overlap | -0.020 pp | +0.051 pp |

The lone evaluation Top-1 loss is noise-scale; KL/JSD are large and consistent, Top-5 improves on both splits, and
Top-10 reverses positive on the larger split. P25 is therefore a validated positive. Combined with P20/P24, W2
complete-family selection is beneficial for the tested Q and V modules but rejected for K. The production policy
must remain module-conditional and replay-gated rather than role- or rate-hard-coded.

## O-projection role replication at W2 (P26)

P26 completed the layer-2 attention-role sweep with `o_proj`. Every alternate improved fold 0 but regressed fold 1:

| Family | Fold 0 final KL | Fold 1 final KL | Worst-fold ratio | Nonzero selectors |
|---:|---:|---:|---:|---:|
| canonical V2 | 0.00276171 | 0.00370304 | 1.000000 | 0.00% |
| 1 | 0.00265508 | 0.00373741 | 1.009282 | 49.75% |
| 2 | 0.00244199 | 0.00388863 | 1.050119 | 49.87% |
| 3 | 0.00240872 | 0.00381050 | 1.029020 | 50.02% |

The minimax gate correctly rejected cancellation across folds and skipped candidate confirmation. Canonical
V2+YAQA was serialized; its packed evaluation remained finite (`KL=0.00408568`, Top-1 `97.65%`, Top-5 `96.15%`,
Top-10 `96.06%`). P26 is a confirmed O/W2 rejection under this contract.

The complete layer-2 W2 role result is therefore:

| Role | Decision | Winning alternate |
|---|---|---:|
| Q | accept | family 3 |
| K | rollback | none |
| V | accept | family 1 |
| O | rollback | none |

This is direct evidence against a role-wide fixed bank policy. The optional algorithm must encode all candidates and
make a per-module propagated decision; a static Q/K/V/O rule is not justified from one layer and cannot replace the
search/confirmation gates.

## Conditional Q-then-V composition at W2 (P27)

P20 and P25 accepted Q and V independently against the same canonical prefix. P27 installed the exact packed P20 Q
artifact first, then repeated complete-family selection for V. This measures V's conditional marginal gain rather
than adding two standalone percentages.

Family 1 remained the V winner, and its worst-fold search ratio improved from `0.980618` standalone to `0.971632`
after Q was installed. Family 3 also remained eligible at `0.980554`; family 2 regressed at `1.026907`.

| Conditional V metric after accepted Q | 30-row confirmation | 64-row untouched evaluation |
|---|---:|---:|
| Final KL | **-4.016%** | **-4.606%** |
| JSD | **-4.340%** | **-4.878%** |
| Top-1 agreement | +0.019 pp | -0.093 pp |
| Top-5 overlap | +0.044 pp | +0.078 pp |
| Top-10 overlap | -0.001 pp | +0.029 pp |

The evaluation Top-1 loss is noise-scale, while the primary KL/JSD gains are larger than the standalone V gain and
Top-5 improves on both splits. P27 is a validated positive and demonstrates complementary Q/V error directions under
live packed execution. It also validates the intended greedy conditional algorithm: after each accepted module,
subsequent candidates must be replayed against the updated live artifact; standalone gains must not be summed or
transplanted without remeasurement.

## Conditional Q/V-then-K reevaluation at W2 (P28)

P28 installed the accepted packed Q and V artifacts, then reran the complete K family search. No K alternate
improved both folds:

| Family | Fold 0 final KL | Fold 1 final KL | Worst-fold ratio |
|---:|---:|---:|---:|
| canonical V2 | 0.00259885 | 0.00380873 | 1.000000 |
| 1 | 0.00265924 | 0.00380031 | 1.023239 |
| 2 | 0.00260823 | 0.00375113 | 1.003611 |
| 3 | 0.00261484 | 0.00393544 | 1.033269 |

Canonical K was retained before confirmation, and the packed Q/V plus canonical-K evaluation remained finite
(`KL=0.00422634`, Top-1 `97.65%`, Top-5 `96.14%`, Top-10 `96.02%`). K's rejection therefore survives the changed
live error state. In the current greedy solution, K is conditionally exhausted after Q and V and need not be retried
again unless a later accepted module or factor estimator materially changes the prefix.

## Artifacts

- `artifacts/qvq_p4_signed_fixed_boundary_gate/layer2_q_w2_rows1826_1954_with_yaqa_diagnostics.json`
- `artifacts/qvq_p4_signed_fixed_boundary_gate/layer2_q_w2_rows1826_1954_with_yaqa_diagnostics.safetensors`
- `artifacts/qvq_p4_gradient_ranked_fixed_boundary_gate/layer2_q_w2_rows1954_2048.json`
- `artifacts/qvq_p4_gradient_ranked_fixed_boundary_gate/layer2_q_w2_rows1970_2048_nogradient.json`
- `artifacts/qvq_p10_propagation_shaped_spectral/layer2_q_w2_rows1954_2048.json`
- `artifacts/qvq_p11_realized_gradient_gate/layer2_q_w2_rows1954_2048.json`
- `artifacts/qvq_p12_crossfit_gradient_gate/layer2_q_w2_rows1954_2048.json`
- `artifacts/qvq_p13_crossfit_expanded_gate/layer2_q_w2_expanded.json`
- `artifacts/qvq_p14_complete_family_gate/report_packed.json`
- `artifacts/qvq_p14_complete_family_gate/selected_packed.safetensors`
- `artifacts/qvq_p15_complete_family_k_gate/report_packed.json`
- `artifacts/qvq_p15_complete_family_k_gate/selected_packed.safetensors`
- `artifacts/qvq_p16_complete_family_v_gate/report_packed.json`
- `artifacts/qvq_p16_complete_family_v_gate/selected_packed.safetensors`
- `artifacts/qvq_p17_complete_family_o_gate/report_packed.json`
- `artifacts/qvq_p17_complete_family_o_gate/selected_packed.safetensors`
- `artifacts/qvq_p18_complete_family_q_seed0_gate/report_packed.json`
- `artifacts/qvq_p18_complete_family_q_seed0_gate/selected_packed.safetensors`
- `artifacts/qvq_p19_fisher_seed0_gate/yaqa_seed0_prepare.json`
- `artifacts/qvq_p19_fisher_seed0_gate/yaqa512_seed0_factors.pt`
- `artifacts/qvq_p19_fisher_seed0_gate/report_packed.json`
- `artifacts/qvq_p19_fisher_seed0_gate/selected_packed.safetensors`
- `artifacts/qvq_p20_fisher_ensemble_gate/yaqa_seed0_1_ensemble.json`
- `artifacts/qvq_p20_fisher_ensemble_gate/yaqa512_seed0_1_ensemble.pt`
- `artifacts/qvq_p20_fisher_ensemble_gate/report_packed.json`
- `artifacts/qvq_p20_fisher_ensemble_gate/selected_packed.safetensors`
- `artifacts/qvq_p22_w1p5_ensemble_gate/report_packed.json`
- `artifacts/qvq_p22_w1p5_ensemble_gate/selected_packed.safetensors`
- `artifacts/qvq_p23_w2p5_ensemble_gate/report_packed.json`
- `artifacts/qvq_p23_w2p5_ensemble_gate/selected_packed.safetensors`
- `artifacts/qvq_p24_k_w2_ensemble_gate/report_packed.json`
- `artifacts/qvq_p24_k_w2_ensemble_gate/selected_packed.safetensors`
- `artifacts/qvq_p25_v_w2_ensemble_gate/report_packed.json`
- `artifacts/qvq_p25_v_w2_ensemble_gate/selected_packed.safetensors`
- `artifacts/qvq_p26_o_w2_ensemble_gate/report_packed.json`
- `artifacts/qvq_p26_o_w2_ensemble_gate/selected_packed.safetensors`
- `artifacts/qvq_p27_q_then_v_w2_gate/report_packed.json`
- `artifacts/qvq_p27_q_then_v_w2_gate/selected_packed.safetensors`
- `artifacts/qvq_p28_qv_then_k_w2_gate/report_packed.json`
- `artifacts/qvq_p28_qv_then_k_w2_gate/selected_packed.safetensors`
