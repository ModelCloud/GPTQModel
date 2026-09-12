# PR 212 GSQ math and protocol audit

## Verdict

- The measured W4 GSQ arms hurt their matched staged signed-GPTQ control:
  `38.05% -> 34.49%` with Lion and `38.05% -> 33.33%` with AdamW. Held-out
  final-logit KL, MSE, and Top-1/5/10 agreement also regress.
- This does not establish that correctly trained GSQ intrinsically hurts GPTQ.
  The scalar forward/backward, candidate geometry, masking, signed scales,
  hard export, Lion trajectory, Q/K objective, and save/reload path have no
  confirmed mathematical defect.
- The strongest current causal candidate is protocol under-training and
  mismatch, not a sign error or reversed teacher/student objective.
- A matched `GPTQ +/- GSQ` and `RTN +/- GSQ` matrix is required. The RTN arms
  distinguish GSQ's contribution from GPTQ Hessian compensation.
- The staged GPTQ initializer had one concrete paper deviation: the repository
  default normalized its Hessian by total calibration tokens after the
  bucketed mode fell back to disabled, while the author GPTQ normalizes raw
  token Grams by calibration sequence count. The staged path now selects an
  explicit sequence-count mode; the earlier quality numbers remain pre-fix
  results and need a rerun. On the current signed prior with proportional
  damping, this is a global Hessian rescaling and a small CPU comparison left
  the GPTQ codes and scales unchanged, so it is a parity correction rather
  than a confirmed explanation for the quality regression.
- GSQ must remain experimental and disabled by default.

## Math audit

QVQ matches the pinned author quantizers for W2, W3, and W4 when noise and
precision are matched:

- forward reconstruction is exact;
- assignment-logit and scale gradients are exact;
- twenty-step Lion trajectories are exact;
- hard exported weights are exact.

The stored author-parity artifacts report zero maximum error for all tested
rate, weight-dtype, and logit-dtype combinations. A focused run of the GSQ
math, initializer, staging, packing, reload, and evaluation-setting tests
passes all 74 cases.

The audited QVQ equations are consistent with the paper and reference code:

1. `softmax((multiplier * logits + Gumbel noise) / temperature)`.
2. W2 selects from the full signed grid `[-2, -1, 0, 1]`.
3. W3/W4 selects a local shift from `[-2, -1, 0, 1, 2]` around the initialized
   integer code, with out-of-grid shifts masked.
4. Hard export selects the maximum valid learned logit and multiplies by the
   learned group scale.
5. Negative group scales remain legal. Clamping them would change the signed
   method.
6. The scale derivative sums `output_gradient * expected_code` over columns
   sharing the scale.
7. The assignment derivative is the standard softmax Jacobian-vector product,
   multiplied by `multiplier / temperature`.
8. The Q/K loss using a lower Cholesky factor is equivalent to the author's
   quadratic Hessian/Gram reconstruction objective.
9. Q/K independent optimization, V/O joint attention reconstruction, MLP
   joint full-block reconstruction, prefix propagation, packing, and exact
   reload are structurally aligned.

The initial older BF16 parity artifact had small gradient differences caused by
where dtype conversion occurred. The later explicit-backward and mixed-logit
artifacts remove that difference and report exact parity.

## Material protocol deviations

### Attention and MLP anneal over only ten updates

The W4 Lion evidence uses:

- 128 calibration documents;
- batch size 64;
- five epochs;
- two attention/MLP updates per epoch;
- 10 total attention updates and 10 total MLP updates per block.

The paper's Llama recipe uses:

- 4,096 sequences;
- length 4,096;
- batch size 64;
- 20 epochs;
- 64 updates per epoch;
- 1,280 total attention and MLP updates per block.

The current run therefore uses 128 times fewer V/O and MLP updates. The
temperature and logit multiplier traverse their complete schedules in only ten
steps. Q/K separately receives 2,000 updates, so the stage budgets are also
strongly imbalanced.

The AdamW ten-epoch rerun has 20 updates, still 64 times fewer than the paper's
Llama budget. Changing the optimizer did not address the dominant budget gap.

### Calibration-token exposure is much smaller

QVQ records 47,005 tokens per epoch. The paper's fixed-length Llama recipe uses
16,777,216 tokens per epoch.

- Per epoch, the current run uses about 357 times fewer tokens.
- Across the respective five versus twenty epochs, it uses about 1,428 times
  fewer token presentations.

Repeating the existing 128 documents until the update count reaches 1,280 would
not be paper parity. It would preserve the data-diversity mismatch and could
overfit.

### The tested operating point is not reported in the paper

- PR 212 tests Llama-3.2-1B at W4.
- The paper's scalar Llama headline experiments use Llama-3.1-8B/70B at W2/W3.
- W4 is not a reported Llama headline setting.
- The paper uses FineWeb-Edu fixed-length sequences. PR 212 uses variable-length
  task-adjacent documents with token-weighted padding masks.

The W4 experiment is useful integration evidence, but it is not a paper
reproduction.

### The paper's GPTQ comparison is not an incremental GSQ ablation

The paper gives GSQ 4,096 calibration sequences and blockwise optimization,
while its GPTQ baseline follows the usual 512-sample baseline recipe. Its GPTQ
baseline may also use asymmetric per-group zero-points, while GSQ uses a
symmetric scalar representation.

The paper's headline `GSQ > GPTQ` result therefore does not prove that applying
GSQ to a fixed, matched GPTQ checkpoint improves it. PR 212's signed-GPTQ
control is more causal for that question, but uses a much smaller GSQ training
protocol.

### The main W2 paper result includes an extra scale-only stage

The paper applies one end-to-end scale-only epoch for its 2-bit Llama result.
That stage is absent from the current evidence. It is relevant to a W2
reproduction, but should not be added to a W4 test and called paper parity.

### Hard assignments are not monitored during training

The author trainer evaluates both soft and hard validation loss after each
epoch. The current artifact records stochastic soft training loss. A decreasing
soft loss can coexist with worse hard assignments after annealing and export.

Hard held-out stage loss should be recorded before training and after every
epoch for Q, K, V/O, and MLP. This is primarily a diagnostic gap; the pinned
author code does not appear to use the metric for rollback.

### The pinned author W4 Q/K path is ambiguous

The pinned author GPTQ Q/K construction creates `GumbelQuantizerInt(...)`
without passing `bits=gsq_bits`; the class defaults to `bits=3`. The ordinary
trainer construction does pass the requested bit width.

QVQ explicitly applies W4 bounds and is consistent with the paper's general
signed-grid definition. Literal parity with this apparent author-code omission
would not be a justified QVQ fix, especially because the paper does not report
a W4 Llama result.

### Staged GPTQ Hessian alignment

The author initializer accumulates each captured activation matrix as a raw
token Gram and materializes

`H = (2 / N_sequences) * sum(X.T @ X)`.

The staged QVQ initializer previously constructed the repository default
bucketed length-aware configuration without bucket boundaries. That configuration
was disabled by GPTQ, leaving `H = (2 / N_tokens) * sum(X.T @ X)`. With variable
length documents, this changes the absolute Hessian scale. `LengthAwareMode.SEQUENCE_COUNT`
now preserves the raw Gram and applies the author sequence-count normalization
only to the staged initializer. Because GPTQ uses proportional damping and the
signed prior does not use Hessian-weighted range search, the scale cancels in
the ideal code trajectory; it can still matter to scale-sensitive objectives,
conditioning, and finite-precision behavior. A focused variable-length
regression test covers the exact formula, and a small signed-W4 CPU comparison
matched the author trajectory's scales, codes, and quantized weights.

## Required causal matrix

Use one immutable dense checkpoint and one locked calibration/evaluation
manifest for all causal arms:

| Prior | GSQ off | GSQ on | Purpose |
|---|---|---|---|
| Symmetric RTN | RTN control | RTN + GSQ | Tests whether GSQ improves simple rounding without Hessian compensation |
| Matched signed GPTQ | GPTQ control | GPTQ + GSQ | Tests the incremental effect of GSQ after the same GPTQ initializer |

Add dense BF16/FP16 and package-default true-sequential GPTQ as non-causal
reference arms. Package-default GPTQ must not replace the matched signed-GPTQ
control because different zero-point freedom and initialization confound the
GSQ delta.

The public staged configuration now accepts `rtn` and uses the native symmetric
RTN quantizer for that initializer. This enables the strict four-arm experiment
matrix without changing the default GPTQ path. RTN is an experiment control,
not a fix for a confirmed GSQ equation bug.

## Experiment order

1. Run a W3 stage gate because it is within the paper's evaluated scalar regime
   and is less likely than W2 to begin from a collapsed 1B control.
2. Use fixed-length FineWeb-Edu calibration and disjoint held-out sequences.
3. Record hard held-out Q/K, attention, and block losses before GSQ and after
   each epoch, plus propagated final-logit KL/MSE/Top-K.
4. Compare an update-count ladder before committing to the complete 1,280-update
   full-model recipe.
5. Run a full model only if hard held-out and propagated final-logit metrics do
   not regress.
6. Reproduce W2 after W3 protocol validation; include the paper's scale-only
   epoch and use Llama-3.1-8B as the actual reproduction target. Llama-3.2-1B
   remains a debugging proxy.

Every arm must hold fixed model revision, calibration rows/order/token IDs,
tokenizer, sequence construction, bits, group size, symmetry, activation
ordering, seed policy, precision, backend, software SHA, packing/reload path,
and evaluator. Preserve raw per-example outputs and paired uncertainty.

## Bottom line

Current QVQ GSQ hurts the matched staged GPTQ control at W4, and the negative
result is statistically and numerically credible. The audit does not identify
a basic GSQ math defect. The current run anneals the most important joint stages
over far too few updates and far less data to test the paper's claim. RTN with
and without GSQ is necessary, together with matched GPTQ with and without GSQ,
to identify whether the degradation comes from the GSQ optimizer itself or its
interaction with the GPTQ initializer.
