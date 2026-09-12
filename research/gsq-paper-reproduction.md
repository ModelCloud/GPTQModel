# Paper-faithful scalar GSQ implementation work

User requested the full training procedure, not only the existing independent
projection fitter. This is active implementation scope; no reproduction claim yet.

Reference: [GSQ v2](https://arxiv.org/html/2604.18556v2), especially Sections 3–4
and Appendices B, C, G. Author implementation freshly fetched at
`03fc16484c369e3127225615d5e03e8d3a6043e3` from
https://github.com/IST-DASLab/GSQ. Local reference checkout:
`/tmp/gsq-author-reproduction` (reference only, not a runtime dependency).

### W4 NM128 training-history diagnostic

The completed five-epoch, batch-64 W4 run recorded 2,000 updates for each
Q/K projection but only ten updates for each attention and MLP stage
(128 documents / 64 documents per batch × five epochs). This is an observed
training-budget imbalance, not an established explanation for the Platinum
regression. All 32 Q/K histories finish above their observed minimum stochastic
loss. Neither that fact nor the lower final-versus-initial logged losses proves
that an earlier checkpoint would have better hard-weight or held-out accuracy:
sampling temperature and multiplier change, and attention/MLP batches shuffle.

The source-bound 64-stage audit is
`artifacts/gsq-staged/full-model-w4-nm128-signed-comparison-v1/training-history-audit.json`.
The existing records lack matched initializer-versus-final hard-weight objectives,
per-stage held-out outputs, and relaxed-versus-hard comparisons. These measurements
are needed before attributing the regression to overfitting, accumulated layer
error, hard assignment, or insufficient attention/MLP optimization. Do not present
decreasing logged training losses as proof of improvement over the GPTQ baseline.

Implementation acceptance requirements:

- Separate explicit training mode; existing GSQ-inspired behavior and disabled
  defaults remain identifiable in serialized configuration and experiment reports.
- Match author scalar grid parameterization, initialization, local-shift rules,
  trainable group scales, Lion updates and parameter groups, temperature and
  logit-multiplier schedules, gradient accumulation, and hard export semantics.
- Implement within-block staging: Q/K linear objectives, V/O attention objective,
  and MLP block reconstruction, preserving teacher/student inputs, masks, rotary
  embeddings, and nonlinear gate/up interaction. Independent Hessian fitting is
  not a substitute for the attention or block objectives.
- Establish numerical parity against pinned author code for forward relaxation,
  gradients, optimizer updates, schedule endpoints, and discrete export before
  making a reproduction claim.
- Start real validation on Llama 3.2 1B, seed 7, using locked disjoint calibration
  and held-out data and matched initializer controls. Preserve requested F6/N
  campaign controls where applicable; label model/protocol deviations from the
  paper explicitly. Measure local losses and propagated KL/MSE/Top-K separately.
- Verify GPTQ packing/reload/inference first, then compatible scalar deployment
  adapters. AWQ transform/scales must preserve the fitted coordinate system.
  QQQ/FP8 activation quantization requires matching training/replay semantics;
  compatibility with weight storage alone is insufficient.
- QVQ whole-tile categorical search remains an adaptation, not scalar GSQ.
  MXFP4/GGUF/nonuniform adapters require their own parameterization and export
  checks before being called compatible with this training path.

Initial source audit: author `src/trainer.py` constructs Lion with configured
betas. `configs/local/verify_llama32_1b.yaml` is a one-epoch smoke configuration,
not the paper's full training budget; do not use it as publication parity evidence.

## First real staged smoke run and diagnosed mismatch

Artifact: `artifacts/gsq-staged/llama-block0-w4-seed7-v1`.
Real Llama 3.2 1B Instruct block 0, W4/group128, FP32, seed7,
16 unweighted calibration documents, 32 disjoint held-out documents, two epochs.
Held-out block MSE: GPTQ 1.9465117556922764e-5; staged
5.6991410334775954e-5 (+192.787%). No promotion or paper-parity claim.

Source audit after the run found the driver used Lion beta2=.99 rather than the
pinned author `src/config.py` default .95. The driver is corrected for subsequent
runs; the existing artifact remains unchanged and must retain its executed-source
identity. This mismatch is not proven to explain the regression.

Saved state has one or more negative O-projection scales (minimum
-7.83326686359942e-6). Author scales are unconstrained parameters; silently
clamping them would change the method. Signed-scale export compatibility requires
an explicit decoder/packing audit. Stage endpoint losses use different shuffled
documents, so their differences are not matched evidence of convergence.

## Q/K training procedure gap (pinned source audit)

`src/prior/gptq.py` uses a separate 2,000-update Q/K training loop, with constant
parameter-group learning rates (no trainer LR scheduler in that loop). It linearly
schedules temperature and multiplier over those 2,000 updates. Our initial driver
instead treated Q/K as ordinary shuffled-input stages with 32 updates. This is a
material remaining reproduction gap, not an equivalent implementation.

Author `src/config.py` main defaults: temperature [2,.05], multiplier [100,500],
assignment LR .0001, scale LR .00005, ten attention/MLP epochs, cosine LR decay,
minimum fraction .1, Lion betas [.9,.95]. The first experiment used the local smoke
sampling/LR ranges, two epochs, linear decay to zero, and FP32. Every such choice
must remain visible in its report; default-source settings are not automatically
the exact paper Appendix G configuration.

Pinned Q/K source calls `GumbelQuantizerInt` for both W3 and W4 without forwarding
`bits`; that constructor defaults to W3. Do not silently reproduce this apparent
W4 source inconsistency as the paper specification. Resolve paper-vs-code behavior
and record an explicit interpretation before claiming W4 parity.

## Late MLP initializer ordering

Pinned author `main.py` lines 269–286 trains attention before calling the MLP
initializer. The initial staged experiments instead initialized all projections
against the original attention. The driver now refreshes MLP-only GPTQ inputs and
seeds after attention fitting; records carry `initializer_timing` and initializer
metadata. Earlier saved artifacts remain evidence for the earlier ordering and
must not be relabeled as having this correction. The real W2 run with a 43.62%
local-MSE improvement also predates this correction.

## Explicit configuration and block lifecycle

`GSQTrainingConfig` now identifies the staged Lion procedure independently of
`GSQConfig`'s projection-level refinement. `quantize_llama_gsq_block` consumes
that configuration, initializes GPTQ, performs late-MLP staged training when
enabled, and optionally packs the resulting block. Defaults bypass GSQ training.
The experiment driver uses the same entry point and records its effective
configuration. Full-model public capture/replay and checkpoint integration remain
required; this block API must not be presented as completing them.


## CUDA repeatability control

Real W2 full-size repeats exposed scale-gradient scatter-add nondeterminism even
with seed7, identical initializers and schedules. The isolated reduction changed
up to 25,674 elements by at most 1.67e-6 across 20 identical replays. Enabling
PyTorch deterministic algorithms made that check exact. Two complete staged
runs with deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8 then
matched all stage losses/schedules, learned scales, final weights and held-out
MSE exactly. This is scoped same-runtime repeatability, not cross-hardware or
full-author-training parity. Keep the failed strict parity assertion and ordinary
CUDA repeats as evidence; do not silently replace their artifacts.


## Inference capture to staged autograd

The shared lifecycle wraps dense projections in HookedLinear, whose forward
runs in inference mode. Captured hidden/rotary tensors can also be inference
tensors. The staged bridge must copy them outside inference mode and unwrap
the private block before computing attention/block gradients. It rejects online
Hadamard wrappers rather than dropping their transforms. A real HF-forward
InputCache experiment matches the complete deterministic W2 payload byte for
byte; full shared-looper dispatch and checkpoint integration remain open.

## Initializer scale orientation remains a reproduction gap

The pinned author's `src/prior/quant.py` range search evaluates **both positive
and negative group scales**, using a weight-error exponent of 2.4. With the
asymmetric signed integer endpoints (for example -2 through 1 at W2), changing
scale sign changes which side receives the extra endpoint. The current staged
adapter instead inherits this repository's activation-weighted scale search
with exponent 2.0 and requires positive initializer scales. Jointly learned
scales can subsequently become negative, but that does not reproduce the prior.

An independent CPU check on real Llama block-0 Q-projection weights (256 rows,
all 2,048 columns, group128) confirmed that merely setting repository MSE search
to exponent 2.4 does not match the author: W2/W3/W4 scale mismatches numbered
2,581/2,272/2,239 out of 4,096 groups. This is a range-search arithmetic audit,
not a model-quality comparison. Raw results are in
`artifacts/gsq-staged/initializer-scale-parity-v1/report.json`.

The 512-document experiments deliberately retain the same repository initializer
in both arms; they must not be described as author-initializer parity. A clean
reproduction needs the signed-scale prior, nonzero signed initializer support,
and matched full GPTQ trajectories, in addition to the remaining batching and
precision requirements. Do not change a running experiment's implementation or
rebind its source snapshots after discovering this difference.

`gsq_initialization.signed_scalar_range_search` now implements that signed
range-search primitive separately. Its CPU audit matches every scale and
reconstructed weight exactly against the pinned author on all 4,096 real groups
at each of W2/W3/W4. The author selects negative scales in 2,080/2,036/2,027 groups,
respectively. See `artifacts/gsq-staged/initializer-scale-parity-signed-v3`.
Contract tests cover signed endpoints, exact zero teachers and rejected inputs.
The optional `GSQTrainingConfig(initializer='gptq_signed')` now uses this prior
inside repository GPTQ and in late MLP initialization. Staged training accepts
finite nonzero signed initializer scales. The ordinary default remains `gptq`.
The training `enabled` flag is independent: disabling training with the signed
initializer selected constructs the matched signed-GPTQ control.

On a real block-0 Q-projection slice (32 outputs, 256 inputs), with actual
embedding/RMSNorm inputs from all 512 NM documents, CPU W2/W3/W4 integer
assignments match the author exactly. Maximum weight differences are
7.15e-7/1.34e-7/4.47e-8; the whole prior is not bitwise identical. See
`artifacts/gsq-staged/signed-gptq-prior-nm512-v2`. A 69-test CPU suite passed;
the subsequently expanded five-test model suite verifies signed-prior public
checkpoint export/reload and preserved configuration. These include synthetic
lifecycle fixtures, not complete-model quality evidence. CUDA prior validation
is queued. Full-model validation of the signed prior, paper batching/precision
and remaining lifecycle integration are still required. The running NM512
experiments use their previously recorded repository initializer.

### NM512 batching audit (2026-09-09)

The live NM512 run uses 512 one-document optimizer batches per epoch for
attention and MLP, hence 5,120 updates per stage over ten epochs. Increasing
calibration from 16 to 512 did not reproduce the author batching schedule.
At pinned author commit `03fc16484c369e3127225615d5e03e8d3a6043e3`,
`src/config.py` defaults to global batch 64 and device microbatch 16;
`src/trainer.py:253` resamples quantized weights inside each microbatch and
accumulates four equally weighted microbatch losses for a full single-device
batch. With 512 examples and ten epochs, that recipe would make 80 optimizer
updates per attention/MLP stage, rather than 5,120. Q/K's separate 2,000-step
fit is not included in that comparison.

The current `train_stage_update` already resamples each microbatch, but
`fit_llama_stages` constructs singleton batches. Wiring only gradient
accumulation over 64 singleton documents would still draw 64 quantized weights
per update instead of the author's four. Reproduction therefore needs actual
16-document forward microbatches and an explicit treatment of variable-length
NM inputs, padding and reconstructed-element weighting. The live NM512 data
are native documents of 59–1,490 tokens, not fixed-length concatenated windows.
These are implementation requirements, not measured accuracy effects; the
running experiment retains its recorded recipe.

The staged API now exposes `GSQTrainingConfig(batch_size=64,
microbatch_size=16)` and the full-model validation CLI exposes matching flags.
Defaults remain one document per update/forward. The new eager-Llama path
right-pads captured documents, preserves valid-token additive attention masks
and rotary tensors, and excludes padding from reconstruction MSE. Each actual
forward microbatch samples fresh quantized weights; valid reconstructed-element
counts weight accumulated losses, including partial batches. For equal-length
full batches this agrees with equal microbatch weighting; for variable-length
NM documents it is an explicitly token-weighted extension of the author setup.

The staged GPTQ initializer now selects an explicit sequence-count Hessian
normalization matching the author path: raw token Grams are materialized as
`H = (2 / N_sequences) * sum(X.T @ X)`. The previous repository default could
fall back to token-count normalization when bucket boundaries were absent.
Existing quality artifacts retain their original source snapshot and must be
rerun to measure this correction. In the current signed GPTQ path this is a
global Hessian rescaling that cancels under proportional damping in the ideal
code trajectory; it is required for reference parity but is not a demonstrated
quality fix. `GSQTrainingConfig(initializer='rtn')` also provides the symmetric
RTN control required for the matched RTN +/- GSQ matrix.

CPU checks exercised variable-length decoder output equivalence with explicit
and absent eager masks, partial optimizer batches, staged fitting, and packing.
`staged-batching-tests-v2.log` records 27 passing configuration/lifecycle tests;
`staged-batching-tests-v3.log` records three passing final mask/batching cases.
These overlapping suites are correctness evidence, not model-quality evidence.
GPU batching, full-model memory behavior, and matched task scores remain pending.
The existing NM512 run loaded the earlier singleton recipe and is unaffected.

### Precision audit follow-up (2026-09-09)

At the pinned author revision, `src/config.py` defaults both model dtype and
assignment-logit dtype to BF16. `src/models/base.py:444` applies its configured
`MSELoss` directly to teacher/student outputs, without an explicit FP32 cast
in that function; actual operator execution remains dependent on the runtime
and any enclosing precision context. `src/trainer.py:40` permits an explicit
FP32 assignment-logit override. This is not evidence that the author's complete
training executes in FP32.

The active NM512 full-model run and queued signed/batched block run explicitly
load FP32 model weights and execute FP32 reconstruction. Their records must
retain that scope. Initializer and batching corrections alone therefore do not
establish reproduction of the author's default BF16 trajectory. A matched
precision experiment remains necessary; do not relabel these existing runs.


### Completed NM128 five-epoch evidence and W4 retest (2026-09-09)

The preceding live/queued NM512 descriptions are historical. The user stopped
the unfinished NM512 singleton run and selected 128 documents and five epochs.
Both full W2/group128 signed-prior arms subsequently completed on the real GPU,
with batch64/micro16, FP32 training and FP16 export. All 32 held-out document
logits match exactly through public checkpoint reload. This verifies the exercised
GPU batching and full-model export path, not author-procedure parity.

The locked 128 documents contain 47,005 tokens. Full Platinum evaluation uses
all 1,209 questions, eight-shot CoT, greedy 256-token generation, seed 7 and
FP16 eager Torch. The strict analyzer verified exact prompts, targets and token
IDs across dense, baseline and GSQ arms, plus source/settings and calibration
bindings. Dense scored 593/1,209; signed GPTQ scored 0/1,209; staged GSQ scored
20/1,209. The paired task gain is 1.6543 percentage points (95% bootstrap interval
0.9926 to 2.3987). Invalid numeric answers fell from 1,170 to 482. Absolute
quantized accuracy remains severely degraded. Held-out final-logit KL, MSE and
Top-1/5/10 all regress with GSQ, with paired intervals excluding zero. This mixed
metric evidence does not support a recovery/default-promotion claim.

The full five-epoch GSQ run took 879.96 seconds. Synchronized wall time for all
training stages totaled 573.68 seconds (Q 216.05, K 74.46, attention 59.47,
MLP 223.71). These measurements do not establish a matched speedup or identify
kernel bottlenecks. Q/K retain 2,000 updates per projection; attention and MLP
each use ten optimizer updates per block (two global batches times five epochs).

The user requested a W4 GPTQ retest after seeing the W2 failure. Matched W4
baseline and GSQ jobs now retain the same 128 documents, signed initialization,
five epochs, seed, grouping and precision. Their results are pending. The public
BaseQModel.quantize staged dispatch, additional compatible methods and full
paper precision/RNG/calibration parity remain unfinished requirements.

Evidence: `artifacts/gsq-staged/full-model-w2-nm128-signed-comparison-v1` and
`artifacts/gsq-staged/gsm8k-platinum-nm128-signed-v1`; W4 protocol:
`artifacts/gsq-staged/gsm8k-platinum-w4-nm128-signed-v1/protocol.json`.


### Historical public dispatch boundary audit

At the original audit, `GPTQConfig.gsq` and `AWQConfig.gsq` normalized only
`GSQConfig`, the independent-projection adapter configuration. The staged
`GSQTrainingConfig` is consumed by `quantize_llama_gsq_model`; it is not consumed
by `BaseQModel.quantize`. The full-model experiments therefore prove the dedicated
trainer and public checkpoint loader, not public staged quantization dispatch.

Integration must preserve the existing optional adapter's configuration semantics,
make staged versus projection fitting explicit, and serialize the effective staged
recipe. It must also route normal calibration preparation, materialized placement,
layer scope, backend selection and packing through validated lifecycle boundaries.
The present dedicated exporter only supports complete uniform Llama W2/W3/W4
models; passing arbitrary dynamic scopes through it would be an unsupported shortcut.
AWQ compatibility requires preserving its transformed weights and activation replay,
not silently substituting the GPTQ signed prior. These are outstanding implementation
and lifecycle tests, not restrictions proving other methods scientifically incompatible.

An affine staged parameterization is now available in `gsq_training_affine.py`.
It accepts existing integer codes, positive initial scales, and fixed per-group
integer zero points, using `(code - zero) * scale` throughout the relaxation.
W2 full-grid and W3/W4 local-shift candidates retain the initializer's legal code
bounds. Synthetic correctness checks cover boundary zero points, hard codes,
forward values, and assignment/scale gradients against a direct softmax formula.
This is not yet AWQ lifecycle support or real-model quality evidence. Integration
must capture the teacher after AWQ scaling and before clipping, retain the AWQ
initializer rather than rerun GPTQ, replay complete attention/block objectives,
and validate learned scales against each actual AWQ packing format. Scales remain
unconstrained during training; a positive initializer alone does not prove that
the learned checkpoint satisfies a positive-scale deployment requirement.

The shared Llama stage fitter now accepts explicit affine initializers. It runs
the same Q/K quadratic, V/O attention, and MLP block objectives while retaining
per-group zero points and hard codes in the returned stage records. This mode
requires `reinitialize_mlp=False`: silently running the GPTQ prior after attention
would replace the supplied AWQ initializer. A synthetic full-block correctness
test forbids any GPTQ initializer call, checks all seven hard affine weights and
zero points, and verifies that the supplied teacher remains unchanged. Separate
W4 AWQ GEMM tests force changed codes and scales through packing, disk state
serialization, and reload for FP16/BF16 scale storage. Native AWQ transform/capture,
processor installation/replay and real-model quality validation remain outstanding.

The subsequent public integration adds separate `gsq_training` configuration for
GPTQ and AWQ. AWQ explicitly requires its own initializer (`initializer='awq'`),
retains transforms/clipping, captures the scaled pre-clip teacher, and replaces
weights/scales/zeros together after staged fitting. Initial support is uniform
W4 GEMM with complete eager Llama blocks. CPU FP16 public quantize/save/reload
passes exact logits on a tiny two-layer correctness fixture; this is not
real-model AWQ recovery evidence. FP16 testing also exposed the signed fitter's
`-1e9` invalid-candidate mask overflow; negative infinity now masks invalid
candidates without a finite value outside FP16 range. Real-model AWQ quality,
additional packing formats and full paper-procedure parity remain unverified.
