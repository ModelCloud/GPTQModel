<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# W4A accuracy campaign notes (appendix)

This appendix is the working log behind the accuracy claims in the
[W4A guide](gptq-w4a-gb10.md). It records every experiment that was run,
including the ones that failed, so the accepted numbers can be audited. It is
not required reading for using the feature.

**Current status.** The accepted Llama 3.2 1B configuration is GPTQ INT4
weights with NVFP4 MLP activations and FP8 attention activations, using the
`least_squares_grid` block-scale recipe. It scores 0.39206 (474/1209) on full
GSM8K Platinum against 0.41522 (502/1209) for the same weights at W4A16, a drop
of 2.32 percentage points.

## W4A4 accuracy work: token-scale RMSNorm (experimental version 4)

The version-3 corrected NVFP4 checkpoint still loses 13.23 percentage points
against its matched W4A16 GSM8K Platinum reference. This is an unresolved
accuracy failure. Version 4 is a new explicit experimental policy; version-2/3
checkpoint behavior and the separate FP8 PR remain unchanged.

With rotation and RMSNorm-weight fusion, each decoder RMSNorm has unit weights.
Its transformation is then a scalar per token. Version 4 preserves the input
FP4 code buffer and the E4M3 block-scale buffer, and carries the FP32 inverse RMS
as an outer token multiplier. The native FP4 GEMM epilogue applies this multiplier
after the correctly scaled GPTQ group reduction and before bias. There is no
additional per-channel tensor or hidden high-precision residual. This removes
32 redundant FP4 re-encodings in a 16-layer model. Residual sums still produce
single packed FP4 carriers, and the same scalar-adjusted operand is shared by
Q/K/V and gate/up.

This extends the carrier with one FP32 value per token; it is not the plain
NVFP4 per-tensor global-scale layout alone. Hardware GEMMs still directly consume
E2M1 values and E4M3 block scales. Configuration and runtime installation reject
version 4 without NVFP4, rotation, and unit fused RMSNorm weights. Calibration
and HookedLinear replacement preserve the no-requantization norm-input flag.
For fresh quantization, set `GPTQMODEL_W4A_ACTIVATION_VERSION=4` and
`GPTQMODEL_W4A_ROTATION=hadamard` in the dedicated 1B lifecycle test.

Validation on GB10: 35 NVFP4 kernel/config tests passed, 27 replay tests passed,
and eight tiny-model lifecycle cases passed (four unsupported combinations
skipped). The new independent Torch tests cover FP16/BF16, zero rows, exact code
and scale pointer preservation, token-scale normalization at 1e-6 tolerances,
and grouped GEMM with bias at 2e-3 tolerances. The full saved-model audit passed
all 112 projections and 16 layers, checking norm code/scale reuse as well as
cached generation.

The first full 1,209-row GSM8K evaluation completed on a metadata-only view of
the corrected version-3 weights, retaining its frozen September 26 prompt date
and batch size 32. This isolates the runtime change; it is not a fresh version-4
GPTQ solve. The result rejects it as a sufficient accuracy-recovery recipe.

| Policy | Correct / rows | Accuracy | Delta vs matched W4A16 |
| --- | ---: | ---: | ---: |
| Matched W4A16 | 524 / 1,209 | 43.3416% | — |
| Version 3 | 364 / 1,209 | 30.1075% | -13.2341 pp |
| Version 4 token-scale norms | 399 / 1,209 | 33.0025% | -10.3391 pp |

Version 4 recovers 35 correct answers. Its paired 95% interval against W4A16
is [-13.0339, -7.6443] percentage points, with 207 baseline-only correct and
82 A4-only correct rows. It changes 788 extracted answers. The provisional
2-point regression gate remains failed. Artifacts:

- Checkpoint: `/root/models/Llama-3.2-1B-Instruct-W4A-NVFP4-token-norm-v4-view`
- Audit: `/root/models/w4a-quality/nvfp4_token_norm_v4_dtype_audit.json`
- Candidate results: `/root/models/w4a-quality/gsm_nvfp4_token_norm_v4_w4a4_full.json`
- Paired report: `/root/models/w4a-quality/gsm_nvfp4_token_norm_v4_paired.json`
- Matched baseline: `/root/models/w4a-quality/gsm_nvfp4_consumer_stream_v3_hookfix_fresh64_w4a16_full.json`

Do not infer downstream acceptance from the numerical tests. The full-row paired
regression gate remains required; fresh calibration and subsequent validation
remain pending if this runtime change proves useful.

### Adaptation lifecycle corrections discovered during the version-4 run

Further inspection found three training/inference consistency problems in the
experimental QAD tools. These changes affect future adaptation runs and do not
change the in-flight version-4 inference evaluation:

1. Weight QAD hard-coded policy version 3, while scale/norm adaptation relied on
   the replay default. All now read the source checkpoint's version and recipe
   so training matches the exported activation contract. Frozen-headroom
   recipes are rejected until their calibrated replay is supported. Norm-only
   training rejects version 4 because changing the unit norm weights would
   invalidate code-preserving normalization.
2. Master-weight initialization regenerated a random Hadamard rotation. It now
   recovers the checkpoint's signed Hadamard matrix from unchanged native and
   saved embedding witnesses, before applying the norm fusion and weight
   transformations. On 64 real embedding rows at width 2,048, reconstruction
   matches saved BF16 bytes exactly; the FP64-to-stored-BF16 relative difference
   is 0.00166312. Directly seeding a new rotation with 42 differs in 1,027 signs.
   This establishes that the seed alone cannot identify this saved rotation;
   historical QAD RNG states were not saved, so their exact matrices cannot be
   retrospectively asserted from their reports. The witness report is
   `/root/models/w4a-quality/nvfp4_master_rotation_recovery.json`.
3. Weight/scale adaptation produced FP16/BF16 group partials before applying
   GPTQ scales, unlike the inference kernel's FP32 partials. A new independent
   FP64 oracle rejected the old weight path (3/896 elements outside tolerance;
   maximum reported failing absolute difference 0.00683594). Both adaptation
   paths now accumulate group products in FP32 and round only the final output.

Ten CPU tests now pass, including exact signed-rotation recovery, rejection of
unrelated embedding witnesses, checkpoint-policy propagation, and FP16/BF16
weight/scale forward checks with finite gradients. Group tests use M=7, K=256,
N=128, two distinct GPTQ groups, and bias, with rtol=atol=2e-3 against a separate
FP64 oracle rounded to the visible dtype. GPU adaptation smoke tests and a new
full-row adapted checkpoint evaluation remain pending; CPU parity is not proof
of recovered downstream quality.

The prior corrected version-3 result was also checked for generation truncation:
only 1 of its 241 lost baseline answers lacked a stop token, and no candidate row
had an empty numeric extraction. Extending the generation limit is therefore
not the primary recovery experiment.

### Larger-corpus adaptation support and queued GPU checks

NVIDIA's current example QAD configuration uses 20,000 training samples and
2,000 evaluation samples, with maximum sequence length 8,192, warmup, and cosine
learning-rate decay ([source recipe](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/llm_qat/configs/train/qad_nvfp4.yaml)).
Our previous 64–128-row trials are far smaller; their failure is not evidence
that adequate, correctly aligned distillation cannot recover this model. The
NVIDIA recipe uses a different weight format, so it is guidance for training
structure rather than a directly interchangeable GPTQ checkpoint recipe.

Weight QAD now accepts `--teacher-cache-dir NEW_DIRECTORY`. It writes lossless
BF16 teacher logits and optional hidden states as safetensors, and loads only
one sample at a time during training/validation. It rejects existing cache
directories to prevent accidental reuse with different weights or tokenization.
The report includes the cache path and bytes written. This removes the
all-targets-in-RAM constraint for larger training corpora without truncating
vocabulary logits or altering the distillation objective. Disk capacity and
per-step activation memory still bound a run; the same 2 GiB headroom guard
remains required.

Eleven CPU tests pass after the cache addition. Four additional cases exercise
FP16/BF16 weight/scale adaptation on CUDA. They are queued behind the running
full-row version-4 evaluation, followed by a two-step layer-15 adaptation/export
smoke check using the recovered native rotation and version-4 replay, then a
full-coverage dtype audit. The short smoke run validates the lifecycle only;
it cannot establish accuracy recovery. Queue log: `/tmp/w4a4-after-v4-eval.log`.

### Preserve the native reference for activation-aware GPTAQ

Inspection found that installing W4A replay before the looper also rounded the
`NativeProcessor` reference pass used by GPTAQ/FOEM. That prevented the native
reference from representing the unquantized activation path. The layer stage
now disables activation replay for the native processor and enables it for the
quantization processor. The switch follows layer replicas and HookedLinear
replacement; rotations remain active in both coordinate-aligned paths.

A CPU regression proves that disabled replay reproduces the original dense
Llama forward at 1e-6 tolerances for activation policies 2, 3, and 4, and that
re-enabling replay restores rounding. HookedLinear replacement preserves the
switch. The replay plus stage suites pass 63 tests. A GB10 tiny-model test is
queued to require nonzero GPTAQ native/quantized cross terms at all 14 selected
projections, followed by quantize/save/reload and encoded-handoff checks.

The tiny GPTAQ lifecycle check subsequently passed, requiring nonzero reference
cross terms, save/reload, and encoded handoffs. It is separate algorithm coverage
and does not establish an improvement to plain GPTQ. The experimental 1B GPTAQ
run was stopped after the user clarified that this work must target plain GPTQ.
The dedicated 1B acceptance test no longer accepts the GPTAQ environment switch;
it explicitly sets GPTAQ and FOEM to `None` and checks the saved algorithm metadata.


The corrected adaptation test suite subsequently passed **15 tests on GB10**,
including the four CUDA FP16/BF16 weight/scale oracle cases and the lossless
disk-cache test. The version-4, two-step layer-15 training/export smoke run and
its full-coverage dtype audit passed. It selected step zero and changed zero
INT4 codes: this validates replay/export and recovered coordinates, with no
evidence of quality recovery. The audit is
`/root/models/w4a-quality/nvfp4_v4_corrected_qad_smoke_dtype_audit.json`.

### Fresh plain-GPTQ version-4 calibration

The completed version-4 evaluation reused version-3 GPTQ weights. Its score
was 399/1,209 versus the same-weight W4A16 score of 524/1,209, a confirmed
10.34-percentage-point regression. This remains a failed acceptance result.

A fresh plain-GPTQ run now calibrates all 112 projections across 16 layers using
version-4 replay, Hadamard rotation, least-squares activation scales, and 64 calibration
sequences concatenated to length 2,048. GPTAQ and FOEM are disabled. The output
is `/root/models/Llama-3.2-1B-Instruct-W4A-NVFP4-v4-gptq-full64-concat2048`.
Configuration and source hashes are recorded in
`/root/models/w4a-quality/nvfp4_v4_gptq_full64_manifest.json`.

Validation is queued after successful quantization: full encoded-boundary
audit, a fresh same-weight W4A16 evaluation, and a W4A4 evaluation, each using
all 1,209 GSM8K Platinum rows. Prompt dates are frozen to the prior baseline's
date. Both absolute scores and their paired difference must be reported;
changing the weights cannot be treated as recovering activation accuracy merely
by lowering the W4A16 reference. The existing frozen results remain unchanged.
Each GPU job runs sequentially under the 2 GiB headroom guard with swap disabled.
Logs: `/tmp/w4a4-v4-gptq-full64-quant.log` and
`/tmp/w4a4-v4-gptq-validate.log`. Pending jobs do not establish acceptance.

Fresh quantization subsequently passed (145 seconds), including plain-GPTQ
metadata and all 112 saved native INT4 projections. The initial full-coverage
dtype/cached-generation audit passed. The paired full-row evaluation is active.

### Separate A4 rounding error from carrier transport errors

The same-weight A16/A4 comparison isolates the total activation-path effect;
it does not alone prove whether the loss arises from quantization, an incorrect
scale handoff, or incorrect consumption. The transport audit now checks every
prefill and cached-decode invocation, including global-scale and token-scale
pointers and metadata alongside codes and E4M3 block scales. Norm-to-attention,
norm-to-MLP, shared projection inputs, and consecutive decoder boundaries are
checked. Ordinary tensors' version counters are recorded when available;
inference tensors do not expose these counters, so pointer identity is not
claimed as proof against arbitrary in-place mutations.

An independent FP64 decoder reconstructs the E2M1 values, inverts the block-scale
layout with tensor reshapes/permutations, and applies global and token scales.
Every audited NVFP4 value must match production decoding at rtol=atol=1e-6.
Twenty-eight CPU tests pass, including all 16 codes, positive/negative values,
multiple K groups, scale row tiles at 33 and 129 rows, optional token multipliers,
and detection of bad prefill handoffs hidden by correct final decode calls.
These are diagnostic test results, not downstream quality acceptance.

After the active quality pipeline, guarded GPU runs are queued for both the
399-correct version-4 checkpoint and the fresh plain-GPTQ checkpoint. They use
an exact saved GSM8K prompt and cached decoding, then profile the reconstruction
error at each materialized A4 boundary. The expanded GPU checks remain pending.
Log: `/tmp/w4a4-v4-transport-audit.log`; results will use the
`nvfp4_v4_{reused,fresh}_{scale_transport_audit,boundary_profile}.json` names under
`/root/models/w4a-quality/`. No BF16 carrier fallback is introduced by this audit.

The same audit now checks consumption at all NVFP4 Linear calls for policy 3+
by sampling row-bit transitions at 16, 32, 64, 96, and 128 rows, the next tile,
and the masked tail (one row during single-token decode).
Its FP64 oracle unpacks saved INT32 GPTQ weights and zero points directly,
applies each original weight-group scale, and includes activation block,
global, and token scales plus bias. It does not use prepared FP4 weight planes
or the production GEMM. Results must match at rtol=atol=2e-3 after rounding to
the visible output dtype. There are now **38 passing CPU diagnostic tests**,
covering both GPTQ zero-point encodings and FP16/BF16 scales. The pending GPU
run will exercise this check on actual model values; the CPU tests alone do
not prove native GEMM correctness. Numerical-oracle failure stops the audit.

Boundary profiling additionally records the enclosing decoder layer for free
function residual packing, signed radial error, and reconstructed/reference
energy ratio. These measurements distinguish loss of activation magnitude
from total reconstruction error; they do not by themselves identify a proven
quality-recovery recipe.

The consumer audit was promoted ahead of the fresh A4 benchmark: the ongoing
A16 evaluation finishes first while its coordinator is held, then guarded
audits and profiles run before the coordinator resumes A4 evaluation. This
changes only job ordering. Priority log:
`/tmp/w4a4-priority-consumer-audit.log`; priority result files have
`priority_consumer_audit` / `priority_boundary_profile` suffixes. Source hashes
are in `nvfp4_v4_transport_oracle_manifest.json`. Calibration used 64 source
records totaling 27,455 non-padding tokens, packed into 14 batches of length
2,048; it must not be described as 64 full-length sequences.

### Consumer audit outcome and NVIDIA calibration review

Fresh plain-GPTQ version-4 W4A16 completed every GSM8K Platinum row at
502/1,209 (0.4152191894). Its A4 evaluation was stopped shortly after launch
to investigate the consumer audit; there is no completed fresh A4 score.
The previously queued validation/audit coordinators were stopped as well.

The first consumer oracle compared output rounding directly from FP64 and
flagged one value in layer 5 gate projection. Captured evidence shows
FP64=-0.9394531684173697, hardware FP32=-0.939453125, and BF16=-0.9375.
The correctly modeled independent FP32 accumulation reproduces hardware
exactly. This is a BF16 midpoint crossing from ordinary FP32 rounding, not an
established transport failure. The capture remains in
`/root/models/w4a-quality/nvfp4_v4_consumer_failure_capture/`.

The oracle now explicitly models FP32 plane/group accumulation and epilogue
rounding using independent Torch arithmetic from native INT4 codes, retaining
the FP64 mathematical result as a separate diagnostic. Tolerances remain
1e-6 for decoding and 2e-3 for GEMM. Forty-two CPU tests pass. The real GB10
audit of the 399-correct checkpoint passes an 837-token saved prompt plus
cached generation: 381 complete carrier handoffs, 720 independent decode
comparisons (maximum absolute error 4.7684e-7), and 336 Linear comparisons
across all 112 projections. Sampled GEMM outputs match the FP32 oracle exactly;
FP32-oracle versus FP64 maximum absolute difference is 3.0101e-6. This is strong
evidence for the tested values, not a universal guarantee. Result:
`/root/models/w4a-quality/nvfp4_v4_reused_fp32_consumer_audit.json`.

NVIDIA Model Optimizer source was reviewed at
`23355eda90a25c290f9b1fdfb928ad54caae7d10`, alongside its current public docs:

- [NVFP4 preset](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/configs/ptq/presets/model/nvfp4.yaml)
  runs max calibration and enables weight/input quantizers. Activation block
  scales are dynamic E4M3 per 16 E2M1 values; calibrated tensor amax provides
  the persistent global scale. Dynamic blocks do not imply an uncalibrated
  global scale.
- [Headroom calibration](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_calib.html)
  calibrates activation global scales from block-amax statistics separately
  from weight-scale calibration. Layerwise calibration defaults to consuming
  preceding layers' quantized outputs.
- [Standard QAT/QAD](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/llm_qat/README.md)
  calibrates first, then updates master weights with simulated quantization;
  its ordinary recipe keeps calibrated scales fixed during training.
- [Dual-LSQ](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/general/qad/nvfp4_dual_lsq-mse_init-fp8_kv.yaml)
  learns NVFP4 **weight** scales. The recipe retains dynamic activation
  quantization. Our runtime least-squares `least_squares` is not that training algorithm.

Our active version-4 `least_squares` producer path recomputes global scales dynamically.
GPTQ collects and saves per-Linear activation scale metadata, but that metadata
does not drive this encoded producer path. Earlier headroom experiments are
documented failures; they do not establish a calibrated version-4 stream.
The next calibration experiment should hold all native GPTQ weight tensors
fixed, calibrate the actual producer boundaries (including residual carriers),
and propagate already-quantized upstream activations while collecting/fitting
later boundaries. Version-4 norms must preserve code/block-scale reuse and
their separate token multiplier. Export must bind every consumer to those
calibrated producer scales. This tests A4 calibration without rerunning the
GPTQ weight solve or replacing the frozen W4A16 baseline. Scale-only learning,
if needed, is a separate custom extension and must not be labeled as NVIDIA's
weight-LSQ recipe. Full 1,209-row evaluation remains mandatory.

### Explicit least-squares recipe names

Our runtime fitting recipes are now named `least_squares`,
`least_squares_headroom`, and `least_squares_grid`. Defaults, variables, replay,
kernel dispatch, tests, and newly serialized configs use these expanded names.
The old `lsq`, `lsq_headroom`, and `lsq_grid` spellings are compatibility aliases
normalized when loading earlier checkpoints. They do not select a training
algorithm. Existing artifact filenames and historical scores retain their
original names for provenance. NVIDIA's LSQ / Dual-LSQ continue to mean Learned
Scale Quantization of weights in the discussion above.

All 41 NVFP4 kernel/config tests pass on GB10 after the rename, including exact
packed-code/block-scale equality for every old/new name pair and normalization
of legacy checkpoint metadata on save. This is a naming change with identical
quantization arithmetic; it is not an accuracy improvement.

Replay checks also pass (31 tests), as do the consumer-audit helpers (42 tests)
and the GB10 tiny-model quantize/save/reload suite (9 passed, 4 skipped).

### Disjoint data prerequisite for activation calibration

The activation-only boundary calibration pass must consume the verified text
artifact produced by `tests/models/w4a_calibration_data.py`. Its source is
[WikiText-103 raw](https://huggingface.co/datasets/Salesforce/wikitext), **train**
only, pinned at `b08601e04326c79dfdd32d625aee71d232d685c3`. Article identity
determines the fit/selection partition before any model runs. Each article
contributes at most one sample, capped at 4,096 whitespace-delimited words;
the model runner applies its separate token-length cap. These are article
counts, not guaranteed full-length token sequences.

The exclusion registry pins revisions of GSM8K (train/test), GSM8K Platinum
(test), MMLU-Pro (test/validation), MMLU (test/validation/dev), ARC-Challenge
and ARC-Easy (train/test/validation). This covers scored and few-shot examples
for these benchmarks. Every selected article is screened against **45,748
question rows across 14 splits**, using Unicode NFKC/case normalization and
alphanumeric word boundaries. Any shared 13-word question excerpt excludes
the article; questions shorter than 13 words are checked in full. Answers,
model predictions, and evaluation scores are never used to fit or select
scales. A new evaluation dataset requires extending and rebuilding the
exclusion registry before further calibration.

Prepared artifact:
`/root/models/w4a-calibration/wikitext103-disjoint-v1/`.
It contains 512 fitting articles and 64 separate selection articles, with
zero detected lexical overlap among accepted samples. Forty-two candidate
articles were conservatively excluded; some matched very short generic
questions, so this count is **not** evidence of 42 contaminated articles.
Lexical screening does not prove the absence of paraphrased questions.

`manifest.json` records corpus/benchmark revisions, per-split counts, policy,
seed, excluded article/question IDs, and content-file hashes. `samples.jsonl`
records article IDs, source row IDs, partition, text, and normalized hashes.
The loader verifies file integrity and reference completeness, checks article
and text duplicates across partitions, and repeats overlap checks before
returning text. It rejects missing manifests and old unchecked parquet inputs.
Existing QAD/reconstruction helpers now use this loader; QAD selection uses
the explicit selection partition. Checkpoint export writes a separate
`w4a_calibration_manifest.json` with provenance and selected source IDs without
overwriting the source checkpoint's metadata. Historical artifacts are unchanged.

Reproduce preparation and independent verification under the memory guard:

```bash
GPTQMODEL_TEST_PYTHON=/root/gptqmodel-test-venv/bin/python \
  bash tests/models/run_w4a_gb10_safe.sh calibration-data prepare \
  --output /root/models/w4a-calibration/wikitext103-disjoint-v1 \
  --fit-rows 512 --selection-rows 64
GPTQMODEL_TEST_PYTHON=/root/gptqmodel-test-venv/bin/python \
  bash tests/models/run_w4a_gb10_safe.sh calibration-data verify \
  --directory /root/models/w4a-calibration/wikitext103-disjoint-v1
```

Preparation refuses an existing output directory. Twenty-two CPU tests cover
normalization, excerpts, article separation, incomplete reference coverage,
modified artifacts, rechecks with updated hashes, partition-specific loading,
and source-safe provenance export. This establishes the dataset prerequisite;
the new producer-scale calibration lifecycle and its full 1,209-row quality
evaluation remain in progress. It does not retroactively certify datasets
used for historical weight quantization or prior experiments.

### Producer-scale calibration lifecycle (version 4)

`gptqmodel/quantization/activation_calibration.py` now implements activation-only
maximum calibration on an existing full-coverage INT4 Llama stream. It captures
unpadded decoder inputs once, stores only the current layer's samples on CPU,
and visits each producer in execution order. Each boundary observes all fitting
samples before its FP32 global scale is frozen. Replaying later boundaries uses
the already frozen earlier producers. After a layer is calibrated, its actual
encoded output carriers become the next layer's calibration inputs. Failure
restores the prior scale values and removes observers.

The producers are the entry carrier, attention-to-o-projection operands, the
attention residual, rotated MLP-to-down-projection operands, and layer outputs:
65 boundaries for all 16 Llama layers. Version-4 RMSNorm continues to reuse
codes/block scales and apply its outer token multiplier. No Hessian solve or
GPTAQ is involved in this pass.

`activation.global_scales` serializes the complete producer-to-FP32-scale map.
Post-init checks exact boundary coverage and attaches nonpersistent runtime
buffers on each layer's own device. INT32 bit storage preserves the FP32 values
through model dtype conversions. Local E4M3 block scales remain dynamic and
use the configured recipe. The initial experiment retains `least_squares` for
these local scales; the new global-scale fitting algorithm is named
`producer_maximum`, not Learned Scale Quantization.

The guarded CLI is `producer-calibrate`, implemented by
`tests/models/w4a_nvfp4_calibrate.py`. It requires the verified disjoint dataset
artifact before loading the model, calibrates selected fitting articles,
measures separate held-out articles, and exports a metadata view sharing the
exact source `model.safetensors`. It verifies native tensor digests before and
after calibration, reloads the view, checks the scale map and exact probe
logits, and verifies the restored native tensors and complete weight-file hash.
The artifact records actual fitting token counts and dataset provenance.

The first real run uses 64 fitting articles (115,465 tokens, capped at 2,048
tokens per article) and 16 selection articles. Its source is the previously
399-correct version-4 view, preserving the same weights as the frozen
524/1,209 W4A16 baseline. Output:
`/root/models/Llama-3.2-1B-Instruct-W4A-NVFP4-v4-producer-max64`.
The sequential verification/evaluation coordinator is
`/tmp/w4a4-producer-max64-validate.sh`; calibration log:
`/tmp/w4a4-producer-max64.log`. Its full GSM8K Platinum result and paired
comparison will be written under `/root/models/w4a-quality/` with
`nvfp4_v4_producer_max64` in their names. Accuracy recovery remains unproven
until that evaluation and the unchanged 2-point acceptance gate complete.

Initial verification passed 15 GB10 producer tests, including independent
Torch quantization checks at 1e-6 tolerances for 1x128, 33x256, and 129x128
BF16 inputs; topological calibration; encoded propagation; unchanged tensors;
exact scale/logit restoration; and failure rollback. An additional exact
packed-code test is included in the queued final run. All 33 replay checks
and 64 dataset/audit helper checks pass. Replay now also rejects incomplete
producer maps and avoids a second down-projection input rounding when the MLP
wrapper already quantized a product with identity rotation. The boundary
profiler includes the new producer modules instead of silently missing them.

The real maximum-calibration run has now completed and exported all 65 scales.
Its report confirms exact reload probe logits, exact native tensor digests,
and the identical source weight file (SHA256
`ba204a2bc5953560eb8a2d9ff6ad161bb1d01290be37e9be7106e87c1cf1887c`).
The 16 held-out articles contain 29,733 tokens. Their per-boundary relative
reconstruction RMSE ranges from 0.01222 to 0.08442; these local diagnostics
do not establish downstream quality recovery.

The final guarded checks pass: 16 producer-calibration tests, 41 NVFP4
kernel/config tests, and 9 tiny lifecycle tests (4 skipped). The calibrated
full-model consumer audit also passes on an 837-token prompt plus two cached
generation tokens: 381 handoffs, 720 independent decode checks (maximum
absolute error 4.7684e-7), and 336 independent GEMM checks across all 112
projections. Report:
`/root/models/w4a-quality/nvfp4_v4_producer_max64_consumer_audit.json`.

The initial coordinator stopped after those checks when date-freezing found
an already frozen source template. The calibrated template is byte-identical
to the frozen W4A16 baseline; tokenizer files and the weight file also share
their exact source files. An evaluation alias preserves that existing
26-Sep-2026 template. The remaining full evaluation and paired comparison are
running from `/tmp/w4a4-producer-max64-eval.sh`, logging to
`/tmp/w4a4-producer-max64-gsm.log`. Do not rerun the stopped initial coordinator
or treat the pending GSM8K result as an accepted accuracy improvement.

### Standard calibration API and loaded-checkpoint export

`model.calibrate_activations(samples)` now connects producer calibration to the
normal save lifecycle. `samples` contains unpadded token-ID vectors; the caller
must establish evaluation-data separation before supplying them. The guarded
CLI continues to enforce the disjoint artifact check and now calls this API.
The method requires an already quantized model in evaluation mode with a
complete version-4 NVFP4 stream. It commits runtime scale values,
`quantize_config.activation.global_scales`, and Hugging Face configuration
metadata together. A failed fit or config validation restores the previous
runtime and metadata. Standalone core calibration without a config remains
available to tools that explicitly serialize its returned map.

Review of ordinary `save()` found a separate loaded-model writer defect: its
generic CPU reconstruction discarded all 14 activation-scale bit buffers in
a two-layer fixture and cast BF16 checkpoint parameters to FP16. The new
NVFP4-specific CPU reconstruction declares those metadata buffers and loads
the source tensors with assignment, preserving every dtype and tensor value.
It supports single-file and sharded safetensors, restores omitted tied
weights, and rejects missing, unexpected, or duplicate native tensor entries.
Runtime FP4 weight planes and producer buffers remain nonpersistent.

The defect was reproduced and fixed using CPU only while evaluation continued;
the real tiny fixture's 77 checkpoint tensors now match exactly. Twelve CPU
tests pass for the calibration/config transaction and native writer, including
mixed BF16/FP16/INT32 source data, shards, tied weights, and invalid checkpoints.
The broader CPU dataset/replay/audit group passes 102 tests, including the five
API transaction checks. A real GPU `calibrate_activations()` → ordinary
`save()` → reload test is queued after the current full evaluation, through
`/tmp/w4a4-calibration-api-validate.sh`; its result remains pending. These save
changes do not alter the already exported checkpoint being evaluated.

The current held-out maximum-scale report also shows that all observed values
fit within the global representable range: the largest observed maximum is
83.34% of `6 * 448 * global_scale`. This checks global-range saturation only;
it does not exclude clipping selected by local block-scale optimization or
establish the cause of downstream answer changes.

### Held-out same-weight reconstruction diagnostics

`layer-trace --calibration <verified-artifact>` now traces the selection
partition of the disjoint corpus. The original `--prompt-result` diagnostic
mode remains available; the two input modes are mutually exclusive. The new
mode uses 16 held-out articles capped at 2,048 tokens and 64 uniformly chosen
positions per article, including the first position and excluding the final
position so every probe has a known next token. Probe positions are fixed
before observing model error. Only those hidden rows and logits are saved,
bounding host memory and avoiding full-sequence vocabulary-logit caches.

The trace comparison verifies corpus-manifest and native-weight hashes,
runtime versions, identical token IDs/positions/targets, sample coverage, and
trace-file hashes. It computes layer reconstruction error both weighted by
activation energy and with equal weight per token, along with angular and
radial error diagnostics. The distinction matters: a large-magnitude token
can otherwise hide substantial error in the other tokens. The logit comparison
uses FP64 arithmetic and the full vocabulary at sampled positions, reporting
teacher-to-student KL, next-token NLL, and argmax agreement. These are held-out
diagnostics, not substitutes for the full downstream accuracy gate.

Eighteen CPU tests pass, including independent scalar KL/NLL checks, the
outlier-masking example, zero vectors, mismatched data/weights, inconsistent
probe positions, missing layers, and corrupt trace files. The real GPU traces
are queued after the full evaluation and ordinary-save lifecycle test via
`/tmp/w4a4-heldout-diagnostics.sh`. They compare the frozen same-weight W4A16,
dynamic-global version-4 A4, and calibrated-global version-4 A4 checkpoints on
the same selection articles. Outputs will be under
`/root/models/w4a-quality/heldout-v4-producer-max64/`; no result from that queued
diagnostic is available yet. It does not read benchmark answers or update any
weight or activation scale.

### Producer calibration validation update (27 September)

The ordinary GPU calibration → save → reload suite has completed with
10 passing tests and 4 skips; the producer suite passes all 16 tests.

The full producer-max64 evaluation completed with 352/1,209 correct
(29.11497%), but used batch size 8 while the frozen W4A16 and dynamic A4
results used 32. The strict paired comparison rejected that mismatch. Keep
`gsm_nvfp4_v4_producer_max64_full.json` as an unpaired batch-8 observation;
it does not establish a valid paired regression magnitude.

`quality-eval --baseline-result` now checks full baseline coverage, fixed
engine settings, the same native weight file, and identical tokenizer/template
files before model loading. It inherits the baseline batch size and rejects
an explicit mismatch. Existing result files cannot be overwritten. Fifteen
CPU preflight tests pass, alongside 22 dataset exclusion tests. The actual
producer-max64 checkpoint passes preflight with batch size 32. A fresh full
run was launched through the 2 GiB reserve/no-swap guard, saving to
`gsm_nvfp4_v4_producer_max64_batch32_full.json`; paired acceptance remains
pending. Its log is `/tmp/w4a4-producer-max64-batch32-gsm.log`.

The held-out traces have also completed on 16 selection articles and 1,024
sampled token positions, using unchanged native weights:

| Diagnostic | Dynamic A4 | Producer-max64 A4 |
| --- | ---: | ---: |
| Mean teacher-to-student logit KL | 0.201750 | 0.195785 |
| Sampled next-token NLL (W4A16: 2.871472) | 3.045600 | 3.048562 |
| Argmax agreement with W4A16 | 77.1484% | 77.3438% |
| Final-layer equal-token relative RMSE | 0.363851 | 0.361217 |
| Final-layer mean radial error | -0.088413 | -0.085186 |

The small, mixed changes do not establish an accuracy recovery. All these
diagnostics use the verified WikiText selection partition, separate from
fitting articles and benchmark questions. Outputs are in
`/root/models/w4a-quality/heldout-v4-producer-max64/`. The lexical exclusion
audit is not a guarantee against semantic paraphrases, and it does not
retroactively certify the original GPTQ checkpoint's calibration data.

### Token-energy diagnostic queued after the paired evaluation

The held-out final-layer radial errors above motivate a limited hypothesis:
preserving each token's norm at encoding might reduce accumulated shrinkage.
This is not established by those aggregate errors; directional errors and
nonlinear propagation could dominate. The diagnostic computes
`gain = ||x||_2 / ||decode(pack(x))||_2` at a producer and multiplies the
existing version-4 FP32 token multiplier by that gain. It preserves the FP4
codes and E4M3 block-scale tensors exactly. It keeps no dense residual, changes
no native weights, and writes no activation policy to the checkpoint. This
is a local norm-preservation experiment, not trained scale learning.

`layer-trace --calibration ... --experimental-token-energy {residual,all}`
enables the removable hooks only for verified selection data. `residual`
covers stream entry and residual boundaries; `all` also covers attention
output and the rotated MLP product. The manifest records the scope and
per-producer token-gain statistics. The helper normalizes operands before
sum-of-squares accumulation, handles zero/zero as unity, and rejects a
nonzero token encoded entirely to zero. It cannot repair lost directions.

Twenty CPU correction checks and 18 held-out trace checks pass, including
independent scalar FP64 norm oracles at widths 128/2048/8192 for FP16, BF16,
and FP32, extreme finite magnitudes, zero tokens, and exact carrier-storage
identity through removable hooks. Target-hardware oracle checks at `1e-6`
and a tiny encoded-stream/native-state check are queued in the producer
suite. Only if those pass will the two full 16-article selection diagnostics
run. The coordinator `/tmp/w4a4-token-energy-diagnostics.sh` waits for the
live batch-32 evaluation PID, then uses the usual serialized GPU memory guard.
These experiments have no quality result yet and are not a production recipe.

The queued producer suite now collects 25 tests. Two additional GB10 cases
check the corrected carrier through the real NVFP4 GEMM at 1 and 33 rows,
256 input channels, and 128 output channels. They independently unpack native
INT4 weights, fit the norm multiplier in FP64, apply distinct scales for two
GPTQ groups, and include bias. The test observes the FP32 result before output
packing and uses the inference limit of `2e-3`; every native tensor must remain
exact. These target-hardware cases are collected but have not run yet.

A CPU scale-range diagnostic also inspected the saved W4A16 layer-output
probes: 2,097,152 nonzero blocks across 16 articles and 16 layers. With the
saved producer global scales, neither the M=4 nor M=6 candidate needed an
E4M3 scale below the smallest normal value. The M=4 candidate exceeded 448
in 608 blocks; the M=6 candidate did so in zero blocks. This weighs against
small-scale range loss on these reference output probes, but does not measure
the actual calibrated upstream distribution, attention/MLP producer inputs,
or clipping introduced by least-squares selection. It is not an accuracy
result. Details are in
`heldout-v4-producer-max64/reference_output_scale_range.json` under the quality
artifact directory.

### Completed matched producer-max64 quality gate

The corrected batch-32 run completed all 1,209 GSM8K Platinum rows and passed
the strict pairing checks against the frozen same-weight W4A16 baseline.

| Variant | Correct / rows | Accuracy | Difference from W4A16 |
| --- | ---: | ---: | ---: |
| Frozen W4A16 | 524 / 1,209 | 43.3416% | — |
| Dynamic version-4 A4 | 399 / 1,209 | 33.0025% | -10.3391 pp |
| Producer-max64 A4, matched batch 32 | 379 / 1,209 | 31.3482% | -11.9934 pp |

The calibrated-versus-W4A16 paired 95% interval is [-14.7629, -9.2238] points:
227 baseline-only correct and 82 calibrated-only correct answers. The verdict
is `confirmed_regression` against the 2-point budget. The calibrated run also
has 139 losses and 119 gains against dynamic A4 with identical row prompts,
targets, task metadata, and checked engine settings: net 20 fewer correct.
The result does not support promoting producer maximum calibration as an
accuracy recovery. Keep both baseline artifacts fixed.

Result: `gsm_nvfp4_v4_producer_max64_batch32_full.json`; paired report:
`gsm_nvfp4_v4_producer_max64_batch32_paired.json`, under the quality directory.
The earlier batch-8 observation remains a separate unpaired artifact.

The next producer GPU suite initially stopped on a test-harness import target
error in the two new raw-GEMM checks (23 other tests passed). After correcting
the monkeypatch target to the module imported inside `forward`, all 25 GPU
tests pass in 8.75 seconds, including FP32 quantization checks at `1e-6`,
independent GEMM checks at `2e-3`, exact packed-code preservation, and unchanged
native state. Log: `/tmp/w4a4-token-energy-gpu-tests.log`; the initial failure
log is preserved separately. The serialized coordinator has now started the
16-article residual-boundary norm-preservation trace, followed by the all-
producer trace. No accuracy benefit from either experiment is established yet.

### Completed token-energy selection diagnostics

Both experiments finished on the same 16 disjoint selection articles and
1,024 fixed probe positions, using the dynamic version-4 checkpoint:

| Policy | Logit KL | Next-token NLL | Argmax agreement | Final equal-token RMSE |
| --- | ---: | ---: | ---: | ---: |
| Existing dynamic A4 | 0.201750 | 3.045600 | 77.1484% | 0.363851 |
| Preserve norm at residual boundaries | 0.198865 | 3.074019 | 78.0273% | 0.368097 |
| Preserve norm at all producers | 0.203169 | 3.043005 | 77.3438% | 0.367573 |

The W4A16 sampled NLL is 2.871472. Residual-only correction improves KL
slightly but worsens NLL and hidden reconstruction. All-producer correction
improves NLL slightly but worsens KL and hidden reconstruction. Neither
supports adoption as an accuracy recovery; no full GSM8K rerun is justified
by these mixed diagnostics alone. No checkpoint policy or native weight was
changed. The experimental hooks remain opt-in diagnostic code.

Artifacts: `dynamic_energy_residual_comparison.json` and
`dynamic_energy_all_comparison.json` in the held-out quality directory.
The final-layer mean radial errors improve from -0.088413 to -0.080032 and
-0.079264 respectively, while cosine similarity falls slightly. This shows
that improving the radial metric alone does not remove the directional error
or reliably improve output quality. The next calibration investigation must
account for the actual downstream reconstruction objective.

### Producer scales selected by layer reconstruction (experimental)

`producer-reconstruct` now evaluates candidates through the actual encoded
NVFP4 runtime, using a same-file native W4A16 teacher. For each decoder layer,
it freezes the teacher's unrounded outputs and searches each producer scale
in execution order. The candidate factors are 0.75, 0.875, 1.0, 1.125, 1.25,
and 1.5 relative to that producer's starting scale. Its FP64 objective is the
mean per-token relative squared error of decoder outputs. Each token has
equal weight, so large activations cannot mask other tokens' errors. Ties
retain the current scale. Fitted encoded student outputs feed the next
student layer; the teacher follows its own W4A16 trajectory throughout.

This changes only the existing FP32 producer global scales. It retains the
runtime least-squares E4M3 block-scale recipe, native INT4 codes/GPTQ scales,
and the version-4 transport format. No gradient approximation or additional
token multiplier is used. Failed fitting restores every producer scale and
the prior policy. The standalone tool exports a metadata-only checkpoint view
and requires exact reload logits and unchanged native tensor/file digests.
It validates the disjoint corpus before loading models and uses only the fit
partition for candidate selection.

Seventeen CPU reconstruction/helper checks plus five calibration lifecycle
checks pass. The expanded producer GPU suite passes all 27 tests, including
successful two-layer fitting, encoded handoffs, unchanged native tensors,
and failure rollback. Checkpoint-shell copying now excludes stale producer
reports, preventing a later report write from modifying its source through
an inherited symlink; a CPU test covers this explicitly.

The first 1B trial uses 16 fitting articles capped at 512 tokens and starts
from producer-max64. Its output will be
`/root/models/Llama-3.2-1B-Instruct-W4A-NVFP4-v4-producer-reconstruct-fit16-L512`.
Coordinator: `/tmp/w4a4-producer-reconstruct-fit16.sh`; fitting log:
`/tmp/w4a4-producer-reconstruct-fit16.log`. After successful fitting/export,
it traces the same 16 selection articles at 2,048 tokens and compares against
the saved W4A16 probes. The trial is running under the 2 GiB/no-swap guard;
neither fitting improvement nor held-out accuracy recovery is established yet.

### Reconstruction result and remaining replay precision mismatch

The fit16/L512 trial finished all 65 producers on 8,192 fitting tokens. It
changed 43 producer scales; each layer's local fitting loss decreased by
0.0157% to 1.0314%. Export/reload logits and every native GPTQ tensor match
exactly, and the weight file retains SHA256
`ba204a2bc5953560eb8a2d9ff6ad161bb1d01290be37e9be7106e87c1cf1887c`.

The held-out result is worse: KL 0.211496, sampled NLL 3.106882, argmax
agreement 76.2695%, and final-layer equal-token RMSE 0.372248. The starting
producer-max64 checkpoint had KL 0.195785 and NLL 3.048562. These results
reject this small coordinate-search trial as an accuracy improvement; they
do not justify a new GSM8K run. The report is
`reconstruct_fit16_L512_comparison.json` in the held-out quality directory.

Separate inspection found that version-4 tensor replay still casts decoded
carriers back to the model dtype: `round_w4a_activation` returns its input
dtype, residual replay casts to `residual.dtype`, and ordinary RMSNorm returns
that same dtype. Inference instead preserves the FP32 product of FP4 codes,
E4M3 block scales, global scale, and token scale until projection output.
The existing version-4 checks ensure norm inputs are not quantized twice but
do not establish numerical equality with the deployed stream.

The deployed version-4 stream keeps the residual value in compute precision.
Its fused RMSNorm rescales the *rounded* carrier by the inverse RMS of the
*pristine* residual, so the norm denominator is never quantized. A replay that
instead normalizes the decoded value produces a different operand and a
different Hessian. An earlier deterministic CPU probe using an independent
NVFP4 codebook search and FP32 unit-RMSNorm arithmetic measured that
decoded-denominator mismatch at 17 tokens × 2,048 channels with global scale
0.01234567:

| Model dtype | Norm max absolute error | Relative L2 error | Elements beyond rtol=atol=1e-6 |
| --- | ---: | ---: | ---: |
| BF16 | 0.0151429 | 0.00226628 | 32,076 / 34,816 |
| FP16 | 0.00224638 | 0.00029587 | 32,121 / 34,816 |

Evidence: `/root/models/w4a-quality/nvfp4_v4_replay_precision_probe.json`.
This is an identified replay mismatch, not proof of the downstream
regression's magnitude or sole cause. Producer maximum and reconstruction
trials above used the actual encoded runtime and are unaffected by it.

### Version-4 replay precision correction and fresh audited quantization

Version-4 replay now reproduces the deployed stream operator for operator. It
carries the pristine residual in FP32 and rebuilds the rounded GEMM operand at
each producer, so its RMSNorm uses a rounded numerator with a pristine
denominator exactly like inference. The dynamic global-scale grid is selected
from the original producer dtype — the raw model-dtype stream entry, FP32 once
a residual sum has promoted the value — before promoting the operand to FP32.
Dense and HookedLinear consumers accept FP32 operands, compute in FP32, and
return the model dtype at projection outputs. An explicit exit cast matches
inference at an unselected decoder layer or final norm. Existing custom INT4
adaptation forwards remain in use; their output is cast at the same projection
boundary. Replacing a dense module with HookedLinear preserves the policy, and
installing replay on an existing HookedLinear retains capture hooks without a
second input rounding.

Each interior layer rebuilds its input operand from the pristine value instead
of retaining a rounded tensor across layers, so reverse-order recomputation
under non-reentrant gradient checkpointing cannot corrupt it.

The adaptation tools now disable the entire replay policy for their A16
preservation pass, including its precision changes. Replacing only the round
function with identity is insufficient for version 4. Native disabled replay
matches the BF16 and FP16 reference model exactly in the CPU fixtures.

Validation completed on GB10 (SM121) through the guarded harness:

| Coverage | Result |
| --- | --- |
| `tests/models/test_w4a_producer_calibration.py` | 36 passed |
| `tests/models/test_w4a_hardware_forward.py` | 32 passed |
| `tests/models/test_w4a_replay_stream.py` | 43 passed |
| Full W4A suite (kernels, policy, lifecycle, diagnostics) | 608 passed, 5 skipped |

The GB10 norm checks compare the replay Tensor, an actual encoded NVFP4
carrier, and an independent codebook/FP32 normalization oracle at
rtol=atol=1e-6 for both BF16 and FP16 models. CPU tests cover projection and
layer boundary dtypes, native bypass, an unselected-layer exit, HookedLinear
replacement, capture preservation, single quantization per producer, and
finite nonzero gradients through a custom adaptation forward. The oracle keeps
the pristine denominator, matching the deployed stream; a bit-exact probe
confirms the replay now equals the runtime at the norm operand. This closes
the demonstrated norm-operand mismatch; it does not prove that this defect
accounts for the entire model-level accuracy gap.

A fresh plain-GPTQ 1B solve has been launched as
`/root/models/Llama-3.2-1B-Instruct-W4A-NVFP4-v4-fp32replay-disjoint64`.
The original checkpoints and frozen scores are retained. The 1B test now
requires `GPTQMODEL_W4A_CALIBRATION_ARTIFACT`, selects only its fit partition,
caps each article at 2,048 tokens before concatenation, and records article
IDs, source-manifest hash, and actual token count in the checkpoint. It cannot
fall back to an unchecked parquet dataset. The trial uses 64 fitting articles,
all 16 layers/112 projections, Hadamard rotation, and least-squares activation
scales with GPTAQ/FOEM disabled. Source hashes and settings are recorded in
`nvfp4_v4_fp32replay_disjoint64_run_manifest.json` in the quality directory.

Coordinator: `/tmp/w4a4-fp32replay-disjoint64.sh`; quantization log:
`/tmp/w4a4-fp32replay-disjoint64-quant.log`. After quantization, the coordinator
prepares a same-weight A16 view, runs the consumer audit, and compares both
variants on the disjoint selection trace. GPU phases remain serialized under
the 2 GiB/no-swap guard.

The 64-article/2,048-token attempt terminated with a scope OOM in layer 0
after subset 3/4. Its 115,465 nonpadding tokens formed 57 concatenated
batches. The scope reached its 38,438,244,352-byte limit (35.8 GiB); no
checkpoint or quality result was produced. The failed run manifest and logs
are retained. After exit, root cgroup accounting was dominated by file cache
(about 66.6 GiB), with about 115.8 GiB host MemAvailable and zero swap; this
observation does not implicate the earlier NVIDIA slab-retention issue.

The retry retains the same 64 disjoint fit articles and caps each at 512
tokens, using `GPTQMODEL_W4A_CALIBRATION_MAX_TOKENS=512`. Concatenation remains
2,048 tokens. This bounds total source tokens at 32,768 while retaining
article coverage. The token cap is validated before model work and recorded
in checkpoint provenance. The default cap remains 2,048. Dataset/loader
validation now passes 28 checks, including both caps and invalid settings.
The memory guard and frozen reference are unchanged. Retry artifacts use
`fp32replay-disjoint64-L512` to distinguish this fitting protocol.

The retry completed: the 1B quantize/save test passed in 183.98 seconds,
covering all 112 projections. Its checkpoint provenance records exactly
32,768 source tokens from the 64 fit articles. The consumer audit passed
381 encoded handoffs, 720 independent decode checks (maximum absolute error
4.7684e-7), and 336 independent GEMM checks. The sampled scope peak during
quantization was 23.33 GiB, with no OOM events or swap; sampled host available
memory stayed above 92.99 GiB. Full-system memory and per-scope samples are
saved in `nvfp4_v4_fp32replay_disjoint64_L512_memory.jsonl`.

On the same 16 selection articles/1,024 probe positions used previously,
the new same-weight A16/A4 comparison has mean KL 0.190113, reference NLL
2.844775, A4 NLL 3.015975, and argmax agreement 0.776367. Final-layer
equal-token relative RMSE is 0.367079. These are mixed diagnostic changes
against the previous checkpoint (KL 0.201750, A4 NLL 3.045600, final-layer
RMSE 0.363851), not an end-to-end accuracy result. The weights and fitting
protocol changed, so downstream evaluation requires a new same-weight A16
pair as well as comparison with the frozen 524/1,209 reference. The original
reference remains intact.

Evidence: `heldout-v4-fp32replay-disjoint64-L512/comparison.json` and
`nvfp4_v4_fp32replay_disjoint64_L512_consumer_audit.json` in the quality
directory. The coordinator originally wrote its audit under the failed
attempt's unused filename; it was moved to the `L512` name above after
checking that its checkpoint field identifies this successful retry.

Full 1,209-row GSM8K Platinum evaluation is now queued serially by
`/tmp/w4a4-fp32replay-disjoint64-L512-quality.sh`: the new same-weight A16
reference first, then A4 with the paired preflight inheriting batch 32 and
checking weights/tokenizer/settings. Both use the frozen 26 Sep 2026 prompt
date, seed 42, BF16, and 256 generated tokens. Results will use the prefix
`gsm_nvfp4_v4_fp32replay_disjoint64_L512_` in the quality directory. Scores
remain pending; the frozen previous reference remains 524/1,209.

### Full-decoder replay and per-token outer-scale diagnostics

The corrected replay is now covered by four additional GB10 cases comparing
zero-update INT4 weight-adaptation forwards with two complete encoded decoder
layers and final logits. They cover BF16/FP16 and dynamic/fixed producer
scales, unpack INT4 codes independently from serialized qweight, and retain
the 2e-3 inference tolerance. They are queued behind the full quality pair
by `/tmp/w4a4-fp32replay-after-quality-checks.sh`; results are pending in
`/tmp/w4a4-full-decoder-replay-checks.log`.

A CPU diagnostic computes the best possible single scalar adjustment to each
recorded decoder output under the existing equal-token relative-error
objective. It is a selection-only lower bound, not fitted calibration metadata.
For the new checkpoint, that adjustment can remove only 0.108% of layer-0
squared error, 0.081% at layer 7, and 1.486% at layer 15. The final-layer RMSE
would change from 0.367079 to 0.364341. This gives little support for a simple
output-amplitude correction. Evidence:
`nvfp4_v4_fp32replay_disjoint64_L512_scalar_error_bound.json`.

`tests/models/w4a_token_global.py` introduces a separate diagnostic for
per-token FP32 outer scales. For each row, it divides by max(abs(x))/(448*4),
packs the normalized row into native E2M1/E4M3 with global scale 1, and stores
the row scale in the existing scalar token multiplier. Zero rows use scale
1. It retains a single FP4 code tensor and E4M3 block scales; no dense residual
is carried. This removes other rows' magnitudes from the packing decision.
The producer post-hook intentionally repacks for diagnosis and is not a
production packing implementation or an exported checkpoint recipe.

The option `layer-trace --calibration ... --experimental-token-global` runs
only against the verified selection partition. It rejects combination with
the energy experiment, calibrated producers, or unrelated producer hooks,
and records its policy and token-scale ranges in the trace manifest. Existing
checkpoint policy remains untouched. The CPU scale and trace checks pass
28 tests. Added GB10 cases cover FP16/BF16 against the independent codebook
oracle at 1e-6, exact per-row code/decode independence from other batch rows,
encoded interlayer handoff, and unchanged native parameters. They are part
of the queued producer suite. Selection and downstream results are pending;
this experiment has not been selected as the recovery method.

### Fresh checkpoint weight-only result and larger adaptation preparation

The fresh `fp32replay-disjoint64-L512` checkpoint completed all 1,209
GSM8K Platinum rows in its A16 lane: 500 correct (41.3565%). The frozen
original A16 checkpoint remains 524 correct (43.3416%). On identical prompts,
the new weights lose 152 originally correct answers and gain 128, a net
-1.9851 percentage points with paired 95% interval [-4.6967, 0.7264]. This is
a comparison between two weight checkpoints, not an activation-only effect.
The fresh A4 lane is running and must be judged against both its same-weight
pair and the original reference. Evidence:
`nvfp4_v4_fp32replay_disjoint64_L512_weight_only_comparison.json`.

Weight adaptation now supports opt-in `--gradient-checkpointing`, using
non-reentrant decoder recomputation. Frozen embeddings are supported. Each
lane completes backward before switching the replay policy, so recomputation
uses that lane's activation policy. The training report records the option.
Four CPU cases cover BF16/FP16 and enabled/disabled A4 replay through two
decoder layers; outputs and every latent-weight gradient match without
checkpointing exactly. The full CPU subset passes 15 tests. Four additional
CUDA cases use the real NVFP4 packer in the straight-through forward and are
queued by `/tmp/w4a4-checkpointing-after-selection.sh`, after the current
evaluation and selection work. No full-model memory-saving claim is made
before a measured training run.

Preparation of an expanded WikiText artifact is running at
`/root/models/w4a-calibration/wikitext103-disjoint-4096-v1`, requesting 4,096
fit and 512 selection articles. It retains the pinned corpus and all 14
benchmark/few-shot exclusion splits, seed 787, and article-based partitioning.
It is a separate CPU-only job with a 4 GiB memory cap, two-CPU quota, and
zero swap. The original calibration artifact remains intact. Preparation log:
`/tmp/w4a4-corpus4096-prepare.log`.

Preparation and independent reload verification have completed. The expanded
artifact contains 4,096 fit and 512 selection articles, with zero accepted
lexical overlaps against 45,748 questions across the 14 exclusion splits.
Both original partitions are exact prefixes of their expanded counterparts:
no original selection article entered the expanded fit partition and no
original fit article entered expanded selection. All hashes and overlap
checks were rerun. Evidence:
`/root/models/w4a-quality/wikitext103_disjoint4096_verification.json`.
This supplies data for a larger adaptation run; it does not establish a
recovered checkpoint or certify absence of paraphrased overlap.

Weight-QAD preflight now runs before loading a GPU model. It rejects existing
output/cache paths, a teacher that does not share the exact native weight
file, mismatched quantization settings or tokenizer artifacts, unverified
calibration input, and out-of-range fit/selection counts. The teacher cache
and training report record the source manifest hash and exact article IDs
used in each partition. Export rejects a changed calibration manifest and
still revalidates the calibration artifact when constructing the output.
The dataset/preflight suite passes 36 CPU tests, including each rejection
case and exact partition provenance. Log: `/tmp/w4a4-qad-preflight-cpu.log`.

Read-only inspection also confirmed that the established version-4 checkpoint
(`Llama-3.2-1B-Instruct-W4A-NVFP4-token-norm-v4-view`) and frozen 524-correct
A16 view share their packed-weight file, tokenizer files, and native
quantization settings. Thus future adaptation can start from the established
same-weight pair; it need not use the weaker fresh 500-correct checkpoint as
its teacher. This is a compatible starting point, not a new adaptation result.

The real established checkpoint pair passed preflight against the expanded
corpus with 4,096 fit and 64 selection articles. Exact IDs and the manifest
hash are recorded in `nvfp4_v4_qad4096_preflight.json` in the quality directory.
This preflight does not launch training or select an adaptation checkpoint.

### Memory preparation and checkpointed full-model training smoke

Parent-cgroup memory accounting during the A4 evaluation included about
70.34 GiB of file cache. A file-specific `POSIX_FADV_DONTNEED` operation was
applied to 17 completed weight-QAD checkpoint files after excluding symlinks
and any file observed open or mapped by a live process. File metadata remained
unchanged. Parent file cache decreased to 49.36 GiB and root-cgroup headroom
rose to about 45.13 GiB. No files were deleted, no global cache drop was used,
and the running scope limits were not changed. The 2 GiB/no-swap guard remains
in effect. Evidence: `completed_qad_cache_advice_20260927.json`.

The common adaptation export helper now excludes every `w4a_*_report.json`
from inherited symlinks. Otherwise a second adaptation run could write its
report through the link and overwrite the source run's evidence. Six cases
verify independent new reports for weight, scale, norm, and producer tools;
the combined dataset/export/reconstruction CPU suites pass 59 tests.

`/tmp/w4a4-checkpointed-qad-smoke.sh` is queued after the current evaluation,
producer checks, selection trace, and checkpointing checks. It requires all
36 producer/replay and 23 weight/checkpointing GPU-suite cases to pass before
starting. The smoke run uses the established 524-correct A16 teacher and its
version-4 A4 view, the recovered native master rotation, all 16 trainable
decoder layers, four fit articles, two selection articles, 64-token sequences,
two optimizer steps, two-way accumulation, and non-reentrant checkpointing.
It caches teacher targets on disk, exports standard packed INT4, and then
runs the saved-model consumer audit. This validates lifecycle and practical
memory feasibility only; a two-step smoke run cannot establish accuracy
recovery. Training and audit results remain pending.

### Teacher scale-loading diagnostic

A CPU load of the frozen 1B teacher through `GPTQ_TORCH` with requested BF16
compute dtype converts all 112 saved FP16 GPTQ scale tensors to BF16. The
largest absolute scale change is 0.0001220703125. The initial diagnostic's
assertion that runtime scale values equal the raw FP16 file values therefore
fails; the observed values are retained in
`nvfp4_v4_teacher_scale_load_audit.json`.

A follow-up CPU deserialization probe constructed Torch, TritonV2, and NVFP4
backend shells using the shared loader. All three produce exactly the same
requested BF16 conversion of the saved FP16 scales. This identifies common
runtime dtype behavior, not a demonstrated teacher-only mismatch. No scale
restoration or loader change was made on this evidence, since doing so in
only the adaptation lane would itself change the paired numerical function.
The probe covers loading, not hardware inference or runtime post-init.
Native re-export's separate exact-file preservation tests remain applicable.
Follow-up report: `nvfp4_scale_loader_backend_comparison.json`.

### September 27: disjoint calibration and completed replay validation

The expanded calibration artifact is verified and ready. Activation fitting
uses only the 4,096 WikiText-103 training articles in its `fit` partition;
recipe selection uses its separate 512-article `selection` partition.
Benchmark questions are exclusion references only. GSM8K train/test,
GSM8K Platinum test, MMLU-Pro test/validation, MMLU test/validation/dev,
and both ARC configurations' train/test/validation splits are covered.
There are zero accepted lexical overlaps against 45,748 questions across
these 14 splits. Normalized full short questions and shared 13-word excerpts
are excluded; paraphrased contamination is outside this check's guarantee.

`load_calibration_artifact` rechecks hashes, pinned source/benchmark revisions,
complete exclusion coverage, duplicate content, article partition ownership,
and question overlap before returning text. The producer calibration entry
point invokes it before allocating the GPU model. Its saved view shares the
original native weight file, checks the native tensor digest, and requires
exact logits and native tensors after reload. These checks apply to the new
activation calibration data; they do not retroactively certify the original
GPTQ checkpoint's calibration data.

The fresh FP32-replay checkpoint's full GSM8K Platinum pair has completed:
W4A16 scored 500/1,209, and W4A4 scored 368/1,209. The activation delta is
-10.9181 percentage points, with paired 95% interval [-13.7347, -8.1016].
This is a confirmed regression and is not an accepted accuracy improvement.
The established 524-correct A16 baseline and 399-correct A4 result remain
preserved. Report: `gsm_nvfp4_v4_fp32replay_disjoint64_L512_paired.json`.

The diagnostic per-token outer-scale quotient now uses FP64 division before
storing FP32, matching the independent rounding oracle. The previous CUDA
FP32 scalar-division difference could change FP4 codes near a tie. No
production Triton kernel was changed. All 10 CPU diagnostic tests and 36
producer/replay GPU tests pass, including exact outer-scale comparison and
the existing 1e-6 decoded quantization tolerance. Logs:
`/tmp/w4a4-token-global-fixed-cpu.log` and
`/tmp/w4a4-token-global-fixed-gpu.log`.

The earlier selection and training coordinator scripts stopped at their
numerical gate; they did not run selection or training. A new selection-only
retry, `/tmp/w4a4-token-global-selection-fixed.sh`, uses the passing log and
the disjoint selection partition under the existing 2 GiB/no-swap guard.
It completed all 16 articles and 1,024 sampled token positions. Compared with
the same-weight A16 trace, per-token outer scaling gives mean KL 0.188365
(previous tensor-global diagnostic: 0.190113), sampled A4 NLL 3.008595
(previous: 3.015975), and argmax agreement 0.793945 (previous: 0.776367).
Report: `heldout-v4-fp32replay-disjoint64-L512/token_global_comparison.json`.
These small held-out changes do not establish downstream accuracy recovery.
The checkpointed training smoke remains unstarted, and the saved production
recipe is unchanged.

### Established checkpoint: per-token outer-scale selection rejected

The same diagnostic has now completed on the established 399-correct A4
checkpoint, paired with its frozen 524-correct W4A16 trace. The 16 disjoint
selection articles and 1,024 uniform sampled positions use the same protocol
as the existing comparison.

| Metric | Existing tensor-global scaling | Experimental per-token outer scaling |
| --- | ---: | ---: |
| Mean KL to same-weight A16 | 0.201750 | 0.243874 |
| Sampled A4 NLL | 3.045600 | 3.061415 |
| Argmax agreement | 0.771484 | 0.765625 |
| Final-layer equal-token relative RMSE | 0.363851 | 0.389483 |

This rejects the current per-token outer-scale policy for adoption. It does
not justify a full GSM8K rerun. The small improvement on the fresh, weaker
checkpoint did not transfer to the established checkpoint. Packed-code row
independence alone therefore does not establish better model accuracy.
Report: `heldout-v4-producer-max64/dynamic_token_global_comparison.json`.

The corrected coordinator `/tmp/w4a4-checkpointed-qad-smoke-fixed.sh` has
started the previously prepared two-step training smoke. It gates on the
36 passing producer/replay cases and 23 passing weight/checkpointing cases,
uses only the audited fit/selection partitions, and writes a separate native
INT4 checkpoint. Initial process memory cap: 62,335 MiB, with zero swap and
the unchanged 2 GiB host/root guard. Unlike activation-only calibration, this
separate adaptation experiment can update native INT4 codes; its source and
frozen baseline files remain unchanged. Results are pending.

### Completed checkpointed smoke and larger disjoint-data adaptation

The two-step checkpointed smoke completed and exported successfully. It
optimized 973,078,528 latent parameters but crossed no INT4 code thresholds;
selection therefore retained step zero. This establishes training/export
feasibility, not accuracy improvement. The saved-model consumer audit covers
all 16 decoder layers and 112 projections, with 381 encoded handoffs, 720
independent decodes (maximum absolute error 4.7684e-7), and 336 independent
GEMM checks. Recorded training peak was 35.1064 GiB, minimum root-cgroup
headroom 29.5605 GiB, with no swap or OOM events. Reports:
`nvfp4_v4_checkpointed_qad_smoke_audit.json` and
`nvfp4_v4_checkpointed_qad_smoke_memory.jsonl`.

The disk teacher cache now synchronizes each newly written private file and
uses file-specific `POSIX_FADV_DONTNEED` where supported. Reading copies only
the requested sample into owned CPU tensors, closes its file mappings, and
advises away cached file pages. It retains every target file and its exact
tensor values. Three focused CPU checks pass, including synchronization order,
owned tensor storage, unchanged file bytes, and fallback without POSIX advice.
The complete weight/checkpointing GPU suite passes all 25 cases. This avoids
intentionally retaining the entire teacher corpus in the training cgroup's
filesystem cache; kernel cache advice remains advisory. Training now also
reports preflight completion, teacher-cache progress, and initial validation.

A larger run is active using the established same-weight pair and the audited
expanded corpus: 512 fit articles, 32 selection articles, 256-token sequences,
256 planned steps, four-way accumulation, all 16 layers, and non-reentrant
checkpointing. It uses learning rate 5e-6, eight warmup steps, cosine decay to
0.1 of that rate, logit KL objective, A16 preservation weight 4, and cell-space
regularization 1e-4. Every 16 steps, checkpoint selection requires lower held-out
A4 loss and A16 loss no more than 5e-4 above its initial replay discrepancy.
The first preservation breach stops training and exports the last accepted
snapshot. These are selection conditions, not substitutes for full GSM8K gates.

Coordinator: `/tmp/w4a4-replayfixed-qad-r512-v32-s256-t256-gc.sh`.
Training log: `/tmp/w4a4-replayfixed-qad-r512-v32-s256-t256-gc-train.log`.
Output: `Llama-3.2-1B-Instruct-W4A-NVFP4-v4-replayfixed-qad-r512-v32-s256-t256-gc`
under `/root/models`. Source/code hashes, command, calibration manifest hash,
and memory observations are in `nvfp4_v4_replayfixed_qad_r512_v32_s256_t256_gc_`
`run_manifest.json` and `memory.jsonl` under the quality directory. Initial
process cap is 61,088 MiB, zero swap, with the unchanged 2 GiB host/root guard.
No accuracy recovery is established yet.

### Larger adaptation rejected; full-model replay fidelity gate

The 512-fit/32-selection run stopped at step 32 on its preservation limit.
Step 16 had no code changes and unchanged selection losses. Step 32 changed
20,964 INT4 values, increased selection A4 KL from 0.268236 to 0.282546, and
increased selection A16 KL from 0.000984 to 0.038574 (limit 0.001484).
No step was accepted. Export retained the initial INT4 codes, and its saved
consumer audit passed all 16 layers, 112 projections, 381 handoffs, 720
independent decodes, and 336 independent GEMMs. No new full GSM8K run is
justified by this rejected fit. The frozen 524-correct baseline is unchanged.

The teacher cache contains 35,722,935,040 bytes. Recorded process peak was
33.9524 GiB, minimum root-cgroup headroom 27.5966 GiB, with zero swap and no
OOM events. This run exercised the file-cache release changes with a corpus
large enough that retaining all target pages would materially increase memory
pressure. Outcome: `nvfp4_v4_replayfixed_qad_r512_v32_s256_t256_gc_outcome.json`.

The held-out trace tool now supports `--training-replay` for the complete 1B
weight-adaptation forward. It uses the saved native INT4 codes/scales and the
same activation-rounding function as training, without an optimizer or weight
updates. Runtime and replay traces must have identical weights, activation
policy hashes, data partitions, inputs, positions, and runtime versions.
Comparison records every sampled decoder/logit tensor's maximum absolute
error and element count outside rtol=atol=2e-3, and requires exact sampled
logit argmax tokens. It writes failed evidence before returning failure.
Replay traces cannot be used as the runtime lane of a quality comparison.
The comparison/helper CPU suite passes 24 tests; 17 CPU adaptation cases
also pass after extracting the shared forward.

Using the actual training rounder in the existing two-layer GPU test exposed
a separate fixed-scale lifecycle error: saved producer scales are Python
numbers, while the hardware packer expects a device tensor. The shared
rounder now converts these values directly to device FP32. All 36 producer
GPU cases pass after this fix; the 25 weight/checkpointing GPU cases also pass.
The rejected larger fit used dynamic scales and is unaffected by this bug.

`/tmp/w4a4-replay-fidelity-fixed.sh` now runs the full-model comparison on
four disjoint selection articles, 256 tokens each, with 64 fixed uniform
probes per article. Output directory:
`/root/models/w4a-quality/replay-fidelity-v4-qad-r512-v32-s256-t256-gc`.
This establishes a broader numerical gate than the tiny two-layer cases;
its result is pending and will not substitute for full downstream validation.

### Full-model replay mismatch and RoPE buffer correction

The four-article full-model fidelity gate failed: replay versus encoded runtime
has mean logit KL 0.318726 and 59/256 different sampled argmax tokens. The
first layer already differs (equal-token relative RMSE 0.156516). This blocks
treating the current adaptation forward as a faithful runtime simulation.
The failure is retained in the replay-fidelity directory's `comparison.json`.

A first-layer operand trace shows identical embeddings, encoded entry values,
and normalized Q/K/V operands. Projection outputs differ slightly, but raw
attention output has relative L2 error 0.004832 and its dynamic FP4 global scale
differs by 0.7752%. That changes the block-scale grid throughout the tensor.
The trace also identifies an actual lifecycle defect: after dequantization,
the adaptation helpers cast the entire model to BF16, rounding Llama's loaded
FP32 RoPE frequencies. The maximum frequency error is 0.000913501. The tiny
tests had constructed both sides with the same whole-model dtype cast and
therefore did not expose this loaded-model difference.

`_dequantize_for_replay` now casts only the newly replaced Linear modules;
existing model buffers keep their loaded values and dtypes. Weight, scale,
norm, scale-reconstruction, and fidelity-audit helpers share this conversion.
New FP16/BF16 CPU cases require exact FP32 RoPE and FP64 buffer preservation
and exact converted Linear weights. The CPU adaptation subset passes 19 cases,
and the complete GPU-enabled suite passes 27 cases.

The corrected first-layer trace has exact RoPE frequencies and exact attention
global scale. Raw attention relative L2 error falls to 3.18074e-5, and decoder
output relative L2 error falls from 0.120856 to 0.015406 on the full first
article. Small projection rounding differences remain and later FP4 encoding
amplifies them. Full-model fidelity still fails: mean KL 0.306113 and 63/256
different sampled argmax tokens. The correction is necessary but does not
establish a matching training forward or downstream accuracy recovery.
Evidence: `first-layer-rotary`, `first-layer-rope-preserved`, and
`rope_preserved_comparison.json` in the same replay-fidelity directory.

The next adaptation implementation must use the encoded hardware model for
forward values while retaining a differentiable surrogate for gradients:

1. Maintain a separate runtime model with the current rounded native INT4
   codes. Refresh its prepared FP4 weight planes after optimizer updates;
   inference checkpoint storage remains native INT32-packed GPTQ.
2. Capture actual decoded producer boundaries and projection operands/outputs
   during a no-gradient encoded forward on each fitting sample. Correct the
   differentiable replay's values at those boundaries, including both residual
   sums, while preserving the surrogate gradients.
3. Keep the captured values valid through each complete backward, including
   non-reentrant checkpoint recomputation. The disabled A16 lane must bypass
   every A4 correction. No dense residual is added to inference transport.
4. Require full-model forward fidelity before another accuracy fit, exercise
   gradients and changed-code refresh independently, then repeat disjoint
   selection and the original full downstream gates. Do not relax fidelity
   tolerances or substitute the surrogate's loss for actual runtime quality.

This hardware-forward adaptation is planned, not implemented or validated yet.

### Hardware-forward implementation and exact full-model fidelity

`tests/models/w4a_hardware_forward.py` now implements that training path.
Its auxiliary encoded model runs the current rounded native INT4 weights.
It captures producer outputs and Linear operands/results, substitutes those
exact forward values into the differentiable surrogate, and retains identity
straight-through gradients. Both decoder residual boundaries use the captured
hardware values. The inference carrier and checkpoint formats are unchanged.

Each frame owns one sample's snapshots through forward and backward, including
non-reentrant checkpoint recomputation. Overlapping frames, code refresh during
a frame, and recomputation after a frame has expired fail explicitly. The A16
lane bypasses all hardware substitutions. Rounded INT4 codes are packed directly
on the device; only modules with changed packed values rerun post-init to refresh
their prepared weight planes. Source checkpoint files are not modified.

Five CPU/GPU checks pass on GB10. They cover the exact value/gradient rule,
FP16/BF16 exact runtime logits, bit-exact gradients with/without checkpointing,
expired-frame rejection, A16 bypass and hook removal, and a one-code update
that changes exactly one packed word and refreshes the FP4 planes. The combined
held-out/adaptation CPU subset passes 43 cases (eight CUDA cases deselected).

The full 1B comparison on four disjoint 256-token selection articles also
passes: all 41,222,144 values across 16 decoder outputs and sampled logits are
bit-identical to the independently saved encoded-runtime trace. Maximum absolute
error is zero, logit KL is zero, and all 256 sampled argmax tokens agree.
Evidence: `hardware_forward_comparison.json` in the replay-fidelity directory.
This verifies forward fidelity at the tested inputs; gradients remain an
explicit quantization surrogate and accuracy recovery still requires fitting
and the downstream gates.

Weight adaptation now accepts `--hardware-forward`. Every A4 training and
selection forward checks exact equality against the hardware model's logits;
the report records the check count and per-step refreshed-module count. The
flag is opt-in while the training lifecycle is validated. A two-step 1B smoke
at 256-token sequence length is running with four fitting and two selection
articles, two-way accumulation, checkpointing, and the existing memory guard.
Coordinator: `/tmp/w4a4-hardware-forward-smoke.sh`; log:
`/tmp/w4a4-hardware-forward-smoke-train.log`. Memory samples are recorded in
`nvfp4_v4_hardware_forward_smoke_memory.jsonl` under the quality directory.
No new accuracy fit or downstream acceptance is established yet.

The 256-token hardware-forward smoke has now completed. Its ten A4 training
and selection forwards all matched hardware logits exactly, both optimizer
steps completed, and export retained the initial codes. The saved consumer
audit passed 381 handoffs, 720 independent decodes, and 336 independent GEMMs
across all 16 layers and 112 projections. Recorded peak was 40.4679 GiB,
minimum root-cgroup headroom 21.6792 GiB, and swap/OOM counts remained zero.
This is lifecycle validation; the two-step run does not establish accuracy.

Weight adaptation now tokenizes the same verified data snapshot returned by
preflight instead of rereading and re-auditing the corpus separately for fit
and selection. The initial full audit and final export revalidation remain
mandatory. A regression check verifies one loader call and exact correspondence
between tokenized texts and the recorded article IDs. The combined dataset,
adaptation, and held-out CPU subset passes 86 cases (eight CUDA cases deselected).

The larger hardware-forward fit is active with the same 512-fit/32-selection,
256-token, up-to-256-step settings and original preservation/selection limits.
It uses four-way accumulation and checkpointing, and asserts exact hardware
logits on every A4 forward. Coordinator:
`/tmp/w4a4-hardware-qad-r512-v32-s256-t256-gc.sh`; training log:
`/tmp/w4a4-hardware-qad-r512-v32-s256-t256-gc-train.log`.
The separate output checkpoint is
`Llama-3.2-1B-Instruct-W4A-NVFP4-v4-hardware-qad-r512-v32-s256-t256-gc`.
Its source hashes, exact command, and live memory samples use the prefix
`nvfp4_v4_hardware_qad_r512_v32_s256_t256_gc_` in the quality directory.
Initial process cap is 56,863 MiB, zero swap, retaining the 2 GiB host/root
guard. Fit outcome and full downstream acceptance remain pending.

### Hardware-forward fit outcome and optimizer units

The 512-fit/32-selection hardware-forward run stopped at step 32 and retained
step zero. All 224 A4 forwards matched hardware logits exactly. At step 16 no
codes had changed. At step 32, 20,735 codes differed: held-out A4 KL worsened
from 0.264959 to 0.278137 and A16 KL increased from 0.000950787 to 0.0390496,
exceeding the unchanged 0.00145079 preservation limit. The exported checkpoint
therefore contains no code changes. Its full consumer audit passed again; a
full GSM8K rerun would only repeat the unchanged source checkpoint.

The recorded peak was 36.5998 GiB, minimum root-cgroup headroom 20.7613 GiB,
zero swap, and zero OOM/max/high events. Outcome and memory evidence use
`nvfp4_v4_hardware_qad_r512_v32_s256_t256_gc_` in the quality directory.

The next experiment changes the optimizer's parameterization. A common physical
Adam update corresponds to different distances between INT4 codes because each
GPTQ group has its own scale. For example, a physical step of 5e-6 corresponds
to 0.0201 code cells at the smallest layer-0 V scale and about 0.00158 at the
smallest layer-14 attention-output scale. This arithmetic identifies an update
imbalance; it does not establish that changing units will recover accuracy.
Measurements are saved in `nvfp4_v4_weight_scale_update_units.json`.

`weight-qad --weight-parameterization code_cell` now optimizes dimensionless
latent codes directly. The forward still rounds to native INT4, applies the
unchanged group scales, and uses exact hardware values. Clamp, regularization,
snapshot, packed export, and hardware refresh share this representation.
Physical-weight optimization remains the default. Learning rates use the
selected units; the recorded Adam denominator floor defaults to 1e-12 for
code cells and 1e-8 for physical weights, with an explicit override available.
Selection records now retain per-projection code-change counts, including for
rejected candidates before rollback.

Validation passed 23 CPU weight cases and the complete 33-case GB10 weight
suite. A separate eight-case hardware suite passed exact logits, exact
checkpointed gradients, and changed-code refresh in both parameterizations
and FP16/BF16. The independent code-gradient oracle, INT4 clamps, snapshots,
and regularization checks also passed without loosening tolerances.

A separate 512-fit/32-selection, 256-token hardware-forward run is active with
code-cell learning rate 0.005, Adam epsilon 1e-12, four-way accumulation,
checkpointing, eight warmup steps, and selection every eight steps. It retains
the same data snapshot, teacher, initial codes, and preservation gate.
Coordinator: `/tmp/w4a4-hardware-codecell-r512-v32-s256-t256-gc.sh`.
Logs use that prefix; the output checkpoint is
`Llama-3.2-1B-Instruct-W4A-NVFP4-v4-hardware-codecell-r512-v32-s256-t256-gc`.
Source/command manifest and live memory samples use
`nvfp4_v4_hardware_codecell_r512_v32_s256_t256_gc_` in the quality directory.
Its accuracy outcome and the full downstream acceptance gate remain pending.

### Exact native forward for the preservation objective

The code-cell fit remains active. Its initial selection values exactly match
the earlier physical-weight fit, and the first 32 steps have not changed any
INT4 codes. No candidate has been accepted yet.

There is a second forward-fidelity issue to address before interpreting very
small preservation losses: the differentiable A16 groupwise arithmetic has
initial KL 0.000950787 against the unchanged TorchLinear teacher. Native A16
inference rounds dequantized weights to its matmul dtype. Groupwise FP32
arithmetic is useful for the gradient but is not that exact forward function.

`tests/models/w4a_native_forward.py` now supplies actual native W4A16 values
to the disabled replay lane. It captures projection operands after online
Hadamard rotation and captures native projection outputs. A straight-through
substitution retains the student's gradients. Changed rounded codes update
the auxiliary native model and invalidate its dequantized-weight, streaming,
and prefetch caches. Snapshots remain valid through backward and checkpoint
recomputation; stale/overlapping frames and mutation during a frame fail.

Weight adaptation accepts `--native-forward` in addition to
`--hardware-forward`, requires exact logits on every native preservation
forward, and records its check count and refreshed modules. This does not
change the active code-cell run, which started before the new option existed.

Four CPU cases pass for FP16/BF16 and physical/code-cell parameters. They cover
exact native logits, zero initial KL, exact checkpointed gradients, online
Hadamard operands, one-code refresh with weight-cache invalidation, and frame
lifetime. The combined CPU subset passes 29 cases with six CUDA skips and 14
deselections. GPU and real 1B validation remain pending.

`/tmp/w4a4-native-forward-after-fit.py` waits for the verified current
coordinator PID/start time, including its export audit, then runs the guarded
hardware/native unit suite and a separate two-step 1B smoke with both exact
forward paths. The validation coordinator is
`/tmp/w4a4-native-forward-validate.sh`. Logs use `native-forward-gpu` and
`dual-forward-smoke` under `/tmp/w4a4-*`; the output is
`Llama-3.2-1B-Instruct-W4A-NVFP4-v4-dual-forward-smoke-r4-v2-s256-t2-gc`.
No GPU validation or accuracy gain is claimed until those jobs finish.

### Code-cell rejection and missing Hadamard gradients

The code-cell run completed and rejected its first changed-code selection at
step 56. It changed 8,698 codes across 75 projections (2,470 in layer-0
`mlp.down_proj`), but A4 KL worsened from 0.264959 to 0.269774 and A16 KL rose
to 0.0135429. Step zero was exported. All 480 A4 forward checks were exact,
and the saved consumer audit passed. Peak was 37.0616 GiB with minimum root
headroom 18.9815 GiB, zero swap, and no OOM/high/max events. Outcome evidence:
`nvfp4_v4_hardware_codecell_r512_v32_s256_t256_gc_outcome.json`.

The subsequent native-forward GPU tests exposed a separate training defect.
The fused TorchAO Hadamard fallback returns detached CUDA tensors. With online
MLP rotation enabled, upstream projection parameters could receive no gradient.
Earlier hardware-forward tiny tests did not enable online rotation, and the
trainer accepted partial gradient coverage. Thus those completed fits are not
evidence that training all intended parameters cannot recover accuracy.

`matmul_hadU_cuda` now wraps differentiable calls with an explicit symmetric
Walsh backward. The selected backend supplies exactly the same forward bytes;
backward uses a portable FP64 butterfly and one final dtype conversion. No
inference arithmetic or saved weights change. The normalized scale, including
its existing FP32 representation, is preserved in the derivative.

The hardware suite now enables online rotation in its tiny encoded models and
includes an independent Walsh-matrix gradient oracle for widths 128, 256, 2048,
and 8192, on CPU/CUDA in FP16/BF16. All 32 cases pass: forward identity is exact,
gradient tolerance is 1e-6, checkpointed gradients are exact, and all selected
parameters receive gradients. The oracle computes matrix entries from bit
parity instead of reusing the butterfly. Its FP64 reference is converted on
the target device to avoid CPU double-to-half double rounding at a tie.

Every A4 and A16 training backward now checks fresh per-parameter hook coverage;
existing gradients from an earlier microbatch or the other lane cannot hide a
detached branch. True zero gradients remain valid. Nonfinite aggregate gradient
norms fail before optimizer updates. CPU adaptation checks pass 24 cases.

The guarded 1B two-step smoke is now running with both exact forward paths,
fixed rotation gradients, and coverage checks over all 112 projections.
The command, base commit, hashes, and a snapshot of the changed source files
are saved under `nvfp4_v4_dual_forward_smoke_` in the quality directory.
Full-model validation and downstream accuracy recovery remain pending.

The corrected two-step 1B smoke has now passed. All ten A4 and ten A16
forwards matched their respective runtimes exactly, the unchanged native
teacher KL was zero, and all 112 projections participated in each of four A4
and four A16 backward passes. The saved consumer audit passed 381 handoffs,
720 independent decodes, and 336 independent GEMMs across all 16 layers.
Peak memory was 40.2570 GiB, minimum root-cgroup headroom 15.7532 GiB, with
zero swap and no OOM/high/max events. Initial exact native-teacher reproduction
is now enforced before adaptation begins.

A new 512-fit/32-selection, 256-token run is active with code-cell learning
rate 0.005, epsilon 1e-12, both exact forward paths, complete gradient checks,
four-way accumulation, checkpointing, and the same 5e-4 preservation-increase
gate. Coordinator: `/tmp/w4a4-dual-forward-codecell-r512-v32-s256-t256-gc.sh`.
Its checkpoint is
`Llama-3.2-1B-Instruct-W4A-NVFP4-v4-dual-forward-codecell-r512-v32-s256-t256-gc`.
The command, source snapshot, corpus hash, and memory samples use
`nvfp4_v4_dual_forward_codecell_r512_v32_s256_t256_gc_` in the quality directory.
This is the first such fit with verified full gradient coverage and exact
values in both lanes. Its selection result and downstream accuracy remain
unproven; the frozen 524/1,209 W4A16 reference is unchanged.

### Longer-context data readiness

A CPU audit revalidated the same disjoint artifact and measured the existing
512 fitting and 32 selection articles with the trainer's tokenizer settings.
No articles were moved between partitions. The available token counts are:

| Sequence cap | Fitting tokens | Selection tokens | Short fitting articles | Short selection articles |
| --- | ---: | ---: | ---: | ---: |
| 256 | 131,072 | 8,192 | 0 | 0 |
| 512 | 261,681 | 16,384 | 5 | 0 |
| 1,024 | 515,403 | 32,389 | 36 | 2 |
| 2,048 | 933,058 | 61,080 | 173 | 5 |

Evidence: `nvfp4_v4_disjoint_context_length_audit.json` in the quality
directory. The first four fitting and selection token prefixes were checked
against the existing 256-token path and matched exactly. These counts establish
data availability, not that larger contexts fit in GPU memory or improve quality.

Future weight-adaptation reports now distinguish `maximum_tokens_per_step`
from actual `processed_training_tokens`, record fitting/selection length
summaries, and report tokens used at each step. This avoids counting short
articles as full-length sequences. The CPU adaptation suite passes 25 cases
with ten CUDA cases deselected.

The current 256-token fit is still active. A conditional coordinator,
`/tmp/w4a4-long-context-after-fit.py`, waits for its verified PID/start time
through export and audit. If that fit selects a candidate, full downstream
evaluation takes priority and the coordinator does not start another GPU job.
If it exports unchanged step zero, the coordinator runs
`/tmp/w4a4-dual-forward-smoke-s1024.sh`: four fitting articles, two selection
articles, two steps, 1,024-token cap, and one-way accumulation under the same
2 GiB reserve/zero-swap guard. The smoke must establish exact dual forwards,
complete gradients, memory feasibility, and saved-checkpoint consumer behavior
before any longer-context accuracy fit. Its output is
`Llama-3.2-1B-Instruct-W4A-NVFP4-v4-dual-forward-smoke-r4-v2-s1024-t2-gc`.

### Corrected short-context outcome and longer-context continuation

The fully corrected 256-token fit stopped at step 56. Its 15,062 changed codes
covered all 112 projections. Held-out A4 KL improved slightly, from 0.264959
to 0.263272, but actual native A16 KL rose from zero to 0.0817944, exceeding
the unchanged 0.0005 gate. The candidate was rejected and step zero exported.
All 480 A4 and 480 A16 forward checks passed, as did complete gradient coverage
for 224 backwards in each lane. Peak memory was 39.6631 GiB, minimum root
headroom 14.0540 GiB, zero swap, and no OOM/high/max events. Evidence:
`nvfp4_v4_dual_forward_codecell_r512_v32_s256_t256_gc_outcome.json`.

The conditional 1,024-token smoke also completed. Eight forwards in each lane
matched their runtimes exactly; both backwards per lane covered all 112
projections. The two training steps each processed 1,024 tokens. The fitting
set contained 3,640 tokens across four articles (minimum 568), while selection
contained 2,048 tokens across two articles. Initial native KL was zero. Export
and the full 381-handoff/720-decode/336-GEMM consumer audit passed. Peak memory
was 44.7980 GiB, minimum root headroom 7.8858 GiB, with zero swap and no memory
events. Artifacts use `nvfp4_v4_dual_forward_smoke_s1024_`.

A new longer run is active: 512 fitting articles, 32 selection articles,
1,024-token cap, 512 optimizer steps, one-way accumulation, code-cell learning
rate 0.005, eight-step warmup, cosine decay, and selection every 16 steps.
The single epoch visits all fitting articles; reports record their actual
token lengths. Both exact forward paths and complete gradient coverage remain
mandatory.

This run leaves optional early stopping disabled so later optimizer updates
can attempt recovery after the initial discrete code transitions. Snapshot
acceptance is unchanged: native A16 KL must stay within 0.0005 of its initial
zero, and A4 selection loss must improve. Continuing optimization does not
authorize exporting a failing candidate or replacing the frozen reference.

Coordinator:
`/tmp/w4a4-dual-forward-codecell-r512-v32-s1024-t512-gc-continue.sh`.
Checkpoint:
`Llama-3.2-1B-Instruct-W4A-NVFP4-v4-dual-forward-codecell-r512-v32-s1024-t512-gc-continue`.
The command, changed-source snapshot, corpus hash, and live memory samples use
`nvfp4_v4_dual_forward_codecell_r512_v32_s1024_t512_gc_continue_` in the quality
directory. Accuracy recovery and the original full GSM8K acceptance remain
unproven.

### Adapted-checkpoint acceptance against the frozen reference

The quality helper now provides `accept-adapted` with three full GSM8K result
files: frozen original W4A16, candidate W4A16, and candidate A4/A8. It requires:

1. The candidate's native and activation runs use the same packed-weight file.
2. Their existing paired 95% activation interval is within the 2 pp budget.
3. Candidate activation quality versus the frozen original reference also
   meets that paired budget, allowing different weights only for this explicitly
   labeled overall comparison.
4. Candidate W4A16 has at least as many correct answers as the frozen reference.
   This is a point-score floor; the separate native interval remains visible
   and is not described as proof of statistical noninferiority.

Every comparison requires all 1,209 unique row indices, identical prompts and
targets, matching task settings, binary row scores, and agreement between row
scores and the aggregate metric. Adapted acceptance also verifies evaluation
engine settings, recorded software versions, native backend identity, and the
candidate's activation policy/backend agreement. The ordinary `compare`
command retains its same-weight restriction. The new command refuses to
overwrite an existing output artifact and records input result hashes.

The 26-case CPU preflight/acceptance suite passes. A regression case gives both
candidate lanes 523 correct answers against the frozen 524. Their same-weight
activation and frozen-reference activation comparisons pass the 2 pp budget,
but the native point floor rejects the candidate. Other cases reject duplicate
or partial rows, mismatched settings/prompts, nonbinary scores, and inconsistent
aggregate metrics.

The helper was also run on the real frozen 524/1,209 reference and existing
399/1,209 A4 result. It correctly rejects that checkpoint: -10.3391 pp, paired
95% interval [-13.0339, -7.6443] pp. Evidence:
`nvfp4_v4_original_acceptance_audit.json`. The frozen reference, its 524 correct
answers, and hashes of the result, native weights, quantization config, chat
template, and tokenizer are recorded in `nvfp4_v4_frozen_native_reference.json`.
The native weight hash remains
`ba204a2bc5953560eb8a2d9ff6ad161bb1d01290be37e9be7106e87c1cf1887c`.
These checks strengthen final validation; they do not establish accuracy
recovery for the active longer-context fit.

### Conditional full evaluation after the longer-context fit

`/tmp/w4a4-long-context-evaluate-after-fit.py` now waits on a Linux pidfd for
the specific active training coordinator, including its subsequent saved
consumer audit. It does not launch a second GPU job while that coordinator is
alive. Its queue manifest is
`nvfp4_v4_dual_forward_codecell_r512_v32_s1024_t512_gc_continue_evaluation_queue.json`.

After the coordinator exits, the follow-up requires a completed adaptation
report, the full consumer audit, exact forwards in both lanes, full gradient
coverage, and the unchanged native preservation gate. It records the selected
step, learning curve, and memory outcome. An unchanged step-zero export ends
the follow-up without repeating downstream generation.

For a selected changed checkpoint, the follow-up verifies the frozen reference
hashes and identical tokenizer/template artifacts, creates a same-weight native
view, and runs all 1,209 GSM8K Platinum rows in both lanes sequentially under
the existing memory guard. Batch size is 32, inherited from the frozen native
reference; activation evaluation inherits its candidate native settings.
`accept-adapted` then checks both the same-weight activation regression and
the frozen original reference. Evaluation helper and guard hashes are pinned
when queued; a later source change stops this automatic path for review.

At the first observed code transition, step 57 refreshed 107 projections.
Native training loss became nonzero. This is evidence of optimization activity,
not an accepted checkpoint or downstream accuracy recovery; selection was
still unchanged through step 48.

### Step-64 rejection and configurable latent initialization

The longer-context fit's first changed selection point, step 64, contains
99,012 changed codes. Held-out A4 KL worsens from 0.219199 to 0.272283 and
native KL rises from zero to 0.118303, failing the unchanged 0.0005 limit.
80,446 changes (81.25%) are in layers 0–4. This candidate is rejected; the
planned continuation remains active to measure whether later updates recover.

The master initializer previously fixed every latent offset to
`0.25 * tanh(master_offset / 0.25)`. This leaves at least a quarter-cell gap
before a rounding transition. At a 0.005 code-cell learning rate, persistent
directions therefore cannot change their code until roughly 50 updates
(longer with warmup). A burst of simultaneous changes is a plausible
optimization problem, not an established cause of the quality regression.

Future experiments can set `--master-cell-radius` in `(0, 0.49]` to use
`radius * tanh(master_offset / radius)`. The default remains 0.25. A radius of
0.49 retains at least a 0.01-cell gap, allowing earlier transitions while
preserving every starting INT4 code. A nondefault radius requires a master
checkpoint, and reports record its value. This changes only training
initialization; the active run uses its original source snapshot and settings.

The CPU adaptation suite passes 39 cases with 13 CUDA cases deselected.
New cases cover all 16 native codes, large and near-zero offsets, a Python
double-precision tanh oracle at 1e-6 tolerances, both latent parameterizations,
and exact initial code snapshots. A synthetic update confirms earlier
crossing is possible, without claiming an accuracy gain. Target-GB10 checks
and any radius experiment remain pending until the active training and
conditional downstream evaluation release the GPU.

`/tmp/w4a4-master-radius-gpu-after-evaluation.py` queues the guarded weight
suite behind the verified evaluation coordinator via pidfd. The test log will
be `/tmp/w4a4-master-cell-radius-gpu.log`. It queues validation only; a new
accuracy fit still requires the active run's final outcome and passing checks.

### NVIDIA QAD recipe comparison and step-80 recovery

NVIDIA Model Optimizer's HF example was rechecked at commit
`23355eda90a25c290f9b1fdfb928ad54caae7d10`. The read-only source snapshot and
SHA256 manifest are in `nvidia-qad-reference-23355eda90a2` under the quality
directory. The comparison below describes that example, not every NVIDIA
recipe or a guarantee that its hyperparameters transfer to this model.

| Item | NVIDIA HF QAD example | Active GPTQ experiment |
| --- | --- | --- |
| Teacher | Original higher-precision model | Frozen original native W4A16 |
| Weight representation | NVFP4 student | Native GPTQ INT4 with lossless hardware planes |
| Objective | Teacher distillation | A4 logit KL plus a separate native preservation term |
| Training data | Mixed instruction/chat, mathematics, science and code | Audited WikiText-103 articles |
| Configured training rows | 20,000 | 512 |
| Context cap | 8,192 | 1,024 |
| Per-device batch / accumulation | 2 / 2 | 1 / 1 |
| Learning rate | 1e-5 | 0.005 in dimensionless INT4 code-cell units |
| Schedule | One epoch, cosine, 5% warmup | One epoch, cosine, eight warmup steps |
| Selection | Evaluate periodically and load the best checkpoint | A4 improvement and strict native preservation required |

Sources: [QAD training config](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/examples/llm_qat/configs/train/qad_nvfp4.yaml),
[data blend](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/examples/llm_qat/configs/dataset/blend.yaml),
[QAD workflow](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/examples/llm_qat/README.md).
The learning rates use different parameterizations and are not directly
comparable. NVIDIA's example does not establish the encoded decoder/residual
transport or native INT4 preservation required by this project.

Instruction/math coverage is a plausible follow-up variable. Importing the
upstream blend without auditing it would not satisfy this project's benchmark
exclusion policy. Any future corpus must use a separate audited artifact,
exclude evaluation and few-shot content, and preserve separate fitting and
selection partitions. The active run's corpus remains unchanged.

At step 80, the active run changes 419,334 codes. Held-out A4 KL improves to
0.210770 (initial 0.219199), and native KL recovers from step 64's 0.118303 to
0.0686934. The native limit remains 0.0005, so this point is also rejected.
The recovery supports finishing the existing continuation before deciding on
another optimization experiment; it does not yet justify downstream evaluation.

### Separate diagnostic export for measuring the native-KL heuristic

At step 96, held-out A4 KL reaches 0.182319 (initial 0.219199) and native KL
falls to 0.0427648. The strict 0.0005 native selection screen still rejects
this candidate. Native KL is a proxy for downstream preservation; these
measurements alone establish neither GSM8K recovery nor regression.

Future weight-adaptation runs can request `--diagnostic-candidate-output` with
a separate new directory. An optional tracker retains the snapshot with the
lowest finite held-out A4 loss, independently of the strict snapshot. It
exports only if A4 improves over the starting checkpoint. The ordinary output
still requires the same native KL limit; no screening threshold is relaxed.

The diagnostic report explicitly records `export_role: diagnostic_only`,
`downstream_acceptance: not_evaluated`, and whether it passes the native
selection screen. Source tensors and the strict checkpoint are separate from
this diagnostic export. It is a measurement artifact, not an accepted model.
Any such candidate still needs a saved consumer audit, both full 1,209-row
GSM8K lanes, the original frozen native point floor, and both activation
regression intervals checked by `accept-adapted`.

This option lets later experiments establish whether native KL is overly
conservative without hiding a weaker native downstream score. The active run
has not requested it and continues with its original source snapshot.
Its extra snapshot costs approximately one byte per selected weight in host
memory; a full-model lifecycle smoke must verify the reserve before a larger
run enables it.

The CPU adaptation suite passes 42 cases, with 13 CUDA cases deselected.
New tests reject nonfinite candidates, keep the previous snapshot on ties or
capture failures, and independently construct packed INT4 words to verify
separate strict/diagnostic exports. They also verify byte-identical source and
strict checkpoint files, unchanged scales/zero points/group indices/embeddings,
and refusal to overwrite an existing output. GB10 and full training/export
validation remain pending behind the current run.

### Queued full-model smoke for radius and diagnostic export

`/tmp/w4a4-radius-diagnostic-smoke-after-tests.py` waits on the specific GB10
test coordinator using pidfd. It requires a successful test exit and the
current longer fit's completed outcome. If that fit selects a changed strict
candidate, this smoke exits so the downstream result can be reviewed first.

Otherwise it runs eight updates over four fitting articles and two selection
articles, with a 1,024-token cap, radius 0.49, and separate strict/diagnostic
output directories. Exact A4 and native forwards, per-backward coverage of all
112 projections, the native selection screen, and the 2 GiB/zero-swap guard
remain enabled. The process identity, command, corpus hash, changed-source
snapshot, and memory samples use the prefix
`nvfp4_v4_radius049_diagnostic_smoke_r4_v2_s1024_t8` in the quality directory.

The smoke expects 18 exact forward checks per lane and eight complete
backwards per lane. It audits the strict checkpoint and, only if an improved
A4 diagnostic was exported, audits that separate checkpoint too. No diagnostic
export is an explicitly reported outcome, not proof that the diagnostic
export branch ran successfully. This is a lifecycle/memory check; it does not
run GSM8K or establish accuracy recovery.

Meanwhile the original fit's step-112 selection has A4 KL 0.180335 and native
KL 0.0346307, with 685,674 changed codes. Both losses continue to recover from
step 64, but native KL still exceeds 0.0005. No changed strict candidate is
accepted at this point.

### Procedural arithmetic calibration data preparation

`tests/models/w4a_arithmetic_calibration.py` generates plain worked examples
from seeded integer arithmetic. It covers inventory, packs, equal shares,
production rates, discounts, ratios, payment/change, and remaining distance.
Each document contains twelve examples, including every family. No benchmark
question or answer is used to construct them; evaluation questions are used
only by the existing normalized lexical exclusion filter. The existing
tokenization path is reused without template or tokenizer changes.

The first generated artifact, `procedural-arithmetic-disjoint-4096-v1`, passed
benchmark exclusion but was rejected before training: 1,330 normalized
questions occurred in both its fitting and selection documents. Evidence:
`procedural_arithmetic_disjoint_v1_rejected.json` in the quality directory.
That artifact is retained as rejected evidence and its schema is not accepted
by the updated loader.

Version 2 assigns each normalized question to a deterministic hash partition
before allowing it into a document. Repeated generation can therefore never
place that question in both fitting and selection. The loader rechecks this
assignment, validates that document text matches its question records, and
verifies the archived generator source hash alongside the data hashes. The
same 45,748-question, 14-split exclusion registry remains mandatory.

The combined data/adaptation CPU suite passes 92 checks, with 13 CUDA cases
deselected. New checks include 400 independently solved rational-arithmetic
examples, deterministic generation, all-family coverage, source-hash tampering,
and cross-partition question injection even when data hashes are updated.

The 4,096-fit/512-selection v2 artifact is being built in a separate CPU scope
with a 2 GiB cap, one-core quota, and zero swap. Output:
`/root/models/w4a-calibration/procedural-arithmetic-disjoint-4096-v2`.
Preparation log: `/tmp/w4a4-arithmetic-data-v2-prepare.log`.
It is an additional arithmetic-coverage experiment, not a substitute for the
current WikiText fit or evidence of downstream accuracy recovery. Readiness
still requires the complete artifact and loader verification.

### Procedural arithmetic v2 readiness verified

The v2 artifact is complete and passes the calibration loader. Its manifest
SHA256 is `53136d5f94c105248e26cd4dad9d714518b738a4f9d18493900f8aa272bdf166`.
All 55,296 worked examples were independently checked with exact rational
arithmetic. The fitting documents contain 40,802 distinct normalized questions;
selection contains 5,225. Their intersection is empty. The artifact rejects
247 candidate documents for benchmark-text overlap, and every accepted record
passes the 45,748-question/14-split exclusion audit. This remains a lexical
check, not a guarantee against all semantic paraphrases.

The existing raw-text calibration tokenizer path was exercised at a 1,024-token
cap. The first 512 fitting documents yield 409,767 tokens (778–829 each); the
first 32 selection documents yield 25,671 tokens (780–829 each). All are shorter
than the cap, so token budgets must use actual lengths. Input IDs and attention
masks match between direct AutoTokenizer and Tokenicer for four fitting and
four selection documents. No tokenizer normalization change was required.

Evidence: `procedural_arithmetic_disjoint_4096_v2_verified.json` in the quality
directory; verification script `/tmp/w4a4-verify-arithmetic-v2.py`. Preparation
used about 370 MiB at observed peak, zero swap, and no memory events. The
verification ran in a separate 3 GiB CPU scope with CUDA hidden. This corpus
has not been used for training or checkpoint selection yet.

The live WikiText fit's step-176 selection has A4 KL 0.170436, native KL
0.0248464, and 900,985 changed codes. The native selection screen still fails;
the fit remains active and the frozen downstream reference is unchanged.

### Pytest entry point for full saved accuracy evidence

`tests/models/test_w4a_quality_preflight.py::test_nvfp4_full_gsm8k_saved_acceptance`
now consumes a complete saved evidence bundle through these environment values:

| Environment value | Required artifact |
| --- | --- |
| `GPTQMODEL_W4A_FROZEN_MANIFEST` | Original 524/1,209 reference manifest with file hashes |
| `GPTQMODEL_W4A16_FULL_RESULT` | Candidate same-weight native full GSM8K result |
| `GPTQMODEL_W4A4_FULL_RESULT` | Candidate A4 full GSM8K result |
| `GPTQMODEL_W4A4_CONSUMER_AUDIT` | Matching saved-checkpoint version-4 consumer audit |

The entry point verifies the frozen hashes, all 16 decoder layers and 112
Linears in the consumer audit, its 381 handoffs/720 independent decodes/336
independent GEMMs, checkpoint/recipe agreement, and the existing full-row
`compare_adapted` acceptance rules. Legacy `lsq` checkpoint names are normalized
through the shared recipe helper. Providing only part of the bundle fails.
Providing none explicitly skips the case as unverified, which is not an
accuracy pass. This test consumes recorded evidence; it does not regenerate
predictions or replace the numerical/runtime audit that produced it.

The CPU preflight/evidence suite has 36 passing checks and one unconfigured
e2e skip. Running the configured e2e case against the real frozen native
524/1,209 result, existing A4 399/1,209 result, and its complete consumer audit
fails as expected with both activation comparisons reporting a confirmed
regression. Log: `/tmp/w4a4-original-full-acceptance-pytest-final.log`.
This negative control demonstrates rejection of the existing inaccurate
checkpoint; it is not a new successful W4A4 result.

### Disjoint chat calibration input verified

Weight adaptation now accepts `--calibration-format chat_worked_examples` for
the audited procedural arithmetic v2 corpus. Its default remains plain text.
The preflight rejects this format for other corpora, verifies the complete
benchmark exclusion artifact, and records the format, template SHA256, manifest
SHA256, and exact fitting/selection article IDs before model loading. Both
partitions remain separate from evaluation and from each other. The activation
scale calibration and producer reconstruction entry points also require the
same audited artifact loader before fitting.

Question/answer turns use the checkpoint's existing chat template, without an
added generation prompt. Rendered text is tokenized without adding another set
of special tokens. No tokenizer normalization or evaluation prompt was changed.
The combined calibration-data/weight-adaptation CPU suite passes 99 cases;
13 CUDA cases are skipped with CUDA hidden. Log:
`/tmp/w4a4-chat-calibration-cpu.log`.

The real checkpoint tokenizer produces exactly the same IDs through direct
chat tokenization and the calibration path for 512 fitting plus 32 selection
records. Eight records also match Tokenicer's rendered text, IDs, and masks.
Every sequence has exactly one BOS token. At a 1,024-token cap the fitting
sample contains 459,431 tokens (875–926 per record), and selection contains
28,775 tokens (877–926); no records are truncated. Evidence:
`/root/models/w4a-quality/procedural_arithmetic_disjoint_4096_v2_chat_verified.json`.
This CPU audit ran with a 3 GiB limit and swap disabled. The chat corpus has
not yet been used for training or candidate selection.

The exclusion contract covers GSM8K train/test, GSM8K Platinum test, MMLU-Pro
test/validation, MMLU test/validation/dev, and ARC Easy/Challenge train/test/
validation: 45,748 reference questions across 14 splits. Questions from these
datasets serve only as exclusion references. Missing coverage, corrupted data,
or detected normalized lexical overlap fails before fitting. The overlap
check cannot certify the absence of arbitrary semantic paraphrases, and does
not retroactively certify the original GPTQ checkpoint's calibration data.

### Queued arithmetic-chat adaptation and complete downstream pair

`/tmp/w4a4-arithmetic-chat-after-smoke.py` is a live coordinator waiting on the
specific radius/diagnostic smoke process through pidfd. It does not occupy the
GPU while waiting. It requires the preceding fit to have selected no changed
strict candidate, successful GB10 tests and full-model smoke, zero smoke swap,
and at least 2 GiB observed root headroom. If the preceding fit selects a
candidate, the coordinator exits for review of that candidate's full results.
Required source hashes and the verified corpus/template hashes are pinned;
unexpected changes prevent launch. A separate source snapshot is captured
when the experiment actually starts.

| Setting | Follow-up experiment |
| --- | --- |
| Data | Audited procedural arithmetic v2, existing checkpoint chat template |
| Fitting / selection | 512 / 32 disjoint records; 1,024-token cap |
| Updates | 512, accumulation 1, checkpointing, all 112 projections |
| Latent initialization | Radius 0.49, exact original INT4 codes at step zero |
| Objective | Exact encoded A4 forward plus exact native preservation forward |
| Optimizer | Code-cell LR 0.005, Adam epsilon 1e-12, eight-step warmup, cosine to 10% |
| Native screen | Unchanged KL increase limit 0.0005 |
| Extra output | Separate best-held-out-A4 diagnostic checkpoint |
| Memory | Existing serialized GPU guard, 2 GiB reserve, zero swap |

This changes both the corpus/rendering and initialization radius from the
preceding large WikiText experiment, so any outcome cannot isolate their
individual effects. It is an accuracy experiment with the same output contract.
Weight adaptation is explicitly separate from activation-only scale fitting:
it exports a new native INT4 checkpoint while keeping GPTQ scales, zero points,
group indices, and the source checkpoint unchanged.

After training, the coordinator audits saved encoded consumers. It chooses a
changed strict candidate if available, otherwise the separately labeled
diagnostic candidate with improved held-out A4 loss. The latter may fail the
native KL heuristic; this is recorded and never counts as acceptance. Either
candidate must undergo both full 1,209-row GSM8K Platinum lanes, same-weight
pairing, the frozen 524-correct native point floor, and both existing 2-point
paired confidence-interval gates. A passing report is then checked through the
saved-evidence pytest entry point. No improved candidate means no duplicate
evaluation of unchanged weights.

Queue/run/evaluation artifacts use quality-directory prefix
`nvfp4_v4_chat_arithmetic_radius049_r512_v32_s1024_t512`. Coordinator log:
`/tmp/w4a4-arithmetic-chat-after-smoke.log`. The experiment is queued, not yet
trained or evaluated; accuracy recovery remains unproven.

### Saved consumer audits now bind checkpoint bytes

The saved-evidence review found that consumer audits previously identified a
checkpoint by its path and policy only. Future audits hash the single-file
Llama weights, model/quantization configs, tokenizer files, and optional chat,
generation, and special-token files before model loading, then recheck them
after the consumer checks. Missing optional files are recorded explicitly;
missing required files or broken symlinks fail. The audit source SHA256 is
also recorded and must stay unchanged during execution.

The full saved GSM8K pytest entry point now requires these hashes and compares
them with the current checkpoint before accepting its consumer/accuracy
bundle. A replaced weight file at the same path therefore cannot reuse old
consumer evidence. Historical audits without hashes remain historical evidence
and require a new consumer audit for this final gate. This does not backfill
hashes into old records or claim that old evaluation files recorded file hashes.

The combined CPU consumer/evidence suite passes 94 checks with one unconfigured
full-evaluation skip. New cases include changed weights/configs/tokenizers,
added optional files, changed symlink targets, missing files, and rejection of
hashless saved evidence. Log: `/tmp/w4a4-saved-identity-cpu.log`. The currently
running fit and queued training/evaluation hashes are unchanged. Full GB10
consumer execution of the additional file checks will occur in the queued
saved-checkpoint audits; it has not been claimed from these CPU cases.

### Longer disjoint arithmetic-chat records prepared

A separate artifact with 24 worked examples per document is ready at
`/root/models/w4a-calibration/procedural-arithmetic-disjoint-4096-v2-long24`.
It uses the unchanged version-2 generator and seed 9431, with 4,096 fitting and
512 selection records. Manifest SHA256:
`a9c97bfa997515ea88fc15f08aa2fb70e837137805a2c201dffdb72b203b1ea5`.
The same complete 45,748-question/14-split exclusion registry rejects 506
candidate records; accepted records have no detected lexical overlap.

All 110,592 worked-example answers pass independent exact rational-arithmetic
checks. The fitting and selection questions are disjoint within this artifact.
They also remain disjoint across roles when compared with the shorter v2
artifact: neither corpus's fitting questions occur in the other's selection
set. Reusing the seed-based question partition ensures this property even
when question generation repeats an example across experiments.

With the existing checkpoint chat template and a 2,048-token cap, the first
512 fitting records contain 905,863 tokens, 1,712–1,820 per record. The first
32 selection records contain 56,757 tokens, 1,747–1,807 per record. None are
truncated. Direct chat tokenization matches the calibration path exactly for
all 544 records; eight also match Tokenicer text, IDs, and masks, with no
duplicate BOS tokens. This extends available calibration positions beyond
the shorter artifact's sub-1,024-token examples.

Evidence files in the quality directory:

- `procedural_arithmetic_disjoint_4096_v2_long24_verified.json`
- `procedural_arithmetic_disjoint_4096_v2_long24_chat_verified.json`
- `procedural_arithmetic_long24_cross_partition_verified.json`

Preparation and verification ran in separate CPU-only scopes capped at 2 GiB
and 3 GiB respectively, with zero swap. Observed peaks were about 349 MiB and
883 MiB, with no OOM events. Logs are `/tmp/w4a4-arithmetic-long24-prepare.log`
and `/tmp/w4a4-arithmetic-long24-verify.log`.

This artifact has not been used for fitting or candidate selection. A future
2,048-token training experiment requires its own guarded GB10 memory/lifecycle
smoke and full downstream acceptance. The active WikiText run and already
queued 1,024-token arithmetic-chat experiment retain their original inputs.

### Completed longer WikiText fit and GB10 weight suite

The 512-step exact-dual-forward WikiText run completed all updates and 515,403
actual training tokens. It passed 1,568 exact A4 forward comparisons and
1,568 exact native comparisons, with all 112 projection gradients covered in
each of the 512 backwards per mode. Initial native loss was exactly zero.

Best held-out A4 KL was 0.156334 at step 464, versus 0.219199 initially. The
final step had A4 KL 0.156818 and native KL 0.0134155, with 1,105,718 changed
INT4 values. No changed snapshot met the unchanged native KL limit of 0.0005.
The selected export is therefore step zero with zero changed INT4 values.
These results establish optimization progress and rejection by the selection
screen, not a recovered downstream accuracy result. This run did not retain
the separately introduced diagnostic snapshot because that option was added
after it launched.

Its saved checkpoint passes full consumer coverage: 16 decoder layers, 112
Linears, 381 handoffs, 720 independent decodes, and 336 independent GEMMs. The
maximum independent decode error is 4.76837158203125e-7. The audit also completed
the new before/after checkpoint file-hash checks and records its source hash.
Peak run memory was 44.3387 GiB, minimum observed root headroom 7.6246 GiB,
with zero swap and no OOM/high/max events. The completed outcome is
`nvfp4_v4_dual_forward_codecell_r512_v32_s1024_t512_gc_continue_outcome.json`
in the quality directory. The evaluation coordinator correctly skipped a
duplicate full GSM8K evaluation of unchanged weights.

The queued GB10 weight suite then passed all 55 tests, including its 13 CUDA
cases, in 15.60 seconds. Log: `/tmp/w4a4-master-cell-radius-gpu.log`.
The radius-0.49/diagnostic-export full-model smoke has now started under the
same memory guard. Its result and the subsequent arithmetic-chat fit remain
pending; no W4A4 accuracy acceptance is claimed.

### Wider-radius diagnostic export passes full-model smoke

The radius-0.49 smoke completed all eight updates. Each mode passed 18 exact
forward comparisons and eight gradient-coverage checks over all 112
projections. Initial native loss was exactly zero, and the strict export
remained at step zero with unchanged INT4 values. The separate diagnostic
export selected step eight: A4 loss 0.194920 versus 0.234684 initially, native
loss 0.0202744, and 251,324 changed INT4 values. It is labeled diagnostic-only
and fails the native selection limit; these two selection records do not
establish downstream accuracy recovery.

Both saved outputs pass full consumer coverage: 381 handoffs, 720 independent
decodes, and 336 independent GEMMs each, with maximum decode error
4.76837158203125e-7. Their checkpoint/source file hashes are recorded. Smoke
peak memory was 44.8315 GiB, minimum observed root headroom 6.1164 GiB, and
swap/OOM/high/max counts remained zero.

An additional CPU comparison independently checks all 595 saved tensors across
the source, strict, and diagnostic files. Every strict-export tensor equals
the source. All 483 non-code tensors in the diagnostic export—including GPTQ
scales, zero points, group indices, embeddings, and norms—also equal the source.
All 112 qweight tensors remain INT32. A direct eight-nibble comparison of each
packed word counts exactly 251,324 changed INT4 values, matching the report.
Whole-file hashes remain unchanged across the audit. Evidence:
`nvfp4_v4_radius049_diagnostic_smoke_r4_v2_s1024_t8_tensor_audit.json`
in the quality directory. Script/log:
`/tmp/w4a4-verify-smoke-export-tensors.py` and matching `.log`.

With the smoke and saved audits complete, the queued 512-step arithmetic-chat
experiment has started. It uses the verified shorter v2 corpus (512 fit/32
selection records, 1,024-token cap), exact dual forwards, radius 0.49, and
separate strict/diagnostic exports. Process identity and source snapshot are
recorded under prefix
`nvfp4_v4_chat_arithmetic_radius049_r512_v32_s1024_t512` in the quality directory.
It is currently building its lossless teacher cache. Full paired GSM8K
acceptance remains pending and retains the original frozen reference floor.

### Arithmetic-chat run hit its scope limit; fixed-length retry under validation

The first arithmetic-chat fit terminated after ten updates with exit 137.
The systemd user journal explicitly records an OOM kill in
`gptqmodel-w4a-gb10-3115847.scope`; its launch cap was 45,412 MiB (about
44.3 GiB). This was below the preceding smoke's measured 44.8315 GiB peak.
The last five-second memory sample still showed 9.1653 GiB of root headroom
and zero swap. Its sampled event counters had not yet captured the kill;
those samples must not override the terminal journal evidence. No host-reserve
stop message was recorded. The initial native loss was exactly zero and the
initial A4 loss was 0.214515. There was no completed held-out selection or
exported checkpoint from this failed run. Evidence is recorded in
`nvfp4_v4_chat_arithmetic_radius049_r512_v32_s1024_t512_failure.json`.

Variable token lengths may contribute allocator fragmentation, but this run
does not establish that as the cause. The retry uses a uniform 875-token
prefix, the shortest verified length across the selected fitting records.
It retains all 512 fit/32 selection records and avoids introducing padding or
attention-mask changes. This trims at most 51 tokens per record: 510 fitting
and all 32 selection records are truncated. Fitting now contains exactly
448,000 tokens (versus 459,431), and selection 28,000 (versus 28,775). The
tokenizer audit verifies exact prefix IDs for all 544 records and Tokenicer
agreement for eight. Evidence:
`procedural_arithmetic_disjoint_4096_v2_chat_fixed875_verified.json`.

Each optimization-step log now includes PyTorch CUDA allocated/reserved bytes
and their peaks alongside the external cgroup monitor. These counters cover
PyTorch's allocator, not every driver or host allocation; reading them does
not change the allocator or empty caches. The CPU adaptation suite passes
43 cases with 13 CUDA cases skipped. An initial collection indentation error
was corrected before this successful run. Log:
`/tmp/w4a4-memory-counter-cpu-fixed.log`.

The fresh fixed-length smoke uses 16 fitting records, four selection records,
32 updates, eight-step warmup, and eight-step evaluation intervals. It retains
exact dual forwards, full gradient coverage, radius 0.49, and separate
diagnostic export. Expected checks are 52 exact forwards and 32 full backwards
per mode. Its fresh guard budget is 53,961 MiB with the same 2 GiB host reserve;
both the budget and sequence shape differ from the failed run, so memory
improvements cannot be attributed to fixed length alone.

Coordinator: `/tmp/w4a4-fixed875-smoke-coordinator.py`.
Artifacts use prefix `nvfp4_v4_chat_fixed875_smoke_r16_v4_s875_t32`.
The next full fit is queued through
`/tmp/w4a4-arithmetic-chat-fixed875-after-smoke.py`, using prefix
`nvfp4_v4_chat_arithmetic_fixed875_radius049_r512_v32_s875_t512`.
It requires a complete successful smoke/audit and a freshly computed guarded
launch budget exceeding the smoke's observed peak. The guard itself and its
2 GiB/zero-swap policy are unchanged. Full 1,209-row downstream pairing and
the frozen native acceptance floor also remain unchanged.

### Fixed-length smoke passed; targeted cache release enabled the full retry

The 32-step fixed875 smoke completed with 52 exact forward checks and 32 full
gradient-coverage checks per mode. Its best diagnostic is step 32, with A4
loss 0.0950492, native loss 0.0303983, and 1,081,906 changed INT4 values. The
strict export remains unchanged at step zero. Both saved outputs pass full
consumer coverage: 381 handoffs, 720 independent decodes, and 336 independent
GEMMs each. The diagnostic still fails native preservation; the four selection
records are lifecycle evidence, not a downstream acceptance result.

PyTorch reserved CUDA memory stayed at 32,585,547,776 bytes across the logged
updates. Total scope memory nevertheless peaked at 50,786,344,960 bytes
(47.3 GiB), including export. The fresh-budget gate correctly refused the
initial full retry because its available scope budget was below that peak.
Root-cgroup inspection showed about 62.8 GiB in file cache, not an active
training process consuming that memory.

With every previous trainer terminal, POSIX_FADV_DONTNEED was applied first
to completed task-owned teacher-cache files, reclaiming only about 94 MiB.
Applying it to five completed, task-created checkpoint tensor files then
reduced file cache by about 7.25 GiB and increased observed root headroom by
about 7.47 GiB. No files were deleted or modified and no global cache flush
was used. Evidence:
`w4a_completed_teacher_cache_release_after_chat_oom.json` and
`w4a_completed_checkpoint_cache_release_after_chat_oom.json` in the quality
directory.

The restart then passed the same budget gate: 57,740,849,152 bytes available
to the scope versus the smoke's 50,786,344,960-byte peak, retaining the separate
2 GiB host reserve. Its actual guard launch cap is 55,062 MiB. The original
source hashes, audited corpus, fixed875 prefix checks, diagnostic distinction,
and full downstream acceptance rules are retained. No allocator option or
memory-guard threshold was changed.

The full 512-step retry is now live through
`/tmp/w4a4-arithmetic-chat-fixed875-resume.py`, with coordinator log of the same
basename and training log `/tmp/w4a4-chat-arithmetic-fixed875-train.log`.
The original fixed875 queue record and failed budget check are retained;
a separate resume record identifies this launch. Artifacts continue to use
prefix `nvfp4_v4_chat_arithmetic_fixed875_radius049_r512_v32_s875_t512`.
Accuracy acceptance remains pending.

### Benchmark-disjoint calibration contract

The A4 producer-scale pass and the GPTQ solve both consume
`tests/models/w4a_calibration_data.py`. That module already refused legacy
unchecked input, re-hashed every artifact file on load, and rejected any
calibration article whose normalized text matched an evaluation question. What
was missing was a machine-checked link between the exclusion registry and the
evaluation harness: the two lists agreed only by inspection, so a new scored
task could be added without extending the exclusions.

The contract is now explicit and fail-closed. `evaluated_dataset_configs()`
derives the required dataset/config pairs from `w4a_quality_regression.TASKS`;
`excluded_dataset_configs()` derives the covered pairs from `EXCLUSIONS`;
`require_evaluation_exclusions()` raises when the requested set is not a subset.
`load_calibration_artifact(..., required_evaluations=...)` applies that check
after the existing coverage/hash/question audit, and
`tests/models/w4a_nvfp4_calibrate.py` passes the harness registry, so a
producer-calibration run cannot fit on a benchmark it will later be scored on.
The check runs before any GPU model is allocated or any activation is observed,
and the report records `evaluation_exclusions`.

Current coverage is 14 dataset/config/split rows: GSM8K train and test,
GSM8K-Platinum test, MMLU-Pro test and validation, MMLU test/validation/dev,
and ARC-Challenge/ARC-Easy test/train/validation. The harness currently scores
ARC-Challenge and GSM8K-Platinum, both covered. MMLU-Pro is covered even though
the harness does not yet score it, matching the stated requirement that
calibration stay disjoint from the MMLU-Pro lane. Both on-disk artifacts
(`wikitext103-disjoint-v1` and `wikitext103-disjoint-4096-v1`) already carried
this full coverage, so the earlier producer-max64 run was legitimately
disjoint; the new check prevents future drift rather than invalidating it.

### Producer-scale calibration result and queued retry

The first separate A4 calibration pass over an existing GPTQ checkpoint has now
finished its full-row gate. `nvfp4_v4_producer_max64` calibrated 65 producer
boundaries from 64 disjoint fitting articles (115,465 tokens) and 16 held-out
articles, then scored **352/1,209 (0.2911)** on full GSM8K Platinum against the
frozen same-weight W4A16 **524/1,209 (0.4334)**. The paired gate records
**-11.99 pp, `confirmed_regression`**. Its report confirms exact reload probe
logits, unchanged native tensor digests, and the identical source weight file,
so the failure is accuracy, not plumbing: maximum-based global producer scales
do not recover the 4-bit activation grid.

The A4-aware plain-GPTQ checkpoint
`Llama-3.2-1B-Instruct-W4A-NVFP4-v4-gptq-full64-concat2048-quality-2026-09-26`
has no frozen producer scales, so its lane is dynamic A4. Its full-row
W4A16 reference is 502/1,209 (0.4152) and its dynamic-A4 lane is running. A
separate producer-calibration pass over that same checkpoint, using the
verified `wikitext103-disjoint-4096-v1` artifact and the same 64/16 split, is
queued in `/tmp/w4a4-v4-gptq-producer-validate.sh`. It waits for the dynamic-A4
evaluation to release the GB10 lock, verifies the report and manifest hash,
freezes the prompt date, and runs the paired full-row gate. Neither the queued
run nor the dynamic-A4 lane is accepted accuracy until that gate completes.

### A4-aware plain GPTQ, dynamic A4: best NVFP4 result so far

The dynamic-A4 lane for the A4-aware plain-GPTQ checkpoint has now finished all
1,209 GSM8K Platinum rows. It scores **412/1,209 (0.34078)** against its own
same-weight W4A16 **502/1,209 (0.41522)**. The paired gate records
**-7.44 pp, `confirmed_regression`** with a 95% CI of [-10.08, -4.81] and
McNemar exact p = 4.6e-8. There were 180 W4A16-only correct and 90 A4-only
correct; 270 correctness flips and 797 answer changes.

That is the strongest NVFP4 number measured so far. The previous best
(`gsm_nvfp4_token_norm_v4_paired.json`) was -10.34 pp, so making the GPTQ solve
activation-aware recovered about 2.9 pp. The remaining gap is still a
statistically confirmed regression and far outside the 2-point acceptance
budget. For contrast, the FP8 lane on the same harness is +0.74 pp and
`within_budget`, which isolates the failure to the 4-bit activation grid rather
than the pipeline.

Evidence: `gsm_nvfp4_v4_gptq_full64_w4a4_full.json`,
`gsm_nvfp4_v4_gptq_full64_w4a16_full.json`, and
`gsm_nvfp4_v4_gptq_full64_paired.json` in `/root/models/w4a-quality/`.

### Next measurement: activation-aware GPTAQ weight solve

GPTAQ adds the native-versus-rounded input cross term to the GPTQ objective, so
the INT4 codes compensate for the E2M1 grid during the solve instead of being
fitted against unrounded activations. It is already implemented
(`gptqmodel/quantization/gptaq.py`) and wired through `QuantizeConfig`,
`gptq_processor.py`, and the `NativeProcessor` split in `stage_layer.py`, but
the 1B acceptance test deliberately hard-sets `GPTAQ = None`. The experiment
test `tests/models/test_w4a_nvfp4_gptaq_experiment.py` leaves that acceptance
test untouched and records `meta.gptaq` with `meta.foem is None`.

The queued chain `/tmp/w4a4-gptaq-validate.sh` waits for the producer
calibration to release the GB10 lock, builds the GPTAQ checkpoint from the same
disjoint `wikitext103-disjoint-4096-v1` artifact and 64/16 split, verifies the
saved metadata and 112 packed INT4 qweights, then runs its own same-weight
W4A16 lane, the A4 lane, and the paired gate. Acceptance still requires the
full-row paired drop to stay within 2 points.

### Calibration disjointness is now a loader default

Every A4 calibration entry point fits only text that the verified loader
returns: the A4 calibrate pass, scale QAD, weight QAD, norm QAT, producer
reconstruct, held-out trace, and the 1B test all route through
`load_calibration_artifact` (directly or via `_calibration_ids`). The loader
rechecks file hashes, requires the exclusion coverage to equal the full
`EXCLUSIONS` registry (14 dataset/config/split rows), rebuilds the evaluation
question index, and rejects any fit or selection article that shares a
13-word normalized n-gram with an evaluation question.

Previously only the calibrate entry point passed the harness registry
explicitly; the rest relied on the loader's internal coverage check. That
assertion is now the loader default, so a caller that omits
`required_evaluations` still cannot fit on a benchmark the quality harness can
score. If a new task is added to `w4a_quality_regression.TASKS` without a
matching `EXCLUSIONS` entry, every load fails closed.

Verified on the artifacts actually used: `wikitext103-disjoint-4096-v1`
(`manifest_sha256=249d79b5…`, the artifact recorded by the producer
calibration report) and `procedural-arithmetic-disjoint-4096-v2` both satisfy
harness ⊆ registry ⊆ artifact coverage, with `accepted_overlap_count == 0`.
Rejected articles include real MMLU, MMLU-Pro, and ARC question overlaps. The
base W4A16 checkpoint and the A4 passes share this one artifact, so the
inherited INT4 weights are eval-disjoint too. `tests/models/test_w4a_calibration_data.py`
passes 61/61, including the new default-registry regression test.

### Producer-calibrated A4 also fails the gate

The disjoint producer-scale pass completed over the A4-aware plain-GPTQ
checkpoint: 65 boundaries measured, 64 fit and 16 selection articles,
`weight_updates == 0`, and the checkpoint weight file byte-identical to the
base (`weight_file_sha256=feb36f18…`). The full-row GSM8K Platinum gate then
returned `w4a16=0.41522` (502/1209) versus `w4a_float=0.34078` (412/1209),
`delta_pp=-7.444`, 95% CI `[-10.05, -4.84]`, McNemar `p=3.2e-8`,
`verdict=confirmed_regression`. Not acceptance.

The A4 score is identical to the dynamic-A4 lane to all reported digits
(412/1209). Producer calibration changed the saved activation scales but not a
single evaluated answer, which is the open question for the next iteration:
either the eval path is not consuming the calibrated producer scales, or the
scale choice is numerically inert at this grid. This must be resolved before
another A4 accuracy attempt, because it determines whether calibration work
can affect the outcome at all.

Evidence: `gsm_nvfp4_v4_gptq_full64_producer_cal_full.json` and
`gsm_nvfp4_v4_gptq_full64_producer_cal_paired.json` under `/root/models/w4a-quality/`;
report at `/root/models/Llama-3.2-1B-Instruct-W4A-NVFP4-v4-gptq-full64-producer-cal/w4a_producer_calibration_report.json`.

### GPTAQ chain relaunch

The queued GPTAQ chain had aborted before starting: `run_w4a_gb10_safe.sh`
did not list `tests/models/test_w4a_nvfp4_gptaq_experiment.py` in its allowed
test paths, so it exited with `Only the W4A tests, quality tools, and
benchmark are supported.` The allowlist now includes the GPTAQ experiment test
(and was rebuilt to contain only paths that exist on disk). The chain is
relaunched and running its quantization stage.

### Why producer calibration cannot move the score

The identical 412/1209 result is explained by the arithmetic of the NVFP4
packing kernel, not by a broken save path. The scales do survive: the producer
checkpoint carries 65 `activation.global_scales` entries, `QuantizeConfig`
round-trips them into `activation_global_scales`, `gptqmodel/utils/model.py`
passes them to `install_w4a_llama_stream`, and each boundary's
`NVFP4BoundaryQuantizer` overwrites `global_scale` with its calibrated value
for every recipe, including `least_squares`. Boundary names also line up:
`llama_nvfp4_boundaries` yields `(self_attn, "output")`, `(mlp, "product")`,
`(layer, "attention_residual")`, and `(layer, "output")`, which are exactly the
owners `pack_boundary` is called on.

The kernel then cancels the global scale out of the product. In
`nvfp4_pack_and_swizzle` the block scale is
`local = clamp(maximum / (6 * global_scale), 2**-9, 448)`, the stored scale is
`local.to(e4m3) * global_scale`, and the code is
`e2m1(x / (local.to(e4m3) * global_scale))`. Because the same `global_scale`
multiplies the stored block scale and divides the input, it cancels: the
reconstructed value is `maximum/6` regardless of the FP32 global scale. The
only residual effect is where `local` lands on the E4M3 grid, a second-order
perturbation of the per-block scale rounding. That is why replacing a dynamic
per-call amax scale with a calibrated per-boundary maximum changed no evaluated
answer.

`nvfp4_global_scale` confirms the magnitude: for `least_squares` it returns
`amax / (448 * 4)`, so the dynamic and calibrated scales differ only by the
amax of the current batch versus the calibration corpus, both of which land
`local` in the same well-conditioned range.

Consequence for the roadmap: per-boundary FP32 global-scale calibration is a
saturation guard, not an accuracy lever, in this design. The remaining error is
dominated by the E2M1 grid itself, the per-16-value E4M3 block rounding, and
re-quantization at every module boundary. Those are the targets that can still
move GSM8K Platinum: weight-side grid compensation (GPTAQ, in flight) and
preserving the residual stream in compute dtype so fewer tensors are pushed
through the 4-bit grid.

### GPTAQ is dropped for W4A4

GPTAQ is out of scope for W4A4. Its activation-aware cross term retains extra
per-module state through the solve and pushes host memory past the 2 GiB
headroom contract every GB10 W4A run must hold, so it will not be tested or
supported as an A4 path. The allowlist entry was removed again, restoring
`run_w4a_gb10_safe.sh` to ten supported test paths.

The experiment did finish once before being stopped: a valid checkpoint with
`quant_method=gptq`, `bits=4`, `pack_dtype=int32`, `activation={version:4,
mode:w4a_nvfp4, recipe:least_squares}`, `meta.gptaq={"alpha":0.25}` and
`meta.foem=None` was produced (quantization `2 passed` in 19:15). No A4
evaluation ran against it, so it establishes nothing about accuracy. The
directory is parked at
`/root/models/archive-unsupported-gptaq-Llama-3.2-1B-Instruct-W4A-NVFP4-v4-full64`
and must not be cited as an acceptance artifact.

This removes the weight-side grid-compensation lever. Combined with the
global-scale result above, the only remaining accuracy lever that stays inside
the 2 GiB contract is reducing how much of the stream is pushed through the
4-bit grid, chiefly preserving the residual stream in compute dtype.

### How ModelOpt, llm-compressor, and AutoRound handle W4A4

Sources read locally: NVIDIA ModelOpt (`/tmp/modelopt-reference-20260926`),
llm-compressor (`/tmp/llm-compressor-plan-review`), AutoRound
(`/tmp/auto-round-research`).

**The global scale cancels in all three.** AutoRound's reference is explicit
(`auto_round/data_type/nvfp.py`): `global_scale = 448*6/amax`,
`scale = global_scale*(vec_max/6) = 448*vec_max/amax`, and
`output_scale = global_scale/scale = 6/vec_max`, so the reconstructed value is
`e2m1(x*6/vec_max) * vec_max/6`. ModelOpt's kernel takes a pre-computed
`scale = amax/6.0` (`nvfp4_quant.py`). The FP32 global scale therefore only
positions the E4M3 block scale inside its representable range; it is a
saturation guard. This confirms the earlier finding, and it means the NVFP4
"headroom" calibrators (`NVFP4ActHeadroomCalibrator`, percentile anchor x rho)
exist to stop block scales flushing to subnormal, not to raise accuracy.

**What does move accuracy is the effective per-block grid.** llm-compressor
PR #2950 replaced discrete four-over-six with `nvfp4_expanded_mse`, which
MSE-searches `expand` (default 1.8) down to about `0.8` times the observed
per-group range in 112 steps, and states this "is a more effective way to take
advantage of the same range expansion benefit that fouroversix is based on".
Range expansion changes `vec_max` and therefore the grid step `vec_max/6`; that
is a real lever, unlike the global scale.

**Four-over-six is a weight-side technique.** ModelOpt applies
`nvfp4_four_over_six` only to `*weight_quantizer` and plain `nvfp4` to
`*input_quantizer`; the preset comment says "4/6 per-block M=6 vs M=4 is
selected by MSE (amax multipliers [1.0, 1.5]) on the static weight quantizers;
dynamic activation quantizers are not MSE-calibrated". Our `four_six` recipe
sits on activations, the opposite placement.

**The production W4A4 recipes are mixed-precision, not uniform 4-bit.**
NVIDIA's tuned per-model recipe
(`models/Qwen/Qwen3.8-27B/ptq/nvfp4_w4a4_mlp_fp8_attn_max.yaml`, described as a
"5.5-bit NVFP4-max AutoQuantize sweep" assignment) puts NVFP4 on `mlp.*` and
FP8 on `self_attn.*_proj`. ModelOpt ships the same idea as reusable units
(`attention_qkv_fp8`, `w4a8_nvfp4_fp8`) and presets (`nvfp4_mlp_only`,
`nvfp4_omlp_only`, `nvfp4_w4a4_mlp_fp8_attn_max`). `default_disabled_quantizers`
also keeps `lm_head`, routers, gates, and embeddings out of quantization.

**Contrast with our setup.** We quantize all seven Llama projections to NVFP4
activations, including `q/k/v/o`, and we have no FP8 lane. ModelOpt's own
attention unit is FP8-only, and its W4A4 model recipes never put 4-bit
activations on attention. Our FP8-lane result (+0.74 pp, `within_budget`)
already shows attention is the sensitive part.

**Ranked lessons for us.**
1. Step attention activations down to FP8 (`o_proj` first, then `q/k/v`) while
   keeping MLP at NVFP4. This is the change all three projects agree on and it
   needs no weight-format change, since weights stay native INT4.
2. Replace activation-side `four_six` with continuous MSE range expansion over
   the effective per-block scale (llm-compressor's `nvfp4_expanded_mse`), which
   is a genuine grid lever.
3. Keep `lm_head` and any router or gate out of A4, matching
   `default_disabled_quantizers`.
4. Do not invest further in per-boundary global-scale calibration; all three
   upstream implementations treat it as a representability guard.

## Mixed FP8-attention / NVFP4-MLP stream (implemented)

Lesson 1 from the upstream review is now implemented. A single W4A stream can
declare a per-boundary activation policy: attention boundaries carry FP8 and
MLP boundaries carry NVFP4 over the *same* native GPTQ INT4 weights. No saved
weight tensor changes format; the split only changes activation staging.

### Configuration

`activation` accepts an optional `attention` sub-policy:

```python
activation={"version": 3, "mode": "w4a_nvfp4", "attention": {"mode": "w4afp8"}}
```

Constraints enforced by `QuantizeConfig`:

- `activation.attention` only refines an NVFP4 stream, and only supports
  `mode` (`w4afp8` or `w4a_nvfp4`) plus an optional NVFP4 `recipe`.
- FP8 attention must not carry an NVFP4 recipe.
- Version must be 3 or 4; version 2 decodes every Linear output and cannot
  express a mixed carrier.

### Mechanics

- `w4a_llama_stream` records a `(mode, recipe)` policy on each producer
  boundary. `_boundary_group` maps the post-attention residual and the MLP
  product to the MLP group; every other boundary belongs to attention.
- `W4ANVFP4Linear` now stages a derived `_weight_e4m3` plane in `post_init`
  from `_centered_int4_codes()` and dispatches an incoming `w4afp8` carrier to
  `fp8_linear_prepacked`, mirroring `W4AFP8Linear`'s encoded path. The E4M3
  plane is a non-persistent buffer, so it is never saved and never changes the
  INT4 checkpoint.
- Version-4 NVFP4 producer quantizers are attached only to MLP boundaries; FP8
  attention boundaries pack dynamically and therefore need no calibrated
  global scale.

### Boundary contract (tiny Llama, version 3)

`tests/models/test_w4a_tiny_llama_lifecycle.py::test_tiny_llama_mixed_fp8_attention_nvfp4_mlp`
quantizes, saves, reloads, and asserts that `q_proj` consumes `w4afp8` while
`gate_proj` consumes `w4a_nvfp4`, that all 14 projections keep `int32`
`qweight`, and that the derived E4M3 plane matches the INT4 weight size.

Status: implementation and boundary contract verified on the tiny model. The
full 1,209-row paired GSM8K Platinum gate on the real Llama 3.2 1B checkpoint
is the remaining acceptance step.

### Acceptance result: mixed FP8-attention / NVFP4-MLP (Llama 3.2 1B)

Full 1,209-row GSM8K Platinum paired gate against the same-weight W4A16
reference (`gsm_nvfp4_v4_gptq_full64_w4a16_full.json`, 0.41522):

| lane | score | delta vs W4A16 |
| --- | --- | --- |
| W4A16 (same weights) | 0.41522 (502/1209) | -- |
| uniform dynamic NVFP4 A4 | 0.34078 (412/1209) | -7.44 pp |
| FP8 attention + NVFP4 MLP | 0.36559 (442/1209) | -4.96 pp |

Paired verdict: `confirmed_regression` (95% CI [-7.53, -2.39] pp, McNemar
p=2.0e-4). The attention split recovers +2.48 pp over uniform A4 and confirms
that attention is the dominant 4-bit sensitivity, but it is not yet inside the
2 pp budget. Next lever: replace the activation-side `four_six` range heuristic
with a continuous MSE-fit range (llm-compressor's `nvfp4_expanded_mse`).

### Reference-backed carrier: residual stream stays in compute dtype

The mixed split removed attention from the 4-bit grid but left a second,
larger defect in place: the encoded stream was authoritative for the residual.
`_add_stream` summed `left.decode()` (a dequantized 4-bit value) with the branch
output, so every one of the 32 residual adds in the 16-layer stack injected
fresh 4-bit rounding into the running hidden state, and `input_layernorm`
computed its variance from that corrupted value.

The fix separates the two roles a carrier plays:

- `codes`/`scales`/`global_scale` remain the hardware GEMM operand. Every
  selected projection still consumes a genuine FP8 or NVFP4 operand; nothing is
  decoded to BF16 before a GEMM.
- `reference` retains the exact compute-dtype value the operand was packed
  from. `W4AActivation.exact()` returns it, and only the residual adds, the
  RMSNorm inputs, the fused-token-rescale path, and the unquantized head edge
  call it. `decode()` is unchanged and still returns the dequantized hardware
  value, so every existing audit, replay, and kernel test keeps measuring the
  quantization error it was written to measure.

Pack sites now attach the exact source value: `_as_stream` (layer entry),
`_add_stream` (both residual adds), `_norm_forward`, `_attention_forward`
(o_proj operand), and `_mlp_forward` (down_proj operand). `rescale_tokens`
scales the reference alongside the token multiplier so the version-4 fused
RMSNorm stays consistent.

This matches how ModelOpt, llm-compressor, and AutoRound actually deploy W4A4:
quantization is applied at GEMM operands only. The residual stream, RMSNorm,
attention math, and SiLU product stay in compute dtype. The previous behaviour
quantized strictly more than the reference recipe and charged the difference to
"activation quantization".

### Acceptance result: reference-backed carrier (Llama 3.2 1B)

Same checkpoint, same 1,209 GSM8K Platinum rows, same paired gate:

| lane | score | delta vs W4A16 | verdict |
| --- | --- | --- | --- |
| W4A16 (same weights) | 0.41522 (502/1209) | -- | reference |
| uniform dynamic NVFP4 A4 | 0.34078 (412/1209) | -7.44 pp | confirmed_regression |
| FP8 attn + NVFP4 MLP (pre-carrier) | 0.36559 (442/1209) | -4.96 pp | confirmed_regression |
| FP8 attn + NVFP4 MLP + exact residual | 0.38048 (460/1209) | -3.47 pp | inconclusive |

Paired statistics for the carrier run: 95% CI [-5.95, -1.00] pp, McNemar
p=7.2e-3, 138 base-only vs 96 float-only correct, 234 correctness flips.

The reference carrier recovered **+1.49 pp** with no change to any saved
weight tensor and no change to the set of quantized GEMM operands. This
confirms the residual stream had been over-quantized: the previous code charged
4-bit rounding to the running hidden state at every one of the 32 residual adds
and at every RMSNorm input, which no upstream reference implementation does.

### Where the remaining error lives

Per-boundary relative quantization error, measured on real Llama 3.2 1B
activations through the installed stream (`reference` vs `decode`):

| boundary | mode | recipe | mean relative error |
| --- | --- | --- | --- |
| `mlp.gate_proj` input | NVFP4 | least_squares | 0.08391 |
| `mlp.up_proj` input | NVFP4 | least_squares | 0.08391 |
| `mlp.down_proj` input | NVFP4 | least_squares | 0.07986 |
| `self_attn.q_proj` input | FP8 | -- | 0.02604 |
| `self_attn.o_proj` input | FP8 | -- | 0.02631 |

The FP8 attention lane carries 3.2x less error and independently measured
+0.74 pp versus W4A16 (inside the 2 pp budget). The whole remaining regression
is the NVFP4 grid on the two MLP operands, which is exactly the split NVIDIA's
`nvfp4_w4a4_mlp_fp8_attn_max` recipe prescribes.

### Is the scale search leaving anything on the table?

For each captured MLP operand the block scale was re-fit four ways, then
compared against an exhaustive search over all 126 positive finite E4M3 scales:

| recipe | mean relative error |
| --- | --- |
| `nvidia` (max/6, no refinement) | 0.0952 - 0.0971 |
| `least_squares` (current) | 0.0841 |
| `least_squares_grid` | 0.0812 |
| brute force over all 126 E4M3 scales | 0.0812 |

`least_squares_grid` reproduces the exhaustive optimum exactly, so the block
scale selection is already provably optimal for the NVFP4 format. The best
remaining improvement from scale search alone is 3.4% of the activation error,
which cannot close a 1.47 pp gap.

This also cross-checks the Triton packing kernel against the Torch reference:
the runtime `decode()` error (0.0839) matches the offline reference fit
(0.0841), so the kernel is not the source of the loss.

### Why the residual gap is intrinsic here

Three facts bound what any scale-only change can recover:

- The 4-bit activation grid is the entire cost. Attention at FP8 is free.
- The block scale is already at the exhaustive E4M3 optimum.
- The hidden state and the SiLU product are already Hadamard-rotated
  (`rotate_embeddings` / `rotate_mlp_input` fold the rotation into the GPTQ
  weights, `down_proj` applies the online Hadamard), so the activation
  distribution is already the outlier-free one that QuaRot/SpinQuant produce.

The remaining levers all change something other than the scale recipe: finer
weight groups (the NVFP4 weight path is hard-coded to group 128), weight-side
error compensation against the activation error (GPTAQ, dropped for memory), or
learned/trained activation scales. None of them is a scale-only change.

### Acceptance result: optimal block-scale search (`least_squares_grid`)

The scale-search lever was the last sound, format-preserving option. Same
checkpoint, same 1,209 GSM8K Platinum rows, same paired gate, only the NVFP4
activation recipe changed (`least_squares` -> `least_squares_grid`):

| lane | score | delta vs W4A16 | verdict |
| --- | --- | --- | --- |
| W4A16 (same weights) | 0.41522 (502/1209) | -- | reference |
| uniform dynamic NVFP4 A4 | 0.34078 (412/1209) | -7.44 pp | confirmed_regression |
| FP8 attn + NVFP4 MLP (pre-carrier) | 0.36559 (442/1209) | -4.96 pp | confirmed_regression |
| FP8 attn + NVFP4 MLP + exact residual | 0.38048 (460/1209) | -3.47 pp | inconclusive |
| + optimal `least_squares_grid` scales | 0.39206 (474/1209) | -2.32 pp | inconclusive |

Paired statistics for the final run: 95% CI [-4.73, +0.10] pp, McNemar
p=0.070, 125 base-only vs 97 float-only correct, 222 correctness flips,
`statistically_detectable_drop=false`, `material_regression=false`.

Cumulative recovery from the original uniform-A4 result is **+5.12 pp** with
every saved weight tensor byte-identical throughout.

### Verdict on W4A4

The remaining 2.32 pp gap is **not** a scale-selection problem. Three
independent measurements establish that:

1. `least_squares_grid` reproduces an exhaustive brute-force search over all 126
   positive finite E4M3 block scales exactly (0.0812 vs 0.0812 mean relative
   error), so the per-block scale is provably optimal for the NVFP4 format.
2. The runtime Triton packer matches the Torch reference fit to 0.2% relative,
   so the kernel is not the source of the loss.
3. The activations reaching the MLP operands are already Hadamard-rotated by
   the existing rotation pass, so they are already the outlier-free
   distribution that QuaRot/SpinQuant are designed to produce.

What remains is the intrinsic cost of a 4-bit activation grid on the two MLP
operands. The paired test cannot distinguish it from zero at the 95% level.

For reference, the same harness measures FP8 activations everywhere at
+0.74 pp (inside the 2 pp budget), so the cost is specifically the NVFP4
activation grid on the MLP, not activation quantization in general.

Any further improvement requires changing something other than the activation
scale recipe: finer weight groups (the NVFP4 weight path is hard-coded to
group 128), weight-side error compensation against activation error (GPTAQ,
dropped for memory), learned/trained activation scales, or per-layer precision
selection.

### Per-layer MLP precision selection

The last sound lever is per-layer precision selection: keep the NVFP4 MLP
default on most layers and promote the most sensitive blocks to FP8. This is
what ModelOpt's `default_disabled_quantizers` mechanism does, expressed per
decoder layer instead of per projection.

Configuration (metadata only; saved INT4 weights are untouched):

```python
activation={
    "version": 4, "mode": "w4a_nvfp4", "recipe": "least_squares_grid",
    "attention": {"mode": "w4afp8"},
    "mlp": {"mode": "w4afp8", "layers": [11, 15]},
}
```

`activation.mlp.layers` promotes the named decoder layers' MLP boundaries
(`attention_residual`, `product`, and the post-attention RMSNorm) to FP8. Every
other layer keeps the stream default, and attention is unaffected.

### Which layers matter

Each decoder layer was promoted to FP8 in isolation and scored by KL divergence
from the dense BF16 reference, on the benchmark-disjoint calibration artifact
(64 wikitext articles, 8 batches of 8x512). Reference `baseline_kl` for the
unmodified A4 checkpoint is 0.154863.

| layer | KL gain vs A4 | layer | KL gain vs A4 |
| --- | --- | --- | --- |
| 0 | +0.003071 | 8 | +0.001695 |
| 1 | +0.000701 | 9 | +0.002509 |
| 2 | +0.001836 | 10 | +0.003328 |
| 3 | +0.002339 | 11 | +0.003584 |
| 4 | +0.002690 | 12 | +0.003191 |
| 5 | +0.001524 | 13 | +0.003065 |
| 6 | +0.000510 | 14 | +0.002529 |
| 7 | +0.001244 | 15 | +0.003528 |

Ranked best-first: `[11, 15, 10, 12, 0, 13, 4, 14, 9, 3, 2, 8, 5, 7, 1, 6]`.

Two structure facts matter for how far this lever can go:

- Sensitivity is concentrated at the **first layer and the last third**. Layers
  9-15 plus layer 0 hold 74% of the recoverable fidelity; the middle layers
  (1-8) are individually the cheapest to leave on 4-bit activations.
- The distribution is otherwise **flat**: the single best layer recovers only
  9.6% of the total, and no subset of five reaches half. Promoting one layer is
  not a shortcut; the cost is spread across the depth.

Cumulative recovery by rank: top-1 9.6%, top-2 19.0%, top-3 28.0%, top-4 36.5%,
top-6 52.9%, top-8 66.9%, top-12 89.3%.

### How many layers are needed

**Superseded - do not use.** This section originally interpolated between A4
(-2.32 pp) and a "+0.74 pp" endpoint to get a 3.06 pp span, then concluded that
two promoted layers would close the gap. Both steps turned out to be wrong: the
+0.74 pp endpoint came from a different, weaker checkpoint (see "What this
fixes about the earlier estimate"), and the measured mlp8-2L result was +0.08 pp,
not the +0.60 pp predicted. The measured outcomes below replace this entirely.

### Measured KL for candidate sets (non-additive)

The single-layer table above is a ranking probe. Because a decoder stack is
sequential, promoting several layers together is not the sum of the individual
gains. Each candidate set was therefore re-measured end to end against the
dense BF16 reference on the same disjoint calibration text:

`mlp8-NL` names a candidate set that promotes `N` decoder layers to an FP8 MLP
operand while every other layer keeps the NVFP4 MLP default.

| set | layers | KL | gain | share of full MLP gain | est. GSM8K |
| --- | --- | --- | --- | --- | --- |
| pure A4 | -- | 0.154863 | 0.000000 | 0.0% | -2.32 pp |
| mlp8-1L | 11 | 0.151279 | 0.003584 | 9.6% | -2.02 pp |
| mlp8-2L | 11, 15 | 0.147497 | 0.007366 | 19.7% | -1.71 pp |
| mlp8-3L | 11, 15, 10 | 0.144620 | 0.010242 | 27.4% | -1.48 pp |
| mlp8-4L | 11, 15, 10, 12 | 0.140688 | 0.014175 | 38.0% | -1.16 pp |
| mlp8-6L | 11, 15, 10, 12, 0, 13 | 0.135412 | 0.019451 | 52.1% | -0.72 pp |

Promotion is **super-additive**: mlp8-2L recovers 19.7% where the single-layer probe
predicted 19.0%, and mlp8-6L recovers 52.1% against a predicted 52.9%. Combined with
the first-layer / last-third concentration, this means the cheapest way to buy
fidelity is a small number of well-chosen layers, not a broad sweep.

The estimate column maps the KL share onto the two measured GSM8K endpoints
(pure A4 = -2.32 pp, FP8 MLP everywhere = +0.74 pp, a 3.06 pp span). It is an
interpolation, not a measurement; the mlp8-2L row is the one under test.

### Per-layer MLP selection: measured GSM8K outcome

The mlp8-2L view (layers 11 and 15 promoted to FP8 MLP) was evaluated on the full
1,209 GSM8K Platinum rows with the same paired gate:

| lane | score | delta vs W4A16 | verdict |
| --- | --- | --- | --- |
| W4A16 (same weights) | 0.41522 (502/1209) | -- | reference |
| A4 + `least_squares_grid` | 0.39206 (474/1209) | -2.32 pp | inconclusive |
| A4 + FP8 MLP on layers 11, 15 | 0.39289 (475/1209) | -2.23 pp | inconclusive |

mlp8-2L recovered only **+0.08 pp** (95% CI [-4.54, +0.07] pp, McNemar p=0.068),
against the +0.60 pp the KL interpolation predicted.

### The KL-to-accuracy map is not linear

Two corrections invalidate the interpolation that produced the +0.60 pp figure:

- The "+0.74 pp" endpoint came from a *different* checkpoint. That FP8 lane was
  a fresh version-3 stream calibrated on 512 records whose own W4A16 reference
  scored 0.3573, not the 0.41522 of the full64 checkpoint used here. It is not
  a valid second endpoint, so the 3.06 pp span was wrong.
- Even with a valid span, KL is a smooth distributional measure while GSM8K is
  a thresholded exact-answer metric. A 19.7% reduction in KL need not flip
  answers, and here it barely did.

This is the important result of the per-layer experiment: **KL divergence from
the dense reference is not a usable proxy for GSM8K accuracy** at this scale.
The single-layer ranking remains a valid *sensitivity* map (it correctly
identifies layer 0 and the last third as the costly blocks), but it cannot be
converted into a predicted score.

The ceiling question is therefore open and is being measured directly: FP8 MLP
on all 16 layers is the full A8 stream on this checkpoint, and its GSM8K score
is the maximum any per-layer selection could reach. If that ceiling does not
clear -2 pp, no subset can, and the lever is closed.

### The A8 ceiling on this checkpoint

FP8 MLP on all 16 layers is the full A8 stream on the same INT4 weights. Full
1,209-row paired gate:

| lane | score | delta vs W4A16 | verdict |
| --- | --- | --- | --- |
| W4A16 (same weights) | 0.41522 (502/1209) | -- | reference |
| A4 + `least_squares_grid` | 0.39206 (474/1209) | -2.32 pp | inconclusive |
| A4 + FP8 MLP on 2 layers | 0.39289 (475/1209) | -2.23 pp | inconclusive |
| A8 (FP8 MLP on all 16) | 0.41853 (506/1209) | **+0.33 pp** | `within_budget` |

Paired statistics for the ceiling: 95% CI [-1.56, +2.22] pp, McNemar p=0.797,
66 base-only vs 70 float-only correct. The A8 stream is statistically
indistinguishable from W4A16, and is the first configuration in this campaign
to pass the gate outright.

### What this fixes about the earlier estimate

The span between the two endpoints on **this** checkpoint is 2.65 pp
(-2.32 -> +0.33), not the 3.06 pp used earlier. The earlier figure borrowed a
"+0.74 pp" endpoint from a fresh version-3 stream whose own W4A16 reference
scored 0.3573; that checkpoint is weaker and not comparable.

### The KL-to-accuracy map is strongly convex

| set | KL share of total | measured GSM8K share of total |
| --- | --- | --- |
| mlp8-2L (layers 11, 15) | 19.7% | 3.4% (+0.09 pp of 2.65 pp) |
| k16 (all layers) | 100% | 100% (+2.65 pp) |

Fitting a power law to the mlp8-2L point gives `GSM8K_share ~= KL_share^2.1`. In
words: **the first fifth of the distributional error costs almost nothing in
accuracy, and the last part costs almost everything.** A layer ranking built on
KL therefore identifies where the error is, but badly overstates how much
accuracy any small subset of layers will buy back.

This is the concrete reason the earlier +0.60 pp prediction for mlp8-2L failed. It
is not a bug in the per-layer mechanism: the mechanism works, the two-layer set
simply does not move the metric.

### Per-layer MLP selection: complete results

Every configuration was evaluated on all 1,209 GSM8K Platinum rows with the
same paired gate against the same-weight W4A16 reference (0.41522, 502/1209).

| lane | layers on A8 MLP | score | n/1209 | delta vs W4A16 | flips | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| W4A16 (same weights) | -- | 0.41522 | 502 | -- | -- | reference |
| A4, all NVFP4 MLP | 0 / 16 | 0.39206 | 474 | -2.32 pp | 222 | inconclusive |
| A4 + mlp8-2L (11, 15) | 2 / 16 | 0.39289 | 475 | -2.23 pp | 203 | inconclusive |
| A4 + mlp8-4L (10, 11, 12, 15) | 4 / 16 | 0.40033 | 484 | **-1.49 pp** | 212 | inconclusive |
| A4 + mlp8-6L (0, 10, 11, 12, 13, 15) | 6 / 16 | 0.39371 | 476 | -2.15 pp | 216 | inconclusive |
| A8, all FP8 MLP | 16 / 16 | 0.41853 | 506 | **+0.33 pp** | 136 | `within_budget` |

Two things stand out.

**The point estimate can be closed.** mlp8-4L reaches -1.49 pp, inside the 2 pp
budget, recovering 0.83 pp of the 2.32 pp gap with 12 of 16 layers still on
true 4-bit NVFP4 activations. That is a real result and it is reproducible: the
mlp8-4L view is a metadata-only symlink over the same packed INT4 weights.

**The set is not monotone.** mlp8-6L is 0.66 pp *worse* than mlp8-4L despite promoting two
more layers. The difference is 8 samples against a paired CI half-width of
~2.4 pp, so it is noise, not a signal. It does mean the individual layer
ranking cannot be trusted to order adjacent candidates.

### Why no mixed configuration clears the gate

The gate is stricter than "point estimate within 2 pp". It requires the 95%
paired CI lower bound to sit at or above -2 pp:

```python
elif 100 * (delta - half_width) >= -allowed_drop_pp:
    statistical_verdict = "within_budget"
```

The half-width is `1.96 * sqrt(f / n)` in pp for `f` discordant pairs out of
`n = 1209`. At the flip rates these configurations actually produce, that
half-width is larger than the entire budget:

| lane | flips | CI half-width | delta needed to pass |
| --- | --- | --- | --- |
| A4 | 222 | 2.42 pp | +0.42 pp |
| mlp8-2L | 203 | 2.31 pp | +0.31 pp |
| mlp8-4L | 212 | 2.36 pp | +0.36 pp |
| mlp8-6L | 216 | 2.38 pp | +0.38 pp |
| A8 (all 16) | 136 | 1.89 pp | -0.11 pp |

A mixed configuration therefore has to *match or beat* W4A16 to be declared
`within_budget`, because its own confidence interval is wider than the margin
being tested. mlp8-4L's -1.49 pp point estimate is genuinely inside the 2 pp budget;
it fails the verdict because the interval is ±2.36 pp.

The A8 stream passes only because promoting every MLP layer roughly halves the
discordant-pair count (136 vs 212), which both improves the point estimate and
narrows the interval.

### Conclusion

- Per-layer MLP selection **does not close the gap at small k**. mlp8-2L, mlp8-4L and mlp8-6L
  are each statistically indistinguishable from plain A4 and from each other;
  the direct paired test in "Direct paired comparisons" shows every CI spanning
  zero.
- The **ceiling is real**: FP8 MLP on all 16 layers is +2.65 pp over A4 and the
  only mixed configuration whose improvement is statistically detectable. It
  passes the acceptance gate outright at +0.33 pp versus W4A16.
- The **best point estimate is mlp8-4L at -1.49 pp**, inside the 2 pp budget with 12
  of 16 layers still on true 4-bit NVFP4 activations. It is reported as the best
  estimate, not as a confirmed win, because its own 95% CI spans 4.8 pp.
- The honest summary: **W4A4 is measured at -2.32 pp and best-estimated at
  -1.49 pp; it is not shown to reach -0.32 pp.** The 1,209-row paired harness
  has a CI half-width of ~2.4 pp, which is wider than the 2 pp budget it is
  asked to police, so the last fraction of a point is below its resolution.

### Direct paired comparisons between the mixed configurations

The table above compares each lane to W4A16. The more useful test is each
mixed configuration against plain W4A4, since that is the change being claimed.
Pairing the per-row correctness of the same 1,209 rows:

| comparison | delta | 95% CI | flips | significant? |
| --- | --- | --- | --- | --- |
| mlp8-2L vs A4 | +0.08 pp | [-2.22, +2.38] | 201 | no |
| mlp8-4L vs A4 | +0.83 pp | [-1.57, +3.22] | 218 | no |
| mlp8-6L vs A4 | +0.17 pp | [-2.44, +2.77] | 258 | no |
| mlp8-6L vs mlp8-4L | -0.66 pp | [-3.18, +1.86] | 242 | no |
| **A8 vs A4** | **+2.65 pp** | **[+0.22, +5.07]** | 224 | **yes** |

Only the full A8 stream is statistically distinguishable from plain W4A4. Every
partial promotion - two, four, or six layers - lands within the noise of the
baseline it was meant to improve, and within the noise of each other.

This is the honest answer to the per-layer question:

- **mlp8-4L's -1.49 pp point estimate is inside the 2 pp budget**, and it is the best
  measured mixed configuration. It keeps 12 of 16 layers on true 4-bit NVFP4
  activations.
- But that estimate is **not statistically separable from plain A4's -2.32 pp**.
  Reporting "per-layer selection closes the gap" would be overclaiming from a
  10-sample difference whose own confidence interval spans 4.8 pp.
- The lever is real only at the limit: promoting essentially the whole MLP to
  FP8 is what produces a detectable gain, and at that point the model is W4A8
  in substance, not W4A4.

### What the 1,209-row harness can and cannot resolve

The paired CI half-width is `1.96 * sqrt(f / n)` for `f` discordant pairs. Every
configuration measured here produced between 136 and 258 flips, giving a
half-width of 1.9 to 2.6 pp. The acceptance budget is 2 pp. The measurement is
therefore **coarser than the threshold it is being asked to police**, which is
why four different configurations all land in the same `inconclusive` bucket
despite point estimates spanning -2.32 to -1.49 pp.

Resolving a 0.32 pp difference at this effect size would need on the order of
40,000 paired rows, not 1,209, or a lower-variance metric than exact-answer
accuracy.

## Full-model validation on 2026-10-07

Runtime commit `7b6814c2` was evaluated on all 1,209 GSM8K Platinum test rows
for both saved Llama 3.2 1B checkpoints and their same-weight W4A16 controls.
All 16 decoder layers and 112 projections were selected. Each pair shares
the exact INT32-packed weight file and tokenizer artifacts. Separate views
remove the legacy `activation.version` field and freeze the historical prompt
date. The weight files remain unchanged. This run covers saved-checkpoint
inference; fresh GPTQ calibration/weight quantization was outside its scope.

| Policy | W4A16 correct / 1,209 | Activation correct / 1,209 | Change (pp) | Paired 95% CI (pp) | Correctness flips |
| --- | ---: | ---: | ---: | --- | ---: |
| Balanced A4: FP8 attention, NVFP4 MLP, `least_squares_grid` | 502 (41.52%) | 474 (39.21%) | -2.316 | [-4.729, +0.097] | 222 |
| Per-token FP8, `w4afp8` | 432 (35.73%) | 438 (36.23%) | +0.496 | [-1.618, +2.611] | 170 |

The two checkpoints contain different GPTQ weights, so their absolute scores
are not a direct A4-versus-A8 comparison. Both W4A16 controls reproduce all
1,209 historical responses exactly. A4 also reproduces every historical
response and the accepted 28-answer loss. Its statistical budget test remains
`inconclusive`; reproduction of the accepted point score does not establish
noninferiority at the confidence-interval boundary. FP8 is `within_budget`
against its same-weight control at the 2 pp budget. It scores three answers
below its older 441-correct result (-0.248 pp); 675 extracted answers changed,
with 211 correctness flips relative to that older FP8 run.

Settings match the frozen baselines: eight-shot chat prompts, BF16, seed 42,
greedy generation, a 256-token cap, and batch sizes 8 (A4) and 32 (FP8).
The cached dataset revision is `e762492455a1cf7967de89f05b6bef72fc713b66`.
The guarded sequential runs observed at least 102.01 GiB available memory,
zero swap use, and no new cgroup OOM or memory-limit events.

The boundary audit now resolves effective attention/MLP policies, including
inherited defaults, and looks up norms inside the wrapped model. It applies
the FP4 numerical oracle only to FP4 operands. Both full-model audits pass
381 handoff checks across prefill and cached decoding. A4 additionally passes
288 independent decode checks (maximum absolute error `4.77e-7`) and 144
FP4 GEMM checks (exact output agreement with the FP32 Torch oracle after BF16
rounding; FP32-versus-FP64 arithmetic drift at most `4.60e-6`). The FP8 audit
checks transport; its numerical kernel coverage remains in the kernel suite.
All 75 audit unit/integration tests pass.

The [machine-readable evidence](gptq-w4a-gsm8k-validation.json) records scores,
policies, package versions, dataset and weight hashes, memory observations,
and hashes of every result/audit artifact. Full per-row outputs and the frozen
controls are retained under
`/root/models/w4a-quality/pr3228-7b6814c2-gsm8k-platinum/` on the GB10 host.
