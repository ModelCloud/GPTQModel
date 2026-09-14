> AdamW W4 retest complete: 128 calibration documents, seed 7, group128, ten attention/MLP epochs, 2,000 Q/K updates. Platinum: 403/1209 (33.33%, one invalid) versus matched GPTQ 460/1209 (38.05%, zero invalid). Paired delta -4.71 points, CI95 [-7.36, -2.15]. The reused dense reference remains 593/1209 (49.05%). Exact prompts/targets/token IDs match. All 32 held-out reloads are exact; all five held-out logit metrics regress clearly. The optimizer/epoch change did not recover accuracy.

> W4 retest: both full models completed with exact public reload on all 32 held-out documents. GSQ regresses KL (0.278840 → 0.384085), MSE (0.892304 → 1.145021), and Top-1/5/10 agreement (77.509/74.965/75.104% → 73.504/70.950/71.313%). All five paired 95% bootstrap intervals are unfavorable and exclude zero. Full Platinum completed: dense 593/1209 (49.05%), W4 GPTQ 460/1209 (38.05%), W4 GSQ 417/1209 (34.49%). Paired GSQ delta -3.56 points, 95% interval [-6.37, -0.74]. Exact prompts/targets/token IDs match. This is a clear task regression. Both arms use 128 documents, five epochs, seed 7 and matched signed initialization.

# Staged scalar GSQ: implementation and validation status

The PR 212 math, paper-protocol, and control-matrix audit is recorded in
[`gsq-pr212-audit.md`](gsq-pr212-audit.md).

This branch contains an experimental Llama staged trainer, not a completed paper
reproduction or a promoted default. Public GPTQ now has an experimental
`gsq_training` path for uniform, materialized eager Llama models with Torch
packing. Public AWQ has an initial W4 GEMM staged path; real-model AWQ validation
and additional formats remain unfinished. The existing
independent-projection `gsq` fitter remains separate and optional.

AWQ uses `AWQConfig(gsq_training=GSQTrainingConfig(enabled=True, initializer='awq', ...))`.
It retains AWQ scaling/clipping and affine zero points, captures the teacher after
scaling and before clipping, then fits Q/K, attention, and MLP objectives using
the AWQ initializer. It does not run the GPTQ prior. The initial path requires
uniform W4 GEMM, group size 32/64/128, all seven Llama projections, eager attention,
and no fallback, tensor-parallel padding, or AdjacentExact combination. Learned
weights/scales/zeros replace packing metadata together; histories are retained
in configuration metadata. Tiny CPU FP16 public quantize/save/reload reproduces
logits exactly. This is lifecycle correctness evidence, not real-model quality
evidence or full author-procedure parity. AWQ's MLP initializer is retained from
before attention fitting. GSQ remains disabled by default.

The staged opt-in uses `GPTQConfig(gsq_training=GSQTrainingConfig(enabled=True, ...))`.
The training configuration also accepts `optimizer='adamw'`; the default remains
`'lion'`. This selection applies to Q, K, attention, and MLP stages. AdamW uses
the configured learning rates, betas and assignment weight decay, with zero
scale weight decay, epsilon `1e-8`, and non-fused/non-foreach updates. Selecting
AdamW is an explicit experiment, not the paper's Lion procedure. The requested
W4 AdamW retest uses 128 calibration documents, seed 7, group size 128, ten
attention/MLP epochs and 2,000 Q/K updates; completed results are summarized above. Changing optimizer
and epoch count together does not isolate their individual effects.
It requires `format=FORMAT.GPTQ_V2`, `sym=True`, `desc_act=False`,
`act_group_aware=False`, `offload_to_disk=False`, and int32 packed storage.
The model must already be materialized on one CPU/CUDA device with eager attention.
Use `backend=BACKEND.TORCH` in `quantize()`. Calibration uses the normal public
preparation path; choose `calibration_sort=None` to preserve supplied row order.
`GSQTrainingConfig` owns staged initializer, damping and optimization settings;
the requested outer configuration is retained as provenance. Unsupported extra
processors/transforms reject explicitly rather than being silently skipped.

Example configuration for the tested recipe (not an accuracy recommendation):

```python
from gptqmodel.quantization import FORMAT, GPTQConfig, GSQTrainingConfig

config = GPTQConfig(
    bits=4, group_size=128, format=FORMAT.GPTQ_V2,
    sym=True, desc_act=False, act_group_aware=False, offload_to_disk=False,
    gsq_training=GSQTrainingConfig(
        enabled=True, initializer="gptq_signed", seed=7,
        epochs=5, batch_size=64, microbatch_size=16,
    ),
)
```

Omitting `gsq_training`, or setting `enabled=False`, retains ordinary GPTQ dispatch.
That disabled public path is not the dedicated signed-prior experiment control.
Enabling both projection `gsq` and staged `gsq_training` rejects. Tiny CPU public
quantize/save/reload is verified with exact logits and config/provenance retention;
the real 128-row W4 public-lifecycle run completed: all 482 checkpoint tensors and
all 32 held-out logit tensors exactly match the dedicated Lion run; all 32
documents also reload exactly. Evidence: `artifacts/gsq-staged/full-model-w4-nm128-signed-public-v1/checkpoint-parity.json`. The completed quality
results below used the dedicated trainer and remain negative evidence.

The current user-selected full-model experiment uses **128 calibration documents
and five attention/MLP epochs**, seed 7, W2/group128, signed GPTQ initialization,
batch 64 and microbatch 16. Q/K retain 2,000 updates per projection. Both complete
16-layer models have finished quantization and pass exact public save/reload logits
on all 32 held-out documents. The 128 documents contain 47,005 tokens and pass
question-disjointness checks against D300 and all 1,209 Platinum questions.

| Held-out final-logit metric | GPTQ prior only | Staged GSQ | Paired classification |
|---|---:|---:|---|
| KL to dense teacher | 4.464245 | 5.923538 | Clear negative |
| MSE | 7.857785 | 10.020153 | Clear negative |
| Top-1 agreement | 22.8208% | 16.3185% | Clear negative |
| Top-5 agreement | 23.3328% | 15.5081% | Clear negative |
| Top-10 agreement | 24.2061% | 15.8929% | Clear negative |

All five 95% paired document-bootstrap intervals exclude zero in the unfavorable
direction (10,000 draws, seed 7, token-weighted means). This is a failed recovery
experiment. The baseline is the staged-path signed initializer, not the package's
default true-sequential GPTQ recipe. FP32 training and token-weighted variable-length
microbatches remain differences from a complete author-procedure reproduction.

The five-epoch full run took 879.96 seconds. Synchronized stage wall times sum to
573.68 seconds: Q 216.05, K 74.46, attention 59.47, MLP 223.71. The remaining time
includes initialization, capture, export and evaluation; it has not been attributed
by a kernel profiler. These are run timings, not a matched speedup measurement.

Full Platinum evaluation completed with exact three-arm rendered prompts, targets
and input token IDs verified:

| Arm | Correct / 1,209 | Accuracy | Invalid numeric answers |
|---|---:|---:|---:|
| Dense FP16 | 593 | 49.0488% | 0 |
| Signed GPTQ prior only | 0 | 0.0000% | 1,170 |
| Staged GSQ, five epochs | 20 | 1.6543% | 482 |

The paired task gain is +1.6543 percentage points (95% bootstrap interval
+0.9926 to +2.3987 points, 10,000 draws, seed 7). This is a clear positive task
effect relative to a collapsed baseline, alongside clear negative held-out logit
metrics. Absolute task quality remains severely degraded versus dense; this does
not justify promotion or a general recovery claim. The strict audit also verifies
matching evaluator source/settings, all 1,209 ordered examples, and calibration
hash binding for both quantized models. Quantization logs, raw predictions and
evaluation logs are preserved.

Current artifacts are under `artifacts/gsq-staged/`:
`nm128-seed7-v1`, `full-model-w2-nm128-signed-{baseline,staged,comparison}-v1`,
and `gsm8k-platinum-nm128-signed-v1`.

Historical calibration: the earlier 16-document runs below are failed preliminary
experiments. The subsequent 512-document singleton run was superseded and stopped
at user request; its logs and completed baseline remain preserved. Neither is the
current requested 128-document/five-epoch comparison.

The opt-in `GSQTrainingConfig(initializer='gptq_signed')` adds the author's signed,
exponent-2.4 group-scale prior to repository GPTQ and late MLP initialization.
CPU W2/W3/W4 integer assignments match the author on a real 32x256 projection slice
using all 512 NM documents, with maximum weight differences at most 7.2e-7.
The current full models exercise signed initialization on GPU and public reload;
independent author-versus-adapter CUDA prior parity remains unverified.

The staged path implements Q/K prepared quadratic fitting (2,000 updates), V/O
attention-plus-residual reconstruction, and gate/up/down full-block reconstruction.
It uses Lion, jointly trained scales, W2 full-grid or W3/W4 five-shift assignments,
and temperature/logit-multiplier and learning-rate schedules. Quantizer arithmetic
and 20-update Lion trajectories match pinned author code on nine small CUDA
rate/dtype fixtures. This does not establish complete training-procedure parity.

Real Llama 3.2 1B Instruct block-0 W4/group128, seed7, FP32, 16 train and 32 held-out
documents: the dedicated-Q/K, ten-epoch run regressed held-out block MSE by 18.87%.
Its F6 final-logit diagnostic also regressed KL (0.109431 to 0.112593), MSE
(0.455346 to 0.463713), and Top-10 agreement (82.5931% to 82.3842%). Paired document
bootstrap classified those as clear negatives. Top-1/Top-5 changes were
noise-consistent. Preserve all failed runs; do not promote from this evidence.

Artifacts: `artifacts/gsq-staged/llama-block0-w4-seed7-staged-v3` and
`artifacts/gsq-staged/propagation-v3`. The v3 run predates the subsequent explicit
backward and mixed-logit-dtype parity fixes. The subsequent explicit-arithmetic W4 v4 run still regressed local MSE by
19.23%; the arithmetic correction did not resolve that regression.

W2/group128 with the same real model and locked documents produced a different
result. The early-MLP-initializer run reduced local MSE by 43.62%. Against a W2
GPTQ block-0 baseline in the F6 model, final-logit KL fell from 0.665093 to
0.548272 and logit MSE from 2.138095 to 1.851316. Top-1/5/10 agreement rose by
2.152/3.779/3.603 percentage points. All five paired document-bootstrap intervals
favored staged training. This is a small selected-block diagnostic, not a complete
W2 model or general recovery result. Artifacts: `llama-block0-w2-seed7-explicit-v1`
and `propagation-w2-v1` under `artifacts/gsq-staged`.

Correcting the author-required ordering—initialize MLP after attention training—
further reduced local W2 MSE to 0.000332189, versus GPTQ 0.000618326 (46.28%
lower). The matched portable packed-runtime evaluation completed in
`artifacts/gsq-staged/propagation-w2-late-mlp-v2`: KL 0.664968 → 0.506050,
logit MSE 2.137885 → 1.755513, Top-1 67.5986% → 71.9177%. Top-5/10
also improved, and all five paired intervals favored staged training. Seven
projection packing audits found zero assignment mismatches and no zero stored scales. Calibration remains 16
unweighted documents, FP32, not the full paper data/batching/precision setup.
Broader disjoint confirmation remains necessary before promotion.

Remaining gaps include author GPTQ initializer parity, full staged trajectory
parity, exact paper calibration/batching/precision settings, W4 ambiguity in the
author Q/K constructor, public lifecycle integration, other scalar-format adapters,
and broader native model exports. Complete portable TorchLinear model export is
now implemented and checked below. See `research/gsq-paper-reproduction.md`.

Related changes: calibrated FP8 rejects tensorwise scales until native activation
replay matches deployment. MXFP4 has optional fixed-scale GSQ packing and CPU
payload tests; real-model MXFP4 validation and complete public save/reload remain
pending. Its CPU extension now selects the C++ standard required by PyTorch, and
payload replacement invalidates cached VNNI packing.

## Explicit staged block API

`GSQTrainingConfig` is separate from the existing independent-refinement
`GSQConfig`. The block entry point initializes GPTQ, optionally runs the staged
trainer, and can export all seven projections to portable TorchLinear:

```python
from gptqmodel.quantization import GSQTrainingConfig
from gptqmodel.quantization.gsq_training import quantize_llama_gsq_block

packed_block, diagnostics = quantize_llama_gsq_block(
    llama_decoder_layer,
    captured_batches,  # (unpadded hidden states, attention/rotary kwargs)
    bits=2,
    group_size=128,
    gsq=GSQTrainingConfig(enabled=True),
)
```

Defaults are disabled. Omit `gsq` to retain the ordinary GPTQ initializer without
training. `diagnostics['gsq_training']` records the effective configuration;
`stages` contains training histories and learned scales. The caller owns real
calibration capture, device placement, padding-free batch construction, disjoint
evaluation and model checkpoint writing. Packed blocks return on CPU; unpacked
blocks (`pack=False`) retain their input device. This block API is not yet the
full `GPTQModel.quantize()` lifecycle or a complete paper reproduction. Calibration
batching/precision are not implied by optimizer defaults.


A fixed seed alone does not guarantee bitwise staged CUDA training. Repeating the
same W2 configuration produced different learned weights despite identical GPTQ
seeds and schedules. An isolated real-shape scale-gradient check observed
`scatter_add_` repeat differences up to 1.67e-6; deterministic algorithms removed
those differences in 20 repeats. The author implementation uses the same scatter
reduction. This does not invalidate small matched-arithmetic fixtures, but they
do not establish full-size repeatability. The validation runner exposes
`--deterministic` (with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA startup)
and records the actual runtime setting. Two complete W2 runs now match every recorded loss/schedule, learned scale,
final weight and held-out reconstruction metric exactly (v5/v6 artifacts).
Their local MSE is 0.000330969 versus baseline 0.000618326, a 46.47% reduction.
Its final-logit evaluation is complete: packed KL 0.664968 → 0.521725,
MSE 2.137885 → 1.801593, Top-1 67.5986% → 70.8497%; all five paired
intervals favor GSQ. See `artifacts/gsq-staged/propagation-w2-deterministic-v5`. No prior final-logit evidence is silently rebound to different weights.


## Shared capture bridge

`capture_llama_gsq_inputs` captures actual HF Llama decoder calls into InputCache
while replaying the current model prefix. `quantize_llama_gsq_capture` prepares
an autograd-capable private block and trains/exports it even when the caller
uses inference mode. Captured masks and rotary tensors are preserved. Online Hadamard wrappers and
missing rotary state reject explicitly; source-document capture rejects padding.
Callers supplying an existing InputCache must provide unpadded batches. Capture
hooks are removed on success and failure.

The real W2 full-budget capture run (`llama-block0-w2-seed7-shared-capture-v7`)
produces the identical serialized payload SHA256 as deterministic v5, including
all weights/scales and training histories. This binds the same payload to the
completed packed propagation evaluation. This is a verified HF-forward/InputCache
bridge, not completed full-model ModuleLooper dispatch or checkpoint integration.

## Sequential model runtime and uniform export

`quantize_llama_gsq_model` applies the staged block path to every Llama decoder
by default. Each packed block is installed before the next block's actual model
prefix is replayed for capture. Stored FP16 scale rounding therefore participates
in downstream inputs. The input model must be materialized on one device and in
eval mode. A later failure retains completed blocks and a failed run record.

```python
from gptqmodel.looper.gsq_training_model import (
    quantize_llama_gsq_model, save_llama_gsq_model,
)

run = quantize_llama_gsq_model(
    model, documents, bits=2, group_size=128,
    gsq=GSQTrainingConfig(enabled=True),
)
save_llama_gsq_model(
    model, run, output_directory,
    tokenizer=tokenizer, source_model=dense_checkpoint_directory,
)
```

The exporter requires a completed all-layer run and uses the existing public
GPTQ-v2 writer. It explicitly converts floating runtime weights to FP16 and
rebuilds canonical nonpersistent RoPE state, matching a fresh load. Blindly
casting RoPE's FP32 inverse frequencies to FP16 changes the live runtime but is
not serialized, creating a reload discrepancy. The tiny-model public-loader
regression now checks longer-sequence exact FP16/eager logits. CPU construction
of the RoPE frequencies is necessary to match the loader exactly: GPU construction
can differ by one FP32 ULP. Both full-model audits reproduce all 32 reloaded
document logits exactly with CPU-constructed RoPE, and reproduce the old live
logits exactly after rounding that buffer to FP16. This dedicated runtime API
is not yet the shared `GPTQModel.quantize()` dispatcher.

The complete 16-layer W2 experiment has mixed results. Against its matched
no-GSQ initializer, final-logit KL worsened from 9.384694 to 10.028224, while
MSE improved from 10.825406 to 9.930587 and Top-1 agreement rose from 7.7431%
to 12.4706%. The paired 95% interval for KL degradation is [0.424002, 0.869079].
Both models have poor absolute agreement; the favorable selected-block result
does not establish complete-model recovery. The control uses the staged-path
GPTQ initializer with GSQ disabled, not the package-default true-sequential
GPTQ recipe. Calibration is only 16 unweighted documents. GSM8K Platinum
evaluation is being run separately on the reloaded checkpoints and dense model.
