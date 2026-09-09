# Staged scalar GSQ: implementation and validation status

This branch contains an experimental Llama staged trainer, not a completed paper
reproduction or a promoted default. It is not yet wired into public GPTQ/AWQ
quantization configuration. The existing independent-projection GSQ fitter remains
separate and optional.

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
and complete portable model export. See `research/gsq-paper-reproduction.md`.

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
