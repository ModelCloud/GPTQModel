# Paper-faithful scalar GSQ implementation work

User requested the full training procedure, not only the existing independent
projection fitter. This is active implementation scope; no reproduction claim yet.

Reference: [GSQ v2](https://arxiv.org/html/2604.18556v2), especially Sections 3–4
and Appendices B, C, G. Author implementation freshly fetched at
`03fc16484c369e3127225615d5e03e8d3a6043e3` from
https://github.com/IST-DASLab/GSQ. Local reference checkout:
`/tmp/gsq-author-reproduction` (reference only, not a runtime dependency).

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
