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
backward and mixed-logit-dtype parity fixes. Those fixes have not yet been
validated in a new real-model training run.

Remaining gaps include author GPTQ initializer parity, full staged trajectory
parity, exact paper calibration/batching/precision settings, W4 ambiguity in the
author Q/K constructor, public lifecycle integration, other scalar-format adapters,
and complete portable model export. See `research/gsq-paper-reproduction.md`.

Related changes: calibrated FP8 rejects tensorwise scales until native activation
replay matches deployment. MXFP4 has optional fixed-scale GSQ packing and CPU
payload tests; real-model MXFP4 validation and complete public save/reload remain
pending. Its CPU extension now selects the C++ standard required by PyTorch, and
payload replacement invalidates cached VNNI packing.
