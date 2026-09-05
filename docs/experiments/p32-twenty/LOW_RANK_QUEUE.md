# Experiments 31–40: small-rank recovery paired with window P32

All ten experiments extend the same PR and read-only historical F6 seed-7
campaign. The deployed window model is the P32 comparison arm; canonical FP32
remains the numerical teacher and original BF16 remains the secondary model
reference. Every unreplaced projection keeps window/P32 execution. A replaced
projection runs only W4A16 plus its correction, not a second full window path.

## Fixed focused run: 31/32/33/35

Module: `model.layers.0.mlp.down_proj`, K=8192, N=2048. Ranks
0/2/4/6/8/12/16; nine M values 1/2/4/8/16/32/128/512/2048.

The native base is copied exactly from
`/root/p32-native-joint/down_proj/rank16.pt`: the already passing rank-16 export
selected at joint iteration 2. No native quantizer is invoked in this sweep.
Each export's native tensor payload/metadata fingerprint is checked against the
source. The same 8192-token historical Fisher capture is used:
`/root/p32-recovery-calibration/activations16x512`. Source export, capture and
read teacher shards are hashed. Separate C4 captures are evaluation only.

- **31 — fixed-base direct rank fit:** FP64 reduced-rank regression of actual
  deployed output residual `canonical(X_FP16) - W4A16(X_FP16)`. An activation
  SVD with rcond=1e-5 and projected-residual SVD solve every rank of this same
  problem. Reusing this decomposition is not truncation of the weight-space
  rank-16 correction. Report singular values, optimal projected residual energy,
  calibration/held-out MAE/max/relative L2/cosine, exact serialized BPW,
  all nine latencies, and export-reload output equality.
- **32 — rank-16 truncation control:** QR of A16 and B16-transpose followed by
  SVD of their small core computes the weight-space SVD of A16 B16. Keep its
  first r components and compare with independent output-aware fitting.
- **33 — tail-aware fit:** starting from direct L2 factors, fix calibration
  element weights at 10 for residual magnitudes above their initial 99.9th
  percentile and 1 otherwise. Run 40 Adam steps on actual FP16-boundary output
  error, selecting the minimum weighted calibration objective including step 0.
  A separate smooth-max objective sweep remains queued: mean(E²) plus
  lambda*tau*logsumexp(abs(E)/tau), with parameters declared before evaluation.
- **35 — factor precision:** FP32/FP32 and FP16/FP16 are implemented in the
  focused sweep. Actual GEMM inputs/intermediates use factor dtype; final
  correction and native outputs are added in FP32, then cast to FP16. BF16,
  mixed FP16/FP32, and scaled INT8 factors remain to implement and measure.

The complete standalone serialized operator, including codes, scales, both
factors, dimensions, dtype/fit metadata and serialization overhead, determines
BPW. Do not substitute the analytic factor-only estimate. Runtime workspace,
resident model bytes, and full-model BPW remain separate accounting obligations.

## Remaining queued experiments

| # | Required sweep and deliverables |
|---|---|
|34|Ranks 2/4/6/8/12 with 0/32/64/128/256/512 sparse exceptions. Select positions using activation-weighted residual after AB, refit values, include indices/values in serialized BPW, implement actual sparse runtime and measure its full latency.|
|36|Calibration sizes 2048/4096/8192/16384/32768, at least three independent historical ordinary-text subsets each. Capture provenance, held-out document pass/error distributions, singular stability, worst channels and selected rank. No benchmark fitting.|
|37|Fit every P32 down projection in layers 0–15; use ranks 0/2/4/6/8/12/16 and retain window when none passes. Smallest passing local rank is provisional until broader/model checks.|
|38|Progressive replacements: layer 0, layers 0–3, 0–7, 0–11, and every passing down projection. Measure window-relative logits KL/top1/5/10, C4/FineWeb PPL, full ARC/MMLU or historical target evaluations, prefill/decode, resident and serialized model BPW.|
|39|Rank-specific separate GEMMs, shared input base/XA pass, fused base+correction+addition, and CUDA Graphs. Report logical storage rank separately from physical padded compute rank, especially rank6→8 and rank12→16. Profile actual instructions/resources.|
|40|Independently alternate native quantization and real-output residual refit at each rank for 1/2/4/8 iterations. Select with calibration only; compare matched BPW and latency against the fixed-base family.|

Every candidate must pass export reload, all local MAE <=0.003 and max <=0.046875
cases, broader held-out activations, then progressive model quality within a
predeclared uncertainty policy; latency and complete BPW determine acceptance
among eligible candidates. No default changes or automatic exceptions are made.
Human-review escalation for measured gains >25%/>100%/>500% remains active.

## Live automatic dispatch

`run_gpu_queue.py` executes explicit argv-array jobs on idle GPUs identified by
UUID. It respects existing processes, including the ongoing canonical teacher,
and records PIDs, start/finish times, result checks and failures. Candidate model
jobs wait for a complete producing report with nine export-reload equalities.
Failures are recorded, never automatically retried or overwritten.

The active manifest is `/root/p32-low-rank/model-queue.json`, with state in
`model-queue-state.json` and events in `dispatcher.log`. It schedules four
same-device production-window model baselines plus 28 focused L2/tail model
comparisons (seven ranks, two factor dtypes). Failing local candidates are
research controls, not promotion candidates. Follow-up full tasks and experiments
34/36–40 require their implementations/data and are tracked here rather than
misrepresented as executable jobs already running.
