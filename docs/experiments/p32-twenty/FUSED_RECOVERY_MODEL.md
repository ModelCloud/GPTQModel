# Opt-in fused correction model integration

This phase is implemented but GPU validation and fresh profiling are pending.
It does not fuse the native INT4 GEMM. The existing measured expansion/add
Triton kernel is shared between the runtime harness and RecoveredLinear.

Enable only in the experiment driver with `--recovered-fused-expansion`.
Eligibility requires runtime sm80, FP16 contiguous factors, rank 8/12/16,
no sparse correction, same-device FP16 inputs, and disabled gradients.
All other cases preserve the separate implementation. Production defaults
are unchanged. Kernel launch failures surface normally; they do not silently
promote an unvalidated fallback result.

The native base and input projection write separate per-call tensors. The
expansion/add kernel reads both on the current stream after those kernel
boundaries, writes a fresh output tensor, and preserves FP16 correction rounding
before FP32 addition and final FP16 output. No persistent scratch, locks,
cross-CTA barriers, or buffer aliasing are introduced. Reshaping noncontiguous
inputs can allocate a copy; that cost belongs to full-operator timing.

The nine-row runtime harness now includes the actual integrated module alongside
separate and directly invoked fused variants, with eager and Graph replay.
Dependent C4/full-ARC jobs use the unchanged alpha1 rank8 export and retain
window for all other projections. Results are queued, not yet accepted evidence.
Fresh executed SASS/profile comparison is required after extraction/integration;
prior profile results explain the hypothesis but do not certify this revision.

## First integrated runtime validation

The actual RecoveredLinear fused route completed all nine M values, passing
both canonical/window local gates. All nine CUDA Graph outputs equaled their
eager outputs. The full harness completed 81 cases across its nine variants.
[Raw runtime evidence](results/rank8-targeted/fused-integrated-runtime.json).
This validates the bounded real-activation layer path; it does not yet validate
full-model quality, all fallback transitions, or speed under model execution.
Fresh Nsight captures now include the integrated route at M=1/16/2048.

The first fused model C4 slice completed: PPL 26.97508702, versus separate alpha1
26.97691913 and window 26.99152524. These small differences do not establish a
quality improvement. The report records FP16 factors, rank8, no sparse values,
and fused eligibility; full ARC and propagated-logit comparisons remain pending.
[Raw report](results/rank8-targeted/fused-model-c4.json).

Fresh Nsight profiles completed for M=1/16/2048. Direct and integrated epilogues
execute respectively 26,752 / 27,136 / 3,473,408 warp instructions at those shapes,
with 26 registers and zero local spilling requests in each. The M16 executed
opcode histograms match exactly and sum to the Nsight instruction total.
[Resource evidence](results/rank8-targeted/fused-profile-resources.json),
[opcode evidence](results/rank8-targeted/fused-M16-opcodes.json).
This confirms the integration preserved the measured epilogue's instruction work;
it does not prove all full-operator overhead is unchanged. No new masks, address
operations or conversions were introduced into that epilogue. Native GEMM/input
projection remain separate optimization opportunities. Post-profile normal
correctness/Graph timing is queued; NCU replay durations are not speed evidence.

Two saved-logit CPU comparisons are running: fused versus separate alpha1, and
fused versus window. `compare_baselines.py` now accepts an explicit output path
and refuses to overwrite an existing report, allowing both reference comparisons
without clobbering evidence. Early per-document results include nonzero
fused/separate divergence; final aggregate KL/top-k results remain pending.
These comparisons reuse completed C4 model runs without touching the teacher.

## Full ARC and propagation checks

Fused rank8 scored 390 raw / 431 normalized out of 1,172 ARC examples.
Against window, paired wins/losses are 9/9 raw (p=1) and 9/6 normalized
(p=0.60724); against separate alpha1 they are 1/0 and 2/0 (p=1 and 0.5).
No task improvement is established. Prompts, targets and indices matched.
[ARC report](results/rank8-targeted/fused-model-arc.json),
[paired comparisons](results/rank8-targeted/fused-model-arc-paired.json).

C4 fused-versus-separate logits KL is 1.13269545e-5, with top1/5/10 agreement
0.9978027344 / 0.9991210938 / 0.9989013672. The paths are not bit-identical.
[Logits evidence](results/rank8-targeted/fused-logits-vs-separate.json).
Post-profile runtime checks again pass both local gates at all nine row counts.
[Post-profile report](results/rank8-targeted/fused-post-profile-runtime.json).
Full-model 2x speedup remains unproven; only one down projection is replaced.
