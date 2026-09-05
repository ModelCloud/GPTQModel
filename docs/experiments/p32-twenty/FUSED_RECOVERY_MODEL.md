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
