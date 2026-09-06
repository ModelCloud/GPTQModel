# GPUVerify: concurrency verification rather than algebraic folding

## Primary sources and scope

[GPUVerify](https://github.com/mc-imperial/gpuverify) statically checks race
freedom and barrier-divergence freedom in supported CUDA/OpenCL kernels.
[Engineering a Static Verification Tool for GPU Kernels, CAV 2014](https://www.doc.ic.ac.uk/~afd/papers/2014/CAV.pdf)
describes its verification toolchain and engineering experience.
The root source license is
[Microsoft Public License](https://github.com/mc-imperial/gpuverify/blob/49219770aad01231edd0d8e0fa3ed036006cf32a/LICENSE.TXT).

This tool addresses synchronization correctness. It is not an optimizer for
floating-point identities, CSE or Tensor Core utilization.

## QVQ deployment assessment

Treat it as an isolated feasibility experiment for a small supported shared-memory
kernel. The repository metadata observed on 2026-09-06 reports its last push in
2022; that is a maintenance signal, not proof that every component is unusable.

Do not claim that historical CUDA support covers QVQ's modern inline PTX,
TMA, asynchronous barriers or warpgroup operations. Establish frontend and
semantic coverage first. A simplified model must preserve every effect relevant
to the property being proved.

If using an abstracted producer/consumer example, label the result as a proof
about that model. Map it back to actual buffers, participants and completion
rules before relying on it.

For algebraic analysis, begin with [LLVM](llvm-cuda-analysis.md) and
[rewrite verification](rewrite-verification.md). Dynamic sanitizer runs can
complement static checking, but do not prove unexecuted paths or replace it.
No GPUVerify build or QVQ verification run was performed for this survey.
