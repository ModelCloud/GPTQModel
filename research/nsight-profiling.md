# Nsight profiling: timeline, kernel counters and correctness

## Primary documentation

[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
covers application timelines, CUDA activity and NVTX instrumentation.
[Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
provides kernel-level metrics and source correlation.
Its [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
explains replay, cache and clock controls, occupancy, stall analysis and roofline
views. Collecting metrics may replay work and perturb execution. Application
replay requires deterministic matching of kernel activity.
The [CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
documents kernel/range filters and report export.

## Proposed QVQ workflow

1. Establish correctness and an unprofiled timing baseline under the repository's
   existing GPU isolation and accuracy rules. Record the exact loaded binary,
   GPU, driver/toolchain, model, shapes, precision and launch policy.
2. Use Systems to locate host gaps, launches, transfers, stream dependencies and
   synchronization. Label prefill, decode, transforms, P32 product, reduction and
   EoRA work with meaningful ranges.
3. Select representative expensive kernels for Compute. Start with the counters
   needed to test a specific hypothesis; avoid collecting every metric by default.
4. Inspect memory traffic, compute pipelines, eligible warps, registers, spills,
   shared-memory conflicts and source/SASS correlation together.
5. Re-run unprofiled end-to-end timing and matched correctness after the change.
   Retain reports and the exact invocation in the experiment record.

High occupancy is not the optimization objective; sufficient latency hiding and
useful throughput are. A stall sample is a clue, not a causal proof. Distinguish
a tiny underfilled grid from a resource-constrained steady-state kernel.
Separate static SASS size from executed instructions.

For stateful decode or [recirculation](recirculation.md), verify that replay
reconstructs equivalent state and dependencies. Record replay/cache settings;
do not present profiled replay duration as production latency. Measure both
warm and intended cold-cache conditions when relevant.

## Evidence boundary and research

The [P32 ABI record](https://github.com/ModelCloud/QvQ/blob/263ed4baf7be5e9547b4e731c9031bef5f48cf69/docs/kernels/qvq_p32_runtime_abi.md) already distinguishes a correctness/addressing
profile from representative timing. Its historical artifact references and
reported counts are not new measurements from this documentation update.

Use [Roofline](roofline.md) to organize bandwidth-versus-compute hypotheses,
then account for packed decode, control instructions and launch overhead.
Use [SSA/SASS](ssa-sass.md) for compiler evidence and
[CUDA execution](cuda-execution.md) for architecture-specific interpretation.
