---
name: gptqmodel-cuda-kernels
description: Build, port, optimize, review, benchmark, or debug GPT-QModel CUDA, C++, CUTLASS, or Triton kernels and their JIT wrappers for the primary NVIDIA A100-and-newer target. Use for quantized kernel correctness, extension registration, launch validation, memory/layout/synchronization bugs, architecture gating, or performance work; combine with the Ampere, Hopper, or Blackwell skill for architecture-specific tuning.
---

# GPT-QModel CUDA kernels

For QVQ optimization, first read [qvq-kernel-accuracy](../qvq-kernel-accuracy/SKILL.md) for accuracy-preserving math
and the locked numerical contract; apply it before selecting lower precision or ranking performance candidates.

All kernel code and kernel-library integration must satisfy
[graph-safe-kernels](../graph-safe-kernels/SKILL.md), including native and external
capture ownership. Read that skill before changing the runtime path.

## NVIDIA A100+ target envelope

GPU kernel-performance work defaults to NVIDIA compute capability 8.0 and newer. Read
[references/nvidia-a100-plus.md](references/nvidia-a100-plus.md) before changing CUDA
memory layout, shared-memory staging, warp/CTA synchronization, asynchronous copy,
streams, Tensor Core ownership, or occupancy policy. It is the shared architecture
contract for A100/Ampere, H100/H200/Hopper, and Blackwell.

Pre-Ampere CUDA is compatibility/fallback scope unless the task explicitly targets it.
AMD/ROCm and Apple/Metal optimization are separate explicit scopes; do not dilute a
CUDA design to imitate those execution models.


## QVQ external tuning ABI rule

Any QVQ kernel optimization that changes launch geometry or N-side work must be
available through the versioned external bridge ABI used by ZML and other
embedding runtimes. Expose the complete honored policy, not just a kernel name:
logical M/K/N (and E where applicable), group-N16 boundaries, split-K wave,
kernel variant, N tiles per CTA/warp, warp and thread count, row/CTA grouping,
pipeline stage count, static-N selection, vectorization/layout choices, shared
memory or workspace requirements, and the implementation/build identity used by
the autotune cache. The bridge must be able to enumerate candidates, benchmark
them outside graph capture, select one explicitly, and pass that frozen choice
to the launch or launch-plan entry point.

External tuning is authoritative: a requested value must be honored by a
matching compiled specialization or rejected with an actionable error. Never
silently replace it with a native default, hidden retune, or different backend.
If a geometry is derived or currently fixed by a specialization, expose that
invariant in the ABI and reject non-native values until a matching specialization
is implemented. Legacy/non-ZML entry points may retain QVQ-owned autotuning,
but it must happen during preparation and never during execution or graph
capture/replay. Add a header contract test and a correctness test for the
external path whenever this surface changes.

Start from a numerical reference and select the smallest kernel path that can express the operation. Keep correctness tests separate from performance benchmarks.

Read [references/kernel-workflow.md](references/kernel-workflow.md). For crashes or silent corruption, also read [references/cuda-debugging.md](references/cuda-debugging.md).
Use `$gptqmodel-contiguous-memory` when a kernel or caller silently falls back to a slow eager path because a tensor is non-contiguous.

## Choose the implementation path

1. Keep or create a plain PyTorch reference for correctness and fallback.
2. Prefer Triton for a self-contained tensor kernel whose launch and specialization fit the existing Python JIT style.
3. Prefer CUDA/C++ through `TorchOpsJitExtension` when using CUTLASS, substantial templated C++, custom operators, vendored sources, or features Triton cannot express reliably.
4. Extend the existing JIT system rather than introducing a second AOT build route unless the task explicitly requires packaging changes.

For architecture-specific work, use `$gptqmodel-ampere-kernels`, `$gptqmodel-hopper-kernels`, or `$gptqmodel-blackwell-kernels` in addition to this skill.

## Implement from the boundary inward

1. Specify accepted shapes, strides, dtypes, layouts, quantization metadata, devices, and compute capabilities.
2. Add negative validation at the Python/C++ boundary. Reject unsupported inputs before launch with an actionable message.
3. Implement a minimal correct kernel, including tail handling and accumulation width, before tuning tiles or pipeline stages.
4. Launch under the correct CUDA device guard and framework current stream. Cross-stream producer/consumer ordering must use explicit CUDA events or an equivalent dependency; host launch order is not a dependency. Avoid hidden global-device assumptions, implicit private streams, and fixed device indices.
5. Register native sources in `_EXTENSION_SPECS` in `gptqmodel/extension.py` and use the public `extension.load`, `is_available`, `error`, `op`, or `namespace` helpers.
6. Put reusable wrapper logic in `gptqmodel/utils/` or the relevant quantized-linear module and sources in the matching `gptqmodel_ext/` subtree.
7. Gate specialized code by compute capability and preserve a tested portable or reference fallback.

Respect JIT fingerprints and cache behavior in `gptqmodel/utils/cpp.py`. Use `GPTQMODEL_KERNEL_REBUILD`, a supported per-extension rebuild variable, or `TORCH_CUDA_ARCH_LIST` only for isolated build/debug runs; do not force rebuilds during normal imports.

## Control template and generated-IR growth

Treat CUDA template expansion as a build-performance and maintainability constraint before adding a specialization.
The source line count is not a useful proxy: a small launch switch can generate hundreds of large device entry points,
and `cicc` optimization can become strongly superlinear after force-inlining and loop unrolling inflate the IR.

Before implementing or approving templated CUDA dispatch:

1. Write the specialization-count formula across every dimension: rates or transition widths, dtypes, output dtypes,
   tile/row shapes, vector formats, split modes, feature flags, and architectures. Report both host launch sites and
   unique device kernel specializations.
2. Prove every generated combination is reachable through the public runtime contract. Do not use a generic template
   switch that instantiates rates, layouts, dtypes, or features which validation later rejects. Restrict each format's
   compile-time dispatch to its legal set.
3. Include nested `#pragma unroll`, recursive `__forceinline__` helpers, compile-time array sizes, and duplicated kernel
   bodies in the expansion estimate. Multipliers compose: `rates * dtypes * row shapes * formats * kernel families`,
   while unroll factors multiply each generated body.
4. Preserve specialized kernels only where measurements justify them. Prefer a generic runtime fallback or a small
   low/mid/high specialization bucket for cold combinations instead of materializing a complete Cartesian product.
5. Split large independent kernel families or rate/type buckets into separate `.cu` translation units with a small
   registration/dispatch unit. This lets Ninja schedule multiple `cicc` processes across CPU cores and lowers peak
   memory per compiler process. Merely moving the same all-inclusive template switch into a header used by every
   translation unit duplicates work and is not a valid split.
6. Keep common device logic shared when doing so does not harm the measured hot path. Avoid cloning nearly identical
   normal, split-K, banked, or output-conversion kernels solely for compile-time convenience.
7. Re-audit the matrix whenever adding a rate, dtype, vector format, tile shape, or unroll variant. Record the before
   and after specialization counts in the change notes; do not accept unexplained compile-time or object-size growth.

For example, a dispatch with 15 transition widths, four row shapes, two vector formats, five normal dtype pairs, and
three split-K input dtypes already requests `15 * 4 * 2 * (5 + 3) = 960` scalar device kernels before WMMA, reducers,
architecture targets, or unrolled-body growth. If one format legally supports only seven widths, generating all 15 is
dead compiler work even when runtime validation prevents those kernels from launching.

## Keep CUDA builds host-safe

Use the Ninja binary from the active local environment and an extension-specific build
root. Do not share a build directory between agents or manually kill unrelated
`ninja`, `nvcc`, or `cicc` processes. Before a CUDA compile, set explicit parallelism
limits through the environment. Start from half of the currently available CPU
quota, then lower it when the available-memory budget or observed compiler peak
requires it:

```bash
export PATH=/root/vm314-codex-one/bin:$PATH
BUILD_CORES="$(nproc)"
BUILD_JOBS="$((BUILD_CORES / 2))"
if [ "$BUILD_JOBS" -lt 1 ]; then BUILD_JOBS=1; fi
export MAX_JOBS="$BUILD_JOBS"
export NINJAFLAGS="-j$BUILD_JOBS"
export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
export NVCC_THREADS=2
```

The computed value is a ceiling, not a target. Apply the memory-aware
compilation policy in [graph-safe-kernels](../graph-safe-kernels/SKILL.md),
account for `NVCC_THREADS` and concurrent builds rather than multiplying worker
counts without a budget, and reduce `BUILD_JOBS` before swap pressure or OOM.
Verify `command -v ninja` and `ninja --version` before starting, and use an
extension-specific build root so builds do not contend for objects or locks.

## Validate correctness

### QVQ accuracy contract

Treat QVQ quantization and QVQ inference as separate numerical contracts:

- **Quantization kernels:** require 100% algorithmic accuracy against the trusted
  PyTorch/CPU reference. The selected trellis states, packed words, bank IDs,
  and quantization metadata must match exactly (`torch.equal`/bitwise equality
  wherever the reference is deterministic). A faster result with a changed
  path, code, or packed representation is a correctness failure; do not use an
  error tolerance to waive a quantization mismatch.
- **Inference kernels:** compare the CUDA output with the dequantized/reference
  inference output for the identical packed tensors, inputs, dtype, shape, and
  stream. Require repeated synchronized timing to show a positive gain outside
  measurement noise. A gain no greater than 1% requires mean absolute output
  drift `<= 3e-3`; a gain greater than 1% permits mean absolute output drift
  `<= 4e-3`. Equal, slower, or noise-indistinguishable candidates receive no
  drift allowance. Maximum absolute output drift remains `<= 0.046875` for
  every tested case, with finite outputs. Compute both metrics over all valid
  output elements of that case in sufficient precision; neither signed mean
  nor pooling cases is allowed. Report max-absolute,
  mean-absolute, relative-L2, and the tested dtype, shape, batch/token regime,
  and GPU. Exceeding either inclusive limit fails even if aggregate metrics or
  generated text appear acceptable.
  These are localized kernel-output gates on identical inputs, weights, and
  initial state, including the full reference composition for a fused operator.
  Propagated final-logit differences are diagnostics, not acceptance gates;
  do not apply these thresholds to final logits or use that drift alone to
  reject a locally passing kernel.
- Test both contracts on deterministic seeds, adversarial signs/magnitudes,
  long reductions, smallest legal dimensions, non-divisible M/N/K tails, every
  supported bit/layout branch, repeated calls, and each target architecture.
  Quantization exactness is checked before inference drift; never hide a
  quantization mismatch behind the inference tolerance.

Cover:

- exact output shape, dtype, device, and finite values;
- comparison with the dense or dequantized reference using stated absolute and relative tolerances;
- smallest legal shape, representative decode and prefill shapes, non-divisible tails, and padding paths;
- every supported quantization layout and metadata branch;
- the intended architecture plus an explicit skip/fallback on unsupported hardware;
- repeated calls, multiple devices where available, non-default streams when relevant, and save/reload integration through the backend.

A compilation-only result must be labeled compilation-only. A passing kernel unit test does not establish model quality.

## Benchmark after correctness

Put reproducible benchmarks in `scripts/`. Warm up compilation and steady-state launches, synchronize correctly or use CUDA events, report distribution statistics rather than one timing, and compare against both the reference and the nearest production kernel. Include shapes, tokens/batch regime, dtype, quantization config, GPU properties, software stack, latency, throughput, and memory. Present the complete result table in ASCII.

## Audit generated instructions after every kernel commit

Every commit or named phase that can change generated GPU instructions requires
a matched generated-code audit on the target GPU. This includes source changes,
template/specialization changes, launch geometry, compiler flags, and constants
that affect unrolling or control flow.

1. Profile the affected steady-state kernel with Nsight Compute or an equivalent
   profiler that reports executed instructions. Bind the report to the exact
   committed revision, binary/JIT fingerprint, shape, dtype, rate, and launch.
2. Export and inspect source-correlated SASS. Compare with the preceding committed
   implementation at an identical workload; source-level operation counts are
   not evidence of generated instruction reduction.
3. Perform an explicit math/algebra and movement review against the new hot SASS:
   fold equivalent expressions, eliminate common subexpressions, deduplicate
   decode/address work, reuse values across consumers, and remove redundant
   masks, shifts, conversions, permutations, and store/load round trips.
4. Check for compiler regressions even when the source looks simpler. A new
   compilation can reintroduce opcode families removed by an earlier phase or
   exchange fewer instructions for more registers, spills, bank conflicts,
   barriers, dependency latency, or scheduler stalls.
5. Record before/after executed instructions and dominant opcode families,
   registers, spills/local memory, shared-memory conflicts, occupancy, scheduler
   eligibility, dominant stalls, achieved memory/compute throughput, exact
   commands, and report paths.
6. Re-run numerical correctness and warmed CUDA-event end-to-end timing after the
   profiler capture. Promote only when those gates pass; an instruction-count win
   alone is not a latency win.

If the target GPU or instruction profiler is unavailable, mark the result
compilation-only. Do not describe the commit/phase as complete or as a kernel
performance win until this audit can run.

## Modern CUDA execution and handoff rules

A warp is still 32 lanes and issues instructions as a SIMT unit, but Volta+ independent
thread scheduling means implicit warp-synchronous memory ordering is not a correctness
contract. Keep synchronization scoped to the actual producer/consumer set:

- Same warp: prefer shuffles/register routing; use `__syncwarp(mask)` when shared-memory
  ordering or reconvergence is required.
- Different warps in one CTA: `__syncwarp()` cannot synchronize them. Use
  `__syncthreads()`, `cuda::barrier`/mbarrier, or the architecture pipeline primitive
  that owns the shared handoff.
- Different CTAs: use a kernel boundary, a proven cooperative-grid barrier, or Hopper+
  cluster synchronization/DSM when the launch is explicitly cluster-scoped.
- A memory fence is not a rendezvous. TMA/async-copy completion also has proxy/barrier
  semantics that ordinary control synchronization does not replace.

Warp specialization is a performance pattern, not a relaxation of those rules. A loader,
decoder, MMA issuer, and epilogue warp may follow different code paths only when their
handoffs, buffer reuse, barrier arrivals, and tail/inactive cases are explicitly proven.

On Hopper/Blackwell, a single elected thread can initiate a TMA or architecture-specific
Tensor Core operation while other warps do independent work. On Ampere, `cp.async`
can similarly separate global-to-shared movement from arithmetic. Always validate that
the extra stages/roles improve event-timed latency rather than merely increasing overlap
in a source diagram.

## See also

- [Curated GPU performance engineering resources](references/wafer-gpu-perf-resources.md) — External reading list from wafer-ai's performance engineering index.
