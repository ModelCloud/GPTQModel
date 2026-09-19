---
name: gptqmodel-amd-kernels
description: Optimize, benchmark, profile, or review GPT-QModel kernels on AMD ROCm GPUs. Use for MI300/MI350/MI355, gfx942/gfx950, HIP, Triton-on-ROCm, FlyDSL, Gluon, AITER, Primus-Turbo, torch._scaled_mm, hipBLASLt/rocBLAS dispatch, rocprof traces or counters, and AMDGCN ISA/SSA algebra reduction.
---

# GPT-QModel AMD kernels

For QVQ optimization, first read [qvq-kernel-accuracy](../qvq-kernel-accuracy/SKILL.md) for accuracy-preserving math
and the locked numerical contract; apply it before selecting lower precision or ranking performance candidates.

Optimize the measured production path on the runtime-probed AMD architecture. Preserve CPU, CUDA, Metal, and
unsupported-ROCm fallbacks. Also use `$gptqmodel-cuda-kernels` for the common numerical/JIT contract,
`$gptqmodel-gpu-testing` for formal GPU runs, `$gptqmodel-gpu-profiling` for attribution, and
`$gptqmodel-mega-kernels` when reducing launches or fusing phases.

Read [references/backend-lookup.md](references/backend-lookup.md) before selecting or installing a GEMM backend.
Recheck upstream HEADs before adopting an API because these projects move quickly.

## Lock the comparison

1. Fetch `origin/main`, integrate it without discarding user work, and record the resulting source revision.
2. Name the last retained AMD kernel commit as the performance baseline. Do not silently substitute `main`.
3. Record the exact model shapes, dtype, quantization rate/layout, strides, warmup, sample count, and cache state.
4. On QVQ inference, compare identical packed tensors against the canonical FP32 computation. Repeated synchronized
   timing must show a positive gain outside measurement noise: a gain no greater than 1% requires mean-absolute
   drift `<= 3e-3`, while a gain greater than 1% permits mean-absolute drift `<= 4e-3`. Equal, slower, or
   noise-indistinguishable candidates receive no drift allowance. Max-absolute drift remains `<= 0.046875` in every
   case, with finite outputs. All applicable limits are inclusive; a geometric-mean win or pooling cases cannot
   waive a failing case.
   Gate localized kernel outputs on identical inputs, weights, and initial state, not propagated final logits.
   Final-logit differences are diagnostics and do not determine this kernel acceptance decision.
5. Run the complete requested M/N/K/rate sweep before promotion. Treat a targeted-shape run as exploratory evidence.

## Establish an AMD idle gate

Before importing Torch or initializing HIP, resolve the physical device with `amd-smi` or `rocm-smi`. Reject foreign
KFD processes, unexpected VRAM residency, or nonzero GFX utilization for three consecutive samples. Record BDF,
unique ID, device name, `gcnArchName`, CU count, VRAM, ROCm/HIP, PyTorch, Triton, and profiler versions. Recheck for
new PIDs immediately before timing. Never kill an unrelated process to make a benchmark pass.

## Select the smallest viable backend

Benchmark candidates by exact shape; do not assume one library wins the whole sweep.

1. Reuse the retained specialized kernel when it is already best.
2. For gfx950 FP16/BF16 GEMM, try AITER's tuned dispatcher and FlyDSL HGEMM, then a shape-specific Gluon kernel.
3. Try Primus-Turbo when its packaged GEMM/GroupedGEMM contract matches the tensors and deployment dependencies are
   acceptable.
4. Use `torch._scaled_mm` only for its supported scaled low-precision contract. It is not a generic replacement for
   FP16 residual GEMMs. Probe the installed operator schema and correctness at runtime.
5. Use `torch.mm`/`matmul`, or explicit hipBLASLt/rocBLAS through an established wrapper, as the portable ROCm
   fallback. Verify the actual dispatched kernel with a trace rather than inferring it from the Python call.
6. Write a custom Triton, Gluon, FlyDSL, or HIP kernel only when the library paths leave a measured gap or fusion can
   eliminate a material intermediate/launch boundary.

Keep backend imports lazy and architecture gates explicit. Cache only immutable weight transforms, account for their
per-layer VRAM, and preserve stream and graph-capture semantics.

## Profile and reduce generated math after every device-code commit

1. Capture a bounded `rocprofv3` HIP/kernel trace after JIT and allocator warmup. Report launches per forward and
   p50/mean/p95/min/max for the changed stage; use synchronized benchmark timing for the performance claim.
2. Use `rocprof-compute` for counters when its backend works. Record the exact error when PMC/AQL collection is broken;
   never invent or extrapolate counters.
3. Preserve Triton TTIR, TTGIR, LLVM IR, AMDGCN assembly, JIT key, launch geometry, VGPR, AGPR, SGPR, LDS, scratch,
   and occupancy metadata for the changed specialization.
4. Compare before/after basic blocks and count SALU, VALU, VMEM, LDS, branch, compare, conditional-mask, barrier, and
   MFMA instructions. Trace repeated address, mask, conversion, add, and store/load algebra back to source.
5. Prove constexpr flags disappeared from generated control flow. Apply one algebraic rewrite at a time, then confirm
   that the generated ISA actually shrank without new spills, occupancy loss, or longer dependency chains.
6. Re-run focused correctness, the exact shape benchmark, and a matched trace before retaining the commit. Profile
   documentation-only commits only when they change no generated GPU code.

Store compact certification and profile summaries under `artifacts/`; keep raw profiler databases and large compiler
dumps untracked unless explicitly requested. Log rejected experiments so later work does not repeat them.

## CDNA instruction scheduling is a first-class kernel concern

On CDNA GPUs, instruction scheduling can be one of the hardest parts of writing a
high-performance kernel. In regions that need a balanced mix of VALU and matrix
(MFMA) work, source order is only a request: the compiler may reschedule,
cluster, or otherwise transform instructions in ways that destroy the intended
latency hiding and pipeline balance.

- Delimit performance-critical scheduling regions conceptually and define the
  intended VALU/MFMA overlap, dependency chains, and latency-hiding strategy
  before tuning source syntax.
- Never assume a source-level interleave survived compilation. Inspect the
  generated AMDGCN ISA for the exact production specialization and verify the
  actual VALU/MFMA ordering, waits, dependencies, register pressure, spills,
  occupancy, and stalls.
- Treat the compiler scheduler as an optimization participant that must be
  measured, not as an authority whose schedule is automatically better. A
  rewrite is useful only if the emitted schedule and end-to-end timing improve.
- When repeated high-level rewrites cannot make the compiler preserve a required
  schedule, prefer the smallest possible lower-level escape hatch for the hot
  region: inline assembly, an ISA-oriented generator/DSL, or effectively
  assembler with syntactic sugar. Keep surrounding control flow, dispatch, and
  portability at a higher level and retain a tested fallback.
- Do not optimize instruction counts in isolation. A visually cleaner schedule
  can still lose through longer dependency chains, extra waits, VGPR/AGPR
  pressure, spills, reduced occupancy, or worse memory overlap.

For CDNA tuning, emitted ISA and measured hardware behavior are the ground truth;
source appearance is not.

## Reference: lessons from local-inference-lab/b12x

Use [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) as a
design reference when working on latency-sensitive GPU kernels. It targets
SM120/SM121 rather than CDNA, so copy principles and validation discipline, not
architecture-specific instructions.

Highlights worth carrying into QVQ/Inference-Ultra work:

- **Plan -> prepare -> bind/run.** b12x separates declaration and tuning from
  execution. Compilation, autotuning, scratch sizing, materialization, and
  specialization happen before serving; after preparation/freeze, execution is
  not allowed to discover a new kernel. Follow the same principle for CDNA:
  no compilation, hidden autotuning, allocation, or policy discovery in a hot
  request or captured graph.
- **Compile-time specialization instead of runtime branches.** b12x uses
  constexpr-style specialization so one logical kernel family can support
  multiple modes while dead paths disappear from emitted code. For AMD kernels,
  prefer compile-time specialization for layout, dtype, correction, and pipeline
  modes when it avoids divergent control flow or unnecessary register pressure.
  Verify the supposedly dead code is actually absent in AMDGCN ISA.
- **Emitted code is the contract.** b12x validates PTX/SASS, and in some ports
  explicitly compares instruction classes and hot-path code against a known
  implementation. Apply the same discipline on CDNA with LLVM/AMDGCN ISA and
  rocprof counters: source resemblance is not evidence of equivalent scheduling
  or cost.
- **Launch geometry can dominate kernel math.** Its SM120 work found large wins
  from tile choice, split-K/wave balancing, and occupancy-aware launch decisions
  without changing arithmetic. Before rewriting a CDNA mainloop, sweep tile,
  wave, split, work-queue, and residency choices and prove whether the limiter
  is instruction throughput, latency hiding, occupancy, or memory.
- **Reuse proven synchronization protocols.** b12x's attention work scaled a
  known-good producer/consumer and barrier protocol rather than re-deriving a
  larger pipeline from scratch. For CDNA LDS/pipeline work, preserve a proven
  wait/barrier protocol when increasing stages or worker groups; change one
  concurrency dimension at a time and test for deadlock and ordering failures.
- **Separate work scheduling from arithmetic.** Its MoE designs distinguish
  persistent arithmetic domains, materialized queues, and readiness-aware queues
  based on when work becomes knowable and how variable its cost is. Use the same
  reasoning for AMD MoE/grouped GEMM: choose static grid ownership versus atomic
  work stealing from measured task variance and readiness, not from dtype alone.
- **Graph-safe frozen execution.** Prepared scratch and persistent state are
  caller-owned or plan-owned, and graph replay must not allocate, compile, or
  perform host-dependent policy checks. Preserve this model for HIP graphs and
  external runtimes.
- **Numerical truth and performance truth are separate gates.** b12x uses
  operation-specific numerical oracles, graph-replay checks, and representative
  timing rather than accepting a speedup because output merely “looks right.”
  Keep QVQ's existing numerical contract authoritative and treat PTX/ISA parity,
  graph safety, and latency as independent evidence.
- **Regime hints beat live-shape policy in frozen serving.** b12x often selects
  tiles from a declared decode/prefill regime while allowing multiple live sizes
  to reuse the same prepared kernel. For CDNA, prefer an explicit prepared
  regime/capacity key over per-request tuning or shape-sensitive compiler work.
- **Gate architecture assumptions early.** Before a large port, b12x first
  proves critical mechanisms on real hardware (for example async-copy/barrier
  behavior or dynamic shared-memory feasibility). Do the AMD equivalent with
  tiny hardware probes for required MFMA forms, LDS footprint, wave behavior,
  async/global-to-LDS mechanisms, and compiler scheduling before committing to
  a full kernel architecture.

Specific b12x patterns to study include its sparse-MLA warp-specialized
producer/consumer pipeline, wave-balanced split-K launch planning, dense GEMM
tile-regime selection, persistent/dynamic MoE work sources, startup preparation
and tuning cache, frozen graph-safe execution, and independent numerical-oracle
qualification.

Do not cargo-cult SM120 mechanisms such as TMA, mbarrier, or NVIDIA MMA forms
onto AMD. Translate the underlying intent—overlap, bounded synchronization,
compile-time specialization, occupancy, and deterministic prepared execution—
into the native CDNA/ROCm mechanism, then verify the generated ISA and measured
behavior.

