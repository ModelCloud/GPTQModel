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
