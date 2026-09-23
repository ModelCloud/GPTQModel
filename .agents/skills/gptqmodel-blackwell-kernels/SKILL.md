---
name: gptqmodel-blackwell-kernels
description: Optimize, review, port, or validate GPT-QModel CUDA/CUTLASS/Triton kernels for NVIDIA Blackwell GPUs (compute capability 10.x and 12.x). Use for TCGen05, Tensor Memory (TMEM), TMA pipelines, block-scaled FP8/FP6/FP4, Blackwell shared-memory/resource limits, architecture-family targets, and Blackwell benchmarking; combine with the general CUDA-kernel skill.
---

# GPT-QModel Blackwell kernels

Use this skill with `$gptqmodel-cuda-kernels`. Blackwell is not one uniform target:
data-center cc 10.x and client/workstation cc 12.x devices differ in resource limits
and supported architecture-accelerated instructions. Probe the exact live device and
compile the exact supported path before making a runtime or performance claim.

Read [references/blackwell-notes.md](references/blackwell-notes.md) and the shared
[NVIDIA A100+ architecture contract](../gptqmodel-cuda-kernels/references/nvidia-a100-plus.md).

## Establish the exact target

Record compute capability, product/UUID, SM count, shared memory per SM/block,
register limits, clocks/power mode, HBM/GDDR properties, driver, CUDA toolkit,
CUTLASS version, and JIT/build fingerprint.

- Do not treat `sm_100a` or any architecture-family target as a generic PTX
  fallback. Architecture-accelerated code has explicit compatibility rules.
- Do not infer TCGen05/TMEM support solely from the product name "Blackwell".
- Keep cc 12.x client/workstation devices separate from cc 10.x data-center
  tuning when resource limits or instruction support differ.

## TCGen05 and Tensor Memory

On supported Blackwell architecture targets, TCGen05 changes matrix-multiply
operand and accumulator ownership. Tensor Memory (TMEM) is a distinct on-chip
storage mechanism used by these operations; it is not another name for TMA.

Before using TCGen05:

1. Prove the exact instruction/operand shape and dtype combination exists for the
   selected architecture target.
2. Define which operand resides in registers/shared memory and which accumulator
   lives in TMEM.
3. Budget TMEM, registers, shared memory, CTA geometry, and epilogue extraction
   together. Moving accumulators out of ordinary registers can reduce one
   pressure while creating a new TMEM-to-register epilogue cost.
4. Respect asynchronous completion, commit/wait/fence semantics, and accumulator
   lifetime. Do not overwrite producer buffers or TMEM regions early.
5. Inspect generated SASS/TCGen05 activity and event-timed latency. CUTLASS/CuTe
   syntax alone is not proof that the expected hardware path executed.

## TMA and pipelines

Use Blackwell TMA/CUTLASS pipeline abstractions when they remove address/load
instruction streams or enable overlap. Keep producer/consumer roles and barrier
transaction counts explicit.

- Swizzle only structured layouts whose TMA descriptor and consumer agree.
- Sweep stage count with shared/TMEM/register resource use and active/eligible
  warps; deeper is not automatically faster.
- Do not port a Hopper WGMMA schedule mechanically. TCGen05/TMEM can change the
  best tile, epilogue, and role split.

## Narrow formats and block scaling

Treat FP8/FP6/FP4 and microscaled/block-scaled formats as explicit numerical
contracts:

- record element format, scale dtype, scale block shape/orientation, saturation
  and NaN/Inf behavior, accumulator dtype, and output rounding;
- preserve scale-layout ownership across pack/save/load and kernel dispatch;
- compare against the same quantized/reconstructed values and declared reference;
- validate tails where scale blocks are partial;
- do not call two E2M1/FP4 paths equivalent when their scaling schemes differ.

## Shared memory, banks, and movement

The same 32-bank/4-byte-granularity reasoning applies unless the exact instruction
has a documented specialized layout. Derive bank equations for ordinary shared
loads/stores; for WGMMA/TCGen05 operands use the official CUTLASS/CuTe layout
selected for the target.

A 128-byte-aligned allocation is not sufficient to guarantee coalescing or
conflict-free shared access. Prove lane-to-lane address compactness and shared
bank mapping, then verify sectors/wavefronts in Nsight Compute.

## Validation

- Compile the exact Blackwell target and a non-Blackwell fallback.
- Run deterministic numerical tests, long-K/tails, repeated calls, non-default
  streams, graph replay, and the public dispatch path.
- Capture matched Nsight Compute/SASS evidence for changes that affect generated
  instructions: TCGen05/TMA activity, registers, TMEM/shared footprint, spills,
  occupancy/residency, scheduler eligibility, shared conflicts, cache/DRAM
  traffic and dominant stalls.
- Benchmark decode and prefill separately against the closest production kernel.
  Do not project H100 speedups onto Blackwell.
