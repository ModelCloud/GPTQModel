# W4A4 Prototype Plan

Date: 2026-04-15
Current branch: `w4a8`
Prototype commit: `4a2b4076`

## Status

Overall status: `prototype working`

What is complete:

- `QQQConfig` supports `activation_bits={4,8}`
- checkpoint save/load preserves `activation_bits`
- `QQQLinear` can execute `A8` and `A4`
- a dedicated `qqq_w4a4_gemm` torch op exists
- end-to-end quantize, save, reload, and generate works
- smoke validation completed on GPUs `8` and `9`
- unit tests and a local benchmark driver were added

What is not complete:

- true fused `W4A4` kernel
- optimized decode and prefill performance
- accuracy evaluation beyond smoke testing
- kernel-level tuning for occupancy, shared memory staging, and register pressure

## Measured Progress

Validation completed on 2026-04-15 with `/monster/data/model/Llama-3.2-1B-Instruct`.

Results:

- GPU `8`: `A8 41.65 ms`, `A4 56.70 ms`, `1.36x` slower
- GPU `9`: `A8 42.80 ms`, `A4 56.52 ms`, `1.32x` slower

Interpretation:

- functionality is proven
- current performance is not good enough to justify the path as a production kernel
- the unpack bridge is the main reason the prototype loses to `A8`

## Milestones

### M0: Functional Prototype

Status: `done`

Scope:

- config support
- module support
- CUDA op registration
- checkpoint round-trip
- smoke benchmark

Exit criteria:

- quantize, save, reload, generate, and benchmark on local A100s

### M1: Real Fused Kernel

Status: `next`

Scope:

- direct packed `A4` consumption in the GEMM kernel
- remove temporary unpack-to-`int8` tensor
- keep current scale semantics where possible

Exit criteria:

- `W4A4` prefill is no slower than current `W4A8` baseline
- kernel stable on both PCI-order GPUs `8` and `9`

### M2: Accuracy Validation

Status: `pending`

Scope:

- perplexity or task benchmark against `W4A8`
- compare degradation from `A8 -> A4`

Exit criteria:

- acceptable loss relative to runtime savings

### M3: Production Hardening

Status: `pending`

Scope:

- broader shape coverage
- better test coverage
- extension build stability
- documentation updates in main docs

Exit criteria:

- production-ready runtime path with reproducible benchmarks

## Immediate Next Tasks

1. Replace the unpack bridge with a fused kernel path for packed `A4`.
2. Benchmark prefill and decode separately on GPUs `8` and `9`.
3. Measure accuracy against the existing `W4A8` QQQ path.
4. Decide whether the kernel should stay QQQ-specific or become a more general Marlin-style backend path.

## Risks

- `A4` may not recover enough bandwidth savings to beat `A8` once scale handling and scheduling costs are included.
- Accuracy loss from runtime `A4` activation quantization may be too large for production use.
- A100 `sm_80` lacks native FP8/FP4 tensor-core modes, so all gains depend on integer path quality.
- A new fused kernel is materially more complex than the prototype bridge.
