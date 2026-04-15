# W4A4 Prototype Plan

Date: 2026-04-15
Current branch: `w4a8`
Base prototype commit: `4a2b4076`
Current status: `prototype working, packed-A4 staged inside GEMM`

## Status

Overall status: `functional prototype with in-kernel packed-A4 staging`

What is complete:

- `QQQConfig` supports `activation_bits={4,8}`
- checkpoint save/load preserves `activation_bits`
- `QQQLinear` can execute `A8` and packed `A4`
- a dedicated `qqq_w4a4_gemm` torch op exists
- packed `A4` is unpacked inside the GEMM tile staging path
- there is no standalone activation unpack kernel or dense temporary `A8` tensor
- end-to-end reload, generate, and prefill benchmark works on GPUs `8` and `9`

What is not complete:

- tensor-core path that consumes packed `A4` more natively than the current shared-memory unpack staging
- optimized decode and prefill performance
- accuracy evaluation beyond smoke testing
- deeper kernel tuning for occupancy, shared memory staging, and register pressure

## Measured Progress

Validation completed on 2026-04-15 with `/monster/data/model/Llama-3.2-1B-Instruct` and checkpoint `/tmp/llama3_2_1b_instruct_qqq_w4a4_gpu8`.

Results:

- GPU `8`: prefill `A8 40.75 ms`, `A4 55.20 ms`, `1.35x` slower
- GPU `9`: prefill `A8 41.95 ms`, `A4 58.94 ms`, `1.40x` slower
- GPU `8`: generate `32` tokens in `2.146 s`
- GPU `9`: generate `32` tokens in `2.259 s`

Interpretation:

- functionality is proven on both requested A100s
- the unpack bridge limitation is removed
- the remaining loss is now inside the kernel path itself, not a separate activation materialization step
- current performance is still not good enough to justify the path as a production kernel

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

### M1: In-Kernel Packed-A4 Staging

Status: `done`

Scope:

- direct packed `A4` consumption by the `W4A4` GEMM entrypoint
- remove temporary unpack-to-`int8` tensor
- unpack directly into the GEMM shared-memory `A` tile

Exit criteria:

- no standalone unpack kernel
- stable execution on both PCI-order GPUs `8` and `9`

### M2: More Native Packed-A4 Kernel

Status: `next`

Scope:

- reduce the critical-path cost of A4 unpack
- improve overlap between packed-A4 staging, W4 dequant, and MMA
- evaluate whether cp.async-assisted packed-A4 staging or a deeper kernel rewrite is needed

Exit criteria:

- `W4A4` prefill is no slower than current `W4A8` baseline on local A100s
- kernel remains stable on both PCI-order GPUs `8` and `9`

### M3: Accuracy Validation

Status: `pending`

Scope:

- perplexity or task benchmark against `W4A8`
- compare degradation from `A8 -> A4`

Exit criteria:

- acceptable loss relative to runtime savings

### M4: Production Hardening

Status: `pending`

Scope:

- broader shape coverage
- better test coverage
- extension build stability
- documentation updates in main docs

Exit criteria:

- production-ready runtime path with reproducible benchmarks

## Immediate Next Tasks

1. Profile the packed-A4 fetch path and quantify where the remaining slowdown is spent.
2. Decide whether to add cp.async-assisted packed-A4 staging or move to a more native nibble-aware MMA feed path.
3. Benchmark decode separately from prefill on GPUs `8` and `9`.
4. Measure accuracy against the existing `W4A8` QQQ path.

## Risks

- `A4` may not recover enough bandwidth savings to beat `A8` once unpack and scale handling costs are included.
- Accuracy loss from runtime `A4` activation quantization may be too large for production use.
- A100 `sm_80` lacks native FP8/FP4 tensor-core modes, so all gains depend on integer path quality.
- A more native fused kernel is materially more complex than the current staging-level prototype.
