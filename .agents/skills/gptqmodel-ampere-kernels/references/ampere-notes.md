# Ampere reference notes

Architectural limits come from NVIDIA's [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/) and the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). Also apply the repository's [A100+ architecture contract](../../gptqmodel-cuda-kernels/references/nvidia-a100-plus.md). Query the live device before allocating resources or selecting launch geometry.

## A100 / sm_80 facts

- Compute capability 8.0; warp size 32; up to 64 resident warps and 32 resident blocks per SM.
- 64K 32-bit registers per SM and at most 255 registers per thread.
- Shared memory: up to 164 KiB per SM and 163 KiB addressable by one block, with dynamic shared-memory opt-in above the portable 48-KiB static/default boundary.
- The combined L1/texture/shared-memory pool is 192 KiB on cc 8.0.
- Ampere adds hardware-accelerated global-to-shared asynchronous copy. The CUDA async-copy path supports aligned 4/8/16-byte transfers; 16-byte alignment is the preferred performance case.
- BF16 and TF32 Tensor Core modes are available. Treat TF32/reduced precision as an explicit numerical decision.
- Ampere does **not** provide Hopper TMA, WGMMA, thread-block clusters/DSM, or Hopper-native FP8 execution.

## Memory and fragment rules

- Shared memory has 32 banks with 4-byte bank granularity. For byte address `a`, use `bank=floor(a/4)%32`. Two adjacent FP16 values share one bank word.
- Padding a regular 32-wide FP32 transpose tile to stride 33 is a useful pattern, not a universal rule. For arbitrary indexed lookups, derive the bank equation or use a bank-aware replicated/pair layout when its footprint is justified.
- Global coalescing is a warp-address property. Thirty-two contiguous FP32 values span 128 bytes and are serviced through compact sectors; a warp with a large lane stride can waste many sectors even if every individual load is aligned.
- Vectorized 8/16-byte loads require natural alignment. Validate storage offsets and row/tile strides, not only allocation alignment.
- `ldmatrix` and MMA fragments have architecture-defined lane ownership. Prefer shared/load layouts that naturally feed the fragment over an expensive post-load shuffle network.

## Async-copy pipeline

- Use `cp.async`/`cuda::memcpy_async` when it removes register staging or hides useful global latency.
- The consumer must not read the destination before the matching wait. Compute Sanitizer racecheck can diagnose invalid async-copy synchronization on Ampere+.
- Stage count is a resource tradeoff: more stages consume shared memory and can reduce block residency. Sweep stages with registers and eligible warps, not in isolation.
- Do not assume a double buffer wins at small M or tiny K; producer/commit/wait instructions can cost more than the latency hidden.

## Occupancy and spills

Occupancy is not a target percentage. Diagnose whether the kernel has enough active/eligible warps to cover its dependencies.

- Do not hard-code an SM count from any host snapshot. Derive grids from work and query live properties.
- High registers/thread can reduce residency, but forcing `--maxrregcount` may spill and regress. Reduce live ranges or fragment count first.
- Local memory is backed by device memory and cache hierarchy. Spills add instructions and cache/memory pressure even when they hit in L1/L2; they do not necessarily appear as pure DRAM traffic.

## Tensor Core shape claims

Do not use blanket rules such as "FP16 dimensions must be multiples of 8 or Tensor Cores turn off." Legal MMA shapes are instruction/library specific, and libraries may pad, select a different kernel, or use SIMT for tails. Prove Tensor Core use from generated instructions/profiler activity for the exact shape.

## Launch and fusion claims

Do not assume a universal CUDA launch-overhead number. Measure host launch/API gaps with Nsight Systems on the actual stack and graph mode. Fusion is justified only when it removes measured launch/materialization cost without making the resource union slower.

## Validation checklist

- Confirm the generated code contains the intended sm_80 path and no Hopper-only instruction dependency.
- Test tails, long reductions, decode and prefill regimes, current/non-default streams, and graph replay where supported.
- Compare against the matched dense/dequantized reference and closest production backend.
- Capture source-correlated SASS for meaningful kernel changes and report registers, spills, bank conflicts, sectors/traffic, eligible warps, and dominant stalls.
- Record driver, toolkit/runtime, PyTorch, compiler flags, JIT fingerprint, live GPU properties, and benchmark isolation state.
