# NVIDIA A100+ CUDA architecture contract

This is the default hardware contract for GPT-QModel GPU kernel and performance work. The primary target is NVIDIA A100 and newer GPUs (compute capability 8.0+). Preserve portable fallbacks when they already exist, but do not spend optimization effort on pre-Ampere GPUs unless the task explicitly asks for it. AMD/ROCm and Apple/Metal have separate skills and are opt-in scopes.

Authoritative references:
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)
- [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/)

Always query the live device. Product name, HBM size, SM count, shared-memory capacity, cluster limits, clock behavior, and supported architecture-accelerated instructions vary by SKU and compute-capability minor version.

## Architecture map

| Family | Typical target | Warp size | Max resident warps/SM | Shared memory/SM | Shared memory/block | Main movement / Tensor Core mechanisms |
|---|---:|---:|---:|---:|---:|---|
| A100 / Ampere | 8.0 | 32 | 64 | 164 KiB | 163 KiB | coalesced GMEM, `cp.async`, `ldmatrix`, `mma.sync`/WMMA |
| H100/H200 / Hopper | 9.0 | 32 | 64 | 228 KiB | 227 KiB | TMA, mbarrier pipelines, WGMMA, clusters/DSM |
| Blackwell data center | 10.x | 32 | query live device | compute-capability dependent | compute-capability dependent | TMA extensions, TCGen05 on supported architecture/family targets, TMEM |
| Blackwell client/workstation | 12.x | 32 | 48 on 12.0 | 128 KiB on 12.x | 99 KiB on 12.x | query the exact supported Tensor Core/TMA feature set |

CUDA's current compute-capability table lists 228/227 KiB shared memory for cc 10.0 and larger limits for some later 10.x variants. Never copy one Blackwell SKU's resource constants to another. Static shared memory above 48 KiB is not portable; larger per-block allocations require dynamic shared-memory opt-in where supported.

## Global-memory access: coalescing is a warp property

Do not equate tensor contiguity, pointer alignment, or vector width with coalescing.

- A warp memory instruction is efficient when the requested addresses touch as few memory sectors/transactions as possible and most transferred bytes are used. Modern CUDA coalescing is naturally reasoned about in 32-byte sectors at L1TEX, while lower hierarchy transactions can combine aligned sectors.
- Thirty-two contiguous FP32 values span 128 bytes and normally require four 32-byte sectors, not "one transaction." Use profiler sectors/request and requested-versus-transferred bytes instead of folklore.
- Natural alignment still matters. A 16-byte `uint4`/vector load requires a naturally aligned address, but a warp of aligned 16-byte loads can still be badly scattered if lane-to-lane addresses are strided.
- Optimize the lane mapping, not only the base pointer. For `A[k][rank]`, a warp fixed on `rank` sees a large stride when `rank` is the minor dimension; a transposed execution cache `A_T[rank][k]` can make the same warp contiguous.
- Prefer immutable load-time repacks when they remove repeated runtime gathers/address arithmetic and the extra storage is justified. Include the repack/version key in cache invalidation.
- Do not add `.contiguous()` blindly. A strided view can still have a perfectly coalesced hot dimension, and a large materialization can cost more than a stride-aware kernel.

For vectorized global movement, prove:
1. base and per-row/per-tile offsets satisfy the instruction's natural alignment;
2. the vector does not cross logical/allocation tails;
3. lane-to-lane vectors form compact sectors;
4. the compiler emitted the intended vector instruction;
5. the full operator is faster after including any repack/materialization.

## Shared memory: 32 banks, 4-byte bank granularity

A100 and newer CUDA GPUs expose 32 shared-memory banks. Successive 32-bit words map to successive banks:

```text
bank = floor(byte_address / 4) mod 32
```

Each bank supplies 32 bits per clock for ordinary shared-memory traffic. Consequences:

- FP32/32-bit lane-contiguous words naturally map lanes 0..31 across banks 0..31.
- Two adjacent FP16 values occupy one 32-bit bank word. Do not assume `half[lane]` gives 32 independent banks. Same-word reads may broadcast, while different words in the same bank create conflicts.
- Bank conflicts are defined per warp memory request, not across unrelated warps.
- Base alignment rotates the bank mapping but does not by itself remove conflicts. The lane/address relationship determines conflicts.
- Regular transpose-style patterns often benefit from padding (for example a logical 32-wide FP32 row stored with stride 33) or an architecture-supported shared-memory/TMA swizzle.
- Data-dependent LUTs are different. No static index-only swizzle can guarantee unique banks for arbitrary per-lane indices. If a hot 32-bit lookup must be conflict-free for arbitrary indices, a lane-replicated layout such as `table[index][lane]` gives `bank=lane`, at the cost of 32x table storage. For FP16, use a pair-aware layout or a lane-private 32-bit word when zero-conflict ownership is worth the footprint.
- Replication can lose overall performance by increasing shared-memory residency pressure. Profile the complete operation, not only the lookup.
- TMA swizzle is for structured tensor layouts. The tensor-map swizzle and consumer indexing must agree. Do not apply it to an arbitrary lookup table simply because bank conflicts exist.

For every proposed shared layout, write the symbolic bank equation for one issued warp instruction and verify with Nsight Compute's shared requests/wavefronts/conflicts.

## Ampere asynchronous copy

On cc 8.x, use `cp.async` / `cuda::memcpy_async` for global-to-shared staging when it removes register-mediated copies or overlaps useful work.

- Hardware acceleration supports aligned 4/8/16-byte forms; 16-byte source/destination alignment is the preferred fast path.
- Commit/wait groups or `cuda::pipeline` ownership are part of correctness. A later shared load before the required wait is a race.
- Double buffering is useful only when enough independent compute exists to hide the transfer and the extra shared memory does not collapse residency.
- Do not emulate Hopper TMA with a forest of scalar `cp.async` instructions without measuring address/instruction overhead.

## Hopper TMA and WGMMA

TMA moves multidimensional tensors between global and shared memory without requiring each participating thread to calculate and issue every element address. Use it when descriptors are stable and the tile is large enough to amortize producer/barrier machinery.

- One elected thread can initiate a TMA operation; the hardware performs the transfer asynchronously.
- The associated mbarrier/pipeline transaction-byte count must cover the bytes actually produced. Incorrect producer/consumer counts or transaction sizes can hang.
- Use CUTLASS/CuTe TMA and pipeline primitives when possible; hand-written barrier protocols require a stronger proof.
- TMA shared-memory swizzle can eliminate structured bank conflicts, but must match the descriptor and WGMMA/consumer layout.
- WGMMA is asynchronous. Respect operand fences, arrive/commit/wait ordering, fragment lifetime, and the register/shared source contract.
- A register-source operand can save shared traffic but increase register pressure and decode/address work. A shared-source operand can improve reuse but consume enough SMEM to force one CTA/SM. Measure both.
- `sm_90a` architecture-accelerated code is not a generic forward-compatible substitute for `sm_90`.

## Blackwell TCGen05 and Tensor Memory

Blackwell is not "Hopper with a faster WGMMA." On supported architecture/family targets, TCGen05 changes operand/accumulator ownership and introduces Tensor Memory (TMEM), which is distinct from TMA.

- Gate TCGen05/TMEM by the exact supported compute capability and architecture/family target (for example `sm_100a`/supported family targets), not by a product-name substring.
- TCGen05 MMA is asynchronous and accumulators can live in TMEM. Epilogues may need explicit TMEM-to-register movement; this changes register and shared-memory budgeting.
- TMA can feed SMEM while TCGen05 consumes it; use pipeline types that match the real producer and consumer. A mismatched barrier protocol can deadlock or race.
- Blackwell narrow formats (FP8/FP6/FP4 and block-scaled variants) require explicit scale-layout, saturation, and numerical contracts. Do not infer accuracy from format names.
- Data-center and client Blackwell variants do not share identical resource limits or architecture-accelerated instruction availability. Compile and run the exact gated path on the target GPU before a performance claim.

## Warp and synchronization model

Warp size is 32, but Volta+ independent thread scheduling invalidates implicit warp-synchronous producer/consumer assumptions.

- Same-warp handoff: use register shuffles where possible; use `__syncwarp(mask)` when ordering/visibility between participating lanes is required.
- Cross-warp handoff inside one CTA: `__syncwarp()` is insufficient. Use `__syncthreads()`, a correctly scoped `cuda::barrier`/mbarrier pipeline, or another CTA-wide primitive whose memory semantics match the handoff.
- Cross-CTA handoff: a CTA barrier is insufficient. Use a kernel boundary, cooperative grid synchronization when residency is proven, or Hopper+ cluster primitives for cluster-scoped cooperation.
- A fence orders memory; it is not automatically a rendezvous. A barrier is not automatically a substitute for the required async-proxy/TMA completion semantics.
- Every participating thread/warp must obey the same barrier protocol, including inactive/tail paths.

Warp specialization is valuable when load/decode/MMA/store roles have enough independent work to overlap. It is not free: role-specific code can increase registers, shared memory, barrier traffic, and code size.

## Streams, overlap, and CUDA Graphs

Use the framework's current CUDA stream and current device. Do not launch custom work on an implicit private/default stream unless that is the declared API.

- Cross-stream producer/consumer dependencies require events or another explicit ordering mechanism. Host launch order across independent streams is not a dependency.
- `cudaDeviceSynchronize()` is a diagnostic hammer, not a production dependency mechanism.
- Overlap helps only when operations have independent dependencies and enough complementary resources. Two kernels can run concurrently yet both slow down because they contend for SMs, Tensor Cores, L2, or HBM.
- Bind stream-sensitive library handles/descriptors to the correct stream and preserve their lifetime through asynchronous use.
- Prepare JIT compilation, autotuning, immutable repacks/descriptors, and ordinary lazy initialization before graph capture. Stream-ordered allocation can be graph-capturable, but ownership and stable replay addresses still require an explicit lifetime design.
- Cache tuning by every code-generation/dispatch dimension: architecture target, compute capability, shape, dtype, layout, stages, warp/CTA geometry, split policy, quantization mode, and build/JIT identity.

## Occupancy, residency, and scheduler diagnosis

Occupancy is a latency-hiding resource, not a score to maximize.

- Compute theoretical residency from registers/thread, threads/CTA, static+dynamic shared memory, block limits, and architecture limits.
- Then inspect achieved active warps and eligible warps/scheduler/cycle. High occupancy with low issue can still be dependency-bound; one-CTA/SM kernels can still win through strong reuse.
- Do not use `--maxrregcount` as a first-line "occupancy fix." It can force spills into local memory and make the kernel slower. Reduce live ranges/data structures first, then measure any cap.
- Local memory is backed by device memory and can be served by caches; spills are not automatically a DRAM transaction, but they add load/store instructions, latency, cache pressure, and often hurt.
- Shared-memory capacity and register pressure often trade directly against pipeline depth. Sweep the complete resource tuple, not one knob at a time in isolation.

## PTX, SASS, and proof

PTX is a virtual ISA; SASS is what executes. Source simplification or PTX instruction count does not establish a hardware win.

For a credible optimization:
1. bind the binary/JIT fingerprint and exact GPU;
2. inspect source-correlated executed SASS;
3. compare opcode families such as address `IMAD/LEA`, bit `LOP3/SHF/PRMT`, global/shared loads, barriers, and HMMA/WGMMA/TCGen05 activity;
4. record registers, spills/local memory, shared bank wavefronts, occupancy/residency, scheduler eligibility, dominant stalls, L1/L2/DRAM traffic, and tensor-pipe utilization;
5. confirm with warmed CUDA-event full-operation timing and the numerical contract.

Low HBM bandwidth is not automatically a memory-alignment problem. A compressed decoder can read few bytes while spending most cycles on integer decode, dependent shared lookup, synchronization, or tensor-core issue. Diagnose the limiting pipeline before optimizing for GB/s.
