# Hopper reference notes

Architectural limits come from NVIDIA's [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/), the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/), and the repository's [A100+ architecture contract](../../gptqmodel-cuda-kernels/references/nvidia-a100-plus.md). Query the live runtime before allocating resources.

## H100/H200 / sm_90 facts

- Compute capability 9.0; warp size 32; up to 64 resident warps and 32 resident blocks per SM.
- 64K 32-bit registers per SM and at most 255 registers per thread.
- Shared memory: up to 228 KiB per SM and 227 KiB per block, subject to dynamic opt-in and live device limits.
- Hopper adds Tensor Memory Accelerator (TMA), thread-block clusters, distributed shared memory (DSM), warp-group MMA (WGMMA), and native FP8-oriented Tensor Core paths.
- `sm_90a` architecture-accelerated code is not a generic forward/backward-compatible replacement for `sm_90`. Compile and dispatch it explicitly.

## Shared-memory and TMA rules

- Shared memory remains 32 banks with 4-byte bank granularity. FP16 values share banks in adjacent pairs; derive the actual bank equation for the instruction being issued.
- TMA can move 1D through 5D tensors between global and shared memory without a per-element load/address instruction stream. Descriptor setup, alignment/layout, and mbarrier accounting must match the physical transfer.
- TMA swizzle is a structured-layout mechanism. Use the tensor-map/CuTe layout that the consumer expects and verify shared wavefront/conflict metrics.
- Stage count is not free. More stages increase shared-memory footprint and can reduce active CTAs; a two-stage pipeline can beat a conflict-free one-stage layout because it preserves transfer/compute overlap.
- When one elected thread issues TMA, that does not make `__syncwarp` a cross-warp barrier. Consumers use the pipeline/barrier wait that owns the shared tile.

## WGMMA rules

- WGMMA is asynchronous. Treat arrive/commit/wait and operand-fence order as part of the kernel's numerical/correctness contract.
- Register-source versus shared-source is a measured ownership choice. Register source reduces shared storage but can inflate fragment registers, decode work, and dependency chains. Shared source can amortize a decoded tile over consumers but may consume enough SMEM to force one block per SM.
- Do not overwrite/reuse an operand fragment until the matching outstanding WGMMA groups have reached the documented wait point.
- Generated HGMMA/WGMMA activity, not C++/CuTe syntax, proves Tensor Core execution.

## Clusters and DSM

- Use clusters only when cross-CTA reuse or synchronization pays for the scheduling/residency constraint.
- Compute cluster occupancy with the runtime occupancy APIs intended for clusters; do not infer it from ordinary block occupancy.
- A cluster barrier/DSM scope is not a device-wide barrier. Keep global dependencies at a kernel/cooperative-grid boundary when they exceed cluster scope.

## H100 versus H200

H100 and H200 share compute capability 9.0 but are not interchangeable performance baselines. Record live SM count, HBM capacity/bandwidth, clocks/power state, PCI/SXM topology, and software stack. Split/grid/stage policies may differ even when the SASS is identical.

## Validation matrix

| Evidence | Without cc 9.0 hardware | With target Hopper hardware |
| --- | --- | --- |
| Source/static review | Required | Required |
| sm_90/sm_90a compilation | When toolchain supports it | Required |
| Runtime architecture dispatch | Validate rejection/fallback | Validate specialized selection and fallback |
| Numerical result | Not established for Hopper path | Compare tails, long-K, repeated/graph/stream cases |
| Performance | No Hopper claim | Warmed event timing plus matched NCU/NSYS evidence |
| TMA/WGMMA/cluster behavior | Not established | Verify exact generated instructions, resources, barriers and replay |

Record whether the binary used `sm_90` or `sm_90a`, the exact JIT/build fingerprint, and the live device identity.
