# Blackwell reference notes

Primary sources:
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS Blackwell functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)

Also apply the repository's [A100+ architecture contract](../../gptqmodel-cuda-kernels/references/nvidia-a100-plus.md).

## Do not assume one Blackwell resource table

CUDA's current compute-capability table exposes materially different resource
limits across cc 10.x and cc 12.x. Query the live device and use the capability
table/runtime API for the exact target.

Examples from the current CUDA table include:
- cc 10.0: up to 228 KiB shared memory per SM and 227 KiB per block;
- some later cc 10.x variants expose larger shared-memory capacities;
- cc 12.0/12.1: 128 KiB shared memory per SM, 99 KiB per block, and 48 resident
  warps per SM rather than Hopper/A100's 64.

These are architecture-family examples, not product-name mappings.

## Architecture-accelerated targets

Blackwell architecture-accelerated instructions such as TCGen05 are gated by
specific architecture/family targets. Treat `sm_*a` or family targets as
explicit binaries with the documented compatibility rules. Keep a separately
tested fallback when distributing beyond that exact family.

## TCGen05 / TMEM

PTX TCGen05 operations are asynchronous and use Tensor Memory for supported
accumulator/data paths. TMEM has its own allocation/addressing/lifetime rules.
A Blackwell GEMM design must therefore budget at least:

```text
registers + shared memory + TMEM + CTA/cluster residency + epilogue movement
```

Do not compare only "register count" with Hopper and conclude the Blackwell path
has more headroom.

## Narrow precision

Blackwell supports additional low-precision and block-scaled Tensor Core modes.
For GPT-QModel, format identity includes the scale representation and block
geometry. Record and test the complete pair:

```text
element dtype + scale dtype/layout/block size
```

rather than using "FP4" or "FP8" as a sufficient ABI description.
