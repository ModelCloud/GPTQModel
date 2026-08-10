# Ampere reference notes

Source of architectural limits: NVIDIA's [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html). Query the live device and CUDA API before allocating resources.

## A100-class sm_80 facts

- Compute capability: 8.0.
- Combined L1/shared-memory capacity: 192 KiB per SM.
- Shared-memory capacity: up to 164 KiB per SM and 163 KiB per block, subject to opt-in and runtime limits.
- Asynchronous global-to-shared copy can reduce register pressure and overlap memory movement with computation.
- BF16 and TF32 Tensor Core modes are available; numerical tolerances must reflect the mode actually used.

## Audit-host snapshot

On 2026-07-20, the host exposed eight `PG506-230/232` devices in PCI order. PyTorch reported compute capability 8.0, 124 SMs, and approximately 96 GiB per device. This snapshot may change between runs and must not appear in launch constants or device-index assumptions.

## Build and validation checklist

- Confirm the generated code contains the intended `sm_80` path.
- Test a tail shape and an accumulation-heavy shape as well as the tuned shape.
- Measure both decode and prefill regimes when the kernel serves both.
- Compare against a matched dense/dequantized result and the closest production backend.
- Verify fallback or explicit rejection on non-Ampere devices.
- Record PyTorch, CUDA runtime, NVCC, driver, GPU properties, build flags, and JIT cache/rebuild state.

## GPU kernel facts for Ampere/A100 development

- Tensor cores only activate when matrix dimensions are multiples of 8 for FP16 and 16 for INT8. Odd-shaped matrices silently fall back to regular CUDA cores with no warning, so verify shape alignment before claiming Tensor Core speedup.
- Shared memory is split into 32 banks. If multiple threads in a warp hit different addresses in the same bank, the accesses serialize instead of running in parallel. This is why naive matrix transpose kernels are much slower than they look on paper.
- A100 has 108 SMs (consumer/derivative variants may report up to 124), but occupancy is not about SM count. It is bounded by registers per thread; use too many registers and the scheduler cannot fit enough warps per SM to hide memory latency.
- Kernel launch overhead is roughly 5–10 microseconds. For a model with hundreds of small elementwise ops, that overhead can exceed the compute time, which is the entire reason kernel fusion exists.
- Register spilling does not throw an error. It silently pushes variables into local memory, which physically lives in the same slow DRAM as global memory. The kernel just gets quietly slower and the profiler trace is often unhelpful.
- Coalesced access means a warp reading 32 contiguous floats costs one memory transaction; the same warp reading 32 scattered floats can cost 32 transactions. Row-major versus column-major layout decides which one you get.
