# Hopper reference notes

Architectural limits come from NVIDIA's [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) and compatibility semantics from the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/). Query the live runtime before allocating resources.

## H100-class facts

- Compute capability: 9.0.
- Combined L1/shared-memory capacity: 256 KiB per SM.
- Shared-memory capacity: up to 228 KiB per SM and 227 KiB per block, subject to opt-in and runtime limits.
- Hopper adds Tensor Memory Accelerator transfers, thread-block clusters, distributed shared memory, warp-group MMA, and native FP8-oriented Tensor Core paths.
- `sm_90a` architecture-accelerated code is not forward- or backward-compatible; do not substitute it silently for a generic `sm_90` binary.

## Validation matrix

| Evidence | Without H100 | With H100 |
| --- | --- | --- |
| Source/static review | Required | Required |
| Hopper-target compilation | When toolchain supports it | Required |
| Runtime architecture dispatch | Validate rejection/fallback | Validate specialized selection and fallback |
| Numerical kernel result | Not established | Compare tails and long-K cases with reference |
| Performance | No claim | Warmed, synchronized decode and prefill benchmarks |
| Cluster/TMA behavior | Not established | Exercise resource limits, capture, streams, and repeated launches |

Record the exact H100 variant, SM count, memory, clocks/power state when controlled, driver, CUDA toolkit/runtime, PyTorch, compiler flags, and whether the binary used `sm_90` or `sm_90a`.
