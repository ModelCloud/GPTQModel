# Curated GPU performance engineering resources (Hopper/H100 focus)

Curated from <https://github.com/wafer-ai/gpu-perf-engineering-resources> (accessed 2026-08-24).

## Minimum mental model

- [CUDA C++ basics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)
- [GPU Mode lectures](https://github.com/gpu-mode/lectures)

## Hopper architecture and programming

- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) — TMA, thread-block clusters, asynchronous execution, and Hopper-specific limits.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — Coalescing, shared memory, occupancy, synchronization.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Understanding PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)

## Kernel optimization for Hopper

- [CuTe GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html) — Tiling, layouts, copies, and matrix-multiply atoms.
- [CUTLASS 3.x design](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html) — Collectives and kernel structure.
- [CUTLASS pipeline documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html) — Producer-consumer pipelines and asynchronous stages.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — Compact production FP8 GEMM implementation for Hopper.
- [Outperforming cuBLAS on H100: A Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) — Direct Hopper optimization using tensor cores and asynchronous movement.
- [CUTLASS Tutorial: Mastering TMA](https://research.colfax-intl.com/tutorial-hopper-tma/) — Working kernels built around the Tensor Memory Accelerator.

## Tensor cores and low precision

- [OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf) — E4M3 and E5M2 formats.
- [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) — Shared-scale MX formats.
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) — FP8 and FP4 transformer execution with scaling controls.

## Attention

- [FlashAttention-3](https://arxiv.org/abs/2407.08608) — Asynchronous movement and tensor-core overlap on Hopper.
- [FlashAttention-4](https://proceedings.mlsys.org/paper_files/paper/2026/file/ae8b0b5838ba510daff1198474e7b984-Paper-Conference.pdf) — Blackwell attention schedule (future reference for tensor-core scheduling ideas).
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — Attention and related kernels for serving workloads.
