# Curated GPU performance engineering resources

Curated from <https://github.com/wafer-ai/gpu-perf-engineering-resources> (accessed 2026-08-24).

## Minimum mental model

- [How to Scale Your Model: Inference](https://jax-ml.github.io/scaling-book/inference/) — One request from prefill through decode, with batching, KV memory, and parallelism.
- [CUDA C++ basics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) — Shortest official introduction to the CUDA execution model.
- [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) — Main textbook for GPU programming, memory, and kernel design.
- [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) — Compute, memory-bandwidth, and arithmetic-intensity model.
- [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/) — FLOPs, parameter bytes, KV bytes, and communication for transformer inference.
- [GPU Mode lectures](https://github.com/gpu-mode/lectures) — Practical companion for the topics below.

## GPU fundamentals

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) — Normative CUDA reference.
- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) — Threads, warps, blocks, grids, and the memory hierarchy.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — Coalescing, shared memory, occupancy, synchronization, and optimization workflow.
- [NVCC Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) — CUDA compilation trajectory and artifact controls.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) — NVIDIA's virtual instruction set and memory model.
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) — `cuobjdump` and `nvdisasm` for inspecting GPU binaries.
- [Understanding PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/) — Role of PTX between CUDA and machine code.

## Kernel optimization

- [Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) — Coalescing, shared-memory tiling, and bank conflicts.
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) — Synchronization, divergence, occupancy, and instruction cost.
- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867) — Numerically stable online softmax without materialized intermediates.
- [Benchmarking GPUs to Tune Dense Linear Algebra](https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf) — Reasoning from measured hardware behavior instead of occupancy alone.
- [CuTe GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html) — Tiling, layouts, copies, and matrix-multiply atoms.
- [CUTLASS 3.x design](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html) — Collective and kernel structure used by modern CUTLASS.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — Compact production FP8 GEMM implementation for Hopper.
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) — Matmul built from naive CUDA through shared-memory and register tiling.
- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) — Layouts, tiling, PTX, machine code, and roofline analysis.
- [Outperforming cuBLAS on H100: A Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) — Hopper optimization using tensor cores and asynchronous movement.
- [CUTLASS Tutorial: Mastering TMA](https://research.colfax-intl.com/tutorial-hopper-tma/) — Working kernels built around the Tensor Memory Accelerator.
- [FlashAttention](https://arxiv.org/abs/2205.14135) — IO-aware exact attention.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Better work partitioning and parallelism.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) — Asynchronous movement and tensor-core overlap on Hopper.
- [FlashAttention-4](https://proceedings.mlsys.org/paper_files/paper/2026/file/ae8b0b5838ba510daff1198474e7b984-Paper-Conference.pdf) — Blackwell attention schedule.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — Attention and related kernels for serving workloads.

## Programming models and profiling

- [Triton paper](https://eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — Original blocked-program language and compiler design.
- [Triton programming guide](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) — Official programming model.
- [Triton repository](https://github.com/triton-lang/triton) — Compiler, examples, tests, and backend implementation.
- [CuTe layout algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html) — Layouts and layout composition.
- [CUTLASS pipeline documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html) — Producer-consumer pipelines and asynchronous stages.
- [CUDA Tile IR programming model](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html) — NVIDIA's compiler-owned tile abstraction.
- [ROCm Composable Kernel](https://github.com/ROCm/composable_kernel) — AMD tiling, layout, and operator primitives.
- [ROCm AITER](https://github.com/ROCm/aiter) — AMD inference and transformer operator implementations.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) — Kernel metrics, sections, replay, and roofline analysis.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/) — System timelines, CPU-GPU interaction, and distributed traces.
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/) — Memory, race, initialization, and synchronization checks.
