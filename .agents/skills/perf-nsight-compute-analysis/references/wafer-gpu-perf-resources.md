# Curated GPU performance engineering resources (Nsight Compute / kernel profiling)

Curated from <https://github.com/wafer-ai/gpu-perf-engineering-resources> (accessed 2026-08-24).

## Profiling and benchmarking

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) — Kernel metrics, sections, replay, and roofline analysis.
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/) — Memory, race, initialization, and synchronization checks.
- [CUTLASS GEMM measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html) — Reproducible GEMM benchmarking.
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) — AMD performance counters and roofline analysis.

## Roofline and performance models

- [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)
- [Benchmarking GPUs to Tune Dense Linear Algebra](https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf)
- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) — Layouts, tiling, PTX, machine code, and roofline analysis.

## Kernel implementation context

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
