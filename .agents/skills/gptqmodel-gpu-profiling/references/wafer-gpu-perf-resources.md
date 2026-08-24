# Curated GPU performance engineering resources (GPU profiling)

Curated from <https://github.com/wafer-ai/gpu-perf-engineering-resources> (accessed 2026-08-24).

## Profiling and correctness tools

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/) — System timelines, CPU-GPU interaction, and distributed traces.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) — Kernel metrics, sections, replay, and roofline analysis.
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/) — Memory, race, initialization, and synchronization checks.
- [CUTLASS GEMM measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html) — Reproducible GEMM benchmarking.
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) — AMD performance counters and roofline analysis.

## Performance model

- [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)
- [Benchmarking GPUs to Tune Dense Linear Algebra](https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf)

## Inference measurement context

- [Etalon](https://arxiv.org/html/2407.07000) — TTFT, TPOT, goodput, and latency SLOs.
- [MLPerf Inference](https://www.cs.toronto.edu/ecosystem/papers/ISCA_20/MLPerf%20Inference.pdf)
