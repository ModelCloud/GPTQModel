# Curated GPU performance engineering resources (GPU testing and benchmarks)

Curated from <https://github.com/wafer-ai/gpu-perf-engineering-resources> (accessed 2026-08-24).

## Workload and serving benchmarks

- [MLPerf Inference](https://www.cs.toronto.edu/ecosystem/papers/ISCA_20/MLPerf%20Inference.pdf) — Reproducible benchmark scenarios and load generation.
- [Etalon](https://arxiv.org/html/2407.07000) — Goodput under per-request latency SLOs.
- [ServeGen](https://www.usenix.org/system/files/nsdi26-xiang-servegen.pdf) — Workload generation preserving production-trace properties.
- [BurstGPT](https://github.com/HPMLL/BurstGPT) — Public trace for bursty LLM workloads.
- [MLPerf Endpoints](https://mlcommons.org/benchmarks/endpoints/) — Endpoint-level benchmark for interactive generative AI.

## Correctness and performance context

- [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/)
- [Efficiently Scaling Transformer Inference](https://proceedings.mlsys.org/paper_files/paper/2023/file/c4be71ab8d24cdfb45e3d06dbfca2780-Paper-mlsys2023.pdf)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)
- [CUTLASS GEMM measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)
