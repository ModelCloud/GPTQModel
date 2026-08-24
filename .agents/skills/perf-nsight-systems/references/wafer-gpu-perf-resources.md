# Curated GPU performance engineering resources (Nsight Systems / system profiling)

Curated from <https://github.com/wafer-ai/gpu-perf-engineering-resources> (accessed 2026-08-24).

## System-level profiling

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/) — System timelines, CPU-GPU interaction, and distributed traces.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) — Kernel metrics, sections, replay, and roofline analysis.
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/) — Memory, race, initialization, and synchronization checks.
- [CUTLASS GEMM measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html) — Reproducible GEMM benchmarking.

## Distributed inference context

- [NCCL](https://github.com/NVIDIA/nccl) — NVIDIA's collective communication implementation.
- [Multi-node NVLink Systems Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/) — NVLink and InfiniBand topology in GB200 NVL systems.
- [Megatron-LM](https://arxiv.org/abs/1909.08053) — Tensor and pipeline parallelism.
