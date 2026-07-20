---
name: gptqmodel-hopper-kernels
description: Optimize, review, port, or validate GPT-QModel CUDA and Triton quantized kernels for NVIDIA Hopper H100 sm_90 GPUs. Use for TMA, thread-block clusters, distributed shared memory, WGMMA, FP8, Hopper resource limits, sm_90 versus sm_90a builds, or H100 benchmarks; combine with the general CUDA-kernel skill.
---

# GPT-QModel Hopper kernels

Use this skill with `$gptqmodel-cuda-kernels`. Separate portable Hopper code generation from architecture-accelerated code, and separate build evidence from runtime evidence.

Read [references/hopper-notes.md](references/hopper-notes.md) before choosing `sm_90` or `sm_90a`.

## Establish available evidence

Probe the same live properties used by the Ampere workflow. If no compute-capability 9.0 device is available:

- inspect and compile Hopper paths when the installed toolchain supports them;
- run fallback and architecture-gate tests on available hardware;
- mark Hopper runtime tests skipped with the detected capability;
- make no H100 correctness, latency, throughput, occupancy, or speedup claim.

The 2026-07-20 audit host had no Hopper GPU. An actual H100 run is therefore required before merging a performance claim produced solely on this host.

## Choose the target correctly

1. Use `sm_90` for code intended to remain compatible across compute-capability 9.0 implementations.
2. Use `sm_90a` only for architecture-accelerated features whose generated code is not forward- or backward-compatible. Isolate that binary/path and retain a generic or older-architecture fallback.
3. Gate TMA, WGMMA, thread-block clusters, and distributed shared memory in both build selection and runtime dispatch.
4. Treat FP8 format, scaling, accumulation, saturation, and output dtype as explicit quantization contracts. Native FP8 throughput does not remove the need for accuracy and save/load tests.
5. Derive SM count, cluster limits, shared-memory opt-in, and launch bounds from the live device rather than a canonical H100 SKU.

## Tune Hopper features only when justified

- Use TMA for multidimensional transfers whose descriptors and alignment are stable enough to amortize setup.
- Use clusters and distributed shared memory only when cross-block cooperation outweighs scheduling constraints and reduced residency.
- Tune warp-group MMA pipelines with register pressure, stage count, shared memory, and producer/consumer synchronization considered together.
- Compare specialized Hopper paths to the repository's production alternatives, including Machete or Marlin where their method/format contract matches.
- Measure decode and prefill separately; large Hopper GEMM gains do not imply small-M latency gains.

Retain a deterministic dense/dequantized reference and test architecture dispatch independently from kernel math.
