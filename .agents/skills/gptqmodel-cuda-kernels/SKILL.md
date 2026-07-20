---
name: gptqmodel-cuda-kernels
description: Build, port, optimize, review, benchmark, or debug GPT-QModel CUDA, C++, CUTLASS, or Triton kernels and their JIT wrappers. Use for quantized kernel correctness, extension registration, launch validation, illegal memory access, architecture gating, or performance work; combine with the Ampere or Hopper skill for architecture-specific tuning.
---

# GPT-QModel CUDA kernels

Start from a numerical reference and select the smallest kernel path that can express the operation. Keep correctness tests separate from performance benchmarks.

Read [references/kernel-workflow.md](references/kernel-workflow.md). For crashes or silent corruption, also read [references/cuda-debugging.md](references/cuda-debugging.md).

## Choose the implementation path

1. Keep or create a plain PyTorch reference for correctness and fallback.
2. Prefer Triton for a self-contained tensor kernel whose launch and specialization fit the existing Python JIT style.
3. Prefer CUDA/C++ through `TorchOpsJitExtension` when using CUTLASS, substantial templated C++, custom operators, vendored sources, or features Triton cannot express reliably.
4. Extend the existing JIT system rather than introducing a second AOT build route unless the task explicitly requires packaging changes.

For architecture-specific work, use `$gptqmodel-ampere-kernels` or `$gptqmodel-hopper-kernels` in addition to this skill.

## Implement from the boundary inward

1. Specify accepted shapes, strides, dtypes, layouts, quantization metadata, devices, and compute capabilities.
2. Add negative validation at the Python/C++ boundary. Reject unsupported inputs before launch with an actionable message.
3. Implement a minimal correct kernel, including tail handling and accumulation width, before tuning tiles or pipeline stages.
4. Launch under the correct CUDA device guard and current stream. Avoid hidden global-device assumptions and fixed device indices.
5. Register native sources in `_EXTENSION_SPECS` in `gptqmodel/extension.py` and use the public `extension.load`, `is_available`, `error`, `op`, or `namespace` helpers.
6. Put reusable wrapper logic in `gptqmodel/utils/` or the relevant quantized-linear module and sources in the matching `gptqmodel_ext/` subtree.
7. Gate specialized code by compute capability and preserve a tested portable or reference fallback.

Respect JIT fingerprints and cache behavior in `gptqmodel/utils/cpp.py`. Use `GPTQMODEL_KERNEL_REBUILD`, a supported per-extension rebuild variable, or `TORCH_CUDA_ARCH_LIST` only for isolated build/debug runs; do not force rebuilds during normal imports.

## Validate correctness

Cover:

- exact output shape, dtype, device, and finite values;
- comparison with the dense or dequantized reference using stated absolute and relative tolerances;
- smallest legal shape, representative decode and prefill shapes, non-divisible tails, and padding paths;
- every supported quantization layout and metadata branch;
- the intended architecture plus an explicit skip/fallback on unsupported hardware;
- repeated calls, multiple devices where available, non-default streams when relevant, and save/reload integration through the backend.

A compilation-only result must be labeled compilation-only. A passing kernel unit test does not establish model quality.

## Benchmark after correctness

Put reproducible benchmarks in `scripts/`. Warm up compilation and steady-state launches, synchronize correctly or use CUDA events, report distribution statistics rather than one timing, and compare against both the reference and the nearest production kernel. Include shapes, tokens/batch regime, dtype, quantization config, GPU properties, software stack, latency, throughput, and memory. Present the complete result table in ASCII.
