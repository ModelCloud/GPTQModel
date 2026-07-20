# GPT-QModel agent guide

This file governs the whole repository. Keep changes narrowly scoped, preserve CPU and non-target GPU fallbacks, and never infer hardware capabilities from a fixed CUDA index.

## Repository map

- `gptqmodel/`: Python package, model adapters, quantization lifecycle, backend selection, and JIT wrappers.
- `gptqmodel_ext/`: CUDA/C++ extension sources.
- `tests/`: unit, model, kernel, serialization, and integration tests.
- `scripts/`: benchmarks and focused validation helpers; do not turn benchmarks into unit tests.
- `.agents/skills/`: task-specific workflows for quantization, backends, kernels, architectures, and model support.

## Route work to the local skills

- Quantization algorithms, formats, packing, protocols, calibration, GPTQ, AWQ, QQQ, FP8, GGUF, EXL3, ParoQuant, RTN, or bitsandbytes: use `$gptqmodel-quantization`.
- Quantized linear implementations, backend selection, capability declarations, fallback, or availability checks: use `$gptqmodel-backends`.
- CUDA, C++, CUTLASS, Triton, JIT extensions, correctness debugging, or kernel benchmarks: use `$gptqmodel-cuda-kernels`.
- Torch-profiler traces, Nsight analysis, bottleneck attribution, launch gaps, overlap, or fusion opportunities: use `$gptqmodel-gpu-profiling`.
- Ampere or A100 tuning: also use `$gptqmodel-ampere-kernels`.
- Hopper or H100 tuning: also use `$gptqmodel-hopper-kernels`.
- New model families, `module_tree`, MoE adapters, or `MODEL_MAP`: use `$gptqmodel-model-support`.

Read every selected `SKILL.md` completely before editing. Follow its linked references only when relevant to the task.

## Working rules

1. Inspect the nearest implementation and test before adding a new abstraction.
2. For hardware work, record `nvidia-smi`, PyTorch/CUDA versions, compute capability, SM count, memory, dtype, shapes, and exact build flags. Probe at runtime; PCI bus order and device inventory can change.
3. Establish a dense BF16, FP16, or FP32 reference before optimizing quantized behavior. Compare numerical error as well as output shape and dtype.
4. Put correctness and regression checks in `tests/`; put timed performance experiments in `scripts/`. Warm up kernels, use CUDA events or synchronized timing, and report full ASCII tables with shapes, dtype, batch/token regime, latency, throughput, and speedup.
5. Gate architecture-specific code explicitly and retain a tested fallback. A successful compile is not evidence of runtime correctness or speed on hardware that is not present.
6. Keep Python compatible with Ruff's 119-character line limit. Prefer existing test helpers and focused `pytest` invocations before broader suites.
7. Log enough boundary metadata to reproduce kernel failures, but do not add noisy per-element logging to hot paths.

## Typical checks

Use the smallest applicable set, then expand when risk warrants it:

```bash
ruff check <changed paths>
pytest -q <focused test paths>
git diff --check
```

CUDA tests must state whether they ran, skipped, or only compiled. Model-affecting changes should include a small quantize/save/load/inference path and, when practical, an evaluation comparison against the dense baseline.
