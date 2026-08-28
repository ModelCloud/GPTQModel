# GPT-QModel agent guide

This file governs the whole repository. Keep changes narrowly scoped, preserve CPU and non-target GPU fallbacks, and never infer hardware capabilities from a fixed CUDA index.

## Repository map

- `gptqmodel/`: Python package, model adapters, quantization lifecycle, backend selection, and JIT wrappers.
- `gptqmodel_ext/`: CUDA/C++ extension sources.
- `tests/`: unit, model, kernel, serialization, and integration tests.
- `scripts/`: benchmarks and focused validation helpers; do not turn benchmarks into unit tests.
- `.agents/skills/`: task-specific workflows for quantization, backends, kernels, architectures, and model support.

## Route work to the local skills

- Quantization algorithms, calibration, formats, protocols, GPTQ, AWQ, QQQ, FP8, EXL3, ParoQuant, RTN, or
  bitsandbytes: use `$gptqmodel-quantization`.
- CPU-side tensor packing (`pack_block_cpu`, `pack_awq_cpu`, `pack_qqq_cpu`), AVX-512/AVX2 dispatch, dtype conversion,
  and thread-parallel packing: use `$gptqmodel-cpu-packing`.
- Pre/during/post quantization error analysis, risky module/weight/channel/embedding/LM-head discovery, severe quality
  regressions, pre-pack loss or scale spikes, pack/dequant/kernel isolation, or higher-bit/RTN/mixed-precision controls:
  also use `$gptqmodel-quantization-regressions`.
- Quantized linear implementations, backend selection, capability declarations, fallback, or availability checks:
  use `$gptqmodel-backends`.
- CUDA, C++, CUTLASS, Triton, JIT extensions, correctness debugging, or kernel benchmarks: use `$gptqmodel-cuda-kernels`.
- Multi-kernel fusion, cooperative or persistent mega-kernels, cross-phase scratch reuse, grid barriers, fused phase
  scheduling, or launch-count reduction: also use `$gptqmodel-mega-kernels`.
- MoE routing, expert dispatch, grouped GEMM, per-expert loops, and QKV/gate-up fusion inside MoE models:
  use `$gptqmodel-moe`.
- Fused inference QKV/gate-up, `model.fuse()`, `flash_attention_2`, and decode/prefill optimization:
  use `$gptqmodel-inference-fusion`.
- LazyTurtle/meta-device shell, batched safetensors loading, pinned-memory shard cache, and pre-quantize loading:
  use `$gptqmodel-lazy-turtle`.
- Checkpoint sharding (`reshard`), `ShardStrategy`, parallel shard writes, and save/load integration:
  use `$gptqmodel-sharding`.
- Checkpoint formats (MXFP4, GGUF, vLLM/sglang, EoRA adapters, format conversion): use `$gptqmodel-checkpoint-formats`.
- New model families, `module_tree`, MoE adapters, or `MODEL_MAP`: use `$gptqmodel-model-support`.
- Tokenizer initialization, normalization, special-token compatibility, prompt rendering, chat templates, or
  unexpectedly low generation scores: use `$gptqmodel-tokenizer-normalization`.
- Post-quantization evaluation (Evalution, GSM8K, lm-eval, perplexity, sanity generation): use `$gptqmodel-evaluation`.
- GPU correctness tests, performance benchmarks, user-specified GPU IDs, idle-device preflights, or long-running
  live result tables: use `$gptqmodel-gpu-testing`.
- Unit/branch coverage, realistic and adversarial quantization test design, dense-reference numerical gates, or
  nondeterminism-aware CUDA/Triton coverage claims: use `$gptq-coverage-review`; combine it with
  `$gptqmodel-gpu-testing` and `$gptqmodel-cuda-kernels` when real GPU kernels are in scope.
- GPU allocator CLI/client for leasing one or more GPUs: use `$gpu-allocator-cli`.
- Contiguous-memory layout regressions in quantization or kernel paths: use `$gptqmodel-contiguous-memory`.
- Torch-profiler traces, Nsight analysis, bottleneck attribution, launch gaps, overlap, or fusion opportunities:
  use `$gptqmodel-gpu-profiling`.
- Apple silicon, MLX custom Metal kernels, Xcode Metal System Trace/GPU Frame Capture, Metal counters, barriers,
  occupancy clues, or M-chip kernel tuning: use `$gptqmodel-metal-profiling`.
- Nsight Systems capture, `.nsys-rep` reports, CUDA launch gaps, memory copies, or NCCL timeline analysis:
  use `$perf-nsight-systems`.
- Nsight Compute kernel metrics, `.ncu-rep` reports, SOL/roofline, occupancy, memory hierarchy, or warp stalls:
  use `$perf-nsight-compute-analysis`.
- CUDA-event timing, warmed workload benchmarks, or NVTX instrumentation: use `$perf-workload-profiling`.
- Lightweight telemetry injection and interpretation (`module_load`, `module_move`, `torch_sync`, `disk_telemetry`):
  use `$gptqmodel-telemetry`.
- Reproducing a user-reported bug from an exact command, script, or log excerpt (error, warning, or discrepancy)
  and collecting telemetry before source inspection: use `$run-user-command-with-telemetry`.
- Upstream sync, ports, public release notes, or disclosure-boundary review: use `$gptqmodel-upstream`.
- Ampere or A100 tuning: also use `$gptqmodel-ampere-kernels`.
- Hopper or H100 tuning: also use `$gptqmodel-hopper-kernels`.

Read every selected `SKILL.md` completely before editing. Follow its linked references only when relevant to the task.

### Profiling and telemetry quick map

| What you need | Primary skill |
|---|---|
| Add timing/instrumentation to a region | `$gptqmodel-telemetry` (or `$perf-workload-profiling` for CUDA-event benchmarks) |
| Capture a trace or Chrome/Perfetto timeline | `$gptqmodel-gpu-profiling` |
| MLX/Metal trace, Xcode GPU capture, Apple GPU barriers or M-chip tuning | `$gptqmodel-metal-profiling` |
| System-level `nsys` timeline, launch gaps, NCCL | `$perf-nsight-systems` |
| Kernel-level `ncu` metrics, SOL%, roofline | `$perf-nsight-compute-analysis` |

## Tokenizer normalization ownership

GPT-QModel depends on [ModelCloud/Tokenicer](https://github.com/ModelCloud/Tokenicer). Put reusable corrective
tokenizer and chat-template normalization in Tokenicer, add relevant tests, increment its version, run its complete
test suite, then commit, push, and open a Tokenicer pull request. Do not leave model-loader patches in GPT-QModel
merely to avoid fixing the dependency.

Establish a non-quantized baseline before treating bad generation or evaluation scores as a quantization regression.
Compare exact rendered prompts and input IDs as well as aggregate scores.

## Working rules

1. Inspect the nearest implementation and test before adding a new abstraction.
2. For hardware work, record `nvidia-smi`, PyTorch/CUDA versions, compute capability, SM count, memory, dtype, shapes, and exact build flags. Probe at runtime; PCI bus order and device inventory can change.
3. Establish a dense BF16, FP16, or FP32 reference before optimizing quantized behavior. Compare numerical error as well as output shape and dtype.
4. Put correctness and regression checks in `tests/`; put timed performance experiments in `scripts/`. Warm up kernels, use CUDA events or synchronized timing, and report full ASCII tables with shapes, dtype, batch/token regime, latency, throughput, and speedup.
5. Gate architecture-specific code explicitly and retain a tested fallback. A successful compile is not evidence of runtime correctness or speed on hardware that is not present.
6. Keep Python compatible with Ruff's 119-character line limit. Prefer existing test helpers and focused `pytest` invocations before broader suites.
7. Log enough boundary metadata to reproduce kernel failures, but do not add noisy per-element logging to hot paths.
8. Separate **promotion** from **escalation** in model-quality experiments. A candidate that passes every locked
   metric may be promoted. A candidate with material primary-metric gains, no catastrophic failure, and only small
   minority guardrail regressions should advance to a larger disjoint test across more rows, realistic layers,
   seeds, and task-like data instead of being discarded from one small synthetic gate. Escalation is not acceptance:
   keep the baseline artifact and do not enable a default until the expanded confirmation passes.
9. Never use fake or synthetic tensors as evidence for quantization-quality decisions. Synthetic fixtures are only
   for algebra, kernel, serialization, and corner-case correctness tests. Start every accuracy experiment with real
   weights and real tokenized activations from a tractable model such as Llama 3.2 1B, using limited but disjoint
   calibration and evaluation data; expand rows, layers, seeds, and tasks when the result merits confirmation.
10. Do not reduce model-quality decisions to binary all-column pass/fail when changes are close to measurement noise.
    Record absolute and relative effect sizes, paired uncertainty or bootstrap intervals when possible, and the
    practical importance of each metric. A small guardrail regression can be acceptable when it is within noise and
    substantially outweighed by reproducible propagated-loss or task gains; material regressions remain blockers.
    Borderline cases must be escalated on larger disjoint real-model evidence rather than silently accepted or rejected.
    Classify each metric explicitly as clear positive, noise-consistent, or clear negative. One noisy failure among
    otherwise meaningful gains is an escalation signal, not an automatic experiment failure; do not count columns.
    Local reconstruction MSE/KL is a proxy, not the end goal: it may regress and should be tolerated when disjoint,
    propagated final-logit KL and Top-K agreement improve materially, remain finite, and pass the predeclared
    uncertainty/guardrail policy. Do not trade demonstrated final-model recovery for a lower local error solely to
    make the proxy look better.

## Typical checks

Use the smallest applicable set, then expand when risk warrants it:

```bash
ruff check <changed paths>
pytest -q <focused test paths>
git diff --check
```

CUDA tests must state whether they ran, skipped, or only compiled. Model-affecting changes should include a small quantize/save/load/inference path and, when practical, an evaluation comparison against the dense baseline.
