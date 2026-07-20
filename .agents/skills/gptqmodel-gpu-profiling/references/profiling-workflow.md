# GPU profiling workflow

## Capture contract

| Field | Record |
| --- | --- |
| Source | Repository revision, wrapper/backend, extension name, and build/JIT flags |
| Quantization | Method, format, bits, group size, symmetry/zero convention, activation ordering, and pack dtype |
| Workload | Model, input/output lengths, batch/concurrency, prefill/decode label, cache state, graph state, and seed |
| Hardware | Visible device, compute capability, SM count, memory, clocks/power state if controlled, and topology for multi-GPU |
| Software | Driver, CUDA toolkit/runtime, PyTorch, Triton/CUTLASS or backend versions |
| Schedule | Warmup steps, active steps, repeats, profiler activities/options, and synchronization method |
| Artifact | Absolute trace/report path, server arguments when applicable, and file size |

Use an explicit writable artifact directory. Do not upload a trace to an external viewer unless the user authorizes it; traces may contain model paths, operator inputs, annotations, or request metadata.

## Stage separation

Prefill and decode usually exercise different matrix shapes, cache traffic, launch rates, and graph behavior. Capture them independently by default. Avoid repeated prompts that unexpectedly turn prefill into a prefix-cache hit. If a mixed production trace is necessary, label it and avoid attributing a blended kernel share to either stage.

## Three-table report

Render complete ASCII tables rather than truncating to a few favorable rows.

1. **Dominant kernels**: stage, kernel family, source path, calls, total GPU time, share, mean, and shape/config notes.
2. **Overlap opportunities**: stage, gap or operation, time/share, dependency evidence, candidate overlap, and confidence.
3. **Fusion candidates**: producer, consumer, source locations, intermediate layout/dtype, existing local fused path if any, applicability, and evidence.

Keep thresholds explicit. Preserve a residual/other row when filtering so percentages remain interpretable.

## Interpretation checks

- High call count with low total time may be a launch-overhead issue, not a kernel-throughput issue.
- High kernel share does not imply it is improvable; compare achieved bandwidth/compute and the matched reference.
- CUDA graph traces can obscure Python source attribution; use a matched mapping trace rather than guessing.
- A synchronization event may encode a real dependency. Confirm producer-consumer ordering before calling it avoidable.
- Quantized kernels can move the bottleneck to dequantization, scale loads, reductions, epilogues, sampling, or CPU dispatch.
- Multi-rank averages hide stragglers. When communication matters, report rank spread and topology.
- Trace time is not final performance evidence. Confirm the conclusion with warmed, synchronized latency/throughput measurements.
