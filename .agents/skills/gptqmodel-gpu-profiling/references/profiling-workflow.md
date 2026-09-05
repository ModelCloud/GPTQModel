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

## SASS-guided math and movement reduction

Use this workflow after Nsight Systems has identified a custom CUDA kernel and a reproducible microbenchmark exists. It is especially useful for quantized decoders whose C++ expressions can compile into unexpectedly large mask, shift, address, shuffle, or local-memory streams.

1. Capture one representative launch with Nsight Compute. Prefer focused sections such as `SpeedOfLight`, `ComputeWorkloadAnalysis`, `MemoryWorkloadAnalysis`, `SchedulerStats`, `WarpStateStats`, `InstructionStats`, `LaunchStats`, and `Occupancy`; use `--set full` only when the extra replay cost is justified.
2. Export the source-correlated instruction page with `ncu --import <report> --page source --csv`. Preserve the `.ncu-rep`, exact capture command, source revision, kernel demangled name, and workload shape.
3. Aggregate executed SASS opcodes and inspect the hot basic blocks. Use `cuobjdump --dump-resource-usage` or `nvdisasm` as a corroborating resource/disassembly view, not as a substitute for executed NCU counts.
4. Trace each dominant instruction family back to its mathematical or data-movement role. Look specifically for:
   - repeated `LOP3`, `SHF`, `PRMT`, or `IMAD` chains implementing equivalent bit algebra;
   - per-consumer address calculations or loads that could be produced once and routed cheaply;
   - shared/local store-load round trips for values already resident in registers;
   - barriers, waits, and pipeline state updates whose work is too short to amortize them;
   - widened tiles or producer/consumer ownership that can reuse activation, scale, or compressed-weight fetches.
5. Prove any algebraic rewrite over the complete supported input domain or with an exact reference test. Source-level operation counts are hypotheses: verify the generated SASS actually removed instructions and did not add register moves, permutations, spills, or longer dependencies elsewhere.
6. Compare a matched before/after NCU capture. Report total executed instructions, relevant opcode deltas, registers, local memory, shared conflicts, occupancy, scheduler eligibility, dominant stalls, and achieved memory/tensor-pipe throughput.
7. Gate production acceptance separately with warmed CUDA-event timing and numerical/model-quality tests. An exact instruction reduction is useful evidence even when sub-microsecond event timing is noisy; record both facts without presenting profiler duration as application speedup.

Do not infer a fusion opportunity solely from adjacent source expressions. The proposed ownership and routing must cost less than the work removed; fixed warp transposes and gathers can dominate an otherwise cheaper decoder.

### Per-commit kernel audit invariant

Repeat this workflow after every committed phase that can change generated GPU
instructions, not only after large source rewrites. Template selection, constants,
launch geometry, compiler flags, or a nearby helper can change SASS and can
reintroduce address, mask, shift, conversion, permutation, or synchronization
work previously removed. Always compare the exact new committed binary with the
preceding committed binary at a matched workload and revisit algebraic folding,
common-subexpression elimination, decode/address deduplication, and value reuse.
If the target GPU or executed-instruction profiler is unavailable, report
compilation-only status and defer performance promotion.
