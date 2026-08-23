# QVQ quantization 4x roadmap

This is the execution plan for reaching **4x end-to-end quantization speedup** on the
Llama-3.2-1B W2 `qvq_v2b2_p32` YAQA workload. All work starts from `main` after PR #9
(`a19373b1`) and targets `main` through independently reviewable PRs.

## Goal and fixed benchmark

The acceptance workload and profiling recipe are the ones in
[`qvq_nsys_profile_llama32_1b.md`](qvq_nsys_profile_llama32_1b.md): 128 calibration
rows, 128 YAQA rows, two banks, and the same model, datasets, GPU, and environment.

| checkpoint | two-layer wall | speedup vs original |
|---|---:|---:|
| original baseline | 252.4 s | 1.00x |
| PR #8 fused family-grid | 181.6 s | 1.39x |
| 4x acceptance target | **<=63.1 s** | **>=4.00x** |

The full 16-layer run is the final confirmation, but the representative two-layer
capture is the iteration gate. Report medians from three clean runs after one warmup;
reject a result when the 95% confidence interval overlaps zero improvement. Always
record kernel time, wall time, dispatch counts, and output-quality results together.

## Why another local h-loop optimization cannot reach the goal

The original segment-grid kernel was about 56% of wall time. Even deleting it entirely
would cap end-to-end speedup at `1 / (1 - 0.56) = 2.27x`. PR #8 already changed the
dominant W2 path from 9.639 to 4.499 ms/call and reduced two-layer wall by 28%. PR #9
then demonstrated that eight-prefix ILP is neutral. The exact recurrence is now
FP32-issue-bound at 76% issue-slot utilization, and the tested staging, packing,
barrier, pruning, and ILP variants in
[`qvq_grid_kernel_opt_status.md`](qvq_grid_kernel_opt_status.md) should not be repeated.

For normalized original time, a 4x result requires:

```text
new_time / original_time <= 0.25
0.56 / segment_speedup + 0.44 / remainder_speedup <= 0.25
```

Examples: an 8x segment speedup still requires the remainder to improve by 2.44x; a
4x segment speedup requires the remainder to improve by 4x. The remaining work must
therefore remove repeated solves and optimize the whole quantization pipeline.

## Performance budget

The following is the target allocation from the measured post-PR-#8 two-layer run.
It is a budget, not a prediction; later phases inherit any shortfall from earlier ones.

| component | current | target | required gain |
|---|---:|---:|---:|
| fused family-grid | 57.5 s | 18.0 s | 3.2x |
| plain + tail Viterbi | 39.4 s | 12.0 s | 3.3x |
| YAQA feedback + update | 17.7 s | 7.0 s | 2.5x |
| other GPU work | 39.1 s | 16.0 s | 2.4x |
| non-kernel wall | 27.9 s | 10.0 s | 2.8x |
| **total** | **181.6 s** | **63.0 s** | **2.88x more** |

Because several proposed changes eliminate an outer-loop iteration, their gains apply
to multiple rows simultaneously. Isolated kernel tuning is not expected to satisfy
these component budgets.

## Execution order

### Phase 0 — reproducible scorecard

1. Add a command that runs the fixed two-layer workload three times and emits one JSON
   scorecard containing wall time, per-NVTX-range kernel time, call counts, dispatch
   counts, peak memory, and git/environment identity.
2. Add a comparison command that fails when wall time regresses by more than 2%, an
   expected optimized dispatch does not occur, or quality differs from the reference.
3. Capture a fresh `main` baseline. Do not use the historical 181.6 s number as the
   sole comparison if hardware or software differs.

Exit gate: run-to-run wall coefficient of variation <=2%, and attribution accounts for
>=95% of GPU kernel time. This phase changes instrumentation only.

### Phase 1 — eliminate repeated Viterbi solves at the Python/CUDA boundary

**Status: completed 2026-08-23; stop condition triggered, proceed to Phase 2.**

The three live family-grid call sites now report exact solve counts, sequence
counts, state-steps, provisional outputs produced/consumed, and exact-reuse
candidates through `QVQQuantizationTelemetry`. The audit found **zero reusable
calls**:

- Block-LDLQ visits each block/chunk once; its corrected sequence depends on
  errors updated by the preceding block.
- YAQA visits disjoint anti-diagonal coordinates once and mutates both feedback
  tensors after every commit.
- Sampled family selection is a one-shot solve over one sampled tensor.

This is a structural identity/version result, not a probabilistic hash result,
so a module-lifetime result cache would have a 0% hit rate. Per the predeclared
stop rule below, no general cache was added.

Telemetry also confirms that the provisional family solve produces 128 states
per sequence while only state 63 is consumed; provisional loss and all eight
selectors are discarded. A family midpoint-only fused specialization was
implemented and tested. A first four-segment version was correctly rejected by
the bit-exact test because the required midpoint lies on the globally optimal
128-step traceback. The corrected version retained the full forward recurrence
and only halved traceback storage. It passed weighted/unweighted exactness at
family batches 16/40/128 but measured just 4.915 -> 4.875 ms (1.008x) in a paired
microbenchmark, so it was reverted. The focused final gate is **170 passed** for
`-k 'v2_segment or grid'`, plus the new telemetry test.

Outcome: the phase's >=35% state-step/call and >=25% wall gates are not
attainable through exact call reuse. The explicit `<10%` reuse stop criterion
is met (0%), so Phase 1 is complete as a measured negative result and Phase 2
is the next execution phase.

This is the highest-leverage exact track. Instrument the two dominant call sites in
`gptqmodel/quantization/qvq.py` with stable hashes/IDs for sequences, codebooks,
constraints, step weights, and requested outputs. Measure:

- exact duplicate calls and re-selection calls whose inputs did not change;
- families or tiles that differ only in data already shared by the fused kernel;
- convergence: how often an accepted update leaves the selected path unchanged;
- calls whose full path is computed although only loss, endpoint, or a tail is used.

Then implement, in this order:

1. skip provably unchanged re-selection calls and reuse their exact outputs;
2. cache exact results within one module/anti-diagonal lifetime;
3. add endpoint/loss-only APIs where backpointers are not consumed;
4. batch independent pending solves into one persistent work queue, including plain and
   segment-tail jobs, so shared setup and codebook work are paid once.

Every skip must have an invariant proving identical inputs and outputs; hashes are for
telemetry, while production reuse uses version counters or exact tensor identity plus
mutation tracking. Cache lifetime must be bounded to one module.

Exit gate: >=35% reduction in total Viterbi state-steps or calls and >=25% two-layer
wall reduction, bit-exact. Stop caching work if measured reusable calls are <10%; move
directly to Phase 2 rather than building a general cache.

### Phase 2 — reduce exact work per solve

The existing fused kernel still performs the full 16-predecessor min-plus recurrence.
Only pursue transformations that reduce state-steps or arithmetic count:

1. derive exact lower bounds for predecessor groups and skip a group only when its
   bound cannot beat the current best; process candidates in a fixed order so ties and
   FP32 operation order remain stable;
2. specialize endpoint/loss-only solves to avoid backpointer writes and reconstruction;
3. fuse consecutive tail/plain jobs that share sequences or codebooks into the W2
   persistent kernel, with one work descriptor format and no intermediate norm pack;
4. test longer logical segments or cross-segment reconstruction with checkpointed
   frontiers, trading recomputation for fewer full recurrences only when state-step
   count actually falls.

Before CUDA implementation, replay real captured inputs on CPU/GPU and report the warp
maximum candidate count, not only the mean. The prior sorted-predecessor simulation
(mean 9.2/16 but warp maximum 14.95/16) is a rejection baseline.

Exit gate: candidate/state-step count >=2x lower on real inputs, target op >=1.8x faster
than PR #8, and >=15% additional wall reduction, bit-exact. Reject any design that only
raises occupancy or lowers bytes while executed FP32 instructions remain within 20% of
the current kernel.

### Phase 3 — optimize YAQA and preparation as a pipeline

Profile shapes and dependencies for `yaqa_feedback`, `yaqa_feedback_update_`, and
`stage.prepare_yaqa`. The current workload spends about 17.7 s in feedback/update and
additional time in ordinary FP32 GEMMs.

1. combine pointer setup, feedback epilogue, and update operations where dependencies
   permit;
2. replace thousands of small grouped SIMT GEMMs with shape-specialized batched kernels
   or larger grouped launches;
3. retain Sketch-B products across modules when their source and transformation version
   are unchanged;
4. overlap independent preparation for the next module only after profiling proves
   memory headroom and no serialization on the default stream.

Exit gate: combined YAQA/preparation wall >=2x faster, quantized weights bit-exact, and
peak device memory growth <=10%. Stop kernel fusion if GEMM compute remains >=80% of the
range; then reduce call count or matrix work at the algorithm level.

### Phase 4 — quality-gated algorithmic fast path

If the exact phases do not project to <=63.1 s, 4x requires changing the amount of
search performed. This phase must be a separate opt-in mode; it must not silently alter
the exact/default path.

Evaluate, in order, using captured real distributions:

1. adaptive beam width with an exact fallback when the score margin is small;
2. coarse-to-fine candidate screening followed by exact scoring of survivors;
3. early termination of YAQA re-selection after stable paths/loss for a configured
   number of iterations;
4. reduced Sketch-B or calibration work selected by a held-out quality sweep.

The acceptance suite must compare final model quality, not only local Viterbi loss:
quantized-weight deltas, calibration loss, held-out perplexity, and the repository's
generation/acceptance metrics. Predeclare tolerances in the experiment PR before
benchmarking. Any miss automatically dispatches the exact path for that tile/module.

Exit gate: cumulative two-layer wall <=63.1 s and all predeclared model-quality gates
pass. Also report fallback rate; a fast path that passes only synthetic tests is not
eligible.

### Phase 5 — full-model confirmation and hardening

Run the complete 16-layer profile and a second supported model shape. Confirm >=4x
against fresh `main`, no layer-specific slowdown, bounded memory, deterministic outputs
for the exact mode, and stable quality for the optional fast mode. Add dispatch and
quality regression tests, runtime disable flags, and a rollback path for every new
specialization.

## PR sequence and decision rules

Use one measurable hypothesis per PR:

1. scorecard and fresh baseline;
2. call-reuse telemetry (no behavior change);
3. exact skip/cache or endpoint-only API;
4. unified persistent Viterbi work queue;
5. exact work-reduction prototype, only if simulation clears its gate;
6. YAQA/preparation batching;
7. optional quality-gated fast path, if still required;
8. full-model validation and documentation.

Each performance PR includes before/after JSON, the exact command, hardware identity,
focused tests, two-layer wall, and an `nsys` attribution diff. Merge only if its own
predeclared gate passes. Revert failed prototypes and record them in the status document
so later work does not repeat them.

## Non-negotiable correctness gates

- Exact phases: all `-k 'v2_segment or grid'` tests plus affected Viterbi/YAQA tests are
  bit-exact; loss delta is zero and paths/backpointers match.
- End-to-end: quantized checkpoint structure and metadata match; no CPU fallback or
  reference CUDA dispatch appears unexpectedly.
- Approximate phase: opt-in only, exact fallback available, tolerances predeclared, and
  final model-quality evaluation required.
- Every phase: benchmark on an idle GPU, warm the identical JIT build, and preserve the
  profiler environment constraints documented in the status file.

The immediate next action is Phase 0 followed by the Phase 1 reuse/convergence telemetry.
Those measurements determine whether exact call elimination can close most of the
remaining 2.88x gap or whether the opt-in quality-gated track is unavoidable.
