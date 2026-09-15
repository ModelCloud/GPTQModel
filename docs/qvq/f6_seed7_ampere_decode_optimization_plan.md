# F6 Seed-7 Ampere decode optimization plan

This document turns the F6 Seed-7 Ampere decode profile into a staged
investigation plan. It is a tracking plan, not an accepted optimization
record. Each candidate remains a hypothesis until correctness, generated-code
audit, and unprofiled end-to-end decode timing pass on the target workload.

## Scope

Baseline profile: `docs/qvq/f6_seed7_ampere_decode_profile.md`.

Profiled configuration:

- QvQ source: `03c326d1314f707016f5159c9ccfa81dabe78f9a`
- ZML-Ultra source: `ffac15de3b2bf3d39a038edacf39570dd2913221`
- Model: F6 Seed-7 P32/R8 QVQ Llama 3.2 1B snapshot
- Runtime: native ZML Fast API ABI v1 with QVQ P32 CUDA kernels
- GPU: runtime-probed Ampere SM80, 124 SMs, 96 GiB HBM in the captured run
- Context: 131072 tokens
- KV capacity: 8192 pages
- Primary decode target: batch 1 maximum context
- Batch sweep target: 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, skipping OOM only

## Evidence status

Use these labels in all follow-up PRs and benchmark notes:

- **Measured evidence:** a tool-captured result with command, revision, binary
  hash, workload, and artifact path.
- **Inference:** a bounded interpretation of measured evidence.
- **Unresolved ownership:** a source-responsibility question that must be
  answered before patching.
- **Proposed experiment:** a candidate intervention not yet accepted.
- **Accepted optimization:** a candidate that passed local correctness, graph
  safety where applicable, generated-code audit, and warmed end-to-end timing.
- **Rejected/inconclusive:** a candidate that failed gates or lacks enough
  evidence; keep the record so it is not rediscovered as new.

Do not promote any candidate because static SASS is smaller, executed
instruction count drops, occupancy rises, bandwidth percentage changes, or a
replay-profile duration improves. Those metrics explain a timing change; they
are not the timing gate.

## Current measured bottlenecks

Per decode step, the formal Nsight Systems run measured:

| Rank | Region | Measured contribution | Status |
|---:|---|---:|---|
| 1 | Unified attention | 14.776 ms | Underfilled/dependency-limited inference |
| 2 | Dominant P32 M1 projection | 8.495 ms | Eight-block grid, low waves/SM |
| 3 | Per-layer K/V slice materialization | 7.932 ms | 32 copies x 256 MiB = 8 GiB/step |
| 4 | Rank-8 projection | 0.718 ms | Highly occupied, load/dependency pressure |
| 5 | Other small GEMM/fusion work | ~0.7 ms | Lower priority until aggregated |

The 7.932 ms K/V copy total is a non-overlap-adjusted upper bound on removable
work, not a promised decode speedup. The captured attention DRAM throughput
does not show HBM saturation.

## Global gates for every implementation phase

### Correctness and numerical gates

For QVQ inference kernels and any fused equivalent operation, compare the
candidate against the canonical reference on identical packed weights, inputs,
dtype, shapes, stream, and initial state.

- Finite outputs required.
- Mean absolute drift `<= 2e-3` per case.
- Maximum absolute drift `<= 0.046875` per case.
- Both limits apply independently; relative L2, top-token agreement, and
  pooled averages cannot replace either limit.
- For the F6 Seed-7 P32 campaign, the user-approved localized mean-error limit
  is `3e-3`; finite-output and max-error gates remain unchanged.
- Preserve packed state, bank semantics, tail handling, accumulation dtype, and
  reduction order unless a separately reviewed numerical change is proposed.

For ZML/XLA/StableHLO/fusion changes, also compare against a matched unfused
Transformer eager oracle at the earliest useful stage boundaries: K/V cache
update, attention output, QVQ linear output, residual, final logits, and
selected token where available.

### Human review for high-speed accuracy exceptions

An accuracy-gate failure normally blocks promotion, but it must not silently
discard a verified large speedup. Compute speed gain as
`100 * (baseline_latency / candidate_latency - 1)` on matched measurements.

- Above 25% (`>1.25x`), present a scoped exception option to the human reviewer
  before abandoning the candidate.
- Above 100% (`>2x`), escalate promptly after timing and gate failure are
  verified; do not wait for the rest of an experiment wave.
- Above 500% (`>6x`), escalate immediately as a priority review item.

The review packet must state the measured scope, shapes, hardware, baseline,
timing uncertainty, failing mean/max errors and thresholds, whether the
baseline also fails, candidate-versus-baseline error, held-out model-quality
evidence, missing checks, proposed exception scope, and accurate fallback.
Kernel-only or synthetic gains must not be presented as model speedups.

Escalation is not acceptance. Continue authorized profiling and investigation,
retain the original gate decision and failure evidence, and keep the accurate
implementation as the production default until the human reviewer explicitly
approves a narrowly stated exception. Approval for one gate does not waive any
other correctness, graph, fallback, or end-to-end requirement.

### Generated-code audit gate

Every phase that changes generated GPU instructions must:

1. capture the exact committed binary or JIT fingerprint at a matched workload;
2. run targeted Nsight Compute sections on the affected kernels;
3. export source-correlated SASS;
4. compare executed instructions and dominant opcode families;
5. compare registers, spills, shared memory, bank conflicts, occupancy,
   eligible warps, issue activity, stalls, DRAM/L2/L1 throughput, and launch
   geometry;
6. re-run localized numerical correctness;
7. re-run warmed CUDA-event or synchronized application timing;
8. re-run full decode plus relevant prefill benchmarks.

Static SASS size is not dynamic work. Nsight Compute replay duration is not
application latency.

### Profiling methodology gate

Use:

- unprofiled warmed benchmark for promotion timing;
- Nsight Systems for launch order, copies, gaps, synchronization, overlap, and
  graph behavior;
- targeted Nsight Compute for already-identified kernels;
- source-correlated SASS for CUDA/Triton micro-analysis;
- mapping traces only when necessary to recover operator attribution;
- formal production-configuration traces for final conclusions.

Each run records source revisions, model/artifact hashes, GPU properties,
software stack, shape, batch, context, cache state, warmup/measured steps,
launch geometry, graph mode, cache/replay settings, exact commands, and
artifact paths.

### Build and compiler resource gate

All future QvQ/ZML/CUDA/XLA builds must:

- use cgroup `memory.high` and `memory.max` before `/proc/meminfo`;
- reserve at least 25% of effective memory;
- cap local compilation at no more than half of the effective CPU quota;
- lower the cap further when compiler memory pressure requires it;
- use repository-local `./bazel.sh` for ZML Bazel work;
- treat Bazel, NVCC, Ninja, CMake, Zig, LLVM/MLIR, and XLA workers as one
  shared resource budget, not independent half-core pools.

### Graph and runtime gate

Any future XLA, StableHLO, ZML, attention ABI, fusion, cache representation, or
QvQ runtime integration change must:

- prepare compilation, tuning, allocation, descriptors, and metadata outside
  graph capture;
- preserve buffer ownership and lifetimes through graph replay;
- preserve caller stream and ordering;
- test repeated replay with changed input contents at stable addresses;
- test allocator pressure, non-default streams, invalidation, cleanup, and
  unsupported-layout fallbacks;
- test the actual external runtime path, not only a Python CUDA graph;
- preserve donation and aliasing semantics.

### Architecture gate

Probe the live device at runtime. SM80-specific candidates must be explicitly
gated. Do not assume CUDA index, PCI order, SM count, product string, or HBM
size. Do not use Hopper-only TMA, WGMMA, thread-block clusters, or distributed
shared memory in the Ampere path. Preserve CPU, non-Ampere, and
unsupported-layout fallbacks.

## Cross-repository PR ownership

The tracking plan lives in QvQ. Execution should split by source ownership:

| Area | Primary PR home | Parallel PR need |
|---|---|---|
| P32 kernel SASS, split geometry, rank-8 CUDA | QvQ | ZML PR only if launch-plan ABI or tuning surface changes |
| QVQ P32 external tuning ABI/header/tests | QvQ | ZML bridge PR to consume the new explicit config |
| K/V cache slice materialization | ZML-Ultra | QvQ PR only if a QVQ ABI or benchmark artifact must change |
| Unified attention Triton kernel/config | ZML-Ultra | QvQ PR only for cross-stack benchmark records |
| XLA/HLO aliasing, donation, graph capture | ZML-Ultra | QvQ PR for kernel-side fallback or QVQ docs only |
| Native Fast ABI harness and benchmark publication | Usually QvQ docs/artifacts | ZML PR if the ABI/timing contract changes |
| Persistent or megakernel scheduling | Split by runtime boundary | Likely coordinated QvQ and ZML PRs |

Future implementation should avoid one giant cross-repo patch unless a single
behavior cannot be validated independently. Use paired PRs when a ZML caller and
QvQ kernel ABI must land together.

## Micro-area registry

| Required micro-area | Phase |
|---|---|
| K/V cache slice materialization and copy elimination | 1 |
| Unified attention grid/CTA underfill | 2 |
| Attention memory access, pointer arithmetic, registers, dependencies | 3 |
| P32 M1 grid/split parallelism | 4 |
| P32 state extraction and funnel-shift work | 5 |
| PGC16 mixing and bank-mask computation | 5 |
| Codebook/level lookup behavior | 5 |
| P32 shared-memory staging and load layout | 6 |
| P32 activation/input tile reuse | 6 |
| P32 MMA fragment loading and output accumulation | 6 |
| Split-output workspace and reduction overhead | 4 |
| Rank-8 projection load/dependency pressure | 7 |
| Rank-8 projection and epilogue fusion | 7 |
| Small XLA launch/parameter-update overhead | 8 |
| Persistent/megakernel or task-graph work | 9 |
| Graph capture, buffer ownership, stream ordering, fallback | 1, 8, 9 |
| Build and compiler resource safety | 0 |
| End-to-end decode impact at batch 1 and requested batch range | 10 |
| Prefill regression guardrails | 0, 10 |
| Cross-architecture fallback and runtime capability probing | 10 |

## Phase 0: Baseline recertification and harness hardening

**Hypothesis:** Before optimizing, the existing profile and benchmark harness
need a clean, current baseline so follow-up PRs cannot compare against stale or
ambiguous timing.

**Source ownership:** QvQ for benchmark/profiling artifacts and P32 harnesses;
ZML-Ultra for native Fast ABI timing semantics if the ABI call does not
synchronize or expose CUDA-event timing.

**Exact micro-area:** GPU idle gate, UUID filtering, synchronized decode timing,
trace validation failure handling, model/source/binary hashes, cgroup-aware
build limits, and prefill-regression guardrail.

**Required baseline:** Re-run the merged decode profiler and full max-context
batch sweep from fresh QvQ `origin/main` and ZML-Ultra default branch. Record
whether profiler-report review nits have been addressed in the active harness.

**Experiment/intervention:**

- Make the profiling harness fail closed on invalid idle-sample counts.
- Filter idle checks and foreign processes by the leased GPU UUID.
- Confirm native ABI timing synchronizes, or add CUDA-event timing in the
  benchmark path.
- Treat trace-validation failure as blocking for trace-dependent conclusions.
- Add explicit "profiled timing" versus "promotion timing" fields.

**Profiling evidence to collect:** Nsight Systems copy/kernel breakdown,
targeted Nsight Compute for attention/P32/rank-8, and unprofiled warmed
decode/prefill tables.

**Correctness checks:** Graph/eager output match, QVQ P32 activation evidence,
unique prompts, finite outputs, exact packed-model provenance, and P32 localized
kernel reference checks where the harness exercises kernels directly.

**End-to-end benchmark checks:** Batch 1 decode focus plus the requested batch
range at context 131072; skip only OOM and retain OOM diagnostics.

**Expected decision:** Freeze this as the comparison baseline for all later
phases, or stop optimization until timing/correctness ambiguity is resolved.

**Promotion criteria:** The baseline is reproducible, source/hash complete,
uses UUID-scoped idle gates, and has synchronized or CUDA-event decode timing.

**Rollback/fallback criteria:** If the harness changes timing materially,
retain both old and new records but use only the hardened harness for candidate
promotion.

**Dependencies:** None. This gates every later phase.

## Phase 1: K/V cache slice materialization elimination

**Hypothesis:** Passing full K/V cache storage plus a layer index or byte/element
offset into attention can avoid 32 materialized 256 MiB per-layer copies in the
decode graph.

**Source ownership:** ZML-Ultra primary: `examples/llm/models/llama/model.zig`,
`zml/attention/triton_attention.zig`, `zml/attention/triton_kernels/unified_attention.zig`,
and XLA/StableHLO lowering or buffer-aliasing policy. QvQ owns only benchmark
records and any external kernel ABI interactions.

**Exact micro-area:** `PagedKvCache.updateAt`, `PagedKvCache.atLayer`, HLO
`wrapped_slice`/`kCopy` thunks, attention operands `key_cache_ptr` and
`value_cache_ptr`, donation/aliasing, and command-buffer replay.

**Required baseline:** Phase 0 baseline with 32 K/V copies, 8 GiB copied
payload, per-copy durations, HLO copy annotations, thunk list, and
buffer-assignment evidence.

**Experiment/intervention:**

1. Determine ownership: is materialization caused by ZML tensor slicing,
   StableHLO semantics, XLA buffer assignment, FFI custom-call requirements, or
   lack of alias/sub-buffer representation?
2. Prototype one representation at a time:
   - full-cache pointer plus `layer_index`;
   - full-cache pointer plus precomputed byte/element offset;
   - sub-buffer/alias view that does not allocate or copy;
   - update-and-attend fusion only if view semantics cannot be made safe.
3. Keep all preparation and descriptor computation outside graph capture.

**Profiling evidence to collect:** Nsight Systems proof that 256 MiB D2D copies
are gone or reduced; HLO/thunk/buffer-assignment before/after; attention kernel
argument and address arithmetic changes; graph replay timeline.

**Correctness checks:** Eager versus graph decode, repeated replay with changed
tokens at stable addresses, cache update correctness per layer, non-default
stream, allocator pressure, unsupported-layout fallback, and final logits/token
diagnostics. The attention output must match the unfused reference within the
declared model/runtime tolerance; no QVQ arithmetic gate changes are authorized.

**End-to-end benchmark checks:** Batch 1 max-context decode p50/mean/p95 and
throughput; full requested batch range; prefill guardrail because cache
representation can also affect prefill graph construction.

**Expected decision:** If copies vanish and decode improves without graph or
aliasing regressions, promote the narrow representation. If copies remain,
identify the exact compiler/runtime boundary and file a smaller ZML/XLA task.

**Promotion criteria:** No 256 MiB per-layer materialization in the formal
trace; equal outputs; graph-safe external runtime replay; warmed decode gain
outside noise; prefill not materially regressed.

**Rollback/fallback criteria:** Fall back to existing materialized layer cache
for unsupported layouts, non-SM80 devices, unsafe aliasing, failed graph capture,
or no end-to-end gain.

**Dependencies:** Phase 0.

## Phase 2: Unified attention CTA/grid underfill

**Hypothesis:** The attention kernel is the largest measured GPU contributor
and is underfilled at batch 1/context 131072; segmentation, CTA shape, or
head/query scheduling changes can improve dependency hiding without changing
attention semantics.

**Source ownership:** ZML-Ultra primary: unified attention config selection and
Triton kernel implementation. QvQ only records cross-stack benchmark results.

**Exact micro-area:** `KernelUnifiedAttention3dPtr.Config`, grid
`total_q_blocks x num_kv_heads x num_segments_per_seq`, `block_q`, `block_m`,
`tile_size`, segment count, register pressure, and reduce-segment coupling.

**Required baseline:** Phase 0 or Phase 1 attention profile: launch geometry
`1x8x16`, 64 threads, 124 registers/thread, 0.13 waves/SM, low eligible-warps
and issue activity.

**Experiment/intervention:**

- Sweep bounded segment counts and CTA shapes for the batch-1 long-context
  case.
- Evaluate whether more segments reduce attention critical path after the
  segment-reduction cost.
- Test layout-preserving query/head remapping if it increases CTAs without
  extra reduction overhead.
- Keep the current configuration as fallback.

**Profiling evidence to collect:** Nsight Compute LaunchStats, Occupancy,
SchedulerStats, WarpStateStats, SpeedOfLight, memory workload, segment-reducer
profile, and Nsight Systems whole-step attribution.

**Correctness checks:** Attention output against unfused reference over causal
mask, full context, ragged/short contexts, batch >1, tail dims if any, graph and
eager paths.

**End-to-end benchmark checks:** Decode batch 1 and requested batch range after
Phase 1 state is fixed or explicitly held constant; measure attention-only and
full decode to catch reducer overhead.

**Expected decision:** Retain only a shape/config policy that improves complete
decode, not merely attention replay duration.

**Promotion criteria:** Warmed full-decode gain, no new copies, equal outputs,
no unacceptable register/spill/shared-memory regression, and no prefill
regression for shared attention config paths.

**Rollback/fallback criteria:** Revert to existing config for small context,
short batches, unsupported layouts, graph failures, or when segment reduction
erases the attention gain.

**Dependencies:** Phase 0. Prefer after Phase 1 so removed copies do not mask
attention changes, but an isolated ZML branch can explore in parallel.

## Phase 3: Unified attention address, memory, and dependency micro-work

**Hypothesis:** After grid sizing, attention remains limited by register
pressure, pointer arithmetic, block-table loads, K/V address generation, and
long-scoreboard dependencies; simplifying those can reduce the per-layer
attention critical path.

**Source ownership:** ZML-Ultra.

**Exact micro-area:** `findSeqIdx`, `query_start_len` loads, block-table
offsets, `physical_block_idx`, `k_offset`, `v_offset`, cache modifiers,
dequant path, causal/sliding-window masks, Q/K/V load layout, and register live
ranges.

**Required baseline:** Best available Phase 1/2 attention candidate plus
source-correlated SASS and NCU metrics for the same workload.

**Experiment/intervention:**

- Hoist or precompute invariant strides/offsets where graph-safe.
- Reduce repeated add/mul/rem/div chains in K/V pointer math.
- Check whether layer-offset integration from Phase 1 increases or decreases
  address arithmetic.
- Evaluate block-table access locality and cache modifiers.
- Sweep register-pressure reductions only if SASS shows actual live-range or
  spill pressure.

**Profiling evidence to collect:** Executed instruction families, source
hotspots, registers, local memory/spills, eligible warps, long/short scoreboard,
LD/ST sectors, L1/L2 hit rates, and whole-step timeline.

**Correctness checks:** Same as Phase 2, plus stress page boundaries,
non-contiguous slot mappings, maximum context, short context, and repeated
replay with changed K/V contents.

**End-to-end benchmark checks:** Full decode batch 1 and selected sensitivity
batches 2/4/8/16; full requested sweep before promotion.

**Expected decision:** Keep exact value/address reuse that improves application
timing; reject changes that only move stalls or increase register pressure.

**Promotion criteria:** Matched SASS shows the intended dynamic work reduction
or dependency improvement, and warmed full decode improves outside noise.

**Rollback/fallback criteria:** Revert if register count, spills, replay
stalls, graph compatibility, or prefill attention behavior regress.

**Dependencies:** Phase 0; should follow whichever Phase 1/2 variant becomes
the active attention baseline.

## Phase 4: P32 M1 grid, split parallelism, and reduction ownership

**Hypothesis:** The dominant P32 M1 kernel launches only eight CTAs at batch 1,
leaving most SM80 SMs idle. Split-K, grouped launch policy, row grouping, or
external tuning can expose more parallel work, but only if reduction/workspace
cost is lower than the gained product-kernel parallelism.

**Source ownership:** QvQ primary:
`gptqmodel_ext/qvq/p32/qvq_p32_cuda.cu`, public P32 ABI/header/tests, and
QvQ launch-plan policy. ZML-Ultra owns bridge-side enumeration, autotune cache,
and graph-safe launch-plan consumption if external tuning changes.

**Exact micro-area:** `p32_window_ampere_m1_kernel_body`, scalar M1 dispatch,
`split_count`, `StaticSplitCount`, `row_groups`, grouped scalar/block paths,
`finish_grouped_reduction`, partial-output mode, workspace layout, and ZML
external tuning ABI.

**Required baseline:** Dominant P32 M1 profile: 8 CTAs, 128 threads, 64
registers/thread, 6.848 KiB shared memory, 0.01 waves/SM, 8.495 ms/step over
16 launches, native reduction behavior.

**Experiment/intervention:**

- Enumerate existing compiled specializations before adding new kernels:
  scalar grouped path, hoisted/grouped PGC path, split counts, 64/128/256
  threads, stages 1-4, static-N choices, row groups, native versus partials.
- Add only reachable, ABI-visible candidates.
- Benchmark split counts per M/K/N group and include the reducer in the timed
  scope.
- If ZML can own reduction/fusion, test partial-output mode with graph-visible
  ordered FP32 summation.

**Profiling evidence to collect:** Product and reduction timings separately and
together; workspace bytes; launch count; CTAs/waves; SASS instruction mix;
registers/spills/shared memory; reducer memory traffic; Nsight Systems
placement relative to neighboring ops.

**Correctness checks:** P32 reference on identical packed weights/inputs/state,
mean/max gates, finite outputs, split-order classification, tails, repeated
calls, non-default stream, graph launch-plan path, and unsupported-shape
fallback. Use the F6 Seed-7 `3e-3` mean gate only for the scoped campaign.

**End-to-end benchmark checks:** Full decode batch 1 at maximum context, then
requested batch range. Include prefill guardrail because shared P32 dispatch
policy can affect prefill GEMV/GEMM shapes.

**Expected decision:** Either select an existing specialization/tuning policy,
add a small ABI-visible candidate set, or reject split expansion because
reduction/workspace costs dominate.

**Promotion criteria:** Product-plus-reduction improves full decode outside
noise, localized P32 gates pass, generated-code audit passes, and ZML can freeze
the selected policy outside graph capture.

**Rollback/fallback criteria:** Existing native policy remains default for
unsupported shapes, failed accuracy, graph incompatibility, workspace pressure,
or no full-decode gain.

**Dependencies:** Phase 0. Can proceed in QvQ parallel to ZML Phase 1/2 if the
external ABI surface is not changed until a winner is credible.

## Phase 5: P32 state extraction, PGC16 mixing, bank masks, and level lookups

**Hypothesis:** The P32 M1 SASS hotspots show repeated bit extraction,
funnel-shift, PGC16 mix, bank-mask, address, and level-lookup work. Exact reuse
or rate-specific specialization may reduce dynamic instructions without
changing decoded values.

**Source ownership:** QvQ.

**Exact micro-area:** `window_state_pair64`, `decode_pair_bits`,
`decode_state_bits`, `decode_state_pair_bits`, `pgc16_mix`,
`alternate_bank_mask`, `selected_bank_mask`, `UseSharedLevels`,
`accumulate_scalar_grouped_pgc`, and `accumulate_scalar_hoisted_pgc`.

**Required baseline:** Best Phase 4 P32 candidate or current dominant M1 SASS:
LOP3/SHF/IMAD/LDS/HADD2/FFMA opcode mix and source hotspots.

**Experiment/intervention:**

- Reuse extracted states across `k`/`k+distance` pairs only when integer
  equivalence is proven for each transition width.
- Combine PGC16 operations only when masks, signedness, and 16-bit lane
  semantics are unchanged.
- Evaluate shared-level staging versus read-only/global lookup per rate and
  shape; do not assume LUTs win.
- Check existing hoisted/grouped helpers before adding new variants.
- Keep direct arithmetic as the baseline for any LUT-like experiment.

**Profiling evidence to collect:** Source-correlated SASS by rate and shape,
executed SHF/LOP3/IMAD/LDS/LDG counts, registers, spills, shared conflicts,
L1/L2 hit rates, eligible-warps, stalls, and warmed product timing.

**Correctness checks:** Integer decode equivalence tests, localized output
drift gates for real packed weights and real activations, adversarial states,
bank-alt combinations, transition bits, tails, repeated calls, and streams.

**End-to-end benchmark checks:** Decode batch 1 and representative batches;
full requested range only for retained candidates. Prefill guardrail if the
same helper is used in prefill shapes.

**Expected decision:** Promote only exact reuse/specialization that survives
SASS and full-decode timing; keep rejected variants documented by rate/shape.

**Promotion criteria:** Accuracy gates pass; source-correlated SASS confirms
the intended dynamic work reduction; no register/spill/shared conflict
regression; full decode improves.

**Rollback/fallback criteria:** Per-rate fallback to the prior helper if a
candidate slows a rate, changes values, or increases dependency stalls.

**Dependencies:** Phase 4 winner preferred; can run small isolated experiments
against the current M1 baseline.

## Phase 6: P32 staging, input reuse, MMA fragments, and accumulation

**Hypothesis:** After decode/address cleanup, remaining P32 latency may come
from shared-memory staging, activation tile reuse, MMA fragment loading, and
output accumulation order. Pipeline or layout changes can help only if they do
not reduce occupancy or alter numerical contracts.

**Source ownership:** QvQ.

**Exact micro-area:** `input_tile`, `packed_words`, `packed_bank_ids`,
`shared_levels`, `__pipeline_memcpy_async`, `copy_async_cg_16`,
`load_mma_fragment_a`, `load_mma_fragment_a_upper`, `mma_m16n8k16`, scalar
FFMA accumulation, `store_output_pair`, stage counts, and shared-memory layout.

**Required baseline:** Retained Phase 4/5 P32 SASS and timing, including shared
memory, registers, conflicts, and pipeline-stage count.

**Experiment/intervention:**

- Sweep `StageKTiles` and thread count within existing specialization limits.
- Check input tile reuse across output rows and adjacent tiles.
- Revisit scalar versus MMA path selection for M1-like decode shapes only if
  existing paths cannot express the candidate.
- Inspect whether `cp.async` staging is on the critical path or merely moving
  already-hidden work.
- Preserve accumulation order unless classified and reviewed as a numerical
  change.

**Profiling evidence to collect:** Shared-memory wavefronts/conflicts, async
copy issue/wait behavior, barrier stalls, FFMA/MMA instruction counts, tensor
pipe use where applicable, registers, occupancy, and product-plus-reduction
timing.

**Correctness checks:** Same P32 localized gates; additional tests for stage
tails, row grouping, split partials, and repeated graph/native launches.

**End-to-end benchmark checks:** Full decode at batch 1 plus selected larger
batches; full requested range before promotion; prefill guardrail if dispatch
shares these specializations.

**Expected decision:** Retain a bounded stage/layout specialization only when
it improves full decode and does not inflate compile/resource cost.

**Promotion criteria:** Matched timing win, generated-code audit, no spills or
unacceptable shared conflicts, accuracy gates, and ABI-visible configuration.

**Rollback/fallback criteria:** Existing stage/layout path remains default for
unsupported shapes, compile-growth concerns, graph failures, or flat timing.

**Dependencies:** Phase 4 and preferably Phase 5.

## Phase 7: Rank-8 projection load pressure and epilogue fusion

**Hypothesis:** Rank-8 projection is lower total priority but has high
occupancy with load/dependency stalls. Reducing dependent loads or fusing a
nearby epilogue may remove many small launches if the layout contract permits.

**Source ownership:** QvQ for `p32_rank8_project_kernel` and QVQ epilogue
semantics; ZML-Ultra if projection/epilogue fusion crosses the native bridge,
StableHLO, or graph boundary.

**Exact micro-area:** `p32_rank8_project_kernel<8>`, projection input/output
layout, correction/epilogue consumers, HADD2/FFMA/LDG mix, and adjacent
concatenate/convert/add kernels in the decode graph.

**Required baseline:** Rank-8 NCU profile: grid `1x8192x1`, 256 threads, 31
registers/thread, high active warps, LG throttle and long-scoreboard stalls,
0.718 ms/step over 80 launches.

**Experiment/intervention:**

- Analyze load addressing and coalescing before changing arithmetic.
- Test read-only cache/layout tweaks in isolation.
- Identify adjacent epilogue operations with identical layout and dtype
  boundaries.
- Prototype fusion only where the complete fused operation has a clear
  reference composition and graph-safe ownership.

**Profiling evidence to collect:** LDG sectors, L1/L2 hit rates, LG/MIO
throttle, instruction mix, launch count, fused/unfused Nsight Systems spans, and
full decode timing.

**Correctness checks:** Rank-8 output against reference composition, finite
outputs, QVQ local gates if within the quantized operator, graph/eager parity,
tails, repeated calls, and fallback routes.

**End-to-end benchmark checks:** Batch 1 and selected larger batches first;
full sweep only if rank-8 contribution changes materially.

**Expected decision:** Usually reject as lower priority unless it produces a
clear launch-count or load-stall reduction visible in full decode.

**Promotion criteria:** Full decode gain outside noise and no numerical,
layout, or graph regression.

**Rollback/fallback criteria:** Keep existing projection and epilogue when
fusion increases registers, introduces layout copies, or fails graph safety.

**Dependencies:** Phase 0; preferably after higher-ranked Phase 1/4 work.

## Phase 8: Small XLA launch, parameter update, and command-buffer overhead

**Hypothesis:** Many individually small XLA kernels and parameter updates may
become material after the larger K/V, attention, and P32 bottlenecks shrink.
Graph capture or command-buffer cleanup can reduce overhead only if ownership
and replay semantics stay intact.

**Source ownership:** ZML-Ultra primary. QvQ owns benchmark artifacts and any
QVQ launch-plan constraints exposed to the bridge.

**Exact micro-area:** XLA thunks, command-buffer capture, parameter update
nodes, tiny convert/concatenate kernels, launch gaps, stream ordering, buffer
donation, and graph invalidation.

**Required baseline:** Post-Phase 1/2/4 Nsight Systems trace with the remaining
top small kernels and CPU/API launch gaps aggregated.

**Experiment/intervention:**

- Aggregate small-kernel time and launch/API gaps before patching.
- Remove redundant parameter updates or host-side metadata work only when the
  graph replay address and lifetime proof is explicit.
- Prefer a native multi-launch wrapper if host dispatch is the bottleneck and
  GPU launch boundaries are still useful.
- Keep persistent/megakernel work out of this phase.

**Profiling evidence to collect:** Nsight Systems CUDA API, kernel execution,
queue time, stream timeline, graph replay boundaries, and allocation events.

**Correctness checks:** Actual Fast ABI external runtime graph/eager parity,
repeated replay with changed values, non-default streams, allocator pressure,
shape/context invalidation, and fallback on unsupported layouts.

**End-to-end benchmark checks:** Full decode batch 1 and full requested batch
range. Include TTFT/prefill because command-buffer changes can alter prefill.

**Expected decision:** Promote small launch cleanup only if aggregate overhead
is measurable after earlier phases and full decode improves.

**Promotion criteria:** Reduced launch/gap span in formal trace, graph-safe
replay proof, and warmed timing gain.

**Rollback/fallback criteria:** Existing graph/launch path remains default if
capture is unsafe, invalidation is incomplete, or timing is flat.

**Dependencies:** Phase 0; best after high-impact Phase 1/2/4 work.

## Phase 9: Persistent task graph or megakernel exploration

**Hypothesis:** If simpler copy-elimination, attention, and P32 work leaves a
decode critical path dominated by launch gaps, partial waves, or intermediate
traffic, a persistent task graph or megakernel may improve useful SM occupancy.

**Source ownership:** Split. QvQ owns P32/rank-8 kernel internals and any
native fused QVQ operator. ZML-Ultra owns whole-decode task scheduling,
attention/cache graph integration, StableHLO/PJRT/FFI execution, and serving
runtime state.

**Exact micro-area:** Cross-kernel dependency DAG, per-tile readiness counters,
instruction descriptors, persistent workspace, scratch ownership, graph replay,
QvQ launch ABI, attention/P32/rank-8 task scheduling, and fallback dispatch.

**Required baseline:** Post-Phase 1-8 formal trace showing remaining launch
gaps, partial waves, dependency waits, or intermediate traffic large enough to
justify a fused/persistent design.

**Experiment/intervention:**

- Draw the tile-level dependency DAG first; do not overlap truly sequential
  transformer phases.
- Define task opcodes, operand layouts, wait/signal counters, scratch
  ownership, and reset/epoch protocol.
- Start with a native wrapper or limited fused region before a whole-decode
  persistent kernel.
- Prove progress: no resident worker set may wait on producers queued behind
  it.
- Keep the existing multi-kernel path as fallback.

**Profiling evidence to collect:** Per-SM task timelines, dependency wait time,
intermediate traffic removed, launch count, critical-path tail, registers,
shared memory, residency, persistent-state audits, and full decode timing.

**Correctness checks:** Complete fused operation against unfused reference,
QVQ local gates for quantized sub-ops, graph/eager parity, repeated calls using
the same workspace, specialized-to-generic route transitions, timeouts for
deadlock detection, non-default streams, allocator pressure, and fallback.

**End-to-end benchmark checks:** Full decode batch 1 and requested batch range;
prefill and TTFT guardrails; memory footprint and failure recovery.

**Expected decision:** Proceed only if earlier simpler phases leave a measured
opportunity large enough to justify the state/lifetime risk.

**Promotion criteria:** Full decode gain, complete dependency/lifetime proof,
graph-safe replay, no deadlocks or workspace poisoning, generated-code audit,
and fallback coverage.

**Rollback/fallback criteria:** Keep or restore the existing graph/multi-kernel
path for unsupported hardware, layouts, graph modes, workspace sizes, failed
timeouts, or flat timing.

**Dependencies:** Phase 1, Phase 2 or 3, Phase 4, and Phase 8 evidence.

## Phase 10: Full campaign validation and release decision

**Hypothesis:** A candidate that improves batch-1 decode can still regress
larger batches, prefill, TTFT, graph replay, unsupported hardware, or build
stability. Final promotion needs the original benchmark contract.

**Source ownership:** QvQ for final benchmark records and QVQ kernels; ZML-Ultra
for runtime/compiler/attention changes; paired PRs for cross-stack features.

**Exact micro-area:** End-to-end Fast ABI decode/prefill/TTFT, batch OOM
classification, prefill regression guardrail, graph/eager parity, architecture
probing, and fallbacks.

**Required baseline:** Phase 0 hardened baseline and the nearest pre-candidate
commit for every repo involved.

**Experiment/intervention:**

- Run full requested batch sweep at context 131072 with unique prompts.
- Record OOM cases without fabricating metrics.
- Run formal Nsight Systems to confirm bottleneck movement.
- Run targeted Nsight Compute only on changed/dominant kernels.
- Confirm source and binary hashes, commands, exact branches, PRs, and artifact
  manifests.

**Profiling evidence to collect:** Final whole-step activity table, K/V copy
status, changed kernel SASS/NCU records, launch gaps, graph replay evidence,
and benchmark distributions.

**Correctness checks:** All global numerical and graph gates, plus CPU,
non-Ampere, unsupported-layout, and unsupported-graph fallbacks. Device
capability must be probed at runtime.

**End-to-end benchmark checks:** Batch 1, 2, 4, 8, 12, 16, 20, 24, 28, 32;
TTFT, prefill tokens/s, decode latency, decode tokens/s, OOM diagnostics,
memory usage, and unique-prompt proof.

**Expected decision:** Promote, hold for more evidence, split into QvQ/ZML
follow-up PRs, or reject/rollback.

**Promotion criteria:** Full benchmark improvement in the declared target,
no material prefill/TTFT regression, correctness gates pass, graph and fallback
coverage pass, and build resource limits are preserved.

**Rollback/fallback criteria:** Disable by default or revert if any required
batch/graph/fallback path fails, if prefill regresses materially without user
approval, or if source/build complexity is disproportionate to measured gain.

**Dependencies:** All implementation phases included in the candidate.

## Suggested execution order

1. Harden and recertify baseline/harness (Phase 0).
2. Open a ZML-Ultra investigation PR for K/V cache materialization (Phase 1).
3. In parallel, run QvQ-only P32 split/geometry enumeration without changing
   ZML external ABI defaults (Phase 4).
4. Once Phase 1 clarifies cache ABI, tune unified attention grid/config
   (Phase 2), then attention address/register micro-work (Phase 3).
5. Apply P32 SASS micro-work only to the retained P32 geometry (Phases 5-6).
6. Re-evaluate rank-8 and small-launch work after larger bottlenecks move
   (Phases 7-8).
7. Consider persistent/megakernel work only if formal traces still show a
   measured launch/partial-wave/intermediate-traffic opportunity (Phase 9).
8. Run the full original benchmark contract and promote only validated
   candidates (Phase 10).

## PR tracking checklist

Each future optimization PR should include:

- evidence label: measured, inference, experiment, accepted, rejected;
- source/base SHAs for QvQ and ZML-Ultra;
- exact model, shape, batch, context, graph mode, and GPU properties;
- correctness command and result;
- generated-code audit command and artifact path if GPU instructions changed;
- warmed end-to-end timing table;
- prefill/TTFT guardrail result;
- graph-safety and fallback result;
- build-resource settings used;
- explicit rollback/fallback behavior;
- linked parallel PRs when ownership crosses QvQ and ZML-Ultra.
