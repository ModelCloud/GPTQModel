# Mega-kernel development playbook

## Contents

1. [What to inspect first](#what-to-inspect-first)
2. [Choose the least costly fusion level](#choose-the-least-costly-fusion-level)
3. [Prove dependencies and scratch lifetimes](#prove-dependencies-and-scratch-lifetimes)
4. [Budget the resource union](#budget-the-resource-union)
5. [Scheduling and memory tricks](#scheduling-and-memory-tricks)
6. [Downsides and common failure modes](#downsides-and-common-failure-modes)
7. [Profiler signals and decision rules](#profiler-signals-and-decision-rules)
8. [Lessons from Marlin+EoRA](#lessons-from-marlineora)
9. [Review checklist](#review-checklist)

## What to inspect first

Inspect in this order before writing fused code:

1. **Timeline:** Use Nsight Systems to count launches and measure each kernel,
   launch API, gaps, synchronization, allocations, and the total GPU span.
2. **Dependency DAG:** Identify input/output ownership and the earliest legal
   start of each phase. Separate true dependencies from host sequencing.
3. **Resource table:** Record block/grid geometry, registers, shared memory,
   occupancy, waves, spills, and cooperative-launch eligibility for every kernel.
4. **Critical tail:** Find the last CTA/warp to finish each phase. Check
   `work_tiles % gridDim`, stripe imbalance, lock/reduction ordering, and idle
   warps before changing arithmetic.
5. **Memory lifetime:** Mark the final reader and first overwriter for every
   global/shared temporary. Note the fence or barrier between them.
6. **Access pattern:** Inspect transaction coalescing, broadcasts, reuse, atomics,
   and arithmetic intensity. Do not label a kernel bandwidth-bound from load
   instructions alone.
7. **Deployment contract:** Check current stream, device guard, graph capture,
   architecture, SM count, dynamic shared-memory opt-in, and fallback behavior.
8. **Numerics:** Record accumulation precision and order against a dense or
   unfused reference.

Write a table like this before selecting a design:

```text
phase  grid/block  p50_us  regs  dyn_smem  spills  occupancy  dependency  scratch
-----  ----------  ------  ----  --------  ------  ---------  ----------  -------
base   ...         ...     ...   ...       ...     ...        input       C_tmp
down   ...         ...     ...   ...       ...     ...        independent work
up     ...         ...     ...   ...       ...     ...        down result output
```

## Choose the least costly fusion level

Use the smallest level that removes the measured cost:

1. **Python/wrapper cleanup:** Reuse prepared metadata and allocations; avoid
   redundant validation, tuple construction, or device queries.
2. **One native operator, multiple GPU launches:** Remove Python/dispatcher gaps
   while preserving proven kernels and their independent resource envelopes.
3. **CUDA graph:** Amortize stable launch sequences when capture and replay are
   already production-compatible.
4. **Single ordinary kernel:** Fuse producer/epilogue work within one CTA when no
   grid-wide dependency exists.
5. **Cooperative mega-kernel:** Use `cooperative_groups::grid_group::sync()` only
   when phases require cross-CTA visibility and the entire grid can be resident.
6. **Persistent kernel:** Reserve for repeated work with enough reuse to offset
   residency, scheduling, watchdog, and integration complexity.

Fewer launches are not automatically faster. A native multi-launch operator may
beat a mega-kernel whose resource union destroys occupancy. Stream overlap may
beat fusion when phases are independent.

## Prove dependencies and scratch lifetimes

For every boundary, state all four facts:

```text
producer writes -> consumer reads
scope: warp | CTA | grid | stream
visibility primitive: shuffle | syncwarp | syncthreads | grid.sync | kernel boundary
scratch reuse: final old reader -> first new writer
```

Apply these rules:

- Use `__syncwarp()` only for participating lanes with a correct mask.
- Place `__syncthreads()` on a control path reached by every thread in the CTA.
- Use a cooperative grid barrier only in a cooperative launch whose grid fits
  simultaneous residency. Never emulate a global barrier by spinning arbitrary
  oversubscribed CTAs.
- Treat a kernel boundary as both ordering and memory visibility on one stream;
  replacing it requires an explicit equivalent.
- Do not remove a terminal block barrier merely because output looks correct in
  one run. Prove that no thread reads or reuses the shared region afterward.
- Do not alias scratch based only on allocation size. Prove last-use ordering and
  alignment for both types.
- Remember that atomic completion does not by itself prove every consumer sees a
  complete vector; define the barrier after the final producer.

Independent phases may be interleaved only if that shortens the critical path.
Moving equal work before a barrier often changes ordering without changing
`max(block_finish + phase_work)`.

## Budget the resource union

A mega-kernel pays for its worst phase for the whole launch:

- Registers are allocated from the compiled kernel's maximum live requirement.
- Dynamic shared memory is reserved at launch and cannot be released after a
  phase.
- Static shared memory and stack/local memory also reduce residency.
- Cooperative grids are capped by active blocks per SM across this union.
- Code size and phase-specific branches can increase instruction-cache pressure
  and JIT build time even when they are cold at runtime.

Calculate occupancy before implementation with the exact target attributes and
confirm it with NCU after compilation. Inspect ptxas register/spill output. A
successful cooperative launch proves legality, not that the grid is efficient.

Keep specialization-only state inside compile-time gates. Adding a heavy phase
to a generic template can raise registers or shared memory for unrelated shapes
even when a runtime condition skips it.

Low occupancy is not automatically wrong. A one-block-per-SM kernel can be fast
when it has sufficient ILP and each block owns useful work, but it becomes
especially sensitive to long scoreboards and one slow CTA at a grid barrier.

## Scheduling and memory tricks

### Eliminate structural tails

- Compute `tiles / grid`, `tiles % grid`, and waves per SM for every phase.
- Give remainder tiles to idle warps in the first CTAs instead of making those
  CTAs execute another full serial pass.
- Let primary and secondary warp groups write disjoint shared regions, cross one
  CTA barrier, then assign disjoint writer lanes.
- Preserve bounds for every generalized grid even when dispatch currently gates
  an exact SM count.

### Reuse dead storage

- Reuse a base kernel's reduction scratch for an adapter or epilogue only after
  the base result and all scratch readers complete.
- Prefer shared scratch already reserved by the heaviest phase when the later
  reduction fits; this avoids VRAM and launch-resource growth.
- Pack temporary storage by live rows/columns rather than padded logical shape,
  but retain alignment required by vectorized accesses.

### Reduce atomics without breaking coalescing

- Reduce within lanes, then warp, then CTA, and issue global atomics last.
- Map a warp across adjacent matrix columns/ranks when the storage is row-major.
- Compare the full transaction pattern before choosing a smaller rank tile. A
  lower atomic count can lose if it multiplies broadcast or sector traffic.
- Verify that atomic order changes remain within the numerical contract.

### Add bounded memory-level parallelism

- Interleave 2, 4, then 8 independent accumulation chains when NCU attributes
  stalls to serial long-scoreboard dependencies.
- Watch registers, spills, instruction count, p95, and the second dtype.
- Stop when additional chains only shift latency between event-resolution bins
  or worsen the mean/tail.

### Specialize warp roles

- Use otherwise-idle warps for a secondary tile, reduction, prefetch, or
  epilogue only when their work does not extend the critical CTA.
- Keep shared indices disjoint between roles and make writer ownership obvious.
- Prefer compile-time roles for fixed production shapes; avoid hot-loop dynamic
  division and modulo when a specialized mapping can precompute them.

## Downsides and common failure modes

Expect and measure these costs:

- **Occupancy collapse:** The heaviest phase's shared memory or registers govern
  all phases. Two individually fast kernels can become one latency-bound kernel.
- **Barrier tail amplification:** Every grid barrier waits for the slowest CTA.
  Stripe imbalance that was previously hidden by the next launch becomes direct
  critical-path idle time.
- **Incompatible geometries:** A block shape ideal for tensor-core GEMM may leave
  half the warps idle in a vector reduction or epilogue.
- **Lost concurrency:** Fusion can serialize kernels that could overlap on
  separate streams or copy/compute engines.
- **Cache interference:** A later phase may evict data needed by an earlier
  pipeline or inherit an unsuitable cache policy.
- **More instructions:** Bounds, role selection, shared reductions, and phase
  setup may outweigh saved launch time.
- **Numerical drift:** Combining reductions commonly changes FP32 or low-precision
  addition order.
- **Graph restrictions:** Cooperative launch and capture support differ by CUDA,
  driver, framework, and runtime state. Preserve an explicit graph-safe route.
- **Portability loss:** SM count, shared-memory capacity, cluster support, and
  cooperative residency vary within an architecture family.
- **JIT/codegen growth:** Large templates increase fingerprints, build time,
  binary size, and the chance that an unrelated specialization changes.
- **Debugging difficulty:** An illegal access or deadlock now spans several
  phases. Keep phase comments, boundary assertions, and focused tests.
- **Misleading timing:** Nsys host ranges include instrumentation; NCU replays a
  perturbed kernel; CUDA events may have coarse visible buckets at microsecond
  scale.

Reject fusion when launch/gap time is negligible, the resource union loses more
than it saves, a global barrier cannot be made resident, a portable gate cannot
be defined, or correctness requires a broad untested reduction-order change.

## Profiler signals and decision rules

Use Nsight Systems for selection evidence on launch structure:

```text
variant  launches  kernel_p50  kernel_mean  kernel_p95  gpu_span  host_range  peak
-------  --------  ----------  -----------  ----------  --------  ----------  ----
base     ...       ...         ...          ...         ...       ...         ...
fused    ...       ...         ...          ...         ...       ...         ...
```

Use Nsight Compute to explain, not manufacture, a win. Compare:

- duration, waves, achieved/theoretical occupancy, registers, shared, spills;
- eligible warps, issued warps, no-eligible percentage, cycles per issue;
- barrier, long/short scoreboard, wait, not-selected, and throttle stalls;
- DRAM bandwidth percentage and bytes/s, L1/L2 hit rate, compute throughput;
- executed/issued instructions and source hotspots;
- maximum versus average SM active cycles for imbalance.

Decision rules:

1. Require correctness before timing.
2. Use matched source, environment, clocks/state, warmup, and profiler options.
3. Require p50 plus mean and p95 improvement for small raw-kernel claims.
4. Repeat gains near profiler resolution.
5. Cross-check dtypes because register allocation and generated instructions can
   differ.
6. Retain a change only when full-operator impact is positive or the raw benefit
   is clearly attributed and does not harm the full path.
7. Keep allocator-visible peak and persistent per-layer memory in the table even
   when the optimization targets latency.

## Lessons from Marlin+EoRA

The rank-128 `M=1, K=N=4096` W4A16 adapter path provides concrete lessons:

1. **One native launch was worthwhile.** Combining Marlin and EoRA removed one
   GPU launch plus the inter-kernel gap. BF16 GPU work span fell 26.495 to
   23.328 us (-11.95%) and the profiled host range fell 13.34%.
2. **The mega-kernel inherited Marlin's resource ceiling.** A 166,912-byte
   dynamic shared-memory request and 96 BF16 registers allowed one 256-thread
   CTA per each of 124 SMs. Later LoRA phases could reuse the dead shared
   scratch but could not improve occupancy.
3. **Profile classification mattered.** NCU showed low DRAM and compute
   utilization, about 80% scheduler cycles with no eligible warp, no spills,
   and dominant barrier/scoreboard stalls: latency, not bandwidth saturation.
4. **Bounded ILP helped until it did not.** Eight independent LoRA-down
   accumulation chains reduced serial load dependencies. Sixteen chains moved
   p50 slightly but regressed mean and p95, so it was rejected.
5. **Inspect tile remainders before rewriting math.** The 124-CTA grid served
   128 LoRA-up tiles. Blocks 0-3 ran a second tile while upper warps were idle.
   Assigning those four tiles concurrently to warps 4-7 removed the tail and
   improved raw p50 about 3.6-4.3% without changing resources.
6. **Coalescing beat minimum atomics.** Mapping each CTA to 32 adjacent ranks
   reduced atomics from 15,872 to 3,968 and improved both dtypes. A 16-rank tile
   halved atomics again but added input broadcasts and was 1.1-1.6% slower.
7. **Barrier deletion needs distribution evidence.** Removing only a terminal
   up barrier slightly moved p50 but left mean and p95 worse; it was rejected.
8. **Raw and end-to-end timers answer different questions.** The final fused
   scheduling pass reduced raw p50 from 20.544 to 19.456 us FP16 and 20.896 to
   19.632 us BF16, while synchronized full-call p50 stayed in the same 30.72 us
   visible event bucket.
9. **More instructions can still be faster.** The coalesced shared reduction
   increased executed instructions but reduced serialization. NCU replay and
   200-launch Nsys distributions confirmed the win.
10. **Memory stayed flat because storage was reused.** The retained changes used
    existing Marlin scratch and kept the target allocator peak at 24 KiB.

Use `eora_marlin.md` for exact commands, artifacts, full metric tables, rejected
variants, and environment history. Generalize the mechanisms, not fixed CUDA
indices or one board's SM count.

## Review checklist

Before retaining a mega-kernel, verify:

- [ ] Exact dependency DAG and scratch lifetime are documented.
- [ ] Dense/unfused reference passes for every fused dtype.
- [ ] Nsys proves launch/gap/span opportunity and final launch count.
- [ ] NCU records resource union, occupancy, stalls, traffic, and imbalance.
- [ ] Cooperative grid fits simultaneous residency on the runtime-probed device.
- [ ] Every block/warp reaches required barriers on all control paths.
- [ ] Tail tiles and non-divisible dimensions are bounds-checked.
- [ ] Numerical reduction-order impact is measured.
- [ ] Normal p50/mean/p95, raw p50/mean/p95, throughput, and memory are reported.
- [ ] Small gains repeat and cross-check the second dtype/device.
- [ ] Compute Sanitizer memcheck and applicable synccheck/racecheck runs pass on
      the final exact source.
- [ ] Graph capture and unsupported hardware take a tested fallback.
- [ ] Architecture-specific code is compile/runtime gated without fixed indices.
- [ ] Rejected experiments are reverted and logged.
- [ ] Profiler artifacts remain out of commits unless explicitly requested.
