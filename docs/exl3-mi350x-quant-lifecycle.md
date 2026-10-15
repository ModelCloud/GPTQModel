<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# EXL3 quantization-lifecycle concurrency on MI350X (gfx950)

This note records where the EXL3 **quantization lifecycle** (calibration capture
-> Hessian finalize -> LDLQ -> pack) spends its time on a single MI350X, and
which of the obvious levers are already exhausted. It is a measurement record,
not a description of the current implementation: several of the changes that
produced these numbers are the ones merged in #3308, #3309, #3310 and #3311.

The scope is the quantizer only. Inference kernels are out of scope.

## Reference configuration

| Item | Value |
| --- | --- |
| Device | 1x MI350X, gfx950, 256 CUs |
| Model | `tiny-qwen2-base` (896 hidden, 6 layers, 42 quantized modules) |
| Method | `EXL3`, 4-bit, `K = 4`, codebook `mcg` |
| Calibration | 8 samples, 136 tokens |
| Wall clock | 7.97-8.02 s steady state, `warm` 14.6 s |
| State hash | `c8f7f96a...085f6ca` (201 tensors) |

The per-layer module shapes are `(896,128)` x2, `(896,896)` x2, `(896,4864)` x2
and `(4864,896)` x1, so each layer quantizes 7 modules.

## The tile quantizer is 92% of all GPU time

A torch CUDA profiler run over one quantization reports 90 297 kernel records.
Aggregated by kernel:

| Kernel | Calls | GPU time | Share |
| --- | --- | --- | --- |
| `quantize_tiles_kernel<4, 1>` | 5 759 | 10.334 s | 91.9% |
| `at::native::index_elementwise_kernel` | 7 806 | 0.278 s | 2.5% |
| `at::native::vectorized_elementwise_kernel` | 12 858 | 0.066 s | 0.6% |
| `rocsolver::potf2_kernel_small` | 222 | 0.065 s | 0.6% |
| `__amd_rocclr_copyBuffer.kd` | 10 163 | 0.054 s | 0.5% |
| everything else | | < 0.2 s | |

Mean tile-kernel duration is 1.792 ms. Nothing else is worth optimizing at this
level; the rest of the lifecycle is rounding error against this one kernel.

## The kernel is latency, not throughput

The kernel walks a 256-position trellis twice (tail-biting), with a
`__syncthreads()` between every position, so a block's runtime is a fixed
serial depth of roughly 512 dependent steps. The grid dimension only decides how
many CUs are occupied, not how long the block takes. Standalone HIP
microbenchmarks against the same `quantize_tiles_kernel<4, 1>` instantiation,
`mcg` codebook, 1024 threads:

| Tiles (grid) | Time |
| --- | --- |
| 8 | 1.500 ms |
| 56 | 1.514 ms |
| 128 | 1.518 ms |
| 256 | 1.676 ms |
| 304 | 3.120 ms |
| 512 | 3.087 ms |
| 1024 | 6.185 ms |

The step at 304 and 512 is the wave boundary: a launch costs
`ceil(tiles / 256) * ~1.55 ms`, so a 56-tile launch occupies 56 of 256 CUs for
exactly as long as a 256-tile launch occupies all of them. Grid size buys
occupancy, never latency.

56 blocks on a 256-CU device cost the same as 8. The kernel uses 128 VGPRs at
1024 threads, i.e. the whole 131 072-entry VGPR file, which caps residency at one
block per CU. The instantiation is therefore a **latency chain**, and the only
lever that moves wall time is how many independent chains are in flight.

The same flat cost curve appears inside a real run. Grouping the trace's tile
kernels by grid size:

| Grid | Launches | Mean duration | Total |
| --- | --- | --- | --- |
| 256 | 1 374 | 2.368 ms | 3.253 s |
| 168 | 312 | 1.719 ms | 0.536 s |
| 144 | 234 | 2.062 ms | 0.483 s |
| 56 | 2 495 | 1.524 ms | 3.801 s |
| 48 | 672 | 1.864 ms | 1.253 s |
| 8 | 672 | 1.500 ms | 1.008 s |

A grid of 8 and a grid of 56 both cost about 1.5 ms; only at 256 does the mean
rise, and then by less than 60%.

### Why the block takes 1.5 ms

The block is issue-bound, not stall-bound. Emitting gfx950 assembly for
`quantize_tiles_kernel<4, 1>` and counting the steady-state trellis loop
(`.LBB12_3`, the `for (int i = 1; i < 256; ++i)` body) gives **526 instructions
per position iteration**:

| Opcode class | Count | Role |
| --- | --- | --- |
| `v_cndmask` | 120 | 2 per relaxation: `min_err`, `min_in_edge` |
| `v_fma` / `v_fma_mix_f32` | 112 | `dec - wf` and `fmaf(dh, dh, cost)` |
| `ds_read_u16` | 65 | 1 cost read per relaxation |
| `v_cmp` | 60 | 1 per relaxation |
| `s_waitcnt` | 46 | LDS/global scoreboard waits |
| `v_lshl` / `v_or` | 69 | addressing and `in_edge_idx` |
| `v_pk_add_f32` / `v_pk_fma_f32` | 8 | packed fp32 pairs |
| `global_store_short` / `ds_write_b16` | 8 | result and traceback writes |

255 iterations x 2 tail-biting laps x 526 = 268 K instructions per warp. With
1024 threads that is 8 warps per SIMD, so a SIMD issues 8 x 268 K = 2.15 M
instructions; at one issue per SIMD-cycle and ~2.2 GHz that is 0.98 ms against
a measured 1.43 ms, i.e. about 68% issue efficiency. Reducing the thread count
does not help: 512 threads doubles `n_oe` and therefore doubles the instructions
per warp while halving the warps per SIMD, leaving the same product.

The compiler is already at the floor for this formulation. Two codegen facts
matter, and both were confirmed in the assembly:

* The 16 predecessor reads per out-edge are emitted as
  `ds_read_u16 v97, v96 offset:512` ... `offset:3584`. For `K = 4` the state
  stride is `1 << (16 - 2K) = 256` halfs = 512 bytes, so the index arithmetic
  for `temp_costs_inc[((k << Kr) | out_edge_idx) >> K]` collapses into a single
  base register plus a fixed immediate offset. There is no per-relaxation
  address computation to remove.
* `__half2float` is free: `v_fma_mix_f32` takes the f16 operand directly, and
  where the dependency graph allows it the compiler emits `v_pk_add_f32` /
  `v_pk_fma_f32`, doing two fp32 lanes per instruction.

The per-relaxation floor is therefore 5 ops (subtract, fma, compare, two
selects) and the emitted loop is within about one op of it.

## Measured concurrency is 1.15, and the device can do far better

From the chrome trace, the tile kernels have a span of 8.949 s and 10.334 s of
total duration, so the time-weighted mean number of tile kernels executing at
once is **1.155**. Broken down by concurrency level:

| Concurrent tile kernels | Time | Share of span |
| --- | --- | --- |
| 0 | 1.196 s | 13.4% |
| 1 | 5.173 s | 57.8% |
| 2 | 2.580 s | 28.8% |
| >= 3 | 0 | 0% |

The device is not the limit. A standalone benchmark running the same
instantiation on four streams gets **3.59x** over one stream (`kt_streams 56 4`:
6.086 ms for four launches on one stream vs 1.694 ms across four streams), and
that holds with `hipFuncSetAttribute` called per launch, so the per-launch
attribute set is not the serializer. The Python tile path scales 3.32x across
four threads, and a faithful LDLQ replica scales 3.34x.

The real `ldlq` scales only **1.81x** across four threads. The gap between 1.81x
and the 3.32x that the same launch pattern achieves in isolation is the layer
pipeline, not the stream plumbing.

## The layer pipeline runs in lockstep rounds

Instrumenting `quantize_exl3` entry/exit and the subset boundaries produces a
strictly repeating per-layer schedule. Each layer issues 4 sequential subsets,
each with its own forward replay, and the number of modules that can be
quantized at once is set by the subset, not by the worker pool:

| Subset | Modules | Wall | Concurrent chains |
| --- | --- | --- | --- |
| 1/4 | `self_attn.q/k/v_proj` | 0.147 s | 3 |
| 2/4 | `self_attn.o_proj` | 0.119 s | 1 |
| 3/4 | `mlp.gate_proj`, `mlp.up_proj` | 0.431 s | 2 |
| 4/4 | `mlp.down_proj` | 0.611 s | 1 |

Per layer that is 1.31 s, and 6 layers gives 7.86 s of the 8.0 s wall. The gaps
between subsets are ~20 ms, so the pipeline is not stalling between stages; it
is simply that `true_sequential` forces subset N's replay to complete before
subset N+1 may read its activations, and `mlp.down_proj` is alone in the last
subset.

The same structure shows up as a stream census. The device pool has four
workers, and each worker drives its own quantization stream, yet across the
whole run the 5 759 tile launches appear on only three distinct streams and are
very unevenly distributed: 4 156 on one, 1 540 on a second, and 63 on a third.
At most three workers ever contribute tile work. The subset barrier, not the
pool size, is what decides how many chains exist.

## Launch accounting: 22% of launches are the scale search

`quantize_tiles` is called 4 386 times per run, and those calls issue 5 760
`quantize_tiles_kernel` launches: the HIP op splits a call into
`ceil(tiles / 256)` launches of at most 256 blocks. Instrumenting the call sites
splits both counts cleanly:

| Source | `quantize_tiles` calls | Kernel launches |
| --- | --- | --- |
| `ldlq` (one per 16-row block) | 3 840 | 4 512 |
| `g_scale_gss` (global-scale search) | 546 | 1 248 |
| total | 4 386 | 5 760 |

The 5 760 agrees with the 5 759 `quantize_tiles_kernel<4, 1>` records in the
profiler trace, and the scale search is still 22% of them.

`ldlq` calls `quantize_tiles` exactly `size_k / 16` times per module, once per
16-row block, and that count is invariant under `buf_size_k`: a larger span
trades outer iterations for inner ones. How many kernel launches those calls
issue depends on the shape, because the call's grid is `tiles_n`: 56 tiles is
one launch, and the 304-tile call of a `(896,4864)` module is two.
`mlp.down_proj` is `(4864, 896)`, so it makes 304 single-launch calls per layer:
304 of the layer's 752 `ldlq` kernel launches, and every one of them is on the
critical path because the module is alone in its subset.

`g_scale_gss` calls `quantize_tiles` 13 times per module, and each of those
calls is `ceil(sample_tiles / 256)` kernel launches -- one for the 168-tile
sample of the square and `(896,128)` shapes, four for the 912-tile sample of
`(896,4864)` and `(4864,896)`. The 13 is not a tunable: it is a golden-section
search over `[0.1, 1.9]` with `tol = 0.01`, which shrinks the bracket by `1/phi`
per iteration and needs 11 iterations plus 2 probes. `sample_tiles` is
`max(tiles_k, tiles_n) * 3`.

On an idle device the search is pure launch latency:

| Shape | Sample tiles | `g_scale_gss` | 13 bare tile launches |
| --- | --- | --- | --- |
| `(896, 128)` | 168 | 20.77 ms | 20.42 ms |
| `(896, 896)` | 168 | 20.73 ms | 20.00 ms |
| `(896, 4864)` | 912 | 81.44 ms | 80.66 ms |
| `(4864, 896)` | 912 | 81.40 ms | 80.74 ms |

So the search has no overhead beyond the 13 tile launches it performs. Gating it
off entirely drops the wall from 7.98 s to **6.28 s**, which is the measured
ceiling for any change that only removes the search. Note that this is a
ceiling, not a target: the search is what selects the global scale, so removing
it changes the quantized weights.

## Where the waves go: `ldlq`'s grid is only `tiles_n` wide

`ldlq` walks the weight in 16-row blocks, bottom-up, and calls
`quantize_tiles_multigpu` once per block with `tiles.shape[0] == tiles_n`. The
block is a genuine dependency: block `bi` consumes `weight_q[bj:]` produced by
the blocks above it, so the chain is `size_k / 16` deep and cannot be batched.
What it *can* be is concurrent, and it is not, because the grid is narrow.

| Shape | `tiles_n` | Chain depth | CU occupancy per launch | GPU time |
| --- | --- | --- | --- | --- |
| `(896, 128)` | 8 | 56 | 3% | 84 ms |
| `(896, 896)` | 56 | 56 | 22% | 84 ms |
| `(896, 4864)` | 304 | 56 | 100% | 168 ms |
| `(4864, 896)` | 56 | 304 | 22% | 456 ms |

`mlp.down_proj` is the pathological case: 304 sequential launches of 56 blocks,
each of which holds a CU for a full 1.5 ms wave. Packed to 256 blocks per wave
those same 17 024 tiles would need 67 waves, i.e. ~100 ms. The 4.5x difference
is not recoverable inside one module: the chain is sequential along `size_k`,
the tile grid is fixed by the weight shape, and splitting a 16x16 tile across
blocks would need a grid-wide barrier per trellis position.

Counting waves per subset -- one wave is one `ceil(tiles / 256)` kernel launch,
and 256 blocks cost ~1.55 ms -- and measuring the subset walls on the same run:

| Subset | Modules | Chains | `ldlq` waves | `gss` waves | Measured wall |
| --- | --- | --- | --- | --- | --- |
| 1/4 | `q,k,v_proj` | 3 | 168 | 39 | 148 ms |
| 2/4 | `o_proj` | 1 | 56 | 13 | 120 ms |
| 3/4 | `gate,up_proj` | 2 | 224 | 104 | 432 ms |
| 4/4 | `down_proj` | 1 | 304 | 52 | 615 ms |
| total | | | 752 | 208 | **1.31 s** |

960 waves at 1.55 ms would be 1.49 s if every wave queued behind the last, so
the subset walls are 1.31 s because the chains that fit inside one wave do share
it, and only those:

* Subset 1 launches 56, 8 and 8 blocks per step, 72 in all, so its three `ldlq`
  chains fit in a single wave: 56 waves of wall time carry 168 wave-slots. Its
  three 168-tile `gss` searches overlap the same way, 26 waves for 39 slots.
* Subset 3 launches 304 + 304 = 608 blocks per step, three waves' worth, so its
  two chains cannot share one: the subset pays three waves per step, 168 waves
  of wall time for its 224 `ldlq` slots. Its two `gss` searches do share: 52
  waves for 104 slots.
* Subset 4 is a single 56-block chain, so all 304 of its waves are wall time.

That is 584 `ldlq` wave-slots of wall time (~905 ms) plus 143 `gss` wave-slots
(~222 ms), and the remaining ~190 ms per layer is `regularize`'s non-`gss` work,
`finalize_capture_H`, the replay and the subset-boundary syncs. It reconciles
the measured 1.31 s per layer and the 7.86 s of the 8.00 s wall.

This also explains the `metric_h2d` column in the phase table below. The
`H.to(device)` copy at the end of the module is a *synchronous* pageable
host-to-device copy on the quantization stream, so it drains everything already
enqueued on that stream. For `(4864, 896)` it reports 459 ms of the module's
603 ms because that is exactly the `ldlq` chain draining, not because the copy
is expensive. Removing the copy (`EXL3_KEEP_H_DEV=1`) leaves the wall unchanged
because the GPU work still has to run; the wait simply moves.

## Per-module phase costs

Wrapping the phases of `_quantize_exl3_body` gives milliseconds per module
(`gss` is nested inside `regularize`):

| Shape | n | total | `gss` | `ldlq` | `h_finalize` | `regularize` | `metric_h2d` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `(4864, 896)` | 6 | 603.3 | 76.6 | 23.4 | 26.8 | 80.7 | 459.3 |
| `(896, 4864)` | 12 | 418.7 | 134.5 | 4.8 | 5.8 | 141.0 | 265.2 |
| `(896, 128)` | 12 | 137.1 | 33.0 | 4.7 | 11.7 | 34.6 | 85.1 |
| `(896, 896)` | 12 | 124.3 | 26.8 | 4.7 | 4.8 | 28.4 | 85.5 |

Summed over the run, 11.78 thread-seconds elapse inside `quantize_exl3` for
8.00 s of wall, i.e. mean worker occupancy 1.47 out of 4. Non-`gss`
`regularize` work is only ~4 ms per module.

`ldlq`'s column is thread time, not GPU time: the loop only enqueues launches.
The GPU-side cost of `ldlq` is one wave per launch, i.e.
`1.5 ms * (size_k / 16) * ceil(tiles_n / 256)`.

## Negatives that should not be re-derived

Each of these was measured and rejected. They are recorded so the same ground is
not covered twice.

**The proxy-error Hessian copy is free.** `metric_h2d` (`Hd = H.to(device)`,
a pageable host-to-device copy of the CPU-resident `k x k` Hessian) is 7.98 s of
the 11.78 thread-seconds, by far the largest single thread-time item, and for
`(4864, 896)` it is 459 ms of a 603 ms module. It is nonetheless invisible in
wall time: keeping the Hessian device-resident and skipping the copy entirely
gives 7.99 s against a 7.98 s baseline with the state hash unchanged. The
reported time is the quantization stream draining in front of the copy, as the
`ldlq` grid section explains, so removing the copy only moves the wait.

**Worker count does not matter.** 4 workers -> 7.97 s, 8 -> 8.03 s, 16 -> 8.05 s.
Each worker receives exactly 21 modules. The subset barrier, not the pool,
decides parallelism.

**Enlarging the tile scratch does not matter.** The HIP op batches tile launches
at `temp_costs.size(0)`, currently 256. Removing that cap so a whole matrix
launches at once gives no win, because the kernel is latency-bound:
912 tiles in one launch is 6.130 ms against 6.162 ms for four 256-tile launches.
304 tiles in one launch is 3.103 ms against 3.137 ms for two.

**Allocator configuration does not matter.** `PYTORCH_ALLOC_CONF`
`expandable_segments` on and off both give 1.86x/1.87x concurrency scaling.

**Kernel variants are all noise.** `#pragma unroll 4`,
`__launch_bounds__(..., 2)`, 512 and 256 threads, a no-unroll build, and
`dec_cache = false` were each tried. Only `dec_cache = false` moves the needle,
and it is 2.4x slower, so the decode cache stays.

**Inner-loop restructuring does not pay either.** Removing work in pieces shows
the loop is balanced and already minimal, and re-deriving the pieces in a
"cheaper" form makes it slower (grid 56, `K = 4`, `mcg`, 1024 threads, best of
five):

| Variant | Time | Spill |
| --- | --- | --- |
| baseline | 1.425 ms | 60 B |
| predecessor cost read replaced by a constant | 1.240 ms | 44 B |
| decode-table read replaced by a constant | 1.147 ms | 0 B |
| `temp_edges` global store removed | 1.184 ms | 20 B |
| inner `k` loop collapsed to `k = 0` | 0.251 ms | 0 B |
| inner `__syncthreads()` removed | 1.394 ms | 60 B |
| one forward lap instead of two | 0.699 ms | 0 B |
| permuted cost layout (see below) | 1.513 ms | 60 B |
| `min_k` tracked instead of `min_in_edge` | 1.460 ms | 156 B |
| `constrain` check removed | 1.681 ms | 140 B |

The `min_k` variant is bit-exact and still loses: hoisting
`((k << Kr) | out_edge_idx) >> K` out of the loop frees the `v_or` the compiler
was folding into an immediate, but it keeps the same two `v_cndmask`s and pushes
the live range of the loop-carried state across a spill boundary. The
permutation that makes the 16 predecessors of one out-edge consecutive
(`slot(x) = (x & state_hi_mask) * max_q + (x >> state_shift)`, which turns every
cost read into a fixed-immediate LDS offset) also loses, because the compiler
already emits exactly that addressing and the permutation only adds work to the
store side and gives it an 8-way shared-memory bank conflict.

The conclusion is that the tile kernel has no headroom left at the instruction
level. Further gains have to come from occupancy, not from the block.

**The GIL cannot be released.** The interpreter is a GIL build
(`Fatal Python error: config_read_gil: Disabling the GIL is not supported by
this build`) despite a startup banner that suggests `PYTHON_GIL=0`.

**The `finalize_capture_H` mutex is not a steady-state bottleneck.** 42 calls,
95.6 ms of total wait on a 9.12 s run.

**Stream primitives are cheap.** `torch.cuda.Stream.wait_stream` and `Event`
round-trips cost about 5 us even under load.

**Subset-boundary device syncs are free.** A run issues 36
`torch.cuda.synchronize()` calls (one per subset, plus the per-layer sync in
`stage_layer`) and waits a total of 0.002 s. The workers' stream brackets have
already ordered the work by the time each sync is reached, so the syncs are not
what serializes the subsets.

## What is left

The block is at its instruction floor, `ldlq` is irreducibly sequential along
`size_k`, and the subset barrier caps concurrent chains at 1-3. The remaining
headroom is entirely in **occupancy**: putting more independent tile chains in
flight so a 1.5 ms wave carries more than 56 blocks.

Three things follow from the step accounting above.

* `g_scale_gss` and the non-`gss` parts of `regularize` depend only on the
  original weight and the finalized Hessian, both of which exist for every
  module of a layer before the subset loop starts. They are 208 waves per layer
  (~222 ms of wall time) that currently run back-to-back with `ldlq` on the same
  stream rather than beside it. Hoisting them out of the per-subset critical
  path is the largest accuracy-preserving opportunity, but the bound is the free
  CUs, not the 222 ms: in subset 4 `ldlq` leaves 200 CUs idle for 615 ms, so the
  52 waves of `gss` there (81 ms) can hide completely, while in subset 1 the
  168-tile searches are wider than the 72 blocks `ldlq` launches there and would
  contend with it instead.
* Merging all seven layer modules into one subset removes the subset barriers
  and takes the wall from 7.97 s to **6.18 s**, verified twice. It is not a free
  win: it changes the state hash because `true_sequential` replay is what makes
  subset `N+1` capture against subset `N`'s *quantized* weights, and the code
  documents that as required for correctness. It needs an accuracy gate (the
  `logits_mse` / `cosine` end-to-end check) before it could be considered.
* Subset 4's 456 ms is a floor for a single module. `mlp.down_proj` is a
  single-module subset contributing 304 launches per layer, and nothing inside
  it can fill those CUs, so the only way to recover them is to give the chain
  company from outside the subset: any restructuring that lets the next layer's
  independent work start while it runs attacks 3.6 s of the 8.0 s directly.

All three are changes to the layer/subset schedule rather than to the kernel,
and each needs the state hash re-validated before it can ship.
