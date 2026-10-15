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
| 256 | 1.676 ms |
| 1024 | 6.185 ms |

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

`quantize_tiles` is called 5 760 times per run. Instrumenting the call sites
splits them cleanly:

| Source | Launches |
| --- | --- |
| `ldlq` (one per 16-row block) | 4 512 |
| `g_scale_gss` (global-scale search) | 1 248 |
| total | 5 760 |

`ldlq` issues exactly `size_k / 16` launches per module, and that count is
invariant under `buf_size_k`: a larger span trades outer iterations for inner
ones. `mlp.down_proj` is `(4864, 896)` and therefore contributes 304 launches
per layer, about 35% of the layer's 528 critical-path launches.

`g_scale_gss` issues `13 * ceil(sample_tiles / 256)` launches. The 13 is not a
tunable: it is a golden-section search over `[0.1, 1.9]` with `tol = 0.01`,
which shrinks the bracket by `1/phi` per iteration and needs 11 iterations plus
2 probes. `sample_tiles` is `max(tiles_k, tiles_n) * 3`, which is 168 for the
896-square and `(896,128)` modules and 912 for `(896,4864)` and `(4864,896)`.

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
The GPU-side cost of `ldlq` is `1.5 ms * (size_k / 16)`.

## Negatives that should not be re-derived

Each of these was measured and rejected. They are recorded so the same ground is
not covered twice.

**The proxy-error Hessian copy is free.** `metric_h2d` (`Hd = H.to(device)`,
a pageable host-to-device copy of the CPU-resident `k x k` Hessian) is 7.98 s of
the 11.78 thread-seconds, by far the largest single thread-time item, and for
`(4864, 896)` it is 459 ms of a 603 ms module. It is nonetheless invisible in
wall time: keeping the Hessian device-resident and skipping the copy entirely
gives 7.99 s against a 7.98 s baseline with the state hash unchanged. The copy
overlaps the GPU bottleneck.

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

The kernel is a fixed-latency chain, `ldlq` is irreducibly sequential along
`size_k`, and the subset barrier caps concurrent chains at 1-3. The remaining
headroom is therefore in **raising the number of independent tile chains in
flight**, not in shortening any single chain:

* `g_scale_gss` and the non-`gss` parts of `regularize` depend only on the
  original weight and the finalized Hessian, both of which are available for
  every module of a layer before the subset loop starts. Hoisting them out of
  the per-subset critical path would overlap 208 launches per layer that are
  currently serial, worth roughly 1.6 s.
* `mlp.down_proj` is a single-module subset contributing 304 launches per layer.
  Any restructuring that lets the next layer's independent work start while it
  runs attacks 3.6 s of the 8.0 s directly.

Both are structural changes to the layer/subset schedule rather than to the
kernel, and both need the state hash to be re-validated.
