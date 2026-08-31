# QVQ exact-P32 Ampere kernel progression

This ledger tracks the exact continuous-window P32 inference kernel for
A100-class `sm_80` GPUs. It records accepted forward progress and failed or
discarded experiments so later tuning does not repeat unsafe variants.

## Contract and target

- Source base: freshly fetched GitHub `origin/main` at `eb0eefff` (tip after
  PR #75 merged).
- Device: physical GPU 0, `NVIDIA PG506-230`, UUID
  `GPU-14ab23f1-a785-e9df-bbb5-215547154e3c`, CC 8.0, 124 SMs, 96 GiB.
- Software: PyTorch 2.13.0+cu130; CUDA runtime 13.0; NVCC 13.3.
- Input and levels: FP16. Accumulation and output: FP32.
- Rates: exact standard-P32 W2, W2.5, W3, and W3.5 continuous-window payloads.
- Rows: M1 through M16; the current formal matrix covers M1, M2, M4, M8,
  and M16.
- Accuracy gate: identical packed payload and bank metadata, dense exact-P32
  reference, and maximum absolute inference drift `<= 2e-3`.
- Timing: 20 warmups and 100 per-iteration CUDA-event samples. Every formal run
  passed a three-sample 0% utilization/8 MiB idle gate and rejected foreign
  compute processes again before each timed pair.

The Ampere path explicitly rejects devices other than CC 8.0. It contains no
TMA, WGMMA, thread-block clusters, or distributed shared memory.

Marlin is another important teacher for this path because it is a very fast
Ampere-native weight-only kernel. Its warp partitioning, shared-memory layout,
software pipelining, vectorized movement, occupancy tradeoffs, and shape-aware
dispatch are relevant to future P32 tuning. Its packed-weight decode and
quantization contract are different, however, so those parts cannot be copied
directly into the exact continuous-window P32 representation.

## Accepted design

The kernel carries forward the parts of the Hopper work that do not depend on
Hopper-only hardware:

1. Consume the storage-neutral continuous-window P32 representation directly.
2. Assign four MMA warps to four adjacent N16 tiles so they share staged
   activation data. Decode lane-owned FP16 pairs directly into explicit
   `mma.sync.m16n8k16` B fragments instead of writing and reloading a shared
   weight tile.
3. Double-buffer two K16 activation/window tiles per stage with Ampere
   `cp.async`. Processing the K32 group behind one pair of block barriers halves
   synchronization and async-pipeline bookkeeping without adding a kernel
   specialization.
4. Decode states 64 pairs apart together because they share one funnel-shift
   amount and a compile-time word distance.
5. Accumulate in FP32 and use bounded split-K to expose enough CTA work. The
   generic fallback remains capped at eight slices; measured short-K policies
   use up to thirty-two slices, while the long-K M1/M2 scalar MLP-down route
   uses measured 128-way (M1) and 96-way (M2) waves to fill more of the
   A100's 124 SMs.
6. Use an explicit block barrier after MMA before reusing a stage buffer.
7. For M=1 through M=4, use a Marlin-style scalar route even on the measured
   long-K MLP-down shape; for other shapes, M=3-M4 keep that route for K<=6144. One
   128-thread CTA covers sixteen N16 tiles, each warp owns four tiles, and
   each lane accumulates its output columns directly. This avoids executing
   inactive rows of every `m16n8k16` instruction where the scalar route wins.
8. Keep the 512-byte codebook in Ampere's read-only cache (`__ldg`) and issue
   streaming trellis copies as `cp.async.cg`; use 32-bit funnel shifts for
   state extraction instead of 64-bit window promotion.
9. For M3-M4 and K<=6144, reuse the Marlin-style scalar tile schedule while
   carrying exactly the live rows in FP32 accumulators. M1-M4 also use this
   schedule for the long-K MLP-down shape after a scalar four-K16 stage was
   measured to overcome the old WMMA advantage. The short-K dispatch uses up to
   thirty-two K splits, except attention-out (K=6144, N=5120), where a
   measured 24-way wave reduces split-reduction overhead; M4 long-K and M8+
   remain on the tensor-core path.
10. For M8 and M16, retain WMMA arithmetic but choose the K split by M/N shape:
    M8 uses 16-way splits for wide QKV/MLP projections and 32-way splits for
    small-N KV/attention/Z projections; M16 uses 16-way splits for the latter
    group and 32-way splits for full K/V.

Future Ampere experiments should compare the generated instruction schedule,
register pressure, shared-memory bank behavior, and CTA swizzle against Marlin
as well as carrying forward architecture-independent lessons from the Hopper
kernel.

There are eight WMMA device specializations: four transition widths times
full-M16 and partial-row paths. The scalar M1-M4 rows add four exact
transition-width specializations, while one runtime split reducer is shared
by all rates. Unknown shapes use a live-SM-derived fallback; the seven
measured Qwen3.8-27B shapes use recorded split counts without embedding the
local 124-SM inventory.

## Accepted correctness

`tests/test_qvq_p32_ampere.py` passes 22/22 cases on the target GPU:

- W2-W3.5 at M1, M2, M4, M8, and M16;
- an N80 partial N64 block;
- split-K reconstruction;
- long-K accumulation;
- repeated launches on a non-default CUDA stream;
- output shape, FP32 dtype, finite values, exact repeatability, and the
  `2e-3` maximum-error contract.

The full formal benchmark adds 140 dense-reference cases across seven
Qwen3.8-27B projection shapes, four rates, and M1/M2/M4/M8/M16. All 140
pass.

## Accepted performance

Artifacts from the earlier accepted checkpoints and this tuning cycle:

- `artifacts/a100_p32_window/qwen38_m16_p32_ampere.json`
- `artifacts/a100_p32_window/qwen38_m1_p32_ampere.json`
- `artifacts/a100_p32_window/qwen38_m16_p32_ampere_v2.json`
- `artifacts/a100_p32_window/qwen38_m1_p32_ampere_v2.json`
- `artifacts/a100_p32_window/qwen38_m1_p32_ampere_v3.json`
- `artifacts/a100_p32_window/qwen38_m1_p32_ampere_v4.json`
- `artifacts/a100_p32_window/qwen38_m1_p32_ampere_v6.json`
- `artifacts/a100_p32_window/qwen38_m24_p32_ampere_v1.json`
- `artifacts/a100_p32_window/qwen38_mixed_p32_ampere_v1.json`
- `artifacts/a100_p32_window/qwen38_m8_p32_ampere_v2.json`
- `artifacts/a100_p32_window/qwen38_m16_p32_ampere_v3.json`
- `artifacts/a100_p32_window/qwen38_m124_attention_p32_ampere_v2.json`
- `artifacts/a100_p32_window/qwen38_mixed_longk_attn_p32_ampere_v2.json`
- `artifacts/a100_p32_window/qwen38_mixed_p32_ampere_v5.json`
- `artifacts/a100_p32_window/qwen38_mixed_p32_ampere_v7.json`
- `artifacts/a100_p32_window/qwen38_mixed_p32_ampere_v8.json`
- `artifacts/a100_p32_window/qwen38_origin_main_5f45eb2e.json`
- `artifacts/a100_p32_window/qwen38_mixed_p32_ampere_v10.json`
- `artifacts/a100_p32_window/qwen38_origin_main_b7d68545.json`
- `artifacts/a100_p32_window/qwen38_mixed_p32_ampere_v11.json`
- `artifacts/a100_p32_window/qwen38_mixed_p32_ampere_v12.json`
- `artifacts/a100_p32_window/qwen38_origin_main_eb0eefff.json`
- `artifacts/a100_p32_window/qwen38_mixed_p32_ampere_v13.json`

The comparator is the current canonical planar P32 CUDA GEMV built from the
same checkout. It is quality-equivalent, unlike a W4 kernel comparison.

The planar numbers below are oracle context only. The progress target for this
post-merge cycle is speedup versus the freshly fetched `origin/main` control at
`b7d68545`, measured with the same 140-case matrix and the same CUDA-event
protocol.

| Regime | Cases | Geomean speedup vs planar P32 | Speedup range | Worst max abs |
|---|---:|---:|---:|---:|
| M16 (previous) | 28 | 15.651x | 4.094x-21.478x | 2.823e-4 |
| M16 (v2) | 28 | 15.710x | 4.212x-21.543x | 2.823e-4 |
| M1 (previous) | 28 | 7.130x | 1.010x-21.149x | 2.632e-4 |
| M1 (v2) | 28 | 7.701x | 0.952x-29.848x | 2.632e-4 |
| M1 (v3, eight-way scalar split) | 28 | 7.918x | 0.952x-29.839x | 2.632e-4 |
| M1 (v4, sixteen-way scalar split) | 28 | 9.330x | 1.020x-31.519x | 2.632e-4 |
| M1 (v6, thirty-two-way scalar split) | 28 | 9.620x | 1.041x-35.168x | 2.632e-4 |
| M2 (small-M scalar, thirty-two-way split) | 28 | 14.179x | 1.340x-31.019x | 2.632e-4 |
| M4 (small-M scalar, thirty-two-way split) | 28 | 12.578x | 1.340x-25.280x | 2.632e-4 |
| M8 (WMMA partial rows, shape splits) | 28 | 11.775x | 1.422x-22.594x | 2.594e-4 |
| M16 (WMMA full rows, shape splits) | 28 | 16.242x | 4.739x-21.543x | 2.823e-4 |
| M1 attention-out (24-way scalar split) | 4 | 10.481x | 4.479x-23.683x | 1.645e-5 |
| M2 attention-out (24-way scalar split) | 4 | 15.517x | 5.130x-22.827x | 1.717e-5 |
| M4 attention-out (24-way scalar split) | 4 | 14.652x | 5.148x-21.114x | 1.955e-5 |

The previous five-row-count run had Ampere geometric-mean latencies of
0.073924 ms (M1), 0.078358 ms (M2), 0.088391 ms (M4), 0.100471 ms (M8), and
0.102477 ms (M16). The shape-split refresh lowers M8 to 0.094743 ms (`1.060x`)
and M16 to 0.098617 ms (`1.039x`) while preserving the M1-M4 paths. The
additional `2x` stretch target remains open.

M16 per-shape speedup ranges across W2-W3.5:

| Shape | K | N | Split | Speedup range |
|---|---:|---:|---:|---:|
| Full Q+gate | 5120 | 12288 | 5 | 20.570x-21.346x |
| Full K/V | 5120 | 1024 | 8 | 4.094x-4.469x |
| Attention out | 6144 | 5120 | 8 | 16.041x-16.795x |
| Linear QKV | 5120 | 10240 | 6 | 20.029x-20.545x |
| Linear Z | 5120 | 6144 | 8 | 20.352x-20.743x |
| MLP gate/up | 5120 | 17408 | 8 | 20.452x-21.478x |
| MLP down | 17408 | 5120 | 8 | 17.503x-18.783x |

The v2 M1 artifact reduces geometric-mean Ampere latency from 0.100193 ms to
0.092794 ms (`1.080x`), while M16 changes from 0.102317 ms to 0.101903 ms
(`1.004x`). M1 full-Q+gate W2 improves from roughly 0.116 ms to 0.091 ms;
the long-K MLP-down route remains on WMMA and avoids the scalar route's
regression. The v3 scalar split update lowers M1 geometric-mean latency again
to 0.090268 ms (`1.028x` over v2, `1.110x` over the prior checkpoint); full-Q
W2 reaches 0.084992 ms. The v4 sixteen-way scalar split lowers the geometric
mean again to 0.076604 ms (`1.178x` over v3), with full-Q W2 at 0.072704 ms.
The v6 thirty-two-way scalar split lowers the geometric mean again to
0.074286 ms (`1.031x` over v4), with full-Q W2 at 0.066560 ms. This is
accepted forward progress, but the additional `2x` Ampere stretch target
remains open.

The attention-out scalar split refresh uses 24 rather than 32 K slices. In the
formal four-rate artifact it reaches 0.051562 ms (M1), 0.054521 ms (M2), and
0.057835 ms (M4), with exactness preserved; paired split probes showed 24-way
waves 5.7-19.4% faster than 32-way waves on this shape.

## Post-merge versus-main progress

The exact control is `qwen38_origin_main_b7d68545.json`, produced from
`origin/main` at `b7d68545` before this branch. The candidate is
`qwen38_mixed_p32_ampere_v12.json` at `b2b0f8c7`. Each row is the
geometric mean of the 28 cases for that M, using each case's Ampere event
median; this deliberately excludes planar-oracle timing from the target.

| M | Main geomean ms | Candidate geomean ms | Speedup vs fetched main |
|---:|---:|---:|---:|
| 1 | 0.067228 | 0.066702 | 1.008x |
| 2 | 0.072805 | 0.071788 | 1.014x |
| 4 | 0.085102 | 0.084647 | 1.005x |
| 8 | 0.093966 | 0.092688 | 1.014x |
| 16 | 0.096349 | 0.096002 | 1.004x |

The accepted changes are deliberately narrow: the measured long-K Qwen3.8
MLP-down shape (K=17408, N=5120) moves from main's eight-way split to a
128-way (M1) or 96-way (M2) split, M1-M2 use the scalar route on that shape, and M16
attention-out (K=6144, N=5120) uses a measured 12-way split. The focused
MLP-down rows improve 1.008x, 1.022x, 1.002x, 1.000x, and 1.000x for
M1/M2/M4/M8/M16 respectively; M16 attention-out is 1.000x. Because MLP-down
is only one of seven shapes, the complete matrix is currently 1.004-1.014x
versus fetched main in this timing sample, so the requested 2x target remains
open.

## Latest post-merge versus-main progress

The new control is `qwen38_origin_main_eb0eefff.json`, produced from the
freshly fetched `origin/main` after PR #75 merged. The candidate is
`qwen38_mixed_p32_ampere_v13.json`, measured with the same 140 cases, 20
warmups, 100 CUDA-event iterations, and idle/foreign-process gates. Values are
geometric means of the 28 Ampere event medians for each M; planar timings are
not part of this target.

| M | New main geomean ms | Candidate geomean ms | Speedup vs fetched main |
|---:|---:|---:|---:|
| 1 | 0.066583 | 0.066654 | 0.999x |
| 2 | 0.072008 | 0.071404 | 1.008x |
| 4 | 0.084860 | 0.084158 | 1.008x |
| 8 | 0.092692 | 0.092553 | 1.002x |
| 16 | 0.096069 | 0.096158 | 0.999x |

The accepted v13 change specializes the Marlin-style scalar path for the
long-K MLP-down shape (K=17408, N=5120) at M1-M4 and processes four K16 tiles
per software stage, while keeping the proven two-tile WMMA stage for other
shapes and M8/M16. Focused long-K geometric-mean improvements versus the new
main were 1.000x (M1), 1.033x (M2), and 1.066x (M4). The complete matrix is
currently 0.999-1.008x versus the fetched main, so the requested 2x target
remains open.

## Profiler diagnosis

The pre-change M16/W2 full-Q+gate kernel was captured with:

```bash
ncu --section SpeedOfLight --csv --kernel-name regex:p32_window_ampere_kernel \
  --launch-skip 2 --launch-count 1 -- \
  python scripts/benchmark_qvq_p32_ampere.py --physical-gpu 0 --m-values 16 \
  --rates 2 --shapes full_q_gate --warmup 2 --iterations 1
```

Nsight Compute reported 71.50% memory throughput, 4.76% DRAM throughput,
75.96% L1/TEX throughput, and 50.54% compute throughput. The detailed capture
reported 56 registers/thread, 4,128 bytes static shared memory, zero spills,
56.25% theoretical occupancy, and 43.29% achieved occupancy. This classified
the path as on-chip-memory/instruction limited rather than HBM limited and
motivated the Marlin-style register fragments and reduced barrier frequency.
The matched post-change SpeedOfLight capture reduced profiled duration from
139,808 ns to 133,856 ns (`1.044x`) and elapsed cycles from 169,515 to 162,346.
L1/TEX remains the limiting unit at 82.92% while DRAM is only 5.00%, so the
next 2x-target experiments must reduce random on-chip level gathers or remove
more decode instructions without expanding the compact state representation.

## Failed and discarded experiments

| Experiment | Observation | Decision |
|---|---|---|
| WMMA shared operands with 16-byte/natural alignment | M1 failed with `cudaErrorMisalignedAddress` before comparison. | Rejected. All WMMA shared operands and stores now have explicit 32-byte alignment. |
| Double buffering without a post-MMA block barrier | Some low-occupancy cases passed, but denser split grids produced multi-unit output corruption. Fast warps could overwrite a stage still consumed by slower warps under independent thread scheduling. | Rejected and all timings discarded. Added `__syncthreads()` at the producer/consumer handoff. |
| Split counts above eight during the unsafe-buffer experiment | Long-K splits 9-16 showed increasing corruption before the producer/consumer barrier fix. | Rejected for that unsafe revision; the corrected kernel was revalidated separately before enabling wider M=1 scalar split waves. |
| Sixty-four-way split on short-K shapes | Correct in the 28-case M=1 matrix, but geomean latency was 0.065112 ms versus 0.064857 ms for thirty-two-way splitting; larger reduction work provided no net gain. | Rejected for short-K; retain thirty-two-way scalar splitting there. The long-K M1/M2 MLP-down exception is accepted separately. |
| Generic two-wave split heuristic | Correct on the first formal matrix but left substantial performance unused on long-K and wide-N shapes. | Replaced by measured Qwen3.8 splits plus a live-SM fallback for unknown shapes. |
| Cached 512 KiB state-to-FP16-pair LUT | Exact, but representative latency rose from 0.115-0.221 ms to 0.288-0.451 ms because random cache traffic cost more than compact PGC16 arithmetic. | Rejected and reverted. |
| Three-stage K16 pipeline without the post-MMA barrier | Exact, but M16 regressed 3-6% and M1 did not improve; the extra footprint/bookkeeping outweighed the removed barrier. | Rejected and reverted. |
| Eight-warp N128 CTA | Exact, but representative latency regressed 8-12% because the larger block reduced scheduling flexibility. | Rejected and reverted; retain four-warps/N64. |
| Four-K16/K64 staged group | Exact, but regressed 7-16% versus the accepted K32 group as the larger shared footprint dominated further barrier savings. | Rejected and reverted; retain two-K16/K32. |
| Full repository QVQ comparator build | Failed because unrelated YAQA translation units require cuBLAS/cuSPARSE developer headers absent from this local toolkit. | Benchmark builds the current `qvq_gemv_cuda.cu` alone. The GEMV file now uses the lightweight current-stream header and remains source-identical to production GEMV. |
| 256-thread/32-tile M=1 scalar CTA | Correct across the formal M=1 cases, but full-KV latency regressed to 0.058-0.070 ms versus 0.053-0.063 ms for the 128-thread route. | Rejected; retain 128 threads, four warps, and sixteen N16 tiles per CTA. |
| 64-thread/8-tile M=1 scalar CTA | Correct, but full-Q and attention timings were neutral-to-slower and full-KV W2 rose to about 0.055 ms; it did not offset the reduced per-CTA decode parallelism. | Rejected; retain the 128-thread route. |
| Scalar M=1 on K=17408 with the old eight-way wave | Correct, but W3.5 MLP-down rose to about 0.229 ms versus 0.196 ms for WMMA. | Superseded: after widening the long-K wave to 32, scalar M1 is now accepted at about 0.113 ms; do not reuse the old eight-way result. |
| Scalar M=4 on K=17408 with the 32-way wave | Correct, but the four-row scalar route was neutral-to-slower (about 0.159 ms geomean) than WMMA (about 0.158 ms) in the focused probe. | Rejected; retain WMMA for M4 and larger long-K projections. |
| Shared codebook copy | Exact, but the `__ldg` read-only path was consistently lower in the representative matrix and removes a 512-byte per-CTA copy. | Replaced by read-only levels; keep the experiment in history as the prior checkpoint. |
| Direct atomic M=1 split-K accumulation | Correct in the quick matrix and removed the separate reduction launch, but timings were statistically neutral (full-Q about 0.084-0.095 ms, full-KV about 0.053-0.062 ms, attention about 0.070-0.082 ms). Atomic accumulation also sacrifices deterministic summation order. | Rejected; retain deterministic partial-output reduction and do not count this as forward progress. |
| Scalar bank-mask hoisting | Exact, but the compiler already hoisted the row-group selector; the 12-case timing matrix was unchanged at the event-sample resolution. | Rejected as a source change; retain the simpler shared decode helper. |
| Direct global trellis loads for scalar M=1 | Exact in the target-GPU suite, but bypassing the coalesced `cp.async.cg` staging raised the representative full-Q latency to 0.072-0.093 ms and attention to about 0.053-0.057 ms versus the staged route's 0.066-0.074 ms and 0.048-0.055 ms. | Rejected; retain shared trellis staging. |
| Explicit `ld.global.nc.L1::evict_last` codebook loads | Exact and syntactically valid on sm_80, but the 12-case probe was unchanged or slower than `__ldg` at 0.066-0.093 ms for full-Q and 0.044-0.053 ms for the smaller shapes. | Rejected; retain the simpler `__ldg` read-only path. |
| 256-thread/32-tile M2 scalar CTA | Correct, but the representative full-Q/full-KV/attention probe rose to about 0.079/0.048/0.056 ms versus 0.073/0.036/0.047 ms for the 128-thread route. | Rejected; retain 128 threads and sixteen N16 tiles per CTA. |
| Two-tile-per-warp M4 scalar CTA | Correct after fixing the tile-base stride, but leaving half of each warp inactive raised full-Q to about 0.152 ms versus 0.090 ms for four tiles per warp. | Rejected; retain four active N16 tiles per warp. |
| M8 scalar row accumulator | Correct, but eight live FP32 rows raised full-Q/attention to about 0.151/0.088 ms versus 0.117/0.073 ms for WMMA partial rows. | Rejected; keep M8 on the tensor-core path. |
| Two-warps-per-tile M8 scalar cooperative mapping | Correct after fixing an initial tile-coverage bug, but reducing each CTA to two N16 tiles raised full-Q/attention to about 0.399/0.207 ms versus 0.117/0.073 ms for WMMA. | Rejected; preserve the four-warp WMMA CTA and tune only its split wave. |
| Four-K16/K64 WMMA stage for short-K M8/M16 | Correct, but the larger stage regressed representative W2 full-Q to about 0.126 ms (M8) and 0.140 ms (M16), versus about 0.117 ms and 0.119 ms for K32 staging. | Rejected; retain K32 staging and shape-specific split tuning. |
| Conditional final post-MMA barrier elision | Correct, but changing the unconditional handoff to a `has_next` branch regressed representative M16 full-Q to about 0.141 ms versus 0.120 ms; the altered control flow changed compiler scheduling. | Rejected and reverted; retain the unconditional producer/consumer barrier. |
| Forty-eight-way long-K wave for M4/M8/M16 | Correct, but MLP-down was slower than the accepted 32-way wave (about 0.159/0.163/0.171 ms versus 0.158/0.160/0.165 ms). | Rejected; keep 32-way for M4/M8/M16 and 128-way (M1)/96-way (M2) for long-K scalar work. |
| 256-way long-K wave for M1/M2 | Correct, but the reduction wave turned upward: M1/M2 MLP-down were about 0.105/0.121 ms versus about 0.100/0.114 ms for 128-way splitting. | Rejected; retain the 128-way M1 and measured 96-way M2 policies. |
| Four-K16 scalar stage for long-K MLP-down | Correct across W2-W3.5 at M1-M4. Focused geometric means improved 1.003x/1.033x/1.068x versus the new M1/M2/M4 controls, while the full 140-case matrix improved 1.012x/1.014x/1.010x/1.003x/1.000x for M1/M2/M4/M8/M16. | Accepted for the measured `(K,N)=(17408,5120)` M1-M4 path; WMMA shapes retain the two-K16 stage. |
| Eight-K16 scalar stage for long-K MLP-down | Correct, but W2 latency regressed to about 0.096/0.106/0.151 ms for M1/M2/M4 versus about 0.089/0.095/0.130 ms with four-K16 staging; the larger shared stage and unrolled body outweighed fewer handoffs. | Rejected; retain four-K16 scalar staging. |
| Four-CTA-per-SM WMMA launch bound | Correct, but representative M8/M16 full-Q W2 latency rose to 0.119/0.131 ms versus 0.114/0.119 ms under the unconstrained launch bound. | Rejected; the apparent occupancy gain did not translate to throughput. |
| Vectorized four-output split reducer | Correct, but it under-filled the small-output reducer (M1 MLP-down about 0.116 ms versus about 0.100 ms with one output per thread). | Rejected; retain the scalar reducer to preserve enough reduction blocks. |
| Eight-lane shared-activation broadcast | Correct, but replacing repeated shared loads with a packed-half2 shuffle made the scalar M1/M2 probes 1.5-2x slower (full-Q/MLP-down about 0.089/0.130 ms versus about 0.068/0.10 ms). | Rejected; the extra lane-control and shuffle cost outweighs shared-load reuse on sm_80. |
| SM80 `m8n8k4` M1-M4 route | Correct, but the smaller tensor-core instruction still required four K4 slices and extra pair shuffles; M1 full-Q was about 0.261 ms versus about 0.067 ms for the scalar route. | Rejected; retain the scalar M1/M2 and WMMA M4+ dispatch. |

## Reproduction

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-14ab23f1-a785-e9df-bbb5-215547154e3c
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=8 NINJAFLAGS=-j8 CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=2

python -m pytest -q tests/test_qvq_p32_ampere.py -s
python scripts/benchmark_qvq_p32_ampere.py --physical-gpu 0 --m-values 1 2 4 8 16 --warmup 10 --iterations 50
```
