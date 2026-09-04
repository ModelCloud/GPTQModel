# QVQ exact-P32 Ampere kernel progression

This ledger tracks the exact continuous-window P32 inference kernel for
A100-class `sm_80` GPUs. It records accepted forward progress and failed or
discarded experiments so later tuning does not repeat unsafe variants.

## Contract and target

- Source base: freshly fetched GitHub `origin/main` at `ab277a17` (tip after
  PR #86 merged).
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
   measured to overcome the old WMMA advantage. M4 also uses that four-K16
   stage on the measured short-K shapes; M1-M2 use up to thirty-two K splits,
   while M4 uses a measured 40-way wave on short-K shapes. Attention-out
   (K=6144, N=5120) retains a 24-way wave to reduce split-reduction overhead;
   M4 long-K and M8+ remain on the tensor-core path.
10. For M8 and M16, retain WMMA arithmetic but choose the K split by M/N shape:
    M8 uses 16-way splits for wide QKV/MLP projections and 32-way splits for
    small-N KV/attention/Z projections; M16 uses 16-way splits for the latter
    group and 32-way splits for full K/V.
11. M8 uses a compile-time eight-live-row WMMA specialization. It preserves
    the full `m16n8k16` arithmetic contract while removing runtime row-count
    masking from activation staging and output stores.
12. M2 short-K scalar projections use a measured three-K16 software stage,
    reducing barrier/commit overhead while retaining the two-K16 stage for M1
    and the four-K16 stage only where it was already proven.
13. Cache the immutable live SM count per CUDA device in the Python dispatch;
    this removes repeated driver-property queries from the timed auto-split
    path, which is material for sub-50-microsecond small-N projections.
14. For the full-row M16 full-Q, `N=5120`, `N=10240`, `N=6144`, `N=1024`,
    and `N=17408` projections, use a compile-time N-tile count in the WMMA
    path. The fixed Qwen3.8 shapes let trellis staging remove the per-vector
    N-bound predicate while preserving the K-bound check and the exact generic
    fallback for all other shapes. The `N=5120` case is shared by attention-out
    and MLP-down, while `N=17408` covers MLP-gate/up.
15. For M8 full-Q (`N=12288`), combine the compile-time eight-live-row path
    with a compile-time N-tile count. This removes both row and N predicates
    from the measured wide projection while retaining the generic M8 fallback.
16. For M8 attention-out and MLP-down (`N=5120`), use the same compile-time
    N-tile count with the eight-live-row path. The two shapes share the N tile
    geometry despite different K lengths.
17. For M8 linear-QKV (`N=10240`) and linear-Z (`N=6144`), use compile-time
    N-tile counts with the eight-live-row path after matched screens confirmed
    repeatable gains.
18. For M8 full-KV (`N=1024`) and MLP-gate/up (`N=17408`), use compile-time
    N-tile counts as well. This completes fixed-N dispatch coverage for every
    formal M8 projection shape.
19. For scalar M1 and M4, use the fixed-N launcher on the proven shape subset
    while retaining each row count's K-stage policy. M1 specializes the
    N=12288, 1024, 10240, and 17408 short-K projections; M4 specializes all
    formal shapes with its measured four-K16 stage. M2 remains on the cached
    generic scalar dispatch after a matched regression screen.
20. For M4 short-K projections, use a 40-way split wave except for
    attention-out, where 24-way remains faster. M1/M2 retain the 32-way
    short-K wave; this shape-specific policy fills more SMs on the four-row
    scalar route without changing the other row counts.
21. On fixed-N M8 shapes other than full-KV, load only the two live upper-row
    A submatrices with `ldmatrix.x2`. The lower A registers are explicitly
    zeroed to preserve the `m16n8k16` contract, while the matching lower FP32
    outputs remain transient. Full-KV retains `ldmatrix.x4`, which is faster
    once its small launch/reduction overhead dominates.
22. On every formal M16 route, specialize both N and K. Full-Q, full-KV,
    linear-QKV, linear-Z, and MLP-gate/up use `K=5120`; attention-out uses
    `K=6144`; MLP-down uses `K=17408`. The fixed K makes the activation row
    stride and K-tile geometry compile-time values; unknown K values retain
    the runtime fallback.
23. On M8, retain fixed `K=5120` only for full-KV and linear-QKV, where the
    matched repeat remains positive. Other M8 shapes keep runtime K after a
    mixed broad screen.

Future Ampere experiments should compare the generated instruction schedule,
register pressure, shared-memory bank behavior, and CTA swizzle against Marlin
as well as carrying forward architecture-independent lessons from the Hopper
kernel.

There are sixty WMMA device specializations: four transition widths
times full-M16, generic partial-row, compile-time M8 partial-row, and
compile-time M16 `N=12288`, `N=5120`, `N=10240`, `N=6144`, `N=1024`, and
`N=17408`
paths, plus the compile-time M8 `N=12288`, `N=5120`, `N=10240`, and `N=6144`
paths, plus the compile-time M8 `N=1024` and `N=17408` paths. The
scalar M1-M4 rows add four exact transition-width specializations, while one
runtime split reducer is shared by all rates.
Unknown shapes use a live-SM-derived fallback; the seven measured Qwen3.8-27B
shapes use recorded split counts without embedding the local 124-SM inventory.

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
- `artifacts/a100_p32_window/qwen38_mixed_p32_ampere_v14.json`
- `artifacts/a100_p32_window/qwen38_origin_main_492f1f58.json`
- `artifacts/a100_p32_window/qwen38_mixed_p32_ampere_v15_492f1f58.json`
- `artifacts/a100_p32_window/qwen38_m1_m2_stage3_492f1f58.json`
- `artifacts/a100_p32_window/qwen38_m1_cached_sm_492f1f58.json`
- `artifacts/a100_p32_window/qwen38_m2_cached_sm_492f1f58.json`
- `artifacts/a100_p32_window/qwen38_m4_cached_sm_492f1f58.json`
- `artifacts/a100_p32_window/qwen38_m8_cached_sm_492f1f58_retry.json`
- `artifacts/a100_p32_window/qwen38_m16_cached_sm_492f1f58.json`
- `artifacts/a100_p32_window/screen_m16_static_n_fullq.json`
- `artifacts/a100_p32_window/screen_m16_static_n_5120.json`
- `artifacts/a100_p32_window/screen_m16_static_n_10240_6144.json`
- `artifacts/a100_p32_window/screen_m16_static_n_1024.json`
- `artifacts/a100_p32_window/screen_m8_static_n_fullq.json`
- `artifacts/a100_p32_window/screen_m8_static_n_5120.json`
- `artifacts/a100_p32_window/screen_m8_static_n_10240_6144.json`
- `artifacts/a100_p32_window/screen_m8_static_n_1024.json`
- `artifacts/a100_p32_window/qwen38_m4_final_split40_6bc83e5a.json`
- `artifacts/a100_p32_window/screen_m1_m4_static_n_scalar_narrow.json`
- `artifacts/a100_p32_window/screen_m1_m4_static_n_selective.json`

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
`qwen38_mixed_p32_ampere_v14.json`, measured with the same 140 cases, 20
warmups, 100 CUDA-event iterations, and idle/foreign-process gates. Values are
geometric means of the 28 Ampere event medians for each M; planar timings are
not part of this target.

| M | New main geomean ms | Candidate geomean ms | Speedup vs fetched main |
|---:|---:|---:|---:|
| 1 | 0.066583 | 0.065077 | 1.023x |
| 2 | 0.072008 | 0.070520 | 1.021x |
| 4 | 0.084860 | 0.081901 | 1.036x |
| 8 | 0.092692 | 0.091843 | 1.009x |
| 16 | 0.096069 | 0.095498 | 1.006x |

The accepted v14 change specializes the Marlin-style scalar path for the
long-K MLP-down shape (K=17408, N=5120) at M1-M4 and uses four K16 tiles per
software stage for M4's measured short-K shapes. The proven two-tile WMMA stage
remains in place for other shapes and M8/M16. The complete matrix is currently
1.006-1.036x versus the fetched main, so the requested 2x target remains open.

## Current fetched-main checkpoint

The control is `qwen38_origin_main_492f1f58.json`, measured immediately after
fetching the merged PR #76 tip. The v15 candidate is
`qwen38_mixed_p32_ampere_v15_492f1f58.json`; both use the same 140-case matrix,
20 warmups, 100 CUDA-event iterations, and idle/foreign-process gates. Values
are geometric means of the 28 Ampere event medians for each M; planar timing is
excluded from this comparison.

| M | Fetched-main geomean ms | v15 geomean ms | Speedup vs fetched main |
|---:|---:|---:|---:|
| 1 | 0.064880 | 0.065866 | 0.985x |
| 2 | 0.070273 | 0.071145 | 0.988x |
| 4 | 0.081726 | 0.082350 | 0.992x |
| 8 | 0.092139 | 0.089172 | 1.033x |
| 16 | 0.095486 | 0.096163 | 0.993x |
| All 140 cases | 0.080007 | 0.080157 | 0.998x |

The accepted v15 change is the compile-time M8 partial-row specialization.
Matched W2 probes improved full-Q, attention-out, linear-QKV, and MLP gate/up
by approximately 5.7%, 3.2%, 4.2%, and 3.3%, respectively, with all 140 cases
remaining within the exactness gate. The one-pass all-case aggregate is within
timing noise because M8 is only one of five row counts; the cumulative 3%
versus-main target remains open and further row-count-specific work continues.

The next focused checkpoint is `qwen38_m1_m2_stage3_492f1f58.json`. It covers
all 28 cases for M1 and M2. The measured M2 geomean is `0.069539 ms` versus
`0.070273 ms` for the fetched-main control (`1.011x`); M1 is effectively flat
at `1.001x`, so the three-stage policy is narrowed to M2 rather than applied
broadly.

## Isolated row-count checkpoint

The cache checkpoint was validated as separate 28-case runs per row count to
avoid the clock/scheduling excursions seen in one long 140-case process. The
comparison remains against `qwen38_origin_main_492f1f58.json` and excludes
planar timing.

| M | Fetched-main geomean ms | Cached-dispatch geomean ms | Speedup vs fetched main |
|---:|---:|---:|---:|
| 1 | 0.064880 | 0.063994 | 1.014x |
| 2 | 0.070273 | 0.067885 | 1.035x |
| 4 | 0.081726 | 0.081302 | 1.005x |
| 8 | 0.092139 | 0.088444 | 1.042x |
| 16 | 0.095486 | 0.094271 | 1.013x |
| Equal-weight all M | 0.080007 | 0.078309 | 1.022x |

All isolated runs passed the exactness contract; the M8 retry is the artifact
used for the table after an earlier contaminated process was discarded.

The focused M16 fixed-N screen is `screen_m16_static_n_fullq.json`. For the
four full-Q rate cases, compile-time `N=12288` staging lowers the geomean from
0.120576 ms to 0.117728 ms (`1.028x`, or 2.84%) versus the matching rows in
`qwen38_origin_main_492f1f58.json`. The complete 22-case exactness suite still
passes; this is a shape-local checkpoint and does not yet establish a 3%
all-row-count aggregate.

The follow-on `N=5120` screen covers M16 attention-out and MLP-down. It lowers
their geomeans from 0.067327/0.165489 ms on the fetched-main control to
0.065536/0.160763 ms (`1.027x`/`1.029x`) with maximum error `<= 9.6e-5`.
The complete exactness suite is rerun before this checkpoint is retained.

The companion `N=10240`/`N=6144` screen covers M16 linear-QKV and linear-Z.
Their geomeans fall from 0.103680/0.068096 ms to 0.100480/0.066304 ms
(`1.032x`/`1.027x`) against the same fetched-main rows. The exactness suite
passes 22/22 after adding these dispatches.

The remaining M16 full-KV screen (`N=1024`) lowers its geomean from 0.045312
ms to 0.043008 ms (`1.048x`) versus fetched main, with maximum error
`<= 1.8e-5`; the 22-case exactness suite remains green.

The M8 full-Q fixed-N screen lowers its four-rate geomean from 0.115712 ms on
fetched main to 0.110592 ms (`1.046x`), with maximum error `<= 3.5e-5`.

The M8 `N=5120` screen lowers attention-out and MLP-down geomeans from
0.065280/0.159744 ms to 0.062848/0.153344 ms (`1.039x`/`1.042x`) versus
fetched main. Maximum error is `<= 7.3e-5`, and the 22-case exactness suite
passes after the dispatch addition.

The M8 `N=10240`/`N=6144` screen lowers linear-QKV and linear-Z geomeans from
0.098304/0.065280 ms to 0.094464/0.062720 ms (`1.041x`/`1.041x`) versus
fetched main, with maximum error `<= 2.9e-5`; exactness remains 22/22.

The final M8 full-KV (`N=1024`) screen lowers its geomean from 0.045312 ms to
0.042496 ms (`1.066x`) versus fetched main, with maximum error `<= 1.8e-5`.
This completes the fixed-N M8 shape set with the exactness suite still at
22/22.

The final selective scalar fixed-N screen covers all 28 M1 and M4 cases. M1
lowers its geomean from 0.064880 ms on fetched main to 0.063510 ms (`1.022x`),
and M4 lowers 0.081726 ms to 0.080089 ms (`1.020x`). Maximum error is
`<= 4.8e-5`; M2 and M3 remain on their prior dispatches.

The final M4 auto-policy artifact applies 40-way splitting to six short-K
shapes and retains 24-way for attention-out. Its geomean is 0.078690 ms
(`1.039x` versus fetched main); all 28 cases remain within the exactness
gate.

Combining the clean isolated row-count artifacts gives the current cumulative
checkpoint below. The comparator is the fetched `origin/main` control, not the
planar oracle kernel.

| M | Main geomean ms | Candidate geomean ms | Speedup vs fetched main |
|---:|---:|---:|---:|
| 1 | 0.064880 | 0.063510 | 1.022x |
| 2 | 0.070273 | 0.067885 | 1.035x |
| 4 | 0.081726 | 0.078690 | 1.039x |
| 8 | 0.092139 | 0.088308 | 1.043x |
| 16 | 0.095486 | 0.092902 | 1.028x |
| All 140 cases | 0.080007 | 0.077431 | 1.033x |

This exceeds the requested cumulative 3% improvement while preserving the
22/22 exactness result and the documented M2/M3 scalar rejection.

## Post-merge origin/main baseline

PR #78 is merged at `origin/main` commit `1fa22740`. Fresh isolated controls
were captured from that exact tip as `qwen38_newmain_m{1,2,4,8,16}_1fa22740.json`
using the standard 20-warmup/100-iteration CUDA-event protocol and idle gate.
Their Ampere geomeans are 0.062931, 0.068251, 0.078186, 0.087721, and
0.092700 ms for M1/M2/M4/M8/M16. New experiments in this cycle must beat these
controls; planar timing remains oracle context only.

## Dispatch checkpoint: M1 attention-out fixed-N scalar route

Commit `083ec0eb` dispatches the formal M1 attention-out shape
`(K,N)=(6144,5120)` through the existing compile-time-N scalar launcher. The
focused four-rate artifact `exp_v10_m1_staticn_attention.json` is exact and
measures a 0.050733 ms geomean versus 0.051311 ms for the fresh M1 control,
or `1.011x` versus `origin/main` (planar timing is excluded). The apparent
`_auto_split_count` early-return issue was separately tested and rejected as a
red herring; the M-specific caller overrides already execute after the helper.

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
| Four-K16 scalar stage on every short-K M1-M4 shape | Correct, but the six-shape W2-W3.5 probe was neutral-to-slower for M1 (0.998x) and M2 (1.001x), despite a 1.022x M4 gain. | Narrowed to M4 short-K shapes; M1-M2 short-K paths retain two-K16 staging. |
| Eight-K16 scalar stage for long-K MLP-down | Correct, but W2 latency regressed to about 0.096/0.106/0.151 ms for M1/M2/M4 versus about 0.089/0.095/0.130 ms with four-K16 staging; the larger shared stage and unrolled body outweighed fewer handoffs. | Rejected; retain four-K16 scalar staging. |
| Four-CTA-per-SM WMMA launch bound | Correct, but representative M8/M16 full-Q W2 latency rose to 0.119/0.131 ms versus 0.114/0.119 ms under the unconstrained launch bound. | Rejected; the apparent occupancy gain did not translate to throughput. |
| M16 full-Q split-wave 4 or 6 | Correct, but both alternatives measured about 0.128 ms for W2 versus 0.119 ms with the tuned five-way wave. | Rejected; retain the five-way M16 full-Q split. |
| Vectorized four-output split reducer | Correct, but it under-filled the small-output reducer (M1 MLP-down about 0.116 ms versus about 0.100 ms with one output per thread). | Rejected; retain the scalar reducer to preserve enough reduction blocks. |
| Eight-lane shared-activation broadcast | Correct, but replacing repeated shared loads with a packed-half2 shuffle made the scalar M1/M2 probes 1.5-2x slower (full-Q/MLP-down about 0.089/0.130 ms versus about 0.068/0.10 ms). | Rejected; the extra lane-control and shuffle cost outweighs shared-load reuse on sm_80. |
| SM80 `m8n8k4` M1-M4 route | Correct, but the smaller tensor-core instruction still required four K4 slices and extra pair shuffles; M1 full-Q was about 0.261 ms versus about 0.067 ms for the scalar route. | Rejected; retain the scalar M1/M2 and WMMA M4+ dispatch. |
| Adaptive vectorized split reducer | Correct, but large-output rows were neutral at the event-sample resolution and small-output cases lost reducer parallelism; no repeatable full-matrix gain. | Rejected; retain the scalar deterministic reducer. |
| Ordinary/explicit `.ca` level loads | Correct, but matched probes were neutral-to-slower than the `__ldg` read-only path. | Rejected; retain `__ldg` for the 512-byte codebook. |
| M8 dead-row staging elision | Correct, but removing the eight inactive activation rows was slower or neutral versus the compile-time-row specialization alone. | Rejected; retain zero-filled inactive rows for stable pipeline scheduling. |
| Three-K16 scalar stage for M1 | Correct and near-neutral in the full M1 subset (`1.001x`), without a repeatable gain over the two-K16 stage. | Narrowed to M2, where the matched subset measured `1.011x`; M1 retains two-K16 staging. |
| Runtime-N WMMA staging on M16 fixed-N shapes | Correct, but the generic `n_tile < n_tiles` predicate remains on every staged vector even though the measured shapes are fixed at `N=12288`, `N=5120`, `N=10240`, `N=6144`, or `N=1024`. | Replaced by compile-time N specializations, which lower the focused M16 full-Q, attention-out, MLP-down, linear-QKV, linear-Z, and full-KV geomeans by 2.84%, 2.73%, 2.94%, 3.19%, 2.70%, and 4.81%; the remaining N values stay on the generic path pending matched screens. |
| M8 full-Q compile-time N tile count | Correct, but the generic M8 row specialization still checked the fixed `N=12288` tile bound for every staged vector. | Accepted the combined M8 row/N specialization after a 4.63% matched full-Q gain; other M8 N values remain generic pending screens. |
| M8 attention-out/MLP-down compile-time N tile count | Correct, but the generic M8 row specialization still checked the fixed `N=5120` tile bound for every staged vector. | Accepted the combined M8 row/N specialization after 3.87% and 4.18% matched gains; other M8 N values remain generic pending screens. |
| M8 linear-QKV/linear-Z compile-time N tile count | Correct, but the generic M8 row specialization still checked the fixed `N=10240`/`N=6144` tile bounds for every staged vector. | Accepted after 4.07%/4.08% matched gains; the M8 full-KV (`N=1024`) screen followed separately. |
| M8 full-KV compile-time N tile count | Correct, but the generic M8 row specialization still checked the fixed `N=1024` tile bound for every staged vector. | Accepted after a 6.62% matched gain; all seven formal M8 shapes now use fixed-N paths. |
| Scalar M1/M4 fixed-N launcher | Correct, but the scalar stage still checked the fixed output-tile bound at every warp. | Accepted for M1/M4 after 2.16%/2.04% matched gains versus fetched main; M2/M3 use the generic scalar launcher pending a better schedule. |
| Scalar fixed-N launcher on M2/M3 | Correct, but the larger specialized body regressed M2 by 1.10% versus its cached-dispatch candidate (small-N and linear-Z were the largest losses); M3 was not part of the formal target. | Rejected for M2/M3; keep the compile-time launcher only on M1/M4. |
| First v13 lock-free full-matrix sample | Five consecutive early-M2 rows jumped by 3.5-7.5x while their planar controls also jumped, then both paths returned to normal. The unaffected 135 rows improved 0.99% versus main. | Discard the aggregate as a transient-contaminated run; retain `qwen38_v13_lockfree_all_29a69382.json` only as a diagnostic and rerun the complete idle-gated matrix. |
| Four-accumulator split reducer | The matched M1 full-KV probe improved 0.68%, but the clean 140-case refresh regressed from 0.074518 ms to 0.074948 ms, with M1-M4 all slower. | Rejected and reverted; retain the single-accumulator deterministic reducer. The false-positive focused and full results remain in the v13 artifacts. |
| Reusing precomputed tensor data pointers in every CUDA launch branch | Behavior and outputs were unchanged, but the matched 300-iteration M1 full-KV geomean regressed from 0.034557 ms to 0.035309 ms (2.18%). | Rejected and restored to exact `ab277a17` source; repeated `data_ptr()` extraction is not the limiting enqueue cost. |
| 128-thread split reducer | An isolated full-KV run appeared faster, but the complete M1 subset with recorded tuner plans regressed from 0.058178 ms to 0.058862 ms (1.18%). At the same split 64, all four full-KV rates were 3-9% slower. | Rejected and restored to 256 threads; extra reducer blocks do not offset lower per-block throughput. |
| Warp-leader split-bound division | Replacing each thread's uniform K-range divisions with lane-zero division and two shuffles was neutral for M1 full-KV and regressed M16 full-KV from 0.034293 ms to 0.035069 ms (2.26%). | Rejected and reverted; retain the compiler's uniform runtime division. |
| Host-computed alternate-bank mask | Passing the uniform decoded mask instead of deriving it per thread left matched M1 full-KV exactly unchanged at 0.034557 ms; M16 Q/KV results mixed one-tick gains and losses. | Rejected and reverted; retain the simpler bank-ID kernel interface. |
| 512-thread reducer for M8/M16 | M16 regressed from 0.089626 ms to 0.096331 ms (7.48%). At the same split 32, full-KV lost 3-12% and three MLP-down rates lost 73-79%. | Rejected and reverted; 256 threads remain the best broad reducer geometry. |
| Compile-time split-32 reducer | Fully unrolling the common 32-way serial sum preserved exact accumulation order, but M16 full-KV regressed from 0.034293 ms to 0.034816 ms (1.53%); MLP-down was effectively neutral. | Rejected and reverted; runtime loop control is cheaper than the enlarged unrolled reducer on sm_80. |
| Bank-selector hoist across every scalar and WMMA route | The 60-case full-Q/full-KV/MLP-down screen was neutral overall (-0.09%): M8 improved 2.53%, but M1, M2, M4, and M16 regressed by 1.32%, 0.81%, 0.23%, and 0.68%. | Rejected broadly and narrowed to repeatedly positive M8 fixed-N routes; scalar and full-row kernels retain main's decode path. |
| Single allocation for partials and output | Placing the output in the final plane of one `(splits+1)` allocation remained exact, but constructing the returned tensor view raised full-KV latency from 0.035362 ms to 0.038750 ms (9.58%). | Rejected and reverted; retain two direct allocator requests for the split workspace and returned output. |
| Sequential M16 bank-mask hoist | Limiting each selector mask's live range to its two shared decodes remained exact, but the 20-case fixed-N M16 geomean regressed from 0.096235 ms to 0.096689 ms (0.47%); every shape slowed. | Rejected and reverted; keep bank-mask hoisting restricted to the validated M8 routes. |
| Reusing one FP32 `TensorOptions` value | Avoided constructing the same options expression twice, but after excluding one slow control outlier the remaining 19 full-KV cases regressed by 4.98%. | Rejected and reverted; retain the two inline allocator option expressions. |

## Post-merge origin/main baseline (v11)

This cycle starts from the newly fetched and merged `origin/main` tip
`8aa265e0fb11edc61e4d4e143bb4cdf2f6aa651c` (the merge of PR #80). The five
tracked artifacts `qwen38_newmain_m{1,2,4,8,16}_8aa265e0.json` use the same
20-warmup/100-iteration CUDA-event protocol and idle gate. The target metric
is Ampere event geomean latency versus this control; planar-oracle timing is
not included.

| M | Cases | Ampere geomean (ms) | Worst max abs |
|---:|---:|---:|---:|
| 1 | 28 | 0.068855 | 3.052e-5 |
| 2 | 28 | 0.072653 | 3.052e-5 |
| 4 | 28 | 0.080292 | 3.052e-5 |
| 8 | 28 | 0.089952 | 3.052e-5 |
| 16 | 28 | 0.098378 | 3.052e-5 |

The exactness suite passes 22/22 cases on the same checkout. Subsequent
changes are recorded as separate commits only after a matched benchmark shows
repeatable forward progress; failed experiments remain untracked artifacts and
are summarized below rather than being mixed into the control.

### M1 fixed-N dispatch extension

The first v11 progression extends the existing M1 fixed-N scalar dispatch to
the remaining formal `linear_z` projection `(K,N)=(5120,6144)`. A matched
40-warmup/200-iteration screen measured a 0.051132 ms control geomean versus
0.046074 ms with the compile-time-N body (`1.110x`); the full 28-case M1
refresh remains exact and measures 0.064973 ms by median geomean. This is a
dispatch-only change: all other shapes retain their previous route.

The second progression applies the same compile-time-N scalar body to M2
full-KV `(K,N)=(5120,1024)`, while retaining M2's measured three-tile stage.
The focused four-rate screen improves 0.044529 ms to 0.043002 ms (`1.036x`),
and the full M2 refresh improves 0.069963 ms to 0.069192 ms (`1.011x`) by
median geomean, with exact outputs.

The first M2 screen tried fixed-N dispatch for full-Q with the same stage. It
was correct but measured 0.080950 ms versus the 0.080699 ms control (`0.997x`),
so that route was rejected and remains on the generic launcher.

The third progression narrows the M4 full-KV reduction wave to 32 slices on
the 124-SM A100. The matched `(K,N)=(5120,1024)` screen measured 0.036605 ms
versus 0.045014 ms at the previous 40-way policy (`1.230x`); the full M4
refresh improves 0.079040 ms to 0.078673 ms (`1.005x`) by median geomean,
again with exact outputs. The 16- and 24-way alternatives reached 0.037868
and 0.040183 ms; wider waves were not retained after the matched probes. Only
32 is enabled for this shape.

The fourth progression narrows M8 full-KV to 24 slices. Its clean four-rate
screen measures 0.034045 ms versus 0.043262 ms at the 32-way control (`1.271x`)
with exact outputs. Replacing only those four rows lowers the five-M
140-case median geomean from 0.078688 ms to 0.077580 ms (`1.014x`); the other
M8 rows retain the post-merge control timings.

A follow-up wider-wave screen supersedes that provisional setting: 48 slices
measure 0.033784 ms across W2-W3.5 versus 0.034045 ms at 24 (`1.008x`). The
M8 full-KV policy is therefore 48; the in-process launch-plan key version is
bumped so stale entries cannot mask this update during a long-lived process.

### M/K/N launch-plan autotuning

The Python dispatch now runs a first-use tuner by default for new shapes. It
benchmarks up to 12 split waves around the measured fallback on the active CUDA
stream; the selected plan is keyed by
integer CUDA device index, dtype, M, K, N, transition bits, and bank variant.
Entries are memoized only in the current process; no autotune data is read from
or written to disk while the kernel is under active development.
Set `QVQ_AMPERE_AUTOTUNE=0` for the zero-overhead measured/static fallback.
Tuning can be made shorter or broader with `QVQ_AMPERE_AUTOTUNE_WARMUP`,
`QVQ_AMPERE_AUTOTUNE_ITERATIONS`, and `QVQ_AMPERE_AUTOTUNE_CANDIDATES`; clear
stale plans with `clear_qvq_ampere_autotune_cache()`.
Cold CUDA-graph capture uses the measured fallback without memoizing it because
event timing and host synchronization are illegal during capture; shapes tuned
before capture continue to use their cached in-process plan.

The probe budget was validated against an exhaustive comparison of the full
bounded family. A six-probe control chose split 16 for an unseen SM80 shape
K=8192,N=3072,M=1, while explicit timing found split 64 at 0.064512 ms versus
0.069632 ms (`1.079x`). With the 12-probe default, the tuner selected split 64
for that same shape. On the known M1 full-KV shape (K=5120,N=1024), the 12-probe
tuner selected split 96; repeated screens placed splits 64, 96, and 128 within
0.001024 ms, so no hand-tuned-only candidate is assumed to be optimal.

### Post-merge v12 baseline

PR #82 merged as `3bb797de`. The next optimization window uses that exact
`origin/main` tip as the Ampere-kernel control; planar timings remain diagnostic
only and are excluded from improvement calculations. The default-on,
in-process autotuner was enabled for both the control and all candidates.

The clean 20-warmup/100-iteration, 140-case run is stored in
`artifacts/a100_p32_window/qwen38_newmain_all_3bb797de.json`. Ampere median
latency geomeans are 0.074501 ms (M1), 0.079266 ms (M2), 0.086971 ms (M4),
0.095925 ms (M8), and 0.099725 ms (M16), with an all-case geomean of
0.086750 ms. The maximum absolute error across the matrix is 0.000080109.

The first v12 progression removes CUDA device-property queries and string
construction from every process-local autotune cache hit. Because the plan is
never persisted, a tuple containing the tensor device, dtype, M, K, N, rate,
and bank variant is sufficient. On M1 full-KV, the same selected split 64 now
measures 0.047358 ms by four-rate geomean versus 0.060662 ms in the merged-main
matrix (`1.281x`). An explicit split-64 control measures 0.032627 ms, showing
that further Python cache-hit overhead remains available to remove. Exactness
passes 27/27; the focused result is stored in
`artifacts/a100_p32_window/v12_m1_fullkv_fastkey.json`.

The second progression makes a hot autotune hit return directly to the CUDA
operator instead of recomputing the static shape policy and re-entering the
tuning helper. The same M1 full-KV screen falls again from 0.047358 ms to
0.038390 ms (`1.234x` over the first progression and `1.580x` over merged
main), with the selected plans and exact outputs unchanged. The result is in
`artifacts/a100_p32_window/v12_m1_fullkv_fast_hit.json`.

The five-M refresh at `09062aa4` confirms that the cache-hit fixes are broad:
M1 improves from 0.074501 ms to 0.060175 ms (`1.238x`), M2 from 0.079266 ms
to 0.065976 ms (`1.201x`), M4 from 0.086971 ms to 0.075249 ms (`1.156x`),
M8 from 0.095925 ms to 0.087640 ms (`1.095x`), and M16 from 0.099725 ms
to 0.091303 ms (`1.092x`). Across all 140 Ampere cases the geomean falls
from 0.086750 ms to 0.075110 ms: `1.155x`, or 13.42% lower latency versus
merged main. The refresh is stored in
`artifacts/a100_p32_window/qwen38_v12_cached_hit_all_09062aa4.json`.

The third progression keys the process-local cache by the stable integer CUDA
device index rather than a `torch.device` object. A two-million-lookup host
microbenchmark lowers key construction plus dictionary lookup from 289.9 ns to
250.8 ns (`1.156x`). The matched M1 full-KV GPU retry remains within timer
resolution at 0.038912 ms, with the same split-64 plans and exact outputs; it is
stored in `artifacts/a100_p32_window/v12_m1_fullkv_int_device_key_retry.json`.

### Post-merge v13 baseline

PR #85 merged as `db785848`. This optimization window uses that exact
`origin/main` tip as its Ampere-kernel control. Planar measurements remain
diagnostic and are not used in the improvement calculation. Default-on,
process-memory-only autotuning is enabled for the control and candidates.

The clean 20-warmup/100-iteration, 140-case run is stored in
`artifacts/a100_p32_window/qwen38_newmain_all_db785848.json`. Ampere median
latency geomeans are 0.061476 ms (M1), 0.066533 ms (M2), 0.075695 ms (M4),
0.088012 ms (M8), and 0.091889 ms (M16), with an all-case geomean of
0.075809 ms. The maximum absolute error across the matrix is 0.000080109.

The first v13 progression removes the Python reentrant lock from process-local
autotune cache hits. Cold calls still acquire the lock and recheck the entry
before timing, so concurrent misses tune once. In a matched 20-warmup,
300-iteration M1 full-KV screen, the four-rate latency geomean falls from
0.040170 ms with the lock to 0.037096 ms without it (`1.083x`), with exact
outputs. The candidate and control are stored in
`artifacts/a100_p32_window/v13_m1_fullkv_lockfree.json` and
`artifacts/a100_p32_window/v13_m1_fullkv_locked_control.json`.

The second progression reads the default-on autotune setting once per
process-local cache lifetime instead of querying `os.environ` on every launch.
Calling `clear_qvq_ampere_autotune_cache()` refreshes the setting, preserving
an explicit runtime opt-out without charging cache hits for it. A host-only
stub launch drops from 2.78 us on merged main to 1.66 us with both v13 dispatch
changes. The matched M1 full-KV geomean improves again from 0.037096 ms to
0.036836 ms (`1.007x`), with exact outputs; the result is stored in
`artifacts/a100_p32_window/v13_m1_fullkv_cached_env.json`.

The third progression represents M and K directly with the tensor's immutable
`torch.Size` in the in-memory key, while retaining the integer CUDA device
index, dtype, N, rate, and bank variant. It also maps the four supported P32
rates directly on the hot path and removes a redundant cached-split clamp; the
cold and uncommon-rate paths retain full validation. The host stub launch
falls from 1.66 us to 1.24 us. The matched M1 full-KV screen improves from
0.036836 ms to 0.036605 ms (`1.006x`) with exact outputs, stored in
`artifacts/a100_p32_window/v13_m1_fullkv_shape_key.json`.

The clean five-M refresh at `90a686cf` measures 0.060537 ms (M1), 0.065913 ms
(M2), 0.075263 ms (M4), 0.087170 ms (M8), and 0.091496 ms (M16). Its 140-case
geomean is 0.075140 ms, `1.009x` or 0.88% lower latency than the `db785848`
Ampere control. The exact result is stored in
`artifacts/a100_p32_window/qwen38_v13_shape_key_all_90a686cf.json`.

The fourth progression removes a remaining hot-path device query inside the
CUDA operator. Compute capability is immutable for the process lifetime, so
the first launch records it in a lock-free array indexed by integer CUDA device
index and later launches perform only a relaxed atomic load. The matched M1
full-KV screen improves from 0.036605 ms to 0.036334 ms (`1.007x`) with exact
outputs, stored in
`artifacts/a100_p32_window/v13_m1_fullkv_cached_capability.json`.

The corresponding five-M refresh measures 0.059416 ms (M1), 0.065454 ms (M2),
0.074897 ms (M4), 0.086870 ms (M8), and 0.090808 ms (M16). Across all 140
cases, latency falls from 0.075809 ms on `db785848` to 0.074518 ms: `1.017x`,
or 1.70% lower. The result is stored in
`artifacts/a100_p32_window/qwen38_v13_cached_capability_all_c23a15f0.json`.

The attempted fifth progression exposed split-reduction load-level parallelism
with four independent FP32 accumulators. Its M1 full-KV probe improved from
0.036334 ms to 0.036086 ms, but that result did not generalize: the clean
140-case geomean regressed from 0.074518 ms to 0.074948 ms, with M1-M4 all
slower. The experiment is reverted. Its focused and full diagnostic results
are stored in `artifacts/a100_p32_window/v13_m1_fullkv_reducer_ilp4.json` and
`artifacts/a100_p32_window/qwen38_v13_reducer_ilp4_all_09515686.json`.

The fifth accepted progression caches the resolved `torch.ops` callable,
retains already-typed integers in the M/K/N plan key, and defers the explicit
CUDA-input check until a real plan-cache miss. Full validation still occurs in
the CUDA operator on every launch. The Python stub path falls from 1.24 us to
0.99 us, while the matched M1 full-KV geomean falls from 0.036334 ms to
0.032760 ms (`1.109x`) with identical numerical error. The result is stored in
`artifacts/a100_p32_window/v13_m1_fullkv_cached_op.json`.

The final clean refresh at `eee8ad01` clears the cumulative target. Ampere
median latency geomeans fall from 0.061476 ms to 0.059119 ms for M1 (3.83%),
0.066533 ms to 0.064939 ms for M2 (2.40%), 0.075695 ms to 0.074193 ms for M4
(1.98%), 0.088012 ms to 0.086353 ms for M8 (1.88%), and 0.091889 ms to
0.089759 ms for M16 (2.32%). Across all 140 cases, the geomean falls from
0.075809 ms on fetched `db785848` main to 0.073925 ms: `1.025x`, or 2.486%
lower latency. Maximum absolute error remains 0.000080109. Planar timings are
excluded from every improvement figure. The result is stored in
`artifacts/a100_p32_window/qwen38_v13_cached_op_all_eee8ad01.json`.

### Post-merge v14 baseline

PR #86 merged as `ab277a17`. This optimization window uses that exact
`origin/main` tip as its Ampere-kernel control, with default-on in-process
autotuning for both controls and candidates. Planar timings remain diagnostic
and are excluded from all improvement calculations.

The clean 20-warmup/100-iteration, 140-case control is stored in
`artifacts/a100_p32_window/qwen38_newmain_all_ab277a17.json`. Ampere median
latency geomeans are 0.057948 ms (M1), 0.064170 ms (M2), 0.073818 ms (M4),
0.086108 ms (M8), and 0.089626 ms (M16), with an all-case geomean of
0.073316 ms. Maximum absolute error is 0.000080109.

The first v14 progression hoists the two bank-selector masks shared by each
M8 WMMA lane's four decoded pairs. It is enabled only for the five fixed-N
routes that improved in repeated screens: full-Q, attention-out, linear-QKV,
linear-Z, and MLP-down. Full-KV, MLP-gate, scalar M1-M4, and full-row M16 keep
the original decode path. An immediate matched 20-warmup/300-iteration A/B
reduces the enabled 20-case Ampere geomean from 0.091589 ms to 0.090367 ms
(`1.014x`, 1.335% lower latency); every enabled shape improves by 0.93-1.62%.
The control and candidate are stored in
`artifacts/a100_p32_window/v14_m8_bank_selector_matched_control.json` and
`artifacts/a100_p32_window/v14_m8_bank_selector_matched_candidate.json`.
Planar timings are excluded from these comparisons.

The second progression removes a redundant intermediate CUDA launch-error
poll from split plans. The main kernel and reducer are submitted to the same
stream, then the existing post-reducer check validates the two-launch
sequence; the single-kernel path retains its immediate check. In a matched
140-case A/B, the Ampere geomean falls from 0.073926 ms to 0.073650 ms
(`1.004x`, 0.373% lower). M1, M2, M4, M8, and M16 improve by 0.55%, 0.18%,
0.17%, 0.39%, and 0.58%, respectively. The matched control and candidate are
stored in
`artifacts/a100_p32_window/qwen38_v14_launch_poll_matched_control_all.json`
and `artifacts/a100_p32_window/qwen38_v14_launch_poll_all.json`. Planar
timings are excluded.

The third progression writes each naturally aligned adjacent FP32 output pair
with one `float2` store on the WMMA M8/M16 routes. Scalar M1-M4 retains its
original stores after the broad screen was neutral-to-slower there; the
compile-time `N=1024` path also retains scalar stores. In the matched complete
M8/M16 A/B, the 56-case Ampere geomean falls from 0.087619 ms to 0.085996 ms
(`1.019x`, 1.852% lower). M8 improves 1.593% and M16 improves 2.110%.
Artifacts are stored in
`artifacts/a100_p32_window/v14_wmma_float2_matched_control.json`,
`artifacts/a100_p32_window/v14_wmma_float2_matched_candidate.json`, and
`artifacts/a100_p32_window/v14_wmma_float2_selective_all.json`. Planar
timings remain excluded.

The fourth progression stages fixed-N bank IDs with aligned packed loads on
the routes where the instruction reduction pays for itself: one `uint4` load
per K stage on scalar M4 and one `uint32_t` load per K stage on full-row M16.
M1, M2, M8, dynamic/tail paths, and `N=1024` retain byte loads. The initial
broad screen found M1 and M8 regressions and a full-KV regression, so those
variants were rejected rather than averaged into the result. In the immediate
matched 20-warmup/300-iteration 48-case A/B, all twelve enabled M/shape groups
improve: M4 falls from 0.084237 ms to 0.083490 ms (0.895%), M16 falls from
0.103279 ms to 0.101781 ms (1.472%), and their combined geomean falls from
0.093273 ms to 0.092183 ms (`1.012x`, 1.183% lower latency). Exactness passes
28/28. The broad diagnostic, narrowed screen, and matched results are stored
in `artifacts/a100_p32_window/v14_packed_bank_stage_screen.json`,
`artifacts/a100_p32_window/v14_packed_bank_selective_all.json`,
`artifacts/a100_p32_window/v14_packed_bank_matched_control.json`, and
`artifacts/a100_p32_window/v14_packed_bank_matched_candidate.json`. Planar
timings remain excluded.

The fifth progression revisits M2 fixed-N specialization together with packed
bank-ID staging. Fixed N alone had previously regressed M2 by 1.10%; combining
it with one aligned `uint4` bank-ID load per K stage reverses that result.
Across the matched 24 non-KV cases the Ampere geomean falls from 0.071618 ms
to 0.070870 ms (1.055%), with all six shape groups non-regressing. A separate
500-iteration full-KV A/B falls from 0.035574 ms to 0.034808 ms (2.199%).
Together the complete 28-case M2 geomean falls from 0.064805 ms to 0.064026 ms
(`1.012x`, 1.218% lower latency), and exactness passes 28/28. The four matched
artifacts are
`artifacts/a100_p32_window/v14_m2_static_packed_control.json`,
`artifacts/a100_p32_window/v14_m2_static_packed_candidate.json`,
`artifacts/a100_p32_window/v14_m2_fullkv_packed_control.json`, and
`artifacts/a100_p32_window/v14_m2_fullkv_packed_candidate.json`.

The first post-progression 140-case refresh is diagnostic only: six M1/M2
samples suffered isolated 2.6-4.0x timing spikes despite the exclusivity gate,
so its aggregate is invalid and is not used for the cumulative comparison.
The unaffected M16 slice was 2.20% faster than fetched main. The complete
failed refresh is retained in
`artifacts/a100_p32_window/qwen38_v14_packed_bank_all_ec5b2e3e.json` so the
anomaly is visible rather than silently discarded.

Two final broad host/store experiments were rejected. Pairing every M2 scalar
output store regressed the 28-case geomean by 0.356%, and narrowing it to the
initially promising linear-QKV/MLP-gate routes still regressed the immediate
500-iteration A/B by 0.093%. Removing the release-build launch-status poll
also failed to generalize: the matched 20-case full-KV geomean regressed
0.333%, with M1, M4, and M16 non-positive. The source is restored after both
experiments. Diagnostics are retained in
`artifacts/a100_p32_window/v14_m2_float2_candidate.json`,
`artifacts/a100_p32_window/v14_m2_selective_float2_control.json`,
`artifacts/a100_p32_window/v14_m2_selective_float2_candidate.json`,
`artifacts/a100_p32_window/v14_release_launch_check_control.json`, and
`artifacts/a100_p32_window/v14_release_launch_check_candidate.json`.

The sixth progression narrows paired scalar output stores to M1 full-KV only.
The naturally aligned adjacent outputs are written with one `float2` store;
all other scalar routes retain the proven scalar stores. In the matched
40-warmup/1000-iteration A/B, the four-rate Ampere geomean falls from
0.035062 ms to 0.032996 ms (`1.063x`, 6.259% lower latency), and an immediate
repeat measures 0.033022 ms. Exactness passes 28/28. Artifacts are stored in
`artifacts/a100_p32_window/v14_m1_fullkv_float2_control.json`,
`artifacts/a100_p32_window/v14_m1_fullkv_float2_candidate.json`, and
`artifacts/a100_p32_window/v14_m1_fullkv_float2_candidate_repeat.json`.

Using exact artifact medians—not rounded ledger percentages—and weighting
each sequential matched progression by its affected share of the 140-case
matrix, cumulative Ampere latency improves `1.02155x`, or **2.155%**, versus
the fetched `ab277a17` main baseline. Planar measurements are excluded. This
clears the requested cumulative 2% threshold despite the unusable full-refresh
run above; a future quiet-window refresh should confirm the same result in one
continuous matrix.

### Post-merge v15 baseline

PR #87 merged as `3de3fb05`. This window uses that exact `origin/main` tip as
its Ampere control. Its clean 20-warmup/100-iteration, 140-case result is
`artifacts/a100_p32_window/qwen38_newmain_all_3de3fb05.json`: Ampere median
latency geomeans are 0.058310 ms (M1), 0.063643 ms (M2), 0.073368 ms (M4),
0.084338 ms (M8), and 0.086899 ms (M16), with an all-case geomean of
0.072445 ms. Maximum absolute error is 0.000080109. Planar timings remain
diagnostic and are excluded from all improvement figures.

The first v15 progression distributes each M1 full-KV stage's 16 bank-ID
bytes across four lanes as four aligned `uint32_t` loads. The previous
single-lane `uint4` attempt serialized this work and regressed M1; spreading
the loads retains the instruction reduction without that bottleneck. The
specialization is deliberately limited to compile-time `N=1024`, M1, and 16
tiles per block. In a reversal test with 50 warmups and 1,000 iterations, the
four-rate Ampere geomean falls 3.078% by median and 5.234% by mean. W2, W2.5,
and W3 improve by 6.25%, 6.25%, and 2.94% by median; W3.5 selects split 48
instead of 64 and is one timer quantum slower. Exactness is unchanged. The
candidate and restored-control artifacts are
`artifacts/a100_p32_window/v15_m1_fullkv_bankid_u32x4_repeat.json` and
`artifacts/a100_p32_window/v15_m1_fullkv_bankid_clean_repeat.json`.

The second progression removes a per-thread local-memory accumulator array
from the scalar M1/M2/M4 kernel. Each lane always owns exactly one of the four
N tiles, but the old implementation allocated accumulators for all four and
indexed them with `tile_in_warp`. Generated resource usage showed 32, 64, and
128 bytes of stack per thread for representative M1, M2, and M4 kernels. The
lane-owned representation stores only `Rows` accumulators and reduces those
figures to 0, 0, and 8 bytes. In the stable matched 20-warmup/300-iteration
84-case reversal, the Ampere median geomean falls 2.269% and the mean geomean
falls 2.168%. M1, M2, and M4 improve 1.145%, 3.189%, and 2.483% by median;
all seven shapes and all four rates improve. Exactness passes 28/28. The
control and candidate are stored in
`artifacts/a100_p32_window/v15_scalar_lane_accumulator_all_control.json` and
`artifacts/a100_p32_window/v15_scalar_lane_accumulator_all_candidate_retry.json`.

The first expanded candidate refresh for this change is excluded: M1/M2
latencies jumped by 20-38% partway through while M4 remained normal, then
recovered on the cached-binary retry. It is retained as
`artifacts/a100_p32_window/v15_scalar_lane_accumulator_all_candidate.json`.
An explicit scalar decode-mask rewrite was also rejected after regressing the
matched 36-case geomean by 1.136%, with every tested M and shape slower; its
artifacts are `v15_scalar_bankmask_control.json` and
`v15_scalar_bankmask_candidate.json`.

The third progression retunes M4 pipeline depth after the lane-local change.
Transition widths 5-7 use three staged K16 tiles instead of four, reducing
the new register/shared-memory footprint; W2 retains four stages because it
was neutral and noisy in the broad candidate. The complete 28-case M4 repeat
improves the Ampere median geomean by 3.216% and the mean by 2.969%. The first
selective run measured 3.372%/3.147%, and every shape group improved. The
matched lane-local control and repeated candidate are
`artifacts/a100_p32_window/v15_scalar_lane_accumulator_all_candidate_retry.json`
and
`artifacts/a100_p32_window/v15_scalar_lane_stage3_m4_selective_repeat.json`.
The non-selective three-stage diagnostic remains available as
`v15_scalar_lane_stage3_m4_candidate*.json`.

Four-stage short-K staging for M1/M2 was retested after removing the local
accumulator arrays, but still regressed their 56-case median geomean by
0.659%; only W2 was slightly positive. It is rejected and recorded in
`artifacts/a100_p32_window/v15_scalar_lane_stage4_m1_m2_candidate.json`.

Using the conservative repeated results and exact affected-case weighting,
the three sequential progressions improve the full 140-case Ampere target by
`exp((4*ln(1.03078) + 84*ln(1.02269) + 28*ln(1.03216)) / 140) = 1.02087x`,
or **2.087%** lower latency versus fetched `3de3fb05` main. Planar timings
are excluded.

The final continuous 20-warmup/300-iteration matrix is stored in
`artifacts/a100_p32_window/qwen38_v15_final_all_8f04e266.json`. Its mean
geomean is 2.113% faster than the fetched-main artifact and maximum absolute
error remains 0.000080109. Its median aggregate is not used: all five
full-KV M groups shift slower together (8.76% aggregate), including untouched
M8/M16, while the other shapes and the paired reversals remain positive.
This repeats the run-level full-KV timing anomaly seen in the first v15 full
refresh; the stable immediate A/B measurements above remain the acceptance
evidence.

Rejected v15 experiments are retained as diagnostics. Explicit
`cp.async.cg` input staging improved the broad screen by only 0.055%, while
the scalar-only form regressed 0.204%. WMMA `__launch_bounds__(128, 10)` and
`(128, 9)` caused large regressions; the scalar min-blocks variant was
neutral. Wave-aligned fixed M16 splits were slower for all seven model
shapes. A compile-time scalar split-40 specialization produced only one timer
quantum in one rate and no repeatable aggregate gain. The broad four-lane M1
bank-ID variant was positive overall but mixed outside full-KV, so it was
narrowed rather than accepted broadly. Relevant artifacts use the `v15_`
prefix in `artifacts/a100_p32_window/`, including `v15_input_cg_*`,
`v15_wmma_minblocks*`, `v15_wave_m16_*`,
`v15_m1_fullq_static_split40.json`, and
`v15_m1_bankid_u32x4_matched_*`.

### Post-merge v16 baseline

PR #89 merged as `829a8777`. The clean 20-warmup/100-iteration 140-case
Ampere control is
`artifacts/a100_p32_window/qwen38_newmain_all_829a8777.json`. Median latency
geomeans are 0.057675 ms (M1), 0.062415 ms (M2), 0.070734 ms (M4),
0.084479 ms (M8), and 0.087012 ms (M16), with a 0.071523 ms all-case
geomean. Maximum absolute error is 0.000080109. Planar timings are diagnostic
only and excluded from every improvement calculation.

The first v16 progression reduces the scalar M4 stage from three K16 tiles to
two for transition widths 5 and 7 (W2.5 and W3.5). W2 retains four stages and
W3 retains three: the broad two-stage screen improved overall but made W3
slightly slower. The narrowed 21-case M4 W2.5-W3.5 cached-binary repeat lowers
the Ampere median geomean by 1.782% and the mean by 2.131%; exactness passes
28/28. The control and repeat are stored in
`artifacts/a100_p32_window/v16_m4_stage2_control.json` and
`artifacts/a100_p32_window/v16_m4_stage2_selective_repeat.json`.

Two v16 experiments are rejected so far. Directly reading the already-cached
Python operator instead of calling the small resolver was mixed and regressed
the 20-case full-KV median geomean by 1.995%; M4/M8/M16 all slowed. The broad
M4 two-stage form improved 1.346% overall but regressed W3 by 0.274%, so it
was narrowed rather than accepted broadly. Diagnostics are retained as
`v16_cached_op_direct_*.json` and `v16_m4_stage2_candidate.json`.

The second v16 progression removes dead persistent accumulator state from the
fixed-N M8 WMMA routes. The lower eight activation rows are zero and never
stored, so their two FP32 accumulator values per fragment are routed to
temporary MMA outputs instead of being carried through the K loop. Generated
register use falls from 72 to 56 per thread. Generic MLP-gate retains the old
path after a broad screen found it negative. The cached 20-warmup/500-iteration
repeat improves the complete 28-case M8 median geomean by 0.835% and the mean
by 0.734%; MLP-gate is neutral, full-KV improves 1.459%, and MLP-down improves
1.713%. Exactness passes 28/28. Artifacts are
`artifacts/a100_p32_window/v16_m8_live_accumulator_control.json` and
`artifacts/a100_p32_window/v16_m8_live_accumulator_selective_repeat.json`.

Further scalar depth reductions were rejected. M2 two-stage staging regressed
its 28-case median and mean geomeans by 1.053% and 1.119%, with W3 about 4%
slower. One-stage M1 was visibly slower across the short-K shapes. Their
diagnostics are `v16_m2_stage2_*.json` and `v16_m1_stage1_*.json`.

The third v16 progression completes the M8 fixed-N/live-row coverage for
MLP-gate/up (`N=17408`). The prior generic route could not combine its static
N geometry with the newly reduced 56-register accumulator state. In a matched
40-warmup/1000-iteration reversal, every rate improves: 1.399%, 2.721%,
4.110%, and 4.762% for W2 through W3.5. The median geomean gain is 3.240%
and the mean geomean gain is 3.360%. Artifacts are
`artifacts/a100_p32_window/v16_m8_mlpgate_static_live_control.json` and
`artifacts/a100_p32_window/v16_m8_mlpgate_static_live_candidate.json`.

Two additional M8 variants were rejected after the live-accumulator change.
A 256-thread/N128 CTA regressed representative cases by roughly 6-12%, and
eliding inactive lower-row activation staging was broadly slower. Retain the
128-thread/N64 CTA and zero-filled inactive rows. Diagnostics are
`v16_m8_wide_cta_candidate.json` and
`v16_m8_live_rows_stage_candidate.json`.

The fourth v16 progression adds the missing full-row M16 fixed-N route for
MLP-gate/up (`N=17408`). In the matched 40-warmup/1000-iteration pair, all
four rates improve by 5.263-5.960%; the median geomean gain is 5.473% and the
mean geomean gain is 5.348%. Exactness passes 28/28. Artifacts are
`artifacts/a100_p32_window/v16_m16_mlpgate_static_control.json` and
`artifacts/a100_p32_window/v16_m16_mlpgate_static_candidate.json`.

Packing each fixed-N M8 stage's four bank IDs into one cached 32-bit load was
also rejected. It regressed the 24-case median and mean geomeans by 0.506%
and 0.490%, with five of six shape buckets slower; retain distributed byte
loads on partial-row kernels. Diagnostics are `v16_m8_bankid_u32_*.json`.

The fifth v16 progression applies the Marlin-style fragment lesson directly
to M8: because only rows 0-7 are live, `ldmatrix.x2` loads the two required
8x8 A submatrices into operand registers 0 and 2, while registers 1 and 3 are
zeroed. This removes half of the shared-matrix load work without changing the
`m16n8k16` arithmetic or output. Across the matched 30-warmup/500-iteration
24-case non-KV screen, every shape improves; the median and mean geomean gains
are 4.775% and 4.836%, with per-shape median gains from 3.060% to 6.594%.
Exactness passes 28/28. The control and candidate are
`artifacts/a100_p32_window/v16_m8_bankid_u32_control.json` and
`artifacts/a100_p32_window/v16_m8_ldmatrix_x2_candidate.json`.

Applying `ldmatrix.x2` to M8 full-KV was rejected and narrowed out. Its four
rates were neutral-to-slower, with roughly a 2.0% median geomean regression
against the matched live-accumulator control; retain `ldmatrix.x4` for
compile-time `N=1024`. The diagnostic is
`v16_m8_ldmatrix_x2_fullkv_candidate.json`.

The sixth v16 progression adds compile-time `K=5120` to the accepted M16
MLP-gate/up fixed-N kernel. Against that immediate 40-warmup/1000-iteration
control, all four rates improve; the median and mean geomean gains are 0.835%
and 0.726%. Exactness passes 28/28. The candidate is
`artifacts/a100_p32_window/v16_m16_mlpgate_statick_candidate.json`; its control
is `v16_m16_mlpgate_static_candidate.json` from the fourth progression.

The seventh v16 progression expands compile-time `K=5120` to the other four
unambiguous M16 N routes: full-Q, full-KV, linear-QKV, and linear-Z. In the
matched 30-warmup/500-iteration five-shape screen (including the already
specialized MLP-gate control), the median and mean geomeans improve 1.575%
and 1.254%. Every newly changed shape improves by median; full-KV gains
5.206%, full-Q 1.136%, linear-QKV 0.806%, and linear-Z 0.810%. Exactness
passes 28/28. The artifact is
`artifacts/a100_p32_window/v16_m16_statick_short_candidate.json`.

The eighth v16 progression selectively adds compile-time `K=5120` to M8
full-KV and linear-QKV. The narrowed 40-warmup/1000-iteration repeat improves
the eight-case median geomean by 1.390% and mean by 0.418%; full-KV improves
2.199% by median and linear-QKV improves 0.587%. Exactness passes 28/28.
The accepted repeat is
`artifacts/a100_p32_window/v16_m8_statick_selective_repeat.json`. The broader
`v16_m8_statick_short_candidate.json` is diagnostic only: MLP-gate regressed
about 1%, while full-Q and linear-Z did not improve consistently by mean, so
those routes were restored.

The ninth v16 progression completes formal M16 fixed-K coverage for the two
`N=5120` shapes: attention-out uses compile-time `K=6144`, and MLP-down uses
`K=17408`. The matched 40-warmup/1000-iteration eight-case median and mean
geomeans improve 1.201% and 1.176%; both shapes improve by both metrics.
Exactness passes 28/28. The artifact is
`artifacts/a100_p32_window/v16_m16_statick_n5120_candidate.json`.

The tenth v16 progression moves each fixed-N/full-row M16 stage's packed
four-byte bank-selector load into the existing Ampere `cp.async.ca` pipeline.
The selector now overlaps the input and trellis transfers instead of issuing
as a synchronous `__ldg` before the pipeline commit. Full-KV (`N=1024`) keeps
its separate byte-load route. In the matched 30-warmup/500-iteration pair,
all six affected shape buckets improve by both metrics. The 24-case median
and mean geomeans improve 2.333% and 2.498%, with median gains of 0.823% for
attention-out, 1.227% for linear-Z, 2.183% for linear-QKV, 3.035% for full-Q,
3.285% for MLP-down, and 3.475% for MLP-gate/up. The control and candidate are
`artifacts/a100_p32_window/v16_m16_bank_cpasync_control.json` and
`artifacts/a100_p32_window/v16_m16_bank_cpasync_candidate.json`.

Using sequential affected-case log weighting, the first nine progressions
were 1.8859% faster than fetched `829a8777` main. The tenth progression raises
that estimate to
`exp(ln(1.0188591) + 24/140 * ln(1.02333)) = 1.022895x`, or **2.290%**
cumulative median improvement versus main. Planar timings remain excluded.

Further v16 experiments rejected after the x2 checkpoint are retained as
untracked diagnostics. Omitting lower shared rows regressed 0.740%, async
zero-fill regressed 0.230%, and MLP-gate bank-mask hoisting regressed 1.964%
on the relevant M8 screens. M4 W2 three-stage depth, long-K four-stage WMMA,
M16 `float4` output stores, a two-chain split reducer, and named M4 scalar
accumulators all regressed or were neutral. Moving packed M16 bank IDs from
`__ldg` to ordinary global loads was only +0.138% median/+0.026% mean and was
reverted as below the acceptance threshold. Their artifacts use the
`v16_m8_x2_`, `v16_m4_w2_stage3_`, `v16_m816_mlpdown_stage4_`,
`v16_m16_float4_`, `v16_reduce2_`, `v16_m4_named_accumulator_`, and
`v16_m16_bank_global_` prefixes.

Two final fixed-K experiments were also rejected. Specializing the two M8
`N=5120` routes produced only +0.090% median across their eight cases
(attention-out was neutral), too small to accept. Extending fixed K to the
scalar M1/M2/M4 kernels regressed their matched 84-case median and mean
geomeans by 1.902% and 2.355%; every shape bucket was slower. Diagnostics are
`v16_m8_statick_n5120_candidate.json` and `v16_scalar_statickn_*.json`.

### Post-merge v17 baseline

PR #90 merged as `6b3cea54`. The clean 20-warmup/100-iteration 140-case
Ampere control is
`artifacts/a100_p32_window/qwen38_newmain_all_6b3cea54.json`. Median latency
geomeans are 0.057193 ms (M1), 0.061991 ms (M2), 0.068832 ms (M4),
0.079767 ms (M8), and 0.083897 ms (M16), with a 0.069599 ms all-case
geomean. Maximum absolute error is 0.000080109. Planar timings are diagnostic
only and excluded from every improvement calculation.

The first v17 progression extends asynchronous packed bank-selector staging
from full-row M16 to every fixed-N M8 route. Each K16 stage replaces four
synchronous byte loads with one four-byte `cp.async.ca`, overlapping the
selector with the existing input and trellis pipeline. In the matched
30-warmup/500-iteration 28-case pair, the median and mean geomeans improve
2.161% and 2.364%. All shape buckets are non-negative by median; full-KV
improves 2.224%, linear-QKV 2.717%, MLP-gate/up 3.224%, and MLP-down 4.107%.
The control and candidate are
`artifacts/a100_p32_window/v17_m8_bank_cpasync_control.json` and
`artifacts/a100_p32_window/v17_m8_bank_cpasync_candidate.json`.

The second v17 progression applies a 16-byte `cp.async.ca` selector copy to
the six fixed-N M4 routes other than full-KV. The narrowed
40-warmup/1000-iteration repeat improves the 24-case median and mean geomeans
by 0.958% and 1.475%; every shape improves by median, from 0.382% on
linear-QKV to 1.511% on MLP-down. The matched scalar control and accepted
repeat are `artifacts/a100_p32_window/v17_scalar_bank_cpasync_control.json`
and `artifacts/a100_p32_window/v17_m4_bank_cpasync_selective_repeat.json`.

The broader scalar selector experiment was narrowed rather than accepted.
Distributed four-byte async copies regressed M1 full-KV by 4.629% median,
and 16-byte async copies regressed the complete M2 set by 0.142%. Their source
paths were restored; the broad diagnostic is
`v17_scalar_bank_cpasync_candidate.json`.

The third v17 progression applies the Hopper bank-mask-hoisting lesson only
where it survives Ampere resource and schedule constraints. For M2 W2.5 and
W3.5 outside full-KV, two adjacent scalar decode rows share each lower/upper
bank-mask pair instead of recomputing the selectors for every decoded state.
Other rates, rows, and M2 full-KV retain their bit-identical accepted paths.
On the clean matched 40-warmup/1000-iteration 12-case run, every case improves;
the median geomean gain is 3.524%, while the aggregate mean-latency gain is
3.306%. Per-shape median gains are 3.349% for full-Q, 4.545% for
attention-out, 3.150% for linear-QKV, 5.676% for linear-Z, 3.545% for
MLP-gate/up, and 0.939% for MLP-down. Exactness passes 28/28. The control and
candidate are `artifacts/a100_p32_window/v17_m2_bank_mask_group_control.json`
and
`artifacts/a100_p32_window/v17_m2_bank_mask_group_selective_candidate.json`.
The control process preloaded the exact accepted `499e6810` JIT binary at
cache fingerprint `657b7db6c159a655` before running the same benchmark.

Using affected-case log weighting, the three accepted v17 progressions are
`exp(28/140 * ln(1.02161) + 24/140 * ln(1.00958) + 12/140 *
ln(1.03524076)) = 1.008919x`, or **0.892%** cumulative median improvement
versus fetched `6b3cea54` main. Planar timings remain excluded.

Additional v17 failures are recorded to prevent retesting dead ends. A packed
M1 selector was only +0.215% overall and regressed attention-out and linear-Z;
an async M16 full-KV selector was roughly 7-9% slower. Hoisting the WMMA lane
mapping grew live ranges and regressed M16 uniformly. Four skewed shared
codebook replicas were 1-2% slower than the read-only-cache path. Forcing
ordinary L1 `.ca` codebook loads in place of the generated read-only load mode
was neutral-to-mixed in a contention-only screen. A 16-bit inline-PTX PGC mix
lengthened the hot SASS region and was rejected before timing. Finally,
grouping scalar bank masks broadly regressed M1 by 1.968% and M4 by 1.453% in
the diagnostic screen; only the cleanly validated M2 W2.5/W3.5 subset above
is retained. Diagnostics use the `v17_m1_bank_cpasync_`,
`v17_m16_fullkv_bank_cpasync_`, `v17_wmma_lane_hoist_`,
`v17_m8_shared_levels4_`, and `v17_scalar_bank_group_` labels; contention-only
files remain untracked.

The fourth v17 progression pipelines the packed 16-byte M2 bank-selector copy
only on the same validated non-full-KV W2.5/W3.5 routes. The earlier broad M2
async experiment predated bank-mask grouping and was slightly negative; after
grouping shortens the selector's decode use, `cp.async.ca` overlaps its global
load with the scalar input and trellis stage. The clean 60-warmup/2000-iteration
repeat improves the 12-case median and mean geomeans by 1.210% and 0.993%.
Every shape is non-negative by median; gains range from 0.462% on MLP-down to
1.600% on linear-QKV. The accepted repeat is
`artifacts/a100_p32_window/v17_m2_group_async_selector_repeat.json`; its
immediate control is
`artifacts/a100_p32_window/v17_m2_bank_mask_group_selective_candidate.json`.

Including this fourth progression, affected-case log weighting gives
`exp(28/140 * ln(1.02161) + 24/140 * ln(1.00958) + 12/140 *
ln(1.03524076) + 12/140 * ln(1.01210067)) = 1.009959x`, or **0.996%**
cumulative median improvement versus fetched `6b3cea54` main. Planar timings
remain excluded.

An isolated M4 selector cache-policy follow-up was also rejected. Changing
only the selector from `cp.async.ca` to `cp.async.cg` improved the clean
24-case median geomean by just 0.036%, with 22 cases bit-for-bit identical in
the event median. The earlier combined cache-policy diagnostic was therefore
not hiding a material selector-only gain; retain `.ca`.

The fifth v17 progression narrows the earlier broad M1 selector experiment to
the W2.5/W3.5 non-full-KV routes that were consistently positive. One
16-byte `cp.async.ca` now replaces 16 distributed selector-byte loads per
scalar K stage; W2/W3 and full-KV keep their accepted synchronous paths. In
the clean matched 60-warmup/2000-iteration 12-case pair, the median and mean
geomeans improve 1.626% and 0.976%. Every shape improves by median, from
1.021% on MLP-down to 2.613% on linear-QKV. The exact accepted `088f0dbe`
binary was preloaded from JIT fingerprint `9f664b2065b9d1fb` for the control.
Artifacts are
`artifacts/a100_p32_window/v17_m1_async_selector_selective_control.json` and
`artifacts/a100_p32_window/v17_m1_async_selector_selective_repeat.json`.

Including the fifth progression, affected-case log weighting gives
`exp(28/140 * ln(1.02161) + 24/140 * ln(1.00958) + 12/140 *
ln(1.03524076) + 12/140 * ln(1.01210067) + 12/140 * ln(1.01625809)) =
1.011356x`, or **1.136%** cumulative median improvement versus fetched
`6b3cea54` main. Planar timings remain excluded.

The sixth v17 progression extends packed async M1 selector staging to the
three W2 fixed-N routes that remained positive in the broad diagnostic:
full-Q, linear-QKV, and MLP-gate/up. The clean matched
60-warmup/2000-iteration three-case pair improves the median and mean
geomeans by 1.604% and 1.189%, and every case improves. The attempted W3
MLP-gate extension was exactly neutral and was removed. The control preloads
the exact accepted `c023f477` binary from JIT fingerprint
`cdffda1bd75374aa`. Artifacts are
`artifacts/a100_p32_window/v17_m1_w2_async_selector_selective_control.json`
and
`artifacts/a100_p32_window/v17_m1_w2_async_selector_selective_candidate.json`.

Including the sixth progression, affected-case log weighting raises the
cumulative median improvement to **1.170%** versus fetched `6b3cea54` main.
Planar timings remain excluded.

The seventh v17 progression extends M2 packed async selector staging to the
W3 `N=17408` and `N=5120` routes. In the clean matched
60-warmup/2000-iteration six-shape screen, MLP-gate/up improves 1.020% and
MLP-down 1.887%; attention-out (which shares `N=5120`) and the other three
shapes are neutral. The six-case median and mean geomeans improve 0.482% and
0.983%. The control preloads exact accepted `433f778e` from JIT fingerprint
`02ac30b097c0cef4`. Artifacts are
`artifacts/a100_p32_window/v17_m2_w3_async_selector_control.json` and
`artifacts/a100_p32_window/v17_m2_w3_async_selector_candidate.json`.
Affected-case log weighting now gives **1.191%** cumulative median improvement
versus fetched `6b3cea54` main; planar timings remain excluded.

The eighth v17 progression statically unrolls the scalar split reducer for
the split counts selected by the live autotuner, but only on non-full-KV M2,
M4, and M16 routes. The broad 140-case experiment exposed why this must be
selective: static reducers improved the 120 non-full-KV cases by 0.375%
median geomean, while full-KV regressed 1.888%. M1 and M8 were also mixed.
Restricting the identical specialized code paths to the 72 robust cases gives
0.446% median, 0.397% mean, and 0.484% p95 geomean gains, with 27 wins, 43
quantized ties, and two losses by median. M2, M4, and M16 improve 0.187%,
0.469%, and 0.282% respectively across their non-KV cases. The exact accepted
`cf96b17c` JIT binary at fingerprint `cd31af2dd6479ddc` was preloaded for the
control. The broad diagnostic artifacts are
`artifacts/a100_p32_window/v17_static_reducers_candidate.json` and
`artifacts/a100_p32_window/v17_static_reducers_control.json`; the committed
dispatch preserves the runtime reducer on every route rejected by that pair.
Exactness passes 28/28.

Including this progression, affected-case log weighting raises the cumulative
median improvement to **1.423%** versus fetched `6b3cea54` main. Planar timings
remain excluded.

Two more reducer/load experiments were rejected. Packing four output elements
per split-reduction thread reduced the reducer grid fourfold and was
decisively slower because it sacrificed the parallelism that hides strided
partial-output loads. Separately, staging each M8 selector as one aligned
16-byte shared-memory sector with `cp.async.cg` regressed the full screen by
0.312% median geomean and linear-QKV by 1.49%; the accepted distributed
selector layout remains faster.

The ninth v17 progression extends the static split reducer to the 24 M8
non-full-KV cases after the broad screen's aggregate M8 result was obscured by
the losing full-KV cases. In the clean matched 40-warmup/1000-iteration pair,
all 24 medians are non-regressing: the median, mean, and p95 geomeans improve
1.231%, 1.028%, and 0.997%. Attention-out and linear-Z improve 2.691% and
2.667% by median, MLP-down improves 1.507%, and the remaining shapes are
non-negative. Full-KV deliberately retains the runtime reducer. The control
preloads exact accepted `c928aec5` from JIT fingerprint `593b839d5a285eec`.
Artifacts are
`artifacts/a100_p32_window/v17_m8_static_reducer_control.json` and
`artifacts/a100_p32_window/v17_m8_static_reducer_candidate.json`.
Affected-case log weighting now gives **1.636%** cumulative median improvement
versus fetched `6b3cea54` main; planar timings remain excluded.

Compile-time split specialization of the main M16 MLP-gate kernel was also
rejected. At fixed split 10, all four 2000-iteration medians were exactly
identical to the runtime-split control, so the compiler is already reducing
the uniform split arithmetic effectively. Diagnostics use the
`v17_m16_static_split10_` prefix.

The tenth v17 progression extends the static reducer to the three M1 shape
families that were positive by median, mean, and p95 in the broad experiment:
full-Q, linear-QKV, and long-K MLP-down. The clean matched
60-warmup/2000-iteration 12-case pair improves every case. Median, mean, and
p95 geomeans improve 3.684%, 3.931%, and 3.001%; per-shape median gains are
3.026% for full-Q, 3.548% for linear-QKV, and 4.485% for MLP-down. Other M1
shapes and full-KV keep the runtime reducer. The control preloads exact
accepted `31caa82d` from JIT fingerprint `a73a469768c9542b`. Artifacts are
`artifacts/a100_p32_window/v17_m1_static_reducer_control.json` and
`artifacts/a100_p32_window/v17_m1_static_reducer_candidate.json`. Affected-case
log weighting now gives **1.951%** cumulative median improvement versus
fetched `6b3cea54` main; planar timings remain excluded.

The eleventh v17 progression adds M1 MLP-gate/up to the static-reducer set.
The final-source matched 60-warmup/2000-iteration pair improves every rate:
the median, mean, and p95 geomeans improve 2.300%, 2.350%, and 2.280%.
The control again preloads exact accepted `c3c21fbb` from fingerprint
`847808c5f0c8f7c3`. Artifacts are
`artifacts/a100_p32_window/v17_m1_mlpgate_static_reducer_control.json` and
`artifacts/a100_p32_window/v17_m1_mlpgate_static_reducer_candidate.json`.
Attention-out and linear-Z were screened at the same time but remain on the
runtime reducer: attention W2 regressed 4.88%, while linear-Z had a slightly
negative p95 aggregate. With only the robust MLP-gate subset retained,
affected-case log weighting reaches **2.018%** cumulative median improvement
versus fetched `6b3cea54` main. Planar timings remain excluded.

Nsight profiling and follow-up experiments close several additional dead
ends. The M8 full-Q kernel is limited by random codebook traffic through the
unified L1/TEX path and integer ALU work, but its read-only-cache hit rate is
96.2%. Distributing the 256-entry codebook across warp registers required
four shuffles per random half lookup and slowed W2 full-Q from about 0.098 ms
to 0.140 ms. Fetching aligned 32-bit codebook words instead of 16-bit values
slowed it to 0.108 ms. Replacing shared trellis-word loads with warp shuffles
was also 2.08% slower. Marking the split-reducer input loads `__ldg` regressed
the full-KV screen. Finally, wave-adjacent M8 splits 21 (full-Q), 19
(linear-QKV), 31 (linear-Z), and 25 (attention-out) all lost to the existing
autotuned choices, so the profiler's theoretical tail-wave estimate does not
translate into a useful tuning candidate for these imbalanced K slices.

### Post-merge v18 baseline

PR #91 merged as `90c4fa5f`. The clean 20-warmup/100-iteration 140-case
Ampere control is
`artifacts/a100_p32_window/qwen38_newmain_all_90c4fa5f.json`. Median latency
geomeans are 0.057364 ms (M1), 0.060760 ms (M2), 0.068937 ms (M4),
0.078366 ms (M8), and 0.084038 ms (M16), with a 0.069161 ms all-case
geomean. Maximum absolute error is 0.000080109. Planar timings remain
diagnostic only and are excluded from improvement calculations.

The first v18 progression applies the Hopper paired-decode lesson at the
16-bit arithmetic level. On M8 fixed-N routes, adjacent decoded states share
one bank mask. Their two PGC16 transforms now execute as packed 16-bit lanes,
amortizing the fold shifts/XORs while retaining the same four read-only-cache
codebook fetches and exact FP32 accumulation. The matched
40-warmup/1000-iteration 24-case non-full-KV pair improves the median geomean
by **2.319%**, with 21 wins, three event-quantized ties, and no losses.
Full-Q, attention-out, linear-Z, and MLP-down improve 3.167%, 2.740%, 3.653%,
and 3.095%; linear-QKV improves 0.917% and MLP-gate/up 0.384%.
Artifacts are
`artifacts/a100_p32_window/v18_m8_packed_pgc_control.json` and
`artifacts/a100_p32_window/v18_m8_packed_pgc_candidate_repeat.json`.
Affected-case log weighting gives
`exp(24/140 * ln(1.023186)) = 1.003937x`, or **0.394%** cumulative median
improvement versus fetched `90c4fa5f` main. Planar timings are excluded.

The second v18 progression applies the packed transform to M4 scalar decode
groups for W2, W2.5, and W3. Two adjacent K rows share each lower/upper bank
selector, so the PGC transform is packed while each decoded weight pair is
still reused across all four output rows. In the 20-warmup/100-iteration
18-case screen all cases are non-regressing: 12 improve and six tie at CUDA
event resolution. The median geomean improves **1.003%**; per-shape geomean
gains are 1.220% full-Q, 1.042% attention-out, 0.917% linear-QKV, 1.437%
linear-Z, 0.287% MLP-gate/up, and 1.117% MLP-down. W3.5 is deliberately
excluded because all six shapes regressed. The candidate artifact is
`artifacts/a100_p32_window/v18_m14_packed_pgc_candidate.json`; its control is
the fetched-main 140-case artifact above. Affected-case log weighting now
gives
`exp(24/140 * ln(1.023186) + 18/140 * ln(1.010027)) = 1.005226x`, or
**0.523%** cumulative median improvement versus fetched main.

The third v18 progression selectively enables packed scalar groups for 18 M1
cases. Full-Q, attention-out, linear-QKV, and linear-Z use the path at all
four rates; MLP-gate/up uses it only for W2/W2.5. W3/W3.5 MLP-gate/up remains
on the prior scalar decoder because the multi-case diagnostic encountered
the queue-stall failure described below. Across the retained cases, 11
medians improve and seven tie with no losses; the affected geomean improves
**1.154%**. The diagnostic artifact is
`artifacts/a100_p32_window/v18_m1_packed_pgc_candidate.json`. Including this
progression, affected-case log weighting reaches **0.671%** cumulative median
improvement versus fetched `90c4fa5f` main.

The fourth v18 progression packs the existing two-row M2 bank groups for
W2.5/W3.5 on every non-full-KV route except W3.5 MLP-gate/up. The retained
11-case screen has eight wins and three ties with no losses; its affected
geomean improves **1.852%**. Full-Q improves 2.083%, attention-out 1.156%,
linear-QKV 2.456%, MLP-down 2.920%, and linear-Z is neutral; W2.5
MLP-gate/up improves 3.191%. The diagnostic candidate is
`artifacts/a100_p32_window/v18_m2_packed_pgc_candidate.json`. Including this
selective progression, affected-case log weighting reaches **0.816%**
cumulative median improvement versus fetched main.

The fifth v18 progression extends packed M2 groups to ten selective W2/W3
cases. Every retained median improves and their geomean gain is **2.354%**.
W2 attention-out and linear-Z regress and therefore keep the prior decoder;
the distinct four-stage W2 MLP-down path remains enabled. The candidate is
`artifacts/a100_p32_window/v18_m2_packed_pgc_w23_candidate.json`.
Affected-case log weighting now reaches **0.984%** cumulative median
improvement versus fetched main.

The sixth v18 progression extends the paired PGC16 transform to M8 full-KV.
At the same 48-way split, the clean 40-warmup/1000-iteration four-rate screen
has three wins and one event-quantized tie; its median geomean improves
**3.078%**. Maximum absolute error remains below `1.53e-05`. The matched
control and candidate are
`artifacts/a100_p32_window/v18_fullkv_s48_current.json` and
`artifacts/a100_p32_window/v18_m8_fullkv_packed_pgc_candidate.json`.
Affected-case log weighting now reaches **1.071%** cumulative median
improvement versus fetched main. Planar timings remain excluded.

The seventh v18 progression combines the M1 long-K MLP-down fixed-N route
with paired PGC16 decode. That route previously stayed runtime-N because the
fixed specialization alone was neutral; paired decode changes the balance.
In the matched split-128 40-warmup/1000-iteration pair all four rates win,
falling from 0.0870/0.1004/0.1004/0.1055 ms to
0.0840/0.0942/0.0973/0.0963 ms. The affected geomean speedup is **1.0570x**
(5.390% lower latency), and maximum absolute error remains below `2.68e-05`.
Artifacts are
`artifacts/a100_p32_window/v18_m1_mlpdown_static_packed_control.json` and
`artifacts/a100_p32_window/v18_m1_mlpdown_static_packed_candidate_repeat.json`.
Affected-case log weighting now reaches **1.231%** cumulative median
improvement versus fetched main; planar timings remain excluded.

The eighth v18 progression enables paired PGC16 decode for M2 full-KV. In
the matched split-64 40-warmup/1000-iteration pair, three rates win and W3
ties: latency falls from 0.0328/0.0348/0.0328/0.0348 ms to
0.0307/0.0328/0.0328/0.0328 ms. The affected geomean speedup is **1.0475x**
(4.538% lower latency), with maximum absolute error below `1.15e-05`.
Artifacts are
`artifacts/a100_p32_window/v18_m2_fullkv_packed_pgc_control.json` and
`artifacts/a100_p32_window/v18_m2_fullkv_packed_pgc_candidate.json`.
Affected-case log weighting now reaches **1.365%** cumulative median
improvement versus fetched main; planar timings remain excluded.

The ninth v18 progression restores the two M1 MLP-gate/up rates whose first
screen was hidden by a queue stall. In the focused matched split-40
60-warmup/2000-iteration pair, W3/W3.5 fall from 0.094208/0.092160 ms to
0.091136/0.090112 ms. The affected geomean speedup is **1.0282x** (2.743%
lower latency), and exactness remains below `1.34e-05`. The control and
candidate are
`artifacts/a100_p32_window/v18_m1_mlpgate_w33_packed_pgc_control_s40.json`
and
`artifacts/a100_p32_window/v18_m1_mlpgate_packed_pgc_all_candidate_s40.json`.
Affected-case log weighting now reaches **1.405%** cumulative median
improvement versus fetched main; planar timings remain excluded.

The tenth v18 progression resolves the final contaminated scalar case. In a
focused matched split-32 60-warmup/2000-iteration pair, M2 MLP-gate/up W3.5
falls from 0.106496 ms to 0.100352 ms, a **1.0612x** speedup (5.769% lower
latency), with maximum absolute error `9.54e-06`. Artifacts are
`artifacts/a100_p32_window/v18_m2_mlpgate_w35_packed_control.json` and
`artifacts/a100_p32_window/v18_m2_mlpgate_w35_packed_candidate.json`.
Affected-case log weighting now reaches **1.449%** cumulative median
improvement versus fetched main; planar timings remain excluded.

The eleventh v18 progression revisits the Marlin-style `ldmatrix.x2` M8
full-KV load after paired decode changed that kernel's instruction balance.
In the matched split-48 60-warmup/2000-iteration pair, all four rates win:
the x4 control's 0.033792/0.033792/0.033792/0.032768 ms falls to
0.032768/0.031744/0.032768/0.031744 ms. The affected geomean speedup is
**1.0397x** (3.820% lower latency), with maximum absolute error below
`1.53e-05`. Artifacts are
`artifacts/a100_p32_window/v18_m8_fullkv_packed_ldmatrix_x4_control.json` and
`artifacts/a100_p32_window/v18_m8_fullkv_packed_ldmatrix_x2_candidate.json`.
Affected-case log weighting now reaches **1.562%** cumulative median
improvement versus fetched main; planar timings remain excluded.

The twelfth v18 progression bypasses the Python autotune-key lookup for the
measured M8 `(K,N)=(5120,1024)` hot shape and directly launches its stable
48-way plan. Unknown MKNs still use the default-on process-local tuner, and
no plan is written to disk. In the matched 100-warmup/4000-iteration screen,
latency falls from a 0.033533 ms affected geomean to 0.033276 ms, a
**1.0077x** speedup, with maximum absolute error below `1.53e-05`. Artifacts
are `artifacts/a100_p32_window/v18_m8_fullkv_x2_scalar_guard_control.json`
and
`artifacts/a100_p32_window/v18_m8_fullkv_minimal_fastreturn_candidate.json`.
Affected-case log weighting now reaches **1.584%** cumulative median
improvement versus fetched main; planar timings remain excluded.

The thirteenth v18 progression directly dispatches the stable M1 full-Q and
full-KV plans, avoiding the hot Python cache key for these eight cases while
leaving unknown shapes on the live tuner. In the focused screen, the
affected geomean falls by **1.124%** (a **1.0114x** speedup): full-Q has two
wins and two ties, while full-KV has two wins, one tie, and one event-tick
loss. Exactness remains below `1.34e-05`. The matched control and candidate
are `artifacts/a100_p32_window/v18_m1_fastplans_control.json` and
`artifacts/a100_p32_window/v18_m1_fastplans_narrow_candidate.json`.
Affected-case log weighting now reaches **1.650%** cumulative median
improvement versus fetched main; planar timings remain excluded.

The fourteenth v18 progression replaces the serial small-N split reducer
with a four-output-per-warp Ampere reducer for the measured full-KV waves:
split 64 for M1/M2/M4 and split 48 for M8. Four interleaved lane groups load
adjacent outputs while eight lanes parallelize each output's split sum. This
preserves useful coalescing and expands the reducer from 4-16 blocks to
32-256 blocks on the 124-SM A100. M2/M4 now directly dispatch their measured
split-64 plan so normal `split_count=0` calls reach the new reducer; unknown
MKNs remain on the default-on in-memory tuner.

Against the pre-change fixed-split controls, the affected geomean speedups
are **1.0331x** for M1, **1.0588x** for M2, **1.0903x** for M4, and
**1.0580x** for M8. All 16 medians win or tie, for a combined **1.0598x**
speedup (5.645% lower latency); maximum absolute error stays below
`1.15e-05`. Candidate artifacts are
`v18_m124_fullkv_warpreduce4_candidate.json` and
`v18_m8_fullkv_warpreduce4_candidate.json`; controls are
`v18_m1_fullkv_packed_pgc_control.json`,
`v18_m2_fullkv_packed_pgc_candidate.json`,
`v18_m4_fullkv_packed_pgc_control.json`, and
`v18_m8_fullkv_x2_scalar_store_control.json` under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**2.327%** cumulative median improvement versus fetched main; planar timings
remain excluded.

Two structural experiments were rejected before this progression. Direct
FP32 atomic split accumulation for M1 full-KV removed the partial tensor and
reducer launch, but atomic contention plus output zero-fill raised latency
from roughly 0.032-0.034 ms to 0.037-0.039 ms. Warp-private staging on M16
full-Q replaced CTA barriers with duplicated activation/window stages, but
the extra traffic raised latency from 0.106-0.110 ms to 0.116-0.119 ms.
Their diagnostics are `v18_m1_fullkv_atomic_candidate.json` and
`v18_m16_fullq_warp_private_candidate.json`; both source changes were fully
restored.

Further packed-transform expansions were narrowed out. M16 non-KV paths
were event-quantized neutral in normal samples and occasionally suffered a
large schedule outlier, so their dispatch remains unchanged. M2's existing
two-row bank groups and M1 static routes showed promising one-tick wins, but
long multi-case runs intermittently charged 0.22-0.28 ms allocator/queue
stalls to individual CUDA-event intervals; neither path is committed without
a clean aggregate. Diagnostics use the `v18_m16_packed_pgc_`,
`v18_m2_packed_pgc_`, and `v18_m1_packed_pgc_` prefixes.

Three additional experiments were rejected. Replacing the packed helper's
mask/shift lane assembly with explicit PRMT instructions regressed M8 by
roughly 1-2%. Shortening the decoded-pair live ranges was neutral on full-Q
and linear-Z but regressed M8 MLP-down. Finally, a Marlin-style last-CTA
split reducer remained exact but extended the M8 full-Q critical tail:
split-16 latency rose to 0.112-0.117 ms from the accepted 0.095-0.098 ms.
The fused-reducer diagnostics are
`artifacts/a100_p32_window/v18_m8_fused_reduce_s16_candidate.json` and
`artifacts/a100_p32_window/v18_m8_fused_reduce_s24_candidate.json`; all three
source experiments were restored.

M1 full-KV paired decode was also rejected after a matched split-64 control.
The initially promising candidate measured 0.0317-0.0328 ms, but the restored
control measured 0.0307-0.0328 ms and won three rates. Diagnostics are
`artifacts/a100_p32_window/v18_m1_fullkv_packed_pgc_candidate.json` and
`artifacts/a100_p32_window/v18_m1_fullkv_packed_pgc_control.json`.

M4 full-KV paired decode was rejected after its matched split-64 control:
W2/W2.5 were slower and W3 tied, while W3.5 never entered the candidate path.
Diagnostics are `v18_m4_fullkv_packed_pgc_candidate.json` and
`v18_m4_fullkv_packed_pgc_control.json`.

M16 full-KV paired decode was rejected after its matched split-32 control:
W2/W3 tied and W2.5/W3.5 regressed. Replacing the packed hash's two scalar
increments with `__vadd2`, and separately replicating the shared bank mask
with an integer multiply, both made normal M8 full-Q slower. Batched-event
autotuning reduced timer quantization but changed the clean 28-case M1 hot
geomean by only +0.061% versus main, below the acceptance threshold. These
diagnostics use the `v18_m16_fullkv_packed_pgc_`, `v18_m8_packed_vadd2_`,
`v18_m8_mask_replicate_`, and `v18_m1_batch_autotune_` prefixes; all source
changes were restored.

Packing distance-64 states with independent bank masks was exact but
regressed every M4 W3.5 shape by a 4.03% geomean; the same layout was neutral
to slower on M16. Compile-time K on packed M8 full-Q raised every rate by one
event tick. A scalar `(K,N,split)=(5120,1024,64)` specialization also lost
across M1/M2/M4 because its larger scheduled body outweighed eliminating the
uniform divisions. Diagnostics use the `v18_m4_w35_vertical_packed_`,
`v18_m16_vertical_packed_`, `v18_m8_fullq_statick_packed_`, and
`v18_scalar_fullkv_static_ksplit_` prefixes; all source changes were restored.

Four later host/epilogue experiments were rejected. Skipping dead-row shared
staging after the accepted M8 `ldmatrix.x2` load and replacing scalar output
stores with `float2` both regressed their matched full-KV controls. A final
head-to-head autotune pass and a 2% selection hysteresis still mis-ranked the
short split-48 launch because CUDA-event quantization dominated first-use
samples. Finally, generalizing direct measured dispatch across all 28 M8
cases was event-quantized neutral (three wins, 22 ties, two losses after
excluding one queue-stall sample), so only the focused full-KV path remains.
Diagnostics use the `v18_m8_fullkv_x2_deadrows_`,
`v18_m8_fullkv_x2_float2_`, `v18_m8_fullkv_x2_autotune_`, and
`v18_m8_fastplans_` prefixes; rejected source changes were restored.

Generalizing the same direct-plan path across all 28 M1 cases was narrowed
to the retained full-Q/full-KV subset. The broad screen had five wins and 20
ties, but M1 linear-Z W2 and MLP-down W2 each lost one event tick. Those
shapes retain normal in-memory autotuning. Diagnostics are
`artifacts/a100_p32_window/v18_m1_fastplans_control.json` and
`artifacts/a100_p32_window/v18_m1_fastplans_candidate.json`.

Two reducer variants were rejected while finding the retained layout. The
first assigned one complete warp to each output, improving M1/M2 and slightly
improving M4 but issuing split-strided loads; it regressed M8 by 3.03% and
M16 by 6.99%. The retained four-output layout recovered coalescing and won
through M8, but remained 0.70% slower at M16 split 32, so M16 keeps the
serial reducer. Diagnostics use the `v18_m*_fullkv_warpreduce_` and
`v18_m*_fullkv_warpreduce4_` prefixes.

The direct 140-case post-progression refresh is diagnostic only. Isolated
0.22-0.29 ms charges appeared in otherwise stable M2, M16, and full-KV
samples despite the exclusivity gate, matching the previously documented
queued CUDA-event stall signature. It is retained as
`artifacts/a100_p32_window/qwen38_v18_current_all_55b5c849.json` and is not
used in the cumulative comparison.

## v19: post-PR-92 continuation

PR 92 was merged and `origin/main` was fetched at merge commit `03144a22`.
The fresh 140-case main baseline is
`artifacts/a100_p32_window/qwen38_newmain_all_03144a22.json`. Its Ampere-only
geomean is `0.068735 ms`; per-M geomeans are `0.057361`, `0.060111`,
`0.068089`, `0.076959`, and `0.084912 ms` for M1/M2/M4/M8/M16. Planar
timings are excluded from every v19 comparison.

The first v19 progression extends the four-output warp split reducer to M1
attention-out `(K,N)=(6144,5120)` and linear-Z `(5120,6144)`. These shapes
launch only 20-24 blocks with the serial reducer; the interleaved warp layout
raises reducer parallelism while retaining four adjacent output loads. In a
matched 60-warmup/2000-iteration pair, all eight medians win: attention-out
improves **1.0931x** and linear-Z improves **1.0682x**, for a combined
**1.0806x** affected geomean speedup (7.457% lower latency). Maximum absolute
error remains below `1.58e-05`. Artifacts are
`artifacts/a100_p32_window/v19_m1_selective_warpreduce_control.json` and
`artifacts/a100_p32_window/v19_m1_selective_warpreduce_candidate.json`.
Affected-case log weighting gives **0.444% cumulative improvement** versus
fetched main.

The second v19 progression specializes the warp reducer geometry for M16
full-KV at split 32. Sixteen adjacent outputs share a warp while two lanes
parallelize each output's split sum, retaining 128 reducer blocks and wider
coalesced loads than the four-output layout. In the
100-warmup/4000-iteration fixed-split screen all four rates win, improving
from `0.033792/0.033792/0.033792/0.034816 ms` to
`0.031744/0.031744/0.032768/0.032768 ms`. The affected geomean speedup is
**1.0556x** (5.267% lower latency), with maximum absolute error below
`1.53e-05`. The control and candidate are
`artifacts/a100_p32_window/v18_m16_fullkv_packed_pgc_control.json` and
`artifacts/a100_p32_window/v19_m16_fullkv_warpreduce16_candidate.json`.
Affected-case log weighting now reaches **0.599% cumulative improvement**
versus fetched main.

The third v19 progression adds an eight-output warp reducer for M2
attention-out and linear-Z. This geometry provides 160-192 blocks while
loading eight adjacent outputs per warp. Against the fetched-main baseline,
the eight retained cases have three wins and five event-quantized ties with
no losses: their affected geomean improves **1.0184x** (1.807% lower
latency), and maximum absolute error stays below `1.91e-05`. The candidate
is `artifacts/a100_p32_window/v19_m2_warpreduce8_candidate.json`; the control
is the M2 subset of `qwen38_newmain_all_03144a22.json`. Affected-case log
weighting now reaches **0.704% cumulative improvement** versus fetched main.

The fourth v19 progression changes M16 full-KV partial-output stores from
two scalar FP32 writes to one aligned `float2` write per pair. With the
16-output warp reducer already shortening the epilogue, W2/W2.5 tie and
W3/W3.5 each improve one event tick, taking the four-rate geomean from
`0.032251 ms` to `0.031744 ms`: a **1.0160x** speedup (1.575% lower latency).
Maximum absolute error remains below `1.53e-05`. Artifacts are
`v19_m16_fullkv_warpreduce16_candidate.json` and
`v19_m16_fullkv_float2_warpreduce16_candidate.json` under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**0.750% cumulative improvement** versus fetched main.

The fifth v19 progression widens the retained M1 projection reducer from
four to 16 adjacent outputs per warp. Against the matched four-output
artifact, attention-out W2 improves from `0.040960` to `0.039936 ms` and
linear-Z W2 improves from `0.040960` to `0.036864 ms`; the other six rates
tie. The affected eight-case geomean speedup is **1.0165x** (1.620% lower
latency), with maximum absolute error below `1.63e-05`. The candidate is
`artifacts/a100_p32_window/v19_m1_warpreduce16_candidate.json` and the
four-output control is `v19_m1_selective_warpreduce_candidate.json` in the
same directory. Affected-case log weighting now reaches **0.844% cumulative
improvement** versus fetched main.

The sixth v19 progression extends the 16-output M1 reducer to long-K
MLP-down split 96/128. In the 60-warmup/2000-iteration screen all four rates
improve one event tick, from `0.083968/0.094208/0.097280/0.095232 ms` to
`0.082944/0.093184/0.096256/0.094208 ms`. That is a **1.0112x** affected
geomean speedup (1.109% lower latency), with maximum absolute error below
`2.48e-05`. The candidate is
`artifacts/a100_p32_window/v19_m1_mlpdown_warpreduce16_candidate.json`; its
control is the M1 MLP-down subset of `qwen38_v19_current_all_0904e361.json`.
Affected-case log weighting now reaches **0.876% cumulative improvement**
versus fetched main.

The seventh v19 progression applies a 16-output warp reducer to M2 long-K
MLP-down. W2 improves from `0.089088` to `0.088064 ms` and W3 from
`0.103424` to `0.102400 ms`; W2.5 and W3.5 tie. The four-case affected
geomean improves **1.0054x** (0.536% lower latency), with maximum absolute
error below `3.47e-05`. The candidate is
`artifacts/a100_p32_window/v19_m2_mlpdown_warpreduce16_candidate.json`; its
control is the M2 MLP-down subset of `qwen38_v19_current_all_0904e361.json`.
Affected-case log weighting now reaches **0.892% cumulative improvement**
versus fetched main.

The eighth v19 progression bypasses the Python autotune-key lookup for the
measured M16 full-KV split-32 plan, matching the existing direct dispatches
for M1/M2/M4/M8 KV. In the matched 60-warmup/2000-iteration pair, W2,
W2.5, and W3.5 improve from `0.036864/0.037888/0.036864 ms` to
`0.033792/0.034816/0.035840 ms`, while W3 ties at `0.034816 ms`. The
affected geomean speedup is **1.0512x** (4.871% lower latency), and output
accuracy is unchanged because both paths launch the same split-32 kernel.
Artifacts are `v19_m16_fastplans_control.json` and
`v19_m16_fastplans_candidate.json` under `artifacts/a100_p32_window/`.
Affected-case log weighting now reaches **1.036% cumulative improvement**
versus fetched main.

The ninth v19 progression adds a direct measured split-48 dispatch for M1
attention-out. Against the accepted 16-output reducer control, W2 improves
from `0.039936` to `0.038912 ms` and the other three rates tie. The
four-case affected geomean speedup is **1.0065x** (0.647% lower latency),
with identical kernel math and accuracy. The candidate is
`artifacts/a100_p32_window/v19_m1_projection_fastplans_candidate.json`; its
control is `v19_m1_warpreduce16_candidate.json`. Affected-case log weighting
now reaches **1.054% cumulative improvement** versus fetched main.

The tenth v19 progression adds a direct measured split-40 dispatch for M4
linear-Z. In the immediate matched pair, W2 improves from `0.045056` to
`0.044032 ms`, W3.5 improves from `0.051200` to `0.050176 ms`, and the
other two rates tie. The affected geomean speedup is **1.0109x** (1.074%
lower latency), with identical split-40 kernel math. Artifacts are
`v19_m4_projection_fastplans_control.json` and
`v19_m4_projection_fastplans_candidate.json` under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**1.086% cumulative improvement** versus fetched main.

The eleventh v19 progression specializes M2 W3 attention-out at split 48
with 16 outputs per reduction warp; all other M2 projection rates retain the
accepted eight-output geometry. The matched 60-warmup/2000-iteration screen
improves from `0.044032` to `0.043008 ms` (**1.0238x**, 2.326% lower
latency), and a 100-warmup/4000-iteration confirmation reproduces
`0.043008 ms` with maximum absolute error below `1.63e-05`. Artifacts are
`v19_m2_warpreduce8_candidate.json`, `v19_m2_warpreduce16_candidate.json`,
and `v19_m2_w3_attention_warpreduce16_final.json`. Affected-case log
weighting now reaches **1.103% cumulative improvement** versus fetched main.

The twelfth v19 progression pins the measured M2 attention-out plans to
split 48 for W2/W2.5/W3 and split 64 for W3.5. This prevents the bounded
first-use tuner from occasionally selecting the event-quantized split-24 W2
plan. Against the final pre-change refresh, W2 improves from `0.043008` to
`0.039936 ms`; W2.5 and W3 tie their accepted values, and a separate
4000-iteration split-64 W3.5 confirmation ties at `0.044032 ms`. The
affected four-case geomean speedup is **1.0187x** (1.836% lower latency).
Artifacts are `v19_m2_attention_fastplan48_final.json` and
`v19_m2_w35_attention_fastplan64_final.json`. Affected-case log weighting
now reaches **1.156% cumulative improvement** versus fetched main.

The thirteenth v19 progression applies the packed two-state PGC16 transform
to M16 long-K MLP-down. Adjacent decoded states share one bank selector, so
the path amortizes the fold and permutation arithmetic while retaining FP32
accumulation and the existing split-32 launch. Against the matched
60-warmup/2000-iteration control, all four rates improve one event tick:
`0.146432/0.148480/0.148480/0.149504 ms` becomes
`0.145408/0.147456/0.147456/0.147456 ms`. The affected geomean speedup is
**1.0087x** (0.863% lower latency), with maximum absolute error below
`9.54e-05`. Artifacts are
`v19_m16_packed_mlpdown_control.json` and
`v19_m16_packed_fullq_mlpdown_candidate.json` under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**1.181% cumulative improvement** versus fetched main.

The fourteenth v19 progression narrows the same packed PGC16 transform to
M16 W2 full-Q. Against the matched 60-warmup/2000-iteration control, W2
improves from `0.107520` to `0.106496 ms` (**1.0096x**, 0.952% lower
latency). W2.5 and W3.5 were median-neutral in the broad diagnostic and W3
lost one event tick, so all three retain the prior decoder. Maximum absolute
error for the accepted W2 path is `2.68e-05`. Artifacts are
`v19_m16_packed_fullq_control.json` and
`v19_m16_packed_fullq_mlpdown_candidate.json` under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**1.188% cumulative improvement** versus fetched main.

The fifteenth v19 progression enables packed two-state PGC16 decode for M16
W3 linear-QKV only. In a matched 100-warmup/4000-iteration pair, latency
falls from `0.094208` to `0.093184 ms` (**1.0110x**, 1.087% lower latency),
with maximum absolute error `2.87e-05`. Other rates retain the existing
decoder. Artifacts are `v19_m16_w3_linearq_packed_control.json` and
`v19_m16_w3_linearq_packed_candidate.json` under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**1.196% cumulative improvement** versus fetched main.

The sixteenth v19 progression simplifies circular-window wrap selection for
eight attention-out specializations. A wrap can only occur in the first
`floor(32 / transition_bits)` pair positions, so these kernels compare the
pair directly instead of reconstructing and comparing the next word index.
The retained set is all four M1 rates, M2 W2, M4 W3, and M8 W3/W3.5. Against
the matched 40-warmup/1000-iteration control, every retained case wins; the
affected geomean speedup is **1.0423x** (4.060% lower latency). Separate
100-warmup/4000-iteration confirmations measure
`0.036864/0.040960/0.040960/0.040960 ms` for M1, `0.039936 ms` for M2 W2,
`0.050176 ms` for M4 W3, and `0.055296 ms` for both M8 rates. Maximum
absolute error stays below `2.29e-05`, and the complete 38-case Ampere suite
passes. Artifacts are `v19_wrap_pair_predicate_attention_{control,candidate}.json`
and the `v19_selective_wrap_attention_*_deep.json` confirmations under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**1.436% cumulative improvement** versus fetched main.

The seventeenth v19 progression extends the direct pair-wrap predicate to
six additional M1 cases. Full-KV W2/W2.5/W3 improve from
`0.033792/0.034816/0.032768 ms` to
`0.032768/0.032768/0.031744 ms`; W3 full-Q, linear-QKV, and MLP-gate/up
improve from `0.069632/0.059392/0.091136 ms` to
`0.066560/0.057344/0.088064 ms`. W3 linear-Z is neutral at `0.041984 ms`.
Across the seven measured retained cases the affected geomean speedup is
**1.0345x** (3.338% lower latency), with maximum absolute error below
`4.01e-05`. The broad candidate exposed regressions for full-KV W3.5 and
W3 MLP-down, so both are compile-time excluded; narrowed confirmations
restore their controls at `0.032768` and `0.092160 ms`. Artifacts use the
`v19_wrap_m1_{fullkv,w3_other}_{control,candidate}_deep.json` and
`v19_wrap_m1_{fullkv_w35,mlpdown_w3}_narrow_verify.json` names under
`artifacts/a100_p32_window/`. The complete 38-case Ampere suite passes.
Affected-case log weighting now reaches **1.608% cumulative improvement**
versus fetched main.

The eighteenth v19 progression extends the direct pair-wrap predicate to
15 selected M8 WMMA cases. The retained set is full-Q W3/W3.5, all four
linear-QKV and linear-Z rates, all four MLP-down rates, and MLP-gate/up
W3.5. Thirteen medians improve and two tie: full-Q falls from
`0.097280/0.098304 ms` to `0.096256/0.097280 ms`; linear-QKV improves by
one event tick at W2/W2.5/W3.5; every linear-Z rate improves by one tick;
MLP-down improves by one tick at W2/W2.5 and two ticks at W3.5; and
MLP-gate/up W3.5 falls from `0.136192` to `0.135168 ms`. W3 linear-QKV and
MLP-down are neutral. The affected geomean speedup is **1.0114x** (1.127%
lower latency), with maximum absolute error below `7.63e-05`, and the
complete 38-case Ampere suite passes. Matched artifacts are
`v19_wrap_m8_selected_{control,candidate}_{a,b}.json` under
`artifacts/a100_p32_window/`. The broad diagnostic is
`v19_wrap_m8_all_{control,candidate}.json`; its full-KV regressions were
excluded, while full-Q W2.5, attention-out W2.5, and MLP-gate/up W3 failed
to reproduce their screen wins at 2000 iterations and were narrowed out.
Affected-case log weighting now reaches **1.732% cumulative improvement**
versus fetched main.

The nineteenth v19 progression extends the direct pair-wrap predicate to
five selected M4 scalar cases. In the matched 60-warmup/2000-iteration pair,
full-Q W2.5/W3 improve from `0.088064/0.089088 ms` to
`0.087040/0.088064 ms`, linear-QKV W2.5 improves from `0.074752` to
`0.073728 ms`, linear-Z W3 improves from `0.051200` to `0.050176 ms`, and
MLP-gate/up W2.5 improves from `0.116736` to `0.113664 ms`. The affected
geomean speedup is **1.0169x** (1.664% lower latency), with maximum absolute
error below `1.91e-05`, and the complete 38-case Ampere suite passes.
Artifacts are `v19_wrap_m4_selected_{control,candidate}_deep.json` under
`artifacts/a100_p32_window/`. The broad diagnostic is
`v19_wrap_m4_all_{control,candidate}.json`; its full-Q W3.5, full-KV W3.5,
attention-out W2.5, linear-QKV W3.5, linear-Z W2/W3.5, MLP-gate/up W3.5,
and MLP-down W3/W3.5 losses were excluded. MLP-down W2.5 was narrowed out
after its screen win became neutral in the deep pair. Affected-case log
weighting now reaches **1.793% cumulative improvement** versus fetched main.

The twentieth v19 progression extends the direct pair-wrap predicate to
five selected M2 scalar cases. In matched 60-warmup/2000-iteration runs,
full-KV W3 improves from `0.034816` to `0.033792 ms`, MLP-gate/up W3
improves from `0.098304` to `0.097280 ms`, and long-K MLP-down
W2.5/W3/W3.5 improves from `0.100352/0.102400/0.105472 ms` to
`0.096256/0.097280/0.100352 ms`. The affected geomean speedup is
**1.0373x** (3.595% lower latency), with maximum absolute error below
`3.91e-05`, and the complete 38-case Ampere suite passes. Matched artifacts
use the `v19_wrap_m2_selected_{control,candidate}_` prefix under
`artifacts/a100_p32_window/`. The broad
`v19_wrap_m2_all_{control,candidate}.json` screen also suggested full-KV
W2/W3.5 and projection wins, but the deep pair made those regress or tie;
full-Q W2.5/W3.5, attention-out W3.5, linear-QKV W2.5/W3.5, linear-Z W3.5,
and MLP-gate/up W2.5/W3.5 were already screen regressions. All are excluded.
Affected-case log weighting now reaches **1.926% cumulative improvement**
versus fetched main.

The twenty-first v19 progression enables packed two-state PGC16 decode for
M8 MLP-gate/up, complementing its selective pair-wrap path. In a matched
100-warmup/4000-iteration pair, W2 improves from `0.132096` to
`0.131072 ms`, W3 from `0.135168` to `0.134144 ms`, and W3.5 from
`0.136192` to `0.134144 ms`; W2.5 ties at `0.135168 ms`. The affected
geomean speedup is **1.0077x** (0.761% lower latency), with maximum absolute
error below `3.63e-05`, and the complete 38-case Ampere suite passes.
Artifacts are `v19_m8_gate_packed_{control,candidate}_deep.json` under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**1.948% cumulative improvement** versus fetched main.

The twenty-second v19 progression combines pair-wrap selection with packed
PGC16 decode for the remaining M8 MLP-gate/up rates. Relative to the packed
control, W2 improves from `0.131072` to `0.130048 ms`, W2.5 from `0.135168`
to `0.134144 ms`, and W3 from `0.134144` to `0.133120 ms`; W3.5 ties at
`0.134144 ms`. The affected geomean speedup is **1.0058x** (0.576% lower
latency), with maximum absolute error below `4.58e-05`. The candidate is
`v19_m8_gate_packed_wrap_candidate_deep.json`; its control is
`v19_m8_gate_packed_candidate_deep.json` under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**1.965% cumulative improvement** versus fetched main.

The twenty-third v19 progression replaces the W2 circular next-word
compare/select with the power-of-two mask `(first_word + 1) & 15` for five
deep-confirmed cases. M2 full-KV improves from `0.033792` to `0.032768 ms`,
M4 long-K MLP-down from `0.106496` to `0.105472 ms`, M8 full-Q from
`0.095232` to `0.094208 ms`, M8 full-KV from `0.033792` to `0.032768 ms`,
and M8 linear-QKV from `0.082944` to `0.081920 ms`. The affected geomean
speedup is **1.0191x** (1.871% lower latency), with maximum absolute error
below `4.20e-05`, and the complete 38-case Ampere suite passes. Deep
fixed-split artifacts use the `v19_w2_pow2_wrap_*_deep.json` names under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**2.034% cumulative improvement** versus fetched main.

The initial 35-case W2 screen was mixed, so the bitmask path was narrowed
to those five wins. M2 MLP-gate/up, M4 full-KV, and M16 full-KV tied their
controls in deep runs and remain unchanged. A later all-35 selective run
entered a queue-stall cluster after M4 MLP-down; those affected samples were
discarded in favor of explicit known-good-split deep runs. Combined packed
plus pair-wrap M16 MLP-gate/up was also rejected after W2 tied and W2.5/W3
regressed. Giving packed M8 MLP-gate/up a static K regressed W2/W3/W3.5,
and fixed split 8 versus 16 tied at W2.5. Diagnostics are
`v19_w2_pow2_wrap_{control,candidate,selective_final}.json`,
`v19_m16_gate_packed_wrap_{control,candidate}_deep.json`,
`v19_m8_gate_static_k_candidate_deep.json`, and
`v19_m8_gate_wrap_split{8,16}_deep.json`.

The twenty-fourth v19 progression retunes the newly packed and pair-wrapped
M8 MLP-gate/up kernel to split 10. In a 100-warmup/4000-iteration run, W2
improves from `0.130048` to `0.128000 ms`, W2.5 from `0.134144` to
`0.131072 ms`, W3 from `0.133120` to `0.130048 ms`, and W3.5 from
`0.134144` to `0.131072 ms`. The affected geomean speedup is **1.0216x**
(2.116% lower latency), with maximum absolute error below `4.39e-05`.
Because all four rates select the same wave, the measured plan bypasses the
first-use autotuner directly. The deep candidate is
`v19_m8_gate_split10_candidate_deep.json`; its immediate packed/pair-wrap
control is `v19_m8_gate_packed_wrap_candidate_deep.json` under
`artifacts/a100_p32_window/`. The normal default-dispatch retry reproduces
W2 at `0.128000 ms` over 8000 iterations; an earlier first-case interval that
held at `0.258048 ms` was discarded as a queue/clock stall after the isolated
retry. Affected-case log weighting now reaches
**2.096% cumulative improvement** versus fetched main.

Four structural follow-ups were rejected before this progression. Hoisting
W3.5 window-lane plans, as in the Hopper kernel, made M8 mostly neutral but
regressed every M16 shape by roughly 3-5% because the longer live geometry
reinforced Ampere's register occupancy limit. A Hopper-style atomic split
epilogue lost one event tick on M16 full-Q W2-W3 and tied MLP-down, while
small-N full-KV regressed sharply. A two-warp/N32 M16 CTA duplicated enough
activation and staging traffic to lose 9-11% versus N64. Finally, reflecting
the symmetric PGC16 level table into its positive half and a divergent W2
single-word extraction path each added more integer/control cost than the
L1/shared traffic they removed. Diagnostics are
`v19_w35_laneplan_{control,candidate}_deep.json`,
`v19_m16_atomic_narrow_{control,candidate}_deep.json`,
`v19_m16_fullq_n32_candidate_deep.json`,
`v19_m16_fullq_symmetric_levels_candidate_deep.json`, and
`v19_w2_single_word_m16_candidate_deep.json`.

The twenty-fifth v19 progression retunes pair-wrapped M1 full-KV from split
64 to split 56. In a matched 100-warmup/4000-iteration pair, W2 improves
from `0.031744` to `0.030720 ms`, W3 from `0.033792` to `0.032768 ms`, and
W3.5 from `0.032768` to `0.030720 ms`; W2.5 ties at `0.032768 ms`. The
affected geomean speedup is **1.0325x** (3.152% lower latency), with maximum
absolute error below `1.05e-05`. The control and candidate are
`v19_m1_kv_wrap_resplit_s64_control_deep.json` and
`v19_m1_kv_wrap_resplit_s56_deep.json` under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**2.190% cumulative improvement** versus fetched main.

Two post-transform wave retunes were rejected. M8 full-Q and linear-QKV at
splits 12, 14, and 18 were uniformly slower than the retained split 16.
Packed M16 long-K MLP-down at splits 28 and 36 was likewise slower than the
retained split 32. Diagnostics use the `v19_m8_q_qkv_resplit_` and
`v19_m16_down_packed_resplit_` prefixes.

The adjacent full-KV wave did not extend beyond M1. Split 56 regressed M2
W2.5/W3 and M4 W2.5-W3.5. Its apparent M4 W2 screen win also failed the
isolated 200-warmup/8000-iteration check, which measured `0.033792 ms`
against the split-64 control's `0.031744 ms`. M2 pair-wrapped long-K
MLP-down at split 80 was uniformly slower than its retained rate-specific
plans. Diagnostics are `v19_m24_kv_wrap_resplit_s{56,64}_deep.json`,
`v19_m4_w2_kv_wrap_resplit_s56_verify.json`, and
`v19_m2_down_wrap_resplit_s80_deep.json`.

The twenty-sixth v19 progression retunes the packed M16 W3 linear-QKV case
from split 16 to split 10. The matched 100-warmup/4000-iteration control and
candidate improve from `0.093184` to `0.092160 ms`, a **1.0111x** speedup
(1.099% lower latency), with maximum absolute error below `3.82e-05`.
Split 12 measured `0.094208 ms` and was rejected. Artifacts are
`v19_m16_w3_qkv_packed_resplit_s{10,12,16}_deep.json` under
`artifacts/a100_p32_window/`. Affected-case log weighting now reaches
**2.198% cumulative improvement** versus fetched main.

Three register/barrier experiments were rejected before this progression.
Serializing the two M16 N8 weight fragments left both ptxas allocation at 64
registers and W3.5 full-Q latency at `0.109568 ms`; ptxas had already
shortened those live ranges. A nine-CTA launch bound forced 54 registers with
zero spills but regressed the same case to `0.119808 ms`, showing that the
compiler's recomputation/schedule cost exceeded the occupancy gain. A
two-warp-pair staging design duplicated activation data only twice rather
than the rejected warp-private design's four copies, but its named-barrier
path still regressed to `0.147456 ms`. Diagnostics are
`v19_m16_serial_weight_fragment_candidate_screen.json`,
`v19_m16_fullq_minblocks9_candidate_screen.json`, and
`v19_m16_fullq_pair_private_stage_candidate_screen.json`.

The twenty-seventh v19 progression extends the packed M16 linear-QKV split-10
plan to W2. The fixed-split 100-warmup/4000-iteration screen improves from
`0.092160` at split 16 to `0.091136 ms` at split 10, a **1.0112x** speedup
(1.111% lower latency); split 12 regresses to `0.094208 ms`. A reversed
200-warmup/8000-iteration confirmation reproduces both winning and control
medians exactly, with maximum absolute error below `4.20e-05`. Artifacts are
`v19_m16_w2_qkv_resplit_s{10,12,16}_deep.json` and the corresponding
`s{10,16}_verify8000.json` pair under `artifacts/a100_p32_window/`. Affected-
case log weighting now reaches **2.206% cumulative improvement** versus
fetched main. The adjacent W2 full-Q split-10 and split-12 experiments both
measured `0.107520 ms` against the split-16 control's `0.106496 ms` and were
rejected; their diagnostics are `v19_m16_w2_fullq_resplit_s{10,12,16}_deep.json`.

The twenty-eighth v19 progression completes the M16 linear-QKV retune for
W2.5 and W3.5, making split 10 the measured plan for every supported rate on
that exact geometry. The 100-warmup/4000-iteration screen improves W2.5 from
`0.093184` to `0.092160 ms` and W3.5 from `0.094208` to `0.092160 ms`. In the
candidate-first 200-warmup/8000-iteration reversal, split 10 reproduces
`0.092160 ms` for both rates while the conservative split-16 control measures
`0.093184 ms` for both: a **1.0111x** speedup (1.099% lower latency), with
maximum absolute error below `3.63e-05`. Artifacts are
`v19_m16_qkv_remaining_resplit_s{10_candidate,16_control}_deep.json` and the
corresponding `s{10,16}_verify8000.json` pair. Affected-case log weighting now
reaches **2.222% cumulative improvement** versus fetched main.

The adjacent M16 full-Q wave audit found no additional rate-specific win.
Against split 16, split 10 and split 12 both tie W2.5 at `0.109568 ms`,
regress W3 from `0.108544` to `0.109568 ms`, and regress W3.5 from
`0.109568` to `0.110592 ms`; the earlier W2 audit also rejected both shorter
waves. The split-16 policy is retained for every full-Q rate. Diagnostics are
`v19_m16_fullq_remaining_resplit_s{10_candidate,12_candidate,16_control}_deep.json`.

Two further wave audits were rejected. M16 attention-out split 12 measures
`0.061440 ms` at every rate; split 16 regresses to `0.062464-0.063488 ms`
and split 10 to `0.064512-0.065536 ms`. For M8 full-KV, split 40 initially
appeared to improve W2 from `0.032768` to `0.031744 ms`, but the reversed
200-warmup/8000-iteration control also measured `0.031744 ms`; W2.5
regressed at split 40, while split 56 regressed all four rates to
`0.033792 ms`. The retained plans therefore remain split 12 and split 48,
respectively. Diagnostics use the `v19_m16_attention_resplit_` and
`v19_m8_fullkv_resplit_` prefixes, including the W2 `verify8000` pair.

The remaining post-transform full-KV wave checks were also rejected. M4
split 48 regresses every rate versus split 64, from `0.030720-0.032768 ms`
to `0.031744-0.035840 ms`. M16 split 24 and split 40 both regress every
rate versus split 32: the retained plan measures `0.031744-0.032768 ms`,
while the alternatives measure `0.033792-0.034816 ms`. Diagnostics are
`v19_m4_fullkv_resplit_s{48_candidate,64_control}_deep.json` and
`v19_m16_fullkv_resplit_s{24_candidate,32_control,40_candidate}_deep.json`.

The twenty-ninth v19 progression gives the exact M16 full-Q geometry a
Marlin-style N128 CTA. Each of the four warps now owns two N16 tiles and
reuses one `ldmatrix` activation fragment across both pairs of `mma.sync`
instructions, halving activation staging/CTA ownership without relying on
Hopper TMA. In the candidate-first 200-warmup/8000-iteration reversal, the
conservative W2/W2.5/W3/W3.5 controls measure
`0.106496/0.108544/0.108544/0.109568 ms`, while the wide kernel measures
`0.101376/0.104448/0.104448/0.105472 ms`. The affected geomean speedup is
**1.0419x** (4.024% lower latency), maximum absolute error stays below
`3.44e-05`, and affected-case log weighting reaches **2.342% cumulative
improvement** versus fetched main. Artifacts are
`v19_m16_fullq_wide_n128_candidate_{deep,verify8000}.json`,
`v19_m16_fullq_n64_control_verify8000.json`, and the isolated
`v19_m16_fullq_n64_w35_control_verify8000_retry.json` that replaces the
control sweep's contaminated `0.245760 ms` W3.5 interval.

Hoisting `ldmatrix` before decode for every WMMA route was narrowed out: it
cost M16 linear-QKV W2 and W3.5 one event tick. The retained template hoists
and reuses the fragment only for the two-tile specialization; the original
one-tile schedule remains intact. The QKV W2.5 isolated retry reproduces
`0.092160 ms`, and the M8 MLP-gate/up canary matches or improves all four
accepted medians. Diagnostics are
`v19_wide_n128_m16_qkv_nonregression_{deep,v2_deep}.json`,
`v19_wide_n128_m16_qkv_w25_nonregression_retry8000.json`, and
`v19_wide_n128_m8_gate_nonregression_deep.json`.

The thirtieth v19 progression retunes the wider M16 full-Q CTA from split
16 to split 10 and pins the known geometry so it bypasses first-use
autotuning. The default in-memory tuner independently selected split 10 at
all four rates, and the fixed 200-warmup/8000-iteration confirmation
reproduces `0.101376/0.103424/0.103424/0.105472 ms`. Relative to the
pre-wide split-16 controls, the finalized affected geomean speedup is
**1.0471x** (4.496% lower latency), and affected-case log weighting reaches
**2.356% cumulative improvement** versus fetched main. Diagnostics are
`v19_m16_fullq_wide_n128_autotune_audit_deep.json` and
`v19_m16_fullq_wide_n128_split10_verify8000.json`.

Extending the N128 CTA to M16 linear-QKV was rejected. Against the retained
N64 split-10 medians of `0.091136-0.092160 ms`, the wide candidate measures
`0.092160/0.094208/0.093184/0.094208 ms`; every rate regresses because the
smaller N already supplies enough CTA parallelism and cannot amortize the
second accumulator pair. The diagnostic is
`v19_m16_qkv_wide_n128_candidate_deep.json`.

The thirty-first v19 progression extends the Marlin-style N128 CTA to M16
MLP-gate/up, whose larger N amortizes the second accumulator pair. In the
matched 100-warmup/4000-iteration control and candidate, the narrow kernel
measures `0.149504/0.151552/0.152576/0.153600 ms`; the candidate-first
200-warmup/8000-iteration confirmation measures
`0.137216/0.139264/0.140288/0.141312 ms`. The affected geomean speedup is
**1.0881x** (8.095% lower latency), maximum absolute error stays below
`3.63e-05`, and affected-case log weighting reaches **2.604% cumulative
improvement** versus fetched main. Artifacts are
`v19_m16_gate_n64_control_deep.json` and
`v19_m16_gate_wide_n128_candidate_{deep,verify8000}.json`.

The thirty-second v19 progression retunes the wider M16 MLP-gate/up CTA from
split 16 to split 10 and pins the exact geometry. The fixed
200-warmup/8000-iteration confirmation measures
`0.135168/0.137216/0.138240/0.139264 ms`, a **1.0149x** incremental
speedup (1.468% lower latency). Combined with N128 ownership, this is a
**1.1043x** speedup (9.445% lower latency) versus the narrow split-16
control, taking affected-case log weighting to **2.647% cumulative
improvement** versus fetched main. Split 12 regresses every rate to
`0.141312-0.145408 ms` and is rejected. Diagnostics are
`v19_m16_gate_wide_n128_split{10,12}_deep.json` and
`v19_m16_gate_wide_n128_split10_verify8000.json`.

The thirty-third v19 progression extends N128 ownership to the M16 long-K
MLP-down projection. Although N=5120 is smaller than the rejected QKV route,
K=17408 reuses each activation fragment over enough reduction work to repay
the second accumulator pair. The matched split-32 narrow control measures
`0.145408/0.147456/0.146432/0.148480 ms`; the conservative
200-warmup/8000-iteration candidate measures
`0.140288/0.143360/0.143360/0.144384 ms`. The affected geomean speedup is
**1.0287x** (2.790% lower latency), maximum absolute error remains below
`8.40e-05`, and affected-case log weighting reaches **2.730% cumulative
improvement** versus fetched main. Artifacts are
`v19_m16_down_n64_control_deep.json` and
`v19_m16_down_wide_n128_candidate_{deep,verify8000}.json`.

The thirty-fourth v19 progression retunes the wider M16 long-K projection
from split 32 to split 24 and pins the exact geometry. The fixed
200-warmup/8000-iteration confirmation measures
`0.138240/0.141312/0.141312/0.142336 ms`, a **1.0145x** incremental
speedup (1.434% lower latency). Combined with N128 ownership this is a
**1.0437x** speedup (4.184% lower latency) versus the narrow split-32
control, taking affected-case log weighting to **2.772% cumulative
improvement** versus fetched main. Split 20 regresses every rate to
`0.146432-0.150528 ms` and is rejected. Diagnostics are
`v19_m16_down_wide_n128_split{20,24}_deep.json` and
`v19_m16_down_wide_n128_split24_verify8000.json`.

The thirty-fifth v19 progression extends N128 ownership to M8 MLP-gate/up
while retaining the 128-thread CTA. This differs from the rejected v16
experiment, which doubled both N ownership and the CTA to 256 threads and
regressed by 6-12%. Against the accepted narrow split-10 control medians of
`0.128000/0.130048/0.130048/0.131072 ms`, the
200-warmup/8000-iteration candidate measures
`0.116736/0.119808/0.119808/0.121856 ms`. The affected geomean speedup is
**1.0857x** (7.897% lower latency), maximum absolute error stays below
`4.39e-05`, and affected-case log weighting reaches **3.014% cumulative
improvement** versus fetched main. Artifacts are
`v19_wide_n128_m8_gate_nonregression_deep.json` and
`v19_m8_gate_wide_n128_128t_candidate_{v2_deep,verify8000}.json`.

The first M8 prototype exposed a correctness bug rather than a performance
result: the wide bank-ID stage copied eight IDs only for full-row kernels,
leaving the second M8 tile group uninitialized. Extending that stage to
fixed active-row kernels restores exactness before timing; the invalid
artifact without the `v2` suffix is retained only as a failed diagnostic.

The thirty-sixth v19 progression extends the corrected 128-thread N128 M8
path to full-Q. Against the fresh split-16 narrow control medians of
`0.094208/0.097280/0.096256/0.097280 ms`, the
200-warmup/8000-iteration candidate measures
`0.089088/0.092160/0.091136/0.093184 ms`. The affected geomean speedup is
**1.0533x** (5.058% lower latency), maximum absolute error stays below
`3.44e-05`, and affected-case log weighting reaches **3.167% cumulative
improvement** versus fetched main. Artifacts are
`v19_m8_fullq_n64_control_deep.json` and
`v19_m8_fullq_wide_n128_128t_candidate_{deep,verify8000}.json`.

Retuning that wider M8 full-Q path to split 10 and split 12 was rejected.
Their respective medians were
`0.092160/0.094208/0.093184/0.093184 ms` and
`0.092160/0.095232/0.094208/0.094208 ms`, both slower than the retained
split-16 geometry. Diagnostics are
`v19_m8_fullq_wide_n128_split{10,12}_deep.json`.

The thirty-seventh v19 progression extends 128-thread N128 ownership to M8
long-K MLP-down. Against the fresh narrow split-32 control medians of
`0.130048/0.132096/0.132096/0.133120 ms`, the
200-warmup/8000-iteration candidate measures
`0.123904/0.126976/0.126976/0.128000 ms`. The affected geomean speedup is
**1.0426x** (4.081% lower latency), maximum absolute error stays below
`7.25e-05`, and affected-case log weighting reaches **3.290% cumulative
improvement** versus fetched main. Artifacts are
`v19_m8_down_n64_control_deep.json` and
`v19_m8_down_wide_n128_candidate_{deep,verify8000}.json`.

The thirty-eighth v19 progression retunes that wider M8 long-K route from
split 32 to split 40 and pins the measured geometry. The fixed
200-warmup/8000-iteration confirmation measures
`0.118784/0.121856/0.121856/0.123904 ms`, a **1.0400x** incremental
speedup (3.850% lower latency). Combined with N128 ownership this is a
**1.0843x** speedup (7.774% lower latency) versus the narrow split-32
control, taking affected-case log weighting to **3.406% cumulative
improvement** versus fetched main. Split 24 regresses three rates and ties
W3.5, so it is rejected. Diagnostics are
`v19_m8_down_wide_n128_split{24,40}_deep.json` and
`v19_m8_down_wide_n128_split40_verify8000.json`.

The thirty-ninth v19 progression extends the 128-thread N128 layout to the
measured M8 attention-out tuple `(K,N)=(6144,5120)`. Against the fresh
narrow split-32 control medians of `0.055296/0.056320/0.055296/0.056320 ms`,
the 200-warmup/8000-iteration candidate measures
`0.052224/0.054272/0.053248/0.053248 ms`. The affected geomean speedup is
**1.0481x** (4.592% lower latency), maximum absolute error stays below
`2.48e-05`, and affected-case log weighting reaches **3.545% cumulative
improvement** versus fetched main. The specialization is restricted to
that exact K/N pair; other M8/N5120 inputs retain the original narrow path.
Artifacts are `v19_m8_attention_n64_control_deep.json` and
`v19_m8_attention_wide_n128_candidate_{deep,verify8000}.json`.

The fortieth v19 progression applies the same layout to the measured M8
linear-Z tuple `(K,N)=(5120,6144)`. Against the fresh narrow split-32
control medians of `0.055296/0.056320/0.055296/0.056320 ms`, the
200-warmup/8000-iteration candidate measures
`0.052224/0.052224/0.052224/0.053248 ms`. The affected geomean speedup is
**1.0634x** (5.963% lower latency), maximum absolute error stays below
`1.72e-05`, and affected-case log weighting reaches **3.727% cumulative
improvement** versus fetched main. The specialization is restricted to
that exact K/N pair; other M8/N6144 inputs retain the original narrow path.
Artifacts are `v19_m8_linearz_n64_control_deep.json` and
`v19_m8_linearz_wide_n128_candidate_{deep,verify8000}.json`.

The forty-first v19 progression extends the two-tile layout to the measured
M8 linear-QKV tuple `(K,N)=(5120,10240)`, retaining its static-K and W2
power-of-two wrap specializations. Against the fresh narrow split-16 control
medians of `0.081920/0.083968/0.083968/0.083968 ms`, the
200-warmup/8000-iteration candidate measures
`0.077824/0.078848/0.078848/0.079872 ms`. The affected geomean speedup is
**1.0584x** (5.520% lower latency), maximum absolute error stays below
`2.87e-05`, and affected-case log weighting reaches **3.895% cumulative
improvement** versus fetched main. The specialization is restricted to
that exact K/N pair; other M8/N10240 inputs retain the original narrow path.
Artifacts are `v19_m8_qkv_n64_control_deep.json` and
`v19_m8_qkv_wide_n128_candidate_{deep,verify8000}.json`.

The forty-second v19 progression retunes the widened M8 attention-out route
from split 32 to split 24. The fixed 200-warmup/8000-iteration confirmation
measures `0.051200/0.053248/0.053248/0.052224 ms`, a **1.0147x** incremental
speedup (1.446% lower latency) over split 32 and a **1.0635x** combined
speedup (5.972% lower latency) versus the narrow control. The cumulative
affected-case log weighting reaches **3.939% improvement** versus fetched
main. Split 24 is retained; its error remains below `2.67e-05`. Diagnostic
artifacts are `v19_m8_attention_wide_n128_split24_{deep,verify8000}.json`.

The forty-third v19 progression retunes widened M8 linear-Z from split 32 to
split 40. Split 24 regresses all four rates, while split 40 confirms
`0.051200/0.052224/0.052224/0.053248 ms` at 8,000 iterations: a **1.0050x**
incremental speedup (0.494% lower latency) over split 32. The cumulative
affected-case log weighting reaches **3.953% improvement** versus fetched
main. Diagnostic artifacts are
`v19_m8_linearz_wide_n128_split{24,40}_{deep,verify8000}.json`.

The forty-fourth v19 progression retunes widened M8 full-Q from split 16 to
split 14. The fixed 200-warmup/8000-iteration confirmation measures
`0.089088/0.092160/0.091136/0.092160 ms`, a **1.0028x** incremental
speedup (0.276% lower latency) over split 16. The cumulative affected-case
log weighting reaches **3.961% improvement** versus fetched main; maximum
absolute error remains below `3.25e-05`. Diagnostic artifacts are
`v19_m8_fullq_wide_n128_split14_{deep,verify8000}.json`.

The forty-fifth v19 progression applies N128 ownership to the measured M16
attention-out tuple `(K,N)=(6144,5120)`. Against the narrow split-12 control
medians of `0.061440 ms` at every rate, the 200-warmup/8000-iteration
candidate measures `0.060416/0.061440/0.061440/0.061440 ms`: a **1.0042x**
incremental speedup (0.419% lower latency) with no rate regression. Maximum
absolute error remains below `4.01e-05`, and cumulative affected-case log
weighting reaches **3.974% improvement** versus fetched main. The exact
M16/K6144/N5120 plan is pinned at split 12. Artifacts are
`v19_m16_attention_n64_control_deep.json` and
`v19_m16_attention_wide_n128_candidate_{deep,verify8000}.json`.

The forty-sixth v19 progression applies the exact power-of-two circular-word
wrap to W2 in the widened M16 full-Q tuple `(K,N)=(5120,12288)`. The W2
specialization replaces the generic next-word boundary compare with a mask;
W2.5, W3, and W3.5 retain the prior instruction path. In a matched
100-warmup/4000-iteration pair, W2 improves from `0.100352` to `0.097280 ms`
(1.0316x), while the other three rates tie at their control medians. Maximum
absolute error remains below `4.01e-05`, and all 55 Ampere tests pass. The
affected-case log weighting reaches **3.997% cumulative improvement** versus
fetched main. Artifacts are `v19_m16_fullq_pow2_control.json` and
`v19_m16_fullq_pow2_candidate.json`.

The same W2 mask was screened on M16 attention-out and linear-QKV, plus M8
attention-out, linear-Z, and MLP-gate/up. M16 attention and M8 linear-Z tied
all four rates. M8 gate/up W2 tied, while W2.5 appeared one tick faster and
W3.5 one tick slower; since the specialization is W2-only, those unrelated
rate changes are treated as timer noise and the probe is rejected. M16
linear-QKV tied W2/W2.5/W3; an apparent W3.5 one-tick improvement did not
identify a changed instruction path because the power-of-two specialization is
W2-only. M8 attention tied W2/W2.5/W3 and lost one W3.5 tick. None of these
probes is dispatched. Diagnostics are
`v19_m16_attention_pow2_{control,candidate}.json`,
`v19_m16_qkv_pow2_{control,candidate,verify8000}.json`, and
`v19_m8_{attention,linearz,gate}_pow2_{control,candidate}.json`.

Extending the scalar W2 mask to M1 full-Q also failed to produce a clean
rate-wise win: W2/W2.5 tied, while W3 lost one event tick. The fixed-N scalar
launcher therefore retains its existing wrap policy. The diagnostic pair is
`v19_m1_fullq_pow2_{control,candidate}.json`.

The same scalar W2 mask on M1 long-K MLP-down tied all four rates, so the
already tuned four-K16 stage keeps its existing generic wrap. Its diagnostic
pair is `v19_m1_down_pow2_{control,candidate}.json`.

The M2 full-Q scalar W2 mask likewise tied all four rates and was rejected;
the existing scalar wrap policy remains in place. Its diagnostic pair is
`v19_m2_fullq_pow2_{control,candidate}.json`.

The M4 full-Q scalar W2 mask also tied W2 and showed no attributable gain on
the other rates; it was rejected after fixing an initial temporary parenthesis
error in the test condition. Its diagnostic pair is
`v19_m4_fullq_pow2_{control,candidate}.json`.

The forty-seventh v19 progression applies the exact power-of-two circular-word
wrap to W2 in the widened M16 long-K MLP-down tuple
`(K,N)=(17408,5120)`. Against the split-24 control medians of
`0.138240/0.141312/0.141312/0.142336 ms`, the candidate measures
`0.132096/0.141312/0.141312/0.142336 ms` in the matched
100-warmup/4000-iteration run. The W2 gain is 1.0465x (4.44% lower latency),
the other rates tie, and the W2 result reproduces at 8,000 iterations.
Maximum absolute error remains below `9.54e-05`; cumulative affected-case log
weighting reaches **4.031% improvement** versus fetched main. Artifacts are
`v19_m16_down_pow2_control.json`, `v19_m16_down_pow2_candidate.json`, and
`v19_m16_down_pow2_candidate_verify8000.json`.

The forty-eighth v19 progression applies the same exact W2 mask to the fixed-N
M16 linear-Z tuple `(K,N)=(5120,6144)`. Against split-16 control medians of
`0.062464 ms` at every rate, the candidate measures
`0.061440/0.062464/0.062464/0.062464 ms`; the W2 gain is 1.0167x and
reproduces at 8,000 iterations. Maximum absolute error remains below
`2.67e-05`, and cumulative affected-case log weighting reaches **4.043%
improvement** versus fetched main. Artifacts are
`v19_m16_linearz_pow2_control.json`, `v19_m16_linearz_pow2_candidate.json`,
and `v19_m16_linearz_pow2_candidate_verify8000.json`.

A direct Hopper-style `mad.lo.u32` rewrite of the paired PGC16 multiply was
also rejected on Ampere: M16 full-Q W2 regressed from `0.097280` to
`0.100352 ms` while preserving exactness. The compiler’s existing integer
schedule is retained; the diagnostic is
`v19_m16_decode_mad_candidate.json`.

## v20: post-PR-93 continuation

PR #93 was merged at `d66ea238`; the next optimization window starts from
that exact `origin/main` tip. A first Marlin-inspired scalar CTA-width probe
is rejected. M1 full-Q was changed from sixteen to 32 N16 tiles per CTA and
launched with 256 threads so all tiles remained covered. The corrected probe
was exact, but at split 40 its 100-warmup/2000-iteration medians changed from
the 128-thread control `0.061440/0.068608/0.067584/0.068608 ms` to
`0.063488/0.067584/0.067584/0.069632 ms` for W2/W2.5/W3/W3.5: W2 and W3.5
regressed while only W2.5 improved. The source change was reverted; the
diagnostic is `artifacts/a100_p32_window/v20_m1_fullq_wide32_256t_candidate.json`.

The same 256-thread/32-tile geometry was screened on M1 MLP-gate/up
`(K,N)=(5120,17408)`. It remained exact but regressed all four rates at the
100-warmup/2000-iteration split-40 screen: the control was
`0.079872/0.088064/0.089088/0.090112 ms`, while the candidate was
`0.082944/0.089088/0.090112/0.094208 ms`. The source was reverted; see
`artifacts/a100_p32_window/v20_m1_gate_wide32_256t_candidate.json`.

The first accepted v20 specialization applies the direct pair-position wrap
predicate to W2.5 in the widened M16 full-Q tuple `(K,N)=(5120,12288)`. The
predicate removes the generic next-word boundary compare while preserving the
existing W2 power-of-two mask and the W3/W3.5 generic path. Against the
split-10 control `0.103424 ms`, the candidate measures `0.101376 ms` in the
matched 100-warmup/4000-iteration run (1.0202x, 1.984% lower latency), and
the candidate-first 200-warmup/8000-iteration confirmation reproduces
`0.101376 ms`. Maximum absolute error is `3.82e-05`; the all-rate candidate
keeps W2/W3/W3.5 at their control medians. Artifacts are
`v20_m16_fullq_w25_pairwrap_{candidate,control_retry4000}.json` and
`v20_m16_fullq_w25_pairwrap_candidate_verify8000_b.json`.

The second accepted v20 specialization applies the same pair-position wrap
predicate to W2.5 in the widened M16 MLP-gate/up tuple `(K,N)=(5120,17408)`.
The split-10 control measures `0.137216 ms` and the candidate
`0.134144 ms` in the matched 100-warmup/4000-iteration run (1.0229x, 2.239%
lower latency); the candidate reproduces `0.134144 ms` at 8,000 iterations.
Maximum absolute error is `4.01e-05`. The W2/W3/W3.5 paths remain unchanged.
Artifacts are `v20_m16_gate_w25_pairwrap_{candidate,control}.json` and
`v20_m16_gate_w25_pairwrap_candidate_verify8000.json`.

The third accepted v20 specialization applies the pair-position wrap
predicate to W2.5 in widened M16 long-K MLP-down `(K,N)=(17408,5120)`. The
split-24 control is `0.141312 ms`; the candidate is `0.136192 ms` in the
matched 100-warmup/4000-iteration run (1.0376x, 3.624% lower latency), and
reproduces `0.135168 ms` at 8,000 iterations. Maximum absolute error is
`9.92e-05`; W2/W3/W3.5 remain on their existing paths. Artifacts are
`v20_m16_down_w25_pairwrap_{candidate,control}.json` and
`v20_m16_down_w25_pairwrap_candidate_verify8000.json`.

The fourth accepted v20 specialization applies the pair-position wrap
predicate to W2.5 in widened M16 attention-out `(K,N)=(6144,5120)`. The
split-12 control measures `0.061440 ms`; the candidate measures `0.059392 ms`
in the matched 100-warmup/4000-iteration run (1.0345x, 3.333% lower
latency), and reproduces `0.059392 ms` at 8,000 iterations. Maximum absolute
error is `3.43e-05`; the other three rates retain the generic/power-mask
paths. Artifacts are `v20_m16_attention_w25_pairwrap_{candidate,control}.json`
and `v20_m16_attention_w25_pairwrap_candidate_verify8000.json`.

The same predicate was screened on M16 linear-QKV W2.5 `(K,N)=(5120,10240)`
at split 10. It tied the 100-warmup/4000-iteration control at `0.092160 ms`
with exact output, so the source was restored and the path remains unchanged;
the diagnostic is `artifacts/a100_p32_window/v20_m16_qkv_w25_pairwrap_candidate.json`.

The fifth accepted v20 specialization extends the direct pair-position wrap
predicate to W3 (`TransitionBits==6`) in all four widened M16 tuples: full-Q
`(5120,12288)`, attention-out `(6144,5120)`, MLP-gate/up `(5120,17408)`, and
long-K MLP-down `(17408,5120)`. Matched split-10/12/24 controls at 4,000
iterations measured `0.103424/0.061440/0.138240/0.141312 ms`, while the
candidate measured `0.101376/0.059392/0.134144/0.135168 ms` (about
`1.98%/3.33%/2.96%/4.35%` lower latency; 3.26% geometric-mean improvement).
The 8,000-iteration candidate confirmation measured
`0.100352/0.059392/0.134144/0.135168 ms`. Every case remained numerically
exact within the existing tolerances (maximum absolute errors were below
`1.0e-04`). Artifacts are
`v20_m16_w3_pairwrap_wide_{control,screen}.json`,
`v20_m16_w3_pairwrap_attention_control8000.json`,
`v20_m16_w3_pairwrap_down_control8000.json`, and
`v20_m16_w3_pairwrap_wide_verify8000.json`.

The sixth accepted v20 specialization carries the same pair-position wrap
predicate through W3.5 (`TransitionBits==7`) for those four widened M16
tuples. Stable 8,000-iteration controls were
`0.104448/0.061440/0.138240/0.142336 ms` (full-Q, attention-out, gate/up,
down), versus candidate medians
`0.102400/0.060416/0.136192/0.137216 ms`: `1.96%/1.67%/1.48%/3.60%`
lower latency and 2.23% geometric-mean improvement. The 4,000-iteration
screen was directionally consistent, and all outputs stayed within the
existing exactness tolerances (maximum absolute error `9.16e-05`). Artifacts
are `v20_m16_w35_pairwrap_wide_candidate.json`,
`v20_m16_w35_pairwrap_wide_control.json`,
`v20_m16_w35_pairwrap_fullq_control8000.json`, and
`v20_m16_w35_pairwrap_wide_verify8000.json`.

The seventh accepted v20 progression uses one 8-byte `cp.async.ca` for the
eight bank IDs staged by each full-row widened M16 CTA, replacing two 4-byte
copies. The shared bank-ID slab is explicitly 8-byte aligned; M8 active-row
wide CTAs retain the original two-copy path. In matched W2.5 8,000-iteration
runs, full-Q, attention-out, gate/up, and long-K down changed from
`0.101376/0.059392/0.134144/0.135168 ms` to
`0.100352/0.059392/0.133120/0.131072 ms` (about
`1.01%/0%/0.76%/3.03%` lower latency; 1.22% geometric-mean improvement).
The all-rate M16 screen was non-regressing and remained numerically exact;
maximum absolute error was `9.92e-05`. Artifacts are
`v20_m16_ca8_control8000_all.json`, `v20_m16_ca8_w25_verify8000_all.json`,
and `v20_m16_ca8_all_rates_screen.json`.

The eighth accepted v20 progression enables the same 8-byte bank-ID copy only
for the M8 long-K MLP-down tuple `(K,N)=(17408,5120)`; other M8 active-row
wide routes remain on 4-byte copies because attention lost a tick and gate/up
tied in the matched control. At W3.5 and 8,000 iterations, M8 down improves
from `0.122880` to `0.121856 ms` (0.83% lower latency) with exact output.
Artifacts are `v20_m8_ca8_control_w35_verify8000.json` and
`v20_m8_ca8_candidate_w35_verify8000.json` (the 4,000-iteration screen is
`v20_m8_ca8_candidate_w35.json`).

The same M8 down CA8 copy at W3 tied the existing path at `0.120832 ms` and
was not enabled; diagnostic: `v20_m8_down_ca8_w3_candidate.json`.

The W2.5 M8 long-K down CA8 probe was likewise neutral at `0.120832 ms` and
was reverted (`v20_m8_down_ca8_w25_candidate.json`).

The ninth accepted v20 progression specializes `alternate_bank_mask` for the
common `bank_alt_id=3` in full-row widened M16 kernels only, using the
transition-specific constants for W2/W2.5/W3/W3.5. Matched W2.5
8,000-iteration controls/candidates (full-Q, attention-out, gate/up, down)
are `0.101376/0.059392/0.134144/0.135168` and
`0.097280/0.057344/0.129024/0.133120 ms`, respectively: approximately
`4.04%/3.45%/3.82%/1.52%` lower latency and 3.32% geometric-mean gain, with
exact outputs. A global version was rejected after representative M1/M8
routes regressed, so scalar and active-row kernels retain the original
runtime mask path. Artifacts are
`v20_bank_alt3_control8000.json`,
`v20_bank_alt3_fast_all_m16_w25_verify8000.json`,
`v20_bank_alt3_m16_narrow_verify8000.json`, and
`v20_bank_alt3_fast_m1m8_w25.json`.

The initial implementation incorrectly returned the W2.5 mask for every
transition width; the 16-case benchmark caught W3 full-Q corruption
(`max_abs=35.19`) before it was accepted. The fast path now selects the
correct compile-time mask for each width; the corrected full M16 matrix is
`v20_bank_alt3_maskfix_m16_all_rates.json`.

The tenth accepted v20 progression adds a three-K16 software stage to the
widened M16 full-Q tuple `(K,N)=(5120,12288)`. It keeps the two shared stage
buffers but processes three K tiles between handoffs, reducing pipeline
barrier/commit overhead without changing MMA order. In matched 8,000-iteration
runs, stage 2 measured `0.096256/0.096256/0.097280/0.099328 ms` and stage 3
measured `0.092160/0.092160/0.093184/0.094208 ms` for W2/W2.5/W3/W3.5: all
four rates improve, with a 4.69% geometric-mean reduction and exact output
(maximum absolute error `4.01e-05`). The stage-3 template is restricted to
this full-Q dispatch; all other routes retain the proven two-K16 stage.
Artifacts are `v20_m16_fullq_stage3_{control8000,verify8000}.json`.

The eleventh accepted v20 progression applies the same three-K16 stage to the
widened M16 MLP-gate/up tuple `(K,N)=(5120,17408)`. The matched stage-2
4,000-iteration medians were `0.131072/0.129024/0.129024/0.130048 ms`; stage
3 measured `0.119808/0.119808/0.120832/0.121856 ms` (7.10% geometric-mean
improvement), and the 8,000-iteration confirmation remained faster at
`0.119808/0.119808/0.120832/0.122880 ms`. All outputs are exact within the
existing tolerance (maximum absolute error `4.58e-05`). The stage-3 template
is restricted to this gate/up dispatch; other routes retain their measured

The twelfth accepted v20 progression uses a three-K16 stage for widened M16
attention-out `(K,N)=(6144,5120)`. Stage 2's matched medians were
`0.059392/0.057344/0.057344/0.057344 ms`; stage 3 reduced them to
`0.054272/0.052224/0.053248/0.053248 ms` (8.0% geometric-mean improvement).
The 8,000-iteration confirmation reproduced those medians, with exact output
and maximum absolute error `4.01e-05`. This stage depth is restricted to the
M16 attention dispatch; all other kernels keep their existing specialization.
Artifacts are `v20_m16_attention_stage3_{control,verify8000}.json`.

Artifacts are `v20_m16_gate_stage3_{control,verify8000}.json`.

The thirteenth accepted v20 progression applies the three-K16 stage to
widened M16 long-K MLP-down `(K,N)=(17408,5120)`. Stage 2's matched medians
were `0.132096/0.134144/0.133120/0.136192 ms`; stage 3 measured
`0.124928/0.124928/0.125952/0.129024 ms` (5.74% geometric-mean improvement).
The 8,000-iteration confirmation remained lower at
`0.124928/0.124928/0.125952/0.130048 ms`, with exact output and maximum
absolute error `9.54e-05`. The specialization is restricted to this M16
long-K dispatch. Artifacts are `v20_m16_down_stage3_{control,verify8000}.json`.

The fourteenth accepted v20 progression extends the three-K16 stage to the
M8 fixed-N full-Q `(K,N)=(5120,12288)` wide launcher. The matched stage-2
medians were `0.089088/0.092160/0.090112/0.092160 ms`; stage 3 measured
`0.086016/0.090112/0.088064/0.089088 ms` (2.90% geometric-mean improvement).
The 8,000-iteration confirmation reproduced all four medians exactly, with
maximum absolute error `3.24e-05`. Both the pair-wrapped and generic
transition-width branches use the stage-3 template; other M8 shapes retain
stage 2. Artifacts are `v20_m8_fullq_stage3_{control,verify8000}.json`.

The fifteenth accepted v20 progression applies the three-K16 stage to the
M8 fixed-N MLP-gate/up `(K,N)=(5120,17408)` wide launcher. Stage 2's matched
medians were `0.116736/0.120832/0.119808/0.121856 ms`; stage 3 measured
`0.113664/0.115712/0.115712/0.117760 ms` (3.4% geometric-mean improvement).
The 8,000-iteration confirmation remained faster at
`0.113664/0.115712/0.116736/0.118784 ms`, with exact output and maximum
absolute error `4.20e-05`. Both gate/up transition branches use stage 3;
other M8 routes retain stage 2. Artifacts are
`v20_m8_gate_stage3_{control,verify8000}.json`.

The sixteenth accepted v20 progression applies the three-K16 stage to the
M8 fixed-N attention-out `(K,N)=(6144,5120)` wide launcher. Stage 2's matched
medians were `0.052224/0.053248/0.053248/0.052224 ms`; stage 3 measured
`0.052224/0.052224/0.051200/0.051200 ms` (about 1.9% geometric-mean
improvement). The 8,000-iteration confirmation reproduced the same values,
with exact output and maximum absolute error `2.67e-05`. Both transition
branches use stage 3; other M8 routes retain stage 2. Artifacts are
`v20_m8_attention_stage3_{control,verify8000}.json`.

The seventeenth accepted v20 progression applies the three-K16 stage to the
M8 fixed-N linear-Z `(K,N)=(5120,6144)` wide launcher. Stage 2's matched
medians were `0.052224/0.052224/0.052224/0.053248 ms`; stage 3 measured
`0.050176/0.051200/0.051200/0.052224 ms` (about 2.5% geometric-mean
improvement). The 8,000-iteration confirmation reproduced all four values,
with exact output and maximum absolute error `1.72e-05`. This stage depth is
restricted to the fixed-N wide branch. Artifact pair:
`v20_m8_linearz_stage3_{control,verify8000}.json`.

The eighteenth accepted v20 progression applies the three-K16 stage to the
M8 fixed-N linear-QKV `(K,N)=(5120,10240)` wide launcher. Stage 2's matched
medians were `0.077824/0.078848/0.078848/0.080896 ms`; stage 3 measured
`0.074752/0.075776/0.076800/0.077824 ms` (about 3.6% geometric-mean
improvement). The 8,000-iteration confirmation reproduced the same rate-wise
ordering and exact output (maximum absolute error `2.86e-05`). This stage
depth is restricted to the fixed-N wide QKV branch. Artifacts are
`v20_m8_qkv_stage3_{control,verify8000}.json`.

The nineteenth accepted v20 progression applies the three-K16 stage to the
M8 fixed-N long-K MLP-down `(K,N)=(17408,5120)` wide launcher. Stage 2's
matched medians were `0.118784/0.120832/0.121856/0.122880 ms`; stage 3
measured `0.114688/0.116736/0.117760/0.119808 ms` (about 3.2%
geometric-mean improvement). The 8,000-iteration confirmation reproduced
the same values with exact output and maximum absolute error `7.25e-05`.
The existing W3.5-only CA8 bank-ID copy remains enabled inside the stage-3
branch; other M8 routes retain stage 2. Artifacts are
`v20_m8_down_stage3_{control,verify8000}.json`.

The twentieth accepted v20 progression retunes the newly stage-3 M16 full-Q
launcher from split 10 to split 9. The split-9 candidate is faster at every
rate in both 4,000- and 8,000-iteration runs; the 8k medians are
`0.091136/0.089088/0.090112/0.091136 ms` for W2/W2.5/W3/W3.5 (about 2.8%
lower than the stage-3 split-10 geomean), with exact output (maximum absolute
error `4.58e-05`). The Python dispatch and direct-plan test now pin split 9;
other M16 routes are unchanged. Artifacts are
`v20_m16_fullq_stage3_split9_{candidate,verify8000}.json`.

The twenty-first accepted v20 progression fixes the non-wide stage-stride used
by the stage-3 WMMA path. The load address now uses the selected
`StageKTiles` stride instead of the two-tile default, making fixed-N M16
linear-QKV stage 3 exact. With split 10, the 8,000-iteration medians are
`0.086016/0.086016/0.086016/0.087040 ms` for W2/W2.5/W3/W3.5, about 5.9%
lower geomean latency than fetched main; maximum absolute error is
`4.20e-05`. The existing M16 split-10 dispatch is now backed by the
three-K16 launcher. Artifact: `v20_m16_qkv_stage3_stridefix_split10_verify8000.json`.

The twenty-second accepted v20 progression uses the same corrected stride for
fixed-N M16 linear-Z. Split 10 is exact and measures
`0.058368/0.058368/0.059392/0.059392 ms` at 8,000 iterations (about 5.4%
lower geomean latency than fetched main; maximum absolute error
`3.62e-05`). The linear-Z dispatch now uses stage 3 with split 10. Artifact:
`v20_m16_linearz_stage3_stridefix_split10_verify8000.json`.

The corrected-stride full-KV probes were not accepted. M16 full-KV stage 3 at
split 32 was neutral versus its two-stage control, while M8 full-KV stage 3 at
split 48 won only W2 and regressed W3/W3.5 in the 8,000-iteration confirmation;
both routes remain on stage 2. Neighboring M16 long-K down splits 20 and 28
were also slower than the retained split 24. Diagnostics are
`v20_m16_fullkv_stage3_stridefix_split32_probe.json`,
`v20_m8_fullkv_stage3_stridefix_split48_verify8000.json`,
`v20_m16_down_stage3_split{20,28}_screen.json`, and the existing split-24
verification artifact.

The twenty-third accepted v20 progression enables the existing CA8 bank-ID
copy for only the M16 long-K W3.5 specialization. At split 24, the 8,000-
iteration median is `0.129024 ms` (exact maximum absolute error
`8.39e-05`), one event tick below the stage-3 two-4-byte-copy control at
`0.130048 ms`; W2/W2.5/W3 retain the prior copy path. Artifact:
`v20_m16_down_stage3_ca8_w35_split24_verify8000.json`.

The twenty-fourth accepted v20 progression adds the already validated
pair-wrap predicate to fixed-N M16 linear-QKV stage 3. Split 10 remains the
plan: W2 is unchanged, while the 8,000-iteration W2.5/W3/W3.5 medians improve
to `0.082944/0.082944/0.084992 ms`; the full four-rate geomean is about 2.4%
lower than the prior stage-3 plan (about 8.1% lower than fetched main for this
route), with maximum absolute error `4.20e-05`. Artifact:
`v20_m16_qkv_stage3_pairwrap_stridefix_split10_verify8000.json`.

The twenty-fifth accepted v20 progression adds the pair-wrap predicate to
fixed-N M16 linear-Z stage 3. Split 10 remains the plan: W2 is unchanged and
W2.5/W3/W3.5 measure `0.056320/0.056320/0.057344 ms` at 8,000 iterations,
about 3.1% lower than the prior stage-3 plan (about 8.2% lower than fetched
main for this route), with maximum absolute error `3.62e-05`. Artifact:
`v20_m16_linearz_stage3_pairwrap_stridefix_split10_verify8000.json`.

The twenty-sixth accepted v20 progression hoists the bank-selector masks for
all fixed-N M16 linear-QKV rates (the earlier stage-3 pair-wrap path hoisted
W3 only). Split 10 remains exact; 8,000-iteration medians are
`0.082944/0.082944/0.082944/0.083968 ms`, improving W2 and W3.5 and tying the
other two rates versus the prior QKV plan. Maximum absolute error is
`4.20e-05`. Artifact:
`v20_m16_qkv_stage3_pairwrap_hoistall_split10_verify8000.json`.

The twenty-seventh accepted v20 progression hoists bank-selector masks for all
fixed-N M16 linear-Z stage-3 rates. Split 10 is exact and measures
`0.056320/0.056320/0.056320/0.056320 ms` at 8,000 iterations, improving every
rate versus the prior pair-wrap-only plan; maximum absolute error is
`3.62e-05`. Artifact:
`v20_m16_linearz_stage3_pairwrap_hoistall_split10_verify8000.json`.

An all-rate bank-mask-hoist probe for the M16 gate/up stage-3 launcher was
rejected. The 4,000-iteration screen regressed every rate versus the retained
selective-hoist plan (for example W2 moved from `0.119808` to `0.123904 ms`),
so the source was restored without a follow-up 8k run. Diagnostic:
`v20_m16_gate_stage3_hoistall_split10_probe.json`.

The analogous all-rate bank-mask-hoist probe for M16 attention-out was also
rejected. It stayed exact but regressed all four rates in the 4,000-iteration
screen (for example W2 moved to `0.055296 ms` from the retained
`0.054272 ms`), so the original selective policy remains. Diagnostic:
`v20_m16_attention_stage3_hoistall_split12_probe.json`.

A fixed-N M16 full-KV pair-wrap probe was rejected. Although the stage-2
kernel remained exact, the 8,000-iteration split-32 confirmation regressed
W2.5/W3 to `0.034816 ms` and did not beat the matched control consistently;
the source was restored to the existing non-pair-wrapped plan. Diagnostic:
`v20_m16_fullkv_pairwrap_split32_verify8000.json`.

An earlier M8 full-KV small-N stage-3 probe was invalid because the non-wide
WMMA load still used the two-tile shared-memory stride; it consequently failed
random-bank correctness on W2 (`max_abs=37.29`) before timing. After the
stride fix, the route became exact but the 8k result remained mixed as noted
above, so dispatch stays on the two-K16 implementation.

The earlier fixed-N M16 linear-QKV stage-3 probe had the same stride bug and
failed random-bank correctness on W2 (`max_abs=47.61`) before timing. It is
superseded by the corrected-stride result above; fixed-N stage 3 is enabled
only for the explicitly validated M16 QKV and linear-Z routes.

An attempted `.cg` cache-policy variant of the 8-byte transaction was rejected
at compile time: sm_80 `cp.async.cg` accepts only a 16-byte copy in this
toolchain (`ptxas: unexpected value '8'`). The validated `.ca` transaction is
unchanged.

A scalar M1 attention W3.5 probe that replaced its per-byte bank-ID staging
with a 16-byte async copy regressed from the approximately `0.040960 ms`
baseline to `0.045056 ms`; the source was restored. Diagnostic:
`v20_m1_attention_bankca16_candidate.json`.

A shape-specific static `reduce_split_kernel<5>` dispatch for M16 full-Q was
also rejected. Its all-rate 4,000-iteration medians were identical to the
existing runtime-loop reducer (`0.095232/0.100352/0.099328/0.102400 ms`), so
the extra specialization was removed; diagnostic:
`v20_m16_fullq_static_reduce5_candidate.json`.

A warp-parallel five-way reducer (`reduce_split_warp_kernel<5,8>`) for the
same M16 full-Q shape was likewise neutral across W2–W3.5, reproducing
`0.095232/0.100352/0.099328/0.102400 ms`; it was reverted. Diagnostic:
`v20_m16_fullq_warp_reduce5_candidate.json`.

A follow-up W3/W3.5 pair-wrap probe on M16 linear-QKV `(K,N)=(5120,10240)`
was rejected. The exact candidate measured `0.092160/0.093184 ms` for W3/W3.5
in the 4,000-iteration screen, versus fetched-main medians near
`0.091136/0.092160 ms`; both rates regressed by about one event tick. The
dispatch was restored; diagnostic artifact:
`v20_m16_qkv_w35_pairwrap_candidate.json`.

The analogous W3.5-only probe on M16 linear-Z `(K,N)=(5120,6144)` was also
omitted: its exact candidate was `0.062464 ms`, tied to the fetched-main
median at 4,000 iterations. The source was restored; see
`v20_m16_linearz_w35_pairwrap_candidate.json`.

Two scalar M1 W3.5 pair-wrap probes were rejected as well. Full-Q tied at
`0.068608 ms`, and MLP-gate/up regressed from the fetched-main
`0.090112 ms` median to `0.092160 ms`; both retained the original wrap path.
Diagnostics are `v20_m1_fullq_w35_pairwrap_candidate.json` and
`v20_m1_gate_w35_pairwrap_candidate.json`.

Extending pair-wrap to the four-stage M1 long-K MLP-down W3.5 path tied its
control at `0.095232 ms` in the 4,000-iteration screen. It was reverted and
left out of dispatch (`v20_m1_down_w35_pairwrap_candidate.json`).

Adding the W2 power-of-two wrap mask to the widened M16 attention-out and
MLP-gate/up paths was rejected. Attention-out tied its fetched-main median at
`0.060416 ms`, while gate/up regressed from `0.134144` to `0.135168 ms` in the
4,000-iteration screen. The existing generic W2 wrap remains; diagnostic:
`v20_m16_w2_pow2_attention_gate_candidate.json`.

Subsequent probes were rejected and left out of dispatch. The M8 full-KV
N128 layout added one event tick at W2/W2.5/W3 and tied W3.5; M8 long-K
split 36 and split 48 were slower than the retained split 40; M8 gate/up
split 12 and linear-QKV split 12 regressed every rate. The M16 linear-Z
N128 layout tied W2 but regressed the other three rates at split 16 and again
at split 12. Diagnostics are
`v19_m8_fullkv_wide_n128_candidate_deep.json`,
`v19_m8_down_wide_n128_split{36,48}_deep.json`,
`v19_m8_gate_wide_n128_split12_deep.json`,
`v19_m8_qkv_wide_n128_split12_deep.json`,
`v19_m16_linearz_wide_n128_candidate_deep.json`, and
`v19_m16_linearz_wide_n128_split12_deep.json`.

The initial broad M1 reducer experiment was narrowed before acceptance. It
improved attention-out and linear-Z, was neutral on long-K MLP-down, and
regressed MLP-gate/up by about 0.9%; full-Q and linear-QKV were effectively
neutral. The retained dispatch therefore covers only the two clean winning
shapes. The broad diagnostic is
`artifacts/a100_p32_window/v19_m1_warpreduce4_candidate.json`.

Extending the four-output layout to M2 attention-out, linear-Z, and MLP-down
was rejected: attention-out had one win, two ties, and one loss, while the
other shapes were neutral-to-slower. An eight-output M16 full-KV layout was
also narrowed out despite a 1.62% geomean gain because W3.5 lost one event
tick; the retained 16-output layout wins every rate. Diagnostics are
`v19_m2_selective_warpreduce_screen.json`,
`v19_m16_fullkv_warpreduce8_fixed_candidate.json`, and
`v19_m16_fullkv_warpreduce16_candidate.json` under
`artifacts/a100_p32_window/`.

M2 MLP-down remained exactly median-neutral under the later eight-output
layout, so it was omitted from the retained dispatch to avoid changing
summation order without a performance benefit.

Two wider reducer expansions were rejected. The M4 16-output layout was
neutral-to-slower on attention-out and linear-Z. Extending M2's retained
eight-output layout to full-Q, linear-QKV, and MLP-gate/up produced several
one-tick regressions and no wins. Their source changes were restored;
diagnostics are `v19_m4_warpreduce16_candidate.json` and
`v19_m2_warpreduce8_wide_screen.json`.

Three additional reducer/epilogue variants were rejected after the fourth
progression. Reducing warp-reducer launches from 256 to 128 threads regressed
M16 full-KV at every rate and did not provide a clean M1/M2 gain. Loading the
four M16 full-KV bank IDs with one `cp.async.ca` transaction regressed three
rates. An intermediate eight-output M1 reducer was superseded by the retained
16-output layout, which produced the larger clean gain. Diagnostics are
`v19_warpreduce128_screen.json`,
`v19_m16_fullkv_bankid_cpasync4_candidate.json`, and
`v19_m1_warpreduce8_candidate.json` under `artifacts/a100_p32_window/`.

Broadening the 16-output M1 reducer to linear-QKV and MLP-gate/up was
rejected: linear-QKV was median-neutral and MLP-gate W2.5 lost one event
tick. Widening the accepted M2 projection reducer from eight to 16 outputs
was also rejected because both attention-out W2 and linear-Z W2 regressed.
Diagnostics are `v19_m1_warpreduce16_wide_candidate.json` and
`v19_m2_warpreduce16_candidate.json` under `artifacts/a100_p32_window/`.

The later reduction sweep rejected eight- and 16-output layouts for
M1/M2/M4 full-KV, M4/M8/M16 MLP-down, and M8/M16 attention-out/linear-Z;
each was neutral or had at least one rate regression. Vectorizing M2/M4
scalar full-KV output stores likewise regressed the matched control. The
diagnostics use the `v19_m124_fullkv_`, `v19_m4_mlpdown_`,
`v19_m8_mlpdown_`, `v19_m16_mlpdown_`, `v19_m8_projection_`,
`v19_m16_projection_`, and `v19_m24_fullkv_float2_` prefixes.

A broad direct-plan table for every M16 Qwen shape was narrowed to the
retained full-KV entry. Full-Q W2.5/W3.5, attention-out W3.5, and linear-Z
W2 lost one event tick in the matched pair, while most other cases tied.
Those shapes continue to use the default in-memory autotuner.

Direct measured dispatch was also rejected for M2 attention-out/linear-Z:
the attention cases tied and linear-Z W2 lost one event tick. The M1
linear-Z half of the subsequent direct-dispatch screen was narrowed out for
the same reason, while the retained M1 attention-out half had one clean win.
The M2 diagnostic is `v19_m2_projection_fastplans_candidate.json`.

The M4 attention-out half of the direct-plan screen was neutral and was
omitted. Its W2.5/W3.5 autotune result also varied between split 24 and 48
without changing median latency, so retaining autotuning avoids hard-coding
an equivalent summation order.

Later host experiments were rejected. Direct dispatch was neutral for M8
attention-out/linear-Z, M2 and M4 full-Q/linear-QKV, and M2 MLP-down; M1
MLP-down regressed W3.5. Packing the complete autotune identity into one
large Python integer made cache hits 3-6 microseconds slower than the tuple
key, so the cache retains its device-index-first tuple. Computing the bank
mask on the host and passing it into CUDA regressed almost every tested
M1-M8 attention rate by one event tick. Diagnostics use the
`v19_*_fastplan`, `v19_*_fastplans`, `v19_m2_projection_packed_int_cache_`,
and `v19_host_altmask_` prefixes.

Three CUDA cache/specialization experiments were also rejected. A
`__grid_constant__` bank-ID parameter was mixed and regressed M1; compiling
bank-alt 3 as a true template constant regressed the tested M8 N5120 routes.
Adding Marlin's 128-byte L2 prefetch hint to every streaming trellis
`cp.async.cg` load was neutral on M8/M16 but regressed three M1 rates, so it
was not retained globally. Diagnostics are
`v19_gridconstant_bankid_candidate.json`,
`v19_m8_n5120_static_bank3_candidate.json`, and
`v19_trellis_l2_128b_attention_candidate.json`.

The apparent M4 W3 full-KV 16-output reduction win did not reproduce at
4000 iterations: it measured `0.033792 ms`, slower than the four-output
control. The retry is `v19_m4_w3_fullkv_warpreduce16_final.json`, and the
source change was restored.

Four later plan/cache probes were also rejected. Pinning M8 W2.5
MLP-gate/up to split 8 was neutral. Explicit M2 linear-Z plans regressed the
otherwise correct-looking screen, and an explicit split-16 M16 full-Q plan
was exactly neutral. A selective Marlin-style `L2::128B` trellis prefetch on
M4 W3.5 attention-out/MLP-down also tied its matched control at both medians.
Diagnostics are `v19_m8_w25_gate_fixed8.json`,
`v19_m2_linearz_fastplans_final.json`,
`v19_m16_fullq_fastplan16_candidate.json`, and
`v19_m4_w35_l2_128b_{candidate,control}.json` under
`artifacts/a100_p32_window/`.

The packed M16 follow-up initially included full-Q. It improved W2 one event
tick, regressed W3 one tick, and tied W2.5/W3.5, so full-Q was restored to
the existing decoder. Only the all-rate-positive MLP-down specialization is
retained; the combined diagnostic is
`v19_m16_packed_fullq_mlpdown_candidate.json`.

Applying the direct pair-wrap predicate to every specialization was also
rejected. The matched attention screen exposed losses on M2 W3/W3.5, M4
W2.5, and M16 W2.5-W3.5, while a broader M1 screen was mixed outside
attention-out. The initial template flag was therefore narrowed to the
eight deeply confirmed attention cases before the later case-by-case M1
extension described above. The broad diagnostics are
`v19_wrap_pair_predicate_attention_{control,candidate}.json` and
`v19_selective_wrap_predicate_m1_candidate.json`.

The later all-shape M16 pair-wrap screen was rejected completely. Most
non-KV shapes lost one event tick, full-KV W2 and W3.5 regressed, and there
was no clean winning subset. The restored matched diagnostics are
`v19_wrap_m16_all_{control,candidate}.json`.

The M16 bank-mask fast path was briefly broadened from widened kernels to the
fixed-N linear-QKV (`N=10240`) and linear-Z (`N=6144`) routes. Both candidates
were numerically exact, but the 8,000-iteration matched run was effectively
tick-neutral: linear-QKV measured `0.092231 ms` versus `0.092160 ms`, and
linear-Z `0.062408 ms` versus `0.062464 ms`. A 4,000-iteration screen also
showed the same instability and the N=1024 full-KV route regressed by about
9%. The source was restored to the widened-M16-only predicate; diagnostics are
`v20_bank_alt3_fast_m16_fixed_large_verify8000.json`,
`v20_bank_alt3_fast_m16_fixed_large_control8000.json`,
`v20_bank_alt3_fast_m16_narrow_other_candidate.json`, and
`v20_bank_alt3_fast_m16_narrow_other_control.json`.

A packed two-state decode/selector-hoist probe was also rejected on widened
M16 full-Q. Enabling `HoistBankMasks` for W3/W3.5 (W2 already uses it) stayed
exact but lost one event tick at W2.5 and tied the other rates: the candidate
medians were `0.096256/0.097280/0.097280/0.099328 ms` versus control
`0.096256/0.096256/0.097280/0.099328 ms`. The mixed hoist policy was restored;
diagnostics are `v20_m16_fullq_hoist_{candidate,control}.json`.

A 32-output-per-warp split reducer was rejected for M16 full-KV split 32. The
one-lane-per-output geometry is exact, but its 4,000-iteration medians were
`0.035840/0.034816/0.034816/0.035840 ms` versus the retained 16-output
reducer's `0.032768/0.033792/0.034816/0.034816 ms`; it loses W2, W2.5, and
W3.5. The 16-output reducer remains dispatched. Diagnostics are
`v20_m16_fullkv_reduce32_{candidate,control}.json`.

## v21 after merged PR #99

The next optimization cycle started from freshly fetched `origin/main` at
`631411ee3b07ed14c29fb21c0e959b0d811eb5cb` (the merge of PR #99). The fresh
140-row Ampere baseline is `artifacts/a100_p32_window/v21_main_baseline_300.json`;
its median-latency geometric mean is `0.0655313 ms` across M=1,2,4,8,16 and
W2-W3.5. The full candidate screen is
`artifacts/a100_p32_window/v21_candidate_full_300.json`.
The current-tip refresh is `artifacts/a100_p32_window/v21_current_full_300.json`;
against the fresh baseline its median-latency geometric mean is `0.9954x`
(effectively unchanged at the 1.024-us CUDA-event tick), so no additional
10% progression is claimed from this cycle.

The first accepted v21 progression makes the split reducer compile-time for
the two direct wide-Q waves that previously fell through the runtime loop:
split 9 (M16 full-Q) and split 14 (M8 full-Q). The matched 3,000-iteration
medians were exact and improved M8 W2/W2.5/W3 from
`0.086016/0.090112/0.088064` to
`0.084992/0.089088/0.087040 ms`; M16 W3/W3.5 improved from
`0.090112/0.091136` to `0.089088/0.090112 ms`. Diagnostics are
`v21_reducer_9_14_3000.json` and `v21_reducer_control_3000.json`.

The same progression adds a three-stage scalar pipeline only for M1,
K5120/N12288. Its matched 3,000-iteration medians were exact and improved
W2/W2.5/W3 from `0.061440/0.068608/0.067584` to
`0.060416/0.067584/0.066560 ms`, with W3.5 tied. Diagnostics are
`v21_m1_fullq_stage3_3000.json` and `v21_m1_fullq_stage2_control_3000.json`.

The following probes were rejected and reverted. M16 full-Q stage 2 and
stage 4 regressed the three-stage control; a 64-thread Marlin-style CTA was
50--65% slower and initially exposed an 8-byte bank-ID alignment hazard; a
four-output vector reducer tied the scalar reducer; and a global
`-maxrregcount=64` cap was mixed/one-tick-only. Broadening the M1 three-stage
pipeline to N10240/N17408 regressed MLP-gate W3.5 to `0.092160 ms` (versus
`0.090112 ms`). Diagnostics are `v21_m16_fullq_stage2_1000.json`,
`v21_m16_fullq_cta64_safe_1000.json`, `v21_m16_fullq_vec4_1000.json`,
`v21_m16_fullq_rreg64_{1000,3000}.json`, and
`v21_m1_wide_stage3_1000.json`.

Additional follow-up probes were rejected. Explicit M16 full-Q split 8 was
about 9--13% slower than split 9; three N16 tiles per warp was about 50%
slower; explicit PTX for the PGC16 multiply/bit extract regressed W3.5; and
both packed and four-register lane-plan hoists tied or regressed after
register/instruction trade-offs. A global 80-register cap was mixed across
M1 MLP shapes. Diagnostics are `v21_m16_fullq_split8_1000.json`,
`v21_m16_fullq_triplewide_1000.json`, `v21_m16_fullq_pgc_ptx_1000.json`,
`v21_m16_fullq_laneplan_1000.json`,
`v21_m16_fullq_laneplan_packed_1000.json`,
`v21_m16_fullq_wordplan_1000.json`, and
`v21_m1_rreg80_{1000,3000}.json`.

The next probes were also rejected. A correctly scoped scalar bank-mask hoist
for M16 full-Q W3.5 increased the matched 5,000-iteration median from
`0.090112` to `0.091136 ms`; staging the 512-byte level table in shared
memory tied W3.5 at `0.090112 ms` and showed no consistent W2-W3 benefit; and
a fused adjacent-pair state extractor increased W3.5 to `0.091136 ms`.
Enabling the in-process autotuner that is on by default for the three
full-Q hard-coded routes selected noisier waves (including M16 split 10
instead of the retained split 9) without improving the matched screen. A
three-stage scalar M4 W3.5 probe regressed from `0.089088` to `0.092160 ms`.
Replacing the constant-cache `__ldg` level loads with ordinary global loads
also regressed M16/W3.5 from `0.090112` to `0.092160 ms`. All were reverted.
Diagnostics are
`v21_m16_fullq_hoistw35_{control,candidate}_5000.json`,
`v21_m16_fullq_litew35_candidate_5000.json`,
`v21_m16_fullq_sharedlevels_{w35_candidate_5000,w234_candidate_3000}.json`,
`v21_m16_fullq_quad64_w35_candidate_5000.json`,
`v21_fullq_autotune_1000.json`,
`v21_m4_fullq_stage{2_control,3_w35}_5000.json`, and
`v21_m16_fullq_globallevels_w35_candidate_5000.json`.

A warp-level M1 full-Q split reducer was also rejected: with split 40 its
matched 5,000-iteration medians were identical to the existing static
reducer at all four rates. Diagnostics are
`v21_m1_fullq_warpreduce_candidate_5000.json` and
`v21_m1_fullq_staticreduce_control_5000.json`.

Skipping the final WMMA handoff barrier (safe-looking because no later stage
overwrites the buffer) was also rejected: M16/W3.5 moved from the matched
`0.090112` to `0.091136 ms`. The barrier remains unconditional for the
independent-thread-scheduling handoff contract. Diagnostic:
`v21_m16_fullq_finalbarrier_candidate_5000.json`.

Hoisting lane-invariant pair and shuffle coordinates out of the K-stage loop
was neutral in the matched M16/W3.5 run (`0.090112 ms` control and
candidate), so it was also reverted. Diagnostic:
`v21_m16_fullq_lanehoist_w35_candidate_5000.json`.

The Hopper-inspired byte-permute form of the PGC16 pre-mix was bit-identical
but neutral on Ampere: M16/W3.5 measured `0.090112 ms`, the same as control,
so the compiler's original shift/LOP3 form remains. Diagnostic:
`v21_m16_fullq_pgc_byteperm_w35_candidate_5000.json`.

## v22 after the next origin refresh

The next cycle fetched `origin/main` again; it remains
`631411ee3b07ed14c29fb21c0e959b0d811eb5cb`. The existing Ampere branch is
still the open WIP PR #100, so this cycle does not claim a merged upstream
baseline change.

The M16 long-K MLP-down split sweep confirmed the retained split 24 policy:
split 16 measured `0.141312/0.140288/0.142336/0.145408 ms`, split 24 measured
`0.124928/0.124928/0.125952/0.129024 ms`, and split 32 measured
`0.128000/0.128000/0.129024/0.131072 ms` for W2/W2.5/W3/W3.5. The split-24
control remains the fastest; diagnostics are
`v22_m16_mlpdown_split{16,24,32}_2000.json`.

The following candidates were exact but did not improve a matched high-
iteration control and were reverted: a native `__umul24` PGC16 multiply,
static bank-alt-3 specialization, scalar M1 compile-time-K, a warp-parallel
M16 MLP-down reducer, Hopper's `mad.lo.u16` PGC form, cache-at-all-levels
activation copies, and a four-K16 M1 long-K static-N dispatch. Representative
diagnostics are `v22_m16_fullq_umul24_candidate_{5000,w234_5000}.json`,
`v22_m16_fullq_staticalt3_candidate_3000.json`,
`v22_m1_fullq_statick_candidate_5000.json`,
`v22_m16_mlpdown_warpreduce16_candidate_5000.json`,
`v22_m16_fullq_pgc16u16_candidate_5000.json`,
`v22_m16_fullq_input_ca16_candidate_5000.json`, and
`v22_m1_mlpdown_staticn_stage4_candidate_5000.json`.

The current full 140-row refresh remains the measured reference:
`v21_current_full_300.json` has a `0.9954x` median-latency geometric mean
versus `v21_main_baseline_300.json`, so no additional 10% Ampere progression
is claimed from these probes.

## v23 after the September origin refresh

`git fetch origin main` confirms that `origin/main` is still
`631411ee3b07ed14c29fb21c0e959b0d811eb5cb`; PR #100 remains the open WIP
branch, so this cycle continues to use that exact fetched tip as its control.

The accepted progression specializes the M1 long-K MLP-down shape
`(K,N)=(17408,5120)` to the existing three-K16 scalar software stage. The
matched 8,000-iteration CUDA-event runs are exact and measure
`0.082944/0.089088/0.090112/0.092160 ms` for W2/W2.5/W3/W3.5, versus the
two-stage control's `0.082944/0.093184/0.093184/0.095232 ms`. This is a
`0.088508/0.091004 = 1.0282x` (2.82%) geomean reduction for the affected
four-rate shape. The candidate and control are
`artifacts/a100_p32_window/v23_m1_mlpdown_stage3_verify8000.json` and
`artifacts/a100_p32_window/v23_m1_mlpdown_stage2_control8000.json`.
The full formal suite remains the gate; `tests/test_qvq_p32_ampere.py` passes
56/56 cases on the A100.

The following v23 probes were rejected and restored: an eight-block
`__launch_bounds__` register/occupancy cap (M16 full-Q W3.5 rose to
`0.094208 ms`, about 4.5% slower), an eight-half-row activation shared-memory
skew (exact but M16 full-Q W3.5 rose to `0.093184 ms`), and warp or 8-lane
subgroup broadcasts of the shared bank selector (the full-warp version was
not exact because a scalar warp owns four different tiles; the corrected
subgroup version was exact but about 1.5% slower in the M1 full-Q probe).
The diagnostics are `v23_m16_fullq_minblocks8_candidate.json`,
`v23_m16_fullq_inputskew_candidate.json`,
`v23_m16_fullq_bankbroadcast_candidate.json`, and
`v23_m1_fullq_bankbroadcast_probe.json`. Nsight Compute confirms the retained
M16 path is L1/TEX and decode-load bound (95% L1 hit, ~4.3 useful bytes per
32-byte sector), so these probes do not justify claiming the requested 10%
overall gain.

## v24 follow-up from the merged tip

After PR #100 merged, `git fetch origin main` was repeated and the new branch
was based on `origin/main` at `202eb8d7dd9ccdb728051a7a41160b6cb4c21fc1`.
The staged M1 long-K dispatch was cherry-picked as commit `778cfea1` and
opened as WIP PR #101.  The formal Ampere suite passes 56/56 cases on the
A100 sm_80.

To revalidate against that exact fetched tip, matched 8,000-iteration CUDA
event runs were made for M1 MLP-down `(K,N)=(17408,5120)` at W2/W2.5/W3/W3.5.
The origin/main control Ampere medians were
`0.084992/0.094208/0.093184/0.095232 ms`; the three-K16 candidate medians
were `0.082944/0.089088/0.091136/0.092160 ms`.  The geometric mean is
`0.091812/0.088758 = 1.0344x` (3.44%) for this affected four-rate shape,
with exact output checks (`max_abs < 2.4e-5`).  Results are recorded in
`artifacts/a100_p32_window/v24_main_m1_mlpdown_8000.json` and
`artifacts/a100_p32_window/v24_candidate_m1_mlpdown_8000.json`.

This is a shape-local progression, not a claim of a 10% full-matrix gain;
the M16 decode-bound paths and the remaining M values still require a fresh
full-matrix sweep before the target can be assessed.

## v25 M1 long-K static split specialization

The M1 long-K autotuner selects split 128 on the A100.  A compile-time
`StaticSplitCount=128` specialization removes the per-CTA split-count
divisions while preserving the runtime fallback for every other split plan.
The change is exact under the formal suite (56/56 tests) and was repeated at
8,000 iterations: Ampere medians moved from
`0.082944/0.089088/0.091136/0.092160 ms` to
`0.081920/0.088064/0.090112/0.092160 ms` for W2/W2.5/W3/W3.5.  This is a
repeatable `1.0089x` (0.89%) improvement over the v24 staged candidate for
the affected shape.  The static-split diagnostics are
`artifacts/a100_p32_window/v24_candidate_splitstatic_m1_mlpdown_8000.json`
and `artifacts/a100_p32_window/v24_candidate_splitstatic_m1_mlpdown_repeat8000.json`.
The full-matrix 10% target remains open; no broad gain is claimed from this
shape-local specialization alone.

The analogous M2 long-K `StaticSplitCount=96` probe was rejected.  Its
8,000-iteration medians were `0.089088/0.098304/0.098304/0.102400 ms`, while
the dynamic-dispatch control was `0.089088/0.098304/0.099328/0.100352 ms`.
Autotune also changed the W3.5 plan from split 48 to 96 between runs, so the
probe provided no repeatable same-plan improvement and was reverted.  The
diagnostic is `artifacts/a100_p32_window/v25_candidate_m2_mlpdown_static96_8000.json`.

## v26 M1 full-Q static split specialization

The measured M1 full-Q route `(K,N)=(5120,12288)` uses the scalar triple-stage
kernel with split 40.  A guarded `StaticSplitCount=40` launch now removes the
split-count divisions for that plan while retaining the existing dynamic
triple-stage fallback for any other split count.  The formal Ampere suite is
exact (56/56 tests; worst observed `max_abs=1.34e-5`).  Two matched
8,000-iteration runs reproduced the same medians: dynamic
`0.060416/0.067584/0.067584/0.068608 ms` versus static
`0.060416/0.067584/0.066560/0.068608 ms` at W2/W2.5/W3/W3.5.  The resulting
four-rate geometric mean improves from `0.0659635` to `0.0657122 ms`
(`1.0038x`, 0.38%) for this shape.  Diagnostics are
`artifacts/a100_p32_window/v26_candidate_m1_fullq_dynamic_8000.json`,
`artifacts/a100_p32_window/v26_candidate_m1_fullq_static40_8000.json`, and
`artifacts/a100_p32_window/v26_candidate_m1_fullq_static40_repeat8000.json`.
This remains a shape-local progression; the full-matrix 10% target is still
open.

## v27 rejected M1 attention static split probe

The M1 attention-out route `(K,N)=(6144,5120)` uses split 48, so I screened a
matching compile-time split specialization.  It was reverted after the
8,000-iteration run showed a severe W3 regression: dynamic medians were
`0.040960/0.044032/0.043008/0.041984 ms`, while the static probe measured
`0.040960/0.040960/0.281600/0.040960 ms` at W2/W2.5/W3/W3.5.  Outputs stayed
exact (`max_abs <= 1.6e-5`), but the performance result is not acceptable.
The failed diagnostic is `artifacts/a100_p32_window/v27_candidate_m1_attention_static48_8000.json`.

## v28 M1 full-KV static split specialization

The M1 full-KV route `(K,N)=(5120,1024)` uses split 56.  Guarded
`StaticSplitCount=56` launches remove the split-count divisions in both the
scalar main kernel and its matching reducer, while leaving other plans on the
dynamic static-N launcher.  A matched 20,000-iteration control/candidate run
measured dynamic medians `0.033792/0.035840/0.035840/0.033792 ms` versus
static `0.032768/0.034816/0.034816/0.032768 ms` at W2/W2.5/W3/W3.5, a
repeatable `1.0303x` (3.03%) four-rate geometric-mean improvement.  Outputs
remained exact (`max_abs <= 1.1e-5`), and the formal suite passes 56/56.  The
diagnostics are `artifacts/a100_p32_window/v28_candidate_m1_fullkv_dynamic_20000.json`,
`artifacts/a100_p32_window/v28_candidate_m1_fullkv_static56_20000.json`, and
`artifacts/a100_p32_window/v28_candidate_m1_fullkv_static56_reducer_repeat20000.json`
(the shorter 8,000-iteration screens are also retained alongside them).
This is another shape-local gain; the full-matrix 10% target remains open.

## v29 neutral M1 linear-QKV static split probe

The M1 linear-QKV route `(K,N)=(5120,10240)` autotunes to split 40 on all
four rates.  A compile-time split-40 main-kernel specialization was screened,
but matched 8,000-iteration medians were unchanged at
`0.052224/0.057344/0.057344/0.058368 ms` for both dynamic and static launches.
With no measurable gain, the source probe was reverted.  The diagnostic is
`artifacts/a100_p32_window/v29_candidate_m1_linearqkv_static40_8000.json`;
the dynamic control is `artifacts/a100_p32_window/v29_candidate_m1_linearqkv_dynamic_8000.json`.

## v30 rejected M1 linear-Z static split probe

The M1 linear-Z route `(K,N)=(5120,6144)` also autotunes to split 40.  Its
compile-time split specialization regressed the 8,000-iteration medians from
dynamic `0.038912/0.039936/0.039936/0.040960 ms` to static
`0.044032/0.045056/0.044032/0.046080 ms` at W2/W2.5/W3/W3.5.  Exactness was
preserved (`max_abs <= 1.4e-5`), but the source probe was reverted.  The
diagnostics are `artifacts/a100_p32_window/v30_candidate_m1_linearz_dynamic_8000.json`
and `artifacts/a100_p32_window/v30_candidate_m1_linearz_static40_8000.json`.

## v31 M1 MLP gate/up static split specialization

The M1 MLP gate/up route `(K,N)=(5120,17408)` autotunes to split 40 on all
rates.  A guarded compile-time split-40 main-kernel launch is exact under the
formal suite (56/56) and reproduced over two 8,000-iteration runs.  Dynamic
medians were `0.079872/0.088064/0.088064/0.092160 ms`; static medians were
`0.079872/0.087040/0.088064/0.091136 ms` at W2/W2.5/W3/W3.5.  The four-rate
geometric mean improves `0.0869228→0.0864272 ms` (`1.0057x`, 0.57%), with
`max_abs <= 1.4e-5`.  Diagnostics are
`artifacts/a100_p32_window/v31_candidate_m1_mlpgate_dynamic_8000.json`,
`artifacts/a100_p32_window/v31_candidate_m1_mlpgate_static40_8000.json`, and
`artifacts/a100_p32_window/v31_candidate_m1_mlpgate_static40_repeat8000.json`.
This remains shape-local; the full-matrix 10% target is still open.

## v32 neutral M2 full-KV static split probe

The measured M2 full-KV route `(K,N)=(5120,1024)` dispatches split 64.  A
compile-time split-64 scalar main-kernel specialization was screened against
20,000-iteration runs, but the repeats moved individual rates in opposite
directions and did not establish a stable geometric-mean gain.  The source
probe was reverted.  Diagnostics are
`artifacts/a100_p32_window/v32_candidate_m2_fullkv_dynamic_20000.json`,
`artifacts/a100_p32_window/v32_candidate_m2_fullkv_static64_20000.json`, and
`artifacts/a100_p32_window/v32_candidate_m2_fullkv_static64_repeat20000.json`.

## v33 rejected M2 attention static split probe

The M2 attention-out route `(K,N)=(6144,5120)` uses split 48 for W2--W3
(W3.5 uses 64).  A split-48 scalar specialization was mixed: dynamic
8,000-iteration medians were `0.040960/0.044032/0.044032/0.044032 ms`, while
the probe measured `0.041984/0.044032/0.043008/0.045056 ms`.  Since W2 and
W3.5 regressed and there was no stable geometric-mean gain, the source probe
was reverted.  Outputs remained exact (`max_abs <= 2.1e-5`).  Diagnostics are
`artifacts/a100_p32_window/v33_candidate_m2_attention_dynamic_8000.json` and
`artifacts/a100_p32_window/v33_candidate_m2_attention_static48_8000.json`.

## v34 neutral M4 full-KV static split probe

The M4 full-KV scalar route uses split 64.  A compile-time split-64
specialization was screened with 20,000-iteration runs, but repeats swung
from apparent one-tick gains to neutral (`0.034816 ms` at every rate), so no
stable improvement was established at this short latency.  The source probe
was reverted; outputs remained exact.  Diagnostics are
`artifacts/a100_p32_window/v34_candidate_m4_fullkv_dynamic_20000.json`,
`artifacts/a100_p32_window/v34_candidate_m4_fullkv_static64_20000.json`, and
`artifacts/a100_p32_window/v34_candidate_m4_fullkv_static64_repeat20000.json`.

## v35 rejected M16 WMMA static split probe

I screened a compile-time split-count parameter on the M16 WMMA full-KV route
`(K,N)=(5120,1024)`, which dispatches split 32.  The probe regressed W2/W2.5
from dynamic `0.092160/0.034816 ms` to `0.111616/0.065536 ms` (W3/W3.5 were
also not improved), so the WMMA template change and dispatch branch were
removed.  Outputs stayed exact (`max_abs <= 1.6e-5`).  The failed diagnostic
is `artifacts/a100_p32_window/v35_candidate_m16_fullkv_static32_8000.json`.

## v38 M2 MLP gate/up static split specialization

The M2 MLP gate/up route `(K,N)=(5120,17408)` autotunes to split 40.  A
guarded compile-time split-40 scalar main-kernel launch is exact under the
formal suite (56/56) and reproduced across two 8,000-iteration runs.  The
dynamic geometric mean was `0.0958727 ms`; static runs measured
`0.0948473 ms` and `0.0950857 ms` (approximately 0.8--1.1% improvement),
with `max_abs <= 9.6e-6`.  Diagnostics are
`artifacts/a100_p32_window/v38_candidate_m2_mlpgate_dynamic_8000.json`,
`artifacts/a100_p32_window/v38_candidate_m2_mlpgate_static40_8000.json`, and
`artifacts/a100_p32_window/v38_candidate_m2_mlpgate_static40_repeat8000.json`.
This is shape-local progress; the full-matrix 10% target remains open.

## Reproduction

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-14ab23f1-a785-e9df-bbb5-215547154e3c
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=8 NINJAFLAGS=-j8 CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=2

python -m pytest -q tests/test_qvq_p32_ampere.py -s
python scripts/benchmark_qvq_p32_ampere.py --physical-gpu 0 --m-values 1 2 4 8 16 --warmup 10 --iterations 50
```
