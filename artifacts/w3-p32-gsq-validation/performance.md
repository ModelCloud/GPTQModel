# QVQ-GSQ H100 performance and training-budget audit

Profile target: real Llama 3.2 1B `model.layers.0.mlp.down_proj`, shape
`8192 x 2048`, W3 P32, 33 candidates, one Gumbel sample, H100 SM90.

## Performance corrections

- Score a P32 edge edit from only the decoder states it can change. At W3 one
  edge affects three of 128 states, or six of 256 scalar weights. The old path
  decoded and multiplied the complete tile for all four trial shifts.
- Evaluate the exact sparse Fisher delta
  `2<HEG,D> + tr(D^T H D G)` and materialize only the winning legal packed
  payload. Sixteen coordinate groups are screened per dispatch.
- Differentiate the exact full Fisher objective analytically. The new path uses
  two dense GEMMs for `H E G` and injects exact `dL/dp` through the softmax,
  instead of retaining/backpropagating through a large GEMM graph.
- Make hard-check cadence explicit and always check the final update. The
  generic default is every ten updates; the epoch-matched schedule checks once per
  64-update epoch. The baseline/coordinate hard guard remains authoritative.
- Consume H and G directly. Redundant Cholesky factors and factor-form teacher
  products are no longer constructed for the exact-Fisher path.
- Increase coordinate chunks from 1,024 to 8,192 tiles and sparse screen groups
  from four to sixteen. These defaults keep temporary storage bounded on H100
  while eliminating small dispatches.
- Form relaxed tiles and their probability gradients with batched GEMMs. This
  removes the broadcasted candidate/gradient products, including a roughly
  2.2 GiB temporary on the profiled projection.
- Keep finite-state and optimization telemetry on the GPU between hard
  checkpoints. Entropy and maximum-probability diagnostics are sampled at the
  first update and epoch boundaries, rather than forcing host synchronization
  every update.
- Run the soft relaxation in BF16 for the long schedule, matching the official
  GSQ logit precision. Candidate screening, coordinate initialization, hard
  Fisher evaluation, no-regression selection and exported weights remain FP32.

## Performance result

The earlier unprofiled 10-update workload completed in **0.5009 s**, down from
1.047 s. Batched GEMMs reduce that short microbenchmark to **0.4640 s**. The
gain there is only 1.08x because fixed metric, screening and coordinate setup
dominate a ten-update run.

On the then-assumed 640-update half-Llama-paper workload, the new implementation
completes in **2.2173 s**, down from 10.2725 s: a further **4.63x** speedup.
An intermediate FP32 run took 7.1264 s; BF16 soft relaxation plus one hard
check per 64-update epoch took 2.7574 s; device-resident telemetry reduced it
to the final 2.2173 s. The exact FP32-selected result is unchanged in every
case: 37 legal tiles change and normalized Fisher loss improves from
0.0244182274 to 0.0244180374.

The final 640-update Nsight trace records 2.2867 s, 39,548
`cudaLaunchKernel` calls and 107 `cudaStreamSynchronize` calls. Before moving
telemetry to the device, the equivalent trace took 2.9801 s and synchronized
3,290 times. Synchronization count therefore falls 30.7x. Final NVTX ranges
are 1.9821 s relaxation, 81.0 ms sparse decode/screen, 80.3 ms metric
construction and 66.9 ms coordinate initialization.

The earlier Nsight Compute/SASS result remains applicable to the dense FP32
objective GEMM: scalar `FFMA` plus staged `LDGSTS`, no `HMMA`/`HGMMA`. A custom
GEMM was not introduced because preserving the FP32 hard guard while exploiting
P32 locality and an analytic gradient delivered the requested gain.

## Sparse-relaxation speed and FP64 accuracy pass

The 33 legal P32 choices are not dense alternatives: at W3 every local path
choice differs from candidate zero at only six of a tile's 256 decoded values.
The candidate screen now carries those exact indices and deltas into relaxation.
Three SM90 Triton kernels fuse Gumbel softmax, sparse error construction, the
analytic sparse probability gradient and the Lion update. The full decoded bank
is retained for authoritative FP32 hard checkpoints and export.

On two consecutive complete 640-update fits in one process, the first projection
takes **1.3274 s** and the steady-state projection takes **0.6732 s**. The latter
is **3.29x faster** than the previous 2.2173 s implementation. The first fit
pays one-time CUDA library and Triton initialization; real model quantization
reuses the initialized process for 112 projections. A cold-process comparison
is 1.67x, so the 2x claim applies to steady-state GSQ projection throughput, not
startup latency. Both fits preserve the exact result: 37 changed tiles and FP32
Fisher loss 0.0244182274 to 0.0244180374.

The numerical gate follows `inference-ultra` origin/main commit `02665a1`: FP32
matmul precision is `highest`, TF32 is disabled, and 4,096 production-shape
values are compared independently with an FP64 mixture oracle. FP32 sparse
accumulation followed by one BF16 tensor-core cast is more accurate than the
old dense BF16 mixture:

- mean absolute error: 0.00129467 to **0.00112086** (13.4% lower);
- RMSE: 0.00204385 to **0.00170853** (16.4% lower);
- maximum absolute error: 0.0155722 to **0.00809199** (48.0% lower);
- normalized maximum error: 0.0053981 to **0.0028051**;
- head-to-head: sparse closer on 1,432 values, dense closer on 1,115, with
  1,549 ties.

No reduced-step shortcut or relaxed acceptance criterion is used. All 640
updates run, all ten historical checkpoints use the exact FP32 hard objective, and
the baseline/coordinate no-regression guard remains authoritative.

## Compact-metadata follow-up

Nsight's CUDA-graph node trace showed that the next non-GEMM bottlenecks were
streaming a mostly empty `[tile,256,3]` position map and reading int64 scalar
indices in the sparse Lion kernel. The retained implementation sorts the legal
P32 edits once, stores only unique edited positions, and packs both scalar
indices and choice IDs as uint8. Arithmetic is unchanged: position mixtures
still accumulate in FP32, the soft Fisher products remain BF16, and every hard
checkpoint and final acceptance remains FP32.

With the historical 640-update schedule and ten exact hard checkpoints,
steady state is now **0.58331 s**, versus 0.67319 s before this pass: a further
**1.154x** speedup. The complete gain from the corrected 10.2725 s baseline is
**17.61x**. The exact selected payload remains unchanged: 37 legal tiles and
loss 0.0244182274 to 0.0244180374.

A one-final-check timing run reaches **0.52140 s** (**1.291x** versus 0.67319 s),
but it is recorded only as a policy floor; that benchmark policy still
performs all ten hard checkpoints. The requested further 1.5x target would be
0.44879 s and was not reached without changing numerical policy.

Hopper FP8 metric products were explicitly rejected. They are fast enough to
cross the timing target, but on a real Llama q-projection and its captured YAQA
Fisher matrices, row-scaled FP8 produced 5.93% normalized metric-gradient RMSE
and cosine similarity 0.99824, versus 0.41% normalized RMSE for BF16. This pass
therefore does not trade GSQ trajectory accuracy for the headline speedup.

The refreshed FP64 mixture oracle for the compact kernel reports mean absolute
error 0.00113256, RMSE 0.00171733, maximum error 0.00822756 and normalized
maximum error 0.00285207. These remain better than the dense BF16 mixture on
all aggregate error measures. The focused suite passes 90/90 tests and the
leased-H100 lifecycle suite passes 21/21.

## Dataset and epoch audit

Correction after checking arXiv v2: the official dense-Llama schedule uses
4,096 packed 4,096-token training samples, batch size 64, 128 validation
samples and **20** block-wise epochs: 64 updates per epoch and **1,280
updates** total. Ten epochs is the Kimi schedule, not the Llama schedule.
Llama uses FineWeb-Edu in the official setup. The 640-update measurements below
remain valid performance measurements, but are half of the Llama paper budget.

The completed QVQ experiment used 10,178 YAQA/Fisher sequences and 3,961,260
valid tokens. This is more sequences but only 23.6% of the official packed-token
budget. Fisher capture made one exact aggregate pass; it did not feed minibatches
to GSQ. The old optimization requested 100 updates and early-stopped at 10, so
the statement “10 epochs” would have been incorrect.

Corrections:

- QVQ early stopping now defaults to disabled, allowing the temperature and
  kappa schedule to reach their endpoints.
- Diagnostics explicitly report `updates_are_epochs: false`, completed updates,
  hard-check frequency and the `exact_full_fisher` regime.
- `GSQConfig.for_qvq_paper_schedule()` now constructs the 1,280-update Llama
  reference schedule, and the QVQ validation CLI now defaults to 1,280 updates.

A baseline corrected 640-update run completed in 10.272 s. The previous pass
reduced that to 2.217 s, and the fused sparse path now takes 0.673 s after
one-time process initialization. Entropy reaches 0.0001578 and mean maximum
probability reaches 0.9999375, but the result remains exactly the same 37-tile
coordinate solution; the relaxation improves zero tiles. Therefore the
10-update truncation was a paper-parity bug, but it was **not** the cause of the
no-op relaxation. A subsequent 1,280-update run with the old initializer also
changed zero GSQ tiles. The remaining limitation is the frozen 33-path candidate
bank, non-paper initialization, and per-projection/full-Fisher adaptation, not
optimizer duration.

## Validation and artifacts

- The latest focused run passed 90 QVQ/GSQ unit tests and all 21 H100 lifecycle
  cases, including save/reload and forced non-baseline selections. The broader
  prior focused suite passed 162 CPU tests plus the same 21 GPU cases.
- Final 640-update benchmark:
  `/root/qvq-results/gsq-performance-20260916/640-bf16-bmm-h64-device-telemetry.json`
- Intermediate FP32 benchmark:
  `/root/qvq-results/gsq-performance-20260916/640-fp32-bmm-h64.json`
- Final Nsight report:
  `/root/qvq-results/gsq-performance-20260916/640-bf16-final-nsys.nsys-rep`
- Final Nsight database:
  `/root/qvq-results/gsq-performance-20260916/640-bf16-final-nsys.sqlite`
- Final repeated benchmark:
  `/root/qvq-results/gsq-performance-20260916/640-bf16-sparse-triton-fp32acc-repeat2.json`
- Compact-metadata repeated benchmark (ten checkpoints):
  `/root/qvq-results/gsq-performance-20260916/640-bf16-compact-packed-final-repeat3.json`
- One-final-check policy-floor benchmark:
  `/root/qvq-results/gsq-performance-20260916/640-bf16-compact-finalcheck-repeat3.json`
- Sampled FP64 oracle:
  `/root/qvq-results/gsq-performance-20260916/fp64-oracle.json`
- Fused sparse trace:
  `/root/qvq-results/gsq-performance-20260916/640-bf16-sparse-triton-fp32acc-nsys.nsys-rep`
- Fused sparse trace database:
  `/root/qvq-results/gsq-performance-20260916/640-bf16-sparse-triton-fp32acc-nsys.sqlite`
- CUDA-graph node trace used for the compact-metadata pass:
  `/root/qvq-results/gsq-performance-20260916/640-bf16-grouped-position-node.nsys-rep`
