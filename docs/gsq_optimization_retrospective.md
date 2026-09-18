# GSQ optimization retrospective: math, CUDA, and measured results

This document preserves the successful parts of the QVQ + YAQA + GSQ W3/P32
work on Llama 3.2 1B and NVIDIA H100. It is a retrospective and a reusable
optimization playbook, not a claim that GSQ is fully validated at paper scale.
GSQ remains opt-in; the strongest downstream run used only 0.0977% of the
paper's reconstruction-token budget and predates the deterministic-training
correction.

For the operating configuration and promotion rules, see [gsq.md](gsq.md).
For the algorithmic correction history, see the
[paper-alignment audit](experiments/gsq-qvq-paper-alignment.md). For the first
large staged-training speedups, see
[the H100 training report](qvq_gsq_training_speed_2026-09-16.md).

## Executive summary

The work produced three different kinds of positive result. They must not be
collapsed into one number:

1. **Correctness and model quality.** Paper-aligned staged reconstruction made
   GSQ move legal P32 states and produced positive held-out, fresh-reload, and
   downstream results in a bounded experiment. The guarded checkpoint improved
   GSM8K-Platinum from 537/1,209 to 540/1,209 and improved all five untouched
   final-logit endpoints.
2. **Exact projection optimization.** The aggregate Fisher implementation for
   a real `8192 x 2048` projection fell from 14.868 seconds to 1.047 seconds
   after the first correction, and the 640-update steady-state path eventually
   fell from 10.2725 seconds to 0.58331 seconds. That is a 17.61x cumulative
   speedup for that projection workload with the same exact selected payload.
3. **Exact staged-layer optimization.** A complete seven-projection layer fit
   fell from 80.053 seconds to 38.031 seconds, then 17.951 seconds, and finally
   8.372--8.448 seconds with the downstream Q/K guard. The last result is
   9.48x--9.56x faster than the original path and retains a 14.8214% held-out
   block improvement.

Later work attacked the much shorter canonical path. Its individual gains are
necessarily smaller because the large launch, materialization, and autograd
bottlenecks had already been removed. Those small wins are still useful as a
catalog of exact GPU optimization techniques, but their diminishing returns
are the reason optimization has moved back to QVQ and YAQA.

## The math that made GSQ useful

### Preserve the paper's optimization problem

The first QVQ adaptation optimized one aggregate per-projection Fisher loss
over a frozen 33-choice tile bank. A deterministic coordinate prepass had
already minimized the same pool, so the continuous relaxation added no value.
The corrected adaptation retained the deployable P32 representation while
restoring the important parts of paper GSQ:

- candidate zero is the source YAQA W3/P32 hard payload;
- soft logits use the shift-centred noisy Gaussian prior
  `0.01 * (Normal(0, 1) + 6 * centered(-shift^2 / 2))`;
- temperature anneals from 2.0 to 0.05 and the hardening multiplier from 100
  to 500;
- Lion jointly optimizes legal categorical assignments and the serialized
  output scale vector `SV`;
- the objective order is dedicated Q/K quadratic fitting, joint V/O attention
  reconstruction, then joint gate/up/down full-block reconstruction;
- preceding layers are replayed with their deployable quantized state, so the
  fitted layer sees the errors that will exist at inference time; and
- only hard, legal P32 states can be selected or serialized.

The data and schedule also matter. The dense-Llama target is 4,096 packed
4,096-token FineWeb-Edu rows, logical batch 64, 20 block epochs, and 2,000
dedicated Q/K updates. The block schedule is 1,280 optimizer updates, not “20
updates.” Training, hard selection, Q/K metric construction, final reporting,
and GSM8K-Platinum use disjoint data.

### Exploit the exact quadratic instead of differentiating a reconstruction

For error matrix `E`, input Fisher/Hessian factor `H`, and output factor `G`,
the Q/K objective is

```text
L(E) = tr(E^T H E G).
```

For a legal candidate delta `D`, the exact change is

```text
L(E + D) - L(E) = 2 <H E G, D> + tr(D^T H D G).
```

A W3 P32 edge edit changes only six of the 256 decoded scalars in a 16x16
tile. Computing the delta from those six values is mathematically identical to
decoding and multiplying the entire candidate tile, but removes almost all of
the work. This identity powered sparse candidate screening, compact mixtures,
and exact hard-oracle evaluation.

The same objective can be moved through the fixed orthogonal RHT and fixed
input/output scales. Transforming `H` and `G` once into P32 inner coordinates
lets every Q/K update operate on sparse inner-weight error. The optimizer no
longer reconstructs dense rotated weights or backpropagates through two
Hadamard transforms on every update. Once a hard payload is fixed, independent
output scales can be solved in closed form and compared against the original
and edited alternatives on the disjoint Q/K validation split.

For a relaxed error

```text
E(p) = E0 + sum_c p_c D_c,
```

the probability gradient follows directly from the quadratic:

```text
dL/dp_c = 2 <H E(p) G, D_c>.
```

Injecting this analytic gradient through the softmax removed the large generic
autograd graph while retaining the exact FP32 hard objective and selection
guard.

### Hard guards are part of the algorithm

Lower soft loss is not sufficient. The retained acceptance sequence is:

1. restore the best legal hard state on held-out stage data;
2. roll back a complete layer when held-out full-block reconstruction MSE
   regresses;
3. compare Q/K alternatives again after V/O and MLP fitting, because a locally
   better Fisher state can produce a worse BF16 block trajectory;
4. install the cumulative state in a copy-on-write checkpoint, reload through
   the native QVQ runtime, and reject global final-logit regressions; and
5. evaluate report-only and task data only after model selection is frozen.

This guard recovered a subtle case: the transformed Q/K fast path had a better
local Fisher objective but retained 14.8214% rather than 15.1786% held-out
block improvement. Replaying original, scale-only, and edited Q/K pairs after
downstream fitting chose the best available whole-block state without
discarding the speedup.

## Performance journey

The rows below use different workloads and should not be multiplied together.
“Projection” is one `8192 x 2048` Fisher fit. “Layer” is all seven Llama
decoder projections in paper stage order. The later canonical rows use equal
token totals but two different sequence geometries.

| Scope | Before | After | Result | Correctness/quality gate |
| --- | ---: | ---: | ---: | --- |
| Initial coordinate initializer, one projection | 14.868 s | 1.047 s | 14.2x | 265 legal changes across the model; zero Fisher regressions |
| Corrected 640-update projection | 10.2725 s | 2.2173 s | 4.63x | same 37-tile FP32-selected payload |
| Sparse Triton relaxation, steady state | 2.2173 s | 0.6732 s | 3.29x | same payload; FP64 oracle error improved over dense BF16 |
| Compact metadata, steady state | 0.6732 s | 0.58331 s | 1.154x | same payload; ten exact hard checkpoints retained |
| Complete staged layer: fused exact Hadamard | 80.053 s | 38.031 s | 2.105x | 14 tensors and 15.1786% held-out gain bitwise identical |
| Complete staged layer: sparse replay + concurrent Q/K | 38.031 s | 17.951 s | 2.119x | bitwise-identical state and held-out result |
| Complete staged layer: transformed Fisher Q/K | 17.951 s | 8.111--8.190 s | 2.19x--2.21x | deterministic; 14.8214% held-out gain |
| Complete staged layer with downstream Q/K guard | 17.951 s | 8.372--8.448 s | 2.13x--2.14x | best whole-block Q/K pair selected; 14.8214% gain |
| Direct accepted-state materialization | control | 2.46x--4.03x faster | local operation | identical states, guards, and losses |
| Fused identity-Fisher candidate constructor | 20.59 ms | 8.69 ms | 2.37x | exact packed/unpacked oracle parity |
| Fused screen with direct P32 word output | 20.59 ms | 6.77 ms | 3.04x cumulative | identical words, states, and losses |
| Attention CUDA graph, 512-token geometry | 0.3489 s | 0.2089 s | 1.67x | bitwise-identical state and 15.0000% gain |
| Attention CUDA graph, 256-token geometry | 0.5721 s | 0.2522 s | 2.27x | bitwise-identical state and 29.2717% gain |
| Trusted generated candidate buffers | 0.2057/0.2134 s | 0.1521/0.1548 s | 1.35x--1.38x | bitwise-identical states and losses |

The 15.0000% and 29.2717% values are not an accuracy trend. They come from
different sequence lengths, boundaries, and token hashes. Every optimization
was compared only with its matched control at the same geometry.

The initial Nsight result also illustrates the scale of the launch problem:
830,475 CUDA launches and 135,064 stream synchronizations became 12,023
launches and 1,046 synchronizations. In the later attention graph, 9,533 eager
kernel-launch API calls became 2,281 kernel/graph launches.

## CUDA and systems techniques that worked

### Make P32 sparsity the primary representation

- Score only the six W3 scalar changes induced by a legal edge edit.
- Keep candidate zero plus sparse indices and values; do not materialize a
  dense `[tiles, candidates, 16, 16]` bank.
- Scatter exact stored candidate values rather than recomputing
  `baseline + delta`, which would add an FP32 rounding boundary.
- Build the three legal P32 overlaps directly from consecutive trellis starts
  `s`, `s-1`, and `s-2`; avoid sorting generic six-entry maps.
- Store scalar indices and candidate IDs as compact integers and derive overlap
  metadata in the materialization kernel. This removed 899 MiB of per-layer
  allocation volume, including 744 MiB from MLP.
- Emit the selected legal 24-word payload from the screening kernel instead of
  launching clone/gather/mask/scatter repacking operations.

### Fuse around the mathematical boundary

- Fuse legal shift decode, PGC16 scalar decode, local quadratic scoring, stable
  shift selection, and packed-word emission.
- Fuse Gumbel softmax, sparse error construction, analytic probability
  gradient, and Lion update where their FP32 ordering can be preserved.
- Fuse fixed and trainable scales into Hadamard kernels, including right-Fisher
  and K-width specializations.
- Use a dedicated reverse-stage Hadamard backward kernel. A nominally
  self-adjoint transform was not bitwise equivalent because eager autograd
  traversed butterfly stages in reverse order.
- Write the final attention reconstruction as BF16 directly from the fused
  kernel while retaining FP32 transform, saved tensor, and backward arithmetic.
  This improved the isolated operators 1.20x--1.27x.
- Fuse finite checks and hard-loss scalar collection so validation does not
  synchronize once per tensor or microbatch.

### Remove launches and synchronization

- Replay fixed-shape optimizer steps with CUDA graphs, but feed fresh Gumbel
  noise and live temperature, multiplier, learning rate, and decay tensors on
  every replay. Restore all warm-up/capture mutations before real training.
- Batch several exact updates per graph replay when schedule values are
  unchanged over that replay unit.
- Keep telemetry, entropy, max probability, and finite state on device between
  hard checkpoints.
- Check structured hard states at meaningful schedule boundaries and always at
  the final update. For the 2,000-update Q/K schedule, 100 updates was the best
  measured interval while retaining the exact selected state and loss.
- Restrict graph-capture synchronization to the worker streams rather than the
  whole device.

### Exploit safe concurrency

- Build independent projection candidate banks concurrently on private CUDA
  streams.
- Fit independent Q and K projections concurrently with separate seeded RNGs
  and unchanged per-projection update order.
- Use free-threaded Python (`PYTHON_GIL=0`) so host launch threads do not
  serialize; the GIL-enabled oracle path was about 8% slower.
- Preserve deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for
  quality runs. Nondeterministic execution was faster but produced different
  tile assignments and scales from the same seed.

### Reuse immutable and already-validated work

- Cache transformed Fisher factors, dense position-error baselines, and layout
  maps at module construction.
- Reuse sparse candidates for hard checkpoints rather than decoding full banks.
- Decode an accepted hard state directly rather than constructing a temporary
  training module and screening candidates again.
- Cache dense held-out teacher outputs once and replay alternatives directly on
  the teacher layer.
- Trust immutable buffers produced internally by the validated candidate
  builder, while keeping validation and defensive copies in public APIs.
- Convert hard MLP weights from FP32 to BF16 once per checkpoint and retain the
  selected checkpoint for final reporting.

### Tune layout and launch geometry from profiles

- Emit the transposed sparse orientation consumed by the next Hadamard kernel,
  avoiding a materialize-then-transpose chain.
- Coalesce transposed mixture access and remove redundant FP32 transpose copies.
- Tune warp counts by shape: attention and large MLP projections wanted
  different launch widths.
- Use Nsight Systems for launch/synchronization attribution and Nsight Compute
  plus SASS to identify arithmetic and memory limits. The exact reverse
  Hadamard showed 57.99% SM throughput, 67.83% memory throughput, 18 registers
  per thread, no local memory, and no unintended FP64 or FMA across the required
  rounding boundary.

## Precision choices that worked

Precision was changed only where an independent oracle showed that the hard
decision was preserved:

- soft probability and Fisher products can use BF16 on the long schedule;
- sparse accumulation, candidate screening, hard objectives, no-regression
  selection, serialized scales, and exported weights remain FP32;
- final BF16 model-weight stores can be fused after the exact FP32 result is
  complete; and
- FP64 is useful as a sampled development oracle, not as a production compute
  path when FP32 already resolves the decision.

The sparse FP32-accumulation/BF16-product path was more accurate than the old
dense BF16 mixture against FP64: mean absolute error improved 13.4%, RMSE
16.4%, and maximum absolute error 48.0%. Hopper FP8 metric products were
rejected despite their speed because normalized metric-gradient RMSE rose to
5.93%, versus 0.41% for BF16.

## Positive quality results

The staged objective fixed the earlier “GSQ changes nothing” failure:

- with 2,048 training and 1,024 held-out tokens, increasing Q/K updates from 16
  to 256 made Q/K move and improved held-out full-block MSE by 13.18%; using 20
  block epochs also moved V/O and improved MSE by 21.64%;
- on a third report-only split, a freshly packed and SM90-reloaded layer-0
  checkpoint improved forward KL by 2.685%, logit squared error by 1.837%,
  cross-entropy by 0.282%, and perplexity by 0.874%;
- the guarded all-layer bounded run accepted layers 0, 2, 4, and 5 and improved
  untouched forward KL by 3.2908%, logit MSE by 2.1330%, cross-entropy by
  0.0680%, perplexity by 0.2000%, and teacher top-1 agreement by 0.1101 points;
  and
- native SM90 P32 inference with FA2 paged attention completed all 1,209
  GSM8K-Platinum examples and moved from 537 to 540 correct answers, with 39
  wrong-to-correct and 36 correct-to-wrong flips.

These are positive engineering and experimental results, not a paper-scale
accuracy claim. The GSM8K gain is only three answers, and the checkpoint was
created before deterministic training was enforced.

## Small wins retained after the major speedups

The following exact micro-optimizations all moved the canonical path forward
and remain useful implementation patterns:

- defer repeated MLP and attention hard validation into a final refinement
  tail, while retaining the final exact checkpoint;
- reuse reconstructed hard-evaluation weights and batch scalar transfers;
- fuse fixed/trainable `SV` scaling with forward and adjoint Hadamards;
- fuse sparse Lion moment and parameter updates;
- preallocate deterministic workspaces and reuse dense error baselines;
- eliminate full-bank candidate decodes in Q/K hard oracles;
- build P32 overlap maps without radix sorting;
- materialize compact V/O and MLP mixtures directly;
- specialize identity-Fisher screening and emit packed words in the same
  kernel;
- replay attention optimizer updates in one live-schedule CUDA graph; and
- reuse validated candidate buffers and the selected MLP hard weights.

Their isolated gains range from roughly 0.5% to 38%, depending on the stage.
They should be viewed as cleanup after the structural 9--18x gains, not as a
new multiplicative speedup chain.

## What not to repeat

- Do not call optimizer updates “epochs.” Report samples, valid tokens,
  logical batches, epochs, and updates separately.
- Do not pre-optimize the same frozen candidate pool with the same objective
  and then attribute the result to GSQ.
- Do not use one aggregate Fisher objective in place of the staged nonlinear
  reconstruction objectives when evaluating whether GSQ adds value.
- Do not keep `SV` fixed for P32; it is part of the deployable state.
- Do not compare held-out percentages from different token hashes or sequence
  geometries as an accuracy change.
- Do not weaken the final hard guard to report more changed tiles. Changed-tile
  count is diagnostic, not the objective.
- Do not replace FP32 with FP8 solely because a kernel is faster; validate the
  decision boundary against FP64 or an exact FP32 oracle.
- Do not assume CUDA graph replay is automatically faster. Q/K replay depths 8
  and 16 were bitwise exact but did not beat depth 4, and graphing the MLP BF16
  prefix regressed wall time.
- Do not retain a micro-optimization whose work merely moved into another
  range. Measure the combined dependent stages.

## Reusable playbook for QVQ and YAQA

The transferable sequence is:

1. Freeze one representative W3/P32 contiguous-window workload and persist its
   inputs, hard payload, and quality outputs.
2. Profile end-to-end with NVTX ranges; count launches, synchronizations,
   allocation volume, device traffic, and CPU dispatch gaps.
3. Write the mathematical invariant before writing a kernel. Prefer sparse
   deltas, transformed coordinates, and closed-form subproblems over faster
   evaluation of redundant dense work.
4. Keep exact hard selection and serialization separate from approximate soft
   training arithmetic.
5. Fuse producer-to-consumer boundaries so intermediate layouts, casts, and
   reductions never reach global memory.
6. Cache immutable transforms and legal metadata; directly consume trusted
   internally generated buffers.
7. Graph only fixed-shape launch trains, and make every changing scalar or RNG
   input explicit.
8. Parallelize only independent projections, each with a private stream and
   RNG, then verify that per-projection order is unchanged.
9. Compare complete serialized model state, changed-tile counts, stage losses,
   held-out loss, and disjointness—not just kernel output tolerances.
10. Benchmark matched geometries with warmup and alternating control/candidate
    order. Promote every real forward movement, but stop when gains saturate
    and move to the next dominant subsystem.

That final stop rule is the current decision: GSQ micro-optimization has
reached diminishing returns, so the next performance target is the underlying
QVQ + YAQA W3/P32 contiguous-window quantization pipeline.
