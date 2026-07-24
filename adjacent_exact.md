# AdjacentExact optimization log

This file records the reproducible performance and correctness work for the
classical AdjacentExact/GPTQ hybrid. It separates implementation optimization
from the broader quantum-feasibility narrative in `quantum.md`.

## Optimization contract

- Quality is the primary constraint. A faster path is retained only when its
  adjacent states, convergence flags, flip counts, and Hessian costs match the
  existing FP64 reference within the stated tolerance.
- The whole-model method remains complementary to Classic GPTQ. Full-row
  objective selection must continue to prevent a worse adjacent candidate from
  replacing the Classic result.
- Architecture-specific GPU work is selected from runtime device properties.
  Commands bind by UUID and never infer capability from a fixed CUDA index.
- CPU and CUDA fallbacks remain available for unsupported execution paths.
- Timings exclude JIT compilation and allocator warmup unless explicitly
  labeled otherwise.

## 2026-07-24 lifecycle and production configuration checkpoint

AdjacentExact is an explicit, quantization-time extension of GPTQ. It does not
replace Hessian collection, GPTQ's sequential error-feedback loop, range
selection, group scale/zero generation, packing, serialization, or inference.
The module lifecycle is:

1. Normal calibration calls `GPTQ.add_batch` and accumulates the activation
   Hessian.
2. When `QuantizeConfig.adjacent_model` is set, `GPTQ.quantize` preserves the
   original dense weight and undamped Hessian before GPTQ mutates its working
   copies.
3. Classic GPTQ runs to completion and produces its quantized weight, scales,
   zero points, and group index.
4. After GPTQ restores the original column order, AdjacentExact constructs a
   second, pack-compatible candidate from each weight's adjacent lower/upper
   code using the same scales and zero points. `executor` selects the CUDA,
   parallel CPU, or conservative automatic candidate path.
5. The guard evaluates Classic and Adjacent candidates with the original
   full-row Hessian objective. It substitutes an Adjacent row only when that
   row is strictly better beyond `selection_tolerance`; all other rows remain
   Classic GPTQ.
6. The resulting hybrid weight and the unchanged GPTQ scales, zero points, and
   group index continue through the normal pack/save/backend lifecycle.

Activation is explicit rather than model-detected:

```python
from gptqmodel.quantization.adjacent_model import AdjacentModelConfig
from gptqmodel.quantization.config import QuantizeConfig

quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=False,
    adjacent_model=AdjacentModelConfig(executor="auto"),
)
```

With `adjacent_model=None`, GPTQ does not capture the extra references or call
AdjacentExact. With it enabled, the row selector may replace zero, some, or all
Classic rows; if no Adjacent row wins, the returned quantized weight is exactly
the Classic result. Thus AdjacentExact complements Classic GPTQ by guarded
row-level selection, although the selected rows do override the corresponding
rows in the final quantized tensor.

The former `_adjacent_model_config` and `adjacent_config` hooks now fail with a
migration message instead of silently doing nothing. `candidate_device` was
renamed to `executor`; the research runner exposes the matching
`--executor {auto,cpu,cuda}` option and module statistics report
`executor_requested` and `executor`. The runtime policy is deliberately
omitted from checkpoint metadata: inference only needs the already-selected
packed weights, so this rename does not change the saved GPTQ format or Marlin
compatibility.

### Mathematical investigation: why it complements GPTQ

Let one output row of the dense weight be `w`, its quantized row be `q`, and
the calibration inputs be `X` with `N` token rows. GPTQ accumulates

```text
H = (2 / N) XᵀX.
```

For row error `e = w - q`, the calibration output reconstruction error is

```text
(1 / N) ||Xeᵀ||² = (1 / 2) eHeᵀ.
```

The hybrid guard computes `eHeᵀ`; the missing factor of one half is common to
all candidates and therefore cannot change which candidate wins.

Classic GPTQ and AdjacentExact attack this discrete reconstruction problem in
different ways:

| Property | Classic GPTQ | AdjacentExact candidate |
| --- | --- | --- |
| Weight seen by the quantizer | Sequentially compensated working weight | Original dense weight |
| Hessian use | Cholesky factor of the damped inverse Hessian | Original undamped group Hessian; original full Hessian for acceptance |
| Discrete strategy | Ordered greedy quantization plus future-column error feedback | Joint binary lower/upper-code selection |
| Quantization parameters | Searches/produces group scales and zero points | Reuses GPTQ's fixed scales and zero points |
| Reachable codes | May reach codes non-adjacent to the original weight after compensation | Only the original weight's clipped floor or ceiling code |
| Main strength | Propagates a current rounding error into all later columns | Can coordinate several correlated rounding directions at once |
| Main limitation | Greedy and order-dependent; no global discrete certificate | Restricted codebook; group-local search may ignore cross-group couplings |

If `R` is the upper Cholesky factor used for the damped inverse Hessian,
Classic GPTQ performs the following vector operation for column `i`:

```text
qᵢ = Quantize(wᵢ)
δᵢ = (wᵢ - qᵢ) / Rᵢᵢ
W[:, i:] ← W[:, i:] - δᵢ R[i, i:].
```

The next column is therefore quantized after compensation. This is why GPTQ
can choose a code that is not the floor or ceiling of that weight's original
affine coordinate. It is also why a certified optimum inside the adjacent
codebook is not automatically better than Classic GPTQ.

AdjacentExact instead freezes GPTQ's scale `s` and zero point `ζ`. For a group
of `g` original weights, define:

```text
cᵢ = wᵢ / s + ζ
ℓᵢ = clip(floor(cᵢ), 0, 2ᵇ - 1)
uᵢ = clip(ceil(cᵢ),  0, 2ᵇ - 1)
aᵢ = s(ℓᵢ - ζ)
dᵢ = s(uᵢ - ℓᵢ)
q(z) = a + Dz, where D = diag(d) and z ∈ {0,1}ᵍ.
```

With residual `r = w - a` and the group Hessian `H_g`, its objective expands
exactly into the QUBO:

```text
J_A(z) = (r - Dz)ᵀ H_g (r - Dz)
       = c₀ + Σᵢ hᵢzᵢ + Σᵢ<ⱼ Kᵢⱼzᵢzⱼ

c₀   = rᵀH_gr
hᵢ   = dᵢ²(H_g)ᵢᵢ - 2dᵢ(H_gr)ᵢ
Kᵢⱼ = 2dᵢdⱼ(H_g)ᵢⱼ.
```

The pair term is the source of the useful complementarity. RTN treats weights
independently. When `H_g` has off-diagonal correlations, two individually
unattractive flips can jointly reduce activation error, and a joint binary
search can recover that pattern. If `H_g` is diagonal, the QUBO separates into
independent decisions and its fixed-codebook optimum reduces to nearest
rounding, apart from ties and clipped endpoints. The opportunity therefore
comes from correlated activations, and its potential generally grows when
ultra-low-bit steps make rounding residuals larger.

The production candidate is assembled group by group. If the full row is
partitioned into errors `e_g`, then:

```text
eHeᵀ = Σ_g e_g H_gg e_gᵀ + 2 Σ_g<h e_g H_gh e_hᵀ.
```

Group candidate construction optimizes terms using `H_gg`; it does not jointly
optimize every cross-group term `H_gh`. The final row guard deliberately
replays the complete expression, including all cross-group couplings. This
explains how every local group can look reasonable while the concatenated
Adjacent row is worse than GPTQ.

For output row `r`, let `J_G(r)` be the Classic GPTQ full-Hessian cost and
`J_A(r)` the Adjacent cost. The implemented selector is:

```text
use Adjacent iff J_A(r) < J_G(r) - τ(1 + |J_G(r)|);
otherwise use Classic GPTQ.
```

Because the layer reconstruction objective is a sum over output rows, this
row-wise choice yields the lower measured candidate cost independently for
each row. It gives the hybrid a no-regression guarantee relative to Classic
GPTQ on this calibration-Hessian objective, within the stated floating-point
tolerance. It does not guarantee lower perplexity or better downstream task
accuracy: the Hessian is a local layer-output proxy built from finite
calibration data, and later nonlinear layers can amplify small differences.

The current whole-model path must also be described precisely: its multi-start
coordinate search is a bounded candidate generator, and only native
branch-and-bound results that finish their search carry an exact certificate.
It does not optimize scales/zero points or arbitrary `b`-bit codes. The
mathematical value today is therefore candidate diversity plus a safe
full-Hessian selector, not a claim that the entire model is globally
AdjacentExact.

The model experiments support this interpretation:

- Structured group tests often favored AdjacentExact because the constructed
  correlation blocks aligned with its joint binary objective.
- Llama 3.2 1B accepted only 27 of 376,832 rows. Its measured Hessian objective
  improved minutely, but downstream scores were mixed and GSM8K regressed,
  confirming that the guard protects only its stated proxy.
- Qwen3-8B accepted zero of 1,400,832 rows. Classic GPTQ's compensated,
  non-adjacent solution beat the concatenated group-local candidate for every
  full row, so the hybrid correctly collapsed to the exact Classic checkpoint.

The most promising accuracy research direction is consequently not to replace
GPTQ, but to improve the alternative candidate: include cross-group/full-row
couplings, jointly reconsider range parameters where pack compatibility
allows it, or use a downstream-aware acceptance proxy while retaining Classic
GPTQ as the fallback.

### AWQ applicability investigation

Verdict: the adjacent binary optimization is mathematically applicable to AWQ,
but the current whole-model integration is not. The implementation now rejects
`adjacent_model` with `method="awq"` instead of silently accepting an unused
option.

The QUBO is not inherently GPTQ-specific. It needs four objects: a dense
reference weight, a calibration curvature or equivalent activation matrix, a
fixed affine quantization grid, and a baseline candidate. AWQ has three of
these directly:

- it retains calibration input features;
- after activation-aware rescaling and clipping, `pseudo_quantize_tensor`
  produces a fixed scale and zero point for every row-group;
- its normal pseudo-quantized weight is the baseline RTN candidate on that
  grid.

AWQ does not currently materialize a Hessian and never enters `GPTQ.quantize`.
Its scale search uses activation statistics and explicit reconstruction
forwards, and its clip search directly compares group outputs. Therefore the
existing GPTQ hook cannot simply be enabled for AWQ.

A mathematically consistent AWQ adaptation would run in this order:

1. Complete AWQ's activation-aware scaling search and preserve the scaled
   full-precision weight before clipping.
2. Complete clipping and derive AWQ's final scales, zero points, and normal
   RTN weight.
3. Use the correspondingly scaled captured inputs `X_A` to form group
   Hessians `H_g = (2/N) X_A,gᵀX_A,g`.
4. Build the same lower/upper-code QUBO on AWQ's final affine grid.
5. Compare the completed AWQ and Adjacent rows against the preserved
   pre-clipping scaled reference, using either the full Hessian or the
   equivalent direct output error `||X_A(W_ref-Q)ᵀ||²`.
6. Pass the selected grid-exact row through the unchanged AWQ packer.

The pre-clipping reference matters. Comparing only against AWQ's already
clipped weight would guarantee lower post-clipping quantization error but could
still increase total error relative to the full-precision scaled layer. Using
the pre-clipping reference makes the acceptance test cover both clipping and
rounding relative to the same AWQ baseline.

AWQ can avoid storing a full dense Hessian: its captured activations can
produce each at-most-128-column group Hessian on demand, while the final guard
can stream direct output-error chunks. Its symmetric path already exposes the
signed grid through a shifted unsigned zero point, so the existing affine
floor/ceiling algebra remains valid and pack-compatible.

Required engineering work remains substantial: preserve the correct
pre-clipping reference, thread transformed module-specific activations into
`apply_quant`, handle shared AWQ scaling groups and tensor-parallel padding,
generalize the current CUDA-resident GPTQ hook, and verify every AWQ packing
format. An AWQ experiment must compare dense, normal AWQ, and AWQ-plus-Adjacent
with identical calibration data before this becomes a supported config.

Validation used physical GPU 0 UUID
`GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2` (PG506-230, 96 GiB, compute
capability 8.0) with `PYTHON_GIL=0` and `TORCH_CUDA_ARCH_LIST=8.0`:

- 60/60 Adjacent math, public-config, AWQ rejection, whole-model hook, CPU/CUDA
  executor-equivalence, native CUDA, scenario, and quantum-bridge tests passed
  in 14.04 seconds.
- The runner help exposes `--executor {auto,cpu,cuda}` and no
  `--candidate-device` option.
- `git diff --check` passed.
- Ruff passed on every changed Adjacent path when excluding `E722`. The
  unfiltered invocation remains blocked by two pre-existing bare `except`
  clauses in `gptq.py` lines 805 and 810, outside this change.

## 2026-07-23 baseline: Qwen3-8B production-shaped candidate phase

The matched baseline ran on physical GPU 0 UUID
`GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2`, a 96 GiB PG506-230:

| Property | Value |
| --- | --- |
| Compute capability / SMs | 8.0 / 124 |
| Driver / CUDA runtime | 610.43.02 / 13.0 |
| Torch | 2.13.0+cu130 |
| Build target | `-gencode=arch=compute_80,code=sm_80` |
| Python | 3.14.5 free-threading, GIL disabled |
| Visible logical CPUs | 96 |
| Quantization | 4-bit, group size 128, `sym=True` |
| Coordinate search | four starts, 32-flip cap, rebase every 8 flips |
| Samples | three warmed full-task-count runs; median reported |

The benchmark uses real Qwen3-8B layer-0 projection weights and dimensions.
CPU and GPU receive identical deterministic correlated FP64 Hessians. The
Hessians are synthetic, so this is an execution-path benchmark rather than a
model-quality measurement.

| Module | Shape | Tasks | CPU-1 | CPU-4 | GPU wall | GPU speedup over CPU-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q_proj | 4096 x 4096 | 64 | 21.578 s | 5.203 s | 4.016 s | 1.295x |
| k_proj | 1024 x 4096 | 32 | 5.265 s | 1.336 s | 1.941 s | 0.689x |
| v_proj | 1024 x 4096 | 32 | 5.103 s | 1.298 s | 1.925 s | 0.674x |
| o_proj | 4096 x 4096 | 64 | 20.989 s | 5.181 s | 3.940 s | 1.315x |
| gate_proj | 12288 x 4096 | 192 | 65.789 s | 16.065 s | 11.620 s | 1.383x |
| up_proj | 12288 x 4096 | 192 | 63.096 s | 15.418 s | 11.900 s | 1.296x |
| down_proj | 4096 x 12288 | 192 | 68.161 s | 15.252 s | 11.904 s | 1.281x |
| One layer | seven projections | 768 | 249.982 s | 59.753 s | 47.247 s | 1.265x |

Here, GPU speedup is `CPU-4 wall / GPU wall`; values below one mean CPU-4 is
faster. Four CPU workers provide 4.184x aggregate speedup over one CPU worker.
CPU and GPU candidates are bit-identical for all seven types, and the maximum
absolute FP64 cost difference is `8.674e-19`.

The native branch-and-bound tail has no matched 128-variable CPU exhaustive
baseline. Real groups contain 126-128 active decisions, while the CPU exhaustive
reference is intentionally capped at 20. With split depth 6 and 500 nodes per
worker, GPU 0 takes 0.123-0.144 seconds per bounded refinement, visits 32,000
nodes, and does not certify these hard examples.

Artifacts:

- `scripts/quantum_quantization/benchmark_adjacent_model_cpu_gpu.py`
- `scripts/quantum_quantization/results/adjacent_model_cpu_gpu/20260723_gpu0/results.json`
- `scripts/quantum_quantization/results/adjacent_model_cpu_gpu/20260723_gpu0/run.log`

## 2026-07-23 active optimization study

Target: reduce both sequential GPU and outer-parallel CPU time for the
production `_adjacent_group_candidate` phase, then measure whether concurrent
CPU+GPU scheduling is worthwhile.

Candidate mechanisms under test:

1. remove avoidable CUDA scalar synchronizations from the coordinate loop;
2. batch independent coordinate starts when it improves full-task latency;
3. reduce repeated temporary allocation and indexing work;
4. tune CPU task granularity and worker count on the real projection shapes;
5. consider a fused native coordinate operator only if profiling shows launch
   and synchronization overhead remains material after the simpler changes.

No optimization is accepted until the seven-module correctness and timing
comparison is complete.

### Baseline launch/synchronization profile

Nsight Systems 2024.6.2 captured one warmed `q_proj` 2048-by-128 candidate
task inside a CUDA profiler range. Profiling overhead increased the span, so
these numbers are attribution evidence rather than final latency:

| Metric | Baseline |
| --- | ---: |
| NVTX projected GPU span | 114.378 ms |
| GPU operations | 4,298 |
| `cudaLaunchKernel` calls | 4,022 |
| CUDA launch API time | 30.076 ms |
| Stream synchronizations | 247 |
| Device-to-host copies | 247 |
| Summed GPU kernel time | 18.889 ms |

The dominant families were advanced-indexing kernels: 465 calls consumed
19.6% of kernel time and another 352 calls consumed 14.1%. The trace confirms
that data-dependent `nonzero`/scalar checks and eager operator dispatch, rather
than one slow numerical kernel, dominate the baseline.

Artifacts:

- `scripts/quantum_quantization/results/adjacent_optimization/20260723_gpu0/baseline_q_proj.nsys-rep`
- `scripts/quantum_quantization/results/adjacent_optimization/20260723_gpu0/baseline_q_proj_stats.txt`

### Retained checkpoint 1: cached arithmetic and synchronization-free CUDA

The first change caches `steps² * diag(H)` and maintains the signed step
direction incrementally after a flip. It removes repeated full-tensor work on
both CPU and GPU without changing the coordinate decisions.

The second change retains the original dynamic-row implementation as the CPU
and numerical reference path, while CUDA runs a fixed iteration count with
masked no-op updates. This removes the per-iteration host decision and
dynamically shaped `nonzero` result.

| Variant, real q_proj task count | CPU-1 | CPU-4 | GPU |
| --- | ---: | ---: | ---: |
| Original baseline | 21.578 s | 5.203 s | 4.016 s |
| Cached signed-step arithmetic | 19.298 s | 4.450 s | 3.506 s |
| Plus synchronization-free CUDA | 17.352 s | 4.434 s | 2.837 s |

The cached-arithmetic checkpoint improved CPU-1 by 10.6%, CPU-4 by 14.5%, and
GPU by 12.7% relative to the original run. The synchronization-free CUDA path
then improved GPU by another 19.1%, reaching a 29.4% reduction from baseline.
CPU-1 varied between the two post-change runs even though its source was
unchanged; the final seven-module run will provide the accepted CPU comparison.

The real 2048-by-128 Qwen task produced exactly equal candidates, FP64 costs,
convergence flags, and flip counts between the dynamic and masked CUDA
implementations. The complete hybrid test file passed 8/8 cases on physical
GPU 0.

### Retained checkpoint 2: batched CUDA coordinate starts

The four independent `nearest/zero/one/linear` starts now share one larger CUDA
launch sequence. The fallback can disable batching through the internal
`batch_coordinate_starts_on_cuda` research configuration.

| Variant, real q_proj task count | GPU wall | Speedup from prior | Speedup from original |
| --- | ---: | ---: | ---: |
| Synchronization-free, sequential starts | 2.837 s | 1.000x | 1.416x |
| Synchronization-free, batched starts | 0.776 s | 3.656x | 5.175x |

The batched and sequential implementations selected exactly equal states at
2, 3, 4, and 8 bits in focused CUDA tests. On the real 2048-by-128 Qwen weight
slice they also produced bit-identical candidates, FP64 costs, convergence
flags, and flip counts. `tests/test_adjacent_model_hybrid.py` passed 8/8 cases.

Artifacts:

- `scripts/quantum_quantization/results/adjacent_optimization/20260723_gpu0/arithmetic_cache_q_proj.json`
- `scripts/quantum_quantization/results/adjacent_optimization/20260723_gpu0/cuda_masked_q_proj.json`
- `scripts/quantum_quantization/results/adjacent_optimization/20260723_gpu0/batched_starts_q_proj.json`

### Validation incident: unrelated extension cold build

The first combined test invocation passed 59 Adjacent and extension cases, then
the extension registry's `load all` test escaped its fake environment and began
compiling the unrelated GrassHopper CUDA extension. The run was interrupted
after 2 minutes 46 seconds; this was neither an Adjacent numerical failure nor a
CUDA runtime failure.

Root cause: `tests/test_extension_load_api.py` did not install a fake for the
existing GrassHopper extension or include it in the expected `load all` result.
The test harness now supplies that fake. The complete extension test file then
passed 14/14 cases in 9.63 seconds without compiling production kernels.

Checkpoint validation:

- Ruff passed for all Adjacent implementation, benchmark, and test files.
- `git diff --check` passed.
- 57 Adjacent math, CUDA, whole-model, scenario, and quantum-bridge cases passed
  before the extension tests.
- The corrected extension-dispatch file passed 14/14 cases.

### Rejected checkpoint: batching coordinate starts on CPU

The CUDA four-start batching mechanism was tested unchanged on CPU with the
same real Qwen `q_proj` shape and 64 production candidate tasks. It was rejected:

| CPU path | CPU-1 | CPU-4 |
| --- | ---: | ---: |
| Retained sequential starts | 19.235 s | 4.703 s |
| Batched four starts | 59.645 s | 15.343 s |
| Regression | 3.101x slower | 3.262x slower |

CPU batching enlarges the active working set by four, forces every start to run
until the slowest start reaches the stopping point, and loses the cache benefit
of processing starts independently. GPU still measured 0.802 seconds during
the rejected run, consistent with its retained 0.776-second result.

The experimental CPU batching switch and code path were removed. CPU keeps the
cached signed-step arithmetic and dynamic active-row reference loop. The raw
rejection measurement remains local at
`scripts/quantum_quantization/results/adjacent_optimization/20260723_gpu0/cpu_batched_starts_q_proj.json`.

### Verified checkpoint 3: finer CPU tasks plus outer parallelism

The CPU coordinate implementation releases enough work under free-threaded
Python to parallelize independent row-chunk/group candidate tasks. A broad
`q_proj` sweep showed that the earlier four-worker result was limited by its
coarse 2048-row tasks, not by a serial dependency in AdjacentExact.

The selected validation uses the same physical GPU-0 binding and real Qwen
weights as the baseline, although the candidate computation itself is on the
host. This host exposes 96 logical CPUs from two AMD EPYC 7V13 sockets. Torch
intra-op parallelism is fixed at one so each outer worker owns one candidate
task. Five warmed measurements gave:

| CPU scheduler | Median | Range | Speedup |
| --- | ---: | ---: | ---: |
| Original CPU-1, 2048 rows | 21.578 s | baseline run | 1.000x |
| Cached CPU-1, 2048 rows | 18.183 s | broad sweep | 1.187x |
| Original CPU-4, 2048 rows | 5.203 s | baseline run | 4.147x |
| Optimized GPU, 2048 rows | 0.776 s | retained CUDA checkpoint | 27.807x |
| CPU-64, 512 rows | **0.443 s** | 0.438-0.462 s | **48.662x** |

The final column is relative to the original CPU-1 time. Equivalently, the
fine-grained CPU-64 scheduler is 1.750x faster than the current optimized GPU
candidate phase on this `q_proj` workload. All five CPU-64 repetitions and the
CPU-96 comparison returned the same aggregate candidate-cost, convergence, and
flip checksum. A separate seven-repeat neighborhood sweep found 0.460-second
medians for both CPU-80 and CPU-96, but CPU-80 suffered a 1.355-second outlier.
The 64-worker point is retained as the lower-noise setting on this host.

This checkpoint establishes the scheduler and tuning result; wiring CPU
offload into the whole-model hybrid is still pending. It does not change the
default CUDA execution path.

Artifact:

- `scripts/quantum_quantization/benchmark_adjacent_cpu_scaling.py`
- `scripts/quantum_quantization/results/adjacent_optimization/20260723_gpu0/cpu_scaling_q_proj_selected.json`

## 2026-07-24 verified failure: Qwen3-8B whole-model A/B

The serial seed-898 Qwen3-8B experiment on physical GPU 6 completed dense BF16,
Classic GPTQ, and Adjacent-hybrid quantization/evaluation. It used 4-bit,
group-size-128, `sym=True`, identical calibration settings for both quantized
runs, and Marlin evaluation.

| Model | ARC acc | ARC normalized | GSM8K Platinum | Quant wall | Peak process GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense BF16 | 0.558020 | 0.558874 | 0.924731 | n/a | n/a |
| Classic GPTQ | 0.546075 | 0.547782 | 0.911497 | 723.215 s | 8,986 MiB |
| Adjacent hybrid | 0.546075 | 0.547782 | 0.911497 | 2,293.886 s | 11,390 MiB |

The Adjacent search processed 54,263,808 row-groups in 252 modules, attempted
1,008 bounded native refinements, certified 60, and improved 547 local group
objectives. Its unguarded full-Hessian error was `32467.778694544693`, versus
`15583.444859792653` for the run-local Classic replay. The quality guard
selected zero of 1,400,832 rows, making the saved hybrid exactly Classic GPTQ
and therefore giving exactly equal downstream scores.

This checkpoint is rejected as a quality improvement: it took 3.172x longer,
sampled 2,404 MiB more process GPU memory, and improved no guarded row. It is
also evidence that local group-Hessian coordinate improvements and the current
bounded native tail are misaligned with Classic GPTQ's whole-row result on
this 4-bit/group-128 workload. The run predates the retained execution-only
CUDA speedups, so the timing is not used as their whole-model performance
measurement; candidate equivalence means the zero-row conclusion still holds.

Artifact:

- `scripts/quantum_quantization/results/qwen3_8b_adjacent_ab/20260723T222838Z/summary.json`

## 2026-07-24 retained checkpoint 4: conservative CPU offload routing

The fine-grained CPU scheduler was integrated as an explicit whole-module
candidate executor, including CUDA-to-host input transfer, parallel CPU
candidate construction, host-to-CUDA output transfer, and the unchanged CUDA
full-Hessian quality guard. An opt-in `auto` policy selects CPU only when:

- Python's GIL is disabled;
- the process affinity exposes at least the requested worker count;
- quantization is the measured 4-bit, group-size-128 regime;
- the module has at least 393,216 row-groups.

The default `AdjacentModelConfig.executor` remains `cuda`. The whole-model
research runner defaults to the probed `auto` policy and records the requested
and resolved executor per module.

### Prototype correction

The repeated-prototype seven-module sweep measured 5.274 seconds for CPU-64
versus 9.934 seconds for optimized CUDA. It was useful for discovering outer
parallelism, but it repeatedly touched the same 512-by-128 CPU slice. That
overstates production cache locality. It is not retained as the whole-module
speed claim.

The acceptance benchmark therefore loads every full Qwen3-8B layer-0 weight
matrix. It uses deterministic full block-diagonal SPD Hessians so full-row
objective replay is executable while the candidate phase retains genuine
group-128 coupling. Timings include all offload and pool-creation overhead,
disable the native refinement tail, alternate CPU/CUDA order, warm each path
once, and report the median of three runs.

| Module | Row-groups | CUDA candidate | CPU-64 candidate | CPU/CUDA speedup | Auto route |
| --- | ---: | ---: | ---: | ---: | --- |
| q_proj | 131,072 | 0.753 s | 0.762 s | 0.988x | CUDA |
| k_proj | 32,768 | 0.384 s | 0.402 s | 0.955x | CUDA |
| v_proj | 32,768 | 0.349 s | 0.337 s | 1.035x | CUDA |
| o_proj | 131,072 | 0.741 s | 0.666 s | 1.112x | CUDA |
| gate_proj | 393,216 | 2.266 s | 1.784 s | 1.270x | CPU |
| up_proj | 393,216 | 2.266 s | 1.630 s | 1.390x | CPU |
| down_proj | 393,216 | 2.321 s | 1.507 s | 1.540x | CPU |
| Projected serial layer | 1,507,328 | 9.079 s | 7.147 s selected | **1.270x** | hybrid |

`CPU/CUDA speedup` is CUDA time divided by CPU time. The static auto threshold
deliberately leaves V/O on CUDA despite small isolated wins because Q/K share
their shape classes and regress after offload. It avoids online double
execution and makes no model-name assumption.

Including full objective replay, the independently measured module medians sum
to 9.343 seconds for all-CUDA and 7.426 seconds for the conservative route, a
1.258x projected layer speedup. Every one of the seven CPU/CUDA pairs produced
an exactly equal hybrid tensor, convergence count, capped count, total and
maximum flip count, and selected-row count. Full Classic and Adjacent Hessian
errors matched exactly; the largest local-group FP64 sum difference was
`7.105e-14`.

Peak CUDA allocation changes little because full-Hessian objective replay stays
on GPU: the maximum delta was 773.414 MiB on CUDA and 772.388 MiB with CPU
candidate offload. CPU host-memory peaks are not claimed by this benchmark.

Artifacts:

- `scripts/quantum_quantization/benchmark_adjacent_model_cpu_offload.py`
- `scripts/quantum_quantization/results/adjacent_optimization/20260723_gpu0/cpu_offload_*.json`
- `scripts/quantum_quantization/results/adjacent_optimization/20260723_gpu0/post_optimization_7module.json`
