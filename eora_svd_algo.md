# EoRA calibration-time LoRA generation: design and optimization log

## Scope and status

This work accelerates the math that generates EoRA LoRA factors from calibration activations. It does not change the
EoRA inference kernel, Marlin dispatch, or the format of saved adapters.

As of 2026-07-23, the implementation combines:

1. exact reuse of calibration covariance contributions for modules that consume the same activation tensor; and
2. an explicit `EoRAConfig` selecting exact, CUDA-accelerated automatic, or randomized low-rank SVD.

`lowrank` is the default for newly constructed configurations. Across matched rank-32, rank-64, rank-128, and rank-256
full-dataset ARC Challenge and GSM8K Platinum comparisons, it led or tied eight of twelve within-rank task metrics and
was the fastest generation route at every tested rank. `exact` remains available as the reference path and is retained
for serialized descriptors that predate `EoRAConfig`; `auto` remains available as the cuSOLVER route with exact
fallback. The repaired 3-bit sweep retained this default: 3-bit quality leadership was mixed by rank and task, while
`lowrank` remained the fastest route at all four ranks and generated 1.576x to 1.603x faster than `exact`.

## Progress log

- Established the 931.871 s exact generation baseline and recorded the model, calibration, rank, software, and physical
  GPU topology.
- Timed the generation phases and captured representative real `Delta C` matrices. Full SVD, rather than launch count,
  Cholesky, or the surrounding GEMMs, was the dominant cost.
- Benchmarked exact SVD, CUDA `gesvda`, and a deterministic randomized rank-focused SVD on physical GPUs 2 and 3.
- Replaced the development environment variable with serialized `EoRAConfig(algo=...)`. Added `exact`, `auto`, and
  `lowrank` routes with exact portable and runtime-failure fallbacks.
- Added exact same-input covariance reuse. An adversarial distinct-input test exposed allocator address reuse; cache
  entries now retain the source objects until all siblings consume them, eliminating the false-hit opportunity.
- Completed full 512-row generation runs for `auto` and `lowrank`, checked every saved factor for finite values, and
  compared generated outputs and internal hidden states with the dense reference and exact adapter.
- Recreated all three adapters concurrently on physical GPUs 2, 3, and 4, then completed full ARC Challenge and GSM8K
  Platinum evaluation with Evalution at rank 128.
- Repeated the three algorithms at rank 256 on GPUs 2, 3, and 4 while evaluating the no-EoRA base quantized model on
  GPU 5. The combined results selected `lowrank` as the score-leading default and retained the per-task tradeoffs.
- Added rank-64 exact, auto, and lowrank runs on GPUs 2, 3, and 4, plus rank-32 exact on GPU 5. After the rank-64
  accelerated lanes completed, GPUs 3 and 4 ran rank-32 auto and lowrank. All six full generation/evaluation pipelines
  completed with zero invalid GSM8K answers.
- Paused the matched 3-bit extension when the first joint GPTQ+EoRA snapshot produced 64/64 invalid GSM8K answers.
  A clean native 3-bit, group-size-128, activation-scale-search, GAR quantization without EoRA scored 57/64 (89.06%)
  on the same bounded Evalution check with zero invalid outputs, proving that native 3-bit GPTQ was healthy.
- Isolated inference from serialization. Native FP16 TriLin ran on all 252 quantized projections, made 6,300 observed
  native calls, and generated the same 24 tokens as eager Torch. On the healthy base plus the generated adapter, fused
  TriLin+EoRA made 1,656 native fused calls and exactly matched the unfused generated token IDs.
- Traced the failed snapshot to unchecked reconstructed codes in the Python/GPU/original pack paths. Values immediately
  outside `[0, 2**bits - 1]` wrapped during bit packing, while the existing native CPU extension already saturated.
  Added consistent saturation and regression coverage for 2-, 3-, 4-, and 8-bit packing. A separate 2-bit control
  remained coherent and showed no material endpoint-wrap signature.
- Recreated the joint 3-bit/rank-128 artifact with both GPU and CPU/Python packing. The saved model shards and adapter
  were byte-identical across packers; sampled packed-weight cosine recovered from 0.696133 to 0.974279 and bounded
  GSM8K Platinum recovered from 0/64 with 64 invalid outputs to 58/64 with none invalid.
- Measured 46.693 s for all 252 GPU pack finalizations versus 257.603 s for CPU/Python packing, a 5.52x packing
  speedup and 255.5 s reduction in total quantization wall time. Published the isolated fix upstream without the
  Ultra-only packing skill or diagnosis artifacts.
- Completed the repaired 3-bit exact/auto/lowrank sweep at ranks 32, 64, 128, and 256 on physical GPUs 0 through 5,
  plus the full adapter-free baseline. All 12 adapters had 504 finite tensors at the requested rank and zero invalid
  GSM8K outputs; the base had one coherent non-numeric response. Added exact 3-bit-versus-4-bit score deltas below.
- Ran the focused unit/regression suite, Ruff, and whitespace validation.

## EoRA generation math

For a linear layer, calibration replay accumulates an activation covariance from input batches `X`:

```text
H <- scaled running accumulation of X^T X
```

Let `Delta = W - Wq` be the float32 residual between the dense and quantized weights. The normal positive-definite path
computes a Cholesky factor `C` of `H`, then forms the activation-weighted residual:

```text
M = Delta C
M = U S V^T
B = U_r sqrt(S_r)
A = sqrt(S_r) V_r^T C^-1
```

The saved adapter product `B A` is therefore a rank-`r` approximation of `Delta` under the calibration covariance.
When Cholesky is unavailable or `H` is not positive definite, the established eigendecomposition path constructs the
square-root and truncated-pseudoinverse factors instead. Both paths now share the same SVD selector.

## Bottleneck diagnosis and mega-kernel decision

Phase timing on real rank-128 EoRA matrices showed that the full SVD takes about 1.8 seconds on ordinary
4096-dimensional layers, while Cholesky and the surrounding matrix multiplies together remain below roughly 0.12
seconds. The workload is dominated by a global iterative decomposition, not Python dispatch or a sequence of tiny GPU
launches.

A mega-kernel was therefore not retained. Fusing the neighboring factorization, multiply, and solve operations would
not eliminate the SVD's internal global synchronization, and reproducing a robust SVD inside a custom cooperative
kernel would add substantial numerical and architecture risk. Calling the optimized cuSOLVER algorithm and removing
duplicated covariance GEMMs attacks the measured costs directly while preserving portable fallbacks.

## Retained optimization 1: exact covariance reuse

Q/K/V projections and gate/up projections commonly receive the same calibration activation. Previously, every module
independently applied the keep mask and calculated the same `X^T X` contribution.

`EoraProcessor` now identifies compatible modules inside each replay subset and reuses a contribution only when all of
the following match:

- subset and ordered module group;
- calibration batch index;
- input columns and target device;
- source storage, shape, stride, offset, dtype, and device; and
- keep-mask identity and view metadata.

The cached contribution is cloned for each sibling, preserving the pre-existing independent accumulation semantics and
bitwise output. The cache retains references to the source and mask until all expected consumers finish, preventing a
false match if the allocator later reuses a freed storage address. Entries are discarded at subset cleanup.

This optimization is enabled by default. It can be disabled for diagnosis with:

```bash
GPTQMODEL_EORA_SHARED_COVARIANCE=0
```

## Retained optimization 2: explicit SVD configuration

The algorithm is part of the adapter-generation configuration rather than process-global environment state:

```python
from gptqmodel.adapter.adapter import EoRAConfig, Lora

adapter = Lora(rank=128, eora_config=EoRAConfig(algo="lowrank"))
```

`EoRAConfig.algo` controls only calibration-time factor generation:

| Algorithm | Behavior |
|:---|:---|
| `exact` | Always use `torch.linalg.svd(..., full_matrices=False)` with the default exact solver. This matches generation before this change. |
| `auto` | On NVIDIA CUDA, try `torch.linalg.svd(..., driver="gesvda")`; on failure, retry with the exact default solver. CPU and ROCm use the exact solver. |
| `lowrank` | Default for new configs. Use the deterministic randomized rank-focused algorithm described below, with exact fallback on runtime failure. |

The nested config is included by `Lora.to_dict()` and restored by `normalize_adapter()`. A newly constructed
`EoRAConfig()` selects `lowrank`. Old serialized Lora descriptors without `eora_config` still load as `exact`, preserving
their pre-configuration generation behavior. Invalid values fail immediately rather than silently selecting an
unintended algorithm.

### What `EoRAConfig(algo="lowrank")` does

`lowrank` avoids decomposing the full `m x n` matrix when only the leading rank-`r` singular triplets are needed. It:

1. creates a deterministic Gaussian probe using a local generator seeded with `0xE0A`;
2. uses a subspace of `min(min(m, n), 2r)` vectors;
3. performs four alternating QR-based power iterations to concentrate that subspace on the leading singular vectors;
4. projects the large matrix into the small subspace;
5. performs an exact SVD of that smaller projected matrix; and
6. lifts the singular vectors back to the original space.

For rank 128, the usual projected dimension is 256; at rank 256 it is 512. Wide matrices are transposed internally so
the expensive subspace work follows the smaller orientation. If the requested subspace covers the full matrix, or if
the randomized route raises a runtime error, the implementation uses exact SVD.

The fixed local seed makes results repeatable without consuming or changing the application's global random-number
state. This mode changes the generated factors and requires regenerating the adapter. It has no effect when loading an
existing adapter and no effect on inference dispatch.

The tradeoff is numerical: exact SVD decomposes the full matrix, and `gesvda` applies cuSOLVER's approximate algorithm
to the full matrix before rank truncation. `lowrank` instead estimates only the leading subspace. Real captured matrices
were much faster, but their extra rank-128 reconstruction residual ranged from `2.366e-5` to `3.676e-4` of total matrix
energy on ordinary layers and `1.085e-4` on the sampled rank-deficient layer. The complete matched evaluation found no
score regression and selected `lowrank` as the default; explicit `auto` and `exact` remain available for comparison or
more conservative generation policies.

## Hardware and software

The original optimization and controlled 4-bit evaluation used physical GPUs 2, 3, 4, and 5. The 3-bit regression
isolation and fix verification used physical GPUs 0, 1, and 2; the subsequent controlled rank sweep used only physical
GPUs 0 through 5. All were selected with PCI ordering:

| Physical GPU | PCI bus | Device | Compute capability | SMs | Memory |
|---:|:---|:---|:---:|---:|---:|
| 0 | `00000000:25:00.0` | NVIDIA PG506-230/232 | 8.0 | 124 | 98,304 MiB |
| 1 | `00000000:2B:00.0` | NVIDIA PG506-230/232 | 8.0 | 124 | 98,304 MiB |
| 2 | `00000000:64:00.0` | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |
| 3 | `00000000:69:00.0` | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |
| 4 | `00000000:A0:00.0` | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |
| 5 | `00000000:A5:00.0` | NVIDIA PG506-230 | 8.0 | 124 | 98,304 MiB |

```text
Driver                 610.43.02
Python interpreter     /root/vm314t/bin/python
Python                 3.14.5 free-threaded build
PyTorch                2.13.0+cu130
CUDA runtime/toolkit   13.0 / 13.0.88
Triton                 3.7.1
Transformers           5.14.1
```

No capability decision in the implementation depends on these fixed device indices. Runtime tensor device and backend
checks select the CUDA path; CPU, ROCm, unsupported, and failed CUDA cases retain exact behavior.

## Captured-matrix benchmark

The benchmark uses saved float32, two-dimensional `Delta C` matrices from the real calibration run, warmed CUDA-event
timing, rank 128, and a chunked direct reconstruction-residual check. Times are p50.

| Matrix | Route | Latency | Speedup vs exact | Extra residual / total energy |
|:---|:---|---:|---:|---:|
| 4096 x 4096, ordinary layer | exact | 1,892.940 ms | 1.00x | 0 |
| 4096 x 4096, ordinary layer | auto | 191.962 ms | 9.86x | measurement noise |
| 4096 x 4096, ordinary layer | lowrank | 32.670 ms | 57.94x | `2.366e-5` |
| 4096 x 12288, ordinary layer | exact | 1,784.310 ms | 1.00x | 0 |
| 4096 x 12288, ordinary layer | auto | 219.606 ms | 8.13x | measurement noise |
| 4096 x 12288, ordinary layer | lowrank | 50.327 ms | 35.45x | `3.676e-4` |
| 4096 x 12288, rank-deficient layer | exact | 4,097.970 ms | 1.00x | 0 |
| 4096 x 12288, rank-deficient layer | auto | 4,311.970 ms | 0.95x | 0 after exact fallback |
| 4096 x 12288, rank-deficient layer | lowrank | 49.711 ms | 82.44x | `1.085e-4` |

The rank-deficient case is intentionally important: CUDA `gesvda` reported a convergence failure, `auto` caught it,
and exact SVD produced the adapter. That layer pays the failed-attempt overhead, but generation remains correct and the
common layers retain their speedup.

Reproduction helper:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  /root/vm314t/bin/python scripts/benchmark_eora_generation_math.py \
  /path/to/delta_scale_4096x4096.pt /path/to/delta_scale_4096x12288.pt \
  --rank 128 --algos exact,auto,lowrank --warmup 1 --iters 3
```

The source tensors captured during development were stored under `/tmp` and are not repository artifacts.

## Preliminary full generation benchmark

The end-to-end generation workload used the Qwen3-8B GPTQ 4-bit, group-size-128 model, 512 calibration rows / 181,796
tokens, batch size 1, rank 128, and produced 252 adapters.

| Route | Generation | Total run | Generation speedup | Time saved |
|:---|---:|---:|---:|---:|
| exact baseline | 931.871 s | 961.392 s | 1.00x | - |
| auto production candidate | 601.022 s | 638.435 s | 1.55x | 330.850 s |
| lowrank experiment | 558.587 s | 587.800 s | 1.67x | 373.284 s |

All 504 saved factor tensors from both candidates were finite. These exploratory runs predated the explicit config API;
the matched rank-128 and rank-256 recreations below are the policy-setting measurements.

## Preliminary quality checks

The exact baseline and `auto` candidate produced the same continuation and top-1 final token (`12095`) in the smoke
generation check.

| Check | Exact baseline | Auto candidate |
|:---|---:|---:|
| Final logits cosine vs dense | 0.9963679 | 0.9962124 |
| Final logits MAE vs dense | 0.2918788 | 0.2983884 |
| Final logits RMSE vs dense | 0.3752531 | 0.3845199 |
| Layer-2 hidden-state cosine vs dense | 0.9992439 | 0.9992503 |
| Layer-2 hidden-state RMSE vs dense | 0.0470922 | 0.0472392 |

A bounded 64-row GSM8K Platinum comparison scored 11/64 for the exact adapter and 15/64 for `auto`. This is a
regression spot check, not evidence that `auto` improves accuracy: the sample is small and used chat-template settings
that differ from the established full evaluation. The result showed no task-level regression requiring the fast path
to be disabled.

The `lowrank` smoke result was also close to exact, but selected per-layer adapter-product comparisons showed a broader
error distribution than `auto`. The complete matched evaluation below was used for the default decision.

## Controlled rank and no-EoRA evaluation

The controlled experiment recreates all adapters from the same saved Qwen3-8B snapshot and exact first 512 calibration
rows. Rank 128 ran exact, auto, and lowrank concurrently on physical GPUs 2, 3, and 4. Rank 256 repeated those lanes on
the same GPUs while GPU 5 evaluated the quantized snapshot without an adapter. Rank 64 used GPUs 2, 3, and 4 while rank
32 exact used GPU 5; rank-32 auto and lowrank then used GPUs 3 and 4 after their rank-64 lanes exited. The baseline
runtime assertion found 252 quantized linear modules, zero active adapters, no adapter ranks, and finite logits.

The shared contract is GPTQ 4-bit, group size 128, symmetric weights, `desc_act=false`, activation scale search, GAR,
512 rows / 181,796 calibration tokens, batch size 1, and 252 adapted linear modules when EoRA is enabled. Evalution uses
the full 1,172-example ARC Challenge and 1,209-example GSM8K Platinum datasets, batch size 16, no chat template,
deterministic generation, maximum 256 new tokens for GSM8K, and no row cap. ARC used `GPTQ_TORCH`; GSM8K used `MARLIN`.

Effective bits per parameter divides the actual model-plus-adapter safetensor bytes by the 8,190,735,360 dense base
parameters, so it includes packed metadata tensors, unquantized model tensors, and BF16 EoRA factors.

Full saved tensor sizes are decimal MB (`bytes / 1,000,000`). Algorithms at the same rank have identical storage:

| Quantized snapshot | Base | Rank 32 | Rank 64 | Rank 128 | Rank 256 |
|:---|---:|---:|---:|---:|---:|
| 4-bit / group 128 | 6,103.9 MB | 6,278.6 MB | 6,453.2 MB | 6,802.3 MB | 7,500.7 MB |
| 3-bit / group 128 | 5,228.9 MB | 5,403.6 MB | 5,578.2 MB | 5,927.3 MB | 6,625.7 MB |
| 3-bit / group 64 | 5,357.8 MB | 5,532.4 MB | 5,707.0 MB | 6,056.2 MB | 6,754.6 MB |
| 2-bit / group 32 | 4,720.2 MB | 4,894.8 MB | 5,069.4 MB | 5,418.6 MB | 6,117.0 MB |

| Rank | Variant | Adapter weights | Effective bits/parameter | LoRA generation | Speedup vs exact | ARC raw | ARC normalized | GSM8K Platinum |
|:---:|:---|---:|---:|---:|---:|---:|---:|---:|
| - | Base quant, no EoRA | - | 5.961777 | - | - | 625/1,172 (53.33%) | 639/1,172 (54.52%) | 1,069/1,209 (88.42%) |
| 32 | `exact` | 166.6 MiB | 6.132366 | 884.4 s | 1.000x | 642/1,172 (54.78%) | 646/1,172 (55.12%) | 1,088/1,209 (89.99%) |
| 32 | `auto` | 166.6 MiB | 6.132366 | 563.6 s | 1.569x | 635/1,172 (54.18%) | 647/1,172 (55.20%) | 1,091/1,209 (90.24%) |
| 32 | `lowrank` | 166.6 MiB | 6.132366 | 533.0 s | 1.659x | 643/1,172 (54.86%) | 646/1,172 (55.12%) | 1,092/1,209 (90.32%) |
| 64 | `exact` | 333.1 MiB | 6.302889 | 863.7 s | 1.000x | 642/1,172 (54.78%) | 653/1,172 (55.72%) | 1,093/1,209 (90.41%) |
| 64 | `auto` | 333.1 MiB | 6.302889 | 562.3 s | 1.536x | 637/1,172 (54.35%) | 650/1,172 (55.46%) | 1,089/1,209 (90.07%) |
| 64 | `lowrank` | 333.1 MiB | 6.302889 | 518.4 s | 1.666x | 639/1,172 (54.52%) | 651/1,172 (55.55%) | 1,100/1,209 (90.98%) |
| 128 | `exact` | 666.1 MiB | 6.643934 | 872.1 s | 1.000x | 649/1,172 (55.38%) | 651/1,172 (55.55%) | 1,091/1,209 (90.24%) |
| 128 | `auto` | 666.1 MiB | 6.643934 | 590.5 s | 1.477x | 647/1,172 (55.20%) | 653/1,172 (55.72%) | 1,092/1,209 (90.32%) |
| 128 | `lowrank` | 666.1 MiB | 6.643934 | 587.3 s | 1.485x | 650/1,172 (55.46%) | 653/1,172 (55.72%) | 1,099/1,209 (90.90%) |
| 256 | `exact` | 1,332.1 MiB | 7.326024 | 872.0 s | 1.000x | 640/1,172 (54.61%) | 645/1,172 (55.03%) | 1,099/1,209 (90.90%) |
| 256 | `auto` | 1,332.1 MiB | 7.326024 | 602.7 s | 1.447x | 642/1,172 (54.78%) | 652/1,172 (55.63%) | 1,103/1,209 (91.23%) |
| 256 | `lowrank` | 1,332.1 MiB | 7.326024 | 532.1 s | 1.639x | 643/1,172 (54.86%) | 653/1,172 (55.72%) | 1,099/1,209 (90.90%) |

Rank 32 retains useful accuracy at one quarter of rank-128 adapter storage. Its `lowrank` result beats the base by 18
raw ARC, 7 normalized ARC, and 23 GSM8K answers; relative to rank-128 `lowrank`, it is lower by exactly 7 answers on
each metric. Within rank 32, `lowrank` leads raw ARC and GSM8K, while `auto` leads normalized ARC by one answer.

At half the rank-128 storage, rank-64 `lowrank` is lower by 11 raw ARC and 2 normalized ARC answers but higher by one
GSM8K answer. Rank-64 `exact` leads both ARC metrics, while `lowrank` leads GSM8K and generates 345.3 seconds faster
than exact. The score variation is therefore task-specific rather than monotonic in rank.

Across all four ranks, `lowrank` leads or ties eight of twelve within-rank task metrics and is the fastest generation
route at every rank, so it remains the default for new configurations. The explicit `auto` route remains useful when
its cuSOLVER behavior, rank-32 normalized ARC, or rank-256 GSM result is preferred; `exact` remains the reference and
legacy-descriptor behavior.

All twelve adapters contained 504 finite tensors at the requested rank, runtime checks found 252 active adapters and
finite logits, and every full GSM8K run reported zero invalid answers. The no-EoRA baseline is rank-independent and
provides the shared comparison row. Machine-readable results and complete reproduction contracts are in
`tests/benchmark/eora_svd_algorithms_qwen3_8b_a100.json`,
`tests/benchmark/eora_svd_algorithms_qwen3_8b_rank32_rank64_a100.json`, and
`tests/benchmark/eora_svd_algorithms_qwen3_8b_rank256_a100.json`.

## 3-bit regression isolation

The first matched 3-bit run was stopped before extending the rank/algorithm matrix because its adapter-free base emitted
invalid GSM8K answers and repeated `SOLD` for a simple raw completion prompt. The diagnosis kept the requested GPTQ
contract fixed: Qwen3-8B, 3 bits, group size 128, symmetric weights, `desc_act=false`, activation scale search, GAR,
the first 512 calibration rows / 181,796 tokens, and eager Torch loading unless a TriLin comparison is named.

| Saved base | EoRA during quantization | Mean sampled weight cosine vs dense | Last-logit cosine vs dense | Top-1 token | Raw continuation | Bounded GSM8K Platinum |
|:---|:---:|---:|---:|---:|:---|---:|
| Failed joint snapshot | yes | 0.696133 | 0.357972 | 83151 | `SOLD` repeated 24 times | initial run: 0%, 64/64 invalid |
| Fresh native snapshot | no | 0.965302 | 0.984779 | 12095 | coherent Paris/Berlin/Rome/Spain | 57/64 (89.06%), 0 invalid |
| Fresh native snapshot after saturation change | no | 0.965302 | 0.984779 | 12095 | coherent Paris/Berlin/Rome/Spain | byte-identical to the prior native snapshot |
| Recreated joint snapshot after saturation | yes | 0.974279 | 0.969950 base / 0.955333 with adapter | 12095 | coherent Paris/Rome/Madrid/Portugal | 58/64 (90.63%), 0 invalid |

The two native model shards are byte-for-byte identical before and after saturation, so the change does not perturb a
normal GPTQ result whose reconstructed codes are already representable. The failed joint snapshot instead retained the
same saved scales and logical zero-points as the healthy native run but had a mean sampled packed-weight cosine of only
0.696133. Direct bounded rounding with those same scales remained healthy at 0.974280, isolating corruption to
serialization. After the fix, packed reconstruction reaches 0.974279 and agrees with the bounded direct codes for
99.824% of sampled weights; the remaining ordinary GPTQ error-feedback differences no longer include endpoint wrap.

### Inference-kernel exclusion

The native and adapter paths were then exercised on the healthy base. `dtype=float16` is important here: BF16 scales
select the portable Triton kernel, while FP16 scales activate the native TriLin implementation.

| Comparison | Native modules | Observed native calls | Prefill cosine | Decode-score cosine | Generated IDs equal |
|:---|---:|---:|---:|---:|:---:|
| FP16 native TriLin vs eager Torch, no adapter | 252/252 | 6,300 base | 0.999998 | 1.000169 | yes |
| FP16 native TriLin vs eager Torch, rank-128 adapter | 252/252 | 6,300 base in unfused run | 0.999973 | 1.000165 | yes |
| FP16 fused TriLin+EoRA vs unfused TriLin+EoRA | 252/252 | 1,656 fused + 4,644 base | identical at reported precision | 1.000165 | yes |

All three produced the coherent Paris/Berlin/Rome/Spain continuation. The adapter generated by the failed joint run is
therefore usable on the healthy native base, and the fused EoRA kernel agrees with its unfused reference. This excludes
the tokenizer, prompt rendering, native TriLin base kernel, and fused TriLin+EoRA kernel from the serialization failure.

### Packer cause and 2-bit control

The 3-bit `pack_block` route intentionally uses its Python implementation even when the native CPU extension is
available. That implementation, `pack_gpu`, and `pack_original` rounded reconstructed values and immediately shifted
them into packed words. A value of `-1` or `2**bits` therefore wrapped to the opposite endpoint instead of saturating.
The native CPU extension already clamps its reconstructed codes, which made behavior inconsistent across packers.

All three Python/GPU/original routes now round, clamp in floating point to `[0, 2**bits - 1]`, and only then perform the
first integer conversion. The native extension also clamps before its integer cast, which safely handles very large
finite excursions. The regression deliberately feeds codes from -8 through 23 plus `-1e20` and `1e20`, then verifies
exact dequantized values and bitwise-equal packed words for 2, 3, 4, and 8 bits, including the forced Python fallback
and CUDA GPU packer. The full pack suite passed 18 tests plus four saturation subtests on physical GPU 1; the CPU-only
saturation run also passed all four bit-width subtests.

An existing Qwen3-1.7B 2-bit/group-64 control remained coherent, selected the same top-1 token as dense, and measured
0.919323 mean sampled packed-weight cosine versus 0.922526 for direct rounding with saved scales. Across the sampled
projections it had only one endpoint-sized code delta, so the catastrophic 3-bit signature was not present. The
general clamp still covers alternate 2-bit Python/GPU/original paths defensively and matches the already-saturating
native CPU implementation.

### Fixed CPU/GPU recreation and score

Both fixed lanes used the same dense model, calibration rows, quantization contract, default `lowrank` EoRA generation,
and rank 128. Only `pack_impl` and the physical GPU assignment differed. Their two model shards and the complete
698,420,736-byte adapter had identical SHA-256 hashes.

| Variant | Physical GPU | Pack finalization, 252 modules | Total quantization | Pack speedup | Sampled weight cosine | Code agreement | GSM8K Platinum |
|:---|---:|---:|---:|---:|---:|---:|---:|
| CPU/Python `pack_block` | 0 | 257.603 s | 1,687.368 s | 1.00x | byte-identical to GPU | byte-identical to GPU | shared artifact result |
| CUDA `pack_gpu` | 2 | 46.693 s | 1,431.860 s | 5.52x | 0.974279 | 99.824% | 58/64 (90.63%), 0 invalid |

GPU packing saved 210.910 s in the packing stage and 255.508 s end to end, for a 1.178x total quantization speedup.
The fixed adapter score is one answer above the clean native no-EoRA baseline on this bounded sample. This small sample
validates recovery from the serialization failure; it is not a claim that rank-128 EoRA universally improves 3-bit
quality.

Machine-readable contracts, hashes, timing, numerical diagnostics, TriLin isolation, the 2-bit control, and Evalution
results are stored in `tests/benchmark/eora_gptq_3bit_packing_regression_a100.json`.

## Controlled 3-bit rank sweep after packing repair

The controlled sweep generated every adapter afresh against the repaired GPU-packed snapshot. It retained the same
Qwen3-8B dense model, first 512 calibration rows / 181,796 tokens, GPTQ group size 128, symmetric weights,
`desc_act=false`, activation scale search, GAR, batch size 1, and 252 adapted projections used by the 4-bit study.
Evalution again used all 1,172 ARC Challenge and 1,209 GSM8K Platinum examples, batch size 16, no chat template,
deterministic generation, and at most 256 new GSM tokens. ARC used `GPTQ_TORCH`; 3-bit GSM used `GPTQ_TRITON` because
the controlled 3-bit model is not supported by Marlin. The 4-bit GSM reference above used Marlin, so the cross-bit
quality comparison also includes that backend difference; generation timings do not involve either inference backend.

| Rank | Variant | Adapter weights | Effective bits/parameter | LoRA generation | Speedup vs exact | ARC raw | ARC normalized | GSM8K Platinum | Invalid |
|:---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| - | Base quant, no EoRA | - | 5.107149 | - | - | 586/1,172 (50.00%) | 596/1,172 (50.85%) | 1,014/1,209 (83.87%) | 1 |
| 32 | `exact` | 166.6 MiB | 5.277738 | 879.5 s | 1.000x | 599/1,172 (51.11%) | 611/1,172 (52.13%) | 1,051/1,209 (86.93%) | 0 |
| 32 | `auto` | 166.6 MiB | 5.277738 | 593.0 s | 1.483x | 595/1,172 (50.77%) | 609/1,172 (51.96%) | 1,042/1,209 (86.19%) | 0 |
| 32 | `lowrank` | 166.6 MiB | 5.277738 | 548.7 s | 1.603x | 595/1,172 (50.77%) | 606/1,172 (51.71%) | 1,043/1,209 (86.27%) | 0 |
| 64 | `exact` | 333.1 MiB | 5.448260 | 888.6 s | 1.000x | 598/1,172 (51.02%) | 603/1,172 (51.45%) | 1,050/1,209 (86.85%) | 0 |
| 64 | `auto` | 333.1 MiB | 5.448260 | 603.2 s | 1.473x | 596/1,172 (50.85%) | 606/1,172 (51.71%) | 1,048/1,209 (86.68%) | 0 |
| 64 | `lowrank` | 333.1 MiB | 5.448260 | 555.1 s | 1.601x | 598/1,172 (51.02%) | 604/1,172 (51.54%) | 1,050/1,209 (86.85%) | 0 |
| 128 | `exact` | 666.1 MiB | 5.789306 | 874.0 s | 1.000x | 606/1,172 (51.71%) | 606/1,172 (51.71%) | 1,066/1,209 (88.17%) | 0 |
| 128 | `auto` | 666.1 MiB | 5.789306 | 584.8 s | 1.494x | 607/1,172 (51.79%) | 606/1,172 (51.71%) | 1,064/1,209 (88.01%) | 0 |
| 128 | `lowrank` | 666.1 MiB | 5.789306 | 545.2 s | 1.603x | 610/1,172 (52.05%) | 612/1,172 (52.22%) | 1,064/1,209 (88.01%) | 0 |
| 256 | `exact` | 1,332.1 MiB | 6.471395 | 874.6 s | 1.000x | 622/1,172 (53.07%) | 614/1,172 (52.39%) | 1,073/1,209 (88.75%) | 0 |
| 256 | `auto` | 1,332.1 MiB | 6.471395 | 586.0 s | 1.493x | 629/1,172 (53.67%) | 614/1,172 (52.39%) | 1,068/1,209 (88.34%) | 0 |
| 256 | `lowrank` | 1,332.1 MiB | 6.471395 | 554.9 s | 1.576x | 623/1,172 (53.16%) | 611/1,172 (52.13%) | 1,071/1,209 (88.59%) | 0 |

The direct answer-count difference from the previously collected 4-bit table is:

| Rank | Variant | 3-bit minus 4-bit ARC raw | ARC normalized | GSM8K Platinum |
|:---:|:---|---:|---:|---:|
| - | Base quant, no EoRA | -39 | -43 | -55 |
| 32 | `exact` | -43 | -35 | -37 |
| 32 | `auto` | -40 | -38 | -49 |
| 32 | `lowrank` | -48 | -40 | -49 |
| 64 | `exact` | -44 | -50 | -43 |
| 64 | `auto` | -41 | -44 | -41 |
| 64 | `lowrank` | -41 | -47 | -50 |
| 128 | `exact` | -43 | -45 | -25 |
| 128 | `auto` | -40 | -47 | -28 |
| 128 | `lowrank` | -40 | -41 | -35 |
| 256 | `exact` | -18 | -31 | -26 |
| 256 | `auto` | -13 | -38 | -35 |
| 256 | `lowrank` | -20 | -42 | -28 |

EoRA materially improves the repaired 3-bit base at every tested rank. Rank-32 exact gains 13 raw ARC, 15 normalized
ARC, and 37 GSM answers over the base while using one quarter of rank-128 storage. Rank-128 `lowrank` gains 24 raw
ARC and 16 normalized ARC answers, while rank-128 exact gains 52 GSM answers. Rank 256 is strongest overall: `auto`
gains 43 raw ARC answers, exact and auto gain 18 normalized ARC answers, and exact gains 59 GSM answers over the base.
The higher ranks also narrow the 3-bit-to-4-bit gap, especially raw ARC, where rank-256 auto is only 13 answers behind
its 4-bit counterpart.

Quality leadership in the 3-bit sweep is task-specific. Exact leads rank-32 ARC/GSM, exact and lowrank tie rank-64 raw
ARC and GSM, rank-128 lowrank leads both ARC metrics while exact leads GSM by two answers, and rank-256 auto leads raw
ARC while exact ties normalized ARC and leads GSM. `lowrank` nevertheless remains the generation-speed winner at every
rank, saving 319.7 to 333.5 seconds versus exact and staying close to each within-rank quality leader. Combined with
the earlier 4-bit quality result, the sweep does not justify changing the configured `lowrank` default; explicit
`exact` and `auto` remain important when a particular 3-bit task metric is preferred.

All 12 adapters contained 504 finite tensors at the requested rank, loaded on all 252 quantized projections, and
produced zero invalid GSM outputs. The base's one invalid output was a coherent uncertainty response without a numeric
answer, not repetitive gibberish. Its GSM run also emitted 44 recoverable expandable-segment mapping warnings during
transient long-sequence peaks but completed all examples without a CUDA exception. The harness now counts Evalution's
official non-empty `[invalid]` sentinel and repairs stale task metadata when reusing a completed result.

The complete reproduction contract, physical GPU assignments, precise timings, raw scores, invalid counts, and
3-bit-minus-4-bit deltas are stored in
`tests/benchmark/eora_svd_algorithms_qwen3_8b_3bit_a100.json`.

## Controlled group-size sweeps (live)

The controlled matrix holds the dense model, first 512 calibration rows / 181,796 tokens, symmetric GPTQ,
`desc_act=false`, activation scale search, GAR, EoRA generation settings, and full Evalution contract fixed while
testing 3-bit/group-64 and 2-bit/group-32 quantization. Both base snapshots are recreated with the repaired saturating
GPU packer. Physical GPU assignments are restricted to indices 0 through 5. The 2-bit sweep was stopped after its
first ARC results exposed catastrophic quality loss; the 3-bit sweep continues on all six permitted GPUs.

Two storage rates are reported because they answer different questions:

```text
quantized-linear BPW =
    8 * (qweight bytes + qzeros bytes + scale bytes + g_idx bytes)
      / 6,945,767,424 quantized linear parameters

whole-model effective bpp =
    8 * (quantized model safetensor bytes + EoRA safetensor bytes) / 8,190,735,360
```

Quantized-linear BPW is the value comparable to the nominal 3-bit or 2-bit setting. Whole-model effective bpp uses
the actual saved tensor payload and includes packed weights, scales, zero-points, group indices, unquantized model
tensors, and BF16 EoRA factors. It excludes tokenizer, configuration, manifest, and Evalution JSON overhead. The base
row uses zero EoRA bytes.

The whole-model result is substantially higher than the nominal width because 1,244,967,936 of 8,190,735,360
parameters (15.20%) remain BF16. The untied 151,936 x 4,096 token embedding and LM head account for nearly all of
those weights and contribute 2.431351 whole-model bpp; all normalization weights add only 0.000602 bpp. This is not
alias double-counting: `config.json` has `tie_word_embeddings=false`, and
`model.embed_tokens.weight` and `lm_head.weight` occupy two distinct 1,244,659,712-byte ranges in the saved shard.
The measured base-snapshot decomposition is:

| Stored component | 3-bit / g64 | 2-bit / g32 |
|:---|---:|---:|
| Model safetensor bytes | 5,357,789,768 | 4,720,190,648 |
| Packed `qweight` contribution | 2.544009 bpp | 1.696006 bpp |
| `qzeros` contribution | 0.039750 bpp | 0.053000 bpp |
| Scales contribution | 0.212001 bpp | 0.424001 bpp |
| Group indices contribution | 0.005185 bpp | 0.005185 bpp |
| Unquantized BF16 contribution | 2.431953 bpp | 2.431953 bpp |
| Safetensors header contribution | 0.000127 bpp | 0.000127 bpp |
| **Quantized-linear BPW** | **3.302989** | **2.568614** |
| **Whole-model effective bpp** | **5.233024** | **4.610273** |

The identities were checked directly:

```text
3-bit/g64: 8 * 5,357,789,768 / 8,190,735,360 = 5.233024418
2-bit/g32: 8 * 4,720,190,648 / 8,190,735,360 = 4.610272891
```

### 3-bit, group size 64

| Rank | Variant | EoRA storage | Generation | Speedup vs exact | Whole-model effective bpp | ARC raw | ARC normalized | GSM8K Platinum | Invalid | Status |
|:---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| - | Base quant, no EoRA | - | - | - | 5.233024 | 574/1,172 (48.98%) | 596/1,172 (50.85%) | 1,028/1,209 (85.03%) | 0 | complete |
| 32 | `exact` | 166.6 MiB | 885.0 s | 1.000x | 5.403613 | 591/1,172 (50.43%) | 601/1,172 (51.28%) | 1,040/1,209 (86.02%) | 0 | complete |
| 32 | `auto` | 166.6 MiB | 599.6 s | 1.476x | 5.403613 | 588/1,172 (50.17%) | 596/1,172 (50.85%) | 1,040/1,209 (86.02%) | 0 | complete |
| 32 | `lowrank` | 166.6 MiB | 552.6 s | 1.602x | 5.403613 | 590/1,172 (50.34%) | 593/1,172 (50.60%) | 1,043/1,209 (86.27%) | 0 | complete |
| 64 | `exact` | 333.1 MiB | 872.5 s | 1.000x | 5.574136 | 601/1,172 (51.28%) | 611/1,172 (52.13%) | 1,048/1,209 (86.68%) | 0 | complete |
| 64 | `auto` | 333.1 MiB | 600.0 s | 1.454x | 5.574136 | 598/1,172 (51.02%) | 611/1,172 (52.13%) | 1,046/1,209 (86.52%) | 0 | complete |
| 64 | `lowrank` | 333.1 MiB | 538.1 s | 1.622x | 5.574136 | 604/1,172 (51.54%) | 615/1,172 (52.47%) | 1,055/1,209 (87.26%) | 0 | complete |
| 128 | `exact` | 666.1 MiB | 881.0 s | 1.000x | 5.915181 | 598/1,172 (51.02%) | 605/1,172 (51.62%) | 1,044/1,209 (86.35%) | 0 | complete |
| 128 | `auto` | 666.1 MiB | 583.9 s | 1.509x | 5.915181 | 594/1,172 (50.68%) | 604/1,172 (51.54%) | 1,045/1,209 (86.44%) | 0 | complete |
| 128 | `lowrank` | 666.1 MiB | 524.1 s | 1.681x | 5.915181 | 595/1,172 (50.77%) | 602/1,172 (51.37%) | 1,050/1,209 (86.85%) | 0 | complete |
| 256 | `exact` | 1,332.1 MiB | 1,133.7 s | 1.000x | 6.597271 | 610/1,172 (52.05%) | 622/1,172 (53.07%) | 1,073/1,209 (88.75%) | 0 | complete |
| 256 | `auto` | 1,332.1 MiB | 568.1 s | 1.996x | 6.597271 | 610/1,172 (52.05%) | 622/1,172 (53.07%) | 1,076/1,209 (89.00%) | 0 | complete |
| 256 | `lowrank` | 1,332.1 MiB | 560.8 s | 2.022x | 6.597271 | 611/1,172 (52.13%) | 620/1,172 (52.90%) | 1,072/1,209 (88.67%) | 0 | complete |

All twelve group-64 adapters and the base evaluation are complete with zero invalid GSM outputs. At rank 256,
`lowrank` leads raw ARC by one answer and is fastest at 2.022x exact speed; exact and auto tie for normalized ARC,
while auto leads GSM8K Platinum by three answers over exact and four over lowrank.

### 2-bit, group size 32

| Rank | Variant | EoRA storage | Generation | Speedup vs exact | Whole-model effective bpp | ARC raw | ARC normalized | GSM8K Platinum | Invalid | Status |
|:---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| - | Base quant, no EoRA | - | - | - | 4.610273 | 274/1,172 (23.38%) | 312/1,172 (26.62%) | not completed | not recorded | stopped after ARC |
| 32 | `exact` | 166.6 MiB | not completed | - | 4.780862 | pending | pending | not run | - | stopped during generation |
| 32 | `auto` | 166.6 MiB | not run | - | 4.780862 | pending | pending | not run | - | stopped |
| 32 | `lowrank` | 166.6 MiB | 681.7 s | pending | 4.780862 | 405/1,172 (34.56%) | 433/1,172 (36.95%) | not completed | not recorded | stopped after ARC |
| 64 | `exact` | 333.1 MiB | not run | - | 4.951384 | pending | pending | not run | - | stopped |
| 64 | `auto` | 333.1 MiB | not run | - | 4.951384 | pending | pending | not run | - | stopped |
| 64 | `lowrank` | 333.1 MiB | not run | - | 4.951384 | pending | pending | not run | - | stopped |
| 128 | `exact` | 666.1 MiB | not run | - | 5.292430 | pending | pending | not run | - | stopped |
| 128 | `auto` | 666.1 MiB | not run | - | 5.292430 | pending | pending | not run | - | stopped |
| 128 | `lowrank` | 666.1 MiB | not run | - | 5.292430 | pending | pending | not run | - | stopped |
| 256 | `exact` | 1,332.1 MiB | not run | - | 5.974519 | pending | pending | not run | - | stopped |
| 256 | `auto` | 1,332.1 MiB | not run | - | 5.974519 | pending | pending | not run | - | stopped |
| 256 | `lowrank` | 1,332.1 MiB | not run | - | 5.974519 | pending | pending | not run | - | stopped |

### 2-bit stop and initial isolation

The 2-bit base ARC result is close to four-choice chance accuracy, and its interrupted GSM8K run produced only
invalid answers in the observed prefix. Rank-32 lowrank EoRA recovered 131 raw and 121 normalized ARC answers, but
its deterministic continuation remained repetitive. All active and queued 2-bit work was therefore stopped before
spending the remaining sweep budget.

The first boundary checks localize the largest known error before packing and inference:

| Evidence | 4-bit / g128 | 3-bit / g64 | 2-bit / g32 |
|:---|---:|---:|---:|
| Mean logged quantizer loss, 252 projections | 0.000254 | 0.000772 | 0.002947 |
| Layer 6 `mlp.down_proj` quantizer loss | 0.004224 | 0.013688 | **0.226526** |
| Layer 6 `mlp.down_proj` packed-weight cosine vs BF16 | 0.992157 | 0.978445 | 0.920644 |
| Layer 6 direct-round cosine using saved scales vs BF16 | 0.992158 | 0.978447 | 0.920646 |
| Layer 6 eager dequant vs independent manual unpack MAE | 0.000013 | 0.000007 | **0.000000** |

The quantizer loss is computed from the reconstructed weight and Hessian before packing, so neither the GPU packer
nor Triton can cause the 0.226526 spike. On the saved 2-bit checkpoint, all sampled logical zero-points are exactly
2, codes stay in `[0, 3]`, scales and weights are finite, and eager Torch dequantization is exactly equal to an
independent unpack. The sampled packed weights also match direct reconstruction from the saved scales to within
0.000002 cosine, unlike the endpoint wrapping signature from the repaired 3-bit packing bug. Focused synthetic
saturation coverage passes for 2-, 3-, 4-, and 8-bit CPU packers.

Tokenizer normalization is also excluded: the quantized snapshot and dense model produce identical input IDs for
the plain completion and arithmetic prompts. Remaining work is to compare Torch and Triton logits against the same
dequantized reference and recreate a native 2-bit checkpoint without joint EoRA/GAR to determine whether the
destructive error propagation is intrinsic to this configuration or a 2-bit quantizer defect.

The layer-6 loss is concentrated in `mlp.down_proj` output channel 2276. Its maximum saved scale grows from 0.350 at
4-bit to 2.031 at 3-bit and **35.0** at 2-bit; its relative weight RMSE grows from 0.313 to 0.565 and **0.874**.
For this specific 2-bit channel, GPTQ reconstruction has cosine 0.4880 versus BF16 while independent symmetric RTN
reaches 0.8734. Every layer-6 scale above 10 belongs to channel 2276. This pattern is consistent with sequential GPTQ
error feedback amplifying an outlier channel and motivates a controlled static-group or selective-RTN experiment.

A controlled in-memory selective-RTN experiment replaced only layer-6 `mlp.down_proj` channel 2276 with independent
2-bit/group-32 symmetric RTN codes and scales. It improved that channel's weight cosine to 0.8733, but it did not
repair the end-to-end prompt logits:

| CPU eager check | Cosine | RMSE | Top-1 token |
|:---|---:|---:|---:|
| Original 2-bit vs dense BF16 | 0.375457 | 4.669391 | 83151 |
| Selective-RTN hybrid vs original 2-bit | 0.999997 | 0.009585 | 83151 |
| Selective-RTN hybrid vs dense BF16 | 0.375408 | 4.669370 | 83151 |
| Dense BF16 reference | - | - | 12095 |

Channel 2276 is therefore a strong marker of unstable 2-bit error feedback, but repairing that channel after the
full sequential quantization does not undo the broader accumulated failure. The next quantizer controls must rerun
the sequence with static groups or an in-loop fallback rather than patching one saved channel afterward.

The tracked low-bit harness now exposes `--static-groups` / `--no-static-groups` and
`--quantization-diagnostics {off,auto,channel}`. The planned native-versus-static control therefore holds the dense
model, calibration rows, activation scale search, GAR, packing implementation, and pure-Torch evaluation backend
constant; it changes only GPTQ group construction. Both control snapshots will use `channel` diagnostics so the
pre-pack module loss and per-output-channel scale summaries are persisted alongside the checkpoint.

Reviewing that control uncovered and fixed an existing `static_groups=True` plus GAR mapping defect before the run
started. Static quantizers are precomputed in original group order, but GAR processes full groups in a different
global order. The old path selected quantizers by the reordered position and then permuted the serialized scales,
so an original group could receive another group's parameters. The corrected path reorders only the working
quantizer objects into GAR processing order and leaves saved scales and zero-points in original group order after
the quantized weight is restored. A focused test forces a non-identity two-group swap and verifies the returned
scales, zero-points, and `g_idx` against independently precomputed original-group parameters. This affects only the
previously incorrect static-groups-plus-GAR combination; the default dynamic-group path is unchanged.

| 2-bit/group-32 native control | Physical GPU | Static groups | EoRA | Dequant/eval path | Status |
|:---|---:|:---:|:---:|:---|:---|
| Dynamic groups | 2 | no | disabled | pure Torch | saved in 1,015.5 s; dense-logit cosine 0.968555, top-1 matches; GSM64 22/64 (34.38%), 1 invalid |
| Static groups | 3 | yes | disabled | pure Torch | dense-logit cosine 0.886861, wrong top-1; GSM64 1/64 (1.56%), 11 invalid |
| Dynamic groups, identical repeat | 5 | no | disabled | not needed | saved in 1,005.0 s; both safetensor shards are byte-identical to the first dynamic run |
| Dynamic groups, no GAR | 2 | no | disabled | pure Torch | saved in 1,014.7 s; dense-logit cosine 0.958758, top-1 matches; GSM64 17/64 (26.56%), 0 invalid |
| Dynamic groups, no scale search | 3 | no | disabled | pure Torch | saved in 948.2 s; dense-logit cosine 0.800615, wrong top-1; GSM64 1/64 (1.56%), 2 invalid |

The running native no-EoRA control has already reproduced the exact catastrophic pre-pack signature at layer 6
`mlp.down_proj`: loss 0.226526 and maximum scale 34.880. This excludes EoRA generation and adapter application as
the source of that specific local quantizer-loss spike, but not necessarily as the source of the earlier
end-to-end model regression. The live diagnostics also caught an axis-labeling defect in their own first real use:
pre-pack `q_scales` are `[output_channels, groups]`, while saved QuantLinear scales are
`[groups, output_channels]`. The original live reducer reported group 161 as an output channel; the checkpoint
coordinate is `(group=161, output_channel=2276)`. The reducer now scans all dimensions after axis 0 and a
4096-by-384 regression fixture asserts output channel 2276. The two running controls retain valid loss and maximum
scale values; channel labels, per-channel p99, and max/median ratios from those already running processes will be
recomputed from the saved checkpoints.

The corrected static-groups control bounds the layer-6 scale distribution but does not eliminate its pre-pack loss:
maximum scale falls 35.8x from 34.880 to 0.974, while loss falls only 14.2% from 0.226526 to 0.194291. Dynamic
group-parameter refresh therefore amplifies the scale outlier, but the remaining 2-bit reconstruction failure is
broader than that feedback mechanism. The saved-checkpoint and bounded GSM64 comparison will determine whether the
bounded static scales materially recover end-to-end quality.

The corrected scan of the saved native checkpoint read 252 tensors in 34.884 s. Layer-6 `mlp.down_proj` resolves to
output channel 2276 with maximum scale 34.875, per-output-channel p99 0.219257, max/median 630.3x, and all 15 scale
elements above 10. Layer-16 `mlp.down_proj` also resolves to channel 2276 with maximum scale 12.031. The machine-readable
result is stored as `quantization_regression_scale_scan.json` beside the checkpoint.

The saved native no-EoRA checkpoint is coherent under the deliberately isolated pure-Torch dequantization path.
For the smoke prompt, its last-token logits have cosine 0.968555 versus dense BF16, RMSE 1.251588, the same top-1
token 12095, and two overlapping top-5 tokens. The earlier joint 2-bit snapshot measured cosine 0.375457 and selected
wrong top-1 token 83151. Therefore the layer-6 loss/scale anomaly is real but is not sufficient to explain the
catastrophic joint-model output. Generic 2-bit quantization, packing, and eager dequantization are no longer the
leading explanation; the investigation returns to differences introduced by the joint EoRA quantization path. The
bounded native GSM8K Platinum check completed at 22/64 (34.38%) with one invalid output. This is far below the 3-bit
control but is qualitatively different from the joint snapshot's repetitive corruption: 63/64 native outputs remain
parseable.

The identical native no-EoRA repeat is deterministic at the serialized boundary. Its two safetensor shards have the
same sizes and exact SHA-256 hashes as the first coherent native run:

| Safetensor shard | SHA-256 |
|:---|:---|
| `model-00001-of-00002.safetensors` | `700ddfe8d3b7e6fab5ef42ec0b5ae6fafb33a674b8d75fba9a2d208a3819520e` |
| `model-00002-of-00002.safetensors` | `28fdea214c3df9742ac4b25ca653fd4286752fb081db63d9f26e95715f291f98` |

This rules out normal run-to-run packed-code variance as the explanation for the 15.638% layer-0 code mismatch
between the coherent native and joint EoRA snapshots. The zero-correction and full-correction joint repeats therefore
remain the decisive split between EoRA processor lifecycle effects and the numerical `B @ A` correction.

The full joint EoRA repeat is also byte-identical to the original joint run, including both model shards and the
rank-128 adapter:

| Full joint artifact | SHA-256 |
|:---|:---|
| `model-00001-of-00002.safetensors` | `1ffcc97d1415993ea2cad5301a140a2ef45b42ac4350beb6d960abd16e7b0345` |
| `model-00002-of-00002.safetensors` | `850b9905e35725923705338e37ef08551ae93a5983edad63ff86ecc169d0fa5e` |
| `eora-rank128/adapter_model.safetensors` | `2d34606d2a546739a5b87f474c198ad0ded903d3f9b20268fad6d77a883f8261` |

The two deterministic branches therefore start from identical GPTQ losses and metadata but serialize different
packed base codes: native repeatedly produces the coherent checkpoint, while full EoRA repeatedly produces the
corrupted checkpoint.

The zero-correction arm resolves that remaining boundary. Its 504 adapter tensors contain 349,175,808 finite elements
and every element is exactly zero, yet its two base-model shards have the same hashes as both full joint EoRA runs:
`1ffcc97d...0345` and `850b9905...a5e`. They do not match the repeated native hashes. The packed-base divergence is
therefore independent of SVD output and the numerical `B @ A` correction. It is introduced by the shared EoRA
weight-state/restoration lifecycle before packing. The `channel` diagnostics tier now compares up to 4,096 sampled
logical codes per projection immediately after GPTQ, immediately before packing after EoRA finalization, and after
packing. This keeps normal `auto` diagnostics unchanged while distinguishing state restoration from the packer on
the controlled lifecycle runs.

A bounded one-layer rerun then located the first changing boundary. Both arms used the first eight identical
calibration rows, 2-bit/group-32 GPTQ, activation-aware scale search, GAR, the GPU packer, and layer 0 only. The EoRA
arm returned exact zero factors, so it retained the complete second-pass lifecycle without applying a numerical
correction:

| Layer-0 sampled-code lifecycle | Native GPTQ | Zero-correction EoRA |
|:---|---:|---:|
| Projections checked | 7 | 7 |
| Sampled logical codes | 28,672 | 28,672 |
| Post-GPTQ to pre-pack mismatches | 0 (0.0000%) | **4,793 (16.7167%)** |
| Post-GPTQ to packed mismatches | 0 (0.0000%) | **4,793 (16.7167%)** |
| Largest sampled code delta | 0 | 2 |
| Quantization wall time | 22.644 s | 25.375 s |

The zero-correction arm's largest per-module rate is layer-0 `self_attn.o_proj` at 834/4,096 (20.3613%). Its pre-pack
and packed mismatch counts are identical for every sampled projection, proving that the packer faithfully serializes
the already-changed floating-point reconstruction. The extra EoRA pass visibly reloads checkpoint tensors after GPTQ;
the leading mechanism is an aliased saved `wq` tensor whose storage is overwritten when the dense module is
rematerialized. The repair snapshots the reconstructed `wq` into independent storage before the EoRA replay and
requires zero lifecycle mismatches. Machine-readable results are stored in
`artifacts/qwen3_8b_dual_gptq_eora_20260722/2bit_layer0_native_code_lifecycle.json` and
`artifacts/qwen3_8b_dual_gptq_eora_20260722/2bit_layer0_zero_eora_code_lifecycle.json`.

The independent-`wq` control confirms that mechanism and repair. GPTQ now copies its reconstructed BF16 weight to
independent CPU storage before exposing the accelerator tensor through the module parameter. Dense rematerialization
can overwrite the parameter for EoRA's replay without changing the saved reconstruction that is later restored for
packing:

| Zero-correction layer-0 control | Aliased `wq` | Independent CPU `wq` |
|:---|---:|---:|
| Post-GPTQ to pre-pack mismatches | 4,793/28,672 (16.7167%) | **0/28,672 (0.0000%)** |
| Post-GPTQ to packed mismatches | 4,793/28,672 (16.7167%) | **0/28,672 (0.0000%)** |
| Maximum sampled code delta | 2 | 0 |
| Quantization wall time | 25.375 s | 30.254 s |

A focused CPU regression deliberately copies dense weights into the aliased parameter storage and verifies that the
independent snapshot remains bit-exact. The fixed real-model result is stored in
`artifacts/qwen3_8b_dual_gptq_eora_20260722/2bit_layer0_zero_eora_independent_wq.json`.

The full 36-layer production control confirms that the repair holds without the zero-correction harness. It used all
512 calibration rows / 181,796 tokens, activation scale search, GAR, the GPU packer, and a real rank-128 `lowrank`
adapter. All 252 projections retained their post-GPTQ reconstruction through EoRA replay: 0/1,032,192 sampled codes
changed before packing and 0/1,032,192 changed after packing. Both model shards are byte-for-byte identical to the
coherent native checkpoint:

| Full 2-bit/group-32 control | Corrupted joint + matching adapter | Native base | Fixed joint + matching adapter |
|:---|:---:|:---:|:---:|
| Base matches native shards | No; all 252 `qweight` tensors differ | Reference | **Yes; both shards byte-identical** |
| Code evidence | 12.2405% of all logical codes differ from native | Reference | **0/1,032,192 pre-pack and packed mismatches** |
| Dense last-logit cosine | 0.366398 | 0.968555 | **0.971842** |
| Top-1 token agrees with dense | No | Yes | **Yes** |
| GSM8K Platinum, first 64 | 0/64 (0.00%) | 22/64 (34.38%) | **32/64 (50.00%)** |
| Invalid outputs | 64 | 1 | **0** |

The fixed continuation is coherent (`Paris. The capital city of France is Paris...`), all logits are finite, all 252
adapters are active at rank 128, and the model selects the same top-1 token as dense BF16. The persistent layer-6
channel-2276 anomaly remains present, proving it is a real low-bit reconstruction outlier but not the cause of the
old repetitive-output corruption.

| Production-control measurement | Result |
|:---|---:|
| Quantization wall time | 1,478.977 s |
| Save wall time | 5.420 s |
| Model safetensor bytes | 4,720,190,648 (4,720.2 decimal MB) |
| Adapter safetensor bytes | 698,420,736 (698.4 decimal MB) |
| Model + adapter tensor bytes | 5,418,611,384 (5,418.6 decimal MB) |
| Complete artifact bytes | 5,430,163,549 (5,430.2 decimal MB) |

The model-shard SHA-256 values are `700ddfe8...9520e` and `28fdea21...1f98`; the new adapter SHA-256 is
`9eb28f81...c58`. Complete hashes, hardware, software, timing, storage, diagnostics, and quality results are stored in
`artifacts/qwen3_8b_dual_gptq_eora_20260722/2bit_eora_weight_lifecycle_fix_summary.json`. The 2-bit rank sweep stayed
stopped during root-cause validation and was resumed only after this full control passed.

### Repaired 2-bit/group-32 rank sweep

This replacement sweep uses the native-matching fixed snapshot, the same first 512 calibration rows / 181,796 tokens,
and freshly generated ranks 32, 64, 128, and 256 for `exact`, `auto`, and `lowrank`. ARC Challenge and GSM8K Platinum
use the verified `GPTQ_TORCH` path; the unverified 2-bit Triton kernel is deliberately excluded as a quality variable.
All algorithms at a given rank have the same saved tensor size. Full sizes below use decimal MB and include the
4,720,190,648-byte quantized base plus the BF16 adapter tensors:

| Rank | Variant | Full tensor size | Effective bits/parameter | LoRA generation | Speedup vs exact | ARC raw | ARC normalized | GSM8K Platinum | Invalid |
|:---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| - | Fixed base quant, no EoRA | 4,720.2 MB | 4.610273 | - | - | 405/1,172 (34.56%) | 414/1,172 (35.32%) | 495/1,209 (40.94%) | 9 |
| 32 | `exact` | 4,894.8 MB | 4.780862 | 837.0 s | 1.000x | 419/1,172 (35.75%) | 445/1,172 (37.97%) | 589/1,209 (48.72%) | 5 |
| 32 | `auto` | 4,894.8 MB | 4.780862 | 589.7 s | 1.419x | 424/1,172 (36.18%) | 444/1,172 (37.88%) | 585/1,209 (48.39%) | 4 |
| 32 | `lowrank` | 4,894.8 MB | 4.780862 | 518.7 s | 1.614x | 425/1,172 (36.26%) | 445/1,172 (37.97%) | 583/1,209 (48.22%) | 5 |
| 64 | `exact` | 5,069.4 MB | 4.951384 | 851.0 s | 1.000x | 422/1,172 (36.01%) | 452/1,172 (38.57%) | 638/1,209 (52.77%) | 3 |
| 64 | `auto` | 5,069.4 MB | 4.951384 | 561.4 s | 1.516x | 427/1,172 (36.43%) | 457/1,172 (38.99%) | 639/1,209 (52.85%) | 2 |
| 64 | `lowrank` | 5,069.4 MB | 4.951384 | 521.5 s | 1.632x | 427/1,172 (36.43%) | 449/1,172 (38.31%) | 641/1,209 (53.02%) | 2 |
| 128 | `exact` | 5,418.6 MB | 5.292430 | 856.2 s | 1.000x | 440/1,172 (37.54%) | 471/1,172 (40.19%) | 710/1,209 (58.73%) | 1 |
| 128 | `auto` | 5,418.6 MB | 5.292430 | 564.8 s | 1.516x | 434/1,172 (37.03%) | 470/1,172 (40.10%) | 707/1,209 (58.48%) | 2 |
| 128 | `lowrank` | 5,418.6 MB | 5.292430 | 524.1 s | 1.634x | 444/1,172 (37.88%) | 465/1,172 (39.68%) | 701/1,209 (57.98%) | 0 |
| 256 | `exact` | 6,117.0 MB | 5.974519 | 863.8 s | 1.000x | 455/1,172 (38.82%) | 481/1,172 (41.04%) | 784/1,209 (64.85%) | 2 |
| 256 | `auto` | 6,117.0 MB | 5.974519 | 571.4 s | 1.512x | 460/1,172 (39.25%) | 483/1,172 (41.21%) | 787/1,209 (65.10%) | 3 |
| 256 | `lowrank` | 6,117.0 MB | 5.974519 | 541.7 s | 1.594x | 452/1,172 (38.57%) | 478/1,172 (40.78%) | 806/1,209 (66.67%) | 2 |

All twelve repaired adapters were freshly generated after the independent-`wq` lifecycle fix. Rank 256 is the clear
2-bit quality tier: `auto` leads both ARC metrics, while `lowrank` leads GSM8K by 19 answers over `auto`, generates
322.1 seconds faster than exact, and remains the fastest route. Rank-32 EoRA already improves the fixed base by
14-20 ARC answers and 88-94 GSM answers for only 174,656,040 additional tensor bytes. Quality continues to rise with
rank, but even the best 2-bit score remains far below the matched 3-bit and 4-bit results.

The complete reproduction contract, physical GPU assignment, precise timings, raw scores, invalid counts, storage,
and post-fix provenance are stored in
`tests/benchmark/eora_svd_algorithms_qwen3_8b_2bit_a100.json`.

The old stopped table above remains historical evidence from the corrupted pre-fix joint artifact; it is not reused
as the repaired 2-bit baseline.

Disabling GAR reduces the layer-6 `mlp.down_proj` loss from 0.226526 to 0.025585 and eliminates every saved scale
above 10; its largest scale is 7.953 at layer 16, output channel 2276. End-to-end quality does not recover, however:
the last-logit cosine is 0.958758 with the correct dense top-1, while bounded GSM8K Platinum falls to 17/64 (26.56%)
with no invalid outputs. GAR materially amplifies the localized layer-6 outlier, but it is not the primary cause of
weak generic 2-bit accuracy and cannot explain the joint EoRA checkpoint's corrupted packed codes.

Disabling activation scale search is decisively worse. Layer-6 `mlp.down_proj` still has loss 0.206513 and maximum
scale 30.888, while mean loss across all projections rises from 0.002947 to 0.005355. Its last-logit cosine falls to
0.800615 with the wrong top-1 token, and bounded GSM8K Platinum is 1/64 (1.56%) with two invalid outputs. The
activation-aware scale search is therefore beneficial in this 2-bit regime and is ruled out as the regression
source or a viable repair.

Applying the original rank-128 adapter produced by that same joint quantization does not recover the joint snapshot.
The adapter is present on all 252 projections, but the smoke prompt repeats `SOLD`, last-token cosine versus dense
falls to 0.366398 with RMSE 4.677662 and wrong top-1 token 83151, and bounded GSM8K Platinum is 0/64 with all 64
outputs invalid. The failure is therefore not an artifact of evaluating a coupled base without its matching adapter.

A tensor-by-tensor comparison between the joint and coherent native checkpoints finds byte-identical scales,
zero-points, group indices, and unquantized BF16 tensors. All 252 packed `qweight` tensors differ. For layer-0
`self_attn.q_proj`, 15.638% of logical 2-bit codes differ, mostly by one level. Repacking the coherent native weight
after adding the saved rank-128 `B @ A` correction changes only 0.293% of codes and does not reproduce the joint
codes, so the difference is not simply the adapter correction being packed into the base. The logged GPTQ losses
and damping values are identical; only timing differs. Byte-identical native and full-EoRA repeats exclude ordinary
run-to-run variance in both branches. The harness-only zero-correction control retains the complete joint EoRA
processor lifecycle while returning shape-correct zero `A` and `B` factors; its base still matches the corrupted
full-correction branch exactly. This isolates shared state/finalization from the numerical correction without
changing production EoRA behavior.

The complete matched comparison across all 252 projections shows that static groups exchange the layer-6 scale
explosion for several reconstruction-loss regressions. The median static/dynamic loss ratio is 1.093x and the
whole-model mean loss rises from 0.002947 to 0.004407 (1.495x):

| Module | Dynamic-group loss | Static-group loss | Static / dynamic |
|:---|---:|---:|---:|
| Layer 1 `mlp.gate_proj` | 0.001723 | 0.056654 | 32.87x |
| Layer 2 `mlp.gate_proj` | 0.003007 | 0.166856 | 55.50x |
| Layer 2 `mlp.up_proj` | 0.002379 | 0.048798 | 20.52x |
| Layer 3 `mlp.gate_proj` | 0.002763 | 0.046845 | 16.95x |
| Layer 6 `mlp.down_proj` | 0.226526 | 0.194291 | 0.858x |

The corresponding early-module scale ranges remain close between the arms. This indicates that freezing original
group ranges prevents them from adapting to weights changed by sequential GPTQ error feedback: it bounds the scale
distribution but moves severe reconstruction error into other modules. Static groups are therefore an informative
control, not a viable general 2-bit repair.

The corrected saved-checkpoint scan confirms that the static run contains no scale above 10. Layer-6
`mlp.down_proj` still resolves to output channel 2276, but its maximum falls from 34.875 to 0.974 and max/median ratio
falls from 630.3x to 17.8x. This narrower scale range does not recover the model: the static checkpoint's smoke
continuation is malformed, last-token cosine versus dense BF16 is 0.886861 with top-1 token 264 instead of 12095,
and bounded GSM8K Platinum is 1/64 (1.56%) with 11 invalid outputs. Static groups are materially worse than the
dynamic control's 22/64 (34.38%) despite suppressing the conspicuous scale outlier.

### Reusable severe-regression diagnostics

The new repository skill
`.agents/skills/gptqmodel-quantization-regressions/` codifies dense-baseline, pre-pack, packing, eager-dequant,
backend-kernel, and end-to-end isolation. Its snapshot analyzer compares matched layer/module losses by both raw
candidate/reference ratio and the ratio normalized by the matched median. The optional `--scan-scales` path reduces
every saved scale tensor by output channel and performs the same matched comparison.

Running it against these 4-, 3-, and 2-bit snapshots reproduced the manually localized failure:

| Comparison | Matched median loss ratio | Layer-6 `down_proj` loss ratio | Normalized loss regression | Matched median scale ratio | Channel-2276 scale ratio | Normalized scale regression |
|:---|---:|---:|---:|---:|---:|---:|
| 3-bit vs 4-bit | 3.275x | 3.241x | 0.990x | 2.004x | 5.802x | 2.896x |
| 2-bit vs 4-bit | 10.808x | **53.629x** | **4.962x** | 4.343x | **99.615x** | **22.938x** |

The default quantization-time `QuantizeConfig.quantization_diagnostics="auto"` summary is intentionally cheap: it
analyzed all 252 real 2-bit log rows in 0.352 ms averaged over 10,000 runs. It flags the layer-6 module because its
loss is 76.86x the all-module mean and owns 30.50% of total logged loss, then explains that this boundary precedes
packing and inference. `off` disables the summary. `channel` additionally records per-output-channel scale maxima,
p99, median ratio, non-finite counts, and threshold counts in the live CLI table and saved
`quantization_diagnostics.json`.

Detailed channel work is gated: the offline scan read 252 tensors per snapshot in 21.211 s at 4-bit, 28.992 s at
3-bit, and 27.845 s at 2-bit (78.048 s total). It is available through either the analyzer's `--scan-scales` option
or `GPTQMODEL_QUANTIZATION_DIAGNOSTICS=channel`; normal quantization remains on the sub-millisecond loss-only path.

The analyzer also has a gated `--scan-codes` path for same-bit snapshots. It compares logical codes one packed module
at a time and verifies exact scale, zero-point, and group-index metadata. The native-versus-joint 2-bit scan processed
all 6,945,767,424 quantized-linear codes in 17.057 s:

| Packed-code metric | Native vs joint EoRA |
|:---|---:|
| Exact `qweight` tensors | 0 / 252 |
| Logical code mismatches | 850,194,241 / 6,945,767,424 (12.2405%) |
| Mean / maximum absolute code delta | 0.122939 / 3 |
| Exact packed words | 92,206,985 / 434,110,464 (21.2404%) |
| Exact scales / zero-points / group indices | 252 / 252 for each category |

The largest mismatch rates are layer-1 `mlp.gate_proj` at 20.9889%, layer-2 `mlp.gate_proj` at 20.0998%, and layer-3
`mlp.gate_proj` at 18.9355%. The machine-readable summary is stored at
`artifacts/qwen3_8b_dual_gptq_eora_20260722/2bit_native_joint_packed_code_summary.json`. This optional full-checkpoint
scan is deliberately separate from normal quantization-time feedback.

The repaired 2-bit sweep is complete; the historical interrupted table is retained only to document the pre-fix
failure that triggered the lifecycle investigation.

## Tests and validation

Focused tests cover:

- bitwise equality between shared and independently computed covariance accumulation;
- rejection of reuse for distinct same-shape activation views;
- deterministic randomized SVD and reconstruction objective;
- `gesvda` failure followed by exact fallback;
- `EoRAConfig` defaults, validation, serialization, and legacy-payload loading;
- exact and mismatched logical-code accounting for 2-, 3-, 4-, and 8-bit packed layouts;
- invalid algorithm handling; and
- the existing Cholesky, eigensolve, merge, shape, dtype, and device contracts.

Final checks:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5 PYTHON_GIL=0 \
  /root/vm314t/bin/python -m pytest -q \
  tests/test_adapter_config.py tests/test_eora_generation_math.py \
  tests/test_eora_cholesky.py tests/test_eora_merge.py

/root/vm314t/bin/python -m ruff check \
  gptqmodel/adapter/adapter.py gptqmodel/eora/eora.py \
  gptqmodel/looper/eora_processor.py scripts/benchmark_eora_generation_math.py \
  scripts/eora_svd_algorithm_eval.py tests/test_adapter_config.py \
  tests/test_eora_generation_math.py tests/test_eora_cholesky.py

jq empty \
  tests/benchmark/eora_svd_algorithms_qwen3_8b_a100.json \
  tests/benchmark/eora_svd_algorithms_qwen3_8b_rank32_rank64_a100.json \
  tests/benchmark/eora_svd_algorithms_qwen3_8b_rank256_a100.json \
  tests/benchmark/eora_svd_algorithms_qwen3_8b_3bit_a100.json \
  tests/benchmark/eora_gptq_3bit_packing_regression_a100.json

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHON_GIL=1 \
  /root/vm314t/bin/python -m pytest -q tests/test_pack.py

CUDA_VISIBLE_DEVICES='' PYTHON_GIL=1 \
  /root/vm314t/bin/python -m pytest -q \
  tests/test_pack.py::TestPackAccuracy::test_pack_clamps_reconstructed_codes

git diff --check
```

Current result: 23 EoRA configuration, math, merge, and reconstructed-weight lifecycle tests passed; 16 regression
analyzer and quantization-diagnostics tests passed. The pack suite passed 18 tests plus four parameterized saturation
subtests on physical GPU 1; the CPU-only saturation check passed all four bit widths. All twelve 4-bit adapter
pipelines, the adapter-free baseline, both fixed 3-bit pack recreations, the bounded fixed-artifact evaluation, all
twelve full 3-bit adapter pipelines, the full 3-bit base evaluation, and the full 2-bit lifecycle repair control
completed. Ruff, the agent-skill validator, JSON validation, and diff checks passed.

## Follow-up opportunities

1. Repeat the controlled evaluation on additional model families and calibration regimes to validate the `lowrank`
   default beyond Qwen3-8B.
2. Evaluate an adaptive low-rank policy that cheaply checks the projected residual and reruns exact SVD when the error
   exceeds a model-level threshold.
3. Investigate whether similarly shaped independent layer decompositions can be batched without increasing peak memory
   or delaying sequential calibration replay.
4. Track which model families produce `gesvda` convergence failures; if they cluster by numerical condition, skip the
   failed attempt using a cheap and conservative predicate.
