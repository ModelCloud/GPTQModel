# EoRA calibration-time LoRA generation: design and optimization log

## Scope and status

This work accelerates the math that generates EoRA LoRA factors from calibration activations. It does not change the
EoRA inference kernel, Marlin dispatch, or the format of saved adapters.

As of 2026-07-22, the implementation combines:

1. exact reuse of calibration covariance contributions for modules that consume the same activation tensor; and
2. an explicit `EoRAConfig` selecting exact, CUDA-accelerated automatic, or randomized low-rank SVD.

`lowrank` is the default for newly constructed configurations. Across matched rank-32, rank-64, rank-128, and rank-256
full-dataset ARC Challenge and GSM8K Platinum comparisons, it led or tied eight of twelve within-rank task metrics and
was the fastest generation route at every tested rank. `exact` remains available as the reference path and is retained
for serialized descriptors that predate `EoRAConfig`; `auto` remains available as the cuSOLVER route with exact
fallback.

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

Optimization and controlled evaluation use physical GPUs 2, 3, 4, and 5 selected with PCI ordering:

| Physical GPU | PCI bus | Device | Compute capability | SMs | Memory |
|---:|:---|:---|:---:|---:|---:|
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

| Rank | Variant | Adapter weights | LoRA generation | Speedup vs exact | ARC raw | ARC normalized | GSM8K Platinum |
|:---:|:---|---:|---:|---:|---:|---:|---:|
| - | Base quant, no EoRA | - | - | - | 625/1,172 (53.33%) | 639/1,172 (54.52%) | 1,069/1,209 (88.42%) |
| 32 | `exact` | 166.6 MiB | 884.4 s | 1.000x | 642/1,172 (54.78%) | 646/1,172 (55.12%) | 1,088/1,209 (89.99%) |
| 32 | `auto` | 166.6 MiB | 563.6 s | 1.569x | 635/1,172 (54.18%) | 647/1,172 (55.20%) | 1,091/1,209 (90.24%) |
| 32 | `lowrank` | 166.6 MiB | 533.0 s | 1.659x | 643/1,172 (54.86%) | 646/1,172 (55.12%) | 1,092/1,209 (90.32%) |
| 64 | `exact` | 333.1 MiB | 863.7 s | 1.000x | 642/1,172 (54.78%) | 653/1,172 (55.72%) | 1,093/1,209 (90.41%) |
| 64 | `auto` | 333.1 MiB | 562.3 s | 1.536x | 637/1,172 (54.35%) | 650/1,172 (55.46%) | 1,089/1,209 (90.07%) |
| 64 | `lowrank` | 333.1 MiB | 518.4 s | 1.666x | 639/1,172 (54.52%) | 651/1,172 (55.55%) | 1,100/1,209 (90.98%) |
| 128 | `exact` | 666.1 MiB | 872.1 s | 1.000x | 649/1,172 (55.38%) | 651/1,172 (55.55%) | 1,091/1,209 (90.24%) |
| 128 | `auto` | 666.1 MiB | 590.5 s | 1.477x | 647/1,172 (55.20%) | 653/1,172 (55.72%) | 1,092/1,209 (90.32%) |
| 128 | `lowrank` | 666.1 MiB | 587.3 s | 1.485x | 650/1,172 (55.46%) | 653/1,172 (55.72%) | 1,099/1,209 (90.90%) |
| 256 | `exact` | 1,332.1 MiB | 872.0 s | 1.000x | 640/1,172 (54.61%) | 645/1,172 (55.03%) | 1,099/1,209 (90.90%) |
| 256 | `auto` | 1,332.1 MiB | 602.7 s | 1.447x | 642/1,172 (54.78%) | 652/1,172 (55.63%) | 1,103/1,209 (91.23%) |
| 256 | `lowrank` | 1,332.1 MiB | 532.1 s | 1.639x | 643/1,172 (54.86%) | 653/1,172 (55.72%) | 1,099/1,209 (90.90%) |

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

## Tests and validation

Focused tests cover:

- bitwise equality between shared and independently computed covariance accumulation;
- rejection of reuse for distinct same-shape activation views;
- deterministic randomized SVD and reconstruction objective;
- `gesvda` failure followed by exact fallback;
- `EoRAConfig` defaults, validation, serialization, and legacy-payload loading;
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
  tests/benchmark/eora_svd_algorithms_qwen3_8b_rank256_a100.json

git diff --check
```

Current result: 22 tests passed; all twelve adapter pipelines and the adapter-free baseline completed; Ruff, JSON, and
diff checks passed.

## Follow-up opportunities

1. Repeat the controlled evaluation on additional model families and calibration regimes to validate the `lowrank`
   default beyond Qwen3-8B.
2. Evaluate an adaptive low-rank policy that cheaply checks the projected residual and reruns exact SVD when the error
   exceeds a model-level threshold.
3. Investigate whether similarly shaped independent layer decompositions can be batched without increasing peak memory
   or delaying sequential calibration replay.
4. Track which model families produce `gesvda` convergence failures; if they cluster by numerical condition, skip the
   failed attempt using a cheap and conservative predicate.
