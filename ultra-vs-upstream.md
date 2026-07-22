# GPTQModel Ultra vs Upstream GPTQModel

GPTQModel Ultra is no longer just the embedding-requantization fork described by the previous version of this
document. It is now an optimization and experimentation branch that periodically imports upstream GPTQModel while
carrying additional quantization-quality work, calibration/runtime optimizations, adapter compression, and
hardware-specific kernels.

## Comparison snapshot

This snapshot was audited on 2026-07-20 UTC against the tips of both repositories.

| Item | Revision |
|---|---|
| Ultra `main` | `1764f983` (`7.2.0+ultra`) |
| Upstream `main` | `caf51275` (`7.3.1`) |
| Common ancestor | `6edc24d8` |
| Latest upstream import in Ultra | `20e8295f`, a squash sync containing upstream changes through PR `#2953` |

The direct tree comparison is 156 changed files, with 42,365 additions and 1,894 deletions. The raw Git graph says
that Ultra is 361 commits ahead and 39 commits behind upstream, but those counts substantially overstate the product
divergence: Ultra's upstream syncs are squashed, so equivalent changes have different commit identities. Compare the
trees and features, not just `rev-list` counts.

## At a glance

| Area | Upstream at this snapshot | Ultra at this snapshot | Maturity |
|---|---|---|---|
| Core GPTQModel functionality and model definitions | Canonical implementation | Upstream functionality imported through PR `#2953` | Maintained by periodic syncs |
| GPTQ scale/range search | Legacy weight-MSE search through `mse` | Adds `mse`, activation-diagonal, Hessian, and hybrid objectives; activation search is the default | Integrated and tested |
| Pre-quantization analysis | No comparable processor | Optional weight-only quantizability ranking with Markdown/JSON output | Integrated |
| Calibration work sharing | Per-module GPTQ Hessian and AWQ activation work | Shares eligible same-input GPTQ Hessian/inverse work and AWQ activation/x-mean work | Enabled by default, configurable |
| Input/output embedding quantization | `nn.Embedding` is not a quantization target | GPTQ and RTN paths, `TorchQuantEmbeddings`, checkpoint detection, and save/load preservation | Implemented; public API regression noted below |
| Embedding-only checkpoint rewrite | Not present | Loads only selected embedding tensors and can rewrite affected safetensors shards | Implemented; save lifecycle regressions noted below |
| Weight-only CPU parallelism | Standard weight-only lifecycle | Configurable RTN worker count and free-threaded/no-GIL parallel execution | Integrated with serial fallback |
| EoRA/LoRA | Standard EoRA adapter generation and loading | Cholesky fast path, grouped 4/6/8-bit adapter storage, selectable dequantization, and an optional fused CUDA tail | Mixed: integrated plus experimental fast paths |
| CUDA kernels | Upstream kernel set | Adds GrassHopper operators, low-level Triton INT3 work, and a narrowly gated packed Marlin prefill path | Hardware/shape specific |
| Ascend NPU kernels | Upstream Torch NPU paths | Adds Komodo/Cannoe, native GGUF Q4_0 and ParoQuant routes, and extensive 910B tooling | Experimental |

## Ultra-only changes

### GPTQ scale search and quantization analysis

Ultra adds `ScaleSearchConfig` with four objectives:

- `mse`: the legacy uniform weight-error range search;
- `activation`: diagonal Hessian-weighted error;
- `hessian`: group-local correlated Hessian error; and
- `hybrid`: an equal blend of diagonal and full-Hessian error.

The selector can be overridden per module through dynamic config. Omitting it selects `activation`; explicitly passing
`None` disables scale search. This default is a behavioral difference from upstream and should be called out in A/B
comparisons. See [gptq_scale_search.md](gptq_scale_search.md) and
[gptqmodel/quantization/config.py](gptqmodel/quantization/config.py).

`AnalysisConfig` adds an optional preprocessor that estimates grouped quantization error from weights before the main
quantization loop. It ranks difficult modules using relative RMSE, small-value pressure, bad-block rate, and outlier
ratio, then exposes Markdown and JSON reports on `model.quantize_analysis`. It is a heuristic triage report, not an
accuracy evaluation. See [gptqmodel/looper/analysis_processor.py](gptqmodel/looper/analysis_processor.py).

### Quantization runtime and memory work

Ultra deduplicates work for eligible module groups that consume the same activations:

- GPTQ can share calibration Hessian accumulation and the inverse/Cholesky result across compatible modules.
- AWQ can share the captured activation CPU copy and chunked activation-mean reduction.
- AWQ can retain pristine scale-search restore weights on the GPU when memory headroom permits.

The main controls are `enable_shared_hessian_cache`, `enable_activation_x_mean_cache`, and
`scale_search_gpu_weight_restore`. The implementation retains isolated state for incompatible paths such as GPTAQ,
FOEM, and embedding GPTQ. See [docs/quantization_runtime_sharing.md](docs/quantization_runtime_sharing.md).

The GPTQ and AWQ implementations also contain a larger set of allocation, chunking, buffer-reuse, and device-side
indexing optimizations. RTN weight-only quantization adds `weight_only_quant_threads`; automatic parallel execution is
conservative and falls back to serial execution when a free-threaded Python runtime is unavailable.

Ultra also favors lower persistent memory in the Torch EXL3 fallback: reconstructed FP32 inner/full weights are no
longer cached across forwards. That is a memory-versus-recompute policy difference from upstream, not a new EXL3
checkpoint format.

### Embedding quantization and targeted checkpoint updates

Ultra extends the quantizable module set to `nn.Embedding` and adds `QuantizeEmbed.INPUT`, `OUTPUT`, and `BOTH`.
The calibrated path includes embedding-specific GPTQ statistics and uses `TorchQuantEmbeddings` for inference. It can
untie shared input/output weights, preserve quantized embeddings across save/load, and detect them in safetensors
checkpoints. `QuantizeEmbedConfig` is exported from both `gptqmodel` and `gptqmodel.quantization`; `quantize()` and
`requantize()` accept that config while retaining `embed_quant_mode` as a compatibility argument.

The RTN weight-only path can quantize only the requested embedding modules without materializing the full model. The
automatic `save()` path copies unchanged checkpoint artifacts and rewrites only shards containing replaced embedding
tensors; `save_quantized_embeddings()` exposes the same targeted behavior directly. This path depends on the shared
upstream shell/`LazyTurtle` architecture for checkpoint-backed materialization; shell/turtle itself is not an Ultra-only
feature.

### EoRA and compressed adapters

Ultra extends EoRA with:

- a Cholesky/SVD fast path with eigensolve fallback;
- grouped signed 4-, 6-, and 8-bit storage for LoRA A/B tensors;
- dequantization either at load time or during forward execution;
- in-place/addmm adapter application where safe; and
- an optional JIT CUDA `lora_up_add` tail used by eligible Marlin inference shapes.

The serialized low-bit adapter metadata is carried in GPTQModel-specific `LoraConfig` fields. These checkpoints are an
Ultra extension and should not be assumed to load in unmodified upstream runtimes. See
[gptqmodel/adapter/quant.py](gptqmodel/adapter/quant.py) and
[gptqmodel/utils/marlin_lora.py](gptqmodel/utils/marlin_lora.py).

### Hardware-specific kernels

- **Packed Marlin prefill:** an automatic, conservative W4A16 large-`M` route for validated full-K, symmetric,
  group-size-128 GPTQ shapes on `sm_80`. Unsupported hardware, contracts, and shapes fall back to ordinary Marlin.
  The current automatic table was tuned on a 124-SM Ampere target, so it is not a general Marlin replacement. See
  [docs/kernels/marlin_prefill_decode.md](docs/kernels/marlin_prefill_decode.md).
- **GrassHopper:** JIT CUDA GEMV/GEMM operators for grouped 3-, 4-, and 8-bit weights, including LoRA variants. It is
  exposed through the extension API and benchmark tooling, not as a normal `BACKEND` selector. See
  [docs/kernels/grasshopper.md](docs/kernels/grasshopper.md).
- **Triton INT3:** low-level 3-bit packed-word support exists in the Triton matmul kernels, but
  `TritonV2Linear.SUPPORTS_BITS` still excludes 3. Treat it as kernel-level work rather than public backend support.
- **Komodo:** explicit FP16 Ascend NPU GPTQ/AWQ backends using native packed INT4 plans, with Torch fallbacks for
  unsupported layouts. Source quant buffers can be released after prepacking. The same native NPU INT4 machinery is
  also used for eligible GGUF Q4_0 and ParoQuant execution. See [hw/komodo.md](hw/komodo.md).
- **Cannoe:** a separate experimental Ascend CANN/Ascend C path with planner, JIT bridge, raw validators, profiling,
  and fallback behavior. It should not be described as a production replacement for the upstream NPU backend. See
  [hw/cannoe.md](hw/cannoe.md).

## Upstream changes not yet reflected in Ultra

Ultra's `20e8295f` sync includes the functional upstream changes through PR `#2953`. At the comparison baseline,
upstream has three later commits:

| Upstream change | Ultra status |
|---|---|
| PR `#2954`: require `Evalution>=0.0.8` for the `eval` extra | Ultra still declares unversioned `Evalution` |
| PR `#2955`: version `7.3.0` | Ultra retains its own `7.2.0+ultra` version line |
| PR `#2956`: version `7.3.1` | Ultra retains its own `7.2.0+ultra` version line |

The version difference does not by itself mean that Ultra is missing all upstream 7.3 functionality: the preceding
sync imported the code through PR `#2953` in one squash commit.

## Current caveats

- Embedding quantization uses the portable Torch inference module, not every accelerated linear backend.
- GrassHopper and Triton INT3 are low-level extension/kernel surfaces, not generally selectable model backends.
- Packed Marlin prefill, Komodo, Cannoe, and the fused EoRA tail have explicit hardware, dtype, shape, or environment
  gates. Their fallback paths are part of the design and should remain enabled in compatibility testing.
- Many Ultra-only files are benchmark logs, hardware notes, experiments, and tests. The large line-count difference
  should not be read as 42,000 lines of stable public API.

## Refreshing this comparison

Use commit-pinned tree comparisons so squashed sync history does not distort the result:

```bash
git fetch --no-tags origin main
git fetch --no-tags https://github.com/ModelCloud/GPTQModel.git \
  main:refs/remotes/upstream/main

git rev-parse origin/main upstream/main
git merge-base origin/main upstream/main
git diff --shortstat upstream/main origin/main
git diff --name-status upstream/main origin/main
git rev-list --left-right --count upstream/main...origin/main
git rev-list --left-right --cherry-pick --count upstream/main...origin/main
```

After refreshing the revisions, inspect public config defaults, backend selectors, tests, and documented fallback
conditions before changing the capability table. Commit counts alone are not sufficient for this repository.
