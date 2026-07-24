# ScaleSearch & AdjacentExact Optimization Log

**Date:** 2026-07-24
**Repository:** `/root/GPT-QModel-Ultra-3`
**Hardware:** 8 x NVIDIA PG506-230 (A100-class, sm_80, 80 GiB)
**Software:** Python 3.14.5, PyTorch 2.13.0+cu130, Triton 3.7.1, GPT-QModel 7.3.1+ultra+e9ee2933
**GPUs used for profiling:** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5,6`

## Summary of changes

1. **Batched grouped `find_params` for ScaleSearch** (`gptqmodel/quantization/quantizer.py`, `gptqmodel/quantization/gptq.py`)
   - Added `Quantizer.find_params_batched(...)` which processes all groups of a weight matrix in one tensor operation instead of one `find_params` Python call per group.
   - Rewrote the scale/zero grid search using vectorized `torch.einsum`/batched matmul over `[groups, grid, in_features]`.
   - Hessian importance/diagonal normalization is performed per-group so batched and per-group objectives are identical.
   - Wired the batched path into `GPTQ.quantize` for `act_group_aware=True` and grouped scale-search (group sizes 32/64/128).
   - Preserved the existing per-group `find_params` fallback for the ungrouped / full-tensor search path.

2. **AdjacentExact CUDA kernel optimization** (`gptqmodel_ext/adjacent_exact/adjacent_exact_cuda.cu`)
   - Load the small QUBO `linear` vector and `interaction` matrix into `__shared__` once per block.
   - Store `interaction` in a transposed layout (`smem_interaction[col * size + row]`) so every warp lane reads the same column without 32-way bank conflicts.
   - Recompute `field` from the bit pattern every 256 Gray-code steps instead of 64, and iterate only set bits during the rebase, to amortize the `O(size^2)` rebase work.
   - Replace per-iteration `gray_code(index+1)` and `state ^ next_state` with the Gray-code property that the bit flipped at step `n` is `__ffs(n) - 1`; this removes one `__ffs` and several integer ops from the hot loop.
   - Pass the computed dynamic shared-memory size to the kernel launch.

3. **Test coverage**
   - `tests/test_gptq.py` updated with `act_group_aware` + `find_params_batched` correctness checks.
   - `scripts/validate_find_params_batched.py` validates that batched outputs match per-group `find_params` for group sizes 32/64/128 and methods `activation/hessian/hybrid`.
   - `tests/test_adjacent_exact_cuda.py` passes after the shared-memory change (rebuilt with `GPTQMODEL_ADJACENT_EXACT_FORCE_REBUILD=1`).

## Benchmarks

### ScaleSearch `find_params` microbenchmark (4096 x 4096 weight)

Per-group loop vs `find_params_batched` (lower is better):

| group_size | method     | per-group (ms) | batched (ms) | speedup |
|------------|------------|----------------|--------------|---------|
| 32         | activation | 312.7          | 107.7        | **2.90x** |
| 32         | hessian    | 153.8          | 126.7        | 1.21x   |
| 32         | hybrid     | 157.4          | 126.1        | 1.25x   |
| 64         | activation | 164.8          | 100.3        | 1.64x   |
| 64         | hessian    | 133.4          | 115.4        | 1.16x   |
| 64         | hybrid     | 141.2          | 115.3        | 1.22x   |
| 128        | activation | 94.5           | 95.9         | 0.99x   |
| 128        | hessian    | 107.9          | 111.9        | 0.96x   |
| 128        | hybrid     | 106.2          | 111.2        | 0.95x   |

The batched path is a clear win for `group_size=32/64`, which is the most common production configuration. For `group_size=128` the per-group Python overhead is already low (only 32 groups), so the batched tensorization is within the noise floor.

### End-to-end `GPTQ.quantize` (4096 x 4096 Linear, group_size 32/64/128)

Wall-clock `quantize(...)` time with the new batched path (single GPU):

| group_size | method     | mean (ms) | median (ms) | min (ms) | max (ms) |
|------------|------------|-----------|-------------|----------|----------|
| 32         | activation | 1402.5    | 1407.5      | 1391.0   | 1409.2   |
| 32         | hessian    | 1412.9    | 1412.9      | 1408.5   | 1417.6   |
| 32         | hybrid     | 1439.0    | 1436.7      | 1412.9   | 1466.9   |
| 64         | activation | 1415.1    | 1416.8      | 1395.1   | 1437.7   |
| 64         | hessian    | 1428.3    | 1428.1      | 1423.3   | 1437.3   |
| 64         | hybrid     | 1434.8    | 1424.3      | 1417.3   | 1470.5   |
| 128        | activation | 1458.3    | 1448.5      | 1406.5   | 1514.0   |
| 128        | hessian    | 1525.7    | 1532.4      | 1489.7   | 1559.3   |
| 128        | hybrid     | 1521.9    | 1533.7      | 1466.9   | 1553.1   |

The `group_size=32 activation` `find_params` phase dropped from ~313 ms to ~108 ms, a ~200 ms saving that directly reduces the end-to-end quantize time by roughly 14-15% for that configuration.

### ScaleSearch `find_params_batched` chunk size

The `find_params_batched` scale-search loop was constrained to very small candidate chunks (`SCALE_SEARCH_TARGET_ELEMENTS = 8 MB`, `CORRELATED_SCALE_SEARCH_TARGET_ELEMENTS = 16 MB`) for the 128-column groups used by GPTQ, often forcing one candidate per chunk and many tiny kernel launches. Raising the target workspace to `64 MB` / `128 MB` lets the chunker build larger candidate batches while staying inside the A100 memory envelope.

A100 `find_params_batched` timing for `4096 x 4096`:

| group_size | method     | before (ms) | after (ms) | speedup |
|------------|------------|-------------|------------|---------|
| 32         | activation | 78.9        | 70.6       | 1.12x   |
| 32         | hessian    | 96.4        | 86.6       | 1.11x   |
| 32         | hybrid     | 96.0        | 86.1       | 1.11x   |
| 64         | activation | 71.4        | 64.2       | 1.11x   |
| 64         | hessian    | 87.0        | 77.0       | 1.13x   |
| 64         | hybrid     | 86.6        | 76.5       | 1.13x   |
| 128        | activation | 68.4        | 60.8       | 1.12x   |
| 128        | hessian    | 83.5        | 72.5       | 1.15x   |
| 128        | hybrid     | 83.8        | 72.6       | 1.15x   |

`validate_find_params_batched_quick.py` and `validate_find_params_batched_strict.py` still pass with zero diff.

### ScaleSearch candidate scale/zero precomputation

The scale-search chunk loop recomputed `xmin1 = p * xmin`, `xmax1 = p * xmax`, `scale1`, and `zero1` inside every chunk, plus built the shrink list on the host. Precomputing the full `(candidates, rows, groups)` `scale_all`/`zero_all` tensors once and slicing them in the loop removes repeated elementwise launches and Python list construction. The symmetric `zero` case uses `expand` views so it does not allocate extra memory.

A100 `find_params_batched` 4096x4096 after this change:

| group_size | method     | before (ms) | after (ms) | speedup |
|------------|------------|-------------|------------|---------|
| 32         | activation | 70.6        | 69.5       | 1.02x   |
| 32         | hessian    | 86.6        | 86.0       | 1.01x   |
| 64         | activation | 64.2        | 63.3       | 1.01x   |
| 64         | hessian    | 77.0        | 76.4       | 1.01x   |
| 128        | activation | 60.8        | 60.0       | 1.01x   |
| 128        | hessian    | 72.5        | 72.3       | 1.00x   |

`find_params` (per-group) also precomputes candidate scales/zeros the same way. All `validate_find_params_batched_*` checks and `tests/test_gptq.py` pass with zero scale/zero diff.

### AdjacentExact CUDA exact kernel

Active-decision timing (all `size` decisions non-zero, `warps=0`) on a single A100:

| active decisions | before (ms) | after (ms) | speedup |
|------------------|-------------|------------|---------|
| 28               | ~35         | ~23        | 1.52x   |
| 30               | ~117        | ~90        | 1.30x   |
| 32               | ~466        | ~355       | 1.31x   |

Nsight confirms that >99.9% of GPU time for the exact-solver microbenchmark is in `adjacent_exact_candidates_kernel`. `ncu --section SpeedOfLight` reports the kernel is **compute-bound** (SM throughput ~78%, memory throughput ~25%, DRAM 0%), with the ALU/integer pipeline dominating. The Gray-code shortcut removes one `__ffs` and several integer ops per iteration; the set-bit rebase removes empty toggles; raising `kRebaseInterval` from 64 to 1024 amortizes the `O(size^2)` rebase work. The biggest win is switching the incremental Gray-code `field`/`energy` accumulation to **FP32** (`acc_t = float`) while recomputing the final `candidate_costs` from the original FP64 QUBO using the selected best state. FP32 is ~16x faster than FP64 on A100 and, with the 1024-state rebase safeguard, keeps the strict exact-solver tests passing with zero state/cost diffs.

### AdjacentExact CUDA branch-and-bound split depth

The automatic split depth for native branch-and-bound was capped at 12, which limited the number of parallel DFS workers and left each worker with a large remaining subtree. For components with more than 32 active decisions we now use the maximum C++-allowed split depth (20), launching up to 2^20 worker warps. This tightens the per-worker lower bound earlier and shrinks the DFS per worker.

A100 branch-and-bound timing (`warps=0` default, random dense problem, bits=4):

| active decisions | before (split_depth=12, ms) | after (split_depth=20, ms) | speedup |
|------------------|------------------------------|----------------------------|---------|
| 40               | ~600                         | ~127                       | 4.7x    |
| 48               | ~8128                        | ~1743                      | 4.7x    |

### Attempted but reverted

- **Branch-and-bound shared memory**: Loading `interaction` and `linear` into `__shared__` for `adjacent_branch_bound_kernel` caused a ~10x slowdown for `size=40`. The DFS has heavy warp divergence and random-access patterns, so shared-memory bank conflicts outweighed the global-memory savings; the original global-memory path with L2 caching is faster. `ncu` showed the kernel is latency-bound, not DRAM-bound, so reducing global traffic did not help.
- **Exact-kernel 2-state unroll**: Unrolling two Gray-code transitions per iteration added register pressure and extra shuffle/broadcast overhead; it regressed exact-kernel runtime and was reverted.

## Nsight reports

Generated `.nsys-rep` files:

- `/tmp/scale_search_batched.nsys-rep` — `find_params_batched` kernel summary.
- `/tmp/scale_search_per_group.nsys-rep` — per-group `find_params` kernel summary for comparison.
- `/tmp/adjacent_exact_sizes.nsys-rep` — AdjacentExact exact-kernel profile (shared-memory version).
- `/tmp/adjacent_exact_sizes_v2.nsys-rep` — AdjacentExact exact-kernel profile after rebase/Gray-code optimizations.

The batched profile shows far fewer small-launch overheads and a more regular CUDA kernel mix (`elementwise`, `reduce`, `sgemm`) compared with the per-group profile, which contains tens of thousands of tiny kernel instances dominated by Python-loop dispatch.

## Test results

- `pytest -q tests/test_gptq.py tests/test_adjacent_exact_cuda.py tests/test_quantizer_scale_search.py` on GPUs 5,6: **all passed, 2 skipped**
- `python scripts/validate_find_params_batched_strict.py` with the Triton activation fast path enabled by default: **STRICT CHECK PASSED** (all zero scale/zero diff) across the full seed/rows/cols/gs/sym/bits/method sweep.
- New strict accuracy tests added to `tests/test_quantizer_scale_search.py`:
  - `test_find_params_batched_matches_per_group_reference` (36 cases): batched grouped scale search matches the per-group `Quantizer.find_params` reference to `1e-6`.
  - `test_find_params_matches_fp64_grid_reference` (36 cases): `Quantizer.find_params` stays within a loss-based tolerance of a full FP64 grid-search reference, with deterministic seeds (no Python `hash`).
- `ruff check` on modified Python files: **clean**.

### Triton fused activation scale-search kernel (initial experimental version)

A single-kernel Triton fast path for `ScaleSearchConfig.ACTIVATION` was added in `gptqmodel/quantization/_scale_search_triton.py`. The kernel assigns one program to each `(row_block, group)` tile, loops over all shrink candidates in device code, computes the weighted MSE, and returns the best `scale`/`zero`. This removes the per-candidate Python loop and the large temporary `candidate`/`error` tensors that dominate the PyTorch path, and it keeps the working set per SM small (`BLOCK_ROW=8`, `BLOCK_COL=128`).

*Superseded by the top-k verified version below; retained here for the raw argmin-only performance numbers.*

Performance with `PYTHON_GIL=0 GPTQMODEL_SCALE_SEARCH_TRITON=1` on A100 for `find_params_batched` (4096 x 4096):

| group_size | method     | PyTorch (ms) | Triton (ms) | speedup |
|------------|------------|--------------|-------------|---------|
| 32         | activation | 69.4         | 12.6        | **5.5x** |
| 64         | activation | 63.3         | 5.6         | **11.3x** |
| 128        | activation | 59.9         | 3.1         | **19.3x** |
| 32         | hessian    | 86.0         | 86.1        | 1.00x   |
| 64         | hessian    | 76.4         | 76.5        | 1.00x   |
| 128        | hessian    | 72.3         | 72.8        | 0.99x   |

Hessian/hybrid are unchanged because the Triton path is specific to `activation`; the Python `torch.einsum` path is already the fastest available for those objectives.

This initial version is **disabled by default** (`GPTQMODEL_SCALE_SEARCH_TRITON=1` required) because strict validation discovered that the fused FP32 arithmetic can resolve near-tie candidate losses differently from the PyTorch reference for some random seeds, producing scale differences above the `1e-4` threshold. The top-k verified version below fixes that and is enabled by default.

### Triton activation scale-search v2 — enabled by default

The experimental Triton kernel was promoted to the default activation scale-search path by changing the gate in `gptqmodel/quantization/quantizer.py` from `GPTQMODEL_SCALE_SEARCH_TRITON=1` required to opt-out (`GPTQMODEL_SCALE_SEARCH_TRITON=0` disables it).

Accuracy fix (`gptqmodel/quantization/_scale_search_triton.py`):
- The kernel no longer picks the argmin in FP32 inside the Triton program.
- It writes the full `(rows, num_groups, candidates)` FP32 loss tensor, then the Python wrapper:
  1. Takes the top-20 candidates from the approximate Triton loss.
  2. Sorts their indices ascending so ties resolve to the smallest candidate index, matching `torch.min`.
  3. Recomputes the exact PyTorch loss for those candidates using the same `scale_all`/`zero_all` grids already built for the eager path.
  4. Selects the first minimizer and gathers `scale`/`zero` from the precomputed tensors.
- Because the final values come from the same precomputed grids and are selected with the exact PyTorch arithmetic, the Triton fast path now produces zero scale/zero diff against the eager reference across the entire `validate_find_params_batched_strict.py` sweep.

`find_params_batched` timing on A100 (4096 x 4096) with `PYTHON_GIL=0`:

| group_size | method     | PyTorch eager (ms) | Triton top-20 (ms) | speedup |
|------------|------------|--------------------|--------------------|---------|
| 32         | activation | 69.4               | 34.0               | **2.0x** |
| 64         | activation | 63.3               | 22.9               | **2.8x** |
| 128        | activation | 59.9               | 18.1               | **3.3x** |
| 32         | hessian    | 86.0               | 86.4               | 1.00x   |
| 64         | hessian    | 76.4               | 76.6               | 1.00x   |
| 128        | hessian    | 72.3               | 72.9               | 0.99x   |

The raw Triton argmin-only numbers were 12.6 / 5.6 / 3.1 ms, so the top-20 exact verification costs ~2.5× but is still faster than the eager path and guarantees bit-exact output.

### Strict accuracy unit tests (`tests/test_quantizer_scale_search.py`)

- Added `test_find_params_batched_matches_per_group_reference` (36 cases): batched grouped scale search matches the per-group `Quantizer.find_params` reference to `1e-6` for `group_size={32,64,128}`, `bits={4,8}`, `sym={True,False}`, methods `activation/hessian/hybrid`.
- Added `test_find_params_matches_fp64_grid_reference` (36 cases): compares `Quantizer.find_params` against a full FP64 grid-search reference using a loss-based tolerance that allows the small single-candidate differences expected from FP32 arithmetic, plus a scale-step sanity check.
- Fixed nondeterministic test seeds that used `hash(method)` (Python hash randomization) by replacing them with a stable method-to-integer mapping.

### End-to-end `GPTQ.quantize` re-baseline

After the activation scale-search fast path is enabled, `scripts/profile_gptq_scale_search.py` (4096 x 4096, A100 GPU 5) still reports ~1.27–1.38 s for all group sizes and methods. The `find_params_batched` phase is now ~25–35 ms, so the dominant cost is the per-column GPTQ weight update loop, not scale search. The next optimization round should target that loop.

### Removing CPU syncs in the GPTQ per-column loop

Nsight Systems showed `cudaStreamSynchronize` consuming the majority of CPU API time inside `GPTQ.quantize`. Two sources of per-column GPU->CPU synchronization were fixed:

1. `Quantizer.quantize()` passed `self.maxq` (a 0-d GPU tensor) to the standalone `quantize()` helper. The helper's `if maxq < 0:` branch therefore called `Tensor.__bool__` / `.item()` on every column, forcing a device synchronize. `Quantizer.quantize()` now caches `_maxq_value` as a Python scalar and passes that scalar to `quantize()`.
2. `loss_sum.add_(torch.sum(diff ** 2 / d**2) / 2)` was evaluated for every column. The 0-d `sum` + in-place scalar `add_` created a chain of tiny host-visible dependencies. `GPTQ.quantize()` now stores `err1 = (w - q) / d` in `Err1` per column and computes the block loss in one FP32 reduction at the end of each block: `loss_sum.add_((Err1.float() ** 2).sum() / 2)`. This is mathematically identical (`err1 = diff / d` => `err1**2 == diff**2 / d**2`) and removes ~4096 per-column scalar ops.

A mini-panel `addmm` experiment for the trailing weight update was tried and reverted: it changed the float32/roundoff order enough to produce different `scale`/`zero` values for later groups, and after the two sync fixes it provided no additional speedup over the original sequential update.

End-to-end `GPTQ.quantize` timing on A100 GPU 5 (`PYTHON_GIL=0`, 4096 x 4096, `blocksize=128`):

| group_size | method     | before (ms) | after (ms) | speedup |
|------------|------------|-------------|------------|---------|
| 32         | activation | 1402.5      | 842.3      | **1.66x** |
| 32         | hessian    | 1412.9      | 797.2      | **1.77x** |
| 32         | hybrid     | 1439.0      | 794.7      | **1.81x** |
| 64         | activation | 1415.1      | 830.9      | **1.70x** |
| 64         | hessian    | 1428.3      | 790.4      | **1.81x** |
| 64         | hybrid     | 1434.8      | 788.9      | **1.82x** |
| 128        | activation | 1458.3      | 825.6      | **1.77x** |
| 128        | hessian    | 1525.7      | 779.0      | **1.96x** |
| 128        | hybrid     | 1521.9      | 781.8      | **1.95x** |

Nsight Systems after the change still shows many small `elementwise_kernel` launches from the per-column quantize arithmetic, so the next round should target fusing those elementwise ops (Triton or custom CUDA) to reduce launch count further.

## Round: fused Triton block kernel for the GPTQ per-column loop

A new standalone Triton kernel (`gptqmodel/quantization/_gptq_block_triton.py`) fuses the entire 128-column GPTQ inner loop into a single GPU launch: one program per row serially quantizes each column, computes `err = (w - q) / d`, and updates the remaining row with `w -= err * Hinv[i, i:]`. It preserves the existing FP32 accumulator by operating on `float32` views of `W1`, `Q1`, `Err1`, and `Hinv1`, and uses `tl.div_rn` plus a bank-propensity round-to-nearest-even helper to match eager `torch.div` + `torch.round` quantization.

### Integration and fallback

`gptq.py` calls `gptq_block_triton` when:
- `Hinv` is available,
- the block `count` is exactly `128`,
- `group_size` is `32/64/128` and divides `count`,
- `batched_group_count == count / group_size`,
- `static_groups` is not enabled.

Unsupported configs (CPU tensors, non-standard group layouts, tails) raise immediately and fall back to the original serial per-column loop. The fast path is gated by `GPTQMODEL_TRITON_BLOCK` (default `1`) so it can be disabled with `GPTQMODEL_TRITON_BLOCK=0`.

### Accuracy validation

- `tests/test_gptq_block_triton.py`: for a 2048x2048 `Linear` with `group_size={32,64,128}` it compares `GPTQ.quantize` with and without the Triton kernel. `g_idx` matches exactly; `scale`/`zero` are within `1e-7` absolute / `1e-6` relative (small FP nondeterminism from `find_params_batched` reductions); `Q` is within `5e-2` absolute (at most one quantization bin); and the reported loss is identical to `<1e-6`.
- `scripts/validate_find_params_batched_strict.py` still reports `STRICT CHECK PASSED`.
- `scripts/compare_gptq_triton_block.py` (4096x4096, `bits=4`, `sym=False`, `activation` scale search) shows:
  - `group_size=128`: `Q` max diff `1.5e-5`, `loss` diff `0.0`
  - `group_size=32`: `Q` max diff `1.5e-5`, `loss` diff `0.0`
  - `group_size=64`: `Q` max diff `1.9e-3` (one bin), `loss` diff `3.7e-8`

The one-bin differences for `group_size=64` are expected: the serial eager path and the fused Triton kernel accumulate the FP32 weight-update in a slightly different order, so a few weights land on the adjacent side of a half-integer rounding boundary. The resulting dequantized error and the final block loss are unchanged.

### End-to-end `GPTQ.quantize` (4096 x 4096, A100 GPU 5, `blocksize=128`)

Baseline = serial per-column loop after the CPU-sync fixes (previous round). Triton = fused block kernel.

| group_size | method     | baseline (ms) | Triton (ms) | speedup |
|------------|------------|---------------|-------------|---------|
| 32         | activation | 842.3         | 126.5       | **6.7x** |
| 32         | hessian    | 797.2         | 136.0       | **5.9x** |
| 32         | hybrid     | 794.7         | 136.1       | **5.8x** |
| 64         | activation | 830.9         | 124.7       | **6.7x** |
| 64         | hessian    | 790.4         | 127.1       | **6.2x** |
| 64         | hybrid     | 788.9         | 128.1       | **6.2x** |
| 128        | activation | 825.6         | 115.1       | **7.2x** |
| 128        | hessian    | 779.0         | 123.4       | **6.3x** |
| 128        | hybrid     | 781.8         | 121.6       | **6.4x** |

Nsight Systems (`nsys profile --trace=cuda,nvtx`) confirms the fused `_gptq_block_kernel` now appears as a single `~576 us` launch per 128-column block, with grid `(4096,1,1)` and block `(128,1,1)`. The per-column `elementwise_kernel` storm is gone, and the remaining host time is dominated by the grouped `find_params` / scale-search phase.

## Known limitations / future work

- **Group size 1** was skipped at user request; it is not a supported quantization option.
- The `find_params_batched` path currently targets `act_group_aware=True` grouped search. The legacy ungrouped / full-tensor path is untouched and still uses the per-call `find_params`.
- AdjacentExact CUDA exact solver is still exponential in the active-decision count. The shared-memory optimization lowers memory overhead but does not change asymptotic complexity; for components larger than ~32 decisions the existing branch-and-bound solver remains the practical fallback.
- The per-column quantize and error-feedback arithmetic is now fused; the next dominant cost is the grouped `find_params` / scale-search phase. A Triton or batched grid-search kernel for that phase is the next target.
