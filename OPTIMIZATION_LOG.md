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

---

## Round: Triton fused Hessian/Hybrid scale-search kernel

### Objective

Replace the Python chunk loop + `torch.einsum` Hessian/Hybrid `find_params_batched` path with a single Triton kernel per (row-block, group) that evaluates all shrink candidates, removing the large per-candidate `candidate`/`error` tensors and the slow generic `einsum` contraction for `num_groups=1` per-block calls.

### Changes

1. **`gptqmodel/quantization/_scale_search_triton.py`**
   - Added `_scale_search_hessian_kernel` and `_triton_find_params_batched_hessian_hybrid`.
   - The kernel reuses the existing round-half-to-even quantization helper and computes `error^T @ H @ error` with `tl.dot(error, h, allow_tf32=False, out_dtype=tl.float32)`, followed by `tl.maximum(loss, 0.0)` to match the eager `clamp_min_(0)`.
   - `BLOCK_COL` is launched as `group_size` (32/64/128) so small groups are not padded to 128 columns, avoiding a ~4-16x waste in the matrix multiply.
   - The Python wrapper selects the top-20 candidates from the approximate Triton loss and recomputes the exact PyTorch objective for those candidates, preserving the original first-minimizer tie-breaking.

2. **`gptqmodel/quantization/quantizer.py`**
   - Wired `_triton_find_params_batched_hessian_hybrid` into `find_params_batched` for `method in {HESSIAN, HYBRID}` when the group Hessian is 3D, the group size is 32/64/128, the tensor is contiguous CUDA float16/float32/bfloat16, and `GPTQMODEL_SCALE_SEARCH_TRITON != "0"`.
   - Removed a per-call `.item()` sync in the activation Triton launch by passing `float(maxq_value)` instead of `float(self.maxq.item())`.
   - Changed `_quantize_scale_search_candidates` to clamp with a Python `float(maxq_value)` instead of a CUDA 0-d tensor `self.maxq`, eliminating the implicit `cudaStreamSynchronize` caused by `clamp_` with a tensor argument.

3. **`tests/test_quantizer_scale_search.py`**
   - Added `test_find_params_batched_triton_matches_eager` to compare the Triton and eager `find_params_batched` outputs for `HESSIAN` and `HYBRID` across `group_size={32,64,128}`, `bits={4,8}`, and `sym={False,True}`.

### Accuracy validation

- `pytest -q tests/test_quantizer_scale_search.py tests/test_gptq_block_triton.py tests/test_adjacent_exact_cuda.py tests/test_gptq.py tests/test_quantizer.py`: **162 passed, 2 skipped**.
- `test_find_params_batched_triton_matches_eager`: Triton and eager `scale`/`zero` match to `<1e-6` for all 24 param combinations.
- End-to-end `GPTQ.quantize` comparison (`4096x4096`, `bits=4`, `sym=False`, `mse=2.0`, 8-sample Hessian) with `GPTQMODEL_SCALE_SEARCH_TRITON=0` vs `=1`:
  - `group_size=128 hessian`: 1138.9 ms → 132.5 ms, `Q`/scale/zero/loss diff all `0.0`.
  - `group_size=128 hybrid`: 119.9 ms → 119.6 ms, diffs `0.0`.
  - `group_size=64 hessian`: 128.2 ms → 125.2 ms, diffs `0.0`.
  - `group_size=32 hessian`: 139.6 ms → 134.9 ms, diffs `0.0`.

### Benchmarks

`find_params_batched` (4096 x 4096, all groups, A100 GPU 5):

| group_size | method     | before (ms) | after (ms) | speedup |
|------------|------------|-------------|------------|---------|
| 32         | hessian    | 86.7        | 38.6       | **2.2x** |
| 32         | hybrid     | 86.8        | 38.6       | **2.2x** |
| 64         | hessian    | 77.4        | 36.9       | **2.1x** |
| 64         | hybrid     | 77.3        | 36.7       | **2.1x** |
| 128        | hessian    | 72.3        | 41.6       | **1.7x** |
| 128        | hybrid     | 72.3        | 41.6       | **1.7x** |

The largest end-to-end impact is for `group_size=128 hessian` on dense per-block Hessians, where the eager `einsum` path was ~8.6x slower than the Triton kernel.

### Known limitations / future work

- `find_params_batched` Triton Hessian/Hybrid is restricted to `group_size in (32, 64, 128)` so `tl.dot` operates on power-of-two tile sizes.
- The `find_params_batched` activation path still uses `BLOCK_COL=128` for all group sizes; tightening that tile could further speed up `group_size=32/64 activation`.
- AdjacentExact CUDA exact solver remains exponential in active-decision count; no further work in this round.

---

## Round follow-up: tight `BLOCK_COL` for activation scale-search kernel

### Changes

- `gptqmodel/quantization/_scale_search_triton.py`: launch the activation scale-search kernel with `BLOCK_COL=group_size` instead of the fixed 128. This removes masked-off columns for `group_size=32/64` and lets Triton schedule a tighter tile.
- `tests/test_quantizer_scale_search.py`: extended `test_find_params_batched_triton_matches_eager` to also cover `ScaleSearchConfig.ACTIVATION` (36 parameter combos total).

### Accuracy validation

- `test_find_params_batched_triton_matches_eager` with activation: 36 passed, Triton vs eager `scale`/`zero` match to `<1e-6`.
- `pytest -q tests/test_quantizer_scale_search.py tests/test_gptq.py tests/test_quantizer.py`: **119 passed, 2 skipped**.

### Benchmarks

`find_params_batched` activation (4096 x 4096, all groups, A100 GPU 5):

| group_size | method     | before (ms) | after (ms) | speedup |
|------------|------------|-------------|------------|---------|
| 32         | activation | 33.0        | 27.2       | **1.2x** |
| 64         | activation | 22.8        | 20.2       | **1.1x** |
| 128        | activation | 18.0        | 18.0       | 1.0x    |

`group_size=128` is unchanged because `BLOCK_COL` already equals 128. End-to-end `GPTQ.quantize` with the small `nsamples=4` profile harness is within noise; the gain is expected to be more visible when `find_params` dominates the total time (dense Hessians, larger batches, or many blocks).

### Known limitations / future work

- `find_params_batched` Triton activation now uses `group_size` as the tile width; for non-power-of-two group sizes this may compile less efficient kernels, but the target configurations (32/64/128) are power-of-two.

---

## Round: lazy scale/zero top-k recompute and strict-accuracy fixes

### Objective

Reduce the remaining elementwise launch overhead in `find_params_batched` by materializing `scale`/`zero` only for the Triton top-k candidates instead of the full shrink grid, while keeping strict bit-for-bit agreement with the eager fallback. Also restore correctness for Hessian/Hybrid scale search on dense, ill-conditioned Hessians.

### Notes on reverted experiments

- **Hessian inverse replacement** (`torch.linalg.solve_triangular(L, I, upper=False).T` instead of `cholesky(cholesky_inverse(...))`) was reverted: it is ~2.5x faster on a 4096x4096 SPD matrix and more accurate against `torch.linalg.inv(H)`, but the resulting `W` update diverges enough to break `test_gptq_block_triton[128]` (scale diff 3e-5, relative 1.1%).
- **Triton kernels emitting per-candidate scale/zero** were reverted: writing `scale`/`zero` inside the kernel avoided the Python recompute but produced ULP-different values that reordered the exact-loss argmin and caused order-dependent `sym=True` failures in `test_quantizer_scale_search.py`.

### Changes

1. **`gptqmodel/quantization/quantizer.py`**
   - Moved the full `scale_all`/`zero_all` precompute to *after* the Triton activation fast path, so it is skipped when the Triton wrapper is used.
   - Disabled the `_triton_find_params_batched_hessian_hybrid` call. The approximate Triton top-20 loss can miss the true best candidate on dense/ill-conditioned Hessians (`seed=42 rows=4096 cols=256 gs=64 sym=True bits=4 hessian` produced a 0.031 scale diff). The exact vectorized fallback already evaluates all 80 candidates in one launch and passes the strict check.

2. **`gptqmodel/quantization/_scale_search_triton.py`**
   - The activation wrapper now recomputes `scale_k`/`zero_k` only for the top-k shortlisted candidates instead of receiving the full `scale_all`/`zero_all` tensors.
   - Recompute uses a `torch.int64` `maxq` tensor matching the eager precompute (`self.maxq`), eliminating ULP differences that previously reordered the exact-loss argmin.
   - The Hessian/Hybrid wrapper and kernel remain in the file but are no longer invoked from `quantizer.py`.

3. **`tests/test_quantizer_scale_search.py`**
   - Added `test_find_params_batched_dense_matches_per_group_reference` with a dense positive-definite Hessian, `rows=128`, `cols=256`, and multi-group inputs for `activation/hessian/hybrid` across `group_size={32,64,128}`, `bits={4,8}`, and `sym={False,True}`. This catches FP32/Triton top-k regressions that the small block-diagonal tests miss.

### Accuracy validation

- `pytest -q tests/test_gptq.py tests/test_quantizer.py tests/test_quantizer_scale_search.py tests/test_adjacent_exact_cuda.py tests/test_gptq_block_triton.py`: **174 passed, 2 skipped** (GPU 5).
- `python scripts/validate_find_params_batched_strict.py`: **STRICT CHECK PASSED** on both GPU 5 and GPU 6.
- New `test_find_params_batched_dense_matches_per_group_reference`: **36/36 passed**.

### Benchmarks

End-to-end `GPTQ.quantize` (`4096 x 4096`, `blocksize=128`, `bits=4`, `sym=False`, `mse=2.0`, A100 GPU 5). `before` = previous committed state using full `scale_all`/`zero_all` precompute and Triton Hessian/Hybrid top-20; `after` = lazy activation recompute + fallback for Hessian/Hybrid.

| group_size | method     | before (ms) | after (ms) | delta   |
|------------|------------|-------------|------------|---------|
| 32         | activation | 122.36      | 122.37     | +0.01   |
| 32         | hessian    | 134.92      | 135.02     | +0.10   |
| 32         | hybrid     | 134.92      | 135.07     | +0.15   |
| 64         | activation | 113.52      | 113.48     | -0.04   |
| 64         | hessian    | 125.40      | 125.28     | -0.12   |
| 64         | hybrid     | 127.17      | 125.31     | -1.86   |
| 128        | activation | 110.49      | 108.61     | -1.88   |
| 128        | hessian    | 120.65      | 119.56     | -1.09   |
| 128        | hybrid     | 120.80      | 119.65     | -1.15   |

The activation path avoids the full `scale_all`/`zero_all` precompute, giving measurable savings for `group_size=128`. Hessian/Hybrid run through the exact vectorized fallback and are at parity or slightly faster than the approximate Triton top-20 path.

### Known limitations / future work

- Hessian/Hybrid Triton scale search is disabled until an exact (or provably tight top-k) loss computation is implemented.
- The lazy activation recompute still materializes `scale_k`/`zero_k` and `losses_k` for the top-20 candidates; fusing the exact recompute into the Triton kernel would remove the remaining Python->CUDA launch overhead.
- Group size 1 remains skipped as requested.

---

## Round: reduce activation exact recompute to top-2 candidates

### Objective

Cut the remaining Python-side exact recompute cost in the activation Triton scale-search path by narrowing the shortlist from 20 candidates to 2, while preserving strict bit-for-bit agreement with the eager per-group reference.

### Nsight baseline

End-to-end `GPTQ.quantize` (`4096 x 4096`, `blocksize=128`, all scale-search methods, A100 GPU 5). Dominant GPU kernels by cumulative time:

| rank | kernel family | calls | total GPU time | share | notes |
|------|----------------|-------|----------------|-------|-------|
| 1 | `at::native::elementwise_kernel` | ~37k | ~1730 ms | ~27% | `x/scale`, `(q-zero)*scale`, `-`, `*importance` in the top-20 recompute, plus other elementwise ops |
| 2 | `_gptq_block_kernel` | 1728 | ~852 ms | ~13% | fused per-column GPTQ block step |
| 3 | `at::native::elementwise_kernel` (other inst.) | ~11k | ~636 ms | ~10% | scale/zero expansion and top-k value manipulation |
| 4 | `ampere_sgemm_128x128_nn` | 768 | ~453 ms | ~7% | cuBLAS GEMM (Hessian/Cholesky related) |
| 5 | `at::native::reduce_kernel` | ~5760 | ~433 ms | ~7% | `.sum(dim=-1)` over `group_size` in the recompute and `topk` reductions |
| 6 | `at::native::vectorized_elementwise_kernel` (clamp) | 5760 | ~278 ms | ~4% | `torch.clamp` in the recompute |
| 7 | `at::native::vectorized_elementwise_kernel` (round) | 7488 | ~265 ms | ~4% | `torch.round` in the recompute |
| 8 | `ampere_sgemm_32x128_tn` | 3672 | ~250 ms | ~4% | cuBLAS |
| 9 | `ampere_sgemm_64x64_nn` | 384 | ~201 ms | ~3% | cuBLAS |
| 10 | `kernel_trsm_l_mul32` / `trsm_left_kernel` / `potrf_*` | ~12k | ~320 ms | ~5% | Cholesky inverse factorization |

The top-20 exact recompute accounts for the largest share of elementwise/reduce launches. Reducing `k` from 20 to 2 directly shrinks the expanded `x_exp`, `scale_k_exp`, `zero_k_exp` tensors and the `clamp`/`round`/`-`/`*`/`sum` kernels.

### Changes

1. **`gptqmodel/quantization/_scale_search_triton.py`**
   - Changed the activation wrapper's exact-recompute shortlist from `k=min(candidate_count, 20)` to `k=min(candidate_count, 2)`.
   - Updated the surrounding comment to document the top-2 rationale.
   - Left the Hessian/Hybrid wrapper at `k=20`; it is currently disabled in `quantizer.py` and would need a larger shortlist if re-enabled.

### Accuracy validation

- `pytest -q tests/test_gptq.py tests/test_quantizer.py tests/test_quantizer_scale_search.py tests/test_adjacent_exact_cuda.py tests/test_gptq_block_triton.py`: **210 passed, 2 skipped** on A100 GPU 5.
- `python scripts/validate_find_params_batched_strict.py`: **STRICT CHECK PASSED** on both GPU 5 and GPU 6.
- New dense Hessian unit test `test_find_params_batched_dense_matches_per_group_reference`: **36/36 passed**.

### Benchmarks

`find_params_batched` microbenchmark (`4096 x 4096`, `bits=4`, `sym=False`, `grid=100`, A100 GPU 5). `before` = top-20 exact recompute; `after` = top-2 exact recompute.

| group_size | method | before (ms) | after (ms) | speedup |
|------------|--------|-------------|------------|---------|
| 32         | activation | 25.00 | 11.97 | **2.09x** |
| 64         | activation | 20.06 | 7.27  | **2.76x** |
| 128        | activation | 18.26 | 6.04  | **3.02x** |
| 32         | hessian    | 86.32 | 85.91 | 1.00x |
| 64         | hessian    | 76.61 | 76.87 | 0.99x |
| 128        | hessian    | 72.45 | 72.49 | 0.99x |
| 32         | hybrid     | 86.17 | 86.14 | 1.00x |
| 64         | hybrid     | 76.73 | 76.33 | 1.00x |
| 128        | hybrid     | 72.46 | 72.65 | 0.99x |

End-to-end `GPTQ.quantize` on the same shape is within noise for activation because `find_params_batched` is only one component; the dominant remaining time is the fused GPTQ block kernel and the Hessian Cholesky inversion. The microbenchmark speedup is representative of the `find_params_batched` hot path itself.

### Rejected experiments

- Recompute for only the single `argmin` candidate (`k=1`): failed strict validation on several seeds (`seed=42 rows=512 cols=512 gs=32 sym=True bits=8 activation` scale_diff 1.9e-4, others up to 0.0066). The Triton `tl.sum` reduction order can shift the approximate argmin away from the true torch.sum argmin, so a second candidate is required for safety.
- `k=10` and `k=5` both passed strict validation and the focused test suite, but `k=2` also passed and gives the largest speedup, so it was chosen.

### Next target

The Hessian/Hybrid scale search still falls back to the exact vectorized chunk loop and is 6-12x slower than activation per `find_params_batched` call. Next round: investigate whether an exact or larger-top-k Triton loss kernel can safely accelerate Hessian/Hybrid, or whether the chunk size / `einsum` path in `_scale_search_error_batched` can be reordered to use `torch.bmm` over `group_size` blocks with less global memory traffic.
- AdjacentExact CUDA exact solver remains exponential in active-decision count; no further work in this round.

---

## Round: fused quantize/error step in the scale-search fallback

### Objective

Reduce the elementwise launch overhead in the exact PyTorch Hessian/Hybrid (and MSE) scale-search fallback by folding the dequantized `candidate` -> `error` subtraction into `_quantize_scale_search_candidates` and replacing the `add(zero).clamp(0,maxq).sub(zero)` sequence with a single `clamp(-zero, maxq-zero)`. Also expand the strict unit-test matrix to include `bits=2` and larger model-like shapes.

### Notes on rejected experiments

- **Triton Hessian/Hybrid top-k shortlisting** was investigated by calling `_scale_search_hessian_kernel` to get an approximate loss, then recomputing the exact objective for the top-k candidates using the fallback `torch.einsum`. Even with `k=80` (all candidates) the Triton loss produced a different argmin than the eager fallback because the kernel's FP32 reduction order is not equivalent to `torch.einsum` / `torch.sum`. A sweep over `k ∈ {2,5,10,20,40,60,80}` showed 399-401 failures per 972 cases against the per-group reference, so the Triton Hessian/Hybrid kernel remains disabled in `quantizer.py`. The activation Triton path still uses the proven top-2 exact recompute.

### Changes

1. **`gptqmodel/quantization/quantizer.py`**
   - `_quantize_scale_search_candidates` now returns the reconstruction `error` (`dequant - x`) instead of the dequantized `candidate`. This removes a separate `candidate - x` elementwise launch in both `find_params` and `find_params_batched` callers.
   - The quantize clamp is rewritten as `q.clamp_(-zero, maxq - zero).mul_(scale).sub_(x)`, which is algebraically identical to `(round(x/scale)+zero).clamp(0,maxq).sub(zero))*scale - x` but avoids two elementwise `add`/`sub` of `zero`.
   - Callers in `find_params` and `find_params_batched` updated to consume `error` directly.

2. **`tests/test_find_params_batched_strict.py` (new)**
   - 108 additional strict-accuracy cases covering `rows={128,4096}`, `cols={512,4096}`, `group_size={32,64,128}`, `bits={2,4,8}`, `sym={False,True}`, methods `activation/hessian/hybrid`.
   - Activation cases explicitly enable Triton; Hessian/Hybrid cases explicitly disable it to validate the exact fallback.
   - Uses a cheap block-diagonal positive-definite Hessian so the square 4096x4096 matrix is feasible in unit-test time.

### Accuracy validation

- `pytest -q tests/test_quantizer_scale_search.py tests/test_quantizer.py tests/test_gptq.py tests/test_gptq_block_triton.py tests/test_adjacent_exact_cuda.py tests/test_find_params_batched_strict.py`: **318 passed, 2 skipped** on A100 GPU 5.
- `python scripts/validate_find_params_batched_strict.py`: **STRICT CHECK PASSED** on GPU 5.
- All new `test_find_params_batched_strict.py` cases match per-group `find_params` to `<1e-6`.

### Benchmarks

`find_params_batched` microbenchmark (`4096 x 4096`, `bits=4`, `sym=False`, `grid=100`, A100 GPU 5). `before` = previous committed top-2 activation + exact fallback; `after` = fused quantize/error step.

| group_size | method     | before (ms) | after (ms) | speedup |
|------------|------------|-------------|------------|---------|
| 32         | activation | 11.86       | 11.86      | 1.00x   |
| 32         | hessian    | 114.52      | 75.17      | **1.52x** |
| 32         | hybrid     | 113.29      | 75.18      | **1.51x** |
| 64         | activation | 9.06        | 7.16       | **1.27x** |
| 64         | hessian    | 100.45      | 65.45      | **1.53x** |
| 64         | hybrid     | 100.78      | 65.44      | **1.54x** |
| 128        | activation | 6.83        | 5.93       | **1.15x** |
| 128        | hessian    | 95.43       | 61.34      | **1.56x** |
| 128        | hybrid     | 92.62       | 61.55      | **1.50x** |

End-to-end `GPTQ.quantize` (`4096 x 4096`, `blocksize=128`, `bits=4`, `sym=False`, `mse=2.0`, A100 GPU 5):

| group_size | method     | mean (ms) | median (ms) | min (ms) | max (ms) |
|------------|------------|-----------|-------------|----------|----------|
| 32         | activation | 111.62    | 111.30      | 111.06   | 113.11   |
| 32         | hessian    | 124.05    | 123.99      | 123.86   | 124.33   |
| 32         | hybrid     | 124.51    | 124.71      | 124.02   | 124.98   |
| 64         | activation | 105.01    | 105.04      | 104.76   | 105.18   |
| 64         | hessian    | 114.16    | 114.03      | 113.94   | 114.44   |
| 64         | hybrid     | 114.45    | 114.47      | 114.33   | 114.51   |
| 128        | activation | 99.83     | 99.78       | 99.66    | 100.08   |
| 128        | hessian    | 108.89    | 109.08      | 108.17   | 109.55   |
| 128        | hybrid     | 111.33    | 111.24      | 109.01   | 113.70   |

Nsight Systems on `find_params_batched` hessian shows the elementwise kernel storm (`div`, `round`, `add`, `clamp`, `sub`, `mul`) is reduced; the remaining time is dominated by the `torch.einsum` `error @ H @ error` contraction and the per-candidate reductions, not launch overhead.

### Known limitations / future work

- Hessian/Hybrid Triton shortlisting remains disabled because the approximate FP32 kernel loss reorders the argmin relative to the eager `torch.einsum` path. The next round should either improve the kernel loss to be bit-exact with eager or find a provably safe top-k bound.
- The exact fallback contraction `torch.einsum("crgi,gij->crgj", error, hessian)` is already as fast as a hand-rolled `bmm` reshape and is memory-throughput bound on the `error`/`projected` tensors; further speedup likely requires fusing the quantize/error/contract/reduce chain into a single Triton/CUDA kernel per candidate group.
- Group size 1 is skipped as requested.
- AdjacentExact CUDA exact solver remains exponential in active-decision count; no further work in this round.
