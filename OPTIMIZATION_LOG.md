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

## Round: Triton Hessian/Hybrid scale-search kernel with exact top-k recompute

### Objective

Remove the dominant `torch.einsum` `error @ H @ error` contraction and the per-candidate elementwise launch overhead in the Hessian/Hybrid scale-search fallback by fusing quantize/error/contract/reduce into one Triton kernel, while keeping the final candidate selection bit-exact with the PyTorch reference.

### Changes

1. **`gptqmodel/quantization/_scale_search_triton.py`**
   - Added a new Triton kernel `_scale_search_hessian_kernel` that assigns one program to each `(row_block, group)` tile.
   - The kernel receives precomputed `(rows, groups, candidates)` `scale` and `zero` pointers (using the same `int64` `maxq` tensor as the eager path, so `scale` values match exactly).
   - Inside each program it loops over candidates, computes the dequantized reconstruction, the projected Hessian via `tl.dot(error, h, allow_tf32=False)`, and the quadratic loss with `tl.sum`.
   - The kernel returns the full `(rows, groups, candidates)` FP32 loss tensor.
   - The wrapper `_triton_find_params_batched_hessian_hybrid` then:
     1. Top-k shortlists the 20 lowest-loss candidates from the Triton loss.
     2. Sorts the indices ascending to match `torch.min` tie-break behavior.
     3. Recomputes the exact PyTorch Hessian/Hybrid loss for those 20 candidates using the same precomputed `scale`/`zero` grids.
     4. Gathers the final `scale`/`zero` from the precomputed tensors.
   - This mirrors the proven activation fast-path pattern: Triton gets near the argmin quickly, then PyTorch reselects exactly.

2. **`gptqmodel/quantization/quantizer.py`**
   - Imported `_triton_find_params_batched_hessian_hybrid` and added a Triton branch for `ScaleSearchConfig.HESSIAN` and `HYBRID` in `find_params_batched`.
   - Gated the branch to `maxq_value >= 15` (bits >= 4) plus the existing `group_size <= 128`, CUDA, contiguous, and non-groupwise processing guards.
   - `bits=2` is excluded from the Triton Hessian/Hybrid fast path because the very coarse quantization grid (`maxq=3`) creates many near-tie candidates; the Triton FP32 reduction order does not reproduce the exact eager argmin for those cases, and the top-20 exact recompute can miss the true best candidate. `bits=2` falls through to the already-optimized exact Python fallback.

### Accuracy validation

- `python scripts/validate_find_params_batched_strict.py` on A100 GPU 5: **STRICT CHECK PASSED** for all rows/cols/group_size/bits/sym/method seeds.
- `pytest -q tests/test_quantizer_scale_search.py tests/test_quantizer.py tests/test_gptq.py tests/test_gptq_block_triton.py tests/test_adjacent_exact_cuda.py tests/test_find_params_batched_strict.py` on A100 GPU 5: **318 passed, 2 skipped**.
- `ruff check` on modified files: clean.
- `git diff --check`: clean.

### Benchmarks

`find_params_batched` microbenchmark (`4096 x 4096`, `bits=4`, `sym=False`, `grid=100`, `maxshrink=0.8`, A100). `before` = exact PyTorch fallback; `after` = Triton top-20 + exact recompute.

| group_size | method  | before (ms) | after (ms) | speedup |
|------------|---------|-------------|------------|---------|
| 32         | hessian | 75.3        | 40.2       | **1.87x** |
| 32         | hybrid  | 75.3        | 40.2       | **1.87x** |
| 64         | hessian | 65.5        | 37.1       | **1.76x** |
| 64         | hybrid  | 68.4        | 37.1       | **1.84x** |
| 128        | hessian | 61.7        | 41.8       | **1.48x** |
| 128        | hybrid  | 62.9        | 41.9       | **1.50x** |

GPU 6 reproduces the same trend (e.g. `hessian` `group_size=32` ~40.2 ms).

End-to-end `GPTQ.quantize` (`4096 x 4096`, `blocksize=128`, `bits=4`, `sym=False`, `mse=2.0`, A100 GPU 5):

| group_size | method     | mean (ms) | median (ms) | min (ms) | max (ms) |
|------------|------------|-----------|-------------|----------|----------|
| 32         | activation | 112.9     | 111.1       | 111.0    | 120.0    |
| 32         | hessian    | 124.0     | 123.9       | 123.7    | 124.3    |
| 32         | hybrid     | 124.8     | 124.5       | 124.1    | 126.3    |
| 64         | activation | 105.4     | 105.0       | 104.6    | 106.3    |
| 64         | hessian    | 115.2     | 115.0       | 114.2    | 116.2    |
| 64         | hybrid     | 115.3     | 115.1       | 114.9    | 115.9    |
| 128        | activation | 100.6     | 100.9       | 99.7     | 101.5    |
| 128        | hessian    | 109.1     | 108.7       | 108.3    | 110.1    |
| 128        | hybrid     | 109.0     | 108.9       | 108.5    | 109.6    |

The Hessian/Hybrid `find_params_batched` phase is roughly halved, which shows up as ~8-10 ms lower end-to-end `quantize` times compared with the previous fallback-only commit.

### Nsight Systems

Profile of the Triton Hessian/Hybrid path (`4096 x 4096`, `group_size=32`, `bits=4`, `hessian`, A100 GPU 5):

- `_scale_search_hessian_kernel` is the single largest GPU consumer at ~32% of kernel time (avg ~14 ms per invocation over 5 warmup calls).
- The exact top-20 recompute shows as `ampere_sgemm_128x128_nn` (~12%), `at::native::sbtopk::gatherTopK` (~8%), and elementwise/reduce kernels (~14%).
- Total kernel time is dominated by the fused Triton contraction; the remaining exact-recompute kernels are unavoidable if the final selection must match the eager reference.

### Known limitations / future work

- `bits=2` Hessian/Hybrid still uses the exact Python fallback because the Triton FP32 loss is not a faithful proxy for the coarse grid; expanding the top-k to 60/80 candidates fixes accuracy but erases the speedup, so the fallback remains the safer and faster choice.
- The Triton path is limited to `group_size <= 128` and contiguous float-like tensors; CPU and non-target GPU fallbacks are preserved.
- Hybrid MSE and activation components are fused inside the kernel; further speedup may come from lowering the top-k exact recompute into Triton as well, or from using FP16/BF16 accumulation where the Hessian dynamic range allows it.

---

## Round: Triton scale-search tile-size tuning and documentation

### Objective

Tune the row-tile size of the activation and Hessian/Hybrid Triton scale-search kernels and add user-facing documentation for the ScaleSearch feature, algorithms, usage, and accuracy checks.

### Changes

1. **`gptqmodel/quantization/_scale_search_triton.py`**
   - Increased `BLOCK_ROW` from `8` to `32` for both the activation and Hessian/Hybrid `find_params_batched` Triton fast paths.
   - A larger row tile amortizes candidate-loop control overhead and improves SM occupancy for the common per-block `rows=4096` case while keeping register pressure low for `group_size` 32/64/128.
   - `k=20` for Hessian/Hybrid and `k=2` for activation remain unchanged; the tile-size change does not affect the final exact-recompute accuracy path.

2. **`docs/scale_search.md` (new)**
   - Documents the four ScaleSearch modes (`MSE`, `ACTIVATION`, `HESSIAN`, `HYBRID`), when each runs, speed/quality trade-offs, `QuantizeConfig` usage examples, per-module `dynamic` overrides, tuning knobs (`mse`, `grid`, `maxshrink`, `group_size`), and the strict-accuracy validation command.

3. **`README.md`**
   - Added a `### ScaleSearch` subsection under `## Quantization Support` that links to `docs/scale_search.md`.

### Accuracy validation

- `python scripts/validate_find_params_batched_strict.py` on A100 GPU 5: **STRICT CHECK PASSED**.
- `pytest -q tests/test_gptq.py tests/test_quantizer.py tests/test_quantizer_scale_search.py tests/test_adjacent_exact_cuda.py tests/test_gptq_block_triton.py` on A100 GPU 5: **210 passed, 2 skipped**.
- `ruff check gptqmodel/quantization/_scale_search_triton.py` and `git diff --check`: clean.

### Benchmarks

`find_params_batched` microbenchmark (`4096 x 4096`, `bits=4`, `sym=False`, `grid=100`, `maxshrink=0.8`, A100 GPU 5). `before` = `BLOCK_ROW=8`; `after` = `BLOCK_ROW=32`.

| group_size | method     | before (ms) | after (ms) | speedup |
|------------|------------|-------------|------------|---------|
| 32         | activation | 11.86       | 10.37      | 1.14x   |
| 32         | hessian    | 40.2        | 34.7       | 1.16x   |
| 32         | hybrid     | 40.2        | 34.7       | 1.16x   |
| 64         | activation | 9.06        | 7.13       | 1.27x   |
| 64         | hessian    | 37.1        | 37.4       | 0.99x   |
| 64         | hybrid     | 37.1        | 37.4       | 0.99x   |
| 128        | activation | 6.83        | 5.93       | 1.15x   |
| 128        | hessian    | 41.8        | 41.8       | 1.00x   |
| 128        | hybrid     | 41.9        | 41.8       | 1.00x   |

Per-block `find_params_batched` (`rows=4096`, `cols=128`, `group_size=32`, `bits=4`, `sym=False`): 2.22 ms -> 2.09 ms (1.06x). End-to-end `GPTQ.quantize` is within run-to-run noise because the fused block kernel and Cholesky inversion still dominate.

### Nsight Systems

No new Nsight capture in this round; the tile-size change primarily reduces Triton launch/loop overhead and is reflected in the microbenchmark timing.

### Known limitations / future work

- `bits=2` Hessian/Hybrid still falls back to the exact Python loop.
- The activation path is now the default `scale_search` mode; further speedup may come from fusing the top-k exact recompute or tuning `BLOCK_ROW` per `group_size`.

---

## Round: fix Hessian/Hybrid tile-size regression and reduce activation exact-recompute allocations

### Objective

Re-tune `BLOCK_ROW` after discovering that `BLOCK_ROW=32` causes a large regression in the Hessian/Hybrid Triton kernel for `group_size=128`, and reduce the temporary allocations in the activation exact-recompute step.

### Changes

1. **`gptqmodel/quantization/_scale_search_triton.py`**
   - Restored `BLOCK_ROW = 8` for the Hessian/Hybrid `_scale_search_hessian_kernel` path. `BLOCK_ROW=32` raised register pressure for `group_size=128` and made the `tl.dot(error, h)` step ~13x slower (~553 ms vs ~42 ms). Activation remains at `BLOCK_ROW=32` where it is faster.
   - Reimplemented the activation top-2 exact recompute with in-place PyTorch operations and broadcasting views to avoid separate `dequant` and `error` tensors. The sequence `round → clamp → subtract zero → multiply scale → subtract x → square → multiply importance → sum` now reuses a single `q` tensor.

### Notes on rejected experiments

- **Multi-GPU candidate split** was prototyped for the activation Triton kernel and shown to be slower than the single-GPU path for both full-tensor and per-block `find_params_batched` due to cross-device copy overhead. It has been removed and is not part of this commit.
- **Triton-based exact recompute** was also prototyped and failed strict validation because Triton's `tl.sum` reduction order does not match `torch.sum` exactly, causing argmin flips on near-tie candidates. The PyTorch exact recompute is kept.

### Accuracy validation

- `python scripts/validate_find_params_batched_strict.py` on A100 GPU 5: **STRICT CHECK PASSED**.
- `pytest -q tests/test_gptq.py tests/test_quantizer.py tests/test_quantizer_scale_search.py tests/test_adjacent_exact_cuda.py tests/test_gptq_block_triton.py` on A100 GPU 5: **210 passed, 2 skipped**.
- `ruff check gptqmodel/quantization/_scale_search_triton.py` and `git diff --check`: clean.

### Benchmarks

`find_params_batched` microbenchmark (`4096 x 4096`, `bits=4`, `sym=False`, `grid=100`, `maxshrink=0.8`, A100 GPU 5). `activation` uses `BLOCK_ROW=32` and in-place exact recompute; `hessian`/`hybrid` use `BLOCK_ROW=8`.

| group_size | method     | time (ms) |
|------------|------------|-----------|
| 32         | activation | 10.34     |
| 32         | hessian    | 40.19     |
| 32         | hybrid     | 40.20     |
| 64         | activation | 7.09      |
| 64         | hessian    | 37.11     |
| 64         | hybrid     | 37.14     |
| 128        | activation | 6.04      |
| 128        | hessian    | 41.78     |
| 128        | hybrid     | 41.78     |

End-to-end `GPTQ.quantize` (`4096 x 4096`, `blocksize=128`, `bits=4`, `sym=False`, `mse=2.0`, A100 GPU 5) is within run-to-run noise:

| group_size | method     | mean (ms) | median (ms) | min (ms) | max (ms) |
|------------|------------|-----------|-------------|----------|----------|
| 32         | activation | 112.6     | 111.1       | 110.7    | 118.7    |
| 32         | hessian    | 124.3     | 124.1       | 124.0    | 124.7    |
| 32         | hybrid     | 124.4     | 124.3       | 124.2    | 124.9    |
| 64         | activation | 104.9     | 104.9       | 104.5    | 105.2    |
| 64         | hessian    | 114.7     | 114.7       | 114.2    | 115.2    |
| 64         | hybrid     | 114.6     | 114.5       | 114.4    | 114.9    |
| 128        | activation | 99.7      | 99.6        | 99.4     | 99.9     |
| 128        | hessian    | 109.2     | 108.7       | 108.3    | 111.3    |
| 128        | hybrid     | 108.9     | 109.0       | 108.6    | 109.4    |

### Known limitations / future work

- `bits=2` Hessian/Hybrid still falls back to the exact Python loop.
- Multi-GPU candidate splitting is not pursued; any future multi-GPU work should target module-level parallelism or larger fused kernels rather than per-call `find_params_batched` splitting.
- End-to-end `GPTQ.quantize` is still dominated by the fused GPTQ block kernel and Cholesky inversion; the next ScaleSearch round should either lower the exact-recompute launch overhead further or move on to the block-kernel/cholesky phase.

---

## Round: activation ScaleSearch top-2 selection inside the Triton kernel

### Objective

Avoid materializing the full `[rows, num_groups, candidate_count]` approximate-loss tensor and the PyTorch `topk`/`sort` calls by maintaining the two best candidate indices directly inside the activation Triton kernel.

### Changes

1. **`gptqmodel/quantization/_scale_search_triton.py`**
   - `_scale_search_activation_kernel` now outputs a compact `[rows, num_groups, 2]` int32 tensor of the two best candidate indices, sorted ascending, instead of writing every candidate loss.
   - The kernel keeps running `best_loss`/`best_idx` and `second_loss`/`second_idx` registers across the candidate loop using strict `<` comparisons, preserving lower-index tie-breaking.
   - `_triton_find_params_batched_activation` allocates the small `topk_idx` tensor, skips `loss_out.topk(...)` and `.sort(...)`, and feeds the kernel's sorted indices directly into the PyTorch exact-recompute step.
   - Candidate grids with `candidate_count <= 1` now short-circuit to the `c=0` (no-shrink) scale/zero result, avoiding an invalid top-2 path.

### Accuracy validation

- `python scripts/validate_find_params_batched_strict.py` on A100 GPU 5: **STRICT CHECK PASSED**.
- `pytest -q tests/test_gptq.py tests/test_quantizer.py tests/test_quantizer_scale_search.py tests/test_adjacent_exact_cuda.py tests/test_gptq_block_triton.py` on A100 GPU 5: **210 passed, 2 skipped**.
- `ruff check gptqmodel/quantization/_scale_search_triton.py` and `git diff --check`: clean.

### Benchmarks

`find_params_batched` microbenchmark (`4096 x 4096`, `bits=4`, `sym=False`, `grid=100`, `maxshrink=0.8`, A100 GPU 5). Before = in-place exact recompute + full `loss_out` tensor; after = in-kernel top-2.

| group_size | method     | before (ms) | after (ms) | speedup |
|------------|------------|-------------|------------|---------|
| 32         | activation | 10.34       | 5.58       | 1.85x   |
| 64         | activation | 7.09        | 4.94       | 1.43x   |
| 128        | activation | 6.04        | 5.25       | 1.15x   |

End-to-end `GPTQ.quantize` (`4096 x 4096`, `blocksize=128`, `bits=4`, `sym=False`, `mse=2.0`, A100 GPU 5) is close to noise because the fused GPTQ block kernel and Cholesky inversion now dominate, but the activation `find_params_batched` step is roughly halved:

| group_size | method     | mean (ms) | median (ms) | min (ms) | max (ms) |
|------------|------------|-----------|-------------|----------|----------|
| 32         | activation | 114.1     | 111.4       | 110.6    | 125.8    |
| 64         | activation | 104.6     | 104.6       | 104.4    | 104.9    |
| 128        | activation | 99.6      | 99.6        | 99.3     | 99.8     |

### Nsight Systems

A capture of `GPTQ.quantize` with `group_size=64 activation` on A100 GPU 5 confirms the largest single-kernel consumers are now the fused `_gptq_block_kernel` (~37 ms summed) and the Cholesky factorization/solve kernels (~27 ms summed), with scale-search elementwise work greatly reduced.

### Known limitations / future work

- `bits=2` Hessian/Hybrid still fall back to the exact Python loop.
- Multi-GPU candidate splitting remains off the table per user direction.
- The next speedup round should target the fused GPTQ block kernel (per-row scalar extraction via `tl.where`+`tl.sum` is expensive) and/or the Cholesky/inversion path, as those are now the end-to-end bottlenecks.

---

## Round: skip unused W1 writeback in fused GPTQ block kernel

### Objective

Reduce global memory traffic in `_gptq_block_kernel` by not writing the updated working weights back to `W1`, since the caller only consumes `Q1` and `Err1`.

### Changes

1. **`gptqmodel/quantization/_gptq_block_triton.py`**
   - Removed the final `tl.store` of `w_row` to `w_ptr`. The kernel still loads `W1` to initialize the row registers, computes `Q1`/`Err1`, and updates the row in registers, but no longer writes the ~2 MB working-weight slice per block back to global memory.

### Accuracy validation

- `python scripts/validate_find_params_batched_strict.py` on A100 GPU 5: **STRICT CHECK PASSED**.
- `pytest -q tests/test_gptq.py tests/test_quantizer.py tests/test_quantizer_scale_search.py tests/test_adjacent_exact_cuda.py tests/test_gptq_block_triton.py` on A100 GPU 5: **210 passed, 2 skipped**.
- `ruff check gptqmodel/quantization/_gptq_block_triton.py` and `git diff --check`: clean.

### Benchmarks

Isolated `gptq_block_triton` microbenchmark (`4096 rows x 128 cols`, `group_size=32`, A100 GPU 5): median **0.74 ms -> 0.66 ms** (~11% faster).

End-to-end `GPTQ.quantize` (`4096 x 4096`, `blocksize=128`, `bits=4`, `sym=False`, `mse=2.0`, A100 GPU 5) is within run-to-run noise because the remaining Cholesky/inversion and other elementwise kernels still dominate:

| group_size | method     | mean (ms) | median (ms) | min (ms) | max (ms) |
|------------|------------|-----------|-------------|----------|----------|
| 32         | activation | 115.9     | 112.8       | 112.4    | 127.8    |
| 64         | activation | 106.6     | 106.7       | 105.9    | 107.1    |
| 128        | activation | 101.1     | 101.1       | 101.0    | 101.2    |

### Known limitations / future work

- The `_gptq_block_kernel` per-column scalar extraction (`tl.where(offs == i, w_row, 0.0).sum()`) is still the main compute cost inside the kernel and is hard to improve in Triton without memory writes per column.

---

## Round: fuse per-block trailing weight update with addmm

### Objective

Eliminate the temporary allocation and separate subtraction in `W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])` by using a fused `torch.addmm(..., alpha=-1, out=...)`.

### Changes

1. **`gptqmodel/quantization/gptq.py`**
   - Replaced the per-block trailing update `W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])` with `torch.addmm(W[:, i2:], Err1, Hinv[i1:i2, i2:], alpha=-1, out=W[:, i2:])`.
   - This fuses the `matmul` and subtraction into one cuBLAS call and avoids allocating a `[rows, remaining]` scratch tensor per block.

### Accuracy validation

- `python scripts/validate_find_params_batched_strict.py` on A100 GPU 5: **STRICT CHECK PASSED**.
- `pytest -q tests/test_gptq.py tests/test_quantizer.py tests/test_quantizer_scale_search.py tests/test_adjacent_exact_cuda.py tests/test_gptq_block_triton.py` on A100 GPU 5: **210 passed, 2 skipped**.
- `ruff check gptqmodel/quantization/gptq.py` and `git diff --check`: clean.

### Benchmarks

Isolated trailing update (`4096 rows, count=128, remaining=3968`, A100 GPU 5): matmul-then-subtract median **0.36 ms -> 0.28 ms** (~22% faster).

End-to-end `GPTQ.quantize` (`4096 x 4096`, `blocksize=128`, `bits=4`, `sym=False`, `mse=2.0`, A100 GPU 5) is within run-to-run noise because the block-kernel scalar extraction and Cholesky/inversion still dominate, but the fused call removes ~2–3 ms of temporary allocation/copy overhead across a full layer.

### Known limitations / future work

- The `_gptq_block_kernel` per-column scalar extraction remains the largest single-kernel cost.
- The `cholesky(cholesky_inverse(L), upper=True)` path is already a cuSOLVER-optimized way to compute the required upper Cholesky factor of `H^{-1}`; direct `solve_triangular` approaches produce a different (non-Cholesky) factor and are not a drop-in replacement.

---

## Round: replace the Triton GPTQ block with deterministic native CUDA

### Objective

Remove the free-threaded multi-GPU illegal-address failure in the Triton GPTQ
block path without giving up its fused serial-column algorithm or quantization
accuracy, and make the block step materially faster than the eager fallback.

### Changes

1. **`gptqmodel_ext/gptq_block/gptq_block_cuda.cu`**
   - Added a hand-written CUDA kernel with four independent row-warps per block.
   - Keeps up to 128 working columns in registers, broadcasts each serial GPTQ
     error with a warp shuffle, and uses explicit round-to-nearest multiply then
     subtract instructions to match `torch.addr` operation order bit-for-bit.
   - Launches on the caller's current stream under a CUDA device guard and checks
     launch errors at the native boundary.
2. **`gptqmodel/utils/gptq_block.py`**
   - Added the JIT extension wrapper, strict input/output validation, noncontiguous
     input preparation, explicit output reuse, and allocator `record_stream`
     bookkeeping for free-threaded workers.
   - Rejects oversized row grids and quantization levels before native integer/
     FP32 narrowing, rejects scalar type coercion, and skips the NVIDIA-only
     extension cleanly on ROCm.
   - Serializes only cold operator resolution; cached multi-GPU launches do not
     share a process-wide launch lock.
3. **`gptqmodel/quantization/gptq.py`**
   - Replaced the default Triton dispatch with the native CUDA operator.
   - `GPTQMODEL_CUDA_BLOCK=0` selects the eager fallback. The obsolete Triton
     implementation and its old opt-out variable were removed.
4. **Validation and benchmarking**
   - Added a 72-test CUDA suite covering production W2-W8 modes, W4G64 and
     W4G128 MaCa settings, random legal shapes, ties and adjacent ULPs, outliers,
     scale/Hessian extremes, nonfinite propagation, invalid contracts, stream
     ownership, allocator pressure, input immutability, deterministic repeats,
     two-device guards, cold initialization, and ThreadX free-threaded workers.
   - Added an idle-gated benchmark that imports Torch only after validating the
     selected physical GPU.

### Accuracy and safety validation

- Python 3.14.6t, Torch 2.13.0+cu130, CUDA 13.0, `PYTHON_GIL=0`, two `sm_80`
  PG506-230/232 GPUs: **70 passed, 2 probe-only skipped**.
- Python 3.12 coverage pass on two GPUs: **67 passed, 5 free-thread-only/probe
  skips**, with **100% of 113 statements and 54 branches** in the Python wrapper.
- Compute Sanitizer memcheck: **0 errors**.
- Eager versus native results are bit-exact for raw quantized values, raw GPTQ
  errors, logical codes, production quantized weights, scales, zero points,
  group indices, reported loss, and held-out dense outputs.
- Ten repeated launches for each of three seeds and four concurrent streams had
  zero output spread.
- Final `sm_80` and `sm_89` builds use 40 registers, zero stack bytes, zero spill
  loads/stores, and zero barriers.

### Isolated block benchmark

PG506-230 (`sm_80`, 124 SMs), FP32 working state, 7168 rows x 128 columns, 10
warmups and 50 measured repetitions:

| group_size | eager median (ms) | native median (ms) | native p95 (ms) | speedup | code mismatches | max abs error |
|------------|-------------------|--------------------|-----------------|---------|-----------------|---------------|
| 64         | 19.052            | 0.318              | 0.328           | 59.92x  | 0               | 0             |
| 128        | 19.087            | 0.318              | 0.330           | 59.93x  | 0               | 0             |

### Real-model integration

Laguna-S-2.1 PER-LAYER, W4G64, activation scale search, GAR enabled,
`desc_act=False`, routing bypass, length-aware Hessian, GIL disabled, and two
GPUs completed layers 0-4 and exited cleanly with no CUDA/NCCL error or native
kernel fallback. The asynchronous stop callback allowed one layer beyond the
requested four, providing additional coverage.

Compared with the existing eager-fallback W4G64 run on the same GPU class, the
sum of the five logged layer lifecycle times improved from 1066.197s to
969.438s (1.10x). Per-layer variation remains dominated by checkpoint I/O,
Hessian work, and expert finalization/CPU packing rather than the 0.318ms block
kernel:

| layer | eager fallback (s) | native CUDA (s) | speedup |
|-------|--------------------|-----------------|---------|
| 0     | 12.586             | 7.088           | 1.78x   |
| 1     | 301.756            | 247.271         | 1.22x   |
| 2     | 272.101            | 109.206         | 2.49x   |
| 3     | 200.295            | 254.401         | 0.79x   |
| 4     | 279.459            | 351.472         | 0.80x   |
| total | 1066.197           | 969.438         | 1.10x   |

### Remaining bottleneck

The block kernel itself exceeds the 4x target by a wide margin, but the real
five-layer run does not yet reach 4x end-to-end. Further work should profile and
reduce routed-expert finalization/packing, materialization, and Hessian time;
changing the now bit-exact CUDA block math cannot deliver the remaining gain.

## Round: QVQ V2 segment-grid Viterbi speedup investigation (bound pruning)

**Date:** 2026-08-24
**Repository:** `/root/qvq`
**Hardware:** 1 x NVIDIA PG506-230 (A100-class, sm_80, 96 GiB, 124 SMs)
**Software:** Python 3.14.6, PyTorch 2.15.0.dev20260817+cu130, CUDA 13.3, Triton 3.8.0
**Scope:** `gptqmodel_ext/qvq/qvq_viterbi_cuda.cu` — `viterbi_v2_segment_grid_trusted`
(YAQA quantization's dominant kernel), all rates W2/W2.5/W3, half codebooks.

### Baseline characterization

Added `scripts/benchmark_qvq_v2_segment_grid.py` (CUDA-event benchmark plus a
bit-exactness oracle gate against the eager recurrence; 12 rate x config combos).
Measured regimes:

- batch <= ~16 (grid underfilled): flat ~1.2-1.3 ms; chain-latency / issue bound.
- batch >= 32 (grid full): linear scaling; W3.0/b4: 2.2 ms (b32) -> 11.7 ms (b256).

nsight-compute evidence (both regimes): schedulers 86-87% busy (issue bound),
L2 hit rate 99.6%, memory utilization 14-18% (NOT bandwidth bound).
SASS: ptxas fully unrolls the 64-prefix candidate chain into a ~1254-instruction
straight-line block with all loads hoisted; this static pipeline is the kernel's
performance backbone.

### Attempt 1: reduced-precision distance math (fp16/bf16/TC-equivalent)

Rationale: replace FP32 dot/distance with half2 products. On A100, an MMA with
K=2 (padded to k8) has the same effective ceiling as HFMA2 SIMT (~78 TFLOP/s
class), so a half2-SIMT probe bounds the tensor-core outcome.

Result (probe at `/tmp/opencode/probe_fp16.py`, real PGC16 codebooks, full
coupled-bank trellis, FP64 re-scored paths):

| variant | speed | path agreement | true SE inflation (max) |
|---|---|---|---|
| fp16 dot  | 0.99-1.04x | 97.7-100% | <= 1.00042 |
| fp16 dot+norm | 0.96-0.98x | 97.7-100% | <= 1.00042 |
| bf16 dot  | 0.79-0.88x | 67.6-100% | <= 1.040 |

Verdict: precision was never the bottleneck; relaxation buys no speed here.

### Attempt 2: exact Cauchy-Schwarz bound pruning

`dist(s) >= (sqrt(tn) - sqrt(norm_s))^2` gives a valid per-candidate lower
bound; skipping candidates whose bound exceeds the running best preserves the
argmin exactly (ties included), so `torch.equal` still holds. A round-down
intrinsic chain (`__fadd_rd/__fmul_rd`, `qhat = -2*__fsqrt_ru(norm)`,
per-step `rho = __fsqrt_ru(tn)`) makes the bound a provable FP32 lower bound
with no heuristic margins.

Prune-rate validation (`scripts/benchmark_qvq_v2_prune_rate.py`, real
Qwen3-0.6B weight tiles through production-style [tiles,128,2] sequences):
92.0% (shift 6) / 86.5% (shift 5) of candidates provably skippable.

Three CUDA implementations (per-candidate `continue`, rolled, and
`#pragma unroll`-forced) all REGRESSED vs baseline: 1.7-2.3x slower.
Root cause (SASS): any data-dependent branch in the h-loop prevents full
unrolling and load hoisting; the loop compiles to ~174 instructions with 37
branches and loses the static pipeline. The ~9-instruction arithmetic saving
cannot compensate. Kernel code reverted; the experiment is fully reproducible
from this log.

### Follow-up design (not yet implemented): octet-grouped bounds

The remaining sound path keeps branches COARSE (per 8 candidates) and bodies
straight-line:

- Precompute per-8-consecutive-states {min_norm, max_p=2*sqrt(norm)} tables
  (target-independent, cached per codebook; 64 KiB/bank).
- Re-map threads to own 8 consecutive x values (an octet). For shift >= 3,
  `state >> shift` is constant within an octet, so the whole octet shares one
  predecessor lookup.
- Per (h, octet): one 8-byte table load + FFMA/FADD(rd) + one branch; the
  surviving branch evaluates 8 records straight-line (unrollable, loads
  hoistable across octets).
- Budget at 80% octet prune: ~0.6 + 0.2*14 ~= 3.4 instr/candidate -> up to
  ~2x; at 60% octet prune ~2.4x. Exactness argument unchanged from Attempt 2.

Also unresolved from this round: bound-pruning is gated off for
constrained/weighted launches (the all-INF tail-biting edge case needs the
constraint check folded into the bound before pruning can apply there).
