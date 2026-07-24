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

### AdjacentExact CUDA exact kernel

Active-decision timing (all `size` decisions non-zero, `warps=0`) on a single A100:

| active decisions | before (ms) | after (ms) | speedup |
|------------------|-------------|------------|---------|
| 28               | ~35         | ~23        | 1.52x   |
| 30               | ~117        | ~90        | 1.30x   |
| 32               | ~466        | ~355       | 1.31x   |

Nsight confirms that >99.9% of GPU time for the exact-solver microbenchmark is in `adjacent_exact_candidates_kernel`. `ncu --section SpeedOfLight` reports the kernel is **compute-bound** (SM throughput ~78%, memory throughput ~25%, DRAM 0%), with the ALU/integer pipeline dominating. The Gray-code shortcut removes one `__ffs` and several integer ops per iteration; the set-bit rebase removes empty toggles; raising `kRebaseInterval` from 64 to 1024 amortizes the `O(size^2)` rebase work. The biggest win is switching the incremental Gray-code `field`/`energy` accumulation to **FP32** (`acc_t = float`) while recomputing the final `candidate_costs` from the original FP64 QUBO using the selected best state. FP32 is ~16x faster than FP64 on A100 and, with the 1024-state rebase safeguard, keeps the strict exact-solver tests passing with zero state/cost diffs.

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

- `pytest -q tests/test_gptq.py tests/test_adjacent_exact_cuda.py` on GPUs 5,6: **41 passed, 2 skipped**
- `pytest -q tests/test_adjacent_exact_cuda.py` with FP32 incremental accumulator on GPU 5: **20 passed**
- `python scripts/validate_find_params_batched_quick.py`: all group_size 32/64/128 and activation/hessian/hybrid scale/zero outputs match per-group `find_params` exactly.
- `python scripts/validate_find_params_batched_strict.py`: exhaustive sweep across rows `[128, 512, 4096]`, columns `[128, 256, 512]`, group sizes `32/64/128`, `sym={True,False}`, bits `{2,4,8}`, methods `activation/hessian/hybrid`, seeds `42/123/999` — **STRICT CHECK PASSED** (all zero diff, scale diff `0.000`) after per-group Hessian normalization fix.
- `ruff check` on modified Python files: **clean** (also fixed two pre-existing bare `except` clauses in `gptq.py`).

## Known limitations / future work

- **Group size 1** was skipped at user request; it is not a supported quantization option.
- The `find_params_batched` path currently targets `act_group_aware=True` grouped search. The legacy ungrouped / full-tensor path is untouched and still uses the per-call `find_params`.
- AdjacentExact CUDA exact solver is still exponential in the active-decision count. The shared-memory optimization lowers memory overhead but does not change asymptotic complexity; for components larger than ~32 decisions the existing branch-and-bound solver remains the practical fallback.
- A custom Triton megakernel for `find_params_batched` was not implemented because the current vectorized PyTorch path already fuses the grid search into a small set of `elementwise`/`sgemm`/`reduce` kernels and was fast enough for the requested group sizes. A fused Triton kernel could be explored if the grid search becomes the dominant bottleneck.
