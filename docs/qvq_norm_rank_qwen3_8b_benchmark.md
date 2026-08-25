# QVQ norm-rank Qwen3-8B benchmark

This artifact preserves the complete acceptance measurement behind PR #45.
Times are CUDA-event medians in microseconds after 10 warmups and 50 measured
iterations. `pristine` is the unchanged grid recurrence selected by the disable
environment variable; `enabled` is the W2.5/W3 norm-rank path. All values below
are measured values copied from the PR #45 run, not values derived from the
rounded speedups.

The user-facing policy that selects this path per run is documented separately
in [`qvq_viterbi_pruning_config.md`](qvq_viterbi_pruning_config.md); the
deprecated `GPTQMODEL_QVQ_DISABLE_OCTET_GRID` variable used for the `pristine`
column above remains an A/B escape hatch under `viterbi_pruning.mode="auto"`
only.

## Environment

- NVIDIA PG506-230, compute capability 8.0 (`sm_80`), 124 SMs
- CUDA 13.3 (`nvcc V13.3`), `-gencode arch=compute_80,code=sm_80`
- PyTorch 2.13.0+cu130
- Python 3.14.7t
- Model: `/monster/data/model/Qwen3-8B`
- Input: real Qwen3-8B weight tiles produced by the benchmark script

Exact commands, run from the repository root after `source /root/qvq_env.sh`:

```bash
$PY scripts/benchmark_qvq_v2_segment_grid.py --model /monster/data/model/Qwen3-8B --rates 2.0 2.5 3.0 --batches 8 16 32 64 128 256 --warmup 10 --iters 50
GPTQMODEL_QVQ_DISABLE_OCTET_GRID=1 $PY scripts/benchmark_qvq_v2_segment_grid.py --model /monster/data/model/Qwen3-8B --rates 2.0 2.5 3.0 --batches 8 16 32 64 128 256 --warmup 10 --iters 50
```

## Full measured table

| rate | banks | batch | pristine (us) | enabled (us) | speedup |
|---|---:|---:|---:|---:|---:|
| W2.5 | 2 | 8 | 1112.6 | 1116.2 | 1.00x |
| W2.5 | 2 | 16 | 1219.6 | 951.8 | 1.28x |
| W2.5 | 2 | 32 | 1233.9 | 956.4 | 1.29x |
| W2.5 | 2 | 64 | 2257.9 | 1463.3 | 1.54x |
| W2.5 | 2 | 128 | 3668.0 | 2223.1 | 1.65x |
| W2.5 | 2 | 256 | 6375.4 | 3562.5 | 1.79x |
| W2.5 | 4 | 8 | 1236.0 | 956.4 | 1.29x |
| W2.5 | 4 | 16 | 1271.8 | 968.7 | 1.31x |
| W2.5 | 4 | 32 | 2262.0 | 1480.7 | 1.53x |
| W2.5 | 4 | 64 | 3622.4 | 2161.2 | 1.68x |
| W2.5 | 4 | 128 | 6315.5 | 3406.3 | 1.85x |
| W2.5 | 4 | 256 | 11630.1 | 5924.9 | 1.96x |
| W3 | 2 | 8 | 1189.9 | 520.2 | 2.29x |
| W3 | 2 | 16 | 1194.0 | 527.4 | 2.26x |
| W3 | 2 | 32 | 1237.0 | 532.5 | 2.32x |
| W3 | 2 | 64 | 2230.3 | 920.6 | 2.42x |
| W3 | 2 | 128 | 3690.5 | 1369.1 | 2.70x |
| W3 | 2 | 256 | 6323.2 | 2262.5 | 2.79x |
| W3 | 4 | 8 | 1209.3 | 535.6 | 2.26x |
| W3 | 4 | 16 | 1256.4 | 540.7 | 2.32x |
| W3 | 4 | 32 | 2213.9 | 914.4 | 2.42x |
| W3 | 4 | 64 | 3602.4 | 1338.4 | 2.69x |
| W3 | 4 | 128 | 6300.7 | 2191.4 | 2.88x |
| W3 | 4 | 256 | 11500.0 | 3872.8 | 2.97x |

## Final review validation

The final review fix adds one block barrier after the three shared
`group_min` buffers are cleared and before segment-zero initialization or
later-segment `atomicMin` population. The recurrence loop still has exactly
one barrier per step. On the one-GPU acceptance host (PG506-230, `sm_80`), the
CUDA 13.3 rebuild and these checks passed:

- Compute Sanitizer 2026.2.1 racecheck, using batch nine so W2.5/bank2 takes
  the shipped norm-rank grid rather than its small-batch cooperative sibling:
  `PYTHONPATH=/root/qvq-pr45-review-fixes compute-sanitizer --tool racecheck
  --error-exitcode 99 $PY scripts/check_qvq_norm_rank_race.py`. One multi-segment launch
  each at W2.5/bank2/segment16, W2.5/bank4/segment32, W3/bank2/segment16, and
  W3/bank4/segment32 completed with `0 hazards displayed (0 errors, 0
  warnings)`.
- The independent oracle command reported 12/12 configurations `EXACT`.
- `pytest tests/test_qvq_cuda.py -k "viterbi or segment"` reported `396
  passed, 1 skipped, 764 deselected`. The four new cases repeatedly exercise
  the first step of every later segment in the shipped W2.5/W3 bank matrix.
- A batch-256 Qwen3-8B rerun measured W3/bank2 as 6307.3 us pristine versus
  2274.3 us enabled (2.77x), and W3/bank4 as 11654.7 us versus 3891.2 us
  (3.00x).

### Mixed-device evidence boundary

Device-safe event destruction is closed by code inspection: the single event
destruction helper uses RAII `c10::cuda::CUDAGuard`, and both the norm cache
and norm-rank cache call it with the evicted entry's owning device. The
dedicated
`test_qvq_cuda_mixed_device_cache_eviction_destroys_events_on_owner_device`
exists, but it was not executed on this one-GPU acceptance host: its targeted
run accurately reported `1 skipped` because it requires two visible GPUs.

That test deliberately fills each cache with device-0 entries and triggers
eviction while device 1 is current. It asserts that device 1 remains current
after each cross-device eviction and that the caller's original current device
is restored, then checks `norm_cache_size() <= 32` and
`norm_rank_cache_size() <= 8`.

The historical focused result `392 passed, 1 skipped` did **not** exercise or
skip this mixed-device test: the expression `-k "viterbi or segment"`
deselected it by name. Collect-only verification after the four new focused
regressions similarly reported `397 selected, 764 deselected` and did not list
the mixed-device node; the current focused result's one skip remains the
pre-existing selected free-threaded multi-device test. A full collection did
list the dedicated mixed-device node among 1161 tests, and its explicit
targeted run reported the one-GPU skip above. No two-GPU execution is claimed.
