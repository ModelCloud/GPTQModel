# QVQ norm-rank Qwen3-8B benchmark

This artifact preserves the complete acceptance measurement behind PR #45.
Times are CUDA-event medians in microseconds after 10 warmups and 50 measured
iterations. `pristine` is the unchanged grid recurrence selected by the disable
environment variable; `enabled` is the W2.5/W3 norm-rank path. All values below
are measured values copied from the PR #45 run, not values derived from the
rounded speedups.

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
