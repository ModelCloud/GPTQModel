# Real FP8 QKV reconstruction run

run_id: `real-qkv-seed7-v2`; status: completed. This is a selected-layer
experiment, not a complete model artifact. QVQ source commit: `72d2bf7ef21713e34e7c2ef7ae078ee364086865` plus the archived uncommitted sources bound by SHA-256 in provenance.json.
ZML: N/A, not used. Tokenization was not rerun: exact stored token IDs and
activation rows are in documents.json and the hash-bound source fixtures.

Working directory: `/root/polly-work/qvq-gsq`.

```sh
PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python -m scripts.validate_gsq_fp8_layers --output artifacts/gsq-fp8/real-qkv-seed7-v2
```

GPU identity: 0, 00000000:DE:00.0, GPU-737e2423-874a-23a4-1126-dfbe3e77c294, NVIDIA PG506-230, 0, 0. Three idle samples passed.
Torch: 2.15.0.dev20260817+cu130; CUDA: 13.0. FP32 inputs,
FP32 dequantized matmul; TF32 disabled. On this SM80 GPU the FP8 module uses
CPU dequantization followed by CUDA matmul, not native FP8 tensor-core math.
No custom kernel or compilation changes. No timing-performance claim.

FP8 E4M3FN, row inverse scales, frozen scales. Arms: ordinary baseline,
weight-only GSQ, activation-calibrated GSQ. GSQ: 100 steps, 3 candidates,
seed 7, Adam learning rate 0.1, geometric temperature 1.0 to 0.1,
max decoded candidate bytes 1073741824, no module filter. Training uses all
16 documents with sqrt(source-weight) activation scaling (YAQA 1.25, NM 1.0).
All 32 held-out documents are unweighted and never select a checkpoint.

| Projection | Arm | Held-out MSE | Relative change | Changed bytes |
|---|---|---:|---:|---:|
| model.layers.0.self_attn.q_proj | baseline | 0.000169367949933 | +0.000000% | 0 |
| model.layers.0.self_attn.q_proj | gsq_weight | 0.000169367949933 | +0.000000% | 0 |
| model.layers.0.self_attn.q_proj | gsq_calibrated | 0.000151717893994 | -10.421131% | 1882 |
| model.layers.0.self_attn.k_proj | baseline | 0.000270709680237 | +0.000000% | 0 |
| model.layers.0.self_attn.k_proj | gsq_weight | 0.000270709680237 | +0.000000% | 0 |
| model.layers.0.self_attn.k_proj | gsq_calibrated | 0.000247939052742 | -8.411457% | 2192 |
| model.layers.0.self_attn.v_proj | baseline | 3.89517102274e-06 | +0.000000% | 0 |
| model.layers.0.self_attn.v_proj | gsq_weight | 3.89517102274e-06 | +0.000000% | 0 |
| model.layers.0.self_attn.v_proj | gsq_calibrated | 3.84474419167e-06 | -1.294599% | 112 |

All 288 FP32 runtime checks passed, with exact zero drift against the same
exported FP32 decoded weights. This checks dispatch/storage consistency, not
FP16/BF16, native FP8 hardware, graph replay, or a complete-model lifecycle.
Calibrated GSQ lowered local held-out MSE in all three projections. Weight-only
GSQ retained all baseline bytes. No paired uncertainty, final logits, KL,
Top-K agreement or task accuracy was measured; no recovery promotion is made.

The raw report retains every per-document runtime check; MSE is summed squared
output error divided by output element count across 6367 held-out tokens.
The runner did not store per-document quality SSE or output predictions, so
bootstrap uncertainty cannot be recovered from this report alone. That evidence
must be added before broader quality conclusions. Execution log is preserved
in execution.log.gz; source/config/data hashes are in provenance.json.
