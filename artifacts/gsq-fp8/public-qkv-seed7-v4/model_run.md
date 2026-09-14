# Public calibrated FP8 QKV validation

run_id: public-qkv-seed7-v4; arm_id: gsq_calibrated_public; status: completed.
Stored artifact path: /root/polly-work/qvq-gsq/artifacts/gsq-fp8/public-qkv-seed7-v4. Selected QKV payloads only;
this is not a complete model snapshot. Source: Llama-3.2-1B-Instruct at
/monster/data/model/Llama-3.2-1B-Instruct, including its local tokenizer.
QVQ commit: b7c7032df8ba4c2f7670ca3d7865a8b84e4daf23 plus SHA-bound archived uncommitted sources.
ZML: N/A, not used. Source/config/data hashes are inline in provenance.json;
exact selected token IDs and source names are retained in documents.json.

```sh
# cwd: /root/polly-work/qvq-gsq
PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python -m scripts.validate_gsq_fp8_public --output artifacts/gsq-fp8/public-qkv-seed7-v4
```

Full effective config is quantize_config.json. FP8 E4M3FN row inverse scales,
GSQ calibrated enabled, frozen scales, seed 7, 100 Adam steps, 3 candidates,
learning rate 0.1, temperature 1.0 to 0.1, no smoothing, disk offload disabled.
Dynamic exclusions select exactly block-0 Q/K/V. Other model weights stay dense.
Public dataset preparation, capture, fitting, replay and finalization all ran.
Batch 1, original document order, eager attention, FP32 model and inputs,
no generation, no cache or CUDA graphs. Sixteen calibration documents contain
3767 valid tokens; source weights YAQA 1.25/NM 1.0 yield 4279 weighted tokens.
All three projection diagnostics assert both counts. Thirty-two disjoint held-out
documents (6367 tokens) only evaluate already-selected payloads.

GPU 0, PG506-230 SM80, PCI DE:00.0,
UUID GPU-737e2423-874a-23a4-1126-dfbe3e77c294. Three idle samples passed.
Torch 2.15.0.dev20260817+cu130, CUDA 13.0; TF32 disabled.
FP8 storage uses CPU decoding / ordinary matmul on this architecture;
no native FP8 tensor-core or graph claim. Quantization log and full stdout/stderr
are quantization-log.json and execution.log (also execution.log.gz).

| Projection | Held-out output MSE | Documents |
|---|---:|---:|
| model.layers.0.self_attn.q_proj | 0.0001517179508014 | 32 |
| model.layers.0.self_attn.k_proj | 0.00024793910440398 | 32 |
| model.layers.0.self_attn.v_proj | 3.8447442287709e-06 | 32 |

Each saved payload strictly reloads into TorchFP8Linear. report.json preserves
per-document squared-error sums and element denominators. Relative to the prior
ordinary-FP8 fixture baseline on identical held-out inputs: Q -10.4211%,
K -8.4114%, V -1.2946%. This confirms local public-route behavior, not final-model
recovery; bytewise payload comparison now binds these exports to the earlier propagation run.
No new task accuracy, KL or Top-K result is claimed here.

Earlier v1/v2 failed selection (no modules quantized). v3 completed with metadata
loss and was unweighted (3767 weighted tokens); it is not the weighted comparison.
Their logs/reports remain in sibling artifact directories without overwriting.


## Binding to prior propagation

payload-comparison.json proves that all Q/K/V weight bytes and inverse-scale
bytes equal real-qkv-seed7-v2/gsq_calibrated. The stored key sets, E4M3FN format,
row inverse-scale semantics, and FP8 runtime source SHA also match. Serialized
.pt file hashes differ because container metadata is not the tensor payload.
Thus the propagation-seed7-v1 calibrated-arm result applies to these exact
weights under that same eager FP32 evaluation contract. This is a verified
artifact identity binding, not a newly executed propagation run. The earlier
mixed KL/Top-10 result remains unchanged; no promotion is claimed.
