# W2 late-MLP propagation evaluation

Complete: real Llama 3.2 1B Instruct block 0, W2/group128, seed7, 32 locked held-out documents (6,367 tokens), with all other blocks canonical F6 FP32. Calibration used 16 disjoint documents; these held-out rows were reused across earlier experiments, so broader disjoint confirmation remains required.

| Arm | KL | Logit MSE | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| f6 | 0.108126780 | 0.440472766 | 85.4877% | 83.0313% | 82.9260% |
| baseline | 0.665093356 | 2.138095027 | 67.5986% | 64.0019% | 64.4919% |
| staged | 0.506079835 | 1.755596264 | 71.9020% | 68.7828% | 68.6870% |
| baseline_packed | 0.664967634 | 2.137885386 | 67.5986% | 64.0050% | 64.4888% |
| staged_packed | 0.506049827 | 1.755513164 | 71.9177% | 68.7922% | 68.6948% |

Both canonical and matched packed comparisons classify all five metrics as clear positive using 10,000 paired document bootstrap draws, seed7, token-weighted means, 95% percentile intervals. See `bootstrap-canonical.json` and `bootstrap-packed.json`. Packing uses portable TorchLinear, FP16 stored scales, and block state-dict disk reload. This does not establish optimized native backend correctness or full-model export.

Execution CLI, exact source/model binding and log path are in `model_run.md` and `provenance.json`. GPU: NVIDIA PG506-230, UUID GPU-737e2423-874a-23a4-1126-dfbe3e77c294, PCI DE:00.0; exclusive allocator lease with three idle samples. Torch 2.15.0.dev20260817+cu130, CUDA 13.0. Eager, cache disabled, FP32, TF32 disabled.

Analysis commands:

```sh
PYTHONPATH=. /root/venv-py3.14t/bin/python scripts/analyze_gsq_fp8_propagation.py artifacts/gsq-staged/propagation-w2-late-mlp-v2/report.json --arms staged --baseline baseline --output artifacts/gsq-staged/propagation-w2-late-mlp-v2/bootstrap-canonical.json
PYTHONPATH=. /root/venv-py3.14t/bin/python scripts/analyze_gsq_fp8_propagation.py artifacts/gsq-staged/propagation-w2-late-mlp-v2/report.json --arms staged_packed --baseline baseline_packed --output artifacts/gsq-staged/propagation-w2-late-mlp-v2/bootstrap-packed.json
```

Analysis report SHA256: `65b708b710e56d68e796414c8cb7ec34d718d3cb48fea40a72495682681d2112`.
