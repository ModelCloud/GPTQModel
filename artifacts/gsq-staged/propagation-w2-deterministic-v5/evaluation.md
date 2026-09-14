# Deterministic W2 packed propagation

Complete. Real Llama 3.2 1B Instruct block 0 W2/group128, trained with deterministic algorithms and seed7. The other decoder blocks use canonical F6 FP32. 32 held-out documents / 6,367 tokens; training used 16 disjoint documents. Rows were reused in prior experiments, so broader disjoint confirmation remains necessary.

| Arm | KL | Logit MSE | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| f6 | 0.108126780 | 0.440472766 | 85.4877% | 83.0313% | 82.9260% |
| baseline | 0.665093356 | 2.138095027 | 67.5986% | 64.0019% | 64.4919% |
| staged | 0.521752645 | 1.801575064 | 70.8340% | 68.2519% | 68.3053% |
| baseline_packed | 0.664967634 | 2.137885386 | 67.5986% | 64.0050% | 64.4888% |
| staged_packed | 0.521725102 | 1.801593337 | 70.8497% | 68.2582% | 68.2959% |

All five metrics favor GSQ under matched canonical and packed comparisons; paired 10,000-draw document bootstrap, seed7, token-weighted means, 95% percentile intervals. See `bootstrap-canonical.json` and `bootstrap-packed.json`. Compared with the earlier nondeterministic trained payload, local MSE is lower but final-logit KL is higher; this reinforces that local loss is not the promotion criterion. Both improve against their matched GPTQ baseline.

Portable TorchLinear block execution includes FP16 scale storage and state-dict disk reload. This is not a full portable checkpoint or optimized native GPTQ backend validation.

```sh
PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python scripts/validate_gsq_staged_propagation.py --packed --layers artifacts/gsq-staged/llama-block0-w2-seed7-config-deterministic-v5 --inputs artifacts/gsq-scalar/gptq-w4-seed7-v2 --output artifacts/gsq-staged/propagation-w2-deterministic-v5
```

Log: `../logs/propagation-w2-deterministic-v5.log.gz`. Model/source hashes and commit are in `provenance.json`; GPU/runtime inventory is in `report.json`. The final-logit forward uses FP32 with TF32 disabled and cache-free eager attention; deterministic-algorithm mode is a training control in the source artifact.

Analysis:

```sh
PYTHONPATH=. /root/venv-py3.14t/bin/python scripts/analyze_gsq_fp8_propagation.py artifacts/gsq-staged/propagation-w2-deterministic-v5/report.json --arms staged --baseline baseline --output artifacts/gsq-staged/propagation-w2-deterministic-v5/bootstrap-canonical.json
PYTHONPATH=. /root/venv-py3.14t/bin/python scripts/analyze_gsq_fp8_propagation.py artifacts/gsq-staged/propagation-w2-deterministic-v5/report.json --arms staged_packed --baseline baseline_packed --output artifacts/gsq-staged/propagation-w2-deterministic-v5/bootstrap-packed.json
```

Report SHA256: `576cf8dec88033ee2d1fdcd18c199008c9777d4c3549b35d55a4e86637afdcb8`.
