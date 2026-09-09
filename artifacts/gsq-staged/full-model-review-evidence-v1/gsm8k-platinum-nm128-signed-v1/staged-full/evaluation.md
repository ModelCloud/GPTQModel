# NM128 five-epoch staged GSQ: full Platinum evaluation

State: complete; process exited 0. Correct: 20/1209 (1.65426%). Invalid numeric answers: 482/1209.
Matched signed GPTQ initializer; W2/group128, 128 calibration documents, seed 7,
five attention/MLP epochs, batch64/micro16, Q/K 2000 updates each.
QVQ commit: 5cc76af5c715f6089663d00e5cc5cefb61445afd. Exact dirty quantization sources are preserved in the model run.
ZML Ultra: not used. Effective evaluation settings, model hashes, dataset hashes and
source hash are in run.json; complete predictions and prompts are in raw.json.
Elapsed evaluation: 217.278768 seconds.

```sh
PYTHONPATH=.:/root/polly-work/Evalution CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 GPTQ_CACHE_DEQUANTIZED_WEIGHTS=1 OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python scripts/evaluate_gsq_gsm8k.py --model artifacts/gsq-staged/full-model-w2-nm128-signed-staged-v1/model --output artifacts/gsq-staged/gsm8k-platinum-nm128-signed-v1/staged-full --dataset artifacts/gsq-staged/gsm8k-platinum-v1/dataset --arm gptq-signed-staged-gsq-w2-nm128 --batch-size 32
```

Evaluation log SHA256: fc1bb85e79b128d4b1c8cbd174191f1eed79e93eaddfd3735e26fa8fa9529fe8
Raw result SHA256: 0990825e411f998914d0e8267ef7e38b220ab2173a1a8d9dfdea1508eee299ad

This small task gain over a collapsed baseline does not establish useful recovery.
Dense reference: 593/1209. All five held-out final-logit metrics regress with GSQ.
See the matched comparison for exact prompt/input-ID verification and paired uncertainty.
