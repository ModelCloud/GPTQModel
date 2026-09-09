# W4 signed GPTQ baseline: full Platinum evaluation

State: complete; process exited 0. Correct: 460/1209 (38.047974%).
Invalid numeric answers: 0/1209. This is the matched staged-path signed
GPTQ initializer with GSQ disabled, not package-default true-sequential GPTQ.
128 NM calibration documents, seed 7, W4/group128. Full effective evaluation
settings, model/source/dataset hashes and runtime versions are in run.json;
all predictions, scores and exact prompts are in raw.json.
QVQ commit: 5cc76af5c715f6089663d00e5cc5cefb61445afd; ZML Ultra not used.
Evaluation elapsed: 217.725097 seconds.

```sh
PYTHONPATH=.:/root/polly-work/Evalution CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 GPTQ_CACHE_DEQUANTIZED_WEIGHTS=1 OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python scripts/evaluate_gsq_gsm8k.py --model artifacts/gsq-staged/full-model-w4-nm128-signed-baseline-v1/model --output artifacts/gsq-staged/gsm8k-platinum-w4-nm128-signed-v1/baseline-full --dataset artifacts/gsq-staged/gsm8k-platinum-v1/dataset --arm gptq-signed-w4-nm128 --batch-size 32
```

Durable log: evaluation.log; SHA256 6bd9b1ef466c8fdfe4a8c216accf6492770f58d4a8175a652de5dad16f4dbb4c.
Raw result SHA256: a49cba24a69ad975c50a75631c358705fb0a31e79c28aec490e69d966a0fe2a1.
The GSQ arm completed at 417/1209. Exact three-arm prompt/token matching passed.
Paired GSQ-minus-baseline interval: [-6.3689, -0.7444] percentage points.
Dense reference: 593/1209. See ../comparison.md.
