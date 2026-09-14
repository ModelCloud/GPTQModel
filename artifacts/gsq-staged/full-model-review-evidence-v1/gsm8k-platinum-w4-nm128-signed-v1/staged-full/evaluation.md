# W4 staged GSQ: full Platinum evaluation

Complete; exit 0. Correct: 417/1209 (34.491315%); invalid: 0.
Matched baseline: 460/1209; dense: 593/1209. GSQ minus baseline: -3.5567 percentage
points, paired 95% interval [-6.3689, -0.7444]. Exact prompts, targets and input
IDs match all three arms. This is a clear task regression, consistent with all
five held-out logit metrics. No recovery/default promotion is justified.

Run/arm: staged-full / gptq-signed-staged-gsq-w4-nm128. QVQ: 5cc76af5c715f6089663d00e5cc5cefb61445afd; ZML Ultra not used.
Model: /root/polly-work/qvq-gsq/artifacts/gsq-staged/full-model-w4-nm128-signed-staged-v1/model. Evaluation settings, source/dataset hashes and versions:
run.json. Full predictions/prompts/scores: raw.json. Calibration: 128 NM documents,
seed 7; W4/group128, signed GPTQ prior and five staged GSQ epochs.

```sh
PYTHONPATH=.:/root/polly-work/Evalution CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 GPTQ_CACHE_DEQUANTIZED_WEIGHTS=1 OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python scripts/evaluate_gsq_gsm8k.py --model artifacts/gsq-staged/full-model-w4-nm128-signed-staged-v1/model --output artifacts/gsq-staged/gsm8k-platinum-w4-nm128-signed-v1/staged-full --dataset artifacts/gsq-staged/gsm8k-platinum-v1/dataset --arm gptq-signed-staged-gsq-w4-nm128 --batch-size 32
```

Elapsed: 201.685559s. Log SHA256: 0be699e9db98bcc5c81d93c6778119061d6c57efafe6cccbe1c97e3c6629f8c4.
Raw SHA256: e9580828d2624082a2d7d06aaa34a987d9d3aa111b397bc897da54b648784242. Paired audit: ../comparison.json and ../comparison.md.
