# NM128 signed GPTQ baseline: full Platinum evaluation

State: complete; process exited 0. Correct: 0/1209; invalid numeric answers: 1170/1209.
This is the experimental staged-path signed W2/group128 initializer with GSQ disabled,
not package-default true-sequential GPTQ. All 128 calibration documents are fixed at seed 7.

QVQ commit: 5cc76af5c715f6089663d00e5cc5cefb61445afd; dirty quantization sources are preserved in the model run.
ZML Ultra: not used. Torch backend, FP16, eager attention, batch 32, eight-shot CoT,
greedy generation with 256 new tokens. Full effective configuration, model and dataset
hashes are in run.json; all predictions, prompts and scores are in raw.json.
Elapsed evaluation: 229.627893 seconds.

Command (allocator wrapper used):

```sh
PYTHONPATH=.:/root/polly-work/Evalution CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 GPTQ_CACHE_DEQUANTIZED_WEIGHTS=1 OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python scripts/evaluate_gsq_gsm8k.py --model artifacts/gsq-staged/full-model-w2-nm128-signed-baseline-v1/model --output artifacts/gsq-staged/gsm8k-platinum-nm128-signed-v1/baseline-full --dataset artifacts/gsq-staged/gsm8k-platinum-v1/dataset --arm gptq-signed-w2-nm128 --batch-size 32
```

Durable evaluation log SHA256: db935fb23636cdbda3e4fd3674556501abea87016df79edf2cd3596fe535f297
Raw result SHA256: 5f3a03b37c203bd719b1b73266bc8854e711886d6fcb367eb47758d24ec2c554

The matched GSQ evaluation completed at 20/1209 correct versus dense 593/1209.
The strict comparison verified identical prompts, targets and input token IDs.
See ../comparison.md for paired uncertainty. Both quantized arms remain severely
degraded; zero baseline accuracy alone does not identify the cause of collapse.
