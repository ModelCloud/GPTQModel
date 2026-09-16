#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MAX_JOBS=4
export PYTHONPATH=/root/qvq-gsq-validation

exec /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- \
  /root/venv-py3.14t/bin/python /root/qvq-gsq-validation/scripts/run_in_worktree.py \
  --worktree /root/qvq-gsq-validation \
  --script scripts/qvq_quantize.py -- \
  --model /monster/data/model/Llama-3.2-1B-Instruct \
  --output /root/qvq-results/w3-p32-gsq-ab-20260915/gsq \
  --report /root/qvq-results/w3-p32-gsq-ab-20260915/metadata/gsq.qvq_quantize_run.json \
  --quant-config /root/qvq-gsq-validation/artifacts/w3-p32-gsq-validation/gsq.quant-config.json \
  --calibration-dataset /monster/data/model/dataset/nm-calibration/llm.parquet \
  --calibration-dataset-split train \
  --calibration-row-start 0 \
  --calibration-rows 128 \
  --yaqa-dataset /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/calibration/source/yaqa182-nm10000.parquet \
  --yaqa-dataset-split train \
  --yaqa-row-start 0 \
  --yaqa-rows 10178 \
  --batch-size 1 \
  --concat-size 0 \
  --calibration-sort desc \
  --device cuda:0 \
  --disjointness-manifest /root/qvq-gsq-validation/artifacts/w3-p32-gsq-validation/disjointness.audit.json \
  --require-disjointness \
  --qvq-telemetry
