#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONHASHSEED=7
export ZML_LLAMA_ATTENTION=fa2
export ZML_LLAMA_GRAPH_MODE=decode

exec /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- \
  /root/venv-py3.14t/bin/python /root/zml-ultra/examples/llm/evaluate_llama_batched_gsm8k.py \
  --model /root/qvq-results/w3-p32-gsq-corrected-20260916/model \
  --runner /root/zml-ultra/bazel-bin/examples/llm/llama_paged_token_runner \
  --reference /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32/evaluations/f6-seed7-abc-20260909__C-transformers-paged-fa2-b1-tokenids-full1209-v1/evaluation.gsm8k_platinum_cot.samples.json \
  --dataset-arrow /root/.cache/huggingface/datasets/madrylab___gsm8k-platinum/main/0.0.0/e762492455a1cf7967de89f05b6bef72fc713b66/gsm8k-platinum-test.arrow \
  --output /root/qvq-results/w3-p32-gsq-corrected-20260916/evaluations/zml-sm90-full1209/evaluation.json \
  --run-id w3-p32-gsq-corrected-20260916 \
  --arm-id gsq-corrected-zml-sm90-fa2-b8-graph-full1209 \
  --rows 1209 \
  --batch-size 8 \
  --context 8192 \
  --prefill-len 8192 \
  --max-new-tokens 256 \
  --zml-repo /root/zml-ultra \
  --qvq-repo /root/qvq-gsq-validation \
  --log /root/qvq-results/w3-p32-gsq-corrected-20260916/evaluations/zml-sm90-full1209/evaluation.log
