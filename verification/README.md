# vLLM and SGLang verification

This directory contains reusable vLLM/SGLang accuracy verification: standardized Evalution scoring and native versus
post-quantization output-distribution comparison. Throughput remains under `benchmark/laguna_single_gpu_tps/`; it is
not duplicated here. The scripts contain no local model, repository, conda-installation, or fixed-GPU paths.

## Environment and GPU lease

Run vLLM commands in the `vllm_test` conda environment and SGLang commands in `sglang_test`. The examples lease
GPUs by UUID through the repository allocator, so physical devices are resolved at runtime rather than inferred from
a fixed CUDA index.

```bash
python -m gpu_allocator.cli run -n 1 -t 60 --style uuid -- \
  conda run -n vllm_test --no-capture-output \
  python verification/evalution_score.py --help

python -m gpu_allocator.cli run -n 1 -t 60 --style uuid -- \
  conda run -n sglang_test --no-capture-output \
  python verification/compare_logits.py --help
```

Use `-n N` together with `--tensor-parallel-size N` for tensor parallel models. Each GPU command refuses to start
unless exactly that many allocated devices are visible and all requested devices pass three idle samples. Runtime
settings, generated configurations, commands, physical GPU identity, and scores are retained with the artifacts.

## Evalution accuracy suites

`evalution_score.py` generates an exact Evalution YAML for exactly one report suite per invocation:

- ARC-Challenge: 1,172 test rows, multiple-choice log-likelihood, `acc,exam`;
- GSM8K-Platinum CoT: 1,209 test rows, chat template, up to 256 generated tokens, `acc,num`;
- MMLU-STEM: 3,153 test rows over all 19 STEM subjects, 5-shot, `acc,ll`;
- MMLU-History: 930 test rows covering European, US, World History, and Prehistory, 5-shot, `acc,ll`.

Run vLLM:

```bash
python -m gpu_allocator.cli run -n 1 -t 60 --style uuid -- \
  conda run -n vllm_test --no-capture-output \
  python verification/evalution_score.py \
    --engine vllm \
    --model /path/to/model \
    --task arc_challenge \
    --batch-size 32 \
    --output artifacts/vllm_eval.json
```

Run SGLang:

```bash
python -m gpu_allocator.cli run -n 1 -t 60 --style uuid -- \
  conda run -n sglang_test --no-capture-output \
  python verification/evalution_score.py \
    --engine sglang \
    --model /path/to/model \
    --task arc_challenge \
    --batch-size 32 \
    --output artifacts/sglang_eval.json
```

Run the command four times with `--task arc_challenge`, `--task gsm8k_platinum_cot`, `--task mmlu_stem`, and
`--task mmlu_history`. Each invocation starts and stops its own engine, so its summary contains an independent launch
timestamp, engine startup time, and task runtime. This also allows task-specific tensor parallelism and batch sizes.

The SGLang defaults reproduce the validated Ultra evaluation path: deterministic inference, CUDA graphs disabled,
FlashInfer attention, and Triton prefill. Pass `--sglang-attention-backend auto` and
`--sglang-prefill-attention-backend auto` to delegate both choices to SGLang. Extra runtime/model kwargs use
`--model-kwargs-json '{"key": "value"}'` or `--model-kwargs-json @options.json`.
For DeepSeek runtimes where fused WQA/WKV yields missing multiple-choice logprobs, pass
`--sglang-disable-fused-wqa-wkv`; the exact `SGLANG_OPT_FUSE_WQA_WKV=0` child-process override is retained in the
score summary.

For a bounded integration smoke test, add `--max-rows 4`. To create and schema-check the exact YAML without loading a
model or requiring a GPU, add `--dry-run`. Every completed run produces three artifacts:

```text
<output>.json             raw per-sample Evalution result
<output>.evalution.yaml   exact reusable Evalution configuration
<output>.scores.json      validated concise scores and run metadata
```

The wrapper recomputes every aggregate score from per-sample scorer values. ARC follows its reference exam rule and
awards `1/k` credit when the gold answer is among `k` tied top choices; the summary records the credit sum, fully
correct count, and partial-credit count separately. Other supported suites require binary per-sample scores. Full
runs additionally require the exact suite sizes and all expected MMLU subjects before the summary is marked
successful.

The score summary records `gpu_models` plus a deduplicated `gpus` inventory (physical ID, PCI bus ID, UUID, model,
driver, compute capability, and memory). It also records the wrapper start time, Evalution process start/completion
times, engine startup seconds, total Evalution wall time, and separate start/completion/duration values for ARC,
GSM8K-Platinum, MMLU-STEM, and MMLU-History. Task duration starts at Evalution's `running test suite` event and ends
at its matching `completed test` event, so it includes dataset preparation, inference, and scoring for that suite but
excludes shared model startup and final engine shutdown.

## Native versus post-quant logits/KLD

`compare_logits.py` tokenizes held-out prompts once, saves the exact rendered text and token IDs, then launches the
native and quantized models in separate fresh engine subprocesses. Both receive the same raw token-ID prefixes. This
avoids comparing different chat templates, special tokens, cache state, or generated histories.

Run with vLLM:

```bash
python -m gpu_allocator.cli run -n 1 -t 60 --style uuid -- \
  conda run -n vllm_test --no-capture-output \
  python verification/compare_logits.py \
    --engine vllm \
    --native-model /path/to/native-model \
    --quantized-model /path/to/quantized-model \
    --tokenizer /path/to/native-model \
    --prompts /path/to/heldout-prompts.jsonl \
    --positions-per-prompt 4 \
    --output artifacts/vllm_logits.json
```

The SGLang command differs only in the environment and engine selection:

```bash
python -m gpu_allocator.cli run -n 1 -t 60 --style uuid -- \
  conda run -n sglang_test --no-capture-output \
  python verification/compare_logits.py \
    --engine sglang \
    --native-model /path/to/native-model \
    --quantized-model /path/to/quantized-model \
    --prompts /path/to/heldout-prompts.jsonl \
    --positions-per-prompt 4 \
    --output artifacts/sglang_logits.json
```

Prompt JSON may be a list of strings, `{ "text": ... }` objects, or `{ "messages": [...] }` chat records. JSONL
accepts the same record shapes; plain text treats each non-empty line as one prompt. Message records always use the
tokenizer chat template. `--apply-chat-template` wraps string prompts as a user message. Verification data should be
held out from calibration data.

For each bounded prefix position, the engine returns the complete next-token log-probability distribution. vLLM uses
its flat full-vocabulary representation when that API is available and otherwise consumes the standard per-position
mapping; SGLang gathers the requested logprob for every vocabulary token ID, which avoids an unnecessary
full-vocabulary Top-k sort. The selected transport is recorded in artifact metadata. The report computes:

- exact FP64-accumulated `KL(P_native || P_quantized)`: mean, median, p95, p99, and maximum;
- Top-1 agreement/flip count and Top-k set overlap;
- total variation distance;
- centered-logit MAE, RMSE, and cosine;
- log-probability MAE/RMSE and Top-1 margin delta;
- native/quantized NLL delta for sampled positions whose next teacher-forced token is available.

Serving engines expose normalized log probabilities, not the raw additive logit offset. Subtracting each vector's
mean recovers logits up to that unidentifiable constant, so centered-logit deltas are exact for all meaningful relative
logit differences. KLD and Top-k metrics are unaffected by the offset.

Raw `native.log_probs.npz`, `quantized.log_probs.npz`, worker configs, and `inputs.json` are retained beside the report
for independent recomputation. Full-vocabulary output is intentionally bounded by prompts and sampled positions; the
script estimates artifact size and refuses to exceed `--max-artifact-gib` without an explicit override. Optional
`--max-kld-mean` and `--min-top1-agreement` turn the metrics into process exit gates without inventing default quality
thresholds.

Use `--native-tensor-parallel-size` and `--quantized-tensor-parallel-size` when the dense and quantized checkpoints
need different GPU counts. Both default to `--tensor-parallel-size`. A validated dense NPZ can be reused with
`--reuse-native-artifact`; the script verifies its model, tokenizer, dtype, TP size, input hash, shape, and matrix
metadata before launching only the quantized role. The enclosing allocator lease then needs to cover only the active
quantized TP size. Reports must retain the role-specific TP sizes because changing TP topology can change numerical
results.
