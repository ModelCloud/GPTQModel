---
name: gptqmodel-evaluation
description: Run, add, or debug post-quantization evaluation with Evalution, lm-eval, GSM8K, perplexity, and sanity generation. Use when measuring model quality after quantization or when evaluation scores look wrong.
---

# GPT-QModel evaluation

Use when measuring post-quantization quality, comparing quantized vs dense baselines, or debugging
unexpectedly low benchmark scores.

## When to use

- Running `gsm8k`, `gsm8k_platinum`, `wikitext`, `perplexity`, or `lm-eval` tasks.
- Integrating with [ModelCloud/Evalution](https://github.com/ModelCloud/Evalution).
- Diagnosing a drop in generation accuracy, coherence, or benchmark scores.
- Comparing two quantizer settings, dynamic overrides, or fused/unfused inference.
- Adding a new evaluation task or dataset to the test suite.

## Key files

- `tests/tasks/gsm8k/`, `tests/tasks/gsm8k_platinum/`
- `tests/test_simple_quant.py`, `tests/models/test_*.py`
- `scripts/benchmark_fuse_real_laguna.py` (sanity prompts)
- `docs/model_inference_optimize.md` (perf/accuracy tradeoffs)
- External: `Evalution` package and `eval.benchmarks`

## Workflow

1. **Establish a dense or higher-precision baseline.**
   - Compare the quantized model against dense BF16/FP16 or a trusted higher-bit checkpoint.
   - Do not treat generation drift alone as a quantization regression; compare token IDs and logits.

2. **Render prompts exactly.**
   - Compare the exact input IDs the quantized and dense models receive.
   - Check tokenizer chat template, special tokens, and `apply_chat_template`.

3. **Run targeted tasks first.**
   - Use one small task (`gsm8k` subset, sanity prompts, `wikitext2` perplexity) before sweeping 150+ tasks.
   - Use `Evalution` integration for standardized metrics.

4. **Control for inference settings.**
   - Run eval at the same `backend`, `device_map`, `attn_implementation`, and batch size for both models.
   - Note that fused/unfused paths may differ slightly due to BF16 accumulation order; compare within tolerance.

5. **Report.**
   - Include exact benchmark config, task list, batch size, prompt template, and scores.
   - Separate model-quality issues (wrong answers) from inference-implementation issues (wrong outputs/shape errors).

## Anti-patterns

- Do not conclude a quantization regression from a single cherry-picked prompt.
- Do not run eval on a different tokenizer or prompt template than the baseline.
- Do not sweep all 150+ tasks before a targeted sanity check passes.
- Do not compare `tok/s` numbers with different batch/token regimes.
