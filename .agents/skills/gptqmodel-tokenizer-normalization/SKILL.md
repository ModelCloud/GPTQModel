---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-tokenizer-normalization
description: Diagnose and correct GPT-QModel tokenizer initialization, tokenization normalization, special-token handling, prompt rendering, and chat-template problems. Use when inference or evaluation quality suggests wrong token IDs or prompts, when a model needs tokenizer load kwargs or compatibility patches, or when tokenizer behavior is being changed in GPT-QModel; reusable corrective behavior must be implemented, tested, versioned, pushed, and submitted upstream to github.com/ModelCloud/Tokenicer.
---

# GPT-QModel tokenizer normalization

## At a glance

For tokenizer, chat-template or generation-quality changes, compare the dense
baseline, rendered prompts and input IDs first. Repair reusable normalization
in [Tokenicer](https://github.com/ModelCloud/Tokenicer); consume its release
through `Tokenicer.load()` rather than patching individual GPT-QModel loaders.

## Diagnose the boundary

1. Reproduce with the non-quantized model before attributing low scores or malformed output to quantization.
2. Capture the exact raw text, rendered chat-template text, input IDs, attention mask, special-token settings, tokenizer class, tokenizer config, and Transformers/Tokenicer versions.
3. Compare direct `AutoTokenizer` behavior, current `Tokenicer.load()`, and the proposed normalization on the same prompts. Include strings that expose the suspected boundary or regex error.
4. Run a small dense-model evaluation with the same task, generation settings, and evaluator used for the quantized model. If dense and quantized runs fail similarly, fix the tokenizer or template before changing quantization code.
5. Separate reusable tokenizer behavior from evaluator-specific prompt construction. AutoTokenizer kwargs, tokenizer compatibility, special-token repair, and model chat-template normalization belong in Tokenicer. A benchmark task's own few-shot or answer-extraction format remains in the evaluation project.

Do not suppress a correctness warning or hard-code a GPT-QModel model-name check when Tokenicer can normalize the underlying behavior. Preserve explicit caller overrides with patterns such as `setdefault` unless the upstream API documents a mandatory correction.

## Implement upstream first

1. Use a task-specific clone of [Tokenicer](https://github.com/ModelCloud/Tokenicer) under the [workspace and draft PR rules](../gptqmodel-agent-workflow/SKILL.md).
2. Sync Tokenicer's default branch and create a focused branch. Preserve unrelated worktree files.
3. Implement the smallest general normalization in Tokenicer and make every tokenizer load and fallback path apply it consistently.
4. Add focused unit tests for the default correction, explicit override behavior, and fallback paths. Add a lightweight end-to-end tokenizer assertion when it materially strengthens the regression.
5. Increment the Tokenicer version for the release-bound correction and document the change where that repository's conventions require it.
6. Run the focused tests and the complete Tokenicer test suite. Resolve every failure before publishing; report genuine environment skips separately.
7. Review and stage only intended files, commit them, push the branch, and open a pull request against `ModelCloud/Tokenicer`. Include the root cause, before/after tokenization evidence, tests, and downstream GPT-QModel impact.

Do not consider a reusable normalization fixed when it exists only as a GPT-QModel workaround. If temporary downstream code is unavoidable while an upstream release is pending, isolate and label it, link the Tokenicer pull request, and remove it as soon as the dependency can be raised.

## Consume the correction in GPT-QModel

1. Route all normal tokenizer construction through `Tokenicer.load()`; avoid new direct `AutoTokenizer.from_pretrained()` paths in model loaders.
2. After the Tokenicer release is available, raise GPT-QModel's minimum Tokenicer version and remove duplicated compatibility code in the same change.
3. Add a GPT-QModel regression that proves the public load path receives normalized behavior without reimplementing Tokenicer internals.
4. Re-run the dense baseline, then quantized save/reload/inference and the affected evaluation tasks. Compare exact prompt rendering and input IDs before comparing scores.

In the final report, link the Tokenicer commit and pull request, state the Tokenicer version, list all Tokenicer and GPT-QModel checks, and distinguish passed tests from skips or unavailable hardware.
