# GPT-QModel agent guidance

This file governs the whole repository in addition to any parent agent instructions.

## Tokenizer normalization ownership

For tokenizer initialization, tokenization normalization, special-token compatibility, prompt rendering, chat-template correctness, or unexpectedly low inference/evaluation scores, read and follow `$gptqmodel-tokenizer-normalization` in `.agents/skills/gptqmodel-tokenizer-normalization/SKILL.md` before editing.

GPT-QModel depends on [ModelCloud/Tokenicer](https://github.com/ModelCloud/Tokenicer). Put reusable corrective tokenizer and chat-template normalization in Tokenicer, add relevant tests, increment its version, run its complete test suite, then commit, push, and open a Tokenicer pull request. Do not leave model-loader patches in GPT-QModel merely to avoid fixing the dependency.

Establish a non-quantized baseline before treating bad generation or evaluation scores as a quantization regression. Compare exact rendered prompts and input IDs as well as aggregate scores.
