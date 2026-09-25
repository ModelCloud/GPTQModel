<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# GPT-QModel agent guidance

This file governs the whole repository in addition to any parent agent instructions.

## Numerical oracle validation

For any new or changed quantization math, weight packer, or inference kernel,
read and follow `$gptqmodel-torch-oracle-numerics` in
`.agents/skills/gptqmodel-torch-oracle-numerics/SKILL.md` before editing.

For every MLX inference kernel, test FP16 and BF16 activations independently.
Assert that each supported path returns the input activation dtype, and measure
accuracy drift against an independent Torch oracle for both dtypes. Follow the
skill's output-rounding guidance and report arithmetic error separately from
the visible low-precision output error.

## Tokenizer normalization ownership

For tokenizer initialization, tokenization normalization, special-token compatibility, prompt rendering, chat-template correctness, or unexpectedly low inference/evaluation scores, read and follow `$gptqmodel-tokenizer-normalization` in `.agents/skills/gptqmodel-tokenizer-normalization/SKILL.md` before editing.

GPT-QModel depends on [ModelCloud/Tokenicer](https://github.com/ModelCloud/Tokenicer). Put reusable corrective tokenizer and chat-template normalization in Tokenicer, add relevant tests, increment its version, run its complete test suite, then commit, push, and open a Tokenicer pull request. Do not leave model-loader patches in GPT-QModel merely to avoid fixing the dependency.

Establish a non-quantized baseline before treating bad generation or evaluation scores as a quantization regression. Compare exact rendered prompts and input IDs as well as aggregate scores.
