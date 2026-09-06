# Full ARC-Challenge and completed canonical GSM8K slice

All 1172 ARC-Challenge test examples were evaluated with identical prompts,
targets, indices, tokenizer, eager attention and batch1. These are full test-set
results, unlike the earlier 128-example slice. No task data entered fitting.

| Operator | Raw correct | Length-normalized correct |
|---|---:|---:|
|Original BF16|409|439|
|Canonical FP32 P32|390|430|
|Production window|390|428|
|Rank12 L2 FP16|389|431|
|Original rank8 tail FP16|391|432|
|Joint rank16 FP32|391|431|
|New rank8 alpha1 FP16|389|429|

The original rank8 tail FP16 gains 10 and loses 6 normalized answers versus
window: exact paired two-sided p=0.4545. New alpha1 gains8/loses7 normalized
(p=1), and gains8/loses9 raw (p=1). None establishes a statistically persuasive
quality improvement. Original BF16 remains numerically stronger here; these
results do not establish retention of a P32 post-quant advantage on ARC.

The canonical FP32 GSM8K-128 teacher finally completed: **58/128**. Original
BF16 and window each scored59/128; joint rank16 scored64/128. That bounded run
uses the earlier 256-new-token cap and must not be called full GSM8K.

[Per-example ARC archives and paired tests](results/arc-full) retain prompt
hashes and scores. [Canonical GSM8K archive](results/gsm8k-128/canonical.json)
completes the previously missing teacher arm. Future interpretation should use
paired effects and uncertainty, not differences of a few aggregate answers.
