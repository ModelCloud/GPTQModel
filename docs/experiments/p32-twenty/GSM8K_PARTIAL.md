# Bounded GSM8K comparison

These are the first 128 GSM8K CoT examples, batch one, seed 7, identical chat
prompts, and a 256-new-token cap. They are not full-benchmark scores. No GSM8K
example was used for calibration or residual fitting.

| Operator | Correct | Accuracy |
|---|---:|---:|
| Original BF16 | 59/128 | 0.4609375 |
| Window P32 | 59/128 | 0.4609375 |
| Window + joint rank-16 layer-0 down recovery | 64/128 | 0.5000000 |

Prompt text, targets, and example indices matched exactly across these arms.
BF16/window exchanged 11 wins and 11 losses. Joint recovery versus window gained
7 answers and lost 2; exact two-sided discordance p=0.1796875. The five-answer
aggregate gain is promising but inconclusive on this small slice.

The canonical FP32 teacher completed at 58/128; see [the completed evaluation summary](FULL_ARC.md). No P32 post-quant advantage or
unrestricted promotion is established. Archives retain prompt hashes and per-case
scores in [results/gsm8k-128](results/gsm8k-128); full runtime outputs remain outside Git.
