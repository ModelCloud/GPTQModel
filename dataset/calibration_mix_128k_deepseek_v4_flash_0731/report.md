# Calibration coverage report

## Reference
- model: /tmp/dsv4_bf16
- reference: ref:reference.txt (36889 tokens, 446 modules)
- target tokens (floor): 131072 (selection mode: gain_per_token)

## Warnings

- MoE expert-token floor 16 not reached (min routed tokens per reference-active expert: 0); remaining datasets are redundant. Consider --moe-routing-bypass or a larger pool.

## Per-dataset standalone scores

| dataset | tokens | standalone score | fallback modules |
|---------|--------|------------------|------------------|
| code_00.txt | 29345 | 149239.218960 | 0 |
| code_01.txt | 28561 | 156479.349559 | 0 |
| fineweb_edu_00.txt | 29347 | 150348.625603 | 0 |
| fineweb_edu_01.txt | 28646 | 148256.969495 | 0 |
| imatrix_00.txt | 27900 | 140341.897161 | 0 |
| imatrix_01.txt | 27921 | 142751.366266 | 0 |
| math_00.txt | 27368 | 179624.737646 | 0 |
| math_01.txt | 27533 | 180380.024154 | 0 |
| nm_llm_00.txt | 27513 | 145921.676674 | 0 |
| nm_llm_01.txt | 27419 | 147355.298112 | 0 |
| nm_llm_02.txt | 27181 | 148403.235477 | 0 |
| nm_llm_03.txt | 27376 | 146782.082343 | 0 |
| pg19_00.txt | 27979 | 156479.914124 | 0 |
| pg19_01.txt | 27488 | 165488.533307 | 0 |
| tulu_00.txt | 23932 | 125337.533416 | 0 |
| tulu_01.txt | 24356 | 126272.544781 | 0 |
| wiki_ar_00.txt | 18209 | 189610.506712 | 0 |
| wiki_de_00.txt | 23393 | 169143.791934 | 0 |
| wiki_ja_00.txt | 19257 | 168181.715397 | 0 |
| wiki_ko_00.txt | 24644 | 170764.914478 | 0 |
| wiki_ru_00.txt | 20483 | 166167.355558 | 0 |
| wiki_zh_00.txt | 18642 | 165697.261403 | 0 |
| wiki_zh_01.txt | 16512 | 161236.933655 | 0 |

## Greedy ranking (ranked by conditional gain)

| step | dataset | conditional gain | score after | tokens |
|------|---------|------------------|-------------|--------|
| 1 | wiki_zh_01.txt | 3275307.056139 | 161236.933655 | 16512 |
| 2 | tulu_01.txt | 46785.880506 | 114451.053149 | 40868 |
| 3 | nm_llm_03.txt | 2932.924444 | 111518.128705 | 68244 |
| 4 | wiki_ar_00.txt | 1308.267212 | 110209.861493 | 86453 |
| 5 | wiki_ja_00.txt | 1331.910471 | 108877.951022 | 105710 |
| 6 | nm_llm_01.txt | 2104.008342 | 106773.942680 | 133129 |
| 7 | wiki_ko_00.txt | 723.325548 | 106050.617133 | 157773 |
| 8 | tulu_00.txt | 1049.589793 | 105001.027339 | 181705 |

## Selected mix: wiki_zh_01.txt -> tulu_01.txt -> nm_llm_03.txt -> wiki_ar_00.txt -> wiki_ja_00.txt -> nm_llm_01.txt -> wiki_ko_00.txt -> tulu_00.txt
- score at start: 3436543.989794
- final score: 105001.027339
- cumulative gain: 3331542.962455
- total tokens: 181705
- fallback modules: 0

## Complementarity vs selected mix

| dataset | conditional gain | verdict |
|---------|------------------|---------|
| wiki_zh_01.txt | 3275307.056139 | selected |
| tulu_01.txt | 46785.880506 | selected |
| nm_llm_03.txt | 2932.924444 | selected |
| wiki_ar_00.txt | 1308.267212 | selected |
| wiki_ja_00.txt | 1331.910471 | selected |
| nm_llm_01.txt | 2104.008342 | selected |
| wiki_ko_00.txt | 723.325548 | selected |
| tulu_00.txt | 1049.589793 | selected |
| wiki_zh_00.txt | -1290.396118 | redundant |
| imatrix_01.txt | -1301.350645 | redundant |
| wiki_de_00.txt | -1318.192245 | redundant |
| nm_llm_02.txt | -1402.035633 | redundant |
| nm_llm_00.txt | -1581.493079 | redundant |
| fineweb_edu_00.txt | -1770.698060 | redundant |
| imatrix_00.txt | -1777.267422 | redundant |
| pg19_00.txt | -1837.276552 | redundant |
| code_00.txt | -1987.093154 | redundant |
| fineweb_edu_01.txt | -2174.827930 | redundant |
| code_01.txt | -2244.751189 | redundant |
| math_00.txt | -2652.840978 | redundant |
| math_01.txt | -2753.994505 | redundant |
| wiki_ru_00.txt | -2898.887616 | redundant |
| pg19_01.txt | -2963.597633 | redundant |

## Timing

| stage | seconds |
|-------|---------|
| scan | 28631.883 |
| standalone_scores | 43.144 |
| greedy | 449.248 |
| fallback | 3.001 |
| complementarity | 89.914 |
| total | 29217.190 |

## How to read this report

*Score* is the importance-weighted tail under-coverage versus the held-out reference. For every target module and input channel we compute:

- ``importance = diag / mean(diag)`` (with NaNs clamped to 0) from the reference Hessian diagonal.
- ``gap = relu(ref_p99 - calib_p99) / ref_p99`` per channel.
- ``score = sum(importance * gap)`` over all modules/channels.

A lower score is better. A score of ``0`` means the calibration data covers every reference tail. A high score means important reference channels are not seen in the calibration set.

*Standalone score* is ``score(dataset, ref)`` for each dataset alone.

*Conditional gain* for a dataset is ``score(mix_before, ref) - score(mix_before + dataset, ref)``: how much adding that dataset to the current mix reduces the score. Positive values are complementary; zero or negative values are redundant. The greedy ranking always selects the next dataset by this gain. *Targets are floors, not ceilings*: once ``--target-gain`` or ``--target-tokens`` is reached, the search continues as long as the next conditional gain remains positive. The cumulative gain is ``score_start - score_final`` for the selected mix.
