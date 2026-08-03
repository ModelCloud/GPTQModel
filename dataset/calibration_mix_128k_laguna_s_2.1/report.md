# Calibration coverage report

## Reference
- model: /monster/data/model/Laguna-S-2.1-PER-LAYER
- reference: reference (47729 tokens, 288 modules)
- target tokens (floor): 131072 (selection mode: gain_per_token)

## Warnings

- Target token floor 131072 not reached (reached 59075); remaining datasets are redundant (negative conditional gain) and are never added just to fill the floor.

## Per-dataset standalone scores

| dataset | tokens | standalone score | fallback modules |
|---------|--------|------------------|------------------|
| code_00 | 31927 | 69835.480560 | 0 |
| code_01 | 31812 | 70344.477931 | 0 |
| fineweb_edu_00 | 32306 | 58904.444817 | 0 |
| fineweb_edu_01 | 31524 | 58045.433754 | 0 |
| imatrix_00 | 31645 | 57254.144539 | 0 |
| imatrix_01 | 31743 | 60239.109657 | 0 |
| math_00 | 31032 | 77665.642102 | 0 |
| math_01 | 30971 | 80357.497384 | 0 |
| nm_llm_00 | 31041 | 67243.397675 | 0 |
| nm_llm_01 | 30907 | 60062.274799 | 0 |
| nm_llm_02 | 30314 | 65715.336983 | 0 |
| nm_llm_03 | 30742 | 62589.566788 | 0 |
| pg19_00 | 32187 | 67083.086174 | 0 |
| pg19_01 | 31359 | 66674.540268 | 0 |
| tulu_00 | 31591 | 43018.083450 | 0 |
| tulu_01 | 31381 | 46151.055458 | 0 |
| wiki_ar_00 | 31422 | 65120.590733 | 0 |
| wiki_de_00 | 31528 | 70195.678476 | 0 |
| wiki_ja_00 | 27622 | 52347.902958 | 0 |
| wiki_ko_00 | 31339 | 50622.954884 | 0 |
| wiki_ru_00 | 32033 | 57174.881462 | 0 |
| wiki_zh_00 | 29673 | 49433.121931 | 0 |
| wiki_zh_01 | 27484 | 51861.784548 | 0 |

## Greedy ranking (ranked by conditional gain)

| step | dataset | conditional gain | score after | tokens |
|------|---------|------------------|-------------|--------|
| 1 | wiki_zh_01 | 1543530.201048 | 51861.784548 | 27484 |
| 2 | tulu_00 | 14773.618723 | 37088.165825 | 59075 |

## Selected mix: wiki_zh_01 -> tulu_00
- score at start: 1595391.985596
- final score: 37088.165825
- cumulative gain: 1558303.819771
- total tokens: 59075
- fallback modules: 0

## Complementarity vs selected mix

| dataset | conditional gain | verdict |
|---------|------------------|---------|
| wiki_zh_01 | 1543530.201048 | selected |
| tulu_00 | 14773.618723 | selected |
| tulu_01 | -722.233227 | redundant |
| nm_llm_01 | -1257.286274 | redundant |
| nm_llm_03 | -1294.524002 | redundant |
| nm_llm_02 | -1604.536331 | redundant |
| wiki_zh_00 | -1994.638828 | redundant |
| nm_llm_00 | -2052.014339 | redundant |
| math_00 | -2090.329945 | redundant |
| math_01 | -2156.598251 | redundant |
| wiki_ko_00 | -2271.759930 | redundant |
| wiki_ja_00 | -2386.220295 | redundant |
| code_01 | -2468.509151 | redundant |
| code_00 | -2502.151497 | redundant |
| imatrix_01 | -2779.920673 | redundant |
| fineweb_edu_01 | -2923.257854 | redundant |
| fineweb_edu_00 | -3395.275551 | redundant |
| imatrix_00 | -3572.951889 | redundant |
| wiki_de_00 | -3917.164322 | redundant |
| wiki_ru_00 | -4315.049408 | redundant |
| pg19_00 | -4878.569813 | redundant |
| pg19_01 | -5176.034939 | redundant |
| wiki_ar_00 | -5438.532784 | redundant |

## Timing

| stage | seconds |
|-------|---------|
| load | 65.456 |
| scan | 6563.006 |
| standalone_scores | 26.125 |
| greedy | 56.777 |
| fallback | 1.833 |
| complementarity | 69.403 |
| total | 6788.295 |

## How to read this report

*Score* is the importance-weighted tail under-coverage versus the held-out reference. For every target module and input channel we compute:

- ``importance = diag / mean(diag)`` (with NaNs clamped to 0) from the reference Hessian diagonal.
- ``gap = relu(ref_p99 - calib_p99) / ref_p99`` per channel.
- ``score = sum(importance * gap)`` over all modules/channels.

A lower score is better. A score of ``0`` means the calibration data covers every reference tail. A high score means important reference channels are not seen in the calibration set.

*Standalone score* is ``score(dataset, ref)`` for each dataset alone.

*Conditional gain* for a dataset is ``score(mix_before, ref) - score(mix_before + dataset, ref)``: how much adding that dataset to the current mix reduces the score. Positive values are complementary; zero or negative values are redundant. The greedy ranking always selects the next dataset by this gain. *Targets are floors, not ceilings*: once ``--target-gain`` or ``--target-tokens`` is reached, the search continues as long as the next conditional gain remains positive. The cumulative gain is ``score_start - score_final`` for the selected mix.
