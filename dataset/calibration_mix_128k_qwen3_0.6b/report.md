# Calibration coverage report

## Reference
- model: Qwen/Qwen3-0.6B
- reference: reference (15665 tokens, 168 modules)
- target tokens (floor): 131072 (selection mode: gain_per_token)

## Per-dataset standalone scores

| dataset | tokens | standalone score | fallback modules |
|---------|--------|------------------|------------------|
| imatrix_00 | 32474 | 27659.696865 | 0 |
| imatrix_01 | 30825 | 27030.957657 | 0 |
| nm_llm_00 | 32633 | 21375.393707 | 0 |
| nm_llm_01 | 31926 | 20569.932030 | 0 |
| nm_llm_02 | 31881 | 21639.773708 | 0 |
| nm_llm_03 | 32442 | 20382.992767 | 0 |

## Greedy ranking (ranked by conditional gain)

| step | dataset | conditional gain | score after | tokens |
|------|---------|------------------|-------------|--------|
| 1 | imatrix_01 | 202184.334396 | 27030.957657 | 30825 |
| 2 | nm_llm_03 | 5968.951996 | 21062.005661 | 63267 |
| 3 | nm_llm_00 | 858.789909 | 20203.215752 | 95900 |
| 4 | nm_llm_01 | 288.631954 | 19914.583797 | 127826 |
| 5 | imatrix_00 | 158.377701 | 19756.206097 | 160300 |
| 6 | nm_llm_02 | 325.757925 | 19430.448172 | 192181 |

## Selected mix: imatrix_01 -> nm_llm_03 -> nm_llm_00 -> nm_llm_01 -> imatrix_00 -> nm_llm_02
- score at start: 229215.292053
- final score: 19430.448172
- cumulative gain: 209784.843882
- total tokens: 192181
- fallback modules: 0

## Complementarity vs selected mix

| dataset | conditional gain | verdict |
|---------|------------------|---------|
| imatrix_01 | 202184.334396 | selected |
| nm_llm_03 | 5968.951996 | selected |
| nm_llm_00 | 858.789909 | selected |
| nm_llm_02 | 325.757925 | selected |
| nm_llm_01 | 288.631954 | selected |
| imatrix_00 | 158.377701 | selected |

## Timing

| stage | seconds |
|-------|---------|
| load | 3.959 |
| scan | 1214.364 |
| standalone_scores | 1.323 |
| greedy | 17.267 |
| fallback | 0.658 |
| complementarity | 0.000 |
| total | 1237.572 |

## How to read this report

*Score* is the importance-weighted tail under-coverage versus the held-out reference. For every target module and input channel we compute:

- ``importance = diag / mean(diag)`` (with NaNs clamped to 0) from the reference Hessian diagonal.
- ``gap = relu(ref_p99 - calib_p99) / ref_p99`` per channel.
- ``score = sum(importance * gap)`` over all modules/channels.

A lower score is better. A score of ``0`` means the calibration data covers every reference tail. A high score means important reference channels are not seen in the calibration set.

*Standalone score* is ``score(dataset, ref)`` for each dataset alone.

*Conditional gain* for a dataset is ``score(mix_before, ref) - score(mix_before + dataset, ref)``: how much adding that dataset to the current mix reduces the score. Positive values are complementary; zero or negative values are redundant. The greedy ranking always selects the next dataset by this gain. *Targets are floors, not ceilings*: once ``--target-gain`` or ``--target-tokens`` is reached, the search continues as long as the next conditional gain remains positive. The cumulative gain is ``score_start - score_final`` for the selected mix.
