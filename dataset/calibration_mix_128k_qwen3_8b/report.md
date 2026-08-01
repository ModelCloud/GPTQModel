# Calibration coverage report

## Reference
- model: Qwen/Qwen3-8B
- reference: reference (15504 tokens, 216 modules)
- target tokens (floor): 131072 (selection mode: gain_per_token)

## Warnings

- Target token floor 131072 not reached (reached 127402); all remaining datasets are redundant.

## Per-dataset standalone scores

| dataset | tokens | standalone score | fallback modules |
|---------|--------|------------------|------------------|
| imatrix_00 | 32474 | 123214.205181 | 0 |
| imatrix_01 | 30825 | 125696.749397 | 0 |
| nm_llm_00 | 31886 | 128002.478668 | 0 |
| nm_llm_01 | 32286 | 120485.328698 | 0 |
| nm_llm_02 | 32021 | 118270.127073 | 0 |
| nm_llm_03 | 32082 | 123692.027466 | 0 |

## Greedy ranking (ranked by conditional gain)

| step | dataset | conditional gain | score after | tokens |
|------|---------|------------------|-------------|--------|
| 1 | imatrix_01 | 1053951.257927 | 125696.749397 | 30825 |
| 2 | nm_llm_03 | 14339.037155 | 111357.712242 | 62907 |
| 3 | imatrix_00 | 4040.059410 | 107317.652832 | 95381 |
| 4 | nm_llm_02 | 3740.834091 | 103576.818741 | 127402 |

## Selected mix: imatrix_01 -> nm_llm_03 -> imatrix_00 -> nm_llm_02
- score at start: 1179648.007324
- final score: 103576.818741
- cumulative gain: 1076071.188583
- total tokens: 127402
- fallback modules: 0

## Complementarity vs selected mix

| dataset | conditional gain | verdict |
|---------|------------------|---------|
| imatrix_01 | 1053951.257927 | selected |
| nm_llm_03 | 14339.037155 | selected |
| imatrix_00 | 4040.059410 | selected |
| nm_llm_02 | 3740.834091 | selected |
| nm_llm_00 | -3201.279572 | redundant |
| nm_llm_01 | -3546.486665 | redundant |

## Timing

| stage | seconds |
|-------|---------|
| load | 23.413 |
| scan | 7643.242 |
| standalone_scores | 36.372 |
| greedy | 64.411 |
| fallback | 0.777 |
| complementarity | 13.741 |
| total | 7784.595 |

## How to read this report

*Score* is the importance-weighted tail under-coverage versus the held-out reference. For every target module and input channel we compute:

- ``importance = diag / mean(diag)`` (with NaNs clamped to 0) from the reference Hessian diagonal.
- ``gap = relu(ref_p99 - calib_p99) / ref_p99`` per channel.
- ``score = sum(importance * gap)`` over all modules/channels.

A lower score is better. A score of ``0`` means the calibration data covers every reference tail. A high score means important reference channels are not seen in the calibration set.

*Standalone score* is ``score(dataset, ref)`` for each dataset alone.

*Conditional gain* for a dataset is ``score(mix_before, ref) - score(mix_before + dataset, ref)``: how much adding that dataset to the current mix reduces the score. Positive values are complementary; zero or negative values are redundant. The greedy ranking always selects the next dataset by this gain. *Targets are floors, not ceilings*: once ``--target-gain`` or ``--target-tokens`` is reached, the search continues as long as the next conditional gain remains positive. The cumulative gain is ``score_start - score_final`` for the selected mix.
