# Llama 3.2 1B Instruct QVQ Divergence-32 audit

## Protocol correction

The historical `token_top1_agreement` implementation compared positions 0--31 of every source row. PR #35 then
replaced that statistic with independently generated trajectories. Neither value is comparable to llama.cpp's
`Same top p` quantization diagnostic: llama.cpp teacher-forces a common corpus and begins scoring only after a
half-context warmup (`tools/perplexity/perplexity.cpp`, `first = n_ctx/2`).

The unified evaluator now reports all protocols without aliasing them:

- `shared_prefix_300_at_32.top1_agreement`: 32 common-prefix positions beginning at half context; headline metric.
- `legacy_shared_prefix_first_32.top1_agreement`: former position-0 statistic, retained for old artifact comparison.
- `divergence_300.trajectory_survival`: exact independently generated 32-token trajectory survival.

## Disjoint data contract

- ordinary calibration: `nm-calibration/llm.parquet`, rows 0--127;
- YAQA Sketch-B: locally materialized optimized mix, 182 rows / 302,193 tokens;
- damping selection: rows 128--159, followed by confirmation on rows 128--427;
- locked evaluation: `nm-calibration/llm.parquet`, rows 512--811.

The locked slice was not used for calibration or candidate selection. Of its 300 rows, 298 were long enough for
the half-context warmup plus the fixed 32-position horizon.

## Locked results

| Checkpoint | Effective precision | Final KL | Token top-1 | Top-5 overlap | Top-10 overlap | Warm-context @32 top-1 | Legacy first-32 top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer-damping YAQA | W2, V2B2-P32 (2.03125 BPW payload estimate) | 0.271032 | 81.7096% | 72.6876% | 71.7525% | **82.0365%** | 76.9753% |
| MLP-down diagnostic | W2.5 down; all other projections W2 | 0.229711 | 82.3401% | 74.3826% | 73.5418% | **83.5570%** | 68.4469% |

The flat-W2 checkpoint clears the predeclared 80% target. The W2.5 arm is diagnostic only; it confirms that MLP
sensitivity remains the next optimization target but is not needed to pass the corrected gate.

The reproducible W2 configuration uses regularization 0.10 for layers 0, 6, 10, and 12 and 0.05 elsewhere. Dynamic
`yaqa_regularization` support was added so the selected checkpoint can be produced directly rather than assembled
from payload shards.
