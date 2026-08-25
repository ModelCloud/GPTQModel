# Llama 3.2 1B Instruct QVQ quantization-fidelity audit

## Protocol correction

The historical `token_top1_agreement` implementation compared positions 0--31 of every source row. PR #35 then
replaced that statistic with independently generated trajectories. Neither value is comparable to llama.cpp's
`Same top p` quantization diagnostic: llama.cpp teacher-forces a common corpus and begins scoring only after a
half-context warmup (`tools/perplexity/perplexity.cpp`, `first = n_ctx/2`).

The unified evaluator now reports all protocols without aliasing them:

- `sp_top1_32_w50.top1_agreement`: compact JSON name for **Shared-prefix top-1 agreement@32, with 50% context
  warmup**; 32 common-prefix positions beginning halfway through the context.
- `legacy_shared_prefix_first_32.top1_agreement`: former position-0 statistic, retained for old artifact comparison.
- `divergence_300_at_32.independent_token_top1_agreement_at_32`: aligned token-ID agreement across the 32 positions
  of two independently generated trajectories.
- `divergence_300_at_32.exact_trajectory_agreement_at_32`: fraction of 300 prompts whose independently generated
  trajectories remain exactly identical through all 32 tokens.

The shared-prefix score is not [Unsloth's **Divergence-300 @32**](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs#divergence-300-32). Unsloth independently greedy-decodes 32 tokens
from BF16 and quantized models on 300 held-out task prompts, so its test measures compounding trajectory drift. The
shared-prefix score teacher-forces the same source context at every position and measures isolated next-token
agreement. Unsloth's test is conceptually closest to `divergence_300_at_32`; numerical comparison additionally requires
the same prompts, chat templates, decoding settings, and scalar aggregation rule.

## Disjoint data contract

- ordinary calibration: `nm-calibration/llm.parquet`, rows 0--127;
- YAQA Sketch-B: locally materialized optimized mix, 182 rows / 302,193 tokens;
- damping selection: rows 128--159, followed by confirmation on rows 128--427;
- locked evaluation: `nm-calibration/llm.parquet`, rows 512--811.

The locked slice was not used for calibration or candidate selection. Of its 300 rows, 298 were long enough for
the half-context warmup plus the fixed 32-position horizon.

## Locked results

| Checkpoint | Effective precision | Final KL | Token top-1 | Top-5 overlap | Top-10 overlap | SP-Top1@32-W50 | Legacy first-32 top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer-damping YAQA | W2, V2B2-P32 (2.03125 BPW payload estimate) | 0.271032 | 81.7096% | 72.6876% | 71.7525% | **82.0365%** | 76.9753% |
| Layer damping + output alignment | W2, V2B2-P32 (2.03125 BPW payload estimate) | **0.248805** | **82.0343%** | **73.9342%** | **73.1259%** | **82.9698%** | 71.2688% |
| MLP-down diagnostic | W2.5 down; all other projections W2 | 0.229711 | 82.3401% | 74.3826% | 73.5418% | **83.5570%** | 68.4469% |

The best flat-W2 checkpoint cleared the historical 80% **shared-prefix** target without a precision exception. Output
alignment improved every final-logit metric and SP-Top1@32-W50 over the layer-damping-only checkpoint. That result
does not imply an 80% Divergence-300 @32 score; independent trajectories are a separate and substantially stricter
gate. The W2.5 arm is diagnostic only and confirms that MLP sensitivity remains the next optimization target.

The reproducible W2 configuration uses regularization 0.10 for layers 0, 6, 10, and 12 and 0.05 elsewhere. Dynamic
`yaqa_regularization` support was added so the selected checkpoint can be produced directly rather than assembled
from payload shards.

## Independent-rollout development results

The flat-W2 layer-damping checkpoint and the output-aligned candidate were measured on the same pinned,
content-disjoint QVQ Divergence-300 development manifest with FP16 compute and SDPA on both sides. These are
development results; the locked Divergence-300 split remains untouched.

| Horizon | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer damping: exact-prefix survival | **71.0000%** | **56.6667%** | 34.3333% | 10.0000% | 2.3333% | 0.3333% |
| Layer damping + alignment: exact-prefix survival | 60.3333% | 50.3333% | **38.6667%** | **17.0000%** | **3.0000%** | **0.6667%** |

| Checkpoint | Independent aligned-token Top-1 through 32 | Matching positions | Exact trajectories at 32 | Mean first divergence |
| --- | ---: | ---: | ---: | ---: |
| Layer damping | 13.9896% | 1,343 / 9,600 | 1 / 300 (0.3333%) | 4.1667 |
| Layer damping + output alignment | **15.1667%** | **1,456 / 9,600** | **2 / 300 (0.6667%)** | **4.5733** |

Output alignment gained 113 aligned positions, or 1.1771 percentage points (+8.41% relative), and one additional
exact trajectory. The candidate's exact-trajectory 95% Wilson interval is 0.1830%--2.3978%; the baseline interval is
0.0589%--1.8637%. Exact-prefix survival is lower for the candidate at tokens 1 and 2 but higher from token 3 onward,
which is why the full horizon curve and the aggregate aligned-token score are both useful.

The prompt manifest SHA-256 is `701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b2`.

This result formally rules out interpreting the historical 82.0365% shared-prefix score as Divergence-300 @32.
Reaching 82% independent-rollout aligned-token Top-1 would require 7,872 of 9,600 positions to match; the candidate
matches 1,456. Reaching 82% exact survival would separately require 246 of 300 prompts to match all 32 generated
tokens; the candidate matches two. The >82% target was therefore **not reached under either reduction**. Configuration
sweeps must report both reductions without renaming shared-prefix agreement.
