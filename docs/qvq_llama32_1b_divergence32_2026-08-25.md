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

The published Unsloth scale also resolves the earlier target mismatch: its narrative places UD-Q2_K_XL at roughly
25% and lower 1-bit quants at roughly 8--10%, even though ordinary teacher-forced Top-1 can be about 77%. Therefore
the current flat-W2 development target is **at least 25% independent aligned-token agreement through 32**, not 82%.
This is a directional Q2 reference rather than a direct leaderboard comparison because the models, quant formats,
prompt manifest, and unpublished Unsloth scalar reduction are not identical.

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
| Two-epoch / 64-batch output alignment | W2, V2B2-P32 (2.03125 BPW payload estimate) | **0.239397** | **82.6090%** | **74.3085%** | **73.6837%** | **83.0222%** | 75.0000% |
| MLP-down diagnostic | W2.5 down; all other projections W2 | 0.229711 | 82.3401% | 74.3826% | 73.5418% | **83.5570%** | 68.4469% |

The best flat-W2 checkpoint cleared the historical 80% **shared-prefix** target without a precision exception.
Increasing output alignment from one epoch / 32 train batches to two epochs / 64 train batches improved every
reported final-logit metric: KL fell 3.78%, Top-1 gained 0.5747 percentage points, and SP-Top1@32-W50 gained 0.0524
points over the prior aligned checkpoint. These gains do not imply that the independent-trajectory target has been
reached; Divergence-300 @32 remains a separate and substantially stricter gate. The W2.5 arm is diagnostic only and
confirms that MLP sensitivity remains the next optimization target.

The reproducible W2 configuration uses regularization 0.10 for layers 0, 6, 10, and 12 and 0.05 elsewhere. Dynamic
`yaqa_regularization` support was added so the selected checkpoint can be produced directly rather than assembled
from payload shards. The larger positive alignment run is captured by
`scripts/configs/llama32_1b_v2b2_p32_yaqa_layer_damping_align2e64.json`; it changes only the alignment search budget,
not the flat-W2 payload format.

## Independent-rollout development results

The flat-W2 layer-damping checkpoint and the output-aligned candidate were measured on the same pinned,
content-disjoint QVQ Divergence-300 development manifest with FP16 compute and SDPA on both sides. These are
development results; the locked Divergence-300 split remains untouched.

| Horizon | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer damping: exact-prefix survival | **71.0000%** | **56.6667%** | 34.3333% | 10.0000% | 2.3333% | 0.3333% |
| Layer damping + alignment: exact-prefix survival | 60.3333% | 50.3333% | **38.6667%** | **17.0000%** | **3.0000%** | **0.6667%** |
| Two-epoch / 64-batch alignment: exact-prefix survival | 64.0000% | 51.0000% | 33.0000% | 14.6667% | 2.6667% | **1.0000%** |

| Checkpoint | Independent aligned-token Top-1 through 32 | Matching positions | Exact trajectories at 32 | Mean first divergence |
| --- | ---: | ---: | ---: | ---: |
| Layer damping | 13.9896% | 1,343 / 9,600 | 1 / 300 (0.3333%) | 4.1667 |
| Layer damping + output alignment | **15.1667%** | **1,456 / 9,600** | **2 / 300 (0.6667%)** | **4.5733** |
| Two-epoch / 64-batch output alignment | **17.0938%** | **1,641 / 9,600** | **3 / 300 (1.0000%)** | 4.5633 |

The larger alignment budget gained another 185 aligned positions over the one-epoch candidate, or 1.9271 percentage
points (+12.71% relative), and one additional exact trajectory. Its exact-trajectory 95% Wilson interval is
0.3407%--2.8983%. It does not dominate exact-prefix survival at every early horizon, which is why the full horizon
curve and the aggregate aligned-token score are both reported. Its per-source aligned-token scores are 24.2188% on
MathArena, 19.2500% on LongBench v2, 16.4062% on SWE-bench Verified, 12.0313% on Terminal-Bench 2.1, and 11.8125%
on non-English Multi-IF.

The prompt manifest SHA-256 is `701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b2`.

This result formally rules out interpreting the historical 82.0365% shared-prefix score as Divergence-300 @32. The
corrected development target of 25% aligned-token agreement requires 2,400 of 9,600 positions to match; the current
candidate matches 1,641, leaving a gap of 759 positions. If the target were instead interpreted as exact-trajectory
survival, it would require 75 of 300 prompts; the candidate matches three. The 25% target is therefore **not yet
reached under either reduction**. Configuration sweeps must report both
reductions without renaming shared-prefix agreement or silently switching which reduction is used for the target.
