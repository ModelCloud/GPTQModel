# Llama 3.2 1B Instruct QVQ quantization-fidelity audit

The append-only [experiment ledger](qvq_llama32_1b_experiment_log_2026-08-25.md) records the complete configuration,
data slices, commands, artifacts, positive results, negative results, invalid evidence, and test outcomes for every
arm in this campaign.

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

## Rendering and template-weighting contract

The optimized 182-row YAQA mix is stored as `messages`. Both the coverage scanner and the quantization lifecycle
therefore render every row through the Llama 3.2 Instruct tokenizer's native chat template. The scanner artifact's
`apply_chat_template: false` means only that raw strings are not wrapped as synthetic user messages; it does not
disable rendering for rows that already carry a `messages` conversation.

The selected checkpoint records `yaqa.chat_template.enabled: false`. Its serialized `content_weight: 0.97` is thus
inactive: YAQA did **not** apply the optional 97% content / 3% template-structure Fisher weighting. That option changes
the direct token weights while retaining all template tokens in the forward context; it is not a row-fraction control.

Every Divergence-300 row is also message-shaped. The evaluator calls the dense model tokenizer's
`apply_chat_template(..., tokenize=True, add_generation_prompt=True)`, left-truncates only above 16,384 prompt tokens,
and supplies the exact same encoded prompt tensors to dense and QVQ models before their independent greedy rollouts.

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
| Layer damping: exact-prefix survival | **71.0000%** | 56.6667% | 34.3333% | 10.0000% | 2.3333% | 0.3333% |
| Layer damping + alignment: exact-prefix survival | 60.3333% | 50.3333% | 38.6667% | **17.0000%** | 3.0000% | 0.6667% |
| Two-epoch / 64-batch alignment: exact-prefix survival | 64.0000% | 51.0000% | 33.0000% | 14.6667% | 2.6667% | 1.0000% |
| Uniform `0.10`: exact-prefix survival | 67.3333% | 55.0000% | 42.6667% | 14.6667% | 4.6667% | 1.0000% |
| Uniform `0.125`: exact-prefix survival | 63.3333% | 50.0000% | 30.3333% | 7.0000% | 2.3333% | 0.6667% |
| Uniform `0.10` + output alignment: exact-prefix survival | 68.6667% | 56.3333% | **44.3333%** | 14.3333% | 3.6667% | 0.6667% |
| Uniform `0.10` + 97/3 chat weighting: exact-prefix survival | **71.0000%** | **59.0000%** | 40.0000% | 16.6667% | **5.3333%** | **1.3333%** |

| Checkpoint | Independent aligned-token Top-1 through 32 | Matching positions | Exact trajectories at 32 | Mean first divergence |
| --- | ---: | ---: | ---: | ---: |
| Layer damping | 13.9896% | 1,343 / 9,600 | 1 / 300 (0.3333%) | 4.1667 |
| Layer damping + output alignment | 15.1667% | 1,456 / 9,600 | 2 / 300 (0.6667%) | 4.5733 |
| Two-epoch / 64-batch output alignment | 17.0938% | 1,641 / 9,600 | 3 / 300 (1.0000%) | 4.5633 |
| Uniform `0.10`, no output alignment | **17.8854%** | **1,717 / 9,600** | 3 / 300 (1.0000%) | 4.8533 |
| Uniform `0.125`, no output alignment | 13.4375% | 1,290 / 9,600 | 2 / 300 (0.6667%) | 3.6233 |
| Uniform `0.10` + output alignment | 17.2188% | 1,653 / 9,600 | 2 / 300 (0.6667%) | 4.7500 |
| Uniform `0.20`, no output alignment | 16.3333% | 1,568 / 9,600 | 3 / 300 (1.0000%) | 4.6100 |
| Uniform `0.10` + 97/3 chat weighting | 15.2604% | 1,465 / 9,600 | **4 / 300 (1.3333%)** | **5.0333** |

The larger hybrid alignment budget gained another 185 aligned positions over the one-epoch candidate, or 1.9271
percentage points (+12.71% relative), and one additional exact trajectory. The corrected damping control then found
that uniform `0.10` regularization without output alignment improves another 76 positions over that aligned hybrid,
making it the current development leader. Its exact-trajectory 95% Wilson interval is 0.3407%--2.8983%. It does not
dominate every source: its per-source aligned-token scores are 15.1563% on MathArena, 19.3750% on LongBench v2,
22.0000% on SWE-bench Verified, 17.5781% on Terminal-Bench 2.1, and 11.6875% on non-English Multi-IF. This is why the
full source breakdown, horizon curve, and aggregate aligned-token score are all reported.

The existing uniform-`0.10` output-aligned checkpoint improves early prefix survival but loses 64 aligned positions
over the complete 32-token horizon. Its teacher-forced alignment objective therefore does not stack with the corrected
damping gain. The unaligned uniform-`0.10` checkpoint remains selected; the aligned arm is retained as negative
evidence rather than promoted from its stronger token-1/4 behavior.

The 97% content / 3% template-structure YAQA weighting control shows the same objective tension more strongly. It
raises token-1 agreement from 67.3333% to 71.0000%, exact 32-token trajectories from 3 to 4, and mean first divergence
from 4.8533 to 5.0333, but loses 252 aligned positions over the complete horizon. The control is therefore rejected
for the aligned-token target despite its stronger early-prefix and exact-survival reductions.

The prompt manifest SHA-256 is `701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b2`.

This result formally rules out interpreting the historical 82.0365% shared-prefix score as Divergence-300 @32. The
corrected development target of 25% aligned-token agreement requires 2,400 of 9,600 positions to match; the current
candidate matches 1,717, leaving a gap of 683 positions. If the target were instead interpreted as exact-trajectory
survival, it would require 75 of 300 prompts; the candidate matches three. The 25% target is therefore **not yet
reached under either reduction**. Configuration sweeps must report both
reductions without renaming shared-prefix agreement or silently switching which reduction is used for the target.

## Corrected-metric damping control

The layers 0/6/10/12 regularization override was originally selected by greedily splicing `0.10`-regularized layer
payloads into an otherwise `0.05` checkpoint on the older teacher-forced proxy. A full corrected-protocol control on
the same pinned development manifest confirms that the layer-specific choice is positive before output alignment:

| Damping checkpoint | Independent aligned-token Top-1 through 32 | Matching positions | Exact trajectories at 32 |
| --- | ---: | ---: | ---: |
| Uniform `0.05` | 12.6042% | 1,210 / 9,600 | 2 / 300 (0.6667%) |
| Layers 0/6/10/12 at `0.10`, all others `0.05` | 13.9896% | 1,343 / 9,600 | 1 / 300 (0.3333%) |
| Uniform `0.10` | **17.8854%** | **1,717 / 9,600** | **3 / 300 (1.0000%)** |
| Uniform `0.125` | 13.4375% | 1,290 / 9,600 | 2 / 300 (0.6667%) |
| Uniform `0.20` | 16.3333% | 1,568 / 9,600 | **3 / 300 (1.0000%)** |

The hybrid gains 133 aligned positions over uniform `0.05`, but uniform `0.10` gains another 374 positions over the
hybrid and 507 over uniform `0.05`. The old proxy-selected dynamic override is therefore positive relative to `0.05`
but is not the corrected-metric optimum. Uniform `0.20` then loses 149 positions relative to `0.10`, bracketing the
best tested damping region instead of supporting still stronger regularization. Exact survival is sparse and does not
rank these arms consistently, so neither reduction is silently substituted for the aligned-token optimization target.

The follow-up `0.125` midpoint also failed: it matched 1,290 positions, losing 427 to `0.10` and 278 to `0.20`.
Behavioral fidelity is therefore not a smooth interpolation of the damping scalar at W2; this arm is negative
evidence, not a reason to average the endpoints. Its per-source aligned-token scores were 12.7500% on LongBench v2,
11.5625% on MathArena, 11.1250% on non-English Multi-IF, 16.5000% on SWE-bench Verified, and 12.3438% on
Terminal-Bench 2.1.

## Teacher-rollout alignment diagnostic

An exact fixed-trellis post-quant arm trained only the existing SU/SV alignment tensors on 32 dense-teacher greedy
continuation tokens for each of the 182 disjoint YAQA prompts. Template and prompt tokens remained in context but were
excluded from the loss. On separate NM slices it moved evaluation KL from `0.273135` to `0.265773` and Top-1 from
`80.6132%` to `81.1481%`, but JSD regressed from `0.059245` to `0.059727`. The strict all-metric gate therefore rejected
the arm and wrote no derived checkpoint. This is directional training evidence, not a promoted Divergence-300 result.
