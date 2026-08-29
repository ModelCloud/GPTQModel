# Llama 3.2 1B Instruct QvQ experiment results

Consolidated report of completed arms and current queued work. All canonical
D300 values below use the pinned divergence-300 development manifest
(`701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b`) and the
independent greedy 32-token protocol. `D300 Top-1` means aligned-token
agreement through horizon 32; exact counts are exact 32-token trajectories.
Regularization labels are the effective W2 `regularization_by_rate` values;
legacy `reg010`/`reg0125` controls retain a separate base value of 0.05 and
override only the W2 rate.

The table is a descriptive ranking on one fixed 300-prompt manifest. The 9,600
token positions are clustered within prompts, so small deltas are not treated
as independent-sample significance; promotion decisions should use paired
prompt-level bootstrap (and paired GSM8K tests where available).
Rows generated before the literal-EOS rollout fix (`55adb7e1`) use the former
EOS-suppressing decoder and are historical until rerun under the corrected
protocol; teacher-forced metrics are unaffected.

Every new checkpoint with a passing strict disjointness contract is also
screened by the fast, task-matched Mini-GSM protocol documented in
[`docs/qvq_micro_math_metrics.md`](qvq_micro_math_metrics.md).  Its immutable
JSON report records numeric-answer rollout accuracy, reasoning ΔCE/ΔKL,
answer-token log-probability and margin retention, critical numeric/operator
Top-1, invalid-answer rate, and paired dense→quantized correctness transitions.
The 128-row Mini-GSM train slice is cryptographically bound and disjoint from
calibration, replay, D300 development/locked, and GSM8K Platinum test rows.
These are screening proxies; full GSM8K Platinum remains the promotion metric.

## Current budget anchors (updated 2026-08-28 UTC)

The following anchors are the comparison points for every future allocation
wave. They are selected by completed full GSM8K Platinum results, not by
Mini-GSM or D300. A new arm may replace an anchor only after its checkpoint,
effective serialized BPW, and full GSM8K report have been verified.

| Budget band | Anchor arm | Allocation | Effective BPW | GSM8K Platinum | Role |
| --- | --- | --- | ---: | ---: | --- |
| ~2.02 | `b5283c` | Flat W2, reg 0.15 | 2.0232 | 23.8213% (288/1209) | flat-W2 reference |
| ~3.02 | `1040a5` | Flat W3 | 3.0232 | 42.9280% (519/1209) | flat-W3 reference |
| ~3.18 | `8980aa` | W3.2 anchor + Up4 layers 6 and 8 | 3.178340 | **45.5749% (551/1209)** | **best ~3.2 anchor** |
| ~3.52 | `9769b1` | Flat W3.5, seed 1 | 3.5232 | **46.65% (564/1209)** | current high-rate reference |
| ~3.56 | `4e7424` | Flat W3.5 + Up4 layers 12–15, seed 1 | 3.5577 | 46.15% (558/1209) | late-Up control |
| ~3.63 | `049d0c` | Flat W3.5 + V4 all + Up4 layers 9–15 | 3.6266 | 45.9884% (556/1209) | V/Up mixed control |

The ~3.2 anchor is the current Wave-5 `L6+L8` result; Wave-6's best
`L8+L9` arm reached 543/1209 at 3.178340 BPW and did not replace it. The
~3.52 seed-1 result remains the highest completed GSM8K count in the current
ledger, while the ~3.6 rows are useful allocation controls rather than an
automatic improvement over flat W3.5.

## Completed arms (ordered by D300 Top-1, descending)

| Arm ID | Family / arm | Rate | Eff. BPW* | Calibration | D300 Top-1 | Exact / 300 | Mean first divergence | GSM8K Platinum | Status |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| a07075 | Q07 uniform | W3 | 3.0232 | YAQA 302k, reg .10 | **33.6667%** | 16 | 9.9133 | 41.27%* | higher-rate reference |
| e616e5 | Q07 uniform | W2.5 | 2.5232 | YAQA 302k, reg .10 | **25.1771%** | 5 | 7.5000 | 33.58%* | higher-rate reference |
| 94710c | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .020 | **18.7813%** | **6** | **5.3633** | 22.08% | **current W2 leader** |
| 2d08b1 | YAQA + output alignment | W2 | 2.0232 | YAQA 302k, reg .010 | 18.4063% | 4 | 5.1667 | 21.67% | D300 complete; alignment arm |
| b5283c | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .015 | 17.9792% | 3 | 5.3200 | 23.82% | D300 complete |
| 6a0ee7 | Full-reference | W2 | 2.0232 | full-reference mix, reg .30 | 17.7292% | 3 | 4.5700 | 19.60% | D300 complete |
| 64876c | Seed control | W2 | 2.0232 | YAQA 302k, seed 1 | 17.7083% | 2 | 4.9667 | not run | complete |
| d64c9f | YAQA + 2-epoch alignment | W2 | 2.0232 | YAQA 302k, reg .05/.10 layers | 17.0938% | 3 | 4.5733 | 22.75% | teacher-forced control |
| 7661a7 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0225 | 16.9271% | 6 | 4.6667 | 21.26% | D300 complete |
| 702443 | YAQA + `mlp.down_proj` W3 | W2 + W3 down | ~2.3565 | AIME mix | 16.7396% | 2 | 4.8267 | 27.79% | mixed-rate diagnostic |
| 048f31 | Fixed-block LDLQ | W2 | 2.0232 | YAQA 302k, reg .005 | 16.3021% | 5 | 4.9133 | 20.43% | complete |
| 7616fa | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .030 | 16.2188% | 5 | 4.4333 | 19.11% | D300 complete |
| 1a6f78 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .025 | 15.3854% | 2 | 4.2667 | 20.60% | D300 complete |
| 62ba43 | YAQA + output alignment | W2 | 2.0232 | YAQA 302k, reg .05/.10 layers | 15.1667% | 2 | 4.5733 | not run | superseded |
| 48252b | layer damping + 2-epoch alignment | W2 | 2.0232 | YAQA 302k, layer-damped | 15.0729% | 2 | 4.6000 | 22.75% | D300 complete |
| d7ee8e | YAQA AIME mix | W2 | 2.0232 | AIME 2526, 216 rows | 14.9792% | 0 | 3.9967 | 17.37% | complete |
| 6c2a99 | YAQA + `mlp.down_proj` W2.5 | W2 + W2.5 down | ~2.1898 | YAQA 302k, reg .020 | 14.8229% | 1 | 4.4267 | 26.88% | mixed-rate diagnostic |
| cd6559 | Full-reference | W2 | 2.0232 | full-reference mix, reg .10 | 14.5417% | 4 | 4.3300 | 21.42% | D300 complete |
| c763d1 | Fixed-block LDLQ | W2 | 2.0232 | YAQA 302k, reg .020 | 14.4583% | 5 | 4.7033 | 23.41% | D300 complete |
| f111c3 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .005 | 14.1042% | 2 | 4.5167 | 20.43% | complete |
| 1a635f | Full-reference | W2 | 2.0232 | full-reference mix, reg .20 | 14.0313% | 4 | 4.1467 | 23.33% | reverified complete |
| 0b4049 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0175 | 13.9271% | 1 | 4.2433 | 19.19% | D300 complete |
| b12aa6 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0125 | 13.4375% | 2 | 3.6233 | 22.50% | complete |

`*` Eff. BPW is logical payload rate plus the common 0.023168-bpw auxiliary
overhead. The mixed W2 + W3-down estimate assumes one-third of projection
weights use W3. GSM8K values for Q07 W2.5/W3 are reported in the existing Q07 ledger;
they are not flat-W2 results. The W2.5/W3 rows are included for rate/fidelity
context only and do not count toward the flat-W2 target.

### Newly completed review follow-ups

| Artifact | Configuration | D300 token top-1 | Exact / 300 | GSM8K Platinum | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `llama32-1b-w2-yaqa302k-reg015-align-tip-gpu1` | YAQA reselect reg 0.15 + output alignment | 14.9271% | 5 | 22.9942% (278/1209) | complete |
| `llama32-1b-w2-yaqa302k-reg015-mlpdown-w25-tip-gpu7` | YAQA reselect reg 0.15 + `mlp.down_proj` W2.5 | 17.8021% | 3 | 23.4078% (283/1209) | complete |

| `llama32-1b-w2-reg015-mlpall-w25` | all MLP projections W2.5 | 19.0833% | 3 | 31.5964% (382/1209) | complete |
| `llama32-1b-w2-reg015-attnall-w25` | all attention projections W2.5 | 20.4167% | 3 | 26.0546% (315/1209) | complete |
| `llama32-1b-w2-yaqa302k-reg015-gateup-w25` | gate/up projections W2.5 | 17.2813% | 2 | 25.0620% (303/1209) | complete |
| `llama32-1b-w2-yaqa302k-reg015-qk-w25` | attention Q/K W2.5 | 16.1042% | 2 | 21.3400% (258/1209) | complete |
| `llama32-1b-w2-reg015-smooth-swiglu` | analytical Smooth-SwiGLU | 17.3021% | 2 | 20.5128% (248/1209) | complete (pre-fix) |
| `llama32-1b-w2-reg015-smooth-swiglu-tip-d68c9d00` | analytical Smooth-SwiGLU, latest native path | 17.3021% | 2 | 20.5128% (248/1209) | complete; latest-code rerun matches pre-fix metrics |

## Invalidated arms

These arms produced numerical results but should not be used for leaderboard
or comparison purposes because the calibration artifact failed the strict
row-level contamination preflight.

| Arm ID | Family / arm | Rate | Eff. BPW* | Calibration | D300 Top-1 | Exact / 300 | Mean first divergence | GSM8K Platinum | Status |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 4aa38f | D300-source-shaped | W2 | 2.0232 | source-shaped 500k, reg .20 | 17.0729% | 5 | 4.9933 | 11.91% | **invalid / contaminated** — calibration artifact `calibration_div300_sources.parquet` failed the disjointness audit (1 normalized D300 overlap, 33 internal duplicate groups) |

## Current leader and protocol

The flat-W2 leader is checkpoint
`/root/qvq-results/llama32-1b-w2-reg020-yaqa302k-main-a12ce4e3`, using
`scripts/configs/llama32_1b_v2b2_p32_yaqa_reg020.json`, ordinary NM rows 0--127,
and YAQA rows 0--181 (302,193 valid tokens). Its canonical D300 score is
18.7813% (1,803/9,600), with 6/300 exact trajectories and mean first
divergence token 5.3633. Its full GSM8K Platinum score is 17.8660%.

## Active, queued, and recently completed follow-ups

| Arm ID | Arm | Eff. BPW | GPU | State | Evaluation |
| --- | ---: | ---: | --- | --- | --- |
| 66c46d | layer damping + output alignment | 2.0232 | 2 | quantization complete; D300 pending | canonical D300 |
| 89c6b9 | YAQA reg .025 + output alignment | 2.0232 | 4 | quantization complete; D300 pending | canonical D300 |
| 4db5cf | fixed-block LDLQ reg .015 | 2.0232 | 1 | queued behind D300 | canonical D300 |
| 5ac42a | fixed-block LDLQ reg .0225 | 2.0232 | 3 | queued behind D300 | canonical D300 |
| 46d985 | clean fixed-block LDLQ reg .020 replica | 2.0232 | 6 | quantization in progress | canonical D300 |
| 37238d | clean fixed-block LDLQ reg .0225 replica | 2.0232 | 7 | quantization in progress | canonical D300 |
| d71136 | V + O projections W2.5, reg .15 | **2.0663** | 1 | D300 complete: **19.1771%**, exact 3/300, mean first divergence 5.0633 | GSM8K complete: **25.4756% (308/1209)** |
| 738a13 | V + O projections W3, reg .15 | **2.1094** | 7 | complete; D300 **17.9271%**, exact 2/300, mean divergence 4.5933 | GSM8K complete: **24.7312%** |
| 09674b | O projection W2.5, reg .15 | **2.0447** | 2 | complete; D300 **17.3021%**, exact 3/300, mean divergence 4.4500 | GSM8K complete: **22.9942%** |
| 0cd45d | V projection W2.5, reg .15 | **2.0447** | 1 | complete; D300 **17.2917%**, exact 2/300, mean divergence 4.6133 | GSM8K complete: **21.4227%** |
| bb0aa2 | gate + down MLP projections W2.5, reg .15 | **2.2990** | 3 | complete; D300 **19.4792%**, exact 7/300, mean divergence 5.4000 | GSM8K complete: **27.9570%** |
| 45a387 | up + down MLP projections W2.5, reg .15 | **2.2990** | 5 | complete; D300 **19.9583%**, exact 3/300, mean divergence 5.0200 | GSM8K complete: **28.1224%** |
| 19d89a | Atomic SwiGLU triplet replay, reg .15 | 2.0232 | 4 | complete; D300 **16.5938%**, exact 1/300, mean divergence 4.2667 | GSM8K complete: **19.9338%** |
| ebec00 | Smooth + Atomic SwiGLU triplet replay, reg .15 | 2.0232 | 6 | complete; D300 **17.6146%**, exact 1/300, mean divergence 4.9933 | GSM8K complete: **22.2498%** |
| 0f642c | Up + Down projections W3, reg .15 | **2.5749** | 0 | complete; D300 **21.7396%**, exact 7/300, mean divergence 5.6300 | GSM8K complete: **30.6865% (371/1209)** |
| 569a95 | Up + Down projections W3.5, reg .15 | **2.8508** | 1 | complete; D300 **21.3021%**, exact 6/300, mean divergence 5.7000 | GSM8K complete: **31.6791% (383/1209)** |
| cdfa75 | Up + Down projections W4, reg .15 | **~3.1266*** | 2 | **failed during quantization** with model-wide `qvq_v4`; trusted CUDA Viterbi rejects V4 sequences; no checkpoint | No D300/GSM8K result |
| ef21af | V + O projections W3, reg .15 | **2.1094** | 3 | complete; D300 **17.9271%**, exact 2/300, mean divergence 4.5933 | GSM8K complete: **24.7312% (299/1209)** |
| 60a68a | V + O projections W3.5, reg .15 | **2.1525** | 4 | complete; D300 **19.0833%**, exact 4/300, mean divergence 5.0133 | GSM8K complete: **26.6336% (322/1209)** |
| 98daf5 | V + O projections W4, reg .15 | **~2.1956*** | 5 | **failed during quantization** with model-wide `qvq_v4`; trusted CUDA Viterbi rejects V4 sequences; no checkpoint | No D300/GSM8K result |
| d13602 | V + O projections W2 + Smooth + Atomic SwiGLU replay, reg .15 | **2.0232** | 6 | complete; D300 **17.6146%**, exact 1/300, mean divergence 4.9933 | GSM8K complete: **22.2498% (269/1209)** |
| b7d172 | V + O projections W2.5 + Smooth + Atomic SwiGLU replay, reg .15 | **2.0663** | 7 | complete; D300 **19.0833%**, exact 4/300, mean divergence 5.9333 | GSM8K complete: **25.3102% (306/1209)** |

### Current 8-GPU mixed-rate matrix (started 2026-08-27 UTC)

All eight arms use the same clean 182-row YAQA mix, ordinary NM rows 0:128,
`qvq_v2b2_p32`/P32 with two banks, reg `.15`, seed 0, and strict benchmark
disjointness. Atomic arms additionally use replay search rows 0:32 and
confirmation rows 32:64 from the disjoint replay source. Paths below are the
authoritative config and checkpoint locations; the six-character Arm ID is
stable across JSON/Markdown logs and evaluator artifacts.

| Arm ID | GPU | Exact configuration | Output checkpoint | Atomic/Smooth | State | D300 | GSM8K Platinum |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| 595f38 | 0 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_updown30.json` | `/root/qvq-results/llama32-1b-w2-reg015-vo35-updown30-595f38` | — | **complete** | **23.7604% (exact 7/300; mean div 7.263)** | **37.3863% (452/1209)** |
| bf96be | 1 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_updown30_atomic.json` | `/root/qvq-results/llama32-1b-w2-vo35-updown30-atomic-bf96be` | Atomic | **complete** | **24.3125% (exact 13/300; mean div 7.260)** | **36.6419% (443/1209)** |
| bf272e | 2 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_updown30_smooth_atomic.json` | `/root/qvq-results/llama32-1b-w2-vo35-updown30-smooth-atomic-bf272e` | Smooth + Atomic | **complete** | **24.3750% (exact 7/300; mean div 7.463)** | **36.5591% (442/1209)** |
| 56c940 | 3 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo25_updown30.json` | `/root/qvq-results/llama32-1b-w2-reg015-vo25-updown30-56c940` | — | **complete** | **21.1458% (exact 9/300; mean div 6.440)** | **34.6567% (419/1209)** |
| e5ca9f | 4 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_updown35.json` | `/root/qvq-results/llama32-1b-w2-reg015-vo35-updown35-e5ca9f` | — | **complete** | **25.5521% (exact 9/300; mean div 7.327)** | **40.2812% (487/1209)** |
| baeb18 | 5 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_mlpall25.json` | `/root/qvq-results/llama32-1b-w2-reg015-vo35-mlpall25-baeb18` | — | **complete** | **21.8750% (exact 8/300; mean div 6.443)** | **35.2357% (426/1209)** |
| 95ef88 | 6 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_updown30_atomic.json` | `/root/qvq-results/llama32-1b-w2-updown30-atomic-95ef88` | Atomic | **complete** | **17.3750% (exact 4/300; mean div 5.357)** | **31.6791% (383/1209)** |
| 4e4d0d | 7 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_updown30_smooth_atomic.json` | `/root/qvq-results/llama32-1b-w2-updown30-smooth-atomic-4e4d0d` | Smooth + Atomic | **complete** | **19.6354% (exact 6/300; mean div 6.157)** | **31.5964% (382/1209)** |

### Current Pareto-frontier allocation sweep (started 2026-08-27 UTC)

These eight arms use the same clean YAQA/NM calibration and strict benchmark
disjointness manifest as the completed matrix above. They are plain
V2B2/P32 reselect quantizations (no Atomic/Smooth). D300 and GSM8K Platinum
are queued automatically after each checkpoint is published. Eff. BPW values
are weighted payload estimates.

| Arm ID | GPU | Precision allocation | Eff. BPW* | Config | Output checkpoint | State | D300 | GSM8K Platinum |
| --- | ---: | --- | ---: | --- | --- | --- | --- | --- |
| 1fee11 | 0 | V+O W2.5 + all MLP W2.5 | **2.4801** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo25_mlpall25.json` | `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo25-mlpall25-1fee11` | **complete** | **22.7917% (exact 8/300; mean div 6.680)** | **35.6493% (431/1209)** |
| b2dee2 | 1 | V+O W3 + all MLP W2.5 | **2.5232** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo30_mlpall25.json` | `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo30-mlpall25-b2dee2` | **complete** | **23.2813% (exact 10/300; mean div 6.680)** | **36.3110% (439/1209)** |
| dc38d2 | 2 | Q+K W2.5 + V+O W3.5 + all MLP W2.5 | **2.6094** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_qk25_vo35_mlpall25.json` | `/root/qvq-results/llama32-1b-w2-reg015-pareto-qk25-vo35-mlpall25-dc38d2` | **complete** | **24.9063% (exact 11/300; mean div 7.580)** | **37.0554% (448/1209)** |
| 8dd86a | 3 | V+O W3.5 + Gate W2.5 + Up/Down W3 | **2.8421** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_gate25_updown30.json` | `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate25-updown30-8dd86a` | **complete** | **27.6250% (exact 17/300; mean div 8.187)** | **40.5294% (490/1209)** |
| 5fddf5 | 4 | V+O W3.5 + all MLP W3 | **2.9801** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_mlpall30.json` | `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-mlpall30-5fddf5` | **complete** | **32.3854% (exact 19/300; mean div 9.870)** | **40.8602% (494/1209)** |
| 83b60c | 5 | Q+K W2.5 + V+O W3.5 + Up/Down W3.5 | **3.0232** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_qk25_vo35_updown35.json` | `/root/qvq-results/llama32-1b-w2-reg015-pareto-qk25-vo35-updown35-83b60c` | **complete** | **28.9167% (exact 12/300; mean div 8.480)** | **39.9504% (483/1209)** |
| 4f018a | 6 | V+O W3.5 + Gate W2.5 + Up/Down W3.5 | **3.1180** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_gate25_updown35.json` | `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate25-updown35-4f018a` | **complete** | **31.9896% (exact 20/300; mean div 9.097)** | **43.0108% (520/1209)** |
| 8e67f6 | 7 | V+O W3.5 + Gate W3.5 + Up/Down W3 | **3.1180** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_gate35_updown30.json` | `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate35-updown30-8e67f6` | **complete** | **31.3021% (exact 17/300; mean div 9.523)** | **39.7022% (480/1209)** |

### Flat-rate projection baselines (queued 2026-08-27 UTC)

These controls quantize all attention projections (Q/K/V/O) and all MLP
projections (gate/up/down) at one common V2B2/P32 rate. They use the same
clean YAQA/NM slices, reg `.15`, seed 0, and strict benchmark disjointness.
W1.5 is within the V2B2 supported W1--W3.5 rate range. Evaluations are
scheduled automatically after each checkpoint completes.

| Arm ID | GPU | Flat rate | Eff. BPW* | Config | Output checkpoint | State | D300 | GSM8K Platinum |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 88be04 | 0 | W2 | **2.0232** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_flat_w20.json` | `/root/qvq-results/llama32-1b-flat-flat-w20-88be04` | **complete** | **12.9063% (exact 2/300; mean div 3.793)** | **19.9338% (241/1209)** |
| 509b7f | 1 | W2.5 | **2.5232** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_flat_w25.json` | `/root/qvq-results/llama32-1b-flat-flat-w25-509b7f` | **complete** | **25.7917% (exact 13/300; mean div 7.383)** | **34.9876% (423/1209)** |
| 1040a5 | 2 | W3 | **3.0232** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_flat_w30.json` | `/root/qvq-results/llama32-1b-flat-flat-w30-1040a5` | **complete** | **28.7292% (exact 13/300; mean div 8.580)** | **42.9280% (519/1209)** |
| b72667 | 3 | W3.5 | **3.5232** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_flat_w35.json` | `/root/qvq-results/llama32-1b-flat-flat-w35-b72667` | **complete** | **39.5521% (exact 40/300; mean div 12.080)** | **45.2440% (547/1209)** |
| 862367 | 4 | W1.5 | **1.5232** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_flat_w15.json` | `/root/qvq-results/llama32-1b-flat-flat-w15-862367` | **complete** | **5.1354% (exact 0/300; mean div 1.703)** | **3.4739% (42/1209)** |

### W3-anchor reallocation sweep (queued 2026-08-27 UTC)

These six plain V2B2/P32 reselect arms keep the total rate near the flat-W3
anchor while moving bits between Q/K, V/O, and the SwiGLU projections. They
use the same clean YAQA/NM slices, reg `.15`, seed 0, and strict benchmark
disjointness. D300 and GSM8K Platinum are scheduled after quantization.

| Arm ID | GPU | Precision allocation | Eff. BPW* | Config | Output checkpoint | State | D300 | GSM8K Platinum |
| --- | ---: | --- | ---: | --- | --- | --- | --- | --- |
| add422 | 0 | Q/K W2.5, V/O W3.5, Gate/Up/Down W3 | **3.0232** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk25_vo35_mlp3.json` | `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp3-add422` | **quantizing** | pending | pending |
| 3151c6 | 1 | Q/K W3, V/O W3.5, Gate/Up/Down W3 | **3.0663** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk3_vo35_mlp3.json` | `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp3-3151c6` | **quantizing** | pending | pending |
| 2ae00f | 4 | Q/K W2.5, V/O W3.5, Gate W3, Up W3.5, Down W3 | **3.1611** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk25_vo35_gate3_up35_down3.json` | `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up35-down3-2ae00f` | **quantizing** | pending | pending |
| 457643 | 5 | Q/K W2.5, V/O W3.5, Gate/Up W3, Down W3.5 | **3.1611** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk25_vo35_gate3_up3_down35.json` | `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up3-down35-457643` | **quantizing** | pending | pending |
| b3bdcd | 6 | Q/K W2.5, V/O W3.5, Gate/Up/Down W3.5 | **3.2990** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk25_vo35_mlp35.json` | `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp35-b3bdcd` | **quantizing** | pending | pending |
| 370e9f | 7 | Q/K W3, V/O W3.5, Gate/Up/Down W3.5 | **3.3421** | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk3_vo35_mlp35.json` | `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp35-370e9f` | **queued (GPU busy)** | pending | pending |

For historical traceability, an earlier monitor snapshot (2026-08-26 UTC)
listed three replay quantizations with evaluator wrappers reserved for each
output. They are not current active jobs; current arms are listed above:

| Artifact ID | Active checkpoint | Quantization | Evaluation queue |
| --- | --- | --- | --- |
| `f6c9e1` | `llama32-1b-w2-reg020-all-subset-replay-disjoint-tip-gpu5` | active (GPU 0) | D300 pending behind quantization |
| `d3eac8` | `llama32-1b-w2-reg020-replay-aggressive-tip-gpu3` | active (GPU 0) | D300 pending behind quantization |
| `a9a4df` | `llama32-1b-w2-reg020-replay-256x256-tip-gpu2` | active (GPU 2) | D300 pending behind quantization |

These historical rows remain separate from the completed-arm leaderboard.
The monitor state file (`docs/experiments/qvq_eval_monitor_state.json`) is the
source of truth for transitions from active/queued to complete/failed, while
the artifact inventory above records the corresponding filesystem state.

The W4 rows use the model-wide `qvq_v4` codec because `qvq_v2b2_p32` rejects
dynamic rates above W3.5. Their starred BPW values are weighted payload
estimates only and are not directly comparable to the segmented V2B2 rows.

## Complete artifact inventory (live reconciliation)

### Corrected mixed-format high-rate reruns (2026-08-28)

The original high-rate evaluations that returned all-zero/invalid outputs are
excluded. These reruns use the corrected mixed-format loader and independent
GPU-pinned evaluation processes. GSM8K Platinum has 1,209 held-out questions;
D300 uses the disjoint 300-prompt development manifest. Arm IDs below are the
stable cross-reference keys for the companion JSON artifact.

| Arm ID | Allocation | Eff. BPW* | GSM8K Platinum | D300 aligned top-1 | Exact@32 | Mean first divergence |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 7c91e2 | Up W4, layers 8–15 | **3.1611** | 42.9280% (519/1209) | 34.7083% | 20/300 | 10.76 |
| 4a6d1f | Up W5, layers 12–15 | **3.1611** | 41.5219% (502/1209) | 33.2813% | 20/300 | 10.20 |
| c83b70 | Up W7, layers 14–15 | **3.1611** | 41.6873% (504/1209) | 32.1771% | 18/300 | 9.93 |
| e2f49a | Up W4 L12–13 + W6 L14–15 | **3.1611** | 42.3490% (512/1209) | 32.6875% | 19/300 | 10.02 |
| 91d5c4 | Flat W3.5 + Up W4 L12–15 | **3.5577** | 45.5749% (551/1209) | 38.8333% | 38/300 | 11.71 |
| b6a028 | Flat W3.5 + Up W4.5 L14–15 | **3.5577** | 44.5823% (539/1209) | 40.1354% | 35/300 | 11.85 |
| f04e77 | Flat W3.5 + Up W5.5 L15 | **3.5577** | 44.2514% (535/1209) | 39.9479% | 37/300 | 11.84 |

\*Effective BPW is computed from the exact per-module logical rates weighted by
Llama-3.2-1B projection parameter counts, plus the measured common auxiliary
overhead of 0.023168 bpw. The W4+ modules use ordinary QVQ V2 geometry; valid
L16/V2 rates have the same raw rate accounting as V2B2/P32. For example, the
first four arms are exactly 3.161099 bpw (reported as 3.1611), and the final
three are exactly 3.557651 bpw (reported as 3.5577).

This inventory is generated from `/root/qvq-results/llama32-1b*` checkpoint
directories and is intended to include every quantization artifact, including
experiments that have not yet produced canonical evaluations. `active` means a
quantizer or evaluator process currently references the artifact; `pending`
means no result file has been observed yet. The six-character path keys are
stable local cross-reference IDs for artifacts that do not yet have a ledger Arm
ID.

### Corrected Wave-1 dynamic-allocation reruns (2026-08-28)

The first Wave-1 batch was invalid because broad dynamic rules shadowed the
arm-specific rules. These fresh checkpoints were generated after fixing
first-match precedence, adding the W4 YAQA regularization override (`0.02`),
and validating resolved allocation fingerprints. Quantization, the 64-row
disjoint micro-math probe, and full GSM8K Platinum evaluation are complete;
canonical D300 remains pending.

| Arm ID | Allocation | Eff. BPW | Quant | Micro-math exact | ΔCE | ΔKL | Answer logprob Δ | Critical top-1 | D300 | GSM8K Platinum |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `fd7dfc` | Up W4, layers 12–15 | **3.1956** | complete | 3.125% | 0.01725 | 0.04014 | +0.00476 | 97.19% | **35.2708% (19/300 exact)** | **41.8528% (506/1209)** |
| `1e2a4d` | Up W4, layers 10–13 | **3.1956** | complete | 7.8125% | 0.01569 | 0.04056 | −0.04216 | 97.35% | **35.3333% (18/300 exact)** | **42.3490% (512/1209)** |
| `fe7561` | Up W4, layers 8–11 | **3.1956** | complete | 3.125% | 0.01944 | 0.04051 | −0.03588 | 97.09% | **35.2604% (18/300 exact)** | **43.0935% (521/1209)** |
| `f65579` | Up W4, layers 8,10,12,14 | **3.1956** | complete | 4.6875% | 0.02075 | 0.04066 | −0.17996 | 97.22% | **35.3958% (20/300 exact)** | **42.0182% (508/1209)** |
| `0f1294` | Up W4 + Down W3.5, layers 14–15 | **3.1956** | complete | 7.8125% | 0.01883 | 0.04058 | −0.01077 | 97.09% | **35.7396% (20/300 exact)** | **42.6799% (516/1209)** |
| `47f9f1` | Up W4 + Gate W3.5, layers 14–15 | **3.1956** | complete | 7.8125% | 0.01935 | 0.04109 | −0.14875 | 97.16% | **37.3438% (20/300 exact)** | **42.0182% (508/1209)** |
| `b82726` | O W4, all layers | **3.1956** | complete | 6.25% | 0.02224 | 0.03841 | −0.00970 | 97.28% | **34.8646% (16/300 exact)** | **43.7552% (529/1209)** |
| `c58c0a` | V W4 + Up W4, layers 13–15 | **3.1956** | complete | 6.25% | 0.01277 | 0.03994 | +0.04268 | 97.00% | **37.1146% (20/300 exact)** | **42.7626% (517/1209)** |

Machine-readable results and report paths are in
`docs/experiments/frontier_wave1_fixed_micro_math_results.json`. The prior
shadowed Wave-1 artifacts and their identical GSM8K results remain invalid and
are intentionally not merged into this table.

| Artifact ID | Checkpoint / experiment | Quant | D300 | GSM8K Platinum |
| --- | --- | --- | --- | --- |
| `ff5436` | `llama32-1b-v2b2p32-align2e64-e2e-susv-r128n300-v1` | completed | completed | completed |
| `107daa` | `llama32-1b-v2b2p32-align2e64-e2e-susv-yaqa182-v1` | completed | completed | completed |
| `20f598` | `llama32-1b-v2b2p32-yaqa322k-effective-reg010-main-54ccd365` | completed | pending | completed |
| `0a0dc5` | `llama32-1b-v2b2p32-yaqa322k-fixed-reg005-pr34-bb1c250e` | completed | pending | completed |
| `f303b2` | `llama32-1b-v2b2p32-yaqa322k-hybrid-attn005-mlp010` | completed | pending | completed |
| `cdeee7` | `llama32-1b-v2b2p32-yaqa322k-hybrid-attn010-mlp005` | completed | pending | completed |
| `4fefa2` | `llama32-1b-v2b2p32-yaqa322k-layerdamp-align-main-f6375fa5` | completed | pending | completed |
| `043c3f` | `llama32-1b-v2b2p32-yaqa322k-layerdamp-align2e64-main-2f34e1da` | completed | pending | completed |
| `bd20aa` | `llama32-1b-v2b2p32-yaqa322k-layerhybrid-r005-r010-dev32` | completed | pending | completed |
| `3591fa` | `llama32-1b-v2b2p32-yaqa322k-mlpdown-w25-main-ecf7081e` | completed | pending | completed |
| `3c871e` | `llama32-1b-v2b2p32-yaqa322k-reg0025-main-54ccd365` | completed | pending | completed |
| `1dd616` | `llama32-1b-v2b2p32-yaqa322k-reg005-align-main-54ccd365` | completed | pending | completed |
| `9536f5` | `llama32-1b-v2b2p32-yaqa322k-reg005-main-54ccd365` | completed | pending | completed |
| `de750c` | `llama32-1b-v2b2p32-yaqa322k-reg010-align-main-ecf7081e` | completed | pending | completed |
| `c80256` | `llama32-1b-v2b2p32-yaqa322k-reg010-attention-only-main-ecf7081e` | completed | pending | completed |
| `5072f6` | `llama32-1b-v2b2p32-yaqa322k-reg010-attention-replay-main-ecf7081e` | completed | pending | completed |
| `1c69af` | `llama32-1b-v2b2p32-yaqa322k-reg010-chatw97-main-d7eae64a` | completed | pending | completed |
| `83093c` | `llama32-1b-v2b2p32-yaqa322k-reg010-mlp-only-main-ecf7081e` | completed | pending | completed |
| `0f21eb` | `llama32-1b-v2b2p32-yaqa322k-reg010-seed1-main-ecf7081e` | completed | pending | completed |
| `659ec3` | `llama32-1b-v2b2p32-yaqa322k-reg010-spectral-main-ecf7081e` | completed | pending | completed |
| `b8fde6` | `llama32-1b-v2b2p32-yaqa322k-reg0125-main-8ae2db25` | completed | pending | completed |
| `7e67aa` | `llama32-1b-v2b2p32-yaqa322k-reg020-main-ecf7081e` | completed | pending | completed |
| `3c7eab` | `llama32-1b-v2b2p32-yaqa322k-rolehybrid-qk010-rest005-dev32` | completed | pending | completed |
| `4c7cf9` | `llama32-1b-v2b2p32-yaqa322k-w25-reg010-pruneauto-main-329cc0a5` | completed | completed | completed |
| `14ef78` | `llama32-1b-v2b2p32-yaqa322k-w30-reg010-pruneauto-main-329cc0a5` | completed | completed | completed |
| `acb522` | `llama32-1b-w2-aime2526-mlpdown-w3-main-fdfd760e` | completed | completed | completed |
| `d65704` | `llama32-1b-w2-aime2526-yaqa-mix-main-1f6c2132` | completed | completed | completed |
| `fa6724` | `llama32-1b-w2-div300-sources-500k-mlpdown-w3-main-4f0d767e` | completed | completed | completed |
| `838950` | `llama32-1b-w2-div300-sources-500k-reg010-main-41106a47` | completed | completed | completed |
| `d9937c` | `llama32-1b-w2-div300-sources-500k-reg020-main-5544143c` | completed | completed | completed |
| `17a515` | `llama32-1b-w2-div300-sources-500k-reg030-main-41106a47` | completed | completed | completed |
| `a9ce03` | `llama32-1b-w2-fixed-reg015-gpu1` | completed | completed | completed |
| `375733` | `llama32-1b-w2-fixed-reg020-gpu2` | completed | completed | completed |
| `1f01e7` | `llama32-1b-w2-fixed-reg020-gpu6` | completed | completed | completed |
| `51bc8a` | `llama32-1b-w2-fixed-reg020-tip-gpu0` | completed | completed | completed |
| `bbcf3e` | `llama32-1b-w2-fixed-reg0225-gpu3` | completed | completed | completed |
| `bb21bc` | `llama32-1b-w2-fixed-reg0225-gpu7` | completed | completed | completed |
| `6ae421` | `llama32-1b-w2-fixedreg005-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `aa2c5f` | `llama32-1b-w2-full-reference-reg010-main-5544143c` | completed | completed | completed |
| `83261e` | `llama32-1b-w2-full-reference-reg020-main-5544143c` | completed | completed | completed |
| `2d8dc3` | `llama32-1b-w2-full-reference-reg030-main-5544143c` | completed | completed | completed |
| `e0344c` | `llama32-1b-w2-mlp-down-w25-reg020-gpu4` | completed | completed | completed |
| `02c72f` | `llama32-1b-w2-reg0025-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `3ad097` | `llama32-1b-w2-reg005-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `39bff7` | `llama32-1b-w2-reg0125-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `c15676` | `llama32-1b-w2-reg020-replay-256x256-tip-gpu6` | active | pending | pending |
| `c7ef00` | `llama32-1b-w2-reg020-replay-256x256-tip-gpu7` | active | pending | pending |
| `2c5e38` | `llama32-1b-w2-reg020-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `c86bfc` | `llama32-1b-w2-seed1-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `20bc50` | `llama32-1b-w2-shiftcal128-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `899803` | `llama32-1b-w2-spectral-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `169f36` | `llama32-1b-w2-yaqa-cal322k-main-de25f203` | completed | completed | completed |
| `635a92` | `llama32-1b-w2-yaqa-main-de25f203` | completed | completed | completed |
| `a0ac7a` | `llama32-1b-w2-yaqa-optmix322k-main-de25f203` | completed | completed | completed |
| `97c8a8` | `llama32-1b-w2-yaqa-optmix551k-main-54ccd365` | completed | completed | completed |
| `05ede7` | `llama32-1b-w2-yaqa302k-layerdamp-align-gpu2` | completed | pending | completed |
| `1f126d` | `llama32-1b-w2-yaqa302k-layerdamp-align2e64-gpu3` | completed | completed | completed |
| `61047f` | `llama32-1b-w2-yaqa302k-reg010-align-gpu1` | completed | completed | completed |
| `6491a4` | `llama32-1b-w2-yaqa302k-reg015-main-5cfb8314` | completed | completed | completed |
| `872b3e` | `llama32-1b-w2-yaqa302k-reg0175-main-5cfb8314` | completed | completed | completed |
| `0cb22d` | `llama32-1b-w2-yaqa302k-reg020-align-main-ff906097` | completed | completed | completed |
| `acc5d3` | `llama32-1b-w2-yaqa302k-reg020-sensitive-hi-main-39e26ebc` | completed | completed | completed |
| `45b81b` | `llama32-1b-w2-yaqa302k-reg0225-main-5cfb8314` | completed | completed | completed |
| `315984` | `llama32-1b-w2-yaqa302k-reg025-align-gpu4` | completed | pending | completed |
| `43db13` | `llama32-1b-w2-yaqa302k-reg025-main-5cfb8314` | completed | completed | completed |
| `603c9a` | `llama32-1b-w2-yaqa302k-reg030-main-5cfb8314` | completed | completed | completed |

## Arm lookup

The Arm ID is the stable key shared by this report, the JSON ledgers, and
queued-run notes. Metrics are in the row above; configuration and checkpoint
locations are indexed here.

| Arm IDs | Configuration / checkpoint source |
| --- | --- |
| a07075, e616e5 | Q07 rate-reference ledger and checkpoints under `/root/qvq-results/`; see the Q07 experiment ledger in `docs/experiments/`. |
| 94710c, b5283c, 64876c, 7661a7, 7616fa, 1a6f78, 62ba43, f111c3, 0b4049, b12aa6 | YAQA regularization configs/checkpoints under `scripts/configs/` and `/root/qvq-results/`. |
| 6a0ee7, cd6559, 1a635f | Full-reference sweep ledger in `docs/experiments/`; checkpoints under `/root/qvq-results/`. |
| d64c9f | Alignment sweep ledger `docs/experiments/2026-08-26-llama32-w2-alignment-sweep.json`. |
| 4aa38f | Source-shaped 500k ledger in `docs/experiments/`; checkpoint under `/root/qvq-results/`. **Invalidated**: the 993-row `calibration_div300_sources.parquet` artifact used by this arm failed the row-level disjointness audit and is not eligible for quantization/evaluation. |
| 702443, d7ee8e | AIME/precision ledger in `docs/experiments/`; checkpoints under `/root/qvq-results/`. |
| 048f31 | Fixed-block LDLQ ledger in `docs/experiments/`; checkpoint under `/root/qvq-results/`. |
| 2d08b1 | Checkpoint `/root/qvq-results/llama32-1b-w2-yaqa302k-reg010-align-gpu1`; metrics `/root/qvq-results/llama32-1b-w2-yaqa302k-reg010-align-gpu1-div300-dev-v1.json`. |
| 66c46d, 48252b, 89c6b9 | Alignment sweep ledger `docs/experiments/2026-08-26-llama32-w2-alignment-sweep.json`. |
| c763d1, 6c2a99, 4db5cf, 5ac42a | Follow-up ledger `docs/experiments/2026-08-26-llama32-w2-fixed-down-followup.json`; configs under `scripts/configs/`. |
| 46d985 | Clean GPU6 fixed-block LDLQ reg .020 replica; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_fixed_reg020.json`; checkpoint `/root/qvq-results/llama32-1b-w2-fixed-reg020-gpu6`. |
| 37238d | Clean GPU7 fixed-block LDLQ reg .0225 replica; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_fixed_reg0225.json`; checkpoint `/root/qvq-results/llama32-1b-w2-fixed-reg0225-gpu7`. |
| d13602, b7d172 | V+O W2/W2.5 + Smooth + Atomic follow-up arms; configs `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo_w20_smooth_atomic.json` and `...vo_w25_smooth_atomic.json`; checkpoints `/root/qvq-results/llama32-1b-w2-vo-w20-smooth-atomic-d13602` and `/root/qvq-results/llama32-1b-w2-vo-w25-smooth-atomic-b7d172`. |

The authoritative per-arm machine-readable ledgers are in
`docs/experiments/`; the append-only chronology is
`docs/qvq_llama32_1b_experiment_log_2026-08-25.md`.

## Data-contamination audit

Calibration/evaluation separation is now enforced by
`scripts/check_calibration_disjointness.py`. The YAQA-182 and full-reference
292-row mixes both pass normalized user-question checks against the D300
development and locked splits and all 1,209 GSM8K Platinum test rows; D300 and
GSM8K also have no normalized collisions. Manifests:
`docs/experiments/disjointness-yaqa182.json` and
`docs/experiments/disjointness-full-reference.json`. Supplying a manifest now
validates its selected-input bindings; benchmark wrappers additionally pass
`--require-disjointness` so quantization fails closed when the manifest is
missing. Historical full-reference
GSM8K scores pass the available row-level audit and are valid on that evidence.
Semantic similarity from derived datasets (for example OpenMathInstruct) is a
separate, optional source-level audit and is not evidence of contamination.

The first D300-source-shaped 500k artifacts failed this audit: the original
993-row mix had a development collision and 33 internal duplicate groups, and
the subsequently named 959-row filtered artifact still has 56 normalized
collisions with the locked split. Neither is eligible for quantization or
evaluation (`docs/experiments/disjointness-div300-sources-disjoint.json` is
intentionally marked `fail`). The source builder now excludes both D300
development and locked prompts; its regenerated 981-row output passes the
strict audit at `docs/experiments/disjointness-div300-sources-v2.json`.
Arm `4aa38f`, which used the ineligible 993-row artifact, is therefore listed
in the "Invalidated arms" section above and should not be compared as a
completed flat-W2 result.
## Corrected Wave-2 matched-budget rerun (2026-08-28 UTC)

The first Wave-2 launch was invalid because stale configurations required 2,000
YAQA sequences while the run supplied 182. It produced no checkpoints or
metrics. The corrected launch uses regenerated configs with
`minimum_sequences=182`, W4 regularization `0.02`, the corrected dynamic-rule
precedence, and the seed-1 control. A launcher preflight now refuses any config
whose sequence floor exceeds the supplied YAQA slice.

| Arm ID | GPU | Allocation | Effective BPW | State | D300 | GSM8K Platinum |
| --- | ---: | --- | ---: | --- | --- | --- |
| `42c2fc` | 0 | Flat W3.5 + Up W4 layers 8–15 | 3.5921 | evaluation complete (Mini-GSM 4.69%) | pending | **44.25% (535/1209)** |
| `e55f4d` | 1 | Flat W3.5 + Up W4 layers 6–15 | 3.6094 | evaluation complete (Mini-GSM 3.13%) | pending | **45.82% (554/1209)** |
| `428e4d` | 2 | Flat W3.5 + Up W4 layers 4–15 | 3.6266 | evaluation complete (Mini-GSM 1.56%) | pending | **45.57% (551/1209)** |
| `a7e34b` | 3 | Flat W3.5 + Up/Down W4 layers 12–15 | 3.5921 | evaluation complete (Mini-GSM 3.13%) | pending | **44.00% (532/1209)** |
| `7cf6ca` | 4 | Flat W3.5 + Up/Gate W4 layers 12–15 | 3.5921 | evaluation complete (Mini-GSM 7.81%) | pending | **43.59% (527/1209)** |
| `f7f157` | 5 | Flat W3.5 + Up W4 layers 12–15 + O W4 all | 3.6008 | evaluation complete (Mini-GSM 4.69%) | pending | **44.42% (537/1209)** |
| `049d0c` | 6 | Flat W3.5 + Up W4 layers 9–15 + V W4 all | 3.6266 | evaluation complete (Mini-GSM 4.69%) | pending | **45.99% (556/1209)** |
| `9769b1` | 7 | Flat W3.5 seed-1 control | 3.5232 | evaluation complete (Mini-GSM 3.13%) | pending | **46.65% (564/1209)** |

Machine-readable status and cross-references: `docs/experiments/frontier_wave2_rerun_20260828.json` and `docs/experiments/arm_id_index.json`.

## Wave-3 control/reallocation queue (2026-08-28 UTC)

These eight arms were queued behind the corrected Wave-2 quantizations; six
unique allocations have now completed Mini-GSM and GSM8K Platinum, and two
duplicate allocations are reused from Wave-2.
The batch includes a current-
code replication of the historical `2ae00f` allocation, exact-budget O/Up
reallocations, and high-frontier late-Up breadth/seed controls. Mini-GSM and
full GSM8K Platinum are required after each checkpoint; D300 remains
diagnostic.

| Arm ID | GPU | Allocation | Target BPW | State | Mini-GSM | GSM8K Platinum |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| `b6429f` | 0 | Current-code QK2.5 / VO3.5 / G3 / U3.5 / D3 replication | 3.1611 | evaluation complete | 4.69% (3/64) | **42.85% (518/1209)** |
| `42a659` | 1 | O4 all, early Up W3 | 3.1611 | evaluation complete | 6.25% (4/64) | **42.18% (510/1209)** |
| `5dc744` | 2 | O4 all, early Gate W2.5 | 3.1611 | evaluation complete | 4.69% (3/64) | **42.02% (508/1209)** |
| `286d6d` | 3 | O4 all, Q W2, K W2.5 | 3.1611 | evaluation complete | 4.69% (3/64) | **43.59% (527/1209)** |
| `4e7424` | 4 | Flat W3.5, Up W4 layers 12–15, seed 1 | 3.5577 | evaluation complete | 3.13% (2/64) | **46.15% (558/1209)** |
| `d64179` | 5 | Flat W3.5, Up W4 layers 8–15 | 3.5921 | reused Wave-2 arm `42c2fc` | 4.69% (3/64) | **44.25% (535/1209)** |
| `d0be49` | 6 | Flat W3.5, Up W4 layers 7–15 | 3.6008 | evaluation complete | 3.13% (2/64) | **44.09% (533/1209)** |
| `e9c194` | 7 | Flat W3.5, Up W4 layers 12–15 plus O4 all | 3.5921 | reused Wave-2 arm `f7f157` | 4.69% (3/64) | **44.42% (537/1209)** |

Wave-3 GSM8K reports are recorded in
`docs/experiments/frontier_wave3_queue_20260828.json`; all six unique runs
completed with zero invalid generations. The seed-1 high-frontier control
(`4e7424`) currently leads this wave at 558/1209 (46.15%).

Machine-readable queue: `docs/experiments/frontier_wave3_queue_20260828.json`.

## Wave-4 W3.2 V/Up sensitivity queue (2026-08-28 UTC)

This wave uses the corrected W3.2 anchor
`Q2 / K2.5 / V3.5 / O4 / Gate3 / Up3.5 / Down3`. All eight arms completed
quantization on GPUs 0–7 between 14:16 and 14:21 UTC and are now queued for
held-out evaluation. Every arm has a distinct
resolved/effective allocation fingerprint.

The plan's original BPW labels treated V3.5→V4-all as a single-layer-sized
increment. Existing serialized frontier accounting indicates that V4-all costs
approximately +0.043103 BPW, so the corrected estimates below are the values
to use for Pareto comparisons. Final serialized accounting remains authoritative.

| Arm ID | GPU | Allocation | Requested BPW | Corrected est. BPW | State |
| --- | ---: | --- | ---: | ---: | --- |
| `c84a1e` | 0 | Anchor, seed 1 | 3.1611 | 3.161099 | GSM8K **43.51% (526/1209)**; Mini-GSM **1.56%** |
| `f1d903` | 1 | Anchor + V4 all | 3.1697 | **3.204202** | GSM8K **44.09% (533/1209)**; Mini-GSM **4.69%** |
| `7b6e2a` | 2 | Anchor + Up4 layer 6 | 3.1697 | **3.169720** | GSM8K **44.25% (535/1209)**; Mini-GSM **4.69%** |
| `a4c918` | 3 | Anchor + Up4 layers 6–7 | 3.1783 | **3.178340** | GSM8K **44.00% (532/1209)**; Mini-GSM **4.69%** |
| `d2f507` | 4 | Anchor + V4 all + Up4 layer 6 | 3.1783 | **3.212823** | GSM8K **43.59% (527/1209)**; Mini-GSM **4.69%** |
| `8e3b61` | 5 | Anchor + Up4 layers 6–8 | 3.1870 | **3.186961** | GSM8K **44.09% (533/1209)**; Mini-GSM **3.13%** |
| `5a0dce` | 6 | Anchor + Up4 layers 6–9 | 3.1956 | **3.195582** | GSM8K **45.00% (544/1209)**; Mini-GSM **3.13%** |
| `b7f294` | 7 | Anchor + V4 all + Up4 layers 6–8 | 3.1956 | **3.230065** | GSM8K **43.18% (522/1209)**; Mini-GSM **3.13%** |

## Wave-5 sparse Up-layer isolation queue (2026-08-28 UTC)

These eight arms isolate the apparent L6/L9 signal from Wave-4 at the same
corrected W3.2 anchor. Full GSM8K Platinum and Mini-GSM are required; D300 is
diagnostic only. Effective BPW values are projection-payload estimates pending
serialized checkpoint accounting.

| Arm ID | GPU | Allocation | Estimated effective BPW | GSM8K Platinum | Mini exact | Answer-logprob Δ | State |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `a91f6c` | 0 | Anchor + Up4 layer 9 | 3.169720 | **44.0033% (532/1209)** | 4.6875% | +0.124061 | complete |
| `3d7e42` | 1 | Anchor + Up4 layers 6 and 9 | 3.178340 | **43.5070% (526/1209)** | 3.1250% | +0.106767 | complete |
| `e8b5a0` | 2 | Anchor + Up4 layers 7 and 8 (matched negative control) | 3.178340 | **45.1613% (546/1209)** | 6.2500% | −0.036569 | complete |
| `8980aa` | 3 | Anchor + Up4 layers 6 and 8 | 3.178340 | **45.5749% (551/1209)** | 4.6875% | −0.019751 | complete |
| `1450c0` | 4 | Anchor + Up4 layers 6 and 10 | 3.178340 | **43.7552% (529/1209)** | 4.6875% | −0.025394 | complete |
| `a63cf6` | 5 | Anchor + Up4 layers 5 and 9 | 3.178340 | **44.5823% (539/1209)** | 4.6875% | +0.086033 | complete |
| `aeb16f` | 6 | Anchor + Up4 layers 6, 9, and 12 | 3.186961 | **43.8379% (530/1209)** | 3.1250% | +0.137476 | complete |
| `a8920f` | 7 | Anchor + Up4 layers 6, 9, 12, and 15 | 3.195582 | **43.1762% (522/1209)** | 3.1250% | +0.129048 | complete |

Machine-readable queue and arm cross-references: `docs/experiments/frontier_wave5_queue_20260828.json` and `docs/experiments/arm_id_index.json`.

## Wave-6 L8 interaction mapping queue (2026-08-28 UTC)

This wave maps standalone and pairwise L8 interactions around the corrected
W3.2 anchor. Full GSM8K Platinum and Mini-GSM are required; D300 is diagnostic
only. Effective BPW values are projection-payload estimates pending serialized
checkpoint accounting.

| Arm ID | GPU | Allocation | Estimated effective BPW | GSM8K Platinum | Mini exact | Answer-logprob Δ | State |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `35d823` | 0 | Anchor + Up4 layer 8 | 3.169720 | **44.8304% (542/1209)** | 6.25% | −0.007874 | complete |
| `4b5abc` | 1 | Anchor + Up4 layers 5 and 8 | 3.178340 | **44.5823% (539/1209)** | 6.25% | −0.048598 | complete |
| `a35e15` | 2 | Anchor + Up4 layers 8 and 9 | 3.178340 | **44.9132% (543/1209)** | 4.69% | +0.067993 | complete |
| `fd5aff` | 3 | Anchor + Up4 layers 8 and 10 | 3.178340 | **44.4169% (537/1209)** | 4.69% | −0.079265 | complete |
| `8b2e24` | 4 | Anchor + Up4 layers 8 and 12 | 3.178340 | **43.8379% (530/1209)** | 6.25% | +0.022460 | complete |
| `7aef62` | 5 | Anchor + Up4 layers 8 and 15 | 3.178340 | **43.9206% (531/1209)** | 6.25% | −0.021847 | complete |
| `18536a` | 6 | Anchor + Up4 layers 6, 8, and 9 | 3.186961 | **44.3342% (536/1209)** | 3.13% | +0.047220 | complete |
| `4f89fc` | 7 | Anchor + Up4 layers 6, 8, and 12 | 3.186961 | **44.7477% (541/1209)** | 4.69% | +0.012833 | complete |

Wave-6 held-out evaluations completed on all eight arms. The best GSM8K
result is `a35e15` (Up4 layers 8 and 9), 543/1209 at 3.178340 BPW; all
reports and Mini-GSM metrics are recorded in the machine-readable queue.

Machine-readable queue and arm cross-references: `docs/experiments/frontier_wave6_queue_20260828.json` and `docs/experiments/arm_id_index.json`.

## Wave-7 full single-layer Up sensitivity sweep (2026-08-28 UTC)

Sixteen arms are queued from the same corrected W3.2 anchor. Each promotes
exactly one `mlp.up_proj` (W3.5 → W4), so every arm has the same estimated
3.169720 BPW and is directly comparable. Two quantizers may run concurrently
per 96-GB GPU; evaluation is exclusive per GPU and follows quantization
immediately. Full GSM8K Platinum and Mini-GSM are required.

| Layer | Arm ID | GPU | Allocation | Estimated BPW | State |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | `e62172` | 0 | Anchor + Up4 L0 | 3.169720 | evaluation complete |
| 1 | `4bab0e` | 1 | Anchor + Up4 L1 | 3.169720 | evaluation complete |
| 2 | `6fba6c` | 2 | Anchor + Up4 L2 | 3.169720 | evaluation complete |
| 3 | `3ff286` | 3 | Anchor + Up4 L3 | 3.169720 | evaluation complete |
| 4 | `e1b702` | 4 | Anchor + Up4 L4 | 3.169720 | evaluation complete |
| 5 | `788234` | 5 | Anchor + Up4 L5 | 3.169720 | evaluation complete |
| 6 | `5f91ab` | 6 | Anchor + Up4 L6 (repeat) | 3.169720 | evaluation complete |
| 7 | `5ea33c` | 7 | Anchor + Up4 L7 | 3.169720 | evaluation complete |
| 8 | `b8fb5c` | 0 | Anchor + Up4 L8 (repeat) | 3.169720 | evaluation complete |
| 9 | `d693ff` | 1 | Anchor + Up4 L9 (repeat) | 3.169720 | evaluation complete |
| 10 | `d6fab2` | 2 | Anchor + Up4 L10 | 3.169720 | evaluation complete |
| 11 | `bd7c59` | 3 | Anchor + Up4 L11 | 3.169720 | evaluation complete |
| 12 | `bb10df` | 4 | Anchor + Up4 L12 | 3.169720 | evaluation complete |
| 13 | `adff86` | 5 | Anchor + Up4 L13 | 3.169720 | evaluation complete |
| 14 | `254d84` | 6 | Anchor + Up4 L14 | 3.169720 | evaluation complete |
| 15 | `515065` | 7 | Anchor + Up4 L15 | 3.169720 | evaluation complete |

### Wave-7 held-out evaluations (2026-08-29 UTC)

All sixteen checkpoints have completed both full GSM8K Platinum (1,209 rows)
and Mini-GSM (64 rows). All arms use the same 3.169720 effective payload BPW.

| Layer | Arm ID | GSM8K Platinum | Mini exact | Answer-logprob Δ | Answer-margin Δ | Reports |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `e62172` | **44.0033% (532/1209)** | 3.1250% | +0.152838 | +0.745247 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_e62172_llama32_1b_frontier_anchor_up4_l0-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_e62172_llama32_1b_frontier_anchor_up4_l0-micro-math-v1.json) |
| 1 | `4bab0e` | **43.8379% (530/1209)** | 3.1250% | +0.149789 | +0.558371 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_4bab0e_llama32_1b_frontier_anchor_up4_l1-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_4bab0e_llama32_1b_frontier_anchor_up4_l1-micro-math-v1.json) |
| 2 | `6fba6c` | **43.9206% (531/1209)** | 4.6875% | +0.092876 | +0.639518 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_6fba6c_llama32_1b_frontier_anchor_up4_l2-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_6fba6c_llama32_1b_frontier_anchor_up4_l2-micro-math-v1.json) |
| 3 | `3ff286` | **44.2514% (535/1209)** | 4.6875% | +0.178998 | +0.534700 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_3ff286_llama32_1b_frontier_anchor_up4_l3-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_3ff286_llama32_1b_frontier_anchor_up4_l3-micro-math-v1.json) |
| 4 | `e1b702` | **44.3342% (536/1209)** | 4.6875% | +0.133087 | +0.830440 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_e1b702_llama32_1b_frontier_anchor_up4_l4-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_e1b702_llama32_1b_frontier_anchor_up4_l4-micro-math-v1.json) |
| 5 | `788234` | **44.7477% (541/1209)** | 4.6875% | +0.102059 | +0.461900 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_788234_llama32_1b_frontier_anchor_up4_l5-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_788234_llama32_1b_frontier_anchor_up4_l5-micro-math-v1.json) |
| 6 | `5f91ab` | **44.0033% (532/1209)** | 1.5625% | +0.133829 | +0.516292 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_5f91ab_llama32_1b_frontier_anchor_up4_l6-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_5f91ab_llama32_1b_frontier_anchor_up4_l6-micro-math-v1.json) |
| 7 | `5ea33c` | **43.4243% (525/1209)** | 3.1250% | +0.175927 | +0.609579 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_5ea33c_llama32_1b_frontier_anchor_up4_l7-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_5ea33c_llama32_1b_frontier_anchor_up4_l7-micro-math-v1.json) |
| 8 | `b8fb5c` | **44.5823% (539/1209)** | 4.6875% | +0.072855 | +0.506915 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_b8fb5c_llama32_1b_frontier_anchor_up4_l8-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_b8fb5c_llama32_1b_frontier_anchor_up4_l8-micro-math-v1.json) |
| 9 | `d693ff` | **44.4169% (537/1209)** | 3.1250% | +0.216500 | +0.677904 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_d693ff_llama32_1b_frontier_anchor_up4_l9-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_d693ff_llama32_1b_frontier_anchor_up4_l9-micro-math-v1.json) |
| 10 | `d6fab2` | **43.5897% (527/1209)** | 4.6875% | +0.077571 | +0.534136 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_d6fab2_llama32_1b_frontier_anchor_up4_l10-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_d6fab2_llama32_1b_frontier_anchor_up4_l10-micro-math-v1.json) |
| 11 | `bd7c59` | **44.2514% (535/1209)** | 4.6875% | +0.203073 | +0.858973 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_bd7c59_llama32_1b_frontier_anchor_up4_l11-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_bd7c59_llama32_1b_frontier_anchor_up4_l11-micro-math-v1.json) |
| 12 | `bb10df` | **44.6650% (540/1209)** | 3.1250% | +0.168476 | +0.730859 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_bb10df_llama32_1b_frontier_anchor_up4_l12-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_bb10df_llama32_1b_frontier_anchor_up4_l12-micro-math-v1.json) |
| 13 | `adff86` | **43.9206% (531/1209)** | 4.6875% | +0.256177 | +0.834301 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_adff86_llama32_1b_frontier_anchor_up4_l13-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_adff86_llama32_1b_frontier_anchor_up4_l13-micro-math-v1.json) |
| 14 | `254d84` | **44.2514% (535/1209)** | 3.1250% | +0.170530 | +0.627562 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_254d84_llama32_1b_frontier_anchor_up4_l14-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_254d84_llama32_1b_frontier_anchor_up4_l14-micro-math-v1.json) |
| 15 | `515065` | **44.5823% (539/1209)** | 4.6875% | +0.145288 | +0.634211 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w7queued_515065_llama32_1b_frontier_anchor_up4_l15-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w7queued_515065_llama32_1b_frontier_anchor_up4_l15-micro-math-v1.json) |

This is a completed-results snapshot; the queue manifest remains authoritative
for the live status and remaining evaluations.

Machine-readable queue and arm cross-references: `docs/experiments/frontier_wave7_queue_20260828.json` and `docs/experiments/arm_id_index.json`.

Machine-readable queue and cross-references: `docs/experiments/frontier_wave4_queue_20260828.json` and `docs/experiments/arm_id_index.json`.

## Wave-8 pair-interaction mapping completed (2026-08-29 UTC)

Sixteen matched-budget arms test pairwise Up4 interactions from the corrected
W3.2 anchor. Each promotes exactly two Up projections from W3.5 to W4, for
3.178340 effective BPW. Positive (`L6+L8`) and negative (`L8+L12`)
interaction controls are intentionally repeated. Full GSM8K Platinum and
Mini-GSM evaluations are required for every checkpoint.

| # | Arm ID | Up4 layers | GPU | Effective BPW | State |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `fb8247` | L5 + L12 | 0 | 3.178340 | evaluation complete |
| 2 | `735365` | L5 + L15 | 1 | 3.178340 | evaluation complete |
| 3 | `e5488a` | L12 + L15 | 2 | 3.178340 | evaluation complete |
| 4 | `6f37dc` | L9 + L12 | 3 | 3.178340 | evaluation complete |
| 5 | `9cd131` | L9 + L15 | 4 | 3.178340 | evaluation complete |
| 6 | `fdbd68` | L6 + L8 (positive control) | 5 | 3.178340 | evaluation complete |
| 7 | `91a008` | L8 + L12 (negative control) | 6 | 3.178340 | evaluation complete |
| 8 | `4e58f5` | L5 + L6 | 7 | 3.178340 | evaluation complete |
| 9 | `c4cc84` | L6 + L12 | 0 | 3.178340 | evaluation complete |
| 10 | `cef938` | L6 + L15 | 1 | 3.178340 | evaluation complete |
| 11 | `fe2ba7` | L5 + L7 | 2 | 3.178340 | evaluation complete |
| 12 | `446290` | L7 + L12 | 3 | 3.178340 | evaluation complete |
| 13 | `719a90` | L7 + L15 | 4 | 3.178340 | evaluation complete |
| 14 | `cecb41` | L4 + L5 | 5 | 3.178340 | evaluation complete |
| 15 | `1e3a24` | L4 + L12 | 6 | 3.178340 | evaluation complete |
| 16 | `864bb1` | L4 + L8 | 7 | 3.178340 | evaluation complete |

Machine-readable queue: `docs/experiments/frontier_wave8_queue_20260829.json`.

### Wave-8 completed evaluations (2026-08-29 UTC)

All sixteen arms completed both full GSM8K Platinum and Mini-GSM evaluations.

| Arm ID | Up4 layers | GSM8K Platinum | Mini exact | Answer-logprob Δ | Answer-margin Δ | Reports |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `fb8247` | L5 + L12 | **42.3490% (512/1209)** | 3.1250% | +0.409967 | +0.070590 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_fb8247_llama32_1b_frontier_anchor_up4_l5_l12-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_fb8247_llama32_1b_frontier_anchor_up4_l5_l12-micro-math-v1.json) |
| `6f37dc` | L9 + L12 | **43.3416% (524/1209)** | 1.5625% | +0.475437 | −0.038672 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_6f37dc_llama32_1b_frontier_anchor_up4_l9_l12-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_6f37dc_llama32_1b_frontier_anchor_up4_l9_l12-micro-math-v1.json) |
| `9cd131` | L9 + L15 | **43.1762% (522/1209)** | 3.1250% | +0.462224 | −0.032238 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_9cd131_llama32_1b_frontier_anchor_up4_l9_l15-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_9cd131_llama32_1b_frontier_anchor_up4_l9_l15-micro-math-v1.json) |
| `4e58f5` | L5 + L6 | **42.2663% (511/1209)** | 4.6875% | +0.338698 | −0.105156 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_4e58f5_llama32_1b_frontier_anchor_up4_l5_l6-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_4e58f5_llama32_1b_frontier_anchor_up4_l5_l6-micro-math-v1.json) |
| `735365` | L5 + L15 | **42.8453% (518/1209)** | 6.25% | +0.391713 | +0.076822 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_735365_llama32_1b_frontier_anchor_up4_l5_l15-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_735365_llama32_1b_frontier_anchor_up4_l5_l15-micro-math-v1.json) |
| `e5488a` | L12 + L15 | **43.4243% (525/1209)** | 4.6875% | +0.456282 | +0.012602 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_e5488a_llama32_1b_frontier_anchor_up4_l12_l15-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_e5488a_llama32_1b_frontier_anchor_up4_l12_l15-micro-math-v1.json) |
| `fdbd68` | L6 + L8 | **42.9280% (519/1209)** | 4.6875% | +0.362137 | −0.217387 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_fdbd68_llama32_1b_frontier_anchor_up4_l6_l8-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_fdbd68_llama32_1b_frontier_anchor_up4_l6_l8-micro-math-v1.json) |
| `91a008` | L8 + L12 | **44.8304% (542/1209)** | 1.5625% | +0.430909 | −0.022157 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_91a008_llama32_1b_frontier_anchor_up4_l8_l12-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_91a008_llama32_1b_frontier_anchor_up4_l8_l12-micro-math-v1.json) |
| `c4cc84` | L6 + L12 | **44.4169% (537/1209)** | 6.25% | +0.407893 | −0.164052 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_c4cc84_llama32_1b_frontier_anchor_up4_l6_l12-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_c4cc84_llama32_1b_frontier_anchor_up4_l6_l12-micro-math-v1.json) |
| `fe2ba7` | L5 + L7 | **42.8453% (518/1209)** | 4.6875% | +0.257663 | −0.178204 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_fe2ba7_llama32_1b_frontier_anchor_up4_l5_l7-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_fe2ba7_llama32_1b_frontier_anchor_up4_l5_l7-micro-math-v1.json) |
| `446290` | L7 + L12 | **44.3342% (536/1209)** | 4.6875% | +0.334293 | −0.228091 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_446290_llama32_1b_frontier_anchor_up4_l7_l12-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_446290_llama32_1b_frontier_anchor_up4_l7_l12-micro-math-v1.json) |
| `719a90` | L7 + L15 | **43.0108% (520/1209)** | 6.25% | +0.318594 | −0.226357 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_719a90_llama32_1b_frontier_anchor_up4_l7_l15-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_719a90_llama32_1b_frontier_anchor_up4_l7_l15-micro-math-v1.json) |
| `cecb41` | L4 + L5 | **40.8602% (494/1209)** | 3.125% | +0.478479 | +0.217460 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_cecb41_llama32_1b_frontier_anchor_up4_l4_l5-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_cecb41_llama32_1b_frontier_anchor_up4_l4_l5-micro-math-v1.json) |
| `1e3a24` | L4 + L12 | **43.0108% (520/1209)** | 3.125% | +0.545433 | +0.179761 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_1e3a24_llama32_1b_frontier_anchor_up4_l4_l12-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_1e3a24_llama32_1b_frontier_anchor_up4_l4_l12-micro-math-v1.json) |
| `864bb1` | L4 + L8 | **43.0108% (520/1209)** | 3.125% | +0.501772 | +0.141465 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_864bb1_llama32_1b_frontier_anchor_up4_l4_l8-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_864bb1_llama32_1b_frontier_anchor_up4_l4_l8-micro-math-v1.json) |
| `cef938` | L6 + L15 | **42.8453% (518/1209)** | 6.25% | +0.393213 | −0.167946 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w8queued_cef938_llama32_1b_frontier_anchor_up4_l6_l15-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w8queued_cef938_llama32_1b_frontier_anchor_up4_l6_l15-micro-math-v1.json) |

## Wave-9 local marginal sweep queued (2026-08-29 UTC)

Eight arms ran from the corrected W3.2 anchor. The first six spend one
additional Up4 or Down3.5 promotion (3.186961 effective BPW); the final two
spend two targeted Down3.5 promotions (3.195582 effective BPW). All arms use
the same 182-row YAQA calibration and require full GSM8K Platinum plus
Mini-GSM evaluation. Queue manifest:
`docs/experiments/frontier_wave9_queue_20260829.json`.

| # | Arm ID | Allocation | GPU | Effective BPW | State |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `5b847c` | Up4 L7 + L8 + L12 | 0 | 3.186961 | evaluation complete |
| 2 | `611404` | Up4 L6 + L7 + L8 | 1 | 3.186961 | evaluation complete |
| 3 | `6fe5ca` | Up4 L8 + L12; Down3.5 L12 | 2 | 3.186961 | evaluation complete |
| 4 | `228564` | Up4 L6 + L8; Down3.5 L8 | 3 | 3.186961 | evaluation complete |
| 5 | `370c3a` | Up4 L8 + L11 + L12 | 4 | 3.186961 | evaluation complete |
| 6 | `0ea650` | Up4 L6 + L8 + L11 | 5 | 3.186961 | evaluation complete |
| 7 | `186871` | Up4 L8 + L12; Down3.5 L8 + L12 | 6 | 3.195582 | evaluation complete |
| 8 | `37eb97` | Up4 L6 + L8; Down3.5 L6 + L8 | 7 | 3.195582 | evaluation complete |

### Wave-9 completed evaluations (2026-08-29 UTC)

All eight Wave-9 arms completed both required held-out evaluations. The queue manifest records the full
micro-math metrics (including answer-logprob and answer-margin deltas):
`docs/experiments/frontier_wave9_queue_20260829.json`.

| Arm ID | Allocation | Effective BPW | GSM8K Platinum | Mini exact | Answer-logprob Δ | Answer-margin Δ | State |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `5b847c` | Up4 L7 + L8 + L12 | 3.186961 | **529/1209 (43.7552%)** | 6.25% | +0.290639 | −0.265506 | evaluation complete |
| `0ea650` | Up4 L6 + L8 + L11 | 3.186961 | **525/1209 (43.4243%)** | 6.25% | +0.387690 | −0.225125 | evaluation complete |
| `611404` | Up4 L6 + L7 + L8 | 3.186961 | **528/1209 (43.6725%)** | 7.8125% | +0.237316 | −0.422956 | evaluation complete |
| `6fe5ca` | Up4 L8 + L12; Down3.5 L12 | 3.186961 | **529/1209 (43.7552%)** | 4.6875% | +0.436002 | −0.064639 | evaluation complete |
| `228564` | Up4 L6 + L8; Down3.5 L8 | 3.186961 | **524/1209 (43.3416%)** | 9.375% | +0.204094 | −0.556634 | evaluation complete |
| `370c3a` | Up4 L8 + L11 + L12 | 3.186961 | **535/1209 (44.2514%)** | 6.25% | +0.458041 | −0.030357 | evaluation complete |
| `186871` | Up4 L8 + L12; Down3.5 L8 + L12 | 3.195582 | **530/1209 (43.8379%)** | 6.25% | +0.283112 | −0.399024 | evaluation complete |
| `37eb97` | Up4 L6 + L8; Down3.5 L6 + L8 | 3.195582 | **528/1209 (43.6725%)** | 6.25% | +0.273898 | −0.297230 | evaluation complete |
