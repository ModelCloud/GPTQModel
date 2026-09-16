# QVQ/P32 GSQ paper-alignment audit

Reference revisions: QVQ `633461bb7`; official GSQ `03fc16484c`. The schedule,
initialization, optimizer and staged-objective comparisons below were checked
against both the arXiv v2 paper and that official implementation revision.

## Finding

The corrected W3/P32 result was not evidence that GSQ improved QVQ. Its 265
accepted tile changes came from a deterministic coordinate sweep over a
Fisher-screened 33-choice tile bank. The continuous Gumbel relaxation improved
zero modules beyond that initializer. GSM8K-Platinum then moved from 537/1209
to 516/1209 on a strictly disjoint evaluation set.

The failure is structural rather than an optimizer-runtime shortage:

1. Paper GSQ assigns five local shifts to every scalar coordinate. The current
   QVQ adapter assigns 33 whole-path choices to each 16x16 tile. Each
   non-baseline choice edits one transition and only six W3 scalar values.
   Final argmax selects at most one such edit per tile, so the adapter can move
   at most 6/256 = 2.34375% of decoded scalars even if every tile changes. The
   paper's W3 ablation reports 15.365% of scalar coordinates moving from their
   initializer, 6.56 times beyond this adaptation's absolute ceiling. On the
   profiled projection, the learned arm moved only 18/65536 tiles, or at most
   108/16,777,216 decoded scalars (0.000644%). P32 can encode composed edge
   changes; the frozen whole-tile candidate parameterization cannot learn them.
2. Candidate shifts are pre-screened by the same aggregate Fisher objective
   later used for fitting. A deterministic coordinate sweep then minimizes the
   same frozen pool before Gumbel training starts.
3. Paper GSQ trains on shuffled minibatches with fresh Gumbel noise and staged
   reconstruction: Q/K linear, V/O attention, then gate/up/down full-block.
   QVQ repeats one aggregate per-projection Kronecker Fisher objective. It has
   no data-dependent optimizer batches, nonlinear attention/MLP signal,
   cross-projection terms, or prefix-error propagation during GSQ.
4. Paper scalar GSQ jointly learns assignments and group scales. QVQ keeps
   transforms, banks, codebooks, and scales fixed. In particular, the existing
   P32 format already stores a floating output scale vector (`SV`), but the GSQ
   fitter neither differentiates it nor returns an updated value.
5. QVQ initialized logits with zeros plus a small winner margin. Paper
   Appendix A uses `0.01 * (Normal(0,1) + 6 * centered(-shift^2/2))` at W3.
6. ArXiv v2 uses 20 dense-Llama block-wise epochs: 4096/64 * 20 = 1280
   updates. The former helper incorrectly used the 10-epoch Kimi schedule.

## Controlled H100 ablation

The real `model.layers.0.mlp.down_proj` projection has shape 8192x2048.

| Run | Hard coordinate prepass | Initialization | Updates | Selected tiles | Fisher objective |
| --- | ---: | --- | ---: | ---: | ---: |
| Old adaptation | no | winner margin | 1280 | 0 | 0.0244182274 -> 0.0244182274 |
| Paper init | no | local-shift Gaussian | 1280 | 18 | 0.0244182274 -> 0.0244181380 |
| Paper init + comparator | one | local-shift Gaussian | 1280 | 37 | 0.0244182274 -> 0.0244180374 |

The unfused two-sample reference path also changed zero tiles with the old
initializer, excluding the fused CUDA graph/Lion kernel as the cause. With
paper initialization, GSQ itself changes 18 legal P32 tiles and passes the
exact hard no-regression guard. The coordinate comparator still wins on the
objective it already greedily optimized, so it is now opt-in rather than the
default GSQ path.

Artifacts:

- `/root/qvq-results/gsq-performance-20260916/1280-bf16-paper-v2-no-coordinate-repeat3.json`
- `/root/qvq-results/gsq-performance-20260916/1280-bf16-paper-v2-no-coordinate-g2-reference.json`
- `/root/qvq-results/gsq-performance-20260916/1280-paper-init-no-coordinate-repeat3.json`
- `/root/qvq-results/gsq-performance-20260916/1280-paper-init-configurable-repeat3.json`
- `/root/qvq-results/gsq-performance-20260916/1280-paper-init-coordinate-repeat3.json`

The configurable repeat is the final post-wiring confirmation: its three
complete-fit timings were 1.7080, 1.0063 and 1.0048 seconds and it reproduced
the same 18-tile payload and objective exactly.

## Corrections in this branch

- Dense-Llama paper schedule defaults to 1,280 updates; Kimi remains available
  by explicitly selecting 10 epochs.
- P32 candidate screening retains each exact transition shift and initializes
  logits with the paper's Gaussian local-shift prior and isotropic noise.
- Deterministic Fisher coordinate search defaults off and remains an explicit
  comparator. Baseline and best-hard guards remain authoritative, so stochastic
  training cannot introduce a Fisher-objective regression merely to report
  more changed tiles.
- Diagnostics identify the effective initialization and continue to distinguish
  updates from epochs.

## Staged reconstruction gate completed

The legal P32 adapter now supports the paper's data-dependent stage order:
dedicated Q/K quadratic fitting, joint V/O attention reconstruction, then joint
gate/up/down full-block reconstruction. It jointly learns categorical P32 edits
and the serialized output scale vector (`SV`). Every accepted state is an exact
round-trippable P32 payload; soft mixtures are never serialized. Successive
rounds rebuild their legal candidate bank from the preceding accepted hard
payload, permitting composed edge edits.

Hard model selection uses token-hash-disjoint FineWeb-Edu validation chunks.
The initial hard state is forced to choice zero while retaining the paper's
shift-centred noisy soft initialization. This invariant matters: without it,
later rounds compare validation loss against random initial hard edits instead
of the preceding accepted model.

On Llama 3.2 1B layer 0, with 2,048 training and 1,024 held-out tokens:

| Schedule | Q/K edits | V/O edits | MLP edits | Held-out full-block MSE |
| --- | ---: | ---: | ---: | ---: |
| 16 Q/K updates, 2 block epochs | 0 / 0 | 0 / 0 | 4,820 / 5,122 / 6,553 | 3.66271e-5 -> 3.23653e-5 (-11.64%) |
| 256 Q/K updates, 2 block epochs | 31 / 9 | 0 / 0 | 2,481 / 2,628 / 3,538 | 3.66271e-5 -> 3.17991e-5 (-13.18%) |
| 256 Q/K updates, 20 block epochs | 31 / 8 | 33 / 224 | 4,503 / 3,966 / 4,717 | 3.66271e-5 -> 2.86996e-5 (-21.64%) |

The update-budget ablation explains the prior zero-edit result. With only 16
updates, the 33-way whole-tile relaxation had insufficient time to move Q/K or
V/O past their exact baseline. At 256 Q/K updates, both Q/K held-out objectives
improve. At the paper's 20 Llama block epochs, V/O also improves and is accepted.

The strongest state was repacked to canonical planar P32, installed into a
copy-on-write checkpoint, and freshly loaded through the SM90 QVQ runtime.
On a third, report-only 1,020-token FineWeb-Edu split, strictly disjoint from
training and validation, it improves forward KL by 2.685%, logit squared error
by 1.837%, cross-entropy by 0.282%, and perplexity by 0.874%. Dense-teacher
argmax agreement falls by 0.294 percentage points, so this is positive but not
universal across every secondary metric.

Artifacts:

- `/root/qvq-results/gsq-staged-20260916/layer0-all7-e20-qk256.json`
- `/root/qvq-results/gsq-staged-20260916/layer0-all7-e20-qk256-state.safetensors`
- `/root/qvq-results/gsq-staged-20260916/checkpoint-layer0-all7-e20-qk256-report-only-eval.json`

## Paper-layout data and all-layer global guard

The reproducible FineWeb-Edu cache now follows the official loader's ordering,
tokenizer behavior, revision, seed, and buffer size. It contains 4,864 packed
4,096-token rows (19,922,944 tokens) with these non-overlapping ranges:

| Role | Start row | Rows | Paper role |
| --- | ---: | ---: | --- |
| reconstruction train | 0 | 4,096 | block-wise GSQ optimization |
| selection validation | 4,096 | 128 | hard checkpoint selection only |
| Q/K metric | 4,224 | 512 | dedicated Q/K quadratic factors |
| report-only evaluation | 4,736 | 128 | final untouched measurement |

The cache is
`/root/qvq-results/gsq-paper-data-20260916/fineweb-edu-seed0-4864x4096.safetensors`
(SHA256 `60e7f6aa74ad5ea1a80a38afa4fbdb10da3086ed1fa18d42348290b054c39c4f`).
All six pairwise split-overlap checks pass. Training, Q/K fitting, model
selection, and final reporting therefore consume disjoint packed rows.

An intermediate all-layer experiment used the correct ranges and the paper's
20 Llama block epochs plus 2,000 Q/K updates, but deliberately reduced each row
to 512 tokens and used 32 train, 8 validation, and 32 Q/K rows. That is 16,384
reconstruction tokens, 4,096 selection tokens, and 16,384 Q/K metric tokens.
It is useful for testing the objective and acceptance machinery, but is only
0.0977% of the paper's 16.78M-token reconstruction budget.

Every preceding layer is reconstructed through its deployable QVQ hard weight.
Accepted GSQ layers use their cumulative hard state; absent or rejected layers
use the source QVQ payload, never a dense substitute. A per-layer transaction
first rejects full-block held-out MSE regressions. A second global transaction
fresh-loads a copy-on-write QVQ checkpoint and accepts a layer only when forward
KL, logit MSE, cross-entropy, perplexity, and top-1 teacher agreement all avoid
regression against the currently accepted prefix.

The strict sweep accepted layers 0, 2, 4, and 5. Layer 1 was rejected after a
large local improvement because the end-to-end checkpoint regressed every
global metric. Layer 15 similarly improved its local full-block MSE by 6.27%
but regressed global KL and logit MSE by 0.386% and 0.401%. Layer 14 failed the
local transaction and was not evaluated globally. These cases demonstrate why
local reconstruction loss alone cannot authorize a deployed edit.

The final cumulative hard state contains 56 tensors across four layers. On all
128 untouched report-only rows (65,408 next-token positions), a fresh SM90 QVQ
load improves every endpoint relative to the source W3/P32 checkpoint:

| Metric | Source QVQ | GSQ layers 0/2/4/5 | Improvement |
| --- | ---: | ---: | ---: |
| forward KL | 0.08958049 | 0.08663259 | 3.2908% |
| logit MSE | 38,550.846 | 37,728.547 | 2.1330% |
| cross-entropy | 2.9421461 | 2.9401446 | 0.0680% |
| perplexity | 18.956485 | 18.918581 | 0.2000% |
| top-1 teacher agreement | 84.7236% | 84.8337% | +0.1101 pp |

Primary artifacts:

- `/root/qvq-results/gsq-paper-global-guard-20260916/candidate-layer-5.safetensors`
- `/root/qvq-results/gsq-paper-global-guard-20260916/final-layers0-2-4-5-report-only.json`
- `/root/qvq-results/gsq-paper-global-guard-20260916/checkpoint-final-layers0-2-4-5-report-only`

The positive untouched-data gate permits downstream GSM8K-Platinum evaluation,
but does not make this a paper-scale training claim. The next accuracy phase is
to run the same guarded pipeline with all 4,096 x 4,096 reconstruction rows.
Expanding from one-edge whole-tile choices to five choices per independent
trellis edge remains a separate representation change.

## Downstream GSM8K-Platinum result

The final guarded checkpoint was evaluated through Evalution's saved
GSM8K-Platinum prompts and scorer using ZML's native SM90 P32 runner, FA2 paged
attention, greedy generation, batch size 8, 8,192-token context, and at most
256 generated tokens. All 1,209 examples completed with zero invalid or
incomplete samples.

| Checkpoint | Correct | Accuracy |
| --- | ---: | ---: |
| source QVQ W3/P32 | 537 / 1,209 | 44.4169% |
| guarded GSQ layers 0/2/4/5 | 540 / 1,209 | 44.6650% |

GSQ gains three correct answers: +0.2481 percentage points, or +0.5587%
relative accuracy. The per-sample audit contains 39 incorrect-to-correct and
36 correct-to-incorrect flips. This is a positive downstream result, consistent
with the untouched final-logit improvements, but it is small and should not be
treated as a statistically robust paper-scale gain. The full-data run and at
least one deterministic repeat remain necessary before making a stronger
accuracy claim.

A post-run provenance replay found that the original training process was not
bitwise deterministic despite private seeded assignment and shuffle generators.
Two same-seed layer-0 replays changed their selected tiles and scales. The
validation driver now enables PyTorch deterministic algorithms, seeds both CPU
and CUDA, and sets cuBLAS workspace mode `:4096:8` before CUDA initialization.
Two corrected replays produced tensor-identical 14-tensor hard states (zero
differing elements) and identical metrics. Deterministic fitting took 78.71 and
79.96 seconds versus about 63.5 seconds previously, a 24-26% reproducibility
cost. `--allow-nondeterministic` remains available for explicitly labeled
exploratory timing runs.

The 540/1,209 checkpoint predates this correction. Its exact hard state and raw
evaluation remain valid preserved measurements, but it is marked experimental
and non-publishable because the early complete training logs were not retained
and it cannot be regenerated bit-for-bit under the old execution mode. The
full-data phase must start from deterministic mode and produce a new artifact.

Artifacts:

- `/root/qvq-results/gsq-paper-global-guard-20260916/evaluations/zml-full1209/evaluation.json`
- `/root/qvq-results/gsq-paper-global-guard-20260916/evaluations/zml-full1209/evaluation.samples.jsonl`
- `/root/qvq-results/gsq-paper-global-guard-20260916/evaluations/zml-full1209/evaluation.log`
