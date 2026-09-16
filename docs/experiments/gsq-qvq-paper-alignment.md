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

## Remaining gate before another full GSM8K run

These changes repair categorical movement but do not make the QVQ adaptation
equivalent to paper GSQ. The next causal experiment must optimize legal P32
assignments under the staged Llama reconstruction objectives already used by
the scalar-GSQ implementation. Use disjoint fixed-length FineWeb-Edu training
and held-out reconstruction samples, then promote a checkpoint to
GSM8K-Platinum only if hard held-out stage loss and propagated final-logit
metrics do not regress. Expanding from one-edge whole-tile choices to five
choices per independent trellis edge is a separate representation change and
must retain exact P32 round-trip and hard-loss guards.

Do not launch another full GSM8K-Platinum comparison from the current
full-Fisher path: it would only retest a known non-paper objective. The staged
held-out gate is the prerequisite for that expensive downstream evaluation.
