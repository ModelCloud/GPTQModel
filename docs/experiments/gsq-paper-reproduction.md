# Reproducing the GSQ paper on Llama 3.2 1B

Scripts in [`scripts/gsq_paper/`](../../scripts/gsq_paper) reproduce the
staged-GSQ comparison from [GSQ: Accurate-Instant Quantization for LLMs](https://arxiv.org/abs/2604.18556)
(Dadgarnia et al., App. G.1) on a single Llama 3.2 1B Instruct checkpoint:
a GPTQ-only arm versus a GPTQ-initialized arm that also runs the staged
reconstruction training.

```bash
# 1. Build the paper's calibration geometry (~2.2 GB, streams FineWeb-Edu)
python scripts/gsq_paper/build_corpus.py --model meta-llama/Llama-3.2-1B-Instruct \
    --out /models/datasets/gsq-paper

# 2. Quantize W2/W3/W4 x {gptq, gsq}, one process per GPU
scripts/gsq_paper/run_all.sh --model meta-llama/Llama-3.2-1B-Instruct \
    --corpus /models/datasets/gsq-paper --out /models/gsq-paper-out \
    --gpus 0,1,2,3,4,5 --bits "2 3 4"

# 3. Held-out NLL comparison
python scripts/gsq_paper/compare_arms.py --corpus /models/datasets/gsq-paper \
    --models /models/gsq-paper-out/w{2,3,4}-gsq-s7/model \
             /models/gsq-paper-out/w{2,3,4}-gptq-s7/model \
    --dense meta-llama/Llama-3.2-1B-Instruct --n 128
```

## Recipe

Everything except the staged-GSQ switch is shared, so the two arms differ only
in whether the reconstruction stages run.

| Setting | Paper (Llama) | `run_arm.py` |
| --- | --- | --- |
| Calibration | FineWeb-Edu, 4096 seqs x 4096 tokens | same, via `build_corpus.py` |
| GPTQ initialization | 512 seqs x 4096 tokens | same |
| Validation | 128 seqs x 4096 tokens | same |
| Epochs / batch / group | 20 / 64 / 128 | same (`--epochs/--batch-size/--group-size`) |
| Q/K updates | 2000 | same (`--qk-steps`) |
| Optimizer | Lion 1e-4 logits, 5e-5 scales, wd 1.0, cosine to 0.1 | same |
| Sampling | temperature 2 -> 0.05, multiplier 100 -> 500 | same |
| W3 prior | strength 6, noise std 0.01 | same |
| Attention | not stated | SDPA |

`build_corpus.py` cuts 4096-token chunks from one shuffled FineWeb-Edu stream,
so the splits are disjoint by source document but not by position. The split
boundaries, seed and shuffle buffer are recorded in `meta.json`.

## Measured results

Llama 3.2 1B Instruct, W2/W3/W4, group size 128, full 4096/512/128 splits,
20 epochs, 2000 Q/K steps. Held-out NLL is measured on the first **32** of the
128 validation documents (131,040 tokens); the tables below are all at that
fixed document set. Zero-shot is the five-task lm-eval suite.

| Arm | NLL | PPL | ARC-C | ARC-E | Hella | PIQA | Wino | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense BF16 | 2.918 | 18.51 | 34.90 | 68.98 | 45.81 | 74.70 | 58.01 | 56.48 |
| W2 GPTQ | 9.293 | 10861 | 20.39 | 24.24 | 26.01 | 53.26 | 48.93 | 34.57 |
| W2 GSQ | 5.898 | 364.3 | 16.38 | 30.05 | 26.41 | 55.66 | 50.28 | 35.76 |
| W3 GPTQ | 3.787 | 44.14 | 27.39 | 56.60 | 38.27 | 65.67 | 51.62 | 47.91 |
| W3 GSQ (A) | 3.842 | 46.63 | 26.96 | 56.73 | 36.09 | 66.54 | 52.01 | 47.67 |
| W3 GSQ (B) | 3.623 | 37.46 | 26.96 | 56.47 | 36.29 | 65.23 | 51.30 | 47.25 |
| W4 GPTQ | 3.566 | 35.36 | 34.98 | 65.89 | 44.06 | 72.36 | 56.75 | 54.81 |
| W4 GSQ | 3.028 | 20.66 | 35.07 | 66.16 | 43.51 | 73.12 | 57.70 | 55.11 |

Deltas against GPTQ at the same bit width:

| Bits | dNLL | dAvg accuracy |
| --- | --- | --- |
| W2 | **-3.395** | **+1.19** |
| W3 (A) | +0.055 | -0.24 |
| W3 (B) | -0.164 | -0.66 |
| W4 | **-0.537** | **+0.30** |

GSQ helps decisively at W2 and clearly at W4. **W3 is not reproducible at this
scale** — see below. The NLL and accuracy directions agree in all four cases,
which is why W3 reads as noise rather than a real gain: the two W3 runs
disagree in sign on NLL while both lose a little accuracy.

### W3 does not regress, it is noise-dominated

Three independent W3 GSQ runs, same recipe:

| Run | NLL | vs GPTQ (3.787) |
| --- | --- | --- |
| N=2048 sequences | 3.752 | -0.035 |
| N=4096 (A) | 3.842 | +0.055 |
| N=4096 (B) | 3.623 | -0.164 |

Spread 0.219 NLL, sd 0.110, against a nominal effect of 0.05. GPTQ sits inside
that spread. An earlier reading that reported W3 as a 6.3% regression was
withdrawn after run (B) landed on the opposite side of GPTQ.

Per-document NLL over 32 validation documents shows the mechanism: two
documents (indices 7 and 29) are not intrinsically hard (dense 2.99 and 3.08
against a 2.92 mean), but GPTQ collapses on both (+3.33 and +2.72 nats) and
every GSQ run repairs both by 1.9-2.6 nats. Excluding those two documents
reverses the comparison:

| Arm | all 32 docs | excluding docs 7, 29 |
| --- | --- | --- |
| W3 GPTQ | 3.787 | 3.636 |
| GSQ N=2048 | -0.035 | +0.104 |
| GSQ N=4096 (A) | +0.055 | +0.212 |
| GSQ N=4096 (B) | -0.164 | -0.020 |

So on the bulk of the corpus W3 GSQ is neutral-to-slightly-worse, and its
apparent gain is outlier repair on two documents. Zero-shot W3 confirms this:
47.67 and 47.25 for the two GSQ runs against 47.91 for GPTQ.

### Statistical power

Measured per-document delta sd against GPTQ is 0.65-0.92 nats across the three
W3 GSQ runs, so the sem at n=32 documents is 0.11-0.16 and two sem (the
smallest difference the current evaluation can resolve) is 0.23-0.33 nats.
The nominal W3 effect is 0.05 nats, well below that floor. Resolving a 0.05-nat
effect at 80% power needs roughly 1300-2700 evaluation documents; the full
128-document validation split is still short of that and only shrinks the
interval to 0.12-0.16 nats. Paired seeds are the cheaper axis: 5-19 replicates
per arm, or evaluate on the full split with `--n 128`. Either way the tables
above cannot distinguish a 0.05-nat effect from zero.

## Noise cadence and the scale-gradient reduction

`train_stage_update` draws a fresh `uniform` for every microbatch, and that is
faithful to the pinned author: `src/trainer.py` `train_step` rebuilds
`quantized_weights` *inside* the accumulation loop, and
`GumbelSoftmaxFunction.forward` in `src/quantization/gumbel_quantizer_2bit.py`
draws `u = torch.rand_like(quant_logits)` on every call. Reusing one draw across
microbatches would change the estimator, so the relaxed weights are deliberately
**not** hoisted out of the loop; `scripts/gsq_paper/bench_stage_update.py` can
measure that variant, and it is faster but it is not the paper's method.

What is free is the scale-gradient reduction. The backward pass scattered
`[out,in]` terms into `[out,groups]` with `scatter_add_`, which collides on every
element of the contiguous `arange(in)//group_size` map and accounted for 34% of
stage CUDA time. When the group map is contiguous the reduction now reshapes to
a segment sum. This is the same sum in a different reduction order (fp32
relative difference ~1e-7; the forward output is bit-identical), and it matches
what the optional fused kernel already did in `gsq_scale_reduce`. Peak VRAM is
unchanged.

## Reproducibility

The staged training has one duplicate-index `scatter_add_` in the scalar
relaxation backward pass
(`gptqmodel/quantization/gsq_training.py`, `_ScalarRelaxation.backward`).
With `torch.use_deterministic_algorithms(True)` the Q/K stage is bit-identical
across reruns at production settings (3/3 runs, no measurable cost); without
it 2 of 3 reruns diverge. It is off by default; pass `--deterministic` to
enable it. The runner records the effective value in `results.json` and the
library records it in its own run metadata, so a comparison can verify both
arms ran under the same setting.

Independent replicates require distinct `--seed`: with a fixed seed and
determinism enabled, two runs are bit-identical and carry no statistical
information.

## Deviations from the paper

* **Scale-only end-to-end stage is not implemented.** The released reference
  (`IST-DASLab/GSQ` at `03fc164`) does not contain it either. Paper Table 11
  credits it with +3.01 average, so the block-wise number here — not 68.55 —
  is the correct target for this implementation.
* **Q/K is fit on the GPTQ initialization split, not the training split**, so
  the Q/K loss is identical for different `--train-seqs`. This matches the
  reference `fit_qk_projection` call but means Q/K does not see held-out
  training text.
* **Paper results are for Llama 3.1 8B/70B.** A 1B model has less redundancy
  to recover, so absolute gaps are not comparable; only the direction and the
  relative ordering of bit widths are.

## Non-uniform bit allocation (RCO)

The paper's 2.37 and 2.62 bits-per-parameter Llama rows are non-uniform
assignments produced with RCO (arXiv 2604.18556 Sec. 4.2 and Table 1, citing
Helcig & Alistarh, arXiv 2605.00649). `gptqmodel/quantization/rco.py` is a
port of that method; its tangent projection, retraction and momentum transport
match the `IST-DASLab/RCO` reference numerically (`alpha` difference 0.0 at
tolerance 0.05 and 1e-6), and its knapsack rounder is bit-identical to the
reference solver at 16 and 112 groups.

Because a staged GSQ checkpoint stores one width per decoder block, the search
runs at block granularity. That keeps the budget it enforces equal to the
budget a checkpoint can actually express; a per-projection search would have to
be reduced to per-block widths afterwards, which moves the achieved budget.

```bash
python scripts/gsq_paper/rco_allocate.py --model meta-llama/Llama-3.2-1B-Instruct \
    --corpus /models/datasets/gsq-bench/fineweb-paper \
    --out /workspace/rco-2375 --target-bits 2.375 \
    --arms '/workspace/w{bits}gsq4096/ckpt' --layers 16 --steps 200
python scripts/gsq_paper/rco_materialize.py --model meta-llama/Llama-3.2-1B-Instruct \
    --assignment /workspace/rco-2375/assignment.json \
    --arms '/workspace/w{bits}gsq4096/ckpt' --out /workspace/rco-2375/model
```

Both budgets are hit exactly, with the constraint held to floating-point noise
across all 200 steps:

| Target | W3 blocks | W2 blocks | Achieved bpp | max drift |
| --- | --- | --- | --- | --- |
| 2.375 | 6 | 10 | 2.3750 | 9.5e-7 |
| 2.625 | 10 | 6 | 2.6250 | 9.5e-7 |

### Measured result: the allocation does not beat a uniform mix

Held-out NLL on all 128 validation sequences (524,160 tokens), scoring the
materialized checkpoints. `linear` is the NLL a uniform mix of the measured W2
and W3 endpoints would give at the same budget.

| Arm | bpp | NLL | PPL | linear | excess vs linear |
| --- | --- | --- | --- | --- | --- |
| dense | 16 | 2.9034 | 18.24 | — | — |
| W2 GSQ | 2.0 | 5.8078 | 332.9 | — | — |
| **RCO 2.375** | 2.375 | **5.1571** | 173.7 | 5.0681 | **+0.0890** |
| **RCO 2.625** | 2.625 | **4.6308** | 102.6 | 4.5749 | **+0.0559** |
| W3 GSQ | 3.0 | 3.8352 | 46.3 | — | — |

RCO lands between the two uniform arms, which shows the search is choosing
widths that work, but it is **worse than the interpolation it should beat** at
both budgets, and it is nowhere near the paper's claim that a 2.37 bpp
assignment scores close to W3 (69.44 vs 72.32 average).

The paper's gain comes from a model with real per-layer sensitivity structure
to exploit: Llama 3.1 8B/70B, where attention and MLP layers differ markedly in
how much they degrade. Llama 3.2 1B does not offer that, and its 16 blocks are
near-uniformly sensitive, so any 6-of-16 assignment behaves like a uniform mix
of the endpoints and the optimizer has almost nothing to find. This is the
same limitation already recorded above: a 1B model has less redundancy to
recover, so only the direction and the ordering of bit widths transfer, not the
magnitude.

Two things remain untested and would be the next steps: a wider option set than
{W2, W3} (the paper searches more widths, which gives the optimizer more room
to differentiate blocks), and a larger model where per-layer sensitivity is
actually heterogeneous.
