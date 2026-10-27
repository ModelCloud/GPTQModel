# GSQ paper reproduction: Llama 3.2 1B and distributed Llama 3.1 8B

Scripts in [`scripts/gsq_paper/`](../../scripts/gsq_paper) reproduce the
staged-GSQ comparison from [GSQ: Accurate-Instant Quantization for LLMs](https://arxiv.org/abs/2604.18556)
(Dadgarnia et al., App. G.1) on a single Llama 3.2 1B Instruct checkpoint:
a GPTQ-only arm versus a GPTQ-initialized arm that also runs the staged
reconstruction training.

The completed quality tables below concern Llama 3.2 1B. The
[distributed Llama 3.1 8B deployment](#distributed-llama-31-8b) has separate
correctness and MLP timing evidence; full GSQ 8B quality results are still pending.

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

## Measured quality results: Llama 3.2 1B

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
**not** hoisted out of the loop.
[`bench_stage_update.py`](../../scripts/gsq_paper/bench_stage_update.py) compares
scatter and segment reductions while retaining that noise cadence. Its default
16-token objective is not a sequence-4096 stage benchmark; use the distributed
MLP measurements below for that geometry.

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

## Distributed Llama 3.1 8B

Status at the 2026-10-27 integration handoff: two four-rank GSQ arms are live,
W2 on physical GPUs 0–3 and W3 on 4–7. Each imported three completed blocks
and began training block 4 (layer index 3). The focused integration suite was
reported as **98 passed in 83.64 seconds**. These checks and the stage benchmark
do not provide completed GSQ 8B NLL or zero-shot quality results.

The first newly trained production block (layer index 3) completed with all
2000 Q/K and 1280 attention/MLP updates, and both arms advanced to block 5/32:

| Arm | First new block time |
| --- | ---: |
| W2, four ranks | 357.9 seconds |
| W3, four ranks | 352.2 seconds |

These model-driver timings include the first active capture, initialization,
training, validation, and packing. They end before checkpoint publication and
capture advancement to the next block. They are initial measurements, not a
completed-model runtime or quality result.

### Flags, sharding, and the paper budget

[`run_arm.py`](../../scripts/gsq_paper/run_arm.py) exposes the following controls.
`--distributed` requires `torchrun` with at least two ranks and `--variant gsq`;
it selects the device from `LOCAL_RANK`. On ROCm, the PyTorch backend name
`nccl` uses RCCL.

| Flag | Meaning |
| --- | --- |
| `--distributed` | Replicate quantizers and optimizer state across ranks; shard each global batch's training and validation documents into contiguous rank chunks. |
| `--batch-size 64` | Global optimizer batch, shared by the ranks. It must be divisible by world size. |
| `--microbatch 16` | Documents per forward/backward microbatch on each rank; must fit inside the local batch. |
| `--capture cpu\|gpu\|disk` | Storage for captured block inputs. The paired production driver uses `gpu`. |
| `--target-cache none\|cpu\|device` | Explicit fixed-target cache policy; default `none`. Memory target caching requires SDPA, which the runner selects. |
| `--activation-checkpointing` | Recompute student activations during backward to reduce activation storage. |
| `--reuse-capture` | Advance captures through each newly packed block for the next consecutive block. Requires CPU or GPU capture; incompatible with `--capture disk`. |
| `--resume` | Read/write completed packed blocks under `<out>/ckpt`. |
| `--resume-prefix-from PATH` | Explicitly import compatible completed blocks from another checkpoint directory; requires `--resume`. |
| `--layers 32` | Process all Llama 3.1 8B decoder blocks. The runner's default is 16, so this must be explicit. |

The production budget is **4096 training / 512 GPTQ calibration / 128 validation
documents, each 4096 tokens**, with group size 128, 20 epochs, and 2000 Q/K
updates. Global batch 64 gives 64 optimizer updates per epoch and **1280 updates
per reconstruction stage** at every supported world size. Learning-rate,
temperature, and multiplier schedules retain those 1280 update positions; the
number of updates is not divided by the number of ranks. GPTQ initialization
and Q/K fitting use the complete 512-document calibration split on the primary
rank, with results distributed to the other ranks.

| Ranks per arm | Local documents per optimizer update | Microbatch size | Local microbatches per update | Global microbatches per update |
| --- | --- | --- | --- | --- |
| 1 | 64 | 16 | 4 | 4 |
| 2 | 32 | 16 | 2 | 4 |
| 4 | 16 | 16 | 1 | 4 |

Every microbatch draws fresh noise for each quantizer. Four ranks therefore
preserve the original microbatch-16 boundaries at global batch 64. Eight ranks
have only eight documents per rank: microbatch 16 is rejected. Changing it to
8 or 2 preserves the global sample count but changes the grouping and number
of draws, so it is a different stochastic trajectory. Native rank RNG streams
are distinct; preserving the budget and boundaries does not imply bitwise
agreement across topologies.

Quantizers synchronize before the first fit evaluation. Accumulated gradients
are weighted by reconstructed output-element counts and summed globally before
each Lion step. Validation uses the same global weighting, so all ranks select
the same best checkpoint. The model driver rejects training/validation lengths
that do not divide into complete global batches instead of dropping documents.
The pinned reference at `03fc16484c369e3127225615d5e03e8d3a6043e3`
shards its corpus contiguously and redraws inside the microbatch loop; its local
10-epoch default gives 640 updates for 4096/64. The deployment here explicitly
uses the 20-epoch, 1280-update recipe.

### Target caching, capture reuse, and checkpoint import

`--capture` controls block-input storage; `--target-cache` controls teacher
outputs and fixed stage intermediates. With CPU capture and no explicit target
cache, the objective can rerun teacher computation and fixed attention during
MLP training. `--target-cache device` retains exact targets and the fixed
post-attention residual on the GPU, leaving the changing MLP, normalization,
residual, reconstruction loss, and backward pass in each MLP update.
`--target-cache cpu` stores the same targets in host memory and transfers them
when consumed. The existing SDPA disk-capture path can automatically create
temporary target caches even when the explicit policy is `none`.

Activation checkpointing recomputes the student forward from the already
sampled weights. It does not draw a replacement noise sample during backward
or share a sample between microbatches. `--reuse-capture` advances train,
calibration, and validation captures through the installed packed block,
including its stored-scale rounding, instead of replaying the entire model
prefix for each next block. After a resumed prefix, the first unfinished block
is captured from that loaded prefix before incremental advancement continues.

Ordinary `--resume` validates the destination checkpoint signature. Explicit
prefix import permits differences in `device`, `distributed_world_size`,
`target_cache`, and `activation_checkpointing`; it still checks the model and
source-prefix identity, all document digests, bit widths, layer selection,
determinism setting, and GSQ recipe, including microbatch size and seed.
Completed destination blocks take precedence over the import source. Imported
blocks are written atomically under the destination signature and record
`imported_from`; source files are preserved. This resumes completed blocks,
not an interrupted block's optimizer state.

New checkpoints also record the hash of their preceding packed blocks. A
changed predecessor is rejected, and checkpoint gaps are rejected before any
training starts. Legacy checkpoints have no historical predecessor hash: their
source directory must be intact, and its complete earlier packed prefix must
match the prefix installed by this run. Rank-local capture and incremental
replay failures propagate to every rank; completed captures are registered for
cleanup before the next capture begins.

The current source trees are
`/models/datasets/gsq-bench8b-original/w2gsq/ckpt` and
`/models/datasets/gsq-bench8b-original/w3gsq/ckpt`, each with layers 00–02.
Four-rank microbatch-16 execution allows their recipe to remain compatible.

### Measured MLP-only update benchmark

The `GSQ_BENCHMARK_JSON` records in
[rank 1](/models/datasets/gsq-ddp-bench-rank1.log),
[rank 2](/models/datasets/gsq-ddp-bench-rank2.log), and
[rank 4](/models/datasets/gsq-ddp-bench-rank4.log) come from
[`bench_distributed_stage.py`](../../scripts/gsq_paper/bench_distributed_stage.py).
They report the import path
`/monster/home/gptqmodel-ddp/gptqmodel/quantization/gsq_training.py`, AMD Instinct
MI350X VF, Torch `2.16.0.dev20261025+rocm10.1`, and HIP `7.16.26385`.

These runs load actual weights from
`/models/datasets/models/Llama-3.1-8B-Instruct`. Gate and up weights each have
shape **[14336, 4096]**; down weights have shape **[4096, 14336]**. Activations
are **synthetic normal inputs**, and the quantized initialization is
**groupwise symmetric RTN**, rather than the production run's GPTQ
initialization from text. Weights, activations, and logits are BF16. Global
batch **64 × 4096** is 262,144 tokens per update; each microbatch is
**[16, 4096, 4096]**, or 65,536 tokens, with MLP intermediate width 14336.
All runs use device-cached targets and activation checkpointing.

Timing surrounds complete `train_stage_update` calls: fresh noise, relaxed
weights, MLP objective, backward, global reduction where applicable, and Lion.
It excludes model loading, input capture, GPTQ calibration/initialization,
Q/K fitting, target preparation, attention-stage training, validation,
packing/export, and quality evaluation. The reported milliseconds/update are
the maximum rank's mean synchronized elapsed time over the measured updates.
These are not end-to-end block or model timings, and profiler invocation counts
must not be treated as optimizer-update counts.

| Bits | Ranks | ms/update | Peak allocated GiB/rank | Peak reserved GiB/rank | First → last measured reconstruction MSE |
| --- | --- | --- | --- | --- | --- |
| W2 | 1 | 394.16 | 23.19 | 26.70 | 1.80244e-4 → 1.67847e-4 |
| W3 | 1 | 345.23 | 29.04 | 30.83 | 2.96831e-5 → 2.74181e-5 |
| W2 | 2 | 259.06 | 21.19 | 23.20 | 1.80244e-4 → 1.67847e-4 |
| W3 | 2 | 248.89 | 27.04 | 29.08 | 2.96831e-5 → 2.74181e-5 |
| W2 | 4 | 139.47 | 18.87 | 19.70 | 1.74522e-4 → 1.53542e-4 |
| W3 | 4 | 137.21 | 24.39 | 25.58 | 2.90871e-5 → 2.51532e-5 |

One- and two-rank runs use two warmup plus five measured updates (seven total).
Four-rank runs use three warmup plus ten measured updates (thirteen total).
`budget_updates=1280` sets the schedule horizon; these timing runs did not
execute a complete 20-epoch stage. Loss windows therefore differ. All ranks
within each distributed run report the same globally reduced losses.
`draws=native` and `matched_across_topologies=false` mean these timing results
are not matched-draw numerical comparisons. Peak figures are the maximum
per-rank PyTorch allocator values, excluding external RCCL allocations;
they are not total device-memory measurements or full-model memory budgets.

To reproduce the four-rank timing configuration on an available four-GPU group:

```bash
PYTHONPATH=/monster/home/gptqmodel-ddp ROCR_VISIBLE_DEVICES=0,1,2,3 \
  /monster/venvs/venv-py3.14t-gil0/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 scripts/gsq_paper/bench_distributed_stage.py \
  --model /models/datasets/models/Llama-3.1-8B-Instruct --bits 2 3 \
  --hidden 4096 --intermediate 14336 --group-size 128 --sequence-length 4096 \
  --global-batch 64 --microbatch-size 16 --dtype bfloat16 --logits-dtype bfloat16 \
  --cached-targets device --activation-checkpointing --draws native \
  --train-documents 4096 --epochs 20 --warmup 3 --repeats 10
```

### Eight-rank numerical evidence

The `GSQ_CORRECTNESS_JSON` in
[/models/datasets/gsq-ddp-oracle-rank8.log](/models/datasets/gsq-ddp-oracle-rank8.log)
records the real GSQ update compared against independent float64 Torch
softmax/gradient/Lion arithmetic and an actual single-device run over all ranks'
microbatches. Each bit width ran four updates with identical explicit draws
per microbatch. This is a tiny numerical fixture with global batch 64 and
microbatch 2, not the 8B microbatch-16 workload above.

| Bits | Maximum absolute error across checked loss, gradients, parameters, and Lion state |
| --- | --- |
| W2 | 4.440892098500626e-16 |
| W3 | 2.6645352591003757e-15 |

The 564 reconstructed elements are deliberately unequal across ranks:
`[60, 72, 72, 72, 72, 72, 72, 72]`. The validation fixture selected epoch 2
(zero-based), with global loss 0.1875; initial synchronization and distinct
rank RNG streams passed. A separate tiny fit executed all **1280** updates
and checked learning-rate and sampling schedules. This establishes the tested
distributed arithmetic and budget behavior; eight-rank full-geometry timing
and full GSQ model quality are not established by this oracle log.

### Detached production pair, mirroring, and evaluation

[`run_distributed_pair.sh`](../../scripts/gsq_paper/run_distributed_pair.sh)
owns the two independent four-rank launches, completed-prefix import, periodic
mirroring, and subsequent RCO/NLL jobs. Its arguments are
`MODEL CORPUS OUTPUT MIRROR ORIGINAL_ARMS`. The deployment uses the complete
Llama-3.1-tokenized corpus at
`/models/datasets/gsq-bench/fineweb-paper-llama31`, all 32 layers, and the full
4096/512/128, 20-epoch, 2000-Q/K-step budget. It passes `--capture gpu`,
`--target-cache device`, `--activation-checkpointing`, `--reuse-capture`,
`--restore-best auto`, `--resume`, and the original arm's
`--resume-prefix-from` directory. `--skip-eval` defers the runner's zero-shot
suite; it does not disable stage validation or best-checkpoint selection.

The output root is `/models/datasets/gsq-bench8b-ddp`, mirrored to
`/monster/models/datasets/gsq-bench8b-ddp`. The detached launch/restart recipe is:

```bash
cd /monster/home/gptqmodel-ddp
mkdir -p /models/datasets/gsq-bench8b-ddp
GSQ_PYTHON=/monster/venvs/venv-py3.14t-gil0/bin/python \
  setsid nohup bash scripts/gsq_paper/run_distributed_pair.sh \
  /models/datasets/models/Llama-3.1-8B-Instruct \
  /models/datasets/gsq-bench/fineweb-paper-llama31 \
  /models/datasets/gsq-bench8b-ddp \
  /monster/models/datasets/gsq-bench8b-ddp \
  /models/datasets/gsq-bench8b-original \
  > /models/datasets/gsq-bench8b-ddp/driver.log 2>&1 < /dev/null &
```

The driver sets this checkout's `PYTHONPATH` and uses `ROCR_VISIBLE_DEVICES`
0–3 for W2 and 4–7 for W3. A lock prevents two drivers from owning the same
output root. `status.txt`, `driver.pid`, `arm-supervisors.pid`, and each arm's
`run.log`/`exit-code.txt` expose progress; restarts archive earlier arm logs and
exit codes with a timestamp. During training, checkpoint/log
mirroring runs every 30 seconds and excludes temporary files, capture caches,
the lock, and incomplete model exports. Completed models are mirrored after
both training arms succeed; final RCO/NLL outputs are mirrored on completion.

After the arms finish, the driver searches block allocations at 2.375 and
2.625 bpp using W2/W3 options, 64 search documents, and 200 steps, then
materializes both models. It runs NLL on 128 validation documents for those
models, uniform W2/W3 GSQ, the original W2/W3 GPTQ baselines, and dense weights.
Outputs include `rco/nll.json` and `rco/nll-per-document.json`.
**RCO also uses the validation split for its search, so this NLL is a validation
diagnostic, not a held-out test of RCO.** The paper's zero-shot evaluation
remains a separate quality evaluation. No completed full-GSQ 8B quality claim
is made while these production runs and evaluations are pending.

## Deviations from the paper

* **Scale-only end-to-end stage is not implemented.** The released reference
  (`IST-DASLab/GSQ` at `03fc164`) does not contain it either. Paper Table 11
  credits it with +3.01 average, so the block-wise number here — not 68.55 —
  is the correct target for this implementation.
* **Q/K is fit on the GPTQ initialization split, not the training split**, so
  the Q/K loss is identical for different `--train-seqs`. This matches the
  reference `fit_qk_projection` call but means Q/K does not see held-out
  training text.
* **Paper results are for Llama 3.1 8B/70B.** The completed quality tables above
  use Llama 3.2 1B; the new 8B deployment is awaiting quality evaluation.
  A 1B model has less redundancy
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
