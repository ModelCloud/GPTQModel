# GSQ: configuration for positive QVQ recovery

Status: experimental. GSQ remains opt-in for QVQ. The staged W3/P32 path has
produced a positive result on Llama 3.2 1B, but the completed run used a small
training subset and predates the deterministic-training correction. It is not
yet a paper-scale or publication-grade result.

This guide records the configuration and acceptance protocol that made GSQ
useful after QVQ + YAQA. It also records the configurations that looked
reasonable but could not improve the deployed model.

References: [GSQ paper](https://arxiv.org/abs/2604.18556),
[pinned author implementation](https://github.com/IST-DASLab/GSQ/tree/03fc16484c369e3127225615d5e03e8d3a6043e3),
[QVQ/GSQ design](qvq_gsq_design.md), and the
[paper-alignment audit](experiments/gsq-qvq-paper-alignment.md).

## What makes GSQ work

GSQ is not simply a longer Fisher-coordinate search. The successful QVQ
adaptation preserves these parts of the paper's optimization:

| Requirement | Working QVQ/P32 configuration | Why it matters |
| --- | --- | --- |
| Initialization | YAQA W3/P32 hard payload is choice zero; soft logits use the paper's shift-centred Gaussian prior | The optimizer starts from an exactly deployable baseline while retaining gradients toward nearby legal choices |
| Optimizer | Lion, assignment LR `1e-4`, scale LR `5e-5`, weight decay `1.0`, betas `(0.9, 0.95)`, cosine decay | This matches the dynamics used by the staged implementation |
| Annealing | temperature `2.0 -> 0.05`, multiplier/kappa `100 -> 500` | Early exploration becomes a hard categorical decision |
| Trainable state | legal P32 categorical edits and the serialized output scale vector `SV` | Fixed scales removed an important degree of freedom from the QVQ adaptation |
| Objective order | Q/K quadratic fit, joint V/O attention reconstruction, joint gate/up/down full-block reconstruction | Attention and MLP coupling is absent from an aggregate per-projection Fisher loss |
| Data | packed FineWeb-Edu, 4,096 rows x 4,096 tokens for reconstruction | Tiny calibration slices can validate mechanics but do not supply the paper's training signal |
| Schedule | 20 Llama block epochs at logical batch 64; 2,000 dedicated Q/K updates | Updates are not epochs: `4096 / 64 * 20 = 1,280` block updates |
| Propagation | replay every preceding layer with its deployable QVQ hard weight | Later layers must learn against the errors the final checkpoint actually contains |
| Selection | hard held-out stage selection, local layer rollback, then fresh-reload global rollback | Soft loss and local MSE do not reliably predict final-model quality |
| Reproducibility | deterministic PyTorch algorithms, seeded CPU/CUDA RNGs, cuBLAS workspace `:4096:8` | Same-seed nondeterministic runs selected different tiles and scales |

The W3/P32 categorical prior is

```text
logits = 0.01 * (Normal(0, 1) + 6 * centered(-shift^2 / 2))
temperature: 2.0 -> 0.05
kappa:       100 -> 500
```

The soft initialization follows the paper, but the first hard state is forced
to candidate zero. Otherwise the first validation comparison is against a
random hard edit rather than the source QVQ payload.

## The schedule and data contract

For dense Llama, the paper-aligned reconstruction budget is:

```text
4,096 packed samples * 4,096 tokens = 16,777,216 reconstruction tokens
64 logical batches per epoch * 20 epochs = 1,280 block optimizer updates
2,000 dedicated Q/K optimizer updates
```

The local cache builder reserves four contiguous, pairwise-disjoint ranges:

| Purpose | Start row | Rows | Tokens |
| --- | ---: | ---: | ---: |
| reconstruction training | 0 | 4,096 | 16,777,216 |
| hard selection validation | 4,096 | 128 | 524,288 |
| Q/K metric construction | 4,224 | 512 | 2,097,152 |
| final report only | 4,736 | 128 | 524,288 |

Do not use the validation or report rows for optimizer updates. Do not use the
report-only rows for layer acceptance. GSM8K-Platinum, or another downstream
task, must also be disjoint from all quantization and model-selection data.

Build the cache with the pinned FineWeb-Edu revision and official packing
behavior:

```bash
python scripts/cache_gsq_fineweb.py \
  --model /path/to/Llama-3.2-1B-Instruct \
  --output /path/to/fineweb-edu-4864x4096.safetensors \
  --manifest /path/to/fineweb-edu-4864x4096.json
```

The generated manifest records the dataset revision, tokenizer hash, offsets,
row hashes, and cache hash. Treat it as part of the model artifact.

### Updates are not epochs

`GSQConfig.for_qvq_paper_schedule()` prevents the older aggregate QVQ fitter
from silently treating ten updates as ten epochs:

```python
from gptqmodel.quantization.config import GSQConfig

gsq = GSQConfig.for_qvq_paper_schedule(
    num_samples=4096,
    batch_size=64,
    epochs=20,
)
assert gsq.steps == 1280
```

This helper configures the format-aware Fisher adapter in
`qvq_gsq.py`. It does **not** turn that adapter into the staged reconstruction
pipeline and is not, by itself, the positive-return recipe. Use the staged
driver below until that lifecycle is integrated into the main quantizer.

## Staged W3/P32 run

`scripts/validate_qvq_gsq_staged_layer.py` implements the current experimental
path. A paper-budget layer run has this shape:

```bash
PYTHON_GIL=0 python scripts/validate_qvq_gsq_staged_layer.py \
  --dense-model /path/to/Llama-3.2-1B-Instruct \
  --qvq-model /path/to/source-w3-p32-qvq \
  --token-cache /path/to/fineweb-edu-4864x4096.safetensors \
  --layer 0 \
  --sequence-length 4096 \
  --train-offset 0 --train-samples 4096 \
  --validation-offset 4096 --validation-samples 128 \
  --qk-offset 4224 --qk-samples 512 \
  --epochs 20 --qk-steps 2000 --qk-hard-eval-interval 100 \
  --batch-size 64 --microbatch-size 1 \
  --candidates 33 --rounds 1 \
  --offload-capture --capture-directory /path/to/capture/layer-0 \
  --state-output /path/to/states/layer-0.safetensors \
  --output /path/to/logs/layer-0.json
```

`PYTHON_GIL=0` remains recommended on a free-threaded Python build.  It is
performance-critical for `--no-fused-qk-fisher`, where the driver fits
independent Q and K projections on separate CUDA streams while preserving each
projection's exact RNG and update order. GIL-enabled execution remains correct
but serializes enough host dispatch to lose roughly 8% on that oracle path.

Independent P32 candidate banks are also built concurrently by default. Each
projection keeps its own seeded generator and CUDA stream, so candidate words,
the selected model state, and held-out loss remain unchanged. Three-pair H100
serial/parallel comparisons produced these medians:

| Sequence geometry | Held-out gain | Candidate build | Candidate + fit |
| --- | ---: | ---: | ---: |
| 64/16/64 x 256 tokens | 29.2717% | 0.5077s -> 0.4048s (1.254x) | 2.6221s -> 2.5384s (1.033x) |
| 32/8/32 x 512 tokens | 15.0000% | 0.5134s -> 0.4058s (1.265x) | 2.1142s -> 2.0103s (1.052x) |

The rows have equal train, held-out, and Q/K token totals, but different
sequence boundaries and token hashes; their held-out percentages must not be
compared as an accuracy change. Within each geometry, all three parallel runs
were bit-for-bit equal to their matched serial model states. Result JSON and
state metadata record sequence length and split sample counts. Use
`--no-parallel-candidate-build` only for serial diagnostics.

First-round P32 candidate screening also specializes its known identity Fisher
metrics. It skips dense identity tensors, identity GEMMs, and block gathers,
while retaining the original FP32 quadratic multiply-and-reduce order. Across
three paired H100 runs per geometry, candidate words, sparse metadata, all 14
state tensors, guard diagnostics, and held-out loss were bit-for-bit equal. The
median candidate wall time improved from 0.4037s to 0.3716s (1.087x) at
32/8/32 x 512 tokens and from 0.4046s to 0.3777s (1.071x) at 64/16/64 x 256
tokens. Held-out gains remained 15.0000% and 29.2717%, respectively. Use
`--no-fast-identity-candidate-metric` only to compare against the dense
identity-matrix path.

The two whole-block Q/K pair guards also replay alternatives directly on the
teacher layer by default. The driver caches dense held-out outputs first,
temporarily installs each BF16 Q/K pair plus the stage's fixed downstream
weights, and restores every dense projection afterward. This removes repeated
stateless functional-call setup without changing the operation being scored.
Across three paired H100 runs, all guard losses, selected alternatives, and
14-tensor states were bit-for-bit equal. Median guard time improved from
0.2618s to 0.2570s (1.019x) at 32/8/32 x 512 tokens and from 0.2611s to
0.2590s (1.008x) at 64/16/64 x 256 tokens. Use
`--no-direct-qk-pair-guard` for the prior functional-call diagnostic path.

Accepted P32 states are materialized by directly decoding their legal window
words and applying the exact RHT reconstruction. Constructing a new two-choice
training module for this read-only operation repeated Fisher candidate
screening even though the accepted words and scales were already fixed. Nsight
showed these throwaway screens in both the post-MLP Q/K guard and final export.
Direct decode is now the default; `--no-direct-state-materialization` retains
the prior diagnostic path. Across three paired runs per geometry, every guard
measurement and all 14 state tensors were bit-for-bit equal. At 32/8/32 x 512
tokens, guard materialization improved 2.56x, final materialization improved
3.96x, and measured candidate-build-plus-fit-plus-final time improved 1.167x.
At 64/16/64 x 256 tokens the corresponding gains were 2.46x, 4.03x, and
1.130x. Held-out gains remained 15.0000% and 29.2717%, respectively.

The default single-round Q/K path now transforms the same quadratic objective
into P32 inner coordinates and runs the fused sparse Fisher optimizer.  It then
solves the independent output scales in closed form and uses only the disjoint
Q/K validation split to select among the original, scale-only, and edited P32
states. After V/O and MLP fitting, it replays the unique Q/K alternatives with
those downstream states fixed and installs only the pair with the lowest
whole-block held-out loss. Dense teacher outputs are cached once for this
guard. This preserves all requested GSQ updates and the final block-level
held-out guard while avoiding dense RHT reconstruction and autograd on every
Q/K update. Pass `--no-fused-qk-fisher` for the previous differentiable Q/K
oracle. Multi-round P32 composition currently requires that oracle path.

For the 2,000-update staged Q/K schedule, checkpoint the structured hard
oracle every 100 updates. This still evaluates 20 points along the relaxation
trajectory plus the exact dense top-k verification, while avoiding redundant
hard decodes. On the controlled H100 layer run, 100 updates was faster than
40, 80, or 200; it retained the exact final model state and held-out loss.
The 640-update paper helper remains epoch-aligned at 64 updates and is not
changed by this staged-driver default.

Choose `--microbatch-size` from available memory; it is an accumulation/memory
control and does not change the logical batch size. Determinism is enabled by
default. `--allow-nondeterministic` is for explicitly labelled exploratory
timing only, never a publishable model-quality run.

For layer `n > 0`, pass the cumulative state accepted through layer `n - 1`
with `--prefix-state`. A globally rejected candidate must not become the next
prefix. The script reconstructs missing preceding layers from the source QVQ
checkpoint, not from dense weights.

P32's 33-choice bank contains candidate zero plus bounded legal trellis-edge
edits. One categorical decision can select only one edit per tile in one
round. Additional `--rounds` rebuild the legal bank from the accepted hard
payload and can compose edits, but each round must independently pass held-out
selection. More rounds are extra search capacity, not an automatic quality
gain.

## Acceptance protocol

Use three gates in this order:

1. Each stage restores the best hard state measured on the held-out selection
   rows. A lower soft training objective is insufficient.
2. The complete layer is rolled back if held-out full-block reconstruction MSE
   regresses.
3. Install the cumulative state into a copy-on-write checkpoint, freshly load
   it through `BACKEND.QVQ`, and compare it with the currently accepted prefix
   on the same selection rows. Promote only a candidate that satisfies the
   predeclared final-logit policy.

The strict validation used for the first all-layer sweep required no regression
in forward KL, logit MSE, cross-entropy, perplexity, or dense-teacher top-1
agreement. A larger experiment may use an uncertainty-aware promotion policy,
but that policy must be declared before looking at report-only or task results.

Run the fresh-checkpoint gate with
`scripts/validate_qvq_gsq_checkpoint.py`. Use `--role validation`, offset
`4096`, and the incumbent validation JSON while selecting layers. Only after
the final state is frozen, use `--role report_only`, offset `4736`, and all 128
report rows. The driver checks row hashes before creating the candidate
checkpoint.

Finally, evaluate the reloaded checkpoint with the target inference runtime and
task harness. For the Llama 3.2 1B experiment this means ZML's SM90 P32 path
with FA2 paged attention and Evalution's complete 1,209-row
GSM8K-Platinum task. See the
[reproduction protocol](qvq_gsq_gsm8k_reproduction.md) for task settings.

## Configurations that did not work

| Configuration | Observed failure |
| --- | --- |
| 100 requested per-projection steps with early stop after 10 | Ten updates were reported as epochs and were far below the annealing/data budget |
| batch-1 Fisher capture followed by one aggregate objective | No shuffled data-dependent minibatches, attention coupling, MLP coupling, or prefix-error signal |
| deterministic coordinate sweep before GSQ | The comparator consumed the same frozen candidate pool and objective, leaving no attributable GSQ gain |
| zero logits plus a small baseline-winner margin | The hard choice stayed at zero; the paper prior was required to move legal tiles |
| fixed `SV` | Assignment-only optimization omitted the deployable scale variable needed by P32 |
| one frozen single-edge candidate bank | One round can change at most one edge per tile and cannot approach scalar GSQ's independent-coordinate movement density |
| local Fisher or full-block MSE as the only gate | Layers with strong local improvements still regressed end-to-end logits |
| dense weights for rejected or unprocessed prefix layers | Later stages optimized against a model that could not be deployed |
| shared calibration, selection, and reporting rows | Apparent improvement was model selection on the evaluation set |
| nondeterministic final training | Same seed produced different payloads, scales, and metrics |

These are algorithmic failures, not reasons to remove hard no-regression guards
or to accept more changed tiles. Tile count is a diagnostic; held-out deployed
quality is the target.

## What has been demonstrated

The guarded all-layer diagnostic used the correct split ranges and schedule
shape, but only 32 training rows truncated to 512 tokens: 16,384 reconstruction
tokens, or 0.0977% of the paper's 16.78M-token budget. It accepted layers 0, 2,
4, and 5. On all 128 untouched report rows, the freshly reloaded checkpoint
improved forward KL by 3.2908%, logit MSE by 2.1330%, cross-entropy by 0.0680%,
perplexity by 0.2000%, and top-1 agreement by 0.1101 percentage points.

On GSM8K-Platinum, the source QVQ checkpoint scored 537/1,209 (44.4169%) and
the guarded GSQ checkpoint scored 540/1,209 (44.6650%): +3 answers or +0.2481
percentage points. There were 39 incorrect-to-correct and 36
correct-to-incorrect flips. This is a positive direction, not a statistically
robust gain.

That checkpoint predates deterministic training. Its measurement is retained
as experimental evidence, but the next claim must rerun the complete pipeline
in deterministic mode with the full 4,096 x 4,096 reconstruction set, preserve
all logs and manifests, and repeat the disjoint final and task evaluations.

## Promotion checklist

A QVQ + GSQ result is eligible for promotion only when all of these are true:

- the source is a valid QVQ + YAQA checkpoint and candidate zero reproduces it;
- the data manifest proves train, validation, Q/K, report, and task disjointness;
- the actual samples, tokens, logical batches, epochs, and optimizer updates are
  reported separately;
- stage order is Q/K, joint V/O, then joint gate/up/down;
- assignments and native `SV` are optimized and materialized as a legal P32
  payload;
- every prefix layer uses its deployable hard QVQ state;
- local and global hard-state guards use held-out selection data;
- report-only metrics are computed once after model selection on a fresh native
  runtime reload;
- deterministic replay reproduces the hard state and metrics;
- downstream evaluation reports paired sample flips, not only aggregate score.
