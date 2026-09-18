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

The fused Q/K path also reuses the exact FP32 baseline and sparse candidate
values produced during P32 screening. Hard checkpoints scatter those values
into the baseline instead of decoding a dense 33-choice candidate bank. This
does not approximate the structured or dense hard oracle and removes the two
largest remaining candidate-bank decodes. Across three paired H100 runs per
geometry, all 14 state tensors, hard-oracle diagnostics, Q/K guards, and
held-out losses were bit-for-bit equal. Median Q/K fit time improved from
0.5956s to 0.5575s (1.068x) at 32/8/32 x 512 tokens and from 0.6013s to
0.5538s (1.086x) at 64/16/64 x 256 tokens. Candidate-build-plus-fit-plus-final
time improved 1.025x and 1.020x, respectively. Use
`--no-sparse-qk-candidate-bank` only for comparison with the dense decode path.

P32 overlap maps are now built directly from the three consecutive edited
trellis states instead of sorting all six scalar entries from every candidate.
The direct kernel preserves the generic map's candidate accumulation order and
uses a fixed three-entry overlap per scalar. Across three alternating paired
H100 runs per geometry, all Q/K guards, losses, changed-tile counts, and 14
state tensors were bit-for-bit equal. Median candidate wall time improved from
0.3760s to 0.2849s (1.320x) at 32/8/32 x 512 tokens and from 0.3782s to
0.2797s (1.352x) at 64/16/64 x 256 tokens. Candidate-build-plus-fit-plus-final
time improved 1.058x and 1.030x, respectively. Held-out gains remained exactly
15.0000% and 29.2717%. Use `--no-fast-p32-position-map` only for comparison
with the generic radix-sort path.

Candidate screening now returns the exact six changed P32 values directly to
the staged training module instead of materializing a dense 33-choice decoded
bank and gathering those values afterward. For a Llama 3.2 1B layer this
removes about 8 GiB of transient FP32 candidate data across the seven
concurrently built projections. Across three alternating paired H100 runs per
geometry, all guards, losses, changed-tile counts, and 14 state tensors were
bit-for-bit equal. Median candidate wall time improved from 0.2833s to 0.2498s
(1.134x) at 32/8/32 x 512 tokens and from 0.2854s to 0.2486s (1.148x) at
64/16/64 x 256 tokens. Candidate-build-plus-fit-plus-final time improved
1.021x and 1.011x, respectively. Held-out gains remained exactly 15.0000% and
29.2717%. Use `--no-compact-sparse-candidates` only to compare against the
dense decoded-bank path.

The fused sparse mixture materializer is also used for the 2,048-wide V/O
attention projections. This avoids deterministic `scatter_add` sorting of
3.15 million sparse entries on every update. The direct P32 map kernel emits
the transposed orientation consumed by the fused Hadamard path, so training
does not build and permute an unused normal-orientation map. Across three
warm-cache paired H100 runs per geometry, all guards, losses, changed-tile
counts, and 14 state tensors were bit-for-bit equal. Median attention fitting
improved from 0.3443s to 0.3321s (1.037x) at 32/8/32 x 512 tokens and from
0.6048s to 0.5737s (1.054x) at 64/16/64 x 256 tokens. Total fitting improved
1.011x and 1.025x; candidate-build-plus-fit-plus-final improved 1.005x and
1.017x. The new 2,048-square Triton specialization has a one-time compilation
cost and breaks even after roughly eight fitted layers. Held-out gains remained
exactly 15.0000% and 29.2717%. Use `--no-compact-attention-forward` only for
comparison with deterministic scatter materialization.

The fused materializer now keeps only a 128-byte P32 start-choice lookup per
tile and derives each scalar's three legal overlaps from starts `s`, `s-1`, and
`s-2` inside the kernel. It applies the same candidate-ID sort and FP32
multiply/add order as the dense overlap map, so this is an exact metadata
compression rather than an arithmetic shortcut. A 65,536-tile MLP projection
now uses an 8 MiB lookup instead of a 256 MiB dense map. Across a Llama 3.2 1B
layer this removes 899 MiB of map allocation volume, including 744 MiB from
the three-module MLP stage. SM90 launches use two warps for attention and four
for the larger MLP projections. On the final warm H100 checks, MLP fitting
improved 1.038x at 32/8/32 x 512 tokens and 1.054x at 64/16/64 x 256 tokens;
total fitting improved 1.011x and 1.009x. All matched state tensors, Q/K guards,
changed-tile counts, and held-out losses remained bit-for-bit equal, strict
split disjointness passed, and gains remained exactly 15.0000% and 29.2717%.
Use `--no-inline-p32-overlap` to compare against the dense overlap-map path.

W3/P32 identity-Fisher candidate screening now fuses four legal shift decodes,
the six affected PGC16 scalar decodes, the exact FP32 local quadratic, and the
stable shift selection into one Triton program per tile and candidate position.
The fallback remains available with `--no-fused-p32-identity-screen`. On an
H100, the complete 8,192 x 2,048 candidate constructor improved from 20.59ms
to 8.69ms after warmup (2.37x), including legal word and sparse-value output.
With seven projection banks built concurrently, median candidate wall time at
32/8/32 x 512 tokens improved from 0.2656s to 0.2345s (1.13x). A controlled
64/16/64 x 256-token pair improved from 0.2702s to 0.2172s (1.24x). All three
512-token pairs and the 256-token pair produced bit-for-bit equal 14-tensor
states, identical changed-tile counts and guards, and exact held-out gains of
15.0000% and 29.2717%, respectively. Strict train/validation/QK split
disjointness passed. Regression coverage compares the fused and eager oracles
for packed and unpacked selectors with every alternate PGC16 bank.

The same identity-screen kernel now emits each selected legal 24-word P32
payload directly. This removes the eager int64 clone, gather, mask, and scatter
chain that previously repacked the winning shift after fused scoring. On H100,
the complete 8,192 x 2,048 constructor improved again from 8.69ms to 6.77ms
(1.28x, or 3.04x cumulatively versus the original 20.59ms path). Across two
warm alternating 32/8/32 x 512-token pairs, aggregate candidate work improved
from 0.515s to 0.465s (1.11x) and candidate wall time from 0.233s to 0.215s
(1.084x). A 64/16/64 x 256-token pair improved candidate work 1.086x and wall
time 1.070x. Every matched 14-tensor state, guard, changed-tile count, and
held-out loss was bit-for-bit equal; strict split disjointness passed and gains
remained exactly 15.0000% and 29.2717%.

FP32-trained attention reconstruction can also write its final BF16 tensor
directly from the fused Hadamard kernel. The transform, scale multiplication,
saved tensor, and backward reduction remain FP32 in the original operation
order; only the final store converts to BF16. This removes the separate cast
kernel without changing optimization math. On H100, the isolated 2,048 x 512
and 2,048 x 2,048 operators improved 1.27x and 1.20x. Across warm alternating
32/8/32 x 512-token runs, attention fitting improved 1.023x by paired means
(1.055x by three-run medians). The 64/16/64 x 256-token geometry improved
attention fitting 1.059x and total fitting 1.031x. All matched 14-tensor states,
gradients in direct operator tests, guards, changed-tile counts, and held-out
losses were bit-for-bit equal; strict split disjointness passed and gains
remained exactly 15.0000% and 29.2717%. The optimization is enabled by default
in the staged driver and can be isolated with
`--no-direct-bf16-attention-output`.

Fixed-shape attention optimizer updates are replayed through one CUDA graph.
The graph receives fresh private-generator Gumbel draws and live FP32
temperature, multiplier, learning-rate, and decay values before every replay;
it therefore preserves the GSQ schedule instead of freezing capture-time
constants. Reciprocal-temperature multiplication retains the eager
Python-scalar division order exactly, and warm-up/capture mutations are
restored before training begins. On H100, 32/8/32 x 512-token attention fitting
improved from 0.3489s to 0.2089s (1.67x), reducing total fitting from 1.4035s
to 1.2842s (1.093x). Nsight measured 9,533 eager kernel-launch API calls
inside the attention range versus 2,281 kernel/graph launch calls (4.18x fewer).
At 64/16/64 x 256 tokens, attention improved from 0.5721s
to 0.2522s (2.27x) and total fitting from 1.8595s to 1.5988s (1.163x). Both
matched runs produced bit-for-bit equal 14-tensor states, guards, changed-tile
counts, and held-out losses; strict split disjointness passed and gains remained
exactly 15.0000% and 29.2717%. The staged driver enables this path by default;
use `--no-cuda-graph-attention-updates` for eager comparison.

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
| BF16 attention prefix plus six-epoch FP32 tail | Warm attention time was unchanged (~0.339s), while held-out gain fell from 15.0000% to 14.6429% |
| Q/K CUDA graph replay depth 8 or 16 | States were bit-for-bit equal, but warm Q/K time did not improve over replay depth 4 |

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
