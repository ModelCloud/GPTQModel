# GSQ W3 paper-budget test on Llama 3.2 1B

The opt-in test in `tests/models/test_llama3_2_gsq_paper_w3.py` keeps the
Llama 3.2 1B Instruct model and W3/group128 GPTQ format. It matches the
sequence counts and optimizer settings in the [GSQ Llama experiment](https://arxiv.org/pdf/2604.18556),
using the existing nm calibration parquet in place of FineWeb-Edu. The test
retains the project's 128-question GSM8K Platinum evaluation; the paper uses
five different zero-shot tasks on larger models.
The [full FineWeb-Edu repository](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
is about 5.84 TB; the paper's [`sample-10BT` subset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT)
is 28.5 GB, whereas the local nm parquet is 12.8 MB.

| Setting | Paper Llama run | This test |
| --- | --- | --- |
| Calibration source | FineWeb-Edu | Pinned `nm-calibration/llm.parquet` |
| GPTQ initialization | 512 sequences × 4,096 tokens | Same count and length, separate source rows |
| GSQ training | 4,096 sequences × 4,096 tokens | Same count and length, separate source rows |
| Validation | 128 sequences × 4,096 tokens | Same count and length, separate source rows |
| GSQ schedule | 20 epochs, batch 64, microbatch 2 | Same |
| Q/K updates | 2,000 | Same |
| Optimizer | Lion; logits LR 1e-4, scales LR 5e-5; cosine LR | Same |
| Sampling | temperature 2→0.05, multiplier 100→500 | Same |
| W3 logit initialization | prior strength 6, noise std 0.01 | Same |
| Evaluation | Five zero-shot tasks on Llama 3.1 8B/70B | 128 GSM8K Platinum questions on Llama 3.2 1B |

The nm file has 10,000 rows, 12,792,319 bytes, and SHA256
`26122fd822e64d2fc704b0fe84af7a2df8a24a4455d347e16a6b5a7484f5cbef`.
After a seed-42 row shuffle, source rows are partitioned into 8,649 training,
270 validation, and 1,081 GPTQ rows. Each partition repeats only its own
text to fill complete 4,096-token sequences. The resulting streams contain
16,777,216 training tokens, 524,288 validation tokens, and 2,097,152 GPTQ
tokens. Their respective unique source-token streams contain about 2.93M,
0.085M, and 0.354M tokens; the text is therefore repeated roughly six times.
The split is disjoint by source row, but its diversity does not match the paper.

The public staged model path captures each split separately at every decoder
block. GPTQ initialization, Q/K statistics, and late MLP initialization use
the GPTQ split. V/O and MLP training use the training split, with held-out
validation losses recorded each epoch. Completed packed blocks are replayed
when capturing subsequent blocks. Full-length activations use disk-backed
capture because keeping the 4,096 training sequences on the GPU would exceed
practical memory limits on this host.

The test uses SDPA attention and records hard-quantized validation loss; these
are implementation and reporting choices, not a claim of numerical identity
with the authors' training run. Repeating a smaller source corpus, changing the
model, and changing the evaluation task also limit direct score comparisons.

Llama's `module_tree` has explicit `q`, `k`, `v`, `o`, `gate`, `up`, and `down`
role tags, as well as the existing shared-input tags and projection subsets.
The staged GSQ path still selects the seven projections by name; the role tags
do not alter its training objective. The parser and full seven-projection
checkpoint path have focused tests.

Run the full test only when the required long training window and disk space
are available:

```bash
GPTQMODEL_RUN_GSQ_PAPER_W3_E2E=1 python -m pytest -q \
  tests/models/test_llama3_2_gsq_paper_w3.py
```

The dense Llama 3.2 1B model scored 54/128 (42.2%) with the same GSM8K
settings. The full 512-sequence, 4,096-token W3 GPTQ initializer scored
**5/128 (3.9%)** after all 16 blocks were packed, saved, and reloaded. The
initializer is a measured reference; the configurable 20% quality gate applies
to GSQ. A full 20-epoch GSQ quality result has not yet been measured.

The prior small-budget W3 check scored 6/128 for the GPTQ initializer and
2/128 after GSQ. Those scores are historical for the 25-sequence, 10-epoch
recipe and must not be presented as results of the paper-budget test.

A one-block geometry smoke used 64 training documents, two GPTQ documents,
and two validation documents at 4,096 tokens. With batch 64, microbatch 2,
one epoch, and one Q/K update, all four stages completed and the held-out
V/O and MLP hard losses were finite. The block quantization took 21.6 seconds
after dataset preparation, and its disk capture was cleaned afterward. This
checks the full sequence and batch shapes; it is not the 20-epoch quality run.

## Training speed experiment

The attempted full 20-epoch run was stopped after its first block had reached
MLP step 436/1,280. That run produced no GSQ quality score. The temporary
activation capture was used to benchmark the training loop, then removed.

The speed experiment used 128 captured Llama 3.2 1B documents of 4,096 tokens,
W3/group128, batch 64, microbatch 2, BF16 projections, SDPA, and one CUDA GPU.
It compared the same two optimizer batches with the original eager relaxation
and teacher evaluation against the fused CUDA relaxation and fixed-target cache.
Both paths used the same stage geometry and loss weighting. The caches preserve
the original microbatch boundaries. Measurements include disk reads and GPU
synchronization at the end of each two-batch epoch, but exclude one-time cache
construction, capture, GPTQ initialization, hard validation, packing, and
generation.

| Stage | Eager, 2 batches | Optimized, 2 batches | Speedup |
| --- | ---: | ---: | ---: |
| MLP, trial 1 | 18.34 s | 4.41 s | 4.16× |
| MLP, trial 2 | 19.08 s | 4.49 s | 4.25× |
| V/O attention, trial 1 | 3.54 s | 1.98 s | 1.79× |
| V/O attention, trial 2 | 3.36 s | 1.91 s | 1.76× |

Summing the separately measured MLP and V/O stage times gives approximately
3.4–3.5× for those two stages. Their cache construction took 10.4 s and
2.0 s respectively for 128 documents. A separate 2,048×2,048 Q/K projection
microbenchmark with its quadratic loss and FP32 logits measured 33 ms eager
versus 13 ms fused per warmed update, about 2.5×. These results do **not**
establish a 4× end-to-end GSQ speedup or a quality gain.

The optional CUDA runtime-compiled path supports W3/W4 with BF16 or FP32
assignment logits and any positive contiguous group size. It uses the same
scalar relaxation and gradient equations, but GPU reductions and transcendental
functions have small rounding differences from PyTorch eager. The numerical
tests compare weights and gradients with tolerances. Set
`GPTQMODEL_GSQ_DISABLE_CUDA_RELAXATION=1` to use the eager implementation;
CUDA configurations without the runtime compiler also fall back to eager.

The precision audit found and corrected a Q/K discrepancy in the first fused
version: FP32-logit softmax probabilities were stored in BF16 even though the
eager scalar path retains FP32 probabilities. The corrected kernel also
accumulates scalar shifts in FP32, as the eager W3/W4 candidates do. A crafted
near-certain-choice test protects the small but nonzero FP32 gradient at the
lowest scheduled temperature. In a separate 2,048×2,048 W3/group128 probe,
the corrected fused and eager FP32 logit gradients had no sign disagreements
among 20,971,520 entries at each of four schedule points, including
temperature 0.05 and multiplier 500. This is a local numerical probe, not a
full model quality measurement.

Three matched-randomness, 30-update tests compare W3 BF16, W4 BF16, and W3
FP32 training trajectories, including hard choices and reconstruction loss.
A paired tiny-Llama W3 run trained, packed, and evaluated all seven projections
after three epochs and three Q/K updates. The eager and corrected CUDA paths
produced identical packed-model logits and measured hard-stage losses on that
case. The full 20-epoch Llama 3.2 1B GSM8K result remains unmeasured.

### Corrected-kernel stage timing

After the FP32 probability correction, the two-stage comparison was repeated
using the Llama 3.2 1B layer dimensions, W3/group128, 4,096-token sequences,
batch 64, microbatch 2, and 128 synthetic captured documents. Each path saw
the same synthetic activations and two optimizer batches. The source documents
were hard-linked copies of one random BF16 activation, so this is a
shape-matched training-speed measurement rather than a quality experiment.
The source and temporary target caches were removed after the benchmark.

| Stage | Eager trial 1 / 2 | Optimized trial 1 / 2 | Speedup |
| --- | ---: | ---: | ---: |
| MLP | 18.08 / 18.11 s | 4.39 / 4.40 s | 4.11–4.12× |
| V/O attention | 3.51 / 3.39 s | 1.93 / 1.91 s | 1.77–1.82× |
| Sum of measured stages | 21.59 / 21.50 s | 6.32 / 6.30 s | 3.41× |

MLP target-cache construction took 7.60 s and V/O target-cache construction
took 2.06 s for these 128 documents. Those one-time costs, Q/K training,
GPTQ initialization, validation, packing, and evaluation are excluded from the
stage timing. CPU prefetch of the next attention optimizer batch was also
tested; its timing varied between slower and faster than the current path,
so it was not incorporated. A 4× complete-run gain remains unverified.
