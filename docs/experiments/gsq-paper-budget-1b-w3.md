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

The prior small-budget W3 check scored 6/128 for the GPTQ initializer and
2/128 after GSQ, both below the usual 20% GSM8K quality gate. These values
are historical for the 25-sequence, 10-epoch recipe and must not be presented
as results of the paper-budget test. A paper-budget quality result has not
yet been measured.

A one-block geometry smoke used 64 training documents, two GPTQ documents,
and two validation documents at 4,096 tokens. With batch 64, microbatch 2,
one epoch, and one Q/K update, all four stages completed and the held-out
V/O and MLP hard losses were finite. The block quantization took 21.6 seconds
after dataset preparation, and its disk capture was cleaned afterward. This
checks the full sequence and batch shapes; it is not the 20-epoch quality run.
