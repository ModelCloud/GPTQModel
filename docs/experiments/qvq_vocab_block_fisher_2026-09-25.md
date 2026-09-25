# Bounded vocabulary Fisher for the Seed-7 Llama endpoint

The merged endpoint lifecycle in QVQ `7326ec74bfd4fb0c54a05398ea4d0ecdb36826cd`
can requantize a small output head after a quantized decoder. The 128,256-row
Seed-7 Llama head still needs a bounded output metric: one FP32 square output
Gram is 61.28 GiB before the transformed copy and GSQ workspace.

`optimize/qvq_vocab_blocks.py` is an offline preparation experiment. It splits
the untied dense head into contiguous `nn.Linear` row blocks, concatenating
their outputs in vocabulary order. With block size 2,048, the Seed-7 head has
63 blocks: 62 with 2,048 rows and one with 1,280. All blocks obey the current
P32 16-column alignment. It is not yet a checkpoint or serving format.

## Fisher identity and limitation

Let `G_s = dL_s/dW` be a per-sequence head weight score and partition its
output rows into `G_{s,i}`. The principal output Fisher block is exactly

```text
F_out[i,i] = sum_s G_{s,i} G_{s,i}^T.
```

Therefore the existing YAQA collector can observe each `nn.Linear` block
while the **full** concatenated logits still determine the real-Fisher loss.
The input factor for each block uses that block's output width as its
normalizer; to reconstruct the full-head input factor, sum the block factors
weighted by `block_rows / total_vocab_rows`. The CPU fixture verifies exact
logits, these principal output factors, and the weighted input identity.

Independent block solves omit cross-block output Fisher terms. With output
error blocks `E_i`, the complete two-sided quadratic also contains terms
involving `E_i F_out[i,j] E_j^T` for `i != j`. A future cross-block GSQ pass
can keep these interactions bounded with a compact output factor `S` such
that `F_out ≈ S S^T`: accumulate `Z = sum_i E_i S_i`, and score
`tr(Z^T F_in Z)`. Changing one block updates only its contribution to `Z`.
`factored_head_fisher_loss` implements this full factored quadratic in FP32
and FP64. A double-precision fixture matches a dense output-Gram oracle and
verifies that cross-block terms remain present. This is a proposed QVQ
adaptation, not an exact dense Fisher replacement.
The factor rank, diagonal calibration, FP32/FP64 oracle agreement, and
held-out quality must be measured before promotion.

No full-model checkpoint, GSM8K score, or inference speed is claimed from
this preparation experiment. The tied input embedding still occupies its
original dense storage; replacing it needs a separate token lookup operator.

## Real-model diagnostic, 2026-09-25 UTC

With the Seed-7 QVQ Llama 3.2 1B checkpoint and the first original YAQA
calibration sequence (1,955 valid tokens), a warmed H100 run collected all
63 blocks' streaming-projected rank-8 input/output factors in **2.11 s**.
Materializing only block 0 produced two 2,048×2,048 FP32 factors. W3.5 P32
YAQA followed by four GSQ updates over three legal candidates completed in
**2.49 s** for that block. PyTorch peak allocated memory was **5,254 MiB**.
GSQ selected **zero changed tiles** at this deliberately tiny diagnostic
budget; calibration Fisher loss remained `2.25781946e-6` before and after.
An independent first-block quadratic evaluation was `6.86741332e-5` in FP32
and `6.85930701e-5` in FP64 (relative difference about 0.118%); these use
the rank-8 one-sequence sketch and are numerical diagnostics, not quality
gates. The first CUDA extension build took 211 s; subsequent capture timings
exclude that one-time compilation.

The checked-in probe reproduces the observation:

```bash
python -m scripts.experiments.qvq_vocab_block_probe \
  --model-path /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-zml-cuda-v3__qvq-p32__yaqa125x__seed7__20260908 \
  --calibration-parquet /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/calibration/source/yaqa182-nm10000.parquet \
  --output-json /root/work/qvq-vocab-block-w3p5-first-sequence-20260925.json \
  --dataset-size 1 --block-rows 2048 --gram-rank 8 --batch-size 1 \
  --bits 3.5 --gsq-steps 4 --gsq-candidates 3 --seed 7 --block-index 0 \
  --factor-mode independent-blocks
```

The repeat completed at 2026-09-25 11:52:32 UTC. Capture was 2.132 s,
block quantization was 2.479 s, and peak PyTorch allocation was
5,509,279,232 bytes. The independent oracle values reproduced exactly.
The 0.118% FP32/FP64 gap is a consistency diagnostic; it cannot establish
quality and would require closer analysis before accepting a GSQ gain of a
similar magnitude. The GSQ loss and independent oracle have different
normalizations and must not be compared directly.

The input is from the original 10,178-sequence YAQA source; the 1,209
GSM8K-Platinum test rows were not used. One calibration sequence and four
optimization updates do not qualify a W3.5 endpoint artifact. The complete
calibration pass, bit-rate sweep, serialization, ZML serving integration,
and held-out full-suite quality and throughput gates remain outstanding.

## Shared-head factor follow-up, 2026-09-25 UTC

The collector can also target the full dense `lm_head` once, using
`streaming_projected` YAQA. Its 128,256×rank source and exact diagonal are
bounded; only the 2,048×2,048 input Gram is materialized. The new
`YaqaGramSketch.factor()` applies the same diagonal correction and exposes
one normalized full-head output factor. Slicing its rows gives block-local
principal Grams **and compatible off-diagonal terms**. This removes the
independent-block random-projection mismatch described above. The probe's
`--factor-mode shared-head` path uses that factor and leaves model logits
unchanged throughout capture.

The same first calibration sequence and block-0 W3.5/P32 four-step GSQ probe
finished at 2026-09-25 11:56:22 UTC:

| Diagnostic | Shared-head result |
| --- | ---: |
| Capture time | 2.034 s |
| Block-0 quantization time | 2.492 s |
| PyTorch peak allocation | 5,600,208,896 bytes |
| GSQ changed tiles | 0 |
| Independent quadratic FP32 | `5.31350051e-5` |
| Independent quadratic FP64 | `5.31346847e-5` |
| FP32/FP64 relative gap | about 0.00060% |

The two probe modes use different randomized sketches, so their absolute
quadratic values must not be compared as a quality result. This establishes
a bounded, coherent factor for future cross-block optimization; the current
GSQ call still optimizes one block at a time. The full calibration corpus and
held-out serving accuracy have not yet been run.
