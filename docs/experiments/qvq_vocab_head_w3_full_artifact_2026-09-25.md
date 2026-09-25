# Seed-7 W3 vocabulary-head artifact, 2026-09-25

This is an offline quantized **output-head** experiment. It does not yet
establish a GSM8K-Platinum score, an inference speedup, or quantized input
embeddings. The full-corpus Fisher capture and first-block rate screen are
recorded in [the preceding experiment](qvq_vocab_head_full_corpus_2026-09-25.md).

## Fixed inputs and scope

- Source: the Seed-7 Llama 3.2 1B P32/Rank-8 checkpoint named in the preceding
  experiment. Its input embedding stays dense; the tied output head is untied
  for quantization.
- Calibration: 10,178 original YAQA sequences, 3,961,260 valid tokens, disjoint
  from the 1,209 held-out GSM8K-Platinum questions by normalized overlap audit.
- Fisher: shared rank-32 full-head factor with exact output diagonal. Cross-block
  interactions are retained, but off-diagonal Fisher information is approximate.
- Format: W3 P32, 62 blocks of 2,048 output rows plus one block of 1,280 rows;
  128,256 vocabulary rows in total. Both the no-GSQ and GSQ arms use the same
  cached Fisher, damping, weights, and deterministic seed.
- GSQ schedule: 640 requested updates, 33 legal candidates, one coordinate
  sweep. A selected block is not evidence that Gumbel updates themselves won:
  the deterministic coordinate comparator can provide the improvement.

The complete baseline and candidate safetensors are each 100,585,087 bytes.
The GSQ arm changed 123 tiles across 63 blocks. Local block improvements did
not imply a whole-head improvement because the errors interact through the
shared output factor.

## Whole-head arbitration

The oracle reconstructs each serialized **planar** P32 payload through the
canonical QVQ weight decoder, applies the original FP16 output boundary, then
scores its difference from the original output head. It evaluates both the
undamped factor objective and a 5%-damped objective in FP32 and FP64. The
guarded arm greedily admits only block changes that improve both full-head
FP64 objectives; the final FP32 objectives must also avoid regression.

Lower loss is better.

| Arm | FP64 undamped | FP64 damped | FP32 undamped | FP32 damped |
| --- | ---: | ---: | ---: | ---: |
| No GSQ | 2.38191662e-5 | 0.00209481620 | 2.38189132e-5 | 0.00209481595 |
| Every GSQ candidate | 2.41444061e-5 | 0.00209823682 | 2.41441812e-5 | 0.00209823647 |
| Guarded, 22 of 63 candidate blocks | 6.20855428e-6 | 0.00207826438 | 6.20837636e-6 | 0.00207826449 |

Relative to the no-GSQ W3 artifact, the guarded arm improves the FP64
undamped oracle by 73.93% and the FP64 damped oracle by 0.790%. The full
candidate arm regresses those objectives by 1.365% and 0.163%, respectively.
These are calibration-proxy results, not task accuracy. The large undamped
change is particularly sensitive to the rank-32 factor approximation.

The on-disk arbitration report is outside Git at
`/root/work/qvq-vocab-head-w3-full-dual-20260925/arbitration.json`; the
`baseline.safetensors`, `candidate.safetensors`, and `guarded.safetensors`
files are in that same directory. A PyTorch loader installed all 63 guarded
blocks and produced finite FP16 logits of shape `[1, 1, 128256]`. Its 15.05 ms
per-call smoke timing is **not** a ZML serving benchmark.

## Promotion gates still open

1. Load the compressed head through ZML-Ultra/Inference-Ultra without keeping
   a duplicate dense output head. Check the actual P32 layout and dispatcher.
2. Run matched full B128 GSM8K-Platinum against the current Seed-7 baseline
   of 543/1,209 correct and zero invalid; the requested policy is no lower
   aggregate score. Record paired row changes, useful and padded prefill and
   decode throughput, per-stream decode, and VRAM.
3. Complete the full-head W2.5, W3.5, and W4 arms on the same calibration
   factor and repeat serialized FP32/FP64 whole-head arbitration. W4 needs a
   separate serving compatibility check because the available path is planar.
4. Quantizing the input embedding requires its own compressed token-lookup
   operator and separate downstream gate. The W3 output-head result alone does
   not imply any embedding result.

The target is at least the established GSM8K score after quantization, **not**
100% literal question accuracy. The current evidence only establishes that the
guarded W3 artifact is a valid full-head offline candidate for that test.
