# Seed-7 vocabulary-head full-corpus calibration, 2026-09-25

This is an offline **first-block** bit-rate screen for the tied-input,
untied-output Llama 3.2 1B Seed-7 checkpoint. It is not a complete quantized
head, serving artifact, or downstream accuracy result.

## Inputs and capture

- Model: `modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-zml-cuda-v3__qvq-p32__yaqa125x__seed7__20260908`.
- Calibration: the original Seed-7 YAQA parquet; SHA256
  `5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39`.
- Capture: 10,178 independent sequences, 3,961,260 valid tokens, batch size
  1, rank-32 shared-head streaming-projected Fisher with exact diagonal.
  The output off-diagonal is approximate. Capture took 546.71 s on H100.
- Cache: `qvq.yaqa.shared-head-factor.v2` safetensors, 17 MiB, SHA256
  `d198eb6cbbdc874933678edb84396cef81200226d298f708961269fdc34b4f6f`.
  The original source-diagonal reductions are preserved. The cache is kept
  outside Git at `/root/work/qvq-vocab-shared-head-full-20260925.safetensors`.
- This calibration source is disjoint from the 1,209 GSM8K-Platinum test
  questions by the earlier normalized overlap audit. No test row was used
  to fit the factors.

The head has 128,256 output rows and 2,048 input channels. The probe
quantized rows 0–2,047 at each rate, plus the final 1,280-row block at W3.
Every arm reused the exact same cached factor. Quantization requested 640
GSQ updates, 33 legal candidates, and one optional deterministic coordinate
sweep. **Every selected arm below was the coordinate comparator, not an
improvement attributable to Gumbel training.**

## First-block results

Positive numbers mean less quadratic error than the matched no-GSQ
reconstruction for that same rate. The prepared objective uses the
transformed, damped metric; the independent source oracle is computed on
the reconstructed weight. FP32 and FP64 oracles were both measured.

| Rate | Format | Changed tiles | Prepared gain | Undamped source FP64 gain | 5%-damped source FP64 gain |
| --- | --- | ---: | ---: | ---: | ---: |
| W2.5 | P32 | 1 | +0.5967% | +54.8625% | +0.6672% |
| W3 | P32 | 3 | +0.4930% | +32.9424% | +0.3773% |
| W3.5 | P32 | 1 | +0.3456% | +4.8774% | **−0.0387%** |
| W4 | Planar QVQ | 1 | +0.9351% | +35.1669% | +0.7107% |

The W3.5 damped regression is small but measurable: its candidate's
FP32/FP64 damped-oracle gap was 0.00021%, and the baseline gap was
0.00035%. Keep the candidate for mitigation or bounded quality testing;
do not promote it solely from the prepared objective. For W2.5, W3, and
W4, all three local objectives moved favorably. The final 1,280-row W3
block also quantized successfully: two changed tiles, +0.6030% prepared,
+5.2798% undamped FP64, and +0.5507% damped FP64.

W4 currently uses the planar QVQ layout because this probe's W4 P32 route
is not available. It needs its own serving compatibility check. The dense
input embedding remains resident; only the output head is being explored.

## Interpretation and remaining gates

The absolute Fisher value has no universal good/bad threshold. Compare
matched before/after values under the **same data, block, factor rank,
damping, and rate**. Diagonal damping explains most of the large difference
between the undamped and prepared relative gains here. None of these local
numbers predicts the final discrete GSM8K score by itself.

Required next work: quantize and assemble all 63 head blocks, preserve
cross-block Fisher interactions in whole-head arbitration, serialize a
checkpoint, integrate a compressed-head load/dispatch path in ZML-Ultra,
then run the 1,209-row held-out GSM8K-Platinum suite against the current
543-correct/zero-invalid baseline. Useful and padded prefill/decode
throughput must be compared on the same B128 runner before claiming the
requested speed gain. Input embedding quantization requires a separate
compressed token-lookup operator and its own quality gate.
