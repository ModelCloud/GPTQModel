# Expanded calibration and actual FP16 deployment

All four projections were refit on 8192 tokens from 16 independent 512-token
sequences of the verified historical Fisher source. Calibration activations are
rounded to FP16 before both the canonical FP32 teacher and actual W4A16 native
base execute; correction uses those same rounded values in FP32, followed by the
actual FP16 output conversion. C4/ARC data never enter the fit.

The key result is **down-projection rank16 passes all nine local cases**, including
a separate export reload/FP16-boundary check. M=2048 MAE is 0.00183859 and max
0.04452360. Ranks32/64/128 pass only 6/9 cases: increasing rank improves mean error
slightly but worsens worst-element error. q/k/gate remain failing at rank128.

The rank16 standalone operator uses **4.5639 BPW** including actual serialization,
versus 6.7514 at rank128. These are operator BPW, not a newly packed whole-model
checkpoint. Rank16 layer speedup versus window is 7.39x at M=1, 7.86x at M=16,
and 2.05x at M=2048. The measurement includes the native base, FP32 correction and
output conversion; its comparator uses the documented FP32-transform harness.

Replacing only model.layers.0.mlp.down_proj in the real window model produces
C4 perplexity **27.035818912832937**, versus **26.9915252378136** for window and
26.99102051460363 for the FP32 P32 teacher. The same 128 ARC-Challenge examples
score **38.28125% raw / 39.0625% normalized**, versus window's
36.71875% / 37.5%. That is two additional correct answers in each metric. The
small sample and mixed PPL/task effects do not establish a quality improvement.
Original BF16 remains 37.5% / 40.625% on this ARC slice and has better C4 PPL.

The same-device full-model window/replacement timing pair is recorded separately.
Only one projection is replaced, so its layer gain is not a claim of comparable
whole-model speedup. No model-wide promotion or gate exception was applied.
Further calibration/quality uncertainty analysis, broader layer coverage, final
checkpoint storage and native instruction profiling remain outstanding.

[Results](results/native-expanded/model-paired-timing.json), with all rank/case
metrics, serialized-byte counts, reload checks and bounded model results nearby.
