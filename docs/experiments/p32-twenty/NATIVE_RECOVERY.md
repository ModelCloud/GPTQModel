# Experiment 19: deployed W4A16 plus output-residual recovery

Four real layer-0 projections were exported from the read-only F6 seed-7 teacher.
The base uses TorchAO tiled INT4 storage, group size 128, BF16 activations/output,
and the actual tinygemm execution path. This is W4A16, not W4A4 INT4-activation
IMMA. Recovery uses FP32 factors and two GEMMs with FP32 output addition.

Folded FP32 weights reproduce held-out teacher outputs with MAE 2.3e-8–6.5e-7
and max <=2.3e-5. Fitting uses the actual residual Y_teacher(X)-Y_native(X),
including native BF16 rounding and kernel arithmetic. It does not fit only a
weight dequantization residual. Calibration is the separate hash-verified historical
Fisher capture: 2048 token rows. Evaluation uses the earlier C4 capture, never
included in fitting. This bounded calibration is not representative of the full
historical dataset. The FP64 reduced-rank solve truncates activation singular
values below 1e-5 of the largest; exported factors are FP32.

Ranks 16/32/64/128 were measured at all nine required M values, against same-device
planar and window full operators with FP32 transforms. These are layer timings,
not whole-model timings or deployed FP16-model boundary validation.

| Projection | Rank-128 passing row counts | M=2048 MAE | M=2048 speedup vs window | Serialized rank-128 operator BPW |
|---|---:|---:|---:|---:|
|q_proj|0/9|0.014693|2.10x|8.2555|
|k_proj|0/9|0.018198|3.53x|14.2722|
|gate_proj|0/9|0.019200|1.90x|6.7514|
|down_proj|9/9|0.001628|1.96x|6.7514|

At M=1/16, corrected operators measured about 7–9x faster than window, including
conversion, low-rank GEMMs and output addition. Full-loop samples and every
rank/case error are retained. Large-M gain is much smaller than small-M gain.
The down projection is the promising passing candidate. The gate projection's
rank-128 correction reduces calibration MAE to 0.007412 but worsens held-out MAE
from 0.012216 to 0.019200, with max error 3.220186. Do not hide this failure.

## Priority human exception option

Failing candidates exceed the >500% speed-gain threshold at small M. A human may
review a scoped research/opt-in exception for the specific projection, rank and
row counts in the result files; no exception is approved by these measurements.
Recommendation: keep q/k/gate unpromoted and improve calibration/recovery first.
Their errors substantially exceed MAE <=0.003 and max <=0.046875. No downstream
model evidence exists for these exports, and large-M gains are only 1.9–3.5x.
Down-projection-only integration can proceed to validation under the normal gate.

Actual rank-specific serialized operator bytes include packed weights, scales,
format metadata, factors, dimensions, and arithmetic configuration. Files remain
outside Git under /root/p32-native-recovery. The figures are operator BPW, not
whole-model BPW; dense tensors, remaining P32 projections, tokenizer/config and
runtime buffers must be included for a complete model comparison. FP32 factors
make the small k_proj particularly expensive. Export reload validation remains
pending. The original F6 checkpoint was never modified or re-quantized.

[All rank/case summaries](results/native-recovery/summary.json). The experiment
remains partial: wider calibration, full model quality/performance, instruction
profiling, combined storage accounting and promotion decisions are outstanding.
