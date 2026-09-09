# QQQ: W4A8 smoothing and compensation

## Sources and finding

Zhang et al., [QQQ: Quality Quattuor-Bit Quantization for Large Language Models,
v3](https://arxiv.org/html/2406.09904v3);
[authors' code](https://github.com/HandH1998/QQQ).

QQQ combines adaptive smoothing and Hessian-based compensation for four-bit
weights and eight-bit activations. The paper co-designs W4A8 kernels to address
both prefill and decode. Reported speedups depend on the measured kernels,
shapes and hardware; they are not transferable performance guarantees.

## Repository evidence

[gptqmodel/quantization/qqq.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qqq.py) and
[gptqmodel/looper/qqq_processor.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/qqq_processor.py) implement the dedicated route;
[dispatch](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/models/base.py) selects QQQ separately from
GPTQ and AWQ. Check the QQQ consumer's scale tensors and workspace lifetime when
changing packing or execution.

## Recovery implications

This is a useful comparison for activation-aware native low-bit inference, but
W4A8 is neither W4A4 nor NVFP4. Replacing its activation quantizer changes the
error distribution and scale contract.

Ablate smoothing, weight compensation and runtime activation rounding separately.
Capture the actual native output before fitting [EoRA-like recovery](eora.md).
Report full-operator conversion/correction overhead and both token regimes.

## GSQ adapter audit (2026-09-09)

Source inspected at `7c10ccb8f`: `QQQLinear.pack`,
`QQQTorchLinear._dequantize_weight_for_torch`, `QQQ.add_batch`, and
`QQQ.quantize`. These are repository implementation findings, not paper claims.

For grouped W4, let `c` be the stored unsigned nibble, `s` the raw group
scale, and `t` the FP32 extra per-output-channel scale. Packing stores
`r = FP16(s / t)`. The Torch runtime's effective weight is
`clamp(round((c - 8) * FP32(r)), -128, 127) * t`.
It is generally **not** `(c - 8) * s`: group-ratio storage and the second
INT8 rounding/saturation both matter. Different nibbles can decode to the
same value. An affine scalar GSQ decoder would therefore score the wrong
candidate even with frozen scales.

`tests/test_gsq_qqq_contract.py` establishes this through actual CPU packing,
strict packed-state reload, and Torch weight decoding for all 16 nibbles,
two groups, and ratios 0.5, 1.5 and 17.0. These cover ties, duplicate decoded
values and INT8 saturation. All three cases pass; these synthetic fixtures
prove representation behavior only, not model quality or native GPU parity.

For a proposed deployed-output GSQ fit, capture rounded runtime inputs
`Xq = token_scale * INT8(X / token_scale)` as well as teacher inputs `X`.
The existing `QQQ.add_batch` accumulates only the unrounded-input Gram.
With `E = Wcandidate - Wteacher`, use `Hq = Xq Xq.T` and
`D = (X - Xq) Xq.T`; the candidate-dependent loss is
`tr(E Hq E.T) - 2 <E, Wteacher D>`. This is the same asymmetric quadratic
already used by the scalar GSQ fitter, but with activation quantization as
the source of the paired-input difference. Match runtime FP16 input casting
and token scaling before accumulating these statistics. Keep the dense
teacher and candidate in the same smoothing basis.

Next implementation requirements are a QQQ-specific hard/soft decoder,
paired calibration collection, preservation of both scale tables, and a
best-hard-payload guard. Channelwise signed-nibble behavior needs its own
endpoint audit. Do not enable QQQ GSQ merely by adding its format to the
affine adapter allowlist. Native inference and real disjoint Llama/F6 data
remain required before claiming compatible QQQ lifecycle support.

The initial candidate decoder is now implemented in
`gptqmodel/quantization/gsq_qqq.py::qqq_candidate_values`. It handles both
unsigned grouped and signed channelwise stored nibbles, including source-dtype
channel-scale division and FP16 grouped-ratio storage. It validates finite
positive scales and legal integer nibbles. Candidate-axis output permits GSQ
to mix decoded values rather than applying a nondifferentiable INT8 rounding
to the expected nibble. Scale learning is not implemented by this helper.

The expanded CPU suite reports **18 passed** (3.39 seconds): actual grouped
and channelwise packing/strict state reload, invalid codes/scales, and an
analytic gradient check for the mixture of decoded candidate values. The
optimizer, calibration collection and public QQQ GSQ configuration are still
pending; this decoder alone does not enable QQQ GSQ.

`qqq_calibration_moments` now implements the proposed paired statistics as
unnormalized sums, returning the token count so callers can normalize both
moments identically. It reproduces runtime FP16 casting, FP16 maximum/division,
INT8 rounding, and FP32 reconstructed activations. Zero rows contribute zero;
nonzero token-scale underflow and FP16 input overflow fail explicitly rather
than using undefined float-to-integer conversions as calibration evidence.

The CPU contract suite now reports **22 passed** (3.53 seconds). Additional
checks compare activation values through the actual runtime `dynamic_quant`,
verify the asymmetric quadratic against explicit reconstruction-loss
differences, and check additive batch statistics for FP16/BF16/FP32 source
inputs. These are algebra/runtime-contract fixtures, not real-model quality
measurements. The helper is not yet connected to `QQQ.add_batch`; default
quantization behavior remains unchanged.

`refine_qqq_codes` now performs optional fixed-scale Gumbel-Softmax fitting
with a private seeded generator, geometric temperature schedule, and Adam.
It mixes the already-decoded QQQ candidate values, uses the paired quadratic
objective, and retains the baseline unless a hard assignment improves it.
Channelwise local candidates use signed distance across the nibble encoding
boundary. Group ownership and both scale tables remain fixed. Scale learning
is rejected explicitly pending an appropriate two-scale optimizer.

The function returns integer assignments, avoiding an unsupported assumption
that decoded INT8 weights can be fed back through the raw-scale W4 packer
without changing their codes. Binding those assignments to the existing
producer/packer lifecycle is the next requirement. The CPU suite reports
**26 passed** (3.64 seconds), including grouped/channelwise local/full grids,
determinism, preservation of the global RNG, and independent hard-score
recalculation. These fixtures do not establish real-model improvement.
