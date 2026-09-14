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

`qqq_codes_to_packer_weight` now reconstructs transport weights using the raw
producer scales and verifies the existing packer's inverse arithmetic after
casting to the requested producer dtype. It refuses non-finite or changed
assignments. These transport weights must not be confused with the effective
INT8-decoded weights used for scoring. Actual QQQ packing preserves all 16
selected codes for grouped and channelwise layouts with FP16/BF16/FP32
producer weights in the added tests. The CPU suite reports **32 passed**
(4.32 seconds). Quantizer/collector binding and real-model checks remain open.

QQQ's quantizer now collects paired moments only for modules selected by
`gsq`, runs its original initializer, and optionally refines the resulting
packed code assignments. It returns the exact original weight when no hard
candidate improves; accepted assignments pass through the verified transport
helper. Statistics are released after use and by `free`. The existing inherited
`QQQConfig.gsq` field remains disabled by default. Enabled use currently requires
an unpadded Linear module and fixed scales; unsupported requests fail explicitly.

The original initializer had an unconditional CUDA synchronization; it now
synchronizes only CUDA weights, on their actual device. This allows the actual
quantizer to run in the CPU contract tests without mocking numerical operations.
The expanded suite reports **36 passed** (3.64 seconds), including GSQ on/off,
grouped/channelwise initialization, packing and Torch forward. These use
synthetic inputs strictly for lifecycle correctness. Real Llama calibration,
held-out evaluation, GPU parity, broader ordering/fallback cases and full-model
save/load/generate are still required before a QQQ quality/support claim.

Default-off verification found a pre-existing QQQ damping round-trip defect:
the QQQ initializer replaced the effective static damping with 0.005 but left
the legacy serialized scalar at GPTQ's 0.05. Reloading made that stale scalar
authoritative. QQQ now resynchronizes the scalar aliases when installing its
own default static configuration; explicit adaptive configurations retain the
parent behavior. Default 0.005 and explicit 0.02 round trips are tested.

The CPU suite now reports 42 passed (4.07 seconds), including byte-exact
weight/scale/zero/group/extra-scale parity against the original initializer
for GSQ None, disabled and unmatched filters, group sizes -1/128, and both
activation-order settings. Config reload and global RNG preservation are
included. This is lifecycle evidence; real-model QQQ evaluation remains open.

Enabled-path ordering validation now covers all combinations of group size
-1/128, activation ordering on/off, static groups on/off and GSQ on/off.
For enabled GSQ, the test independently decodes the actual packed weight,
quantizes inputs with the runtime routine, and computes deployed reconstruction
loss minus the candidate-independent asymmetric constant. That score agrees
with the quantizer diagnostic (relative 2e-5, absolute 1e-7). The expanded CPU
suite reports 54 passed (4.58 seconds). This verifies score-to-packer binding
across those ordering settings, not real-model recovery or GPU parity.

Native QQQ preflight (2026-09-09, SM80 physical GPU 0) compiled and executed
the existing grouped parity suite: four regular-value cases passed, but the
rounding/saturation fixture failed on two of nine values, with maximum absolute
difference 229. This is an existing native-versus-Torch contract discrepancy,
not a GSQ quality result. Since GSQ currently scores the Torch saturation
contract, native QQQ support must remain unverified until this is resolved.
The lease was released. Raw log: `/tmp/gsq-qqq-native.log`.

## Native saturation root cause and proposed arithmetic correction

`dequant_per_group` in `gptqmodel_ext/qqq/qqq_gemm.cu` uses half2 FMA with
magic value 1152 (`0x6480`), then extracts low bytes and XORs 128. It does not
clamp the resulting value. Thus +154 becomes -102; -160 becomes +64. The
negative side also crosses the FP16 binade at 1024, changing rounding spacing.

A proposed correction clamps the existing fused result to [1024,1279] before
byte extraction (`0x6400`/`0x64ff` per half lane). This preserves the fused
rounding order, avoiding a separate rounded half multiply. An exhaustive CPU
arithmetic audit covers all 16 signed W4 codes and 31,743 positive finite FP16
scale patterns (507,888 pairs, including subnormals). Float64 evaluates exact
products/sums before the final half conversion. The proposed mapping agrees
with round-to-even followed by INT8 saturation for every pair, and leaves the
magic result unchanged for products within [-128,127]. Both audit tests pass
(0.10 seconds), including reproduction of the known native endpoint outputs.

This is a proposed kernel correction backed by exhaustive representation math;
the CUDA source is not changed yet. Actual native tests, executed-instruction
inspection, graph checks and matched correctness/timing remain required.
