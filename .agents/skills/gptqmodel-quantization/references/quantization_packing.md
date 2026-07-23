# Safe quantization packing

Use this reference whenever quantized values are reconstructed, converted to integer codes, bit-packed, serialized,
unpacked, or compared across CPU and accelerator implementations. Treat saturation as a correctness boundary, not as
an optional cleanup step.

## Contents

1. [Mandatory invariants](#mandatory-invariants)
2. [Separate the numeric domains](#separate-the-numeric-domains)
3. [Choose saturation or failure deliberately](#choose-saturation-or-failure-deliberately)
4. [Validate inputs before code generation](#validate-inputs-before-code-generation)
5. [Generate bounded integer codes](#generate-bounded-integer-codes)
6. [Handle post-quantization weight changes](#handle-post-quantization-weight-changes)
7. [Pack bits without changing code semantics](#pack-bits-without-changing-code-semantics)
8. [Validate after packing and after reload](#validate-after-packing-and-after-reload)
9. [Diagnose endpoint wraparound](#diagnose-endpoint-wraparound)
10. [Required test matrix](#required-test-matrix)
11. [Implementation workflow](#implementation-workflow)
12. [Review checklist](#review-checklist)

## Mandatory invariants

Enforce all of these at every packing entry point:

1. Define the logical code domain explicitly. For an unsigned `b`-bit GPTQ-style code, use
   `qmin = 0` and `qmax = 2**b - 1`.
2. Reject non-finite weights, scales, offsets, or zero-points before integer conversion.
3. Require finite positive scales unless the quantization method explicitly defines another contract.
4. Round in a documented floating-point dtype and with a documented tie rule.
5. Saturate the rounded floating-point value before the first integer cast. If the producer already emits integers in
   a proven-wide dtype, saturate before narrowing or shifting.
6. Never use a bit mask as a substitute for saturation. `code & qmax` encodes modulo arithmetic and wraps out-of-range
   values to the opposite endpoint.
7. Validate logical zero-points independently from weight codes. Invalid metadata is normally an error, not a value to
   repair silently.
8. Make every packer implement the same numeric contract: native CPU, Python fallback, GPU, and legacy/original.
9. Unpack and dequantize representative outputs, then compare them with the saturated reference before trusting save or
   inference results.

Do not assume an upstream quantizer always emits exact representable floating-point reconstructions. Packing often
reconstructs integer codes from BF16 or FP16 weights and rounded scales. A value intended to be `0` or `qmax` can become
`-1` or `qmax + 1` after the inverse transform. Post-quantization corrections make this even more common.

## Separate the numeric domains

Name and preserve each domain instead of treating them as interchangeable:

| Domain | Example | Required behavior |
|:---|:---|:---|
| Dense weight | `W` | Finite floating point |
| Scale | `s` | Finite and positive |
| Logical zero-point | `z` | Integer in the method's declared code domain |
| Unrounded code | `q_real = W / s + z` | Wide floating point; may be outside the code domain |
| Rounded value | `q_round = round(q_real)` | Floating point with integral values; may be outside the code domain |
| Saturated code | `q = clamp(q_round, qmin, qmax)` | Exactly representable logical code |
| Packed word | layout-specific bits | Lossless encoding of `q`, never a second quantizer |
| Dequantized weight | `W_hat = s * (q - z)` | Reference for numerical comparison |

For offset-style code, such as `(W + scale_zero) / scale`, apply the same rules. Prove algebraically that its
zero-point convention matches the loader before relying on a superficially similar formula.

Keep signed and unsigned domains distinct:

- Most GPTQ/AWQ packers store unsigned logical codes and apply a logical zero-point during dequantization.
- A true signed format must clamp in its signed domain first. Convert a validated signed code to its physical two's
  complement bit pattern only after the range check.
- Never reinterpret a signed negative intermediate as an unsigned code merely by masking it.

Adapt the boundary to the format instead of forcing every scheme into affine integers:

- For codebook quantization, validate or saturate the value before selecting a codebook entry, then require the stored
  index to be in `[0, codebook_size - 1]`. An invalid index is a packing error, not a value to wrap.
- For FP8, FP4, or another minifloat, follow the format's declared overflow behavior: maximum finite saturation,
  infinity, or a reserved encoding. Test NaN, infinity, subnormal, and signed-zero policy explicitly.
- For binary, ternary, or signed-magnitude formats, define the semantic value set and physical encoding separately.
- For mixed-bit formats, derive the range from each block's actual bit width; never apply one global mask.
- For sparse or sentinel-bearing formats, reserve sentinel codes before quantizing ordinary values and reject collisions.

## Choose saturation or failure deliberately

Use saturation for a valid floating-point reconstruction at the final quantization boundary:

- inverse-transform drift from BF16, FP16, or rounded serialized scales;
- a clipping-based quantizer whose mathematical definition includes saturation;
- a post-quantization floating-point weight that is intentionally requantized with fixed scales;
- a shared packing API that must defensively encode only representable codes.

Fail loudly instead of hiding an upstream contract violation when any of these occur:

- `NaN` or infinity in a weight, scale, offset, or zero-point;
- a zero or negative scale outside a method that explicitly defines it;
- a logical zero-point outside its declared domain;
- an already-integer quantizer result outside its promised domain;
- a shape, group-index, layout, or format mismatch;
- a large or unexpected saturation rate that indicates the wrong weight tensor, scales, zero-point convention, or
  orientation reached the packer.

Saturation prevents memory-safe bit packing from corrupting endpoints, but it must not conceal a lifecycle bug. Record
the number and magnitude of underflows and overflows when diagnosing a new path. If incidence is not explainable by
boundary rounding, trace the producer before accepting the result.

## Validate inputs before code generation

Check the complete packing contract before calculating a code:

- `bits`, logical signedness, and `qmin`/`qmax`;
- weight orientation (`[out, in]` versus `[in, out]`);
- scale and zero-point orientation;
- group size, group count, and exact `g_idx` length;
- negative `g_idx` normalization, if the format supports it;
- `sym`, `desc_act`, activation ordering, and dynamic per-layer overrides;
- quantization method and serialized format;
- qzero format/version and any legacy zero-point offset convention;
- pack word dtype and width;
- source weight, scale, offset, and destination device/dtype.

Use wide arithmetic for the inverse transform. Prefer FP32 for reconstructed codes even when stored weights and scales
are BF16 or FP16. Preserve the method's specified rounding rule; PyTorch `round` uses ties-to-even. For stochastic
rounding, apply the declared stochastic rule before saturation and use a fixed seed in differential tests. Do not
change the rounding rule as part of an unrelated packing optimization.

Validate finiteness before converting to an integer. Floating-to-integer conversion of `NaN` or infinity is not a
portable error-handling mechanism and can produce implementation-dependent sentinel values.

## Generate bounded integer codes

Use this order:

```python
q_real = reconstruct_code(weight.float(), scale.float(), zero)
if not torch.isfinite(q_real).all():
    raise ValueError("non-finite reconstructed quantization code")

q_round = torch.round(q_real)
q = q_round.clamp(qmin, qmax).to(torch.int64)
```

Clamp before converting to `int32` or `int64`; a very large finite float can overflow even a wide integer cast. Use
`int32` instead of `int64` after saturation only when every later shift is proven safe. Do not first cast to `int8`,
`uint8`, or the final packed dtype.

For diagnostics, capture these values before discarding `q_round`:

```python
under = q_round < qmin
over = q_round > qmax
stats = {
    "underflow_count": int(under.sum()),
    "overflow_count": int(over.sum()),
    "minimum_rounded_code": float(q_round.min()),
    "maximum_rounded_code": float(q_round.max()),
    "maximum_underflow": float((qmin - q_round[under]).max()) if under.any() else 0.0,
    "maximum_overflow": float((q_round[over] - qmax).max()) if over.any() else 0.0,
}
```

Keep such aggregation out of production hot paths unless it is behind an explicit diagnostic option. Never log
per-element values from a large model.

## Handle post-quantization weight changes

Identify exactly which tensor is being packed:

- canonical quantized weight `Wq`;
- original dense weight `W`;
- corrected replay weight such as `Wq + B @ A`;
- merged adapter weight;
- pruned, smoothed, rotated, or otherwise transformed weight.

Do not pack a corrected or merged weight with `Wq` scales accidentally. Preserve a pristine canonical `Wq` or its
integer codes across calibration replay and restore it before base-model packing. Serialize adapter factors separately
when that is the public format.

If requantizing a modified weight with fixed scales is intentional, treat it as a new lossy quantization step:

1. calculate codes in FP32;
2. measure underflow and overflow incidence;
3. saturate before packing;
4. measure weight error against both the modified target and the original dense reference;
5. validate the model with and without any separately applied correction to detect double application.

Compare object identity, storage identity, dtype, device, shape, and a bounded numerical fingerprint at lifecycle
handoffs. A correct adapter calculation does not prove that the base packer received the intended tensor.

## Pack bits without changing code semantics

Treat packing as a lossless layout transform of already-valid codes.

- Assert or prove `qmin <= q <= qmax` before shifting.
- Promote codes to a wide unsigned-safe arithmetic type before left shifts.
- Mask the final machine word to its word width only when required by the language's signed integer representation.
- Do not mask each logical code to repair range errors.
- Avoid signed right-shift assumptions for negative inputs.
- Handle codes that straddle word boundaries explicitly. Continuous 3-bit layouts require carry bits across adjacent
  32-bit words and deserve dedicated boundary tests.
- Pack qzeros with the format's logical convention, not the weight-code convention by analogy.
- Preserve non-divisible tail behavior only where the format explicitly supports padding or partial words.

For `b`-bit unsigned logical codes, this is unsafe:

```python
packed |= (q_round & ((1 << b) - 1)) << shift  # wraps -1 and 2**b
```

This is safe:

```python
q = q_round.clamp(qmin, qmax)
packed |= q.to(torch.int64) << shift
```

## Validate after packing and after reload

Perform validation at four boundaries:

1. **Reference codes**: compare each implementation with the same saturated logical-code tensor.
2. **Packed words**: require bitwise equality across original, native CPU, Python fallback, and GPU packers when they
   implement the same format.
3. **Unpacked/dequantized weights**: unpack the words, assert the code domain, and compare exactly with
   `scale * (saturated_code - logical_zero)`.
4. **Saved model**: reload through a real backend, verify metadata and module contracts, then compare logits,
   generation, and a bounded evaluation with a dense or known-good quantized baseline.

Check output shape, dtype, device, finiteness, scale dtype, zero-point format, and group indices in addition to numerical
error. A successful pack, save, or compile is not evidence of correctness.

## Diagnose endpoint wraparound

Suspect missing saturation when any of these appear:

- code deltas of exactly `+(2**bits - 1)` or `-(2**bits - 1)`;
- expected code `0` becoming `qmax`, or expected `qmax` becoming `0`;
- a small number of endpoint errors causing a disproportionate norm or cosine collapse;
- healthy direct rounding with saved scales but poor packed dequantization;
- identical scales and zero-points across good and bad artifacts but different packed words;
- 2-, 4-, or 8-bit paths succeeding while a Python-only 3-bit path fails;
- coherent quantization-time replay followed by gibberish only after save/reload.

Build a transition matrix between expected and unpacked codes. Report endpoint swaps separately from ordinary
off-by-one differences. Repairing endpoint swaps in a diagnostic copy can support the hypothesis, but do not ship a
post-hoc packed-word repair; fix the code-generation boundary and recreate the artifact.

First establish a non-quantized baseline and compare exact rendered prompts and token IDs. Then distinguish these
failure classes:

- quantization math produced bad scales or codes;
- packing changed valid codes;
- loading interpreted the format incorrectly;
- an inference kernel disagreed with eager dequantization;
- tokenizer or prompt normalization changed evaluation inputs.

## Required test matrix

Cover every changed implementation and the nearest risky boundaries.

| Dimension | Minimum coverage |
|:---|:---|
| Bits | 2, 3, 4, and 8 where supported |
| Code inputs | `qmin - 1`, `qmin`, `qmin + 1`, `qmax - 1`, `qmax`, `qmax + 1`, and larger excursions |
| Quantization | symmetric and asymmetric where supported |
| Groups | `-1` plus representative 32, 64, and 128 groups |
| Ordering | `desc_act=false` and `true` where the packer consumes reordered `g_idx` |
| Scale dtype | FP32 source plus BF16/FP16 serialized or runtime forms |
| Shapes | exact words, multiple words, supported tails, and non-square projections |
| Implementations | original/legacy, native CPU, forced Python fallback, and GPU |
| Formats | every affected qzero/version convention |
| Lifecycle | direct quantization and post-quantization corrected/replayed weight |

Create an adversarial tensor whose reconstructed rounded codes extend well below and above the legal range. Assert:

- exact saturation at both endpoints;
- bitwise-equal packed words across implementations;
- exact dequantization against the saturated reference;
- no mutation of source weights, scales, zero-points, or group indices;
- CPU fallback behavior without CUDA;
- the actual GPU implementation on supported hardware.

Add a small quantize/save/load/generate test when a model lifecycle can expose a different tensor to the packer. For a
model-level regression, compare sampled dequantized weights and last-token logits with dense, then run a bounded task
that detects invalid or repetitive output.

## Implementation workflow

1. Record the method, format, bits, group size, symmetry, `desc_act`, dtypes, shapes, pack implementation, hardware,
   and exact source revision.
2. Establish dense and known-good quantized references before editing.
3. Reproduce with eager dequantization and unpacked logical codes before blaming an inference kernel.
4. Trace the exact tensor, scales, zero-points, and `g_idx` into every packer.
5. Add saturation in the shared semantic boundary or consistently in every independent entry point.
6. Add the adversarial tensor regression before optimizing the implementation.
7. Compare packed words and dequantized weights across all implementations.
8. Recreate the affected model; do not reuse a corrupt checkpoint.
9. Reload with a generic reference backend, then compare specialized backends.
10. Run bounded generation/evaluation and report invalid-output counts.
11. Benchmark packing separately with warmup and synchronized timing if performance changed.
12. Preserve CPU and unsupported-device fallbacks.

For GPT-QModel, inspect these paths first:

- `gptqmodel/nn_modules/qlinear/__init__.py` for shared original, block, and GPU packing;
- `gptqmodel_ext/pack_block_cpu.cpp` for native CPU parity;
- `gptqmodel/utils/model.py::pack_module` for implementation selection;
- the producing processor under `gptqmodel/looper/` for weight lifecycle and saved scale/zero state;
- `gptqmodel/utils/model_dequant.py` and a generic quantized linear backend for unpack/dequant reference behavior;
- `tests/test_pack.py` for cross-implementation bitwise checks.

## Review checklist

Before approving a packing change, answer every item:

- Is the logical code range explicit for every supported bit width?
- Are non-finite values rejected before integer conversion?
- Are scales valid and zero-points independently range-checked?
- Does saturation happen after rounding but before narrowing, masking, or shifting?
- Do signed and unsigned formats clamp in the correct semantic domain?
- Do all pack implementations agree bit-for-bit?
- Are 3-bit word-boundary carries tested?
- Is the intended weight tensor packed after post-quantization replay or adapter math?
- Are saturation counts small and explained, or has a high rate been investigated?
- Does unpacked dequantization match the saturated reference?
- Does save/reload preserve metadata and numerical behavior?
- Does a real backend produce coherent output and bounded error versus a reference?
- Were CPU and non-target accelerator fallbacks preserved?
- Are hardware, dtype, shapes, commands, and test outcomes recorded?
