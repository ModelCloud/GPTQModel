---
name: gptqmodel-quantization-regressions
description: Diagnose severe GPT-QModel post-quantization quality regressions by localizing error to a layer, module, output channel, quantization math, packing boundary, backend kernel, tokenizer, or evaluation path. Use when low-bit generations become invalid or repetitive, benchmark scores collapse, quantizer loss or saved scales spike, 2-bit/3-bit behavior unexpectedly diverges from a higher-bit control, or an engineer needs a dense/RTN/static-group comparison before changing quantization code.
---

# GPTQModel Quantization Regression Diagnosis

Use boundary-first evidence to find the earliest stage at which a model diverges. Do not blame packing or an inference
kernel when the quantizer's pre-pack reconstruction loss is already broken.

## Required workflow

1. Read `../gptqmodel-quantization/SKILL.md` and its required packing reference before changing quantization code.
2. Establish a dense BF16/FP16 baseline and a native quantized baseline without adapters or other experimental
   processors. Compare rendered prompts and input IDs before treating low scores as a quantization regression.
   For a joint quantization-plus-adapter pipeline, evaluate both the saved base alone and that base with the exact
   adapter generated in the same run; neither replaces the adapter-free quantization control.
3. Record the exact model revision, quantization config, calibration selection/order/hash, software versions, device
   identity, and backend. Hold them fixed across controls.
4. Stop expensive adapter or evaluation sweeps when the adapter-free quantized baseline is corrupt.
5. Locate the earliest failing boundary in this order:

   - quantizer reconstruction and logged pre-pack loss;
   - reconstructed base weight before and after any coupled adapter processor;
   - saved scale, zero-point, group-index, and integer-code tensors;
   - independent manual unpack/dequantization;
   - eager quantized-linear output;
   - optimized backend/kernel output;
   - end-to-end logits, generation, and evaluation.

6. Compare matched layer/module records across a healthy higher-bit reference and the failing snapshot. Then reduce the
   suspicious module by output channel. Use cosine, RMSE, relative RMSE, maximum scale, scale percentiles, legal code
   range, and error concentration—not only whole-tensor averages.
7. For a suspicious channel, compare GPTQ against independent symmetric RTN with the same bit width and group size.
   Follow with controlled `static_groups` or selective-RTN experiments; change one variable at a time.
8. Preserve CPU, non-target GPU, and backend fallbacks. Put regression assertions in `tests/` and timed investigations
   in `scripts/`.

Read [references/diagnostic_workflow.md](references/diagnostic_workflow.md) for formulas, interpretation, controls,
and the evidence table to report.

## Fast snapshot comparison

Run the bundled tool from the repository root. The default reads only `quant_log.csv` and is cheap:

```bash
python .agents/skills/gptqmodel-quantization-regressions/scripts/analyze_quant_regression.py \
  --snapshot 4bit=/path/to/healthy-4bit \
  --snapshot 3bit=/path/to/3bit \
  --snapshot 2bit=/path/to/failing-2bit
```

Add `--scan-scales` only when module-loss comparison identifies a suspicious area or checkpoint-wide channel evidence
is needed. It reads every saved `.scales` tensor:

```bash
python .agents/skills/gptqmodel-quantization-regressions/scripts/analyze_quant_regression.py \
  --snapshot 4bit=/path/to/healthy-4bit \
  --snapshot 2bit=/path/to/failing-2bit \
  --scan-scales \
  --json /tmp/quantization-regression.json
```

Treat the first snapshot as the comparison reference. Inspect both the raw candidate/reference ratio and that ratio
divided by the matched median ratio; lower bit widths normally worsen globally, while a localized regression worsens
far more than the normal global shift.

When two same-bit snapshots have matching quantizer logs or metadata but different behavior, add `--scan-codes`.
This gated path compares every logical packed-weight code without materializing the entire model unpacked at once,
reports per-module mismatch rates and code deltas, and checks exact equality of scales, zero-points, and group indices:

```bash
python .agents/skills/gptqmodel-quantization-regressions/scripts/analyze_quant_regression.py \
  --snapshot native=/path/to/native \
  --snapshot coupled=/path/to/coupled \
  --scan-codes \
  --json /tmp/packed-code-comparison.json
```

Only compare logical codes between snapshots with the same bit width and packing layout. A code mismatch with exact
scales, zero-points, and group indices proves that the reconstructed weight presented to packing differed; it does
not by itself prove the packer mutated identical inputs. Capture pre-pack fingerprints to distinguish those cases.

## Quantization-time feedback

`QuantizeConfig.quantization_diagnostics` accepts:

- `auto` (default): O(number of quantized modules), summarizes loss and warns when one module is both at least 50x the
  mean and at least 25% of total logged loss.
- `channel`: includes `auto`, scans scale tensors for output-channel candidates, and samples logical codes after GPTQ,
  immediately before packing, and after packing to detect coupled-processor weight-state changes.
- `off`: disables the summary.

`GPTQMODEL_QUANTIZATION_DIAGNOSTICS=off|auto|channel` overrides the config for one process. Keep `auto` enabled in normal
quantization. Use `channel` deliberately because it reduces every scale element and synchronizes summary values to the
host. The code fingerprint samples at most 4,096 logical codes per module instead of unpacking the full model. Saved
models include `quantization_diagnostics.json`.

## Stop conditions

Do not claim a root cause from correlation alone. A diagnosis is ready only when an independent boundary check
reproduces or excludes the suspected failure. Examples:

- Bad pre-pack loss plus direct saved-scale reconstruction parity excludes packing and inference as the original cause.
- Manual unpack unequal to eager dequantization implicates serialization/packing/dequantization.
- Eager output correct but optimized backend output wrong implicates backend selection or kernel behavior.
- Exact token IDs and a healthy dense score exclude tokenizer normalization; otherwise use the tokenizer skill.
- A broken joint base that remains broken with its exact matching adapter excludes only a missing-adapter evaluation
  mistake; compare an otherwise identical adapter-free quantization before assigning the failure to shared
  finalization or replay.
