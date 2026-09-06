# RTN: round-to-nearest baseline and fallback

## Repository sources

[gptqmodel/quantization/rtn.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/rtn.py),
[gptqmodel/looper/weight_only_processor.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/weight_only_processor.py),
and [gptqmodel/quantization/fallback_smooth.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/fallback_smooth.py).
RTN is a general rounding procedure, not a separately identified paper.

## Implementation finding

RTN rounds against a selected scale/codebook without GPTQ's Hessian compensation.
QVQ's repository supports a calibration-free RTN lifecycle and smoothing/fallback
choices. `RTNConfig` uses METHOD.GPTQ and GPTQ-compatible exports in
[config](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/config.py); there is no separate
METHOD.RTN enum.

Smoothing or MSE scale search changes the baseline. Label plain RTN,
smoothed RTN and a fallback invocation separately rather than treating every
GPTQ-labelled artifact as a second-order result.

## Recovery implications

Use RTN as a controlled ablation for codebook/scale error versus compensation.
Keep grouping, zero convention, precision and export identical where possible.

Do not interpret a fallback that avoids a numerical failure as proven quality
recovery. Record which modules fell back and compare actual held-out outputs.
The helper's percentile/MAD/log/row-column smoothing names are not evidence that
the published SmoothQuant method ran; see [rotation and smoothing](rotation-and-smoothing.md).
