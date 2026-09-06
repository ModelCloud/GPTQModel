# FP8: representation versus activation policy

## Sources and finding

[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) introduces
E4M3 and E5M2 formats with different range/precision tradeoffs. A format name
alone does not specify scaling granularity, calibration, accumulation precision
or whether activations and KV caches are quantized.

## Repository evidence

[gptqmodel/nn_modules/qlinear/fp8.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/nn_modules/qlinear/fp8.py) defines a weight-only
`TorchFP8Linear` and tensor/channel/block weight scaling. The quantizer
rejects dense weight conversion to E8M0; that format is reserved there for
dequantizing existing checkpoints.
[gptqmodel/looper/weight_only_processor.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/weight_only_processor.py) routes FP8 direct packing.

Separately, [gptqmodel/quantization/qvq_activation.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq_activation.py) is QVQ's
activation-quantization policy. Do not infer that METHOD.FP8 and a QVQ A8 option
are the same execution path.

## Recovery implications

Record the exact FP8 dtype, scale versus inverse-scale convention, block geometry,
and runtime arithmetic. Weight-only storage, W8A8 execution and FP8 KV storage
are three distinct claims requiring independent evidence.

The [NVFP4 paper's FP8 KV scales](nvfp4-hybrid-ptq.md) do not calibrate the
linear-input FP8 quantizer. Fit [output recovery](eora.md) against whichever
operator actually executes and revalidate after changing the scale policy.
