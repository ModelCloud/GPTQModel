# MXFP4: microscaling and the CPU integration

## Sources and finding

[Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537)
describes shared-scale low-precision formats. MXFP4 uses E2M1 elements and
power-of-two E8M0 block scales; its scale geometry differs from
[NVFP4](nvfp4-hybrid-ptq.md). Similar four-bit elements do not make payloads
or calibration policies interchangeable.

## Repository evidence

[gptqmodel/quantization/config.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/config.py) exposes METHOD.MXFP4 and
`MXFP4Config`. [gptqmodel/nn_modules/qlinear/mxfp4_cpu.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/nn_modules/qlinear/mxfp4_cpu.py)
defines a CPU weight-only backend with 32-divisible input width, symmetric
storage and no activation ordering.
[gptqmodel/utils/mxfp4_cpu.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/utils/mxfp4_cpu.py) supplies quantize/dequantize and
extension helpers.

This proves a named format/config and CPU module integration, not every
end-to-end quantization route. In particular, the audited weight-only
processor's direct-pack method set names GGUF/FP8/bitsandbytes, not MXFP4;
trace the complete caller before claiming equivalent lifecycle support.

## Recovery implications

Distinguish packed four-bit weights from the arithmetic used after unpacking.
A CPU MXFP4 module is not evidence of native GPU W4A4 support.

Include block-scale storage in BPW and verify scale encoding, padding and
round-trip dequantization. For QVQ-to-MXFP4 conversion, fit against the installed
destination operator and exported factor precision, not an idealized dense
weight approximation.
