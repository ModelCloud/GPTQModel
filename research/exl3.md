# EXL3: ExLlamaV3 quantization integration

## Primary source

[ExLlamaV3](https://github.com/turboderp-org/exllamav3) documents EXL3 as a
QTIP-based quantization format and inference implementation. Use its own format
documentation as authority; shared trellis ancestry does not imply QVQ/P32
binary compatibility. No separate EXL3 paper is asserted here.

## Repository evidence

[gptqmodel/looper/exllamav3_processor.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/exllamav3_processor.py) captures calibration inputs,
calls the vendored `quantize_exl3`, stages packed tensors, and installs EXL3
modules. [dispatch](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/models/base.py) requires CUDA/HIP
quantization and explicitly rejects EoRA adapter generation and GPTAQ/FOEM
native activation capture in this route.

The existence of upstream ExLlamaV3 LoRA support does not override those local
quantization-lifecycle exclusions. ExLlamaV2 backends that execute GPTQ/AWQ
weights are different from selecting METHOD.EXL3.

## Recovery implications

Inspect EXL3's decoder, codebook and transforms before proposing format conversion.
Do not reuse a [P32](p32.md) payload or assume [EoRA](eora.md) generation works
without extending and validating the integration. Compare serialized bytes and
actual inference, not only nominal bit rates.
