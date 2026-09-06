# GGUF: container, tensor types and quantization integration

## Primary source

The [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
defines a tensor/metadata container. GGUF itself is not one quantization algorithm:
different tensor types carry different block layouts and quantization schemes.

## Repository evidence

[gptqmodel/quantization/config.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/config.py) distinguishes structured GGUF
bit aliases/subtypes. [gptqmodel/looper/weight_only_processor.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/weight_only_processor.py)
routes the direct-pack lifecycle. Runtime entries include
[gptqmodel/nn_modules/qlinear/gguf.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/nn_modules/qlinear/gguf.py),
[gptqmodel/nn_modules/qlinear/gguf_cpp.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/nn_modules/qlinear/gguf_cpp.py) and
[gptqmodel/nn_modules/qlinear/gguf_triton.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/nn_modules/qlinear/gguf_triton.py).

Use the config parser's accepted alias set and the chosen backend's capability
checks. Upstream GGUF support for a tensor type is not proof of local quantization,
export, and inference support for it.

## Recovery implications

Record the actual tensor type, scales, block metadata, mixed per-tensor choices,
padding and consumer. Effective BPW need not equal the integer in a filename.

A checkpoint conversion must preserve the intended dequantization contract or
be labelled lossy. If converting a QVQ teacher to a GGUF-supported representation,
capture the destination runtime output before fitting correction. Container
support alone does not imply adapter support or W4A4.
