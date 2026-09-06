# ParoQuant: learned pairwise rotations

## Sources and finding

[ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference,
v2](https://arxiv.org/html/2511.10645v2);
[authors' implementation](https://github.com/z-lab/paroquant).

ParoQuant uses scaled pairwise rotations to improve quantization geometry, with
runtime kernels designed around those transforms. This is calibration-time
optimization of a transformed representation, unlike simply estimating an absmax
scale or adding a post-quantization residual branch.

## Repository evidence

[gptqmodel/quantization/paroquant/optimization.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/paroquant/optimization.py) explicitly describes
learning channel scales/Givens angles, optimizing transformed-domain quantization,
and exporting packed tensors matching the pseudo-quantized layer.
[gptqmodel/looper/paroquant_processor.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/paroquant_processor.py) provides the lifecycle.
The method enum is METHOD.PARO with serialized value `paroquant`.

## Recovery implications

Track learned rotations, channel scales, quantizer parameters and runtime
precision independently. A pseudo-quantized calibration win must survive
packing and the actual transform kernel.

[EoRA](eora.md) factors fitted in one coordinate system cannot be applied in
another without a consistent basis change. Compare additional recovery only
after the deployed ParoQuant operator is validated. Do not attribute its learned
scale/rotation procedure to the calibration-only [NVFP4 paper](nvfp4-hybrid-ptq.md).
