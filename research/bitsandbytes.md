# bitsandbytes: FP4/NF4 storage and runtime integration

## Sources and finding

[bitsandbytes Linear4bit documentation](https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit)
describes blockwise four-bit layers with FP4/NF4 choices and configurable compute
dtype. [QLoRA](https://arxiv.org/abs/2305.14314) introduces NF4 and double
quantization in a low-rank fine-tuning workflow.

Loading or quantizing a layer with bitsandbytes does not mean QLoRA training
occurred. NF4 and bitsandbytes FP4 are not interchangeable with native NVFP4.

## Repository evidence

[gptqmodel/nn_modules/qlinear/bitsandbytes.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/nn_modules/qlinear/bitsandbytes.py) is the adapter;
[gptqmodel/looper/weight_only_processor.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/weight_only_processor.py) selects direct packing
for METHOD.BITSANDBYTES.
[configuration](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/config.py) defines the
actual accepted options. Backend dependencies and device restrictions still apply.

## Recovery implications

Report codebook, block size, nested scale compression, compute dtype and packed
state. “Four-bit” storage does not imply A4 or native four-bit tensor-core GEMM.

Both QLoRA and EoRA can produce low-rank factors, but their objectives and
generation procedures differ: QLoRA fine-tunes, whereas standard EoRA solves a
calibration-weighted residual approximation. Do not treat adapter compatibility
as proof that either workflow has been run or validated for every backend.
