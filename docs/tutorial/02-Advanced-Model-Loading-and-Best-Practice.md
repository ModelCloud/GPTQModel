# Advanced Model Loading and Best Practice
Welcome to the tutorial of GPTQModel, in this chapter, you will learn advanced model loading and best practice in `gptqmodel`.

## Arguments Introduction
In previous chapter, you learned how to load model into CPU or single GPU with the two basic apis:
- `.from_pretrained`: by default, load the whole pretrained model into CPU.
- `.from_quantized`: by default, `gptqmodel` will automatically find the suitable way to load the quantized model.
  - if there is only single GPU and model can fit into it, will load the whole model into that GPU;
  - if there are multiple GPUs and model can fit into them, will evenly split model and load into those GPUs;
  - if model can't fit into GPU(s), will use CPU offloading.

However, the default settings above may not meet many users' demands, for they want to have more control of model loading.

Luckily, in GPTQModel, we provide some advanced arguments that users can tweak to manually config model loading strategy:
- `device_map`: an optional `Union[str, Dict[str, Union[int, str]]]` type argument, currently only be supported in `.from_quantized`.

Before `gptqmodel`'s existence, there are many users have already used other popular tools such as [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) to quantize their model and saved with different name without `quantize_config.json` file introduced in previous chapter.

To address this, two more arguments were introduced in `.from_quantized` so that users can load quantized model with arbitrary names.
- `quantize_config`: an optional `QuantizeConfig` type argument, can be used to match model file and initialize model incase `quantize_config.json` not in the directory where model is saved.
- `model_basename`: an optional `str` type argument, if specified, will be used to match model instead of using the file name format introduced in previous chapter.

## Multiple Devices Model Loading

### device_map
So far, only `.from_quantized` supports this argument. 

You can provide a string to this argument to use pre-set model loading strategies. Current valid values are `["auto", "balanced", "balanced_low_0", "sequential"]`

In the simplest way, you can set `device_map='auto'` and let 🤗 Accelerate handle the device map computation. For more details of this argument, you can reference to [this document](https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).

## Best Practice

### At Quantization
It's always recommended to first consider loading the whole model into GPU(s) for it can save the time spend on transferring module's weights between CPU and GPU.

However, not everyone have large GPU memory. Roughly speaking, always specify the maximum memory CPU will be used to load model, then, for each GPU, you can preserve memory that can fit in 1\~2(2\~3 for the first GPU incase CPU offload used) model layers for examples' tensors and calculations in quantization, and load model weights using all others left. By this, all you need to do is a simple math based on the number of GPUs you have, the size of model weights file(s) and the number of model layers.

### At Inference
For inference, following this principle: always using single GPU if you can, otherwise multiple GPUs, CPU offload is the last one to consider.

## Conclusion
Congrats! You learned the advanced strategies to load model using `.from_pretrained` and `.from_quantized` in `gptqmodel` with some best practice advices. In the next chapter, you will learn how to quickly customize an GPTQModel model and use it to quantize and inference.
