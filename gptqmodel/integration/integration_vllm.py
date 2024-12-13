def patch_vllm():
    from vllm.model_executor.layers.quantization import gptq_marlin
    from .src.vllm import gptq_marlin as gptqmodel_marlin

    gptq_marlin.GPTQMarlinConfig = gptqmodel_marlin.GPTQMarlinConfig
    gptq_marlin.GPTQMarlinLinearMethod = gptqmodel_marlin.GPTQMarlinLinearMethod
