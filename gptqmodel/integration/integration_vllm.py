def patch_vllm():
    from vllm.model_executor.layers import quantization
    from vllm.model_executor.layers.quantization import gptq_marlin

    from .src.vllm import gptq_marlin as gptqmodel_marlin

    quantization.QUANTIZATION_METHODS["gptq_marlin"] = gptqmodel_marlin.GPTQMarlinConfig
    gptq_marlin.GPTQMarlinLinearMethod = gptqmodel_marlin.GPTQMarlinLinearMethod
