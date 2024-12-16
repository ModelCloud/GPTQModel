def patch_vllm():
    from typing import Dict, List, Type

    from vllm.model_executor.layers import quantization
    from vllm.model_executor.layers.quantization import gptq_marlin

    from .src.vllm import gptq_marlin as gptqmodel_marlin

    # vllm 0.6.4 refactored the quantization methods to be a list
    if isinstance(quantization.QUANTIZATION_METHODS, List):
        def get_quantization_config(quant: str) -> Type[quantization.QuantizationConfig]:
            if quant not in quantization.QUANTIZATION_METHODS:
                raise ValueError(f"Invalid quantization method: {quantization}")

            # lazy import to avoid triggering `torch.compile` too early
            from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
            from vllm.model_executor.layers.quantization.awq import AWQConfig
            from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
            from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
                CompressedTensorsConfig  # noqa: E501
            from vllm.model_executor.layers.quantization.deepspeedfp import DeepSpeedFPConfig
            from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8Config
            from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8Config
            from vllm.model_executor.layers.quantization.fp8 import Fp8Config
            from vllm.model_executor.layers.quantization.gguf import GGUFConfig
            from vllm.model_executor.layers.quantization.gptq import GPTQConfig
            from vllm.model_executor.layers.quantization.gptq_marlin_24 import GPTQMarlin24Config
            from vllm.model_executor.layers.quantization.hqq_marlin import HQQMarlinConfig
            from vllm.model_executor.layers.quantization.ipex_quant import IPEXConfig
            from vllm.model_executor.layers.quantization.marlin import MarlinConfig
            from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
            from vllm.model_executor.layers.quantization.neuron_quant import NeuronQuantConfig
            from vllm.model_executor.layers.quantization.qqq import QQQConfig
            from vllm.model_executor.layers.quantization.tpu_int8 import Int8TpuConfig

            method_to_config: Dict[str, Type[quantization.QuantizationConfig]] = {
                "aqlm": AQLMConfig,
                "awq": AWQConfig,
                "deepspeedfp": DeepSpeedFPConfig,
                "tpu_int8": Int8TpuConfig,
                "fp8": Fp8Config,
                "fbgemm_fp8": FBGEMMFp8Config,
                "modelopt": ModelOptFp8Config,
                # The order of gptq methods is important for config.py iteration over
                # override_quantization_method(..)
                "marlin": MarlinConfig,
                "gguf": GGUFConfig,
                "gptq_marlin_24": GPTQMarlin24Config,
                "gptq_marlin": gptqmodel_marlin.GPTQMarlinConfig,
                "awq_marlin": AWQMarlinConfig,
                "gptq": GPTQConfig,
                "compressed-tensors": CompressedTensorsConfig,
                "bitsandbytes": BitsAndBytesConfig,
                "qqq": QQQConfig,
                "hqq": HQQMarlinConfig,
                "experts_int8": ExpertsInt8Config,
                "neuron_quant": NeuronQuantConfig,
                "ipex": IPEXConfig,
            }

            return method_to_config[quant]

        quantization.get_quantization_config = get_quantization_config
    elif isinstance(quantization.QUANTIZATION_METHODS, Dict):
        # before vllm 0.6.4, the quantization methods were a dictionary
        quantization.QUANTIZATION_METHODS["gptq_marlin"] = gptqmodel_marlin.GPTQMarlinConfig
        gptq_marlin.GPTQMarlinLinearMethod = gptqmodel_marlin.GPTQMarlinLinearMethod
