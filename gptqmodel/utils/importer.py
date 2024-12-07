from collections import OrderedDict
from typing import Dict, Optional, Type, Union

import torch

from ..models._const import DEVICE
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.bitblas import BitBLASQuantLinear
from ..nn_modules.qlinear.dynamic_cuda import DynamicCudaQuantLinear
from ..nn_modules.qlinear.exllama import ExllamaQuantLinear
from ..nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from ..nn_modules.qlinear.ipex import IPEXQuantLinear
from ..nn_modules.qlinear.marlin import MarlinQuantLinear
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..nn_modules.qlinear.tritonv2 import TRITON_AVAILABLE, TRITON_INSTALL_HINT, TritonV2QuantLinear
from ..quantization import FORMAT
from ..utils.logger import setup_logger
from .backend import BACKEND, get_backend

message_logged = False
logger = setup_logger()

backend_dict = OrderedDict({
    BACKEND.MARLIN: MarlinQuantLinear,
    BACKEND.EXLLAMA_V2: ExllamaV2QuantLinear,
    BACKEND.EXLLAMA_V1: ExllamaQuantLinear,
    BACKEND.TRITON: TritonV2QuantLinear,
    BACKEND.CUDA: DynamicCudaQuantLinear,
    BACKEND.BITBLAS: BitBLASQuantLinear,
    BACKEND.IPEX: IPEXQuantLinear,
    BACKEND.TORCH: TorchQuantLinear,
})

format_dict = {
    FORMAT.GPTQ: [BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA, BACKEND.IPEX, BACKEND.TORCH],
    FORMAT.GPTQ_V2: [BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA, BACKEND.TORCH],
    FORMAT.MARLIN: [BACKEND.MARLIN],
    FORMAT.BITBLAS: [BACKEND.BITBLAS],
    FORMAT.IPEX: [BACKEND.IPEX],
}

# public/stable api exposed to transformer/optimum
def hf_select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        checkpoint_format: str,
        meta: Optional[Dict[str, any]] = None,
        device_map: Optional[Union[str, dict]] = None,
        backend: Optional[Union[str, BACKEND]] = None,
) -> Type[BaseQuantLinear]:
    # convert hf string backend to backend.enum
    if isinstance(backend, str):
        backend = get_backend(backend)

    if device_map is not None:
        devices = [device_map] if isinstance(device_map, str) else list(device_map.values())
        if "cpu" in devices or torch.device("cpu") in devices:
            device = DEVICE.CPU
        elif "xpu" in devices or torch.device("xpu") in devices:
            device = DEVICE.XPU
        else:
            device = DEVICE.CUDA
    else:
        device = DEVICE.CPU

    return select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        backend=backend,
        device=device,
        format=FORMAT.GPTQ,
        pack=True,
        allow_marlin=False, # TODO: remove this after marlin padding is fixed
        dynamic=None,
    )


# auto select the correct/optimal QuantLinear class
def select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        device: Optional[DEVICE] = DEVICE.CUDA,
        backend: BACKEND = BACKEND.AUTO,
        format: FORMAT = FORMAT.GPTQ,
        pack: bool = False,
        allow_marlin: bool = True,  # TODO: remove this after marlin padding is fixed
        dynamic=None,
) -> Type[BaseQuantLinear]:
    if not torch.cuda.is_available():
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = DEVICE.XPU
        else:
            device = DEVICE.CPU

    backend = BACKEND.AUTO if backend is None else backend

    # Handle the case where backend is AUTO.
    if backend in [BACKEND.AUTO, BACKEND.AUTO_TRAINABLE]:
        trainable = backend == BACKEND.AUTO_TRAINABLE

        allow_backends = format_dict[format]

        # TODO: fix marlin padding
        # Since Marlin does not support padding in_features and out_features, Marlin is not allowed for hf_select_quant_linear scenarios
        # for gptq internal use, allow_marlin is set to True
        if format in [FORMAT.GPTQ, FORMAT.GPTQ_V2] and allow_marlin:
            allow_backends = [BACKEND.MARLIN] + allow_backends

        allow_quant_linears = backend_dict
        err = None
        global message_logged
        # Suppose all quant linears in the model should have the same backend.
        for k, v in allow_quant_linears.items():
            in_allow_backends = k in allow_backends
            validate, err = v.validate(bits, group_size, desc_act, sym, dynamic=dynamic, device=device, trainable=trainable)
            if in_allow_backends and validate:
                if pack:
                    check_pack_func = hasattr(v, "pack")
                    if check_pack_func:
                        if not message_logged:
                            logger.info(f"Auto choose the fastest one based on quant model compatibility: {v}")
                            message_logged = True
                        return v
                else:
                    if not message_logged:
                        logger.info(f"Auto choose the fastest one based on quant model compatibility: {v}")
                        message_logged = True
                    return v

        if err:
            raise err

    # Handle the case where backend is not AUTO.
    if backend == BACKEND.TRITON:
        if not TRITON_AVAILABLE:
            raise ValueError(TRITON_INSTALL_HINT)
        return TritonV2QuantLinear
    elif backend == BACKEND.BITBLAS:
        return BitBLASQuantLinear
    elif backend == BACKEND.MARLIN:
        return MarlinQuantLinear
    elif backend == BACKEND.EXLLAMA_V2:
        return ExllamaV2QuantLinear
    elif backend == BACKEND.EXLLAMA_V1:
        return ExllamaQuantLinear
    elif backend == BACKEND.CUDA:
        return DynamicCudaQuantLinear
    elif backend == BACKEND.IPEX:
        from ..nn_modules.qlinear.ipex import IPEX_AVAILABLE
        if not IPEX_AVAILABLE:
            raise ValueError("IPEX is not available.")

        from device_smi import Device

        cpu_vendor = Device("cpu").vendor
        if cpu_vendor != "intel":
            logger.warning(f"Intel/IPEX cpu kernel is only validated and optimized for Intel cpu. Running on non-Intel cpu is not guaranteed. Current cpu vendor: `{cpu_vendor}`.")

        return IPEXQuantLinear
    elif backend == BACKEND.TORCH:
        return TorchQuantLinear
    else:
        return TorchQuantLinear
