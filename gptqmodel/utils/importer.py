import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Type, Union

import torch

from ..models._const import DEVICE, normalize_device
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
from . import BACKEND
from .torch import HAS_CUDA, HAS_MPS, HAS_XPU


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

def normalize_device_device_map(device: Optional[Union[str, torch.device]], device_map: Optional[Union[str, Dict]]) -> Optional[DEVICE]:
    if device is None:
        if device_map is not None:
            devices = {device_map} if isinstance(device_map, str) else set(device_map.values())
            normalized_devices = set()
            for device in devices:
                # Returning None means quant linear will be automatically selected.
                if isinstance(device, str) and device == "auto":
                    return None
                normalized_devices.add(normalize_device(device))
            if len(normalized_devices) == 1:
                d = normalized_devices.pop()
                if d in DEVICE:
                    return d
            elif len(normalized_devices) > 1:
                normalized_devices.discard(DEVICE.CPU)
                return normalized_devices.pop()

        return None
    else:
        if isinstance(device, str):
            return DEVICE(device.split(":")[0])
        elif isinstance(device, torch.device):
            return DEVICE(device.type)
        else:
            raise ValueError(f"device must be a string or torch.device, got {type(device)}")


def auto_select_device(device: Optional[DEVICE], backend: Optional[BACKEND]) -> DEVICE:
    assert device is None or isinstance(device, DEVICE)
    assert backend is None or isinstance(backend, BACKEND)

    if device is None:
        if backend == BACKEND.IPEX:
            device = DEVICE.XPU if HAS_XPU else DEVICE.CPU
        elif HAS_CUDA:
            device = DEVICE.CUDA
        elif HAS_XPU:
            device = DEVICE.XPU
        elif HAS_MPS:
            device = DEVICE.MPS
        else:
            device = DEVICE.CPU
    return device

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
        backend = BACKEND(backend.lower())

    if device_map is not None:
        device = normalize_device_device_map(None, device_map)
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
            validate, err = v.validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, dynamic=dynamic, device=device, trainable=trainable)
            if os.environ.get("DEBUG") and in_allow_backends and not validate:
                logger.info(f"skip {k} for {str(err)}")
            if in_allow_backends and validate:
                if pack:
                    check_pack_func = hasattr(v, "pack")
                    if check_pack_func:
                        if not message_logged:
                            logger.info(f"Auto pick kernel based on compatibility: {v}")
                            message_logged = True
                        return v
                else:
                    if not message_logged:
                        logger.info(f"Auto pick kernel based on compatibility: {v}")
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
        from ..nn_modules.qlinear.ipex import HAS_IPEX
        if not HAS_IPEX:
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
