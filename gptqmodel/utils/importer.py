# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Type, Union

import torch
from gptqmodel.adapter.adapter import Adapter

from ..models._const import DEVICE, normalize_device
from ..nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ..nn_modules.qlinear.bitblas import BitBLASQuantLinear
from ..nn_modules.qlinear.exllama import ExllamaQuantLinear
from ..nn_modules.qlinear.exllama_eora import ExllamaEoraQuantLinear
from ..nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from ..nn_modules.qlinear.ipex import IPEXQuantLinear
from ..nn_modules.qlinear.marlin import MarlinQuantLinear
from ..nn_modules.qlinear.qqq import QQQQuantLinear
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..nn_modules.qlinear.tritonv2 import TRITON_AVAILABLE, TRITON_INSTALL_HINT, TritonV2QuantLinear
from ..quantization import FORMAT
from ..utils.logger import setup_logger
from . import BACKEND
from .rocm import IS_ROCM
from .torch import HAS_CUDA, HAS_MPS, HAS_XPU

message_logged = False
log = setup_logger()

AUTO_SELECT_BACKEND_ORDER = OrderedDict({
    BACKEND.MARLIN: MarlinQuantLinear, # optimized for bs > 1
    # BACKEND.EXLLAMA_EORA: ExllamaEoraQuantLinear, #
    BACKEND.EXLLAMA_V2: ExllamaV2QuantLinear, # optimized for bs > 1
    BACKEND.EXLLAMA_V1: ExllamaQuantLinear, # optimized for bs == 1
    BACKEND.TRITON: TritonV2QuantLinear, # good all around kernel that JIT compiles
    # BACKEND.CUDA: DynamicCudaQuantLinear,
    BACKEND.IPEX: IPEXQuantLinear, # best kernel Intel XPU and CPU with amx/avx512/xmx
    BACKEND.BITBLAS: BitBLASQuantLinear, # super slow AOT pre-compiler but fastest for bs=1
    BACKEND.TORCH: TorchQuantLinear, # slightly slower than Triton but getting close in Torch 2.6.0+

    BACKEND.QQQ: QQQQuantLinear, # qqq kernel based on marlin
})

FORMAT_DICT = {
    FORMAT.GPTQ: [BACKEND.MARLIN, BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.IPEX, BACKEND.TORCH, BACKEND.MARLIN_FP16, BACKEND.EXLLAMA_EORA],
    FORMAT.GPTQ_V2: [BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.TORCH],
    FORMAT.MARLIN: [BACKEND.MARLIN, BACKEND.MARLIN_FP16],
    FORMAT.BITBLAS: [BACKEND.BITBLAS],
    FORMAT.IPEX: [BACKEND.IPEX],
    FORMAT.QQQ: [BACKEND.QQQ],
}

def normalize_device_device_map(device: Optional[Union[str, torch.device]], device_map: Optional[Union[str, Dict]]) -> Optional[DEVICE]:
    normalized_device = None
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
                    normalized_device = d
            elif len(normalized_devices) > 1:
                normalized_devices.discard(DEVICE.CPU)
                normalized_device = normalized_devices.pop()
    else:
        if isinstance(device, str):
            normalized_device = normalize_device(device)
        elif isinstance(device, torch.device):
            normalized_device = DEVICE(device.type)
        else:
            raise ValueError(f"device must be a string or torch.device, got {type(device)}")

    # map fake cuda to actual rocm
    if normalized_device == DEVICE.CUDA and IS_ROCM:
        normalized_device = DEVICE.ROCM
    return normalized_device


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
        pack: Optional[bool] = True,
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
        pack=pack,
        allow_marlin=True, # TODO: remove this after marlin padding is fixed
        dynamic=None,
        pack_dtype=torch.int32,
        adapter=None,
    )


# auto select the correct/optimal QuantLinear class
def select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        device: Optional[DEVICE] = None,
        backend: BACKEND = BACKEND.AUTO,
        format: FORMAT = FORMAT.GPTQ,
        pack: bool = False,
        allow_marlin: bool = True,  # TODO: remove this after marlin padding is fixed
        dynamic=None,
        pack_dtype: torch.dtype = None,
        multi_select: bool = False, # return all valid kernels
        adapter: Optional[Adapter] = None,
) -> Union[Type[BaseQuantLinear], List[Type[BaseQuantLinear]]]:
    if device is None:
        device = DEVICE.XPU if backend == BACKEND.IPEX else DEVICE.CUDA

    backend = BACKEND.AUTO if backend is None else backend

    trainable = backend == BACKEND.AUTO_TRAINABLE

    validated_qlinears = []
    # Handle the case where backend is AUTO.
    if backend in [BACKEND.AUTO, BACKEND.AUTO_TRAINABLE]:
        allow_quant_linears = [(k, v) for k,v in AUTO_SELECT_BACKEND_ORDER.items() if k in FORMAT_DICT[format]]
        err = None
        global message_logged
        # Suppose all quant linears in the model should have the same backend.
        for k, cls in allow_quant_linears:
            validate, err = cls.validate(
                bits=bits,
                group_size=group_size,
                desc_act=desc_act,
                sym=sym,
                pack_dtype=pack_dtype,
                dynamic=dynamic,
                device=device,
                trainable=trainable,
                adapter=adapter,
            )
            if os.environ.get("DEBUG") and not validate:
                log.info(f"skip {k} for {str(err)}")
            if validate:
                if pack:
                    check_pack_func = issubclass(cls, PackableQuantLinear)
                    if check_pack_func:
                        #if not message_logged:
                        #    logger.info(f"Auto pick kernel based on compatibility: {cls}")
                        #    message_logged = True
                        log.info(f"{'Packing' if pack else ''} Kernel: Auto-selection: adding candidate `{cls.__name__}`")
                        validated_qlinears.append(cls)
                        if not multi_select:
                            return cls
                else:
                    #if not message_logged:
                    #    logger.info(f"Auto pick kernel based on compatibility: {cls}")
                    #    message_logged = True
                    log.info(f"{'Packing' if pack else ''} Kernel: Auto-selection: adding candidate `{cls.__name__}`")
                    validated_qlinears.append(cls)
                    if not multi_select:
                        return cls

        if err:
            raise err

        return validated_qlinears

    # Handle the case where backend is not AUTO.
    if backend == BACKEND.TRITON:
        if not TRITON_AVAILABLE:
            raise ValueError(TRITON_INSTALL_HINT)
        qlinear = TritonV2QuantLinear
    elif backend == BACKEND.BITBLAS:
        qlinear = BitBLASQuantLinear
    elif backend in [BACKEND.MARLIN, BACKEND.MARLIN_FP16]:
        qlinear = MarlinQuantLinear
    elif backend == BACKEND.EXLLAMA_EORA:
        qlinear = ExllamaEoraQuantLinear
    elif backend == BACKEND.EXLLAMA_V2:
        qlinear = ExllamaV2QuantLinear
    elif backend == BACKEND.EXLLAMA_V1:
        qlinear = ExllamaQuantLinear
    elif backend == BACKEND.IPEX:
        from ..nn_modules.qlinear.ipex import HAS_IPEX
        if not HAS_IPEX:
            raise ValueError("{'Packing' if pack else ''} Kernel: IPEX is not installed. Please install it via `pip install gptqmodel['ipex']`")

        from device_smi import Device

        cpu_vendor = Device("cpu").vendor
        if cpu_vendor != "intel":
            log.warn(f"{'Packing' if pack else ''} Kernel: IPEX on cpu is only validated and optimized for Intel cpu with AVX512, AMX, or XMX. Current cpu vendor: `{cpu_vendor}`.")

        qlinear = IPEXQuantLinear
    elif backend == BACKEND.QQQ:
        qlinear = QQQQuantLinear
    elif backend == BACKEND.TORCH:
        qlinear = TorchQuantLinear
    else:
        qlinear = TorchQuantLinear

    validate, err = qlinear.validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, pack_dtype=pack_dtype, dynamic=dynamic, device=device, trainable=trainable)

    log.info(f"{'Packing' if pack else ''} Kernel: selected: `{qlinear.__name__}`")
    if not validate:
        raise ValueError(err)
    else:
        if multi_select:
            return [qlinear]
        else:
            return qlinear
