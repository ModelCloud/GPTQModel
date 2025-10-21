# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Type, Union

import torch

from gptqmodel.adapter.adapter import Adapter

from ..models._const import DEVICE, normalize_device
from ..nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ..nn_modules.qlinear.awq_exllama import AwqExllamaQuantLinear
from ..nn_modules.qlinear.awq_exllamav2 import AwqExllamaV2QuantLinear
from ..nn_modules.qlinear.awq_gemm import AwqGEMMQuantLinear
from ..nn_modules.qlinear.awq_gemv import AwqGEMVQuantLinear
from ..nn_modules.qlinear.awq_gemv_fast import AwqGEMVFastQuantLinear
from ..nn_modules.qlinear.awq_machete import AwqMacheteQuantLinear
from ..nn_modules.qlinear.awq_marlin import AwqMarlinQuantLinear
from ..nn_modules.qlinear.bitblas import BitBLASQuantLinear
from ..nn_modules.qlinear.exllama import ExllamaQuantLinear
from ..nn_modules.qlinear.exllama_eora import ExllamaEoraQuantLinear
from ..nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from ..nn_modules.qlinear.machete import MacheteQuantLinear
from ..nn_modules.qlinear.marlin import MarlinQuantLinear
from ..nn_modules.qlinear.qqq import QQQQuantLinear
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..nn_modules.qlinear.torch_fused import TorchFusedQuantLinear
from ..nn_modules.qlinear.tritonv2 import TRITON_AVAILABLE, TRITON_INSTALL_HINT, TritonV2QuantLinear
from ..quantization import FORMAT, METHOD
from ..utils.logger import setup_logger
from . import BACKEND
from .rocm import IS_ROCM
from .torch import HAS_CUDA, HAS_MPS, HAS_XPU


ACCELERATE_DEVICE_MAP_KEYWORDS = {"auto", "balanced", "sequential"}
ACCELERATE_DEVICE_MAP_PREFIXES = ("balanced_low_",)
ACCELERATE_OFFLOAD_TARGETS = {"disk", "meta"}


message_logged = False
log = setup_logger()

AUTO_SELECT_BACKEND_ORDER_MAP = {
    METHOD.GPTQ: OrderedDict({
        BACKEND.MACHETE: MacheteQuantLinear, # optimized for sm90+
        BACKEND.MARLIN: MarlinQuantLinear, # optimized for bs > 1
        # BACKEND.EXLLAMA_EORA: ExllamaEoraQuantLinear, #
        BACKEND.EXLLAMA_V2: ExllamaV2QuantLinear, # optimized for bs > 1
        BACKEND.EXLLAMA_V1: ExllamaQuantLinear, # optimized for bs == 1
        BACKEND.TORCH_FUSED: TorchFusedQuantLinear, # optimized for Intel XPU
        BACKEND.TRITON: TritonV2QuantLinear, # good all around kernel that JIT compiles
        # BACKEND.CUDA: DynamicCudaQuantLinear,
        BACKEND.BITBLAS: BitBLASQuantLinear, # super slow AOT pre-compiler but fastest for bs=1
        BACKEND.TORCH: TorchQuantLinear, # slightly slower than Triton but getting close in Torch 2.6.0+
    }),
    METHOD.QQQ: OrderedDict({
        BACKEND.QQQ: QQQQuantLinear, # qqq kernel based on marlin
    }),
    METHOD.AWQ: OrderedDict({
        BACKEND.MACHETE: AwqMacheteQuantLinear,
        BACKEND.MARLIN: AwqMarlinQuantLinear,
        BACKEND.EXLLAMA_V2: AwqExllamaV2QuantLinear,
        BACKEND.EXLLAMA_V1: AwqExllamaQuantLinear,
        BACKEND.GEMM: AwqGEMMQuantLinear,
        BACKEND.GEMV: AwqGEMVQuantLinear,
        BACKEND.GEMV_FAST: AwqGEMVFastQuantLinear,
    }),
}

SUPPORTS_BACKEND_MAP = {
    METHOD.GPTQ: {
        FORMAT.GPTQ: [BACKEND.MACHETE, BACKEND.MARLIN, BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TORCH_FUSED, BACKEND.TRITON, BACKEND.TORCH_FUSED, BACKEND.TORCH, BACKEND.MARLIN_FP16, BACKEND.EXLLAMA_EORA],
        FORMAT.GPTQ_V2: [BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TORCH_FUSED, BACKEND.TRITON, BACKEND.TORCH],
        FORMAT.MARLIN: [BACKEND.MARLIN, BACKEND.MARLIN_FP16],
        FORMAT.BITBLAS: [BACKEND.BITBLAS],
    },
    METHOD.QQQ: {
        FORMAT.QQQ: [BACKEND.QQQ],
    },
    METHOD.AWQ: {
        FORMAT.GEMM: [BACKEND.MACHETE, BACKEND.MARLIN, BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.GEMM],
        FORMAT.GEMV: [BACKEND.GEMV],
        FORMAT.GEMV_FAST: [BACKEND.GEMV_FAST],
        FORMAT.MARLIN: [BACKEND.MACHETE, BACKEND.MARLIN],
    }
}


def _is_accelerate_device_map_keyword(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ACCELERATE_DEVICE_MAP_KEYWORDS:
        return True
    return any(lowered.startswith(prefix) for prefix in ACCELERATE_DEVICE_MAP_PREFIXES)


def _is_accelerate_offload_target(value: str) -> bool:
    return value.strip().lower() in ACCELERATE_OFFLOAD_TARGETS


def normalize_device_device_map(device: Optional[Union[str, torch.device]], device_map: Optional[Union[str, Dict]]) -> Optional[DEVICE]:
    normalized_device = None
    if device is None:
        if device_map is not None:
            if isinstance(device_map, str):
                if _is_accelerate_device_map_keyword(device_map):
                    return None
                devices = {device_map}
            else:
                devices = set(device_map.values())
            normalized_devices = set()
            for device in devices:
                # Returning None means quant linear will be automatically selected.
                if device is None:
                    continue
                if isinstance(device, str):
                    if _is_accelerate_device_map_keyword(device):
                        return None
                    if _is_accelerate_offload_target(device):
                        continue
                    if device == "auto":
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
        if HAS_CUDA:
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
        quant_method=METHOD.GPTQ,
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
        quant_method: METHOD = METHOD.GPTQ,
        pack: bool = False,
        allow_marlin: bool = True,  # TODO: remove this after marlin padding is fixed
        dynamic=None,
        pack_dtype: torch.dtype = None,
        multi_select: bool = False, # return all valid kernels
        adapter: Optional[Adapter] = None,
) -> Union[Type[BaseQuantLinear], List[Type[BaseQuantLinear]]]:
    # TODO: this looks wrong
    if device is None:
        device = DEVICE.CUDA

    backend = BACKEND.AUTO if backend is None else backend

    trainable = backend == BACKEND.AUTO_TRAINABLE

    validated_qlinears = []
    # Handle the case where backend is AUTO.
    if backend in [BACKEND.AUTO, BACKEND.AUTO_TRAINABLE]:
        allow_quant_linears = [(k, v) for k, v in AUTO_SELECT_BACKEND_ORDER_MAP[quant_method].items() if k in SUPPORTS_BACKEND_MAP[quant_method][format]]
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
                    check_pack_func = issubclass(cls, PackableQuantLinear) or (
                        hasattr(cls, "pack_block") and callable(getattr(cls, "pack_block"))
                    )
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

        if len(validated_qlinears) == 0:
            raise ValueError("No valid quant linear")

        return validated_qlinears

    # TODO check AWQ format supports BACKEND

    # Handle the case where backend is not AUTO.
    if backend == BACKEND.TRITON:
        if not TRITON_AVAILABLE:
            raise ValueError(TRITON_INSTALL_HINT)
        qlinear = TritonV2QuantLinear
    elif backend == BACKEND.BITBLAS:
        qlinear = BitBLASQuantLinear
    elif backend == BACKEND.MACHETE:
        if quant_method == METHOD.AWQ:
            qlinear = AwqMacheteQuantLinear
        else:
            qlinear = MacheteQuantLinear
    elif backend in [BACKEND.MARLIN, BACKEND.MARLIN_FP16]:
        if quant_method == METHOD.AWQ:
            qlinear = AwqMarlinQuantLinear
        else:
            qlinear = MarlinQuantLinear
    elif backend == BACKEND.EXLLAMA_EORA:
        qlinear = ExllamaEoraQuantLinear
    elif backend == BACKEND.EXLLAMA_V2:
        if quant_method == METHOD.AWQ:
            qlinear = AwqExllamaV2QuantLinear
        else:
            qlinear = ExllamaV2QuantLinear
    elif backend == BACKEND.EXLLAMA_V1:
        if quant_method == METHOD.AWQ:
            qlinear = AwqExllamaQuantLinear
        else:
            qlinear = ExllamaQuantLinear
    elif backend == BACKEND.QQQ:
        qlinear = QQQQuantLinear
    elif backend == BACKEND.GEMM:
        qlinear = AwqGEMMQuantLinear
    elif backend == BACKEND.GEMV:
        qlinear = AwqGEMVQuantLinear
    elif backend == BACKEND.GEMV_FAST:
        qlinear = AwqGEMVFastQuantLinear
    elif backend == BACKEND.TORCH:
        qlinear = TorchQuantLinear
    elif backend == BACKEND.TORCH_FUSED:
        qlinear = TorchFusedQuantLinear
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
