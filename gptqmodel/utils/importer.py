from collections import OrderedDict
import torch
from typing import Optional, Union

from .backend import BACKEND
from ..nn_modules.qlinear.qlinear_bitblas import BitBLASQuantLinear
from ..nn_modules.qlinear.qlinear_cuda import CudaQuantLinear
from ..nn_modules.qlinear.qlinear_exllama import ExllamaQuantLinear
from ..nn_modules.qlinear.qlinear_exllamav2 import ExllamaV2QuantLinear
from ..nn_modules.qlinear.qlinear_ipex import IPEXQuantLinear
from ..nn_modules.qlinear.qlinear_marlin import MarlinQuantLinear
from ..nn_modules.qlinear.qlinear_tritonv2 import TRITON_AVAILABLE, TRITON_INSTALL_HINT, TritonV2QuantLinear
from ..nn_modules.qlinear.qlinear_torch import TorchQuantLinear
from ..quantization import FORMAT
from ..utils.logger import setup_logger

logger = setup_logger()

backend_dict = OrderedDict({
    BACKEND.MARLIN: [MarlinQuantLinear],
    BACKEND.EXLLAMA_V2: [ExllamaV2QuantLinear],
    BACKEND.EXLLAMA_V1: [ExllamaQuantLinear],
    BACKEND.TRITON: [TritonV2QuantLinear],
    BACKEND.CUDA: [CudaQuantLinear],
    BACKEND.BITBLAS: [BitBLASQuantLinear],
    BACKEND.IPEX: [IPEXQuantLinear],
    BACKEND.TORCH: [TorchQuantLinear],
})

backend_dict_cpu = OrderedDict({
    BACKEND.IPEX: [IPEXQuantLinear],
    BACKEND.TORCH: [TorchQuantLinear],
})

format_dict = {
    FORMAT.GPTQ: [BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA],
    FORMAT.GPTQ_V2: [BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA],
    FORMAT.MARLIN: [BACKEND.MARLIN],
    FORMAT.BITBLAS: [BACKEND.BITBLAS],
    FORMAT.IPEX: [BACKEND.IPEX],
}

format_dict_cpu = {
    FORMAT.GPTQ: [IPEXQuantLinear, TorchQuantLinear],
    FORMAT.GPTQ_V2: [IPEXQuantLinear, TorchQuantLinear],
    FORMAT.IPEX: [BACKEND.IPEX],
}


# public/stable api exposed to transformer/optimum
def hf_select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        device_map: Optional[Union[str, dict]] = None,
        pack: bool = False,
        dynamic=None,
):
    # force backend to ipex if cpu/xpu is designated device
    if device_map is not None:
        devices = [device_map] if isinstance(device_map, str) else list(device_map.values())
        if any(dev in devices or torch.device(dev) in devices for dev in ["cpu", "xpu"]):
            backend = BACKEND.IPEX
    else:
        backend = BACKEND.AUTO_CPU

    return select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        backend=backend,
        format=FORMAT.GPTQ,
        pack=pack,
        dynamic=dynamic,
    )


# auto select the correct/optimal QuantLinear class
def select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        backend: BACKEND = BACKEND.AUTO,
        format: FORMAT = FORMAT.GPTQ,
        pack: bool = False,
        dynamic=None,
):
    if not torch.cuda.is_available():
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            backend = BACKEND.IPEX
        else:
            backend = BACKEND.AUTO_CPU

    # Handle the case where backend is AUTO.
    if backend == BACKEND.AUTO or backend == BACKEND.AUTO_CPU:
        allow_backends = format_dict[format] if backend == BACKEND.AUTO else format_dict_cpu[format]
        allow_quant_linears = backend_dict if backend == BACKEND.AUTO else backend_dict_cpu
        err = None
        for k, values in allow_quant_linears.items():
            for v in values:
                in_allow_backends = k in allow_backends
                validate, err = v.validate(bits, group_size, desc_act, sym, dynamic=dynamic)
                if in_allow_backends and validate:
                    if pack:
                        check_pack_func = hasattr(v, "pack")
                        if check_pack_func:
                            logger.info(f"Auto choose the fastest one based on quant model compatibility: {v}")
                            return v
                    else:
                        logger.info(f"Auto choose the fastest one based on quant model compatibility: {v}")
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
        return CudaQuantLinear
    elif backend == BACKEND.IPEX:
        from ..nn_modules.qlinear.qlinear_ipex import IPEX_AVAILABLE
        if not IPEX_AVAILABLE:
            raise ValueError("IPEX is not available.")

        # Fallback to IPEX/CPU if cpu supports AVX512
        from device_smi import Device
        if "avx512_vnni" not in Device("cpu").features:
            raise ValueError("IPEX/CPU requires minimum avx512_vnni support.")

        return IPEXQuantLinear
    elif backend == BACKEND.TORCH:
        return TorchQuantLinear
    else:
        return TorchQuantLinear
