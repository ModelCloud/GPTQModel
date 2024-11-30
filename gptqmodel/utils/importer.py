from collections import OrderedDict

import torch

from .backend import BACKEND
from ..nn_modules.qlinear.qlinear_bitblas import BitBLASQuantLinear
from ..nn_modules.qlinear.qlinear_cuda import CudaQuantLinear
from ..nn_modules.qlinear.qlinear_exllama import ExllamaQuantLinear
from ..nn_modules.qlinear.qlinear_exllamav2 import ExllamaV2QuantLinear
from ..nn_modules.qlinear.qlinear_ipex import IPEXQuantLinear
from ..nn_modules.qlinear.qlinear_marlin import MarlinQuantLinear
from ..nn_modules.qlinear.qlinear_tritonv2 import TRITON_AVAILABLE, TRITON_INSTALL_HINT, TritonV2QuantLinear
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
})

format_dict = {
    FORMAT.GPTQ: [BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA],
    FORMAT.GPTQ_V2: [BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA],
    FORMAT.MARLIN: [BACKEND.MARLIN],
    FORMAT.BITBLAS: [BACKEND.BITBLAS],
    FORMAT.IPEX: [BACKEND.IPEX],
}


# auto select the correct/optimal QuantLinear class
def select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        backend: BACKEND,
        format: FORMAT,
        pack: bool = False,
        dynamic=None,
):
    # Handle the case where backend is AUTO.
    if backend == BACKEND.AUTO:
        allow_backends = format_dict[format]
        err = None
        for k, values in backend_dict.items():

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

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return IPEXQuantLinear

        # Fallback to IPEX/CPU if cpu supports AVX512
        from device_smi import Device
        if "avx512_vnni" not in Device("cpu").features:
            raise ValueError("IPEX/CPU requires minimum avx512_vnni support.")

        return IPEXQuantLinear
    else:
        return CudaQuantLinear
