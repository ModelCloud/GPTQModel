from collections import OrderedDict
from logging import getLogger
from typing import Dict, Optional

from ..nn_modules.qlinear.qlinear_bitblas import BitBLASQuantLinear
from ..nn_modules.qlinear.qlinear_exllama import ExllamaQuantLinear
from ..nn_modules.qlinear.qlinear_exllamav2 import ExllamaV2QuantLinear
from ..nn_modules.qlinear.qlinear_marlin import MarlinQuantLinear
from ..nn_modules.qlinear.qlinear_marlin_inference import MarlinInferenceQuantLinear
from ..nn_modules.qlinear.qlinear_qbits import QBitsQuantLinear
from ..nn_modules.qlinear.qlinear_tritonv2 import TritonV2QuantLinear
from ..quantization import FORMAT
from .backend import BACKEND

backend_dict = OrderedDict({
    BACKEND.MARLIN: [MarlinInferenceQuantLinear, MarlinQuantLinear],
    BACKEND.EXLLAMA_V2: [ExllamaV2QuantLinear],
    BACKEND.EXLLAMA: [ExllamaQuantLinear],
    BACKEND.TRITON: [TritonV2QuantLinear],
    BACKEND.BITBLAS: [BitBLASQuantLinear],
    BACKEND.QBITS: [QBitsQuantLinear],
})

format_dict = {
    FORMAT.GPTQ: [BACKEND.EXLLAMA_V2, BACKEND.TRITON],
    FORMAT.GPTQ_V2: [BACKEND.EXLLAMA_V2, BACKEND.TRITON],
    FORMAT.MARLIN: [BACKEND.MARLIN],
    FORMAT.BITBLAS: [BACKEND.BITBLAS],
    FORMAT.QBITS: [BACKEND.QBITS],
}

logger = getLogger(__name__)


# auto select the correct/optimal QuantLinear class
def select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        backend: BACKEND,
        format: FORMAT,
        pack: bool = False,
        dynamic_bits: Optional[Dict[str, int]] = None,
):
    # Handle the case where backend is AUTO.
    if backend == BACKEND.AUTO:
        allow_backends = format_dict[format]
        for k, values in backend_dict.items():
            for v in values:
                in_allow_backends = k in allow_backends
                validate, err = v.validate(bits, group_size, desc_act, sym, dynamic_bits=dynamic_bits)
                check_pack_func = hasattr(v, "pack") if pack else True
                if in_allow_backends:
                    if err:
                        raise NotImplementedError(err)
                    elif validate and check_pack_func:
                        logger.info(f"Auto choose the fastest one based on quant model compatibility: {v}")
                        return v
    # Handle the case where backend is not AUTO.
    if backend == BACKEND.TRITON:
        return TritonV2QuantLinear
    elif backend == BACKEND.BITBLAS:
        return BitBLASQuantLinear
    elif backend == BACKEND.MARLIN:
        return MarlinQuantLinear if pack else MarlinInferenceQuantLinear
    elif backend == BACKEND.EXLLAMA_V2:
        return ExllamaV2QuantLinear
    elif backend == BACKEND.EXLLAMA:
        return ExllamaQuantLinear
    elif backend == BACKEND.QBITS:
        return QBitsQuantLinear
    else:
        return ExllamaQuantLinear
