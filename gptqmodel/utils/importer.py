from collections import OrderedDict
from logging import getLogger

from ..nn_modules.qlinear.qlinear_bitblas import QuantLinear as BitBLASQuantLinear
from ..nn_modules.qlinear.qlinear_cuda import QuantLinear as CudaQuantLinear
from ..nn_modules.qlinear.qlinear_cuda_old import QuantLinear as CudaOldQuantLinear
from ..nn_modules.qlinear.qlinear_exllama import QuantLinear as ExllamaQuantLinear
from ..nn_modules.qlinear.qlinear_exllamav2 import QuantLinear as ExllamaV2QuantLinear
from ..nn_modules.qlinear.qlinear_marlin import QuantLinear as MarlinQuantLinear
from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear as TritonV2QuantLinear
from ..quantization import FORMAT
from .backend import Backend

backend_dict = OrderedDict({
    Backend.MARLIN: MarlinQuantLinear,
    Backend.EXLLAMA_V2: ExllamaV2QuantLinear,
    Backend.EXLLAMA: ExllamaQuantLinear,
    Backend.TRITON: TritonV2QuantLinear,
    Backend.CUDA_OLD: CudaOldQuantLinear,
    Backend.CUDA: CudaQuantLinear,
    Backend.BITBLAS: BitBLASQuantLinear,
})

format_dict = {
    FORMAT.GPTQ: [Backend.EXLLAMA_V2, Backend.EXLLAMA, Backend.CUDA_OLD, Backend.CUDA],
    FORMAT.GPTQ_V2: [Backend.EXLLAMA_V2, Backend.EXLLAMA, Backend.CUDA_OLD, Backend.CUDA],
    FORMAT.MARLIN: [Backend.MARLIN],
    FORMAT.BITBLAS: [Backend.BITBLAS],
}

logger = getLogger(__name__)

# auto select the correct/optimal QuantLinear class
def select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        backend: Backend,
        format: str,
        pack: bool = False,
):
    # Handle the case where backend is AUTO.
    if backend == Backend.AUTO:
        allow_backends = format_dict[format]
        for k, v in backend_dict.items():
            in_allow_backends = k in allow_backends
            validate = v.validate(bits, group_size, desc_act, sym, raise_error=False)
            check_pack_func = hasattr(v, "pack") if pack else True
            if in_allow_backends and validate and check_pack_func:
                logger.info(f"Auto choose the fastest one based on quant model compatibility: {v}")
                return v

    # Handle the case where backend is not AUTO.
    if backend == Backend.TRITON:
        logger.info("Using tritonv2 for GPTQ")
        from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
    elif backend == Backend.BITBLAS:
        from ..nn_modules.qlinear.qlinear_bitblas import QuantLinear
    elif bits == 4 and sym and not desc_act and backend == Backend.MARLIN:
        from ..nn_modules.qlinear.qlinear_marlin import QuantLinear
    elif bits == 4 and backend == Backend.EXLLAMA_V2:
        from ..nn_modules.qlinear.qlinear_exllamav2 import QuantLinear
    elif bits == 4 and backend == Backend.EXLLAMA:
        from ..nn_modules.qlinear.qlinear_exllama import QuantLinear
    elif not desc_act or group_size == -1:
        from ..nn_modules.qlinear.qlinear_cuda_old import QuantLinear
    else:
        from ..nn_modules.qlinear.qlinear_cuda import QuantLinear

    return QuantLinear
