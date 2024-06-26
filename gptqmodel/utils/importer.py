from logging import getLogger

from .backend import Backend

logger = getLogger(__name__)


# auto select the correct/optimal QuantLinear class
def select_quant_linear(
    bits: int,
    group_size: int,
    desc_act: bool,
    sym: bool,
    backend: Backend,
):
    # TODO handle AUTO
    if backend == Backend.TRITON_V2:
        logger.info("Using tritonv2 for GPTQ")
        from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
    else:
        if backend == Backend.BITBLAS:
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
