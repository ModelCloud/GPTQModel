from logging import getLogger

logger = getLogger(__name__)


# auto select the correct/optimal QuantLinear class
def select_quant_linear(
    bits: int,
    group_size: int,
    desc_act: bool,
    sym: bool,
    use_triton: bool,
    disable_exllama: bool = False,
    disable_exllamav2: bool = False,
    use_marlin: bool = False,
    use_bitblas: bool = True,
):
    if use_triton:
        logger.info("Using tritonv2 for GPTQ")
        from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
    else:
        if use_bitblas:
            from ..nn_modules.qlinear.qlinear_bitblas import QuantLinear
        elif bits == 4 and sym and not desc_act and use_marlin:
            from ..nn_modules.qlinear.qlinear_marlin import QuantLinear
        elif bits == 4 and not disable_exllamav2:
            from ..nn_modules.qlinear.qlinear_exllamav2 import QuantLinear
        elif bits == 4 and not disable_exllama:
            from ..nn_modules.qlinear.qlinear_exllama import QuantLinear
        elif not desc_act or group_size == -1:
            from ..nn_modules.qlinear.qlinear_cuda_old import QuantLinear
        else:
            from ..nn_modules.qlinear.qlinear_cuda import QuantLinear

    return QuantLinear
