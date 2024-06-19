from logging import getLogger

logger = getLogger(__name__)


def dynamically_import_QuantLinear(
    use_triton: bool,
    desc_act: bool,
    group_size: int,
    bits: int,
    disable_exllama: bool = False,
    disable_exllamav2: bool = False,
    use_marlin: bool = False,
):
    if use_triton:
        logger.info("Using tritonv2 for GPTQ")
        from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
    else:
        if bits == 4 and use_marlin:
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
