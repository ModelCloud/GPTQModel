from logging import getLogger

logger = getLogger(__name__)

try:
    from intel_extension_for_transformers import qbits  # noqa: F401

    QBITS_AVAILABLE = True
    QBITS_EXCEPTION = None
except Exception as e:
    QBITS_AVAILABLE = False
    QBITS_EXCEPTION = e

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
    use_bitblas: bool = False,
    use_qbits: bool = False,
):
    if use_triton:
        logger.info("Using tritonv2 for GPTQ")
        from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
    else:
        if (bits == 4 or bits == 8) and use_qbits:
            if not QBITS_AVAILABLE:
                raise ValueError(
                    f"QBits appears to be not available with the error: {QBITS_EXCEPTION}. Please install with `pip install intel-extension-for-transformers`."
                )
            from ..nn_modules.qlinear.qlinear_qbits import QuantLinear
        elif use_bitblas:
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

