HAS_OPTIMUM = True
try:
    import optimum.gptq as optimum_gptq
    from optimum.gptq import quantizer as optimum_quantizer
    from .src.optimum.gptq import quantizer as patched_optimum_quantizer
    from optimum.utils import testing_utils as optimum_testing_utils, import_utils as optimum_import_utils
    from .src.optimum.utils import testing_utils as patched_optimum_testing_utils, import_utils as patched_optimum_import_utils
except BaseException:
    HAS_OPTIMUM = False

HAS_PEFT = True
try:
    from peft import import_utils as peft_import_utils
    from .src.peft import import_utils as patched_peft_import_utils
    from peft.tuners.adalora.model import AdaLoraModel as peft_AdaLoraModel
    from .src.peft.tuners.adalora.model import AdaLoraModel as patched_peft_AdaLoraModel
    from peft.tuners.lora import gptq as peft_gptq, model as peft_model
    from .src.peft.tuners.lora import gptq as patched_peft_gptq, model as patched_peft_model
    from peft.utils import other as peft_other
    from .src.peft.utils import other as patched_peft_other
except BaseException:
    HAS_PEFT = False

from transformers.quantizers import quantizer_gptq as transformers_quantizer_gptq  # noqa: E402
from .src.transformers.quantizers import quantizer_gptq as patched_transformers_quantizer_gptq  # noqa: E402
from transformers.utils import import_utils as transformers_import_utils  # noqa: E402
from .src.transformers.utils import import_utils as patched_transformers_import_utils  # noqa: E402
from transformers.utils import quantization_config as transformers_quantization_config  # noqa: E402
from .src.transformers.utils import quantization_config as patched_transformers_quantization_config  # noqa: E402
import transformers.testing_utils as transformers_testing_utils
from .src.transformers import testing_utils as patched_transformers_testing_utils

def monkey_patch_transformers():
    _patch_peft()
    _patch_optimum()
    _patch_transformers()


def _patch_peft():
    if not HAS_PEFT:
        return

    peft_import_utils.is_gptqmodel_available = patched_peft_import_utils.is_gptqmodel_available

    peft_AdaLoraModel._create_and_replace = patched_peft_AdaLoraModel._create_and_replace

    peft_gptq.dispatch_gptq = patched_peft_gptq.dispatch_gptq

    peft_model.LoraModel = patched_peft_model.LoraModel

    peft_other.get_auto_gptq_quant_linear = patched_peft_other.get_auto_gptq_quant_linear
    peft_other.get_gptqmodel_quant_linear = patched_peft_other.get_gptqmodel_quant_linear


def _patch_optimum():
    if not HAS_OPTIMUM:
        return

    optimum_gptq.GPTQQuantizer = patched_optimum_quantizer.GPTQQuantizer
    optimum_quantizer.is_gptqmodel_available = patched_optimum_quantizer.is_gptqmodel_available
    optimum_quantizer.has_device_more_than_cpu = patched_optimum_quantizer.has_device_more_than_cpu
    optimum_quantizer.ExllamaVersion = patched_optimum_quantizer.ExllamaVersion

    optimum_import_utils._gptqmodel_available = patched_optimum_import_utils._gptqmodel_available
    optimum_import_utils.is_gptqmodel_available = patched_optimum_import_utils.is_gptqmodel_available
    optimum_testing_utils.require_gptq = patched_optimum_testing_utils.require_gptq


def _patch_transformers():
    transformers_quantizer_gptq.GptqHfQuantizer = patched_transformers_quantizer_gptq.GptqHfQuantizer

    transformers_import_utils._gptqmodel_available  = patched_transformers_import_utils._gptqmodel_available
    transformers_import_utils.is_gptqmodel_available = patched_transformers_import_utils.is_gptqmodel_available

    transformers_quantization_config.AWQLinearVersion =  patched_transformers_quantization_config.AWQLinearVersion

    transformers_testing_utils.require_gptq = patched_transformers_testing_utils.require_gptq
