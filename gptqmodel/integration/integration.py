from optimum.gptq import quantizer
from .optimum.gptq import quantizer as patched_quantizer
from optimum.utils import import_utils
from .optimum.utils import import_utils as patched_import_utils
from optimum.utils import testing_utils
from .optimum.utils import testing_utils as patched_testing_utils

from peft import import_utils as import_utils
from peft.tuners.adalora.model import AdaLoraModel
from .peft.tuners.adalora.model import AdaLoraModel as patched_AdaLoraModel
from peft.tuners.lora import gptq
from .peft.tuners.lora import gptq as patched_gptq
from peft.tuners.lora import model
from .peft.tuners.lora import model as patched_model
from .peft import import_utils as patched_import_utils

from peft.utils import other
from .peft.utils import other as patched_other

from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
from .transformers.quantizers.quantizer_gptq import GptqHfQuantizer as patched_GptqHfQuantizer


def monkey_patch_peft():
    import_utils.is_gptqmodel_available = patched_import_utils.is_gptqmodel_available

    AdaLoraModel._create_and_replace = patched_AdaLoraModel._create_and_replace

    gptq.dispatch_gptq = patched_gptq.dispatch_gptq

    model.LoraModel = patched_model.LoraModel

    other.get_auto_gptq_quant_linear = patched_other.get_auto_gptq_quant_linear
    other.get_gptqmodel_quant_linear = patched_other.get_gptqmodel_quant_linear


def monkey_patch_optimum():
    quantizer.is_gptqmodel_available = patched_quantizer.is_gptqmodel_available
    quantizer.has_device_more_than_cpu = patched_quantizer.has_device_more_than_cpu
    quantizer.GPTQQuantizer = patched_quantizer.GPTQQuantizer

    import_utils._gptqmodel_available = patched_import_utils._gptqmodel_available
    import_utils.is_gptqmodel_available = patched_import_utils.is_gptqmodel_available
    testing_utils.require_gptq = patched_testing_utils.require_gptq


def monkey_patch_transformers():
    GptqHfQuantizer.required_packages = patched_GptqHfQuantizer.required_packages
    GptqHfQuantizer.validate_environment = patched_GptqHfQuantizer.validate_environment
    GptqHfQuantizer.update_torch_dtype = patched_GptqHfQuantizer.update_torch_dtype
