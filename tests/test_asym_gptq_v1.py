# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from gptqmodel.quantization import FORMAT  # noqa: E402

from models.model_test import ModelTest  # noqa: E402


class Test(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"  # "meta-llama/Llama-3.2-1B-Instruct"
    QUANT_FORMAT = FORMAT.GPTQ
    SYM = False

    def test(self):
        self.quant_lm_eval()
