# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from parameterized import parameterized  # noqa: E402

from gptqmodel.quantization import FORMAT  # noqa: E402

from models.model_test import ModelTest  # noqa: E402


class Test(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"  # "meta-llama/Llama-3.2-1B-Instruct"
    QUANT_FORMAT = FORMAT.GPTQ
    QUANT_ARC_MAX_NEGATIVE_DELTA = 0.2

    @parameterized.expand([True, False])
    def test(self, sym: bool):
        self.SYM = sym
        if sym:
            self.NATIVE_ARC_CHALLENGE_ACC = 0.2269
            self.NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2269
        else:
            self.NATIVE_ARC_CHALLENGE_ACC = 0.2747
            self.NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2935

        self.quant_lm_eval()
