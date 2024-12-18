# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

from models.model_test import ModelTest  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402


class TestLmHead(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"  # "LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"
    DEVICE = "cuda:0"
    NATIVE_ARC_CHALLENGE_ACC = 0.2799
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3046
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2

    def test_load(self):
        model = GPTQModel.load(self.NATIVE_MODEL_ID, device=self.DEVICE)

        # validate lm_head is loaded as quantized layer
        assert isinstance(model.model.lm_head, BaseQuantLinear)

    def test_eval(self):
        self.quant_lm_eval()
