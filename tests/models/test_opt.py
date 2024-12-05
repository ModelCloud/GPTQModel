from gptqmodel import BACKEND
from gptqmodel.utils.importer import backend_dict
from model_test import ModelTest


class TestOpt(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/opt-125m"  # "facebook/opt-125m"
    NATIVE_ARC_CHALLENGE_ACC = 0.1894
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2278

    KERNEL_QUANT = {backend_dict[BACKEND.EXLLAMA_V1]}
    KERNEL_INFERENCE = {backend_dict[BACKEND.MARLIN]}

    def test_opt(self):
        self.quant_lm_eval()
