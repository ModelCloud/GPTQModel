from model_test import ModelTest

class TestDeci(ModelTest):
    NATIVE_MODEL_ID = "Deci/DeciLM-7B-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.5239
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5222
    QUANT_ARC_MAX_NEGATIVE_DELTA = 0.55
    TRUST_REMOTE_CODE = True

    def test_deci(self):
        self.quant_lm_eval()