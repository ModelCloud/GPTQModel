from model_test import ModelTest

class TestStablelm(ModelTest):
    NATIVE_MODEL_ID = "stabilityai/stablelm-base-alpha-3b"
    NATIVE_ARC_CHALLENGE_ACC = 0.2363
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2577
    TRUST_REMOTE_CODE = True

    def test_stablelm(self):
        self.quant_lm_eval()
