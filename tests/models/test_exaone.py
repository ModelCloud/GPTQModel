from model_test import ModelTest # noqa: E402



class TestExaone(ModelTest):
    NATIVE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4232
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4164
    TRUST_REMOTE_CODE = True
    def test_exaone(self):
        self.quant_lm_eval()


