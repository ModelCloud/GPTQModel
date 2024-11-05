from model_test import ModelTest

class TestMiniCpm(ModelTest):
    NATIVE_MODEL_ID = "openbmb/MiniCPM-2B-128k"
    NATIVE_ARC_CHALLENGE_ACC = 0.3848
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4164
    TRUST_REMOTE_CODE = True

    def test_minicpm(self):
        self.quant_lm_eval()
