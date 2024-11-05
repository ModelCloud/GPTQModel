from model_test import ModelTest # noqa: E402



class TestTinyllama(ModelTest):
    NATIVE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    NATIVE_ARC_CHALLENGE_ACC = 0.2995
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3268
    TRUST_REMOTE_CODE = True

    def test_tinyllama(self):
        self.quant_lm_eval()
