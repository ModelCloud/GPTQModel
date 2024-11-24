from tests.model_test import ModelTest


class TestTinyllama(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
    NATIVE_ARC_CHALLENGE_ACC = 0.2995
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3268
    TRUST_REMOTE_CODE = True

    def test_tinyllama(self):
        self.quant_lm_eval()
