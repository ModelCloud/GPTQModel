from model_test import ModelTest  # noqa: E402


class TestFalcon(ModelTest):
    NATIVE_MODEL_ID = "tiiuae/falcon-7b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3993
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4292
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_falcon(self):
        self.quant_lm_eval()
